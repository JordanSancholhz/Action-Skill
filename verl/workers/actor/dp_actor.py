# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import json
import time
import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_jsd_loss,
    compute_policy_loss,
    compute_policy_loss_gspo,
    compute_symmetric_topk_jsd_per_action,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _forward_micro_batch_logits(self, micro_batch, temperature) -> torch.Tensor:
        """Forward pass that returns full logits for response tokens (for JSD computation).

        Returns:
            logits: (bs, response_length, vocab_size)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size)

                output = self.actor_module(
                    input_ids=input_ids_rmpad, attention_mask=None,
                    position_ids=position_ids_rmpad, **multi_modal_inputs,
                    use_cache=False)

                # Return raw logits. Temperature is applied exactly once in
                # compute_jsd_loss; applying it here as well would use an
                # unintended effective temperature of T^2.
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                if self.use_ulysses_sp:
                    # For logits we need vocab_size dim — gather across SP ranks
                    logits_rmpad = gather_outpus_and_unpad(
                        logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad back to (bsz, seqlen, vocab_size)
                full_logits = pad_input(
                    hidden_states=logits_rmpad, indices=indices,
                    batch=batch_size, seqlen=seqlen)
                # response part only
                logits = full_logits[:, -response_length - 1: -1, :]  # (bs, resp_len, V)

            else:
                output = self.actor_module(
                    input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, **multi_modal_inputs,
                    use_cache=False)
                logits = output.logits
                logits = logits[:, -response_length - 1: -1, :]  # (bs, resp_len, V)

            return logits

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_action_information(
        self,
        data: DataProto,
        top_k: int = 64,
        jsd_temperature: float = 1.0,
    ):
        """Score Skill/No-skill conditional mutual information per action.

        The Skill forward is also reused to return the old action log-probability
        and entropy, so Easy training still needs two model forwards (Skill and
        No-skill), not three. No environment action is sampled or executed here.
        """
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        policy_temperature = data.meta_info["temperature"]
        if policy_temperature <= 0:
            raise ValueError(f"policy temperature must be positive, got {policy_temperature}")
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "no_skill_probe_input_ids",
            "no_skill_probe_attention_mask",
            "no_skill_probe_position_ids",
        ]
        if "loss_mask" in data.batch:
            select_keys.append("loss_mask")
        batch = data.select(batch_keys=select_keys).batch
        if "multi_modal_inputs" in data.non_tensor_batch:
            raise NotImplementedError("Action directed-information scoring does not yet support multimodal inputs")

        indices = None
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        action_jsd_list = []
        log_probs_list = []
        entropy_list = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                skill_logits = self._forward_micro_batch_logits(
                    micro_batch=micro_batch, temperature=policy_temperature)

                no_skill_batch = {**micro_batch}
                no_skill_batch["input_ids"] = micro_batch["no_skill_probe_input_ids"]
                no_skill_batch["attention_mask"] = micro_batch["no_skill_probe_attention_mask"]
                no_skill_batch["position_ids"] = micro_batch["no_skill_probe_position_ids"]
                no_skill_logits = self._forward_micro_batch_logits(
                    micro_batch=no_skill_batch, temperature=policy_temperature)

                response_length = micro_batch["responses"].size(-1)
                if "loss_mask" in micro_batch:
                    response_mask = micro_batch["loss_mask"][:, -response_length:]
                else:
                    response_mask = micro_batch["attention_mask"][:, -response_length:]

                action_jsd, _ = compute_symmetric_topk_jsd_per_action(
                    logits_a=skill_logits,
                    logits_b=no_skill_logits,
                    response_mask=response_mask,
                    top_k=top_k,
                    temperature=jsd_temperature,
                )
                del no_skill_logits

                # JSD no longer needs the Skill logits, so temperature scaling
                # can be in-place to avoid a third full-vocabulary tensor.
                skill_logits.div_(policy_temperature)
                log_probs = logprobs_from_logits(skill_logits, micro_batch["responses"])
                entropy = verl_F.entropy_from_logits(skill_logits)

            action_jsd_list.append(action_jsd)
            log_probs_list.append(log_probs)
            entropy_list.append(entropy)

        action_jsd = torch.cat(action_jsd_list, dim=0)
        log_probs = torch.cat(log_probs_list, dim=0)
        entropys = torch.cat(entropy_list, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            action_jsd = action_jsd[revert_indices]
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]

        return action_jsd, log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # ==================== HDPO Mode ====================
        hdpo_mode = data.meta_info.get("hdpo_mode", None)
        if hdpo_mode == "jsd":
            hdpo_cfg = data.meta_info["hdpo_config"]
            return self._update_policy_hdpo_jsd(data, hdpo_cfg)
        elif hdpo_mode == "grpo":
            # GRPO path: falls through to standard update_policy below
            pass

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        loss_agg_mode_override = data.meta_info.get("loss_agg_mode_override", None)
        use_kl_loss_for_update = (
            self.config.use_kl_loss
            and not data.meta_info.get("disable_actor_kl_loss", False)
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if use_kl_loss_for_update:
            select_keys.append("ref_log_prob")

        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = loss_agg_mode_override or self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if use_kl_loss_for_update:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }

                    data["actor/pg_clipfrac"] = pg_clipfrac.detach().item()
                    data["actor/pg_clipfrac_lower"] = pg_clipfrac_lower.detach().item()

                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics

    def _update_policy_hdpo_jsd(self, data: DataProto, hdpo_cfg: dict):
        """JSD distillation update: all samples are cliff R=1, use JSD (guided→plain).

        Called as a separate update_actor pass so all GPUs execute the same operations
        (avoids FSDP deadlock from asymmetric forward passes).
        Each call is an independent optimizer cycle (zero_grad → backward → clip → step).

        Args:
            data: DataProto with JSD samples only (cliff R=1 trajectories)
            hdpo_cfg: dict with keys: jsd_lambda, jsd_top_k, jsd_temperature,
                jsd_micro_batch_size_per_gpu
        """
        multi_turn = data.meta_info.get("multi_turn", False)
        jsd_lambda = hdpo_cfg.get("jsd_lambda", 1.0)
        jsd_top_k = hdpo_cfg.get("jsd_top_k", 64)
        jsd_temperature = hdpo_cfg.get("jsd_temperature", 1.0)
        use_teacher_probe = hdpo_cfg.get("use_teacher_probe", False)
        action_ig_enabled = hdpo_cfg.get("action_ig_enabled", False)
        action_ig_clip = hdpo_cfg.get("action_ig_clip", 1.2)
        action_ig_beta = hdpo_cfg.get("action_ig_beta", 0.2)
        jsd_micro_batch_size = max(
            1,
            int(
                hdpo_cfg.get(
                    "jsd_micro_batch_size_per_gpu",
                    self.config.ppo_micro_batch_size_per_gpu,
                )
            ),
        )

        jsd_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if use_teacher_probe:
            jsd_keys.extend([
                "teacher_probe_input_ids",
                "teacher_probe_attention_mask",
                "teacher_probe_position_ids",
            ])
        else:
            jsd_keys.extend([
                "plain_input_ids", "plain_attention_mask", "plain_position_ids"
            ])
        if multi_turn:
            jsd_keys.append("loss_mask")

        metrics = {}
        self.actor_module.train()

        jsd_batch = data.select(batch_keys=jsd_keys).batch
        # JSD uses a single mini-batch (all samples) to avoid over-updating on few R=1 samples.
        # gradient_accumulation normalizes across micro-batches within this single pass.
        jsd_batch_size = jsd_batch.batch_size[0]
        jsd_mini_batch_size = min(self.config.ppo_mini_batch_size, jsd_batch_size)
        mini_batches = jsd_batch.split(jsd_mini_batch_size)

        for mini_batch in mini_batches:
            micro_batches = mini_batch.split(jsd_micro_batch_size)
            self.gradient_accumulation = len(micro_batches)
            self.actor_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.to(get_torch_device().current_device())
                responses = mb["responses"]
                response_length = responses.size(1)
                attention_mask = mb["attention_mask"]
                if multi_turn:
                    response_mask = mb["loss_mask"][:, -response_length:]
                else:
                    response_mask = attention_mask[:, -response_length:]

                # Teacher forward: guided prompt (input_ids = guided + response) — no gradient
                with torch.no_grad():
                    teacher_mb = {**mb}
                    if use_teacher_probe:
                        teacher_mb["input_ids"] = mb["teacher_probe_input_ids"]
                        teacher_mb["attention_mask"] = mb["teacher_probe_attention_mask"]
                        teacher_mb["position_ids"] = mb["teacher_probe_position_ids"]
                    teacher_logits = self._forward_micro_batch_logits(
                        micro_batch=teacher_mb, temperature=jsd_temperature)

                # Student forward: plain prompt (plain_input_ids + response) — with gradient
                plain_mb = {**mb}
                if not use_teacher_probe:
                    plain_mb["input_ids"] = mb["plain_input_ids"]
                    plain_mb["attention_mask"] = mb["plain_attention_mask"]
                    plain_mb["position_ids"] = mb["plain_position_ids"]
                student_logits = self._forward_micro_batch_logits(
                    micro_batch=plain_mb, temperature=jsd_temperature)

                jsd_loss, jsd_metrics = compute_jsd_loss(
                    teacher_logits=teacher_logits.detach(),
                    student_logits=student_logits,
                    response_mask=response_mask,
                    top_k=jsd_top_k,
                    temperature=jsd_temperature,
                    loss_agg_mode=self.config.loss_agg_mode,
                    action_ig_enabled=action_ig_enabled,
                    action_ig_clip=action_ig_clip,
                    action_ig_beta=action_ig_beta)

                loss = (jsd_lambda * jsd_loss) / self.gradient_accumulation
                loss.backward()

                append_to_dict(metrics, {
                    "actor/jsd_loss": jsd_loss.detach().item(),
                    "actor/jsd_lambda": jsd_lambda,
                    "actor/jsd_mean_per_token": jsd_metrics["jsd/mean_per_token"],
                    "actor/jsd_max_per_token": jsd_metrics["jsd/max_per_token"],
                    "actor/jsd_tail_mass": jsd_metrics["jsd/tail_mass_mean"],
                    "actor/jsd_token_count": jsd_metrics["jsd/token_count"],
                    "actor/jsd_action_ig_mean": jsd_metrics["jsd/action_ig_mean"],
                    "actor/jsd_action_ig_max": jsd_metrics["jsd/action_ig_max"],
                    "actor/jsd_action_loss_mean": jsd_metrics["jsd/action_loss_mean"],
                    "actor/jsd_action_weight_min": jsd_metrics["jsd/action_weight_min"],
                })

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

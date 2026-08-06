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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import random
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
try:
    from gigpo import core_gigpo
except ImportError:
    core_gigpo = None

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_success_centered_information_advantage(
    action_information: torch.Tensor,
    traj_uids: np.ndarray,
    task_uids: np.ndarray,
    step_indices: np.ndarray,
    successful_rows: np.ndarray,
    valid_rows: np.ndarray,
    gamma: float = 1.0,
    advantage_clip: float = 3.0,
    fixed_scale: float = 0.2,
):
    """Build an action advantage from success-centered directed information.

    ``action_information`` is the symmetric Skill/No-skill JSD at one stored
    environment action. For task q, trajectory success is centered by the
    empirical rollout pass rate: ``(R_tau - p_q) * I_tau,t``. Therefore an
    all-success or all-failure rollout group receives zero information update;
    the objective rewards information that covaries with success instead of
    maximizing Skill/No-skill divergence after outcome saturation.

    A discounted reward-to-go is computed in temporal order, centered within
    each task with equal trajectory mass, and divided by a fixed scale. Unlike
    batch z-scoring, this lets the information update naturally shrink when its
    raw signal shrinks. The returned trajectory scale makes a row-mean/action-
    mean policy loss give every trajectory equal total mass.
    """
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"information gamma must be in [0, 1], got {gamma}")
    if float(fixed_scale) <= 0.0:
        raise ValueError(f"information fixed scale must be positive, got {fixed_scale}")

    info = action_information.detach().float()
    traj_uids = np.asarray(traj_uids)
    task_uids = np.asarray(task_uids)
    step_indices = np.asarray(step_indices, dtype=np.int64)
    successful_rows = np.asarray(successful_rows, dtype=bool)
    valid_rows = np.asarray(valid_rows, dtype=bool)
    if not (len(info) == len(traj_uids) == len(task_uids) == len(step_indices) == len(valid_rows)):
        raise ValueError("Action-information metadata lengths do not match")

    device = info.device
    centered_success = torch.zeros_like(info)
    task_pass_rates = []
    for task_uid in dict.fromkeys(task_uids[valid_rows].tolist()):
        task_idx_np = np.where((task_uids == task_uid) & valid_rows)[0]
        task_traj_uids = list(dict.fromkeys(traj_uids[task_idx_np].tolist()))
        if not task_traj_uids:
            continue
        trajectory_success = {}
        for traj_uid in task_traj_uids:
            traj_idx_np = np.where((traj_uids == traj_uid) & valid_rows)[0]
            trajectory_success[traj_uid] = bool(successful_rows[traj_idx_np].any())
        pass_rate = float(np.mean(list(trajectory_success.values())))
        task_pass_rates.append(pass_rate)
        for traj_uid, is_success in trajectory_success.items():
            traj_idx_np = np.where((traj_uids == traj_uid) & valid_rows)[0]
            traj_idx = torch.as_tensor(traj_idx_np, dtype=torch.long, device=device)
            centered_success[traj_idx] = float(is_success) - pass_rate

    info_reward = info * centered_success
    info_return = torch.zeros_like(info)
    directed_sums = []
    success_weighted_directed_sums = []
    success_centered_directed_sums = []

    for traj_uid in dict.fromkeys(traj_uids[valid_rows].tolist()):
        idx_np = np.where((traj_uids == traj_uid) & valid_rows)[0]
        idx_np = idx_np[np.argsort(step_indices[idx_np], kind="stable")]
        running = torch.zeros((), dtype=info.dtype, device=device)
        for row_idx in idx_np[::-1]:
            running = info_reward[row_idx] + float(gamma) * running
            info_return[row_idx] = running
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        directed_sums.append(info[idx].sum())
        success_weighted_directed_sums.append(
            (info[idx] * float(successful_rows[idx_np].any())).sum())
        success_centered_directed_sums.append(info_reward[idx].sum())

    # A task-local baseline reduces variance. Every trajectory contributes
    # equal mass even when environment episode lengths differ.
    info_advantage = torch.zeros_like(info)
    for task_uid in dict.fromkeys(task_uids[valid_rows].tolist()):
        idx_np = np.where((task_uids == task_uid) & valid_rows)[0]
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        values = info_return[idx]
        # Give every trajectory equal mass when estimating the task-local
        # baseline; otherwise long trajectories would dominate the mean/std.
        task_traj_uids = traj_uids[idx_np]
        normalization_weights = torch.empty_like(values)
        for task_traj_uid in dict.fromkeys(task_traj_uids.tolist()):
            local_np = np.where(task_traj_uids == task_traj_uid)[0]
            local_idx = torch.as_tensor(local_np, dtype=torch.long, device=device)
            normalization_weights[local_idx] = 1.0 / len(local_np)
        if values.numel() > 0:
            weight_sum = normalization_weights.sum().clamp(min=1e-8)
            mean = (values * normalization_weights).sum() / weight_sum
            info_advantage[idx] = (values - mean) / float(fixed_scale)
        if advantage_clip is not None and float(advantage_clip) > 0:
            info_advantage[idx] = info_advantage[idx].clamp(
                min=-float(advantage_clip), max=float(advantage_clip))

    # If the actor averages action rows, this scale gives each trajectory total
    # mass 1 / num_trajectories regardless of its number of environment steps.
    trajectory_scale = torch.zeros_like(info)
    real_row_count = int(valid_rows.sum())
    real_traj_uids = list(dict.fromkeys(traj_uids[valid_rows].tolist()))
    num_trajectories = len(real_traj_uids)
    if real_row_count > 0 and num_trajectories > 0:
        for traj_uid in real_traj_uids:
            idx_np = np.where((traj_uids == traj_uid) & valid_rows)[0]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            trajectory_scale[idx] = real_row_count / (num_trajectories * len(idx_np))

    valid = torch.as_tensor(valid_rows, dtype=torch.bool, device=device)
    successful = torch.as_tensor(successful_rows & valid_rows, dtype=torch.bool, device=device)
    directed_sums_tensor = torch.stack(directed_sums) if directed_sums else torch.zeros(1, device=device)
    weighted_directed_sums_tensor = (
        torch.stack(success_weighted_directed_sums)
        if success_weighted_directed_sums else torch.zeros(1, device=device)
    )
    centered_directed_sums_tensor = (
        torch.stack(success_centered_directed_sums)
        if success_centered_directed_sums else torch.zeros(1, device=device)
    )
    metrics = {
        "action_ig/jsd_mean": info[valid].mean().item() if valid.any() else 0.0,
        "action_ig/jsd_max": info[valid].max().item() if valid.any() else 0.0,
        "action_ig/jsd_success_mean": info[successful].mean().item() if successful.any() else 0.0,
        "action_ig/directed_info_traj_mean": directed_sums_tensor.mean().item(),
        "action_ig/success_weighted_directed_info_traj_mean": weighted_directed_sums_tensor.mean().item(),
        "action_ig/success_centered_directed_info_traj_mean": centered_directed_sums_tensor.mean().item(),
        "action_ig/task_pass_rate_mean": float(np.mean(task_pass_rates)) if task_pass_rates else 0.0,
        "action_ig/centered_success_abs_mean": centered_success[valid].abs().mean().item() if valid.any() else 0.0,
        "action_ig/info_fixed_scale": float(fixed_scale),
        "action_ig/info_return_mean": info_return[valid].mean().item() if valid.any() else 0.0,
        "action_ig/info_adv_mean": info_advantage[valid].mean().item() if valid.any() else 0.0,
        "action_ig/info_adv_std": info_advantage[valid].std(unbiased=False).item() if valid.any() else 0.0,
        "action_ig/info_adv_min": info_advantage[valid].min().item() if valid.any() else 0.0,
        "action_ig/info_adv_max": info_advantage[valid].max().item() if valid.any() else 0.0,
        "action_ig/trajectory_scale_mean": trajectory_scale[valid].mean().item() if valid.any() else 0.0,
    }
    return info_advantage, trajectory_scale, metrics


def compute_information_action_weights(
    action_information: torch.Tensor,
    traj_uids: np.ndarray,
    task_uids: np.ndarray,
    successful_rows: np.ndarray,
    valid_rows: np.ndarray,
    schedule_scale: float = 1.0,
    max_weight_delta: float = 0.2,
    z_clip: float = 3.0,
    eps: float = 1e-6,
):
    """Turn Skill/No-skill information into bounded GRPO action weights.

    GRPO supplies the trajectory-level optimization direction.  Conditional
    information only redistributes credit among actions through a positive
    multiplier, so it cannot create an update when GRPO has zero advantage or
    reverse the sign of a task advantage.

    Information is standardized per task with equal mass per trajectory.
    ``4 p_q (1-p_q)`` gates the modulation by empirical task competence, making
    it vanish for all-success and all-failure rollout groups where action-level
    credit assignment is not identifiable from outcomes.
    """
    if not 0.0 <= float(schedule_scale) <= 1.0:
        raise ValueError(
            f"information schedule_scale must be in [0, 1], got {schedule_scale}"
        )
    if not 0.0 <= float(max_weight_delta) < 1.0:
        raise ValueError(
            "information max_weight_delta must be in [0, 1), "
            f"got {max_weight_delta}"
        )
    if float(z_clip) <= 0.0:
        raise ValueError(f"information z_clip must be positive, got {z_clip}")
    if float(eps) <= 0.0:
        raise ValueError(f"information eps must be positive, got {eps}")

    info = action_information.detach().float()
    traj_uids = np.asarray(traj_uids)
    task_uids = np.asarray(task_uids)
    successful_rows = np.asarray(successful_rows, dtype=bool)
    valid_rows = np.asarray(valid_rows, dtype=bool)
    if not (
        len(info)
        == len(traj_uids)
        == len(task_uids)
        == len(successful_rows)
        == len(valid_rows)
    ):
        raise ValueError("Action-information weight metadata lengths do not match")

    device = info.device
    weights = torch.ones_like(info)
    modulation = torch.zeros_like(info)
    competence_gate = torch.zeros_like(info)
    task_pass_rates = []
    effective_delta = float(schedule_scale) * float(max_weight_delta)

    for task_uid in dict.fromkeys(task_uids[valid_rows].tolist()):
        idx_np = np.where((task_uids == task_uid) & valid_rows)[0]
        task_traj_uids = list(dict.fromkeys(traj_uids[idx_np].tolist()))
        if not task_traj_uids:
            continue

        trajectory_success = []
        normalization_weights = torch.empty(
            len(idx_np), dtype=info.dtype, device=device
        )
        task_row_traj_uids = traj_uids[idx_np]
        for traj_uid in task_traj_uids:
            task_local_np = np.where(task_row_traj_uids == traj_uid)[0]
            global_traj_np = idx_np[task_local_np]
            trajectory_success.append(bool(successful_rows[global_traj_np].any()))
            local_idx = torch.as_tensor(
                task_local_np, dtype=torch.long, device=device
            )
            normalization_weights[local_idx] = 1.0 / len(task_local_np)

        pass_rate = float(np.mean(trajectory_success))
        task_pass_rates.append(pass_rate)
        gate = 4.0 * pass_rate * (1.0 - pass_rate)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        values = info[idx]
        weight_sum = normalization_weights.sum().clamp(min=float(eps))
        mean = (values * normalization_weights).sum() / weight_sum
        variance = (
            (values - mean).square() * normalization_weights
        ).sum() / weight_sum
        z_score = (values - mean) / torch.sqrt(variance + float(eps))
        z_score = z_score.clamp(min=-float(z_clip), max=float(z_clip))

        # Center tanh(z) once more because tanh is nonlinear. This preserves
        # unit average weight under the equal-trajectory-mass distribution.
        task_modulation = torch.tanh(z_score)
        task_modulation = task_modulation - (
            task_modulation * normalization_weights
        ).sum() / weight_sum
        max_abs = task_modulation.abs().max().clamp(min=1.0)
        task_modulation = task_modulation / max_abs

        modulation[idx] = task_modulation
        competence_gate[idx] = gate
        weights[idx] = 1.0 + effective_delta * gate * task_modulation

    valid = torch.as_tensor(valid_rows, dtype=torch.bool, device=device)
    successful = torch.as_tensor(
        successful_rows & valid_rows, dtype=torch.bool, device=device
    )
    weights[~valid] = 0.0
    metrics = {
        "action_ig/jsd_mean": info[valid].mean().item() if valid.any() else 0.0,
        "action_ig/jsd_max": info[valid].max().item() if valid.any() else 0.0,
        "action_ig/jsd_success_mean": (
            info[successful].mean().item() if successful.any() else 0.0
        ),
        "action_ig/task_pass_rate_mean": (
            float(np.mean(task_pass_rates)) if task_pass_rates else 0.0
        ),
        "action_ig/competence_gate_mean": (
            competence_gate[valid].mean().item() if valid.any() else 0.0
        ),
        "action_ig/modulation_mean": (
            modulation[valid].mean().item() if valid.any() else 0.0
        ),
        "action_ig/modulation_std": (
            modulation[valid].std(unbiased=False).item() if valid.any() else 0.0
        ),
        "action_ig/modulation_min": (
            modulation[valid].min().item() if valid.any() else 0.0
        ),
        "action_ig/modulation_max": (
            modulation[valid].max().item() if valid.any() else 0.0
        ),
        "action_ig/action_weight_mean": (
            weights[valid].mean().item() if valid.any() else 1.0
        ),
        "action_ig/action_weight_std": (
            weights[valid].std(unbiased=False).item() if valid.any() else 0.0
        ),
        "action_ig/action_weight_min": (
            weights[valid].min().item() if valid.any() else 1.0
        ),
        "action_ig/action_weight_max": (
            weights[valid].max().item() if valid.any() else 1.0
        ),
        "action_ig/effective_weight_delta": effective_delta,
        # Compatibility aliases for existing log parsers.
        "action_ig/info_adv_mean": (
            modulation[valid].mean().item() if valid.any() else 0.0
        ),
        "action_ig/info_adv_std": (
            modulation[valid].std(unbiased=False).item() if valid.any() else 0.0
        ),
        "action_ig/info_adv_min": (
            modulation[valid].min().item() if valid.any() else 0.0
        ),
        "action_ig/info_adv_max": (
            modulation[valid].max().item() if valid.any() else 0.0
        ),
        "action_ig/trajectory_scale_mean": 1.0,
    }
    return weights, modulation, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO

        # Check if contrastive mode provides context types for probe-based advantage
        contrastive_context_types = kwargs.get('contrastive_context_types', None)
        if contrastive_context_types is not None:
            omega = kwargs.get('contrastive_omega', 1.0)
            ema_delta = kwargs.get('ema_delta', None)
            adv2_clip = kwargs.get('adv2_clip', 3.0)
            advantages, returns = core_algos.compute_grpo_decomposed_contrastive_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                traj_index=data.non_tensor_batch['traj_uid'],
                contrastive_context_types=contrastive_context_types,
                omega=omega,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                ema_delta=ema_delta,
                adv2_clip=adv2_clip,
            )
        else:
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            padding_rows = data.non_tensor_batch.get('_is_padding', None)
            valid_rows = None if padding_rows is None else ~np.asarray(padding_rows, dtype=bool)
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                traj_index=data.non_tensor_batch['traj_uid'],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                compute_mean_std_cross_steps=kwargs.get('compute_mean_std_cross_steps', True),
                valid_rows=valid_rows,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        val_dataset_ood: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
        val_envs_ood=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.val_envs_ood = val_envs_ood
        self.traj_collector = traj_collector

        # Previous-W-round statistics for the four-quadrant router.  The
        # current round is appended only after routing, preventing target
        # leakage into its own thresholds.
        ours_cfg = config.env.get('ours', {})
        self._routing_window_size = ours_cfg.get('window_size', 5)
        self._routing_window = deque(maxlen=self._routing_window_size)
        self._routing_std_window = deque(maxlen=self._routing_window_size)
        routing_window_init = ours_cfg.get('routing_window_init', [])
        for value in list(routing_window_init)[-self._routing_window_size:]:
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"routing_window_init values must be in [0, 1], got {value}")
            self._routing_window.append(value)
        std_window_init = ours_cfg.get('std_window_init', [])
        for value in list(std_window_init)[-self._routing_window_size:]:
            value = float(value)
            if not 0.0 <= value <= 0.5:
                raise ValueError(f"std_window_init values must be in [0, 0.5], got {value}")
            self._routing_std_window.append(np.asarray([value], dtype=float))
        self._routing_threshold = None  # Will be computed from window
        self._routing_std_threshold = None
        # Sliding window for delta baseline (cross-task skill utilization)
        utilize_cfg = config.env.get('utilize', {})
        delta_window_size = utilize_cfg.get('delta_window_size', 5)
        self._delta_window = deque(maxlen=delta_window_size)

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler,
                                val_dataset_ood=val_dataset_ood)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        effective_rollout_n = config.actor_rollout_ref.rollout.n
        real_train_batch_size = config.data.train_batch_size * effective_rollout_n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler,
                           val_dataset_ood=None):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # Val ID dataloader
        # In OOD mode with env-generated tasks (e.g. ALFWorld), val_dataset is a
        # Subset placeholder so batch_size = len(dataset).
        # In OOD mode with data-driven tasks (e.g. Search), val_dataset has real
        # rows and should use config.data.val_batch_size to iterate in batches.
        if val_dataset_ood is not None:
            # Use config batch size if explicitly set; otherwise fall back to
            # len(dataset) for backward compat with ALFWorld-style placeholders.
            val_batch_size = self.config.data.get("val_batch_size", None)
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)
        else:
            val_batch_size = self.config.data.val_batch_size
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Val OOD dataloader (if provided)
        self.val_dataloader_ood = None
        if val_dataset_ood is not None:
            val_ood_batch_size = self.config.data.get("val_ood_batch_size", None)
            if val_ood_batch_size is None:
                # Fall back to val_batch_size, then to len(dataset)
                val_ood_batch_size = self.config.data.get("val_batch_size", None)
            if val_ood_batch_size is None:
                val_ood_batch_size = len(val_dataset_ood)
            self.val_dataloader_ood = StatefulDataLoader(
                dataset=val_dataset_ood,
                batch_size=val_ood_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            print(f"Val OOD dataloader: batch_size={val_ood_batch_size}, batches={len(self.val_dataloader_ood)}")

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path,
                          traj_uids=None, uids=None, extra_meta=None):
        """Dump rollout samples as JSONL, grouped by trajectory.

        When traj_uids is provided, steps belonging to the same trajectory are
        merged into a single JSON entry (matching val dump format).  Otherwise
        falls back to one-step-per-line (legacy behaviour).

        Args:
            extra_meta: Optional dict of extra fields to include in each trajectory entry.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)

        if traj_uids is not None and len(traj_uids) == n:
            # ── Grouped-by-trajectory format (aligned with val dump) ──
            from collections import OrderedDict
            traj_groups = OrderedDict()  # traj_uid -> {steps, scores, uid, extras}
            for i in range(n):
                uid_str = str(traj_uids[i])
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {
                        "steps": [],
                        "scores": [],
                        "uid": str(uids[i]) if uids is not None else None,
                    }
                traj_groups[uid_str]["steps"].append({"input": inputs[i], "output": outputs[i]})
                traj_groups[uid_str]["scores"].append(scores[i])

            with open(filename, "w") as f:
                for traj_uid_str, traj_data in traj_groups.items():
                    traj_score = sum(traj_data["scores"])  # total reward for this trajectory
                    entry = {
                        "traj_uid": traj_uid_str,
                        "uid": traj_data["uid"],
                        "score": traj_score,
                        "num_steps": len(traj_data["steps"]),
                        "step_scores": traj_data["scores"],
                        "steps": traj_data["steps"],
                        "global_step": self.global_steps,
                    }
                    if extra_meta:
                        entry.update(extra_meta)
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Dumped {len(traj_groups)} trajectories to {filename}")
        else:
            # ── Legacy flat format (one step per line) ──
            base_data = {
                "input": inputs,
                "output": outputs,
                "score": scores,
                "step": [self.global_steps] * n,
            }

            for k, v in reward_extra_infos_dict.items():
                if len(v) == n:
                    base_data[k] = v

            with open(filename, "w") as f:
                for i in range(n):
                    entry = {k: v[i] for k, v in base_data.items()}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Dumped {n} steps (flat) to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # Per-step data for trajectory dump
        all_step_inputs = []
        all_step_outputs = []
        all_step_traj_uids = []

        # Reset eval cursor for sequential full-coverage evaluation
        if hasattr(self.val_envs, 'reset_eval_cursor'):
            self.val_envs.reset_eval_cursor()

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            ################ agent-environment loop ###############
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                                                    gen_batch=test_gen_batch,
                                                    actor_rollout_wg=self.actor_rollout_wg,
                                                    envs=self.val_envs,
                                                    is_train=False,
                                                    )
            print('validation generation end')
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs (per-step, flattened across all steps)
            step_input_ids = test_output_gen_batch.batch["input_ids"]
            step_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in step_input_ids]
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            step_traj_uids = test_output_gen_batch.non_tensor_batch['traj_uid']

            # Collect per-step data for trajectory dump
            all_step_inputs.extend(step_input_texts)
            all_step_outputs.extend(output_texts)
            all_step_traj_uids.extend(step_traj_uids)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            # success rate
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all success_rate should be the same
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Dump full multi-turn trajectories grouped by traj_uid
        val_dump_path = self.config.trainer.get("val_dump_path", None)
        if val_dump_path and all_step_traj_uids:
            os.makedirs(val_dump_path, exist_ok=True)
            filename = os.path.join(val_dump_path, f"{self.global_steps}.jsonl")
            # Group steps by traj_uid, preserving order
            from collections import OrderedDict
            traj_groups = OrderedDict()
            for inp, out, uid in zip(all_step_inputs, all_step_outputs, all_step_traj_uids):
                uid_str = str(uid)
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {"steps": []}
                traj_groups[uid_str]["steps"].append({"input": inp, "output": out})
            # Compute per-trajectory score from sample_scores (one score per step, same within a traj)
            traj_uids_flat = np.concatenate(traj_uid_list, axis=0)
            scores_flat = np.array(sample_scores)
            uid_to_score = {}
            for uid, score in zip(traj_uids_flat, scores_flat):
                uid_str = str(uid)
                if uid_str not in uid_to_score:
                    uid_to_score[uid_str] = score
            with open(filename, "w") as f:
                for uid_str, traj_data in traj_groups.items():
                    entry = {
                        "traj_uid": uid_str,
                        "score": uid_to_score.get(uid_str, 0.0),
                        "num_steps": len(traj_data["steps"]),
                        "steps": traj_data["steps"],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Dumped {len(traj_groups)} trajectories to {filename}")

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        # The multi-turn batch contains one row per trajectory step. Keep one
        # row per trajectory so questions that take more search steps do not
        # receive more weight in accuracy.
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_rewards = reward_tensor[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(unique_rewards.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(unique_rewards[i].item())

        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each
        # trajectory, so the same unique-trajectory indices are used here.
        data_source_tool_calling = {}

        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)
            normalized_source = str(data_source).lower()
            if normalized_source.startswith('searchr1_'):
                normalized_source = normalized_source[len('searchr1_'):]
            metric_dict[f'val/{normalized_source}/accuracy'] = float(np.mean(rewards))
            metric_dict[f'val/{normalized_source}/num_samples'] = len(rewards)

        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

        # Search-QA domain aggregates. NQ and HotpotQA are the training/ID
        # domains; the remaining Search-R1 evaluation sources are OOD.
        search_id_sources = {'nq', 'hotpotqa'}
        search_ood_sources = {
            'triviaqa', 'popqa', '2wikimultihopqa', 'musique', 'bamboogle'
        }

        def search_domain(data_source):
            source = str(data_source).lower()
            if source.startswith('searchr1_'):
                source = source[len('searchr1_'):]
            if source in search_id_sources:
                return 'id'
            if source in search_ood_sources:
                return 'ood'
            return None

        domain_rewards = {'id': [], 'ood': []}
        for data_source, rewards in data_source_reward.items():
            domain = search_domain(data_source)
            if domain is not None:
                domain_rewards[domain].extend(rewards)

        domain_tool_calls = {'id': [], 'ood': []}
        for data_source, tool_calls in data_source_tool_calling.items():
            domain = search_domain(data_source)
            if domain is not None:
                domain_tool_calls[domain].extend(tool_calls)

        for domain in ('id', 'ood'):
            if domain_rewards[domain]:
                domain_score = float(np.mean(domain_rewards[domain]))
                metric_dict[f'val/{domain}/test_score'] = domain_score
                metric_dict[f'val/{domain}/accuracy'] = domain_score
                metric_dict[f'val/{domain}/success_rate'] = domain_score
                metric_dict[f'val/{domain}/num_samples'] = len(domain_rewards[domain])
            if domain_tool_calls[domain]:
                metric_dict[f'val/{domain}/tool_call_count/mean'] = float(
                    np.mean(domain_tool_calls[domain])
                )

        # Use a balanced ID/OOD score for checkpoint selection. The validation
        # files intentionally contain the same number of ID and OOD examples,
        # but taking the two means explicitly keeps selection balanced even if
        # those file sizes change later.
        populated_domain_scores = [
            float(np.mean(domain_rewards[domain]))
            for domain in ('id', 'ood')
            if domain_rewards[domain]
        ]
        if populated_domain_scores:
            metric_dict['val/search_qa/accuracy'] = float(
                np.mean(populated_domain_scores)
            )

        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v
        metric_dict['val/num_trajs'] = len(unique_traj_uid)

        # === Skill Bank 动态更新 ===
        if self.config.env.get('skills_only_memory', {}).get('enable_dynamic_update', False):
            self._update_skills_from_validation(
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_scores=sample_scores,
                success_rate=success_rate,
            )

        return metric_dict

    def _validate_ood(self):
        """Run validation on OOD environments with 'val_ood/' metric prefix.

        Mirrors _validate() logic but uses self.val_envs_ood and does NOT
        trigger skill dynamic update.
        """
        assert self.val_dataloader_ood is not None, \
            "_validate_ood requires val_dataloader_ood (pass val_dataset_ood to RayPPOTrainer)"
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        all_step_inputs = []
        all_step_outputs = []
        all_step_traj_uids = []

        # Reset eval cursor for sequential full-coverage evaluation
        if hasattr(self.val_envs_ood, 'reset_eval_cursor'):
            self.val_envs_ood.reset_eval_cursor()

        for test_data in self.val_dataloader_ood:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # Use OOD environments
            test_output_gen_batch = self.traj_collector.multi_turn_loop(
                gen_batch=test_gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.val_envs_ood,
                is_train=False,
            )
            print('OOD validation generation end')
            del test_batch
            test_batch = test_output_gen_batch

            step_input_ids = test_output_gen_batch.batch["input_ids"]
            step_input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in step_input_ids]
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            step_traj_uids = test_output_gen_batch.non_tensor_batch['traj_uid']

            all_step_inputs.extend(step_input_texts)
            all_step_outputs.extend(output_texts)
            all_step_traj_uids.extend(step_traj_uids)

            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", np.array(["ood"] * reward_tensor.shape[0])))
            tool_calling_list.append(test_batch.non_tensor_batch.get("tool_callings", np.zeros(reward_tensor.shape[0])))
            traj_uid_list.append(test_batch.non_tensor_batch.get("traj_uid", np.arange(reward_tensor.shape[0])))

            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    vals = test_batch.non_tensor_batch[k]
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    for i in range(len(vals)):
                        if i == 0 or test_batch.non_tensor_batch.get('traj_uid', [None]*len(vals))[i] != test_batch.non_tensor_batch.get('traj_uid', [None]*len(vals))[i-1]:
                            success_rate_dict[k].append(vals[i])

        # Dump OOD trajectories
        val_dump_path = self.config.trainer.get("val_dump_path", None)
        if val_dump_path and all_step_traj_uids:
            ood_dump_path = os.path.join(val_dump_path, "ood")
            os.makedirs(ood_dump_path, exist_ok=True)
            filename = os.path.join(ood_dump_path, f"{self.global_steps}.jsonl")
            from collections import OrderedDict
            traj_groups = OrderedDict()
            for inp, out, uid in zip(all_step_inputs, all_step_outputs, all_step_traj_uids):
                uid_str = str(uid)
                if uid_str not in traj_groups:
                    traj_groups[uid_str] = {"steps": []}
                traj_groups[uid_str]["steps"].append({"input": inp, "output": out})
            traj_uids_flat = np.concatenate(traj_uid_list, axis=0)
            scores_flat = np.array(sample_scores)
            uid_to_score = {}
            for uid, score in zip(traj_uids_flat, scores_flat):
                uid_str = str(uid)
                if uid_str not in uid_to_score:
                    uid_to_score[uid_str] = score
            with open(filename, "w") as f:
                for uid_str, traj_data in traj_groups.items():
                    entry = {
                        "traj_uid": uid_str,
                        "score": uid_to_score.get(uid_str, 0.0),
                        "num_steps": len(traj_data["steps"]),
                        "steps": traj_data["steps"],
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Dumped {len(traj_groups)} OOD trajectories to {filename}")

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]
        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val_ood/{data_source}/test_score'] = np.mean(rewards)
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val_ood/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
        for k, v in success_rate.items():
            metric_dict[f'val_ood/{k}'] = v
        metric_dict['val_ood/num_trajs'] = len(unique_traj_uid)

        return metric_dict

    def _update_skills_from_validation(
        self,
        sample_inputs: list,
        sample_outputs: list,
        sample_scores: list,
        success_rate: dict,
    ):
        """
        根据 validation 结果更新 skill bank。

        仅在特定任务类型成功率低于阈值时触发更新。
        """
        update_config = self.config.env.skills_only_memory
        threshold = update_config.get('update_threshold', 0.5)

        # 检查是否需要更新（某个任务类型成功率低于阈值）
        needs_update = False
        low_success_tasks = []
        for task_key, rate in success_rate.items():
            if rate < threshold:
                needs_update = True
                # 从 key 提取 task_type (e.g., "pick_and_place_success_rate" -> "pick_and_place")
                task_type = task_key.replace('_success_rate', '')
                low_success_tasks.append(task_type)

        if not needs_update:
            print(f"[SkillUpdate] All task success rates above {threshold}, skipping update")
            return

        print(f"[SkillUpdate] Low success tasks: {low_success_tasks}, triggering skill update...")

        # 收集失败 trajectories
        failed_trajectories = self._collect_failed_trajectories(
            sample_inputs, sample_outputs, sample_scores
        )

        if not failed_trajectories:
            print("[SkillUpdate] No failed trajectories found")
            return

        # 初始化 SkillUpdater (lazy init, 使用 Azure OpenAI o3)
        if not hasattr(self, 'skill_updater'):
            from agent_system.memory.skill_updater import SkillUpdater
            self.skill_updater = SkillUpdater(
                max_new_skills_per_update=update_config.get('max_new_skills', 3),
            )

        # 获取当前 skills —— 从 train envs 读取，因为新 skill 会加到 train envs，
        # 如果从 val_envs 读，_next_dyn_index 看不到已有的 dyn_* ID，导致 ID 冲突。
        train_memory = self.envs.retrieval_memory if (
            hasattr(self, 'envs') and hasattr(self.envs, 'retrieval_memory')
        ) else None
        if train_memory is None:
            print("[SkillUpdate] No retrieval_memory found in training envs")
            return

        # 分析失败并生成新 skills
        print(f"[SkillUpdate] Analyzing {len(failed_trajectories)} failed trajectories ...")
        new_skills = self.skill_updater.analyze_failures(
            failed_trajectories=failed_trajectories,
            current_skills=train_memory.skills,
        )

        if new_skills:
            # Add to both training and validation envs so that new skills
            # are used in subsequent training rollouts AND validation rollouts.
            added_train = train_memory.add_skills(new_skills, category='general')
            print(f"[SkillUpdate] Added {added_train} new skills to training envs")

            added_val = 0
            if hasattr(self, 'val_envs') and hasattr(self.val_envs, 'retrieval_memory') and self.val_envs.retrieval_memory:
                added_val = self.val_envs.retrieval_memory.add_skills(new_skills, category='general')
                print(f"[SkillUpdate] Added {added_val} new skills to validation envs")

            # Save updated skill bank to disk.
            if added_train > 0:
                save_dir = self.config.trainer.get('default_local_dir', './outputs')
                save_path = os.path.join(save_dir, f'updated_skills_step{self.global_steps}.json')
                train_memory.save_skills(save_path)
                print(f"[SkillUpdate] Saved updated skill bank to {save_path}")
            else:
                print("[SkillUpdate] All generated skills were duplicates, skipping save")
        else:
            print("[SkillUpdate] No new skills generated")

    def _collect_failed_trajectories(
        self,
        inputs: list,
        outputs: list,
        scores: list,
    ) -> list:
        """收集失败的 trajectories 用于分析"""
        failed = []
        for inp, out, score in zip(inputs, outputs, scores):
            if score <= 0:  # 失败的 trajectory
                task_type = self._detect_task_type_from_input(inp)
                task_desc = self._extract_task_description(inp)
                trajectory = self._parse_conversation_to_steps(inp, out)
                failed.append({
                    'task': task_desc,
                    'trajectory': trajectory,
                    'task_type': task_type,
                })
        return failed[:10]  # 限制数量，避免 prompt 过长

    def _extract_task_description(self, inp: str) -> str:
        """Extract the task description from a full conversation prompt."""
        import re
        # Common patterns used in ALFWorld, WebShop, OpenClaw, etc.
        patterns = [
            r'(?:Your task is to|Task:|task is to|you need to)[:\s]+(.*?)(?:\n|$)',
            r'(?:goal|objective)[:\s]+(.*?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, inp, re.IGNORECASE)
            if m:
                return m.group(1).strip()[:1000]
        # Fallback: first user turn (skip system prompt)
        for marker in ('<|im_start|>user\n', '\nHuman: ', '\nUser: '):
            idx = inp.find(marker)
            if idx >= 0:
                start = idx + len(marker)
                return inp[start:start + 1000]
        return inp[:1000]

    def _parse_conversation_to_steps(self, inp: str, out: str) -> list:
        """
        Parse a full decoded conversation into a list of trajectory steps.

        Each step is ``{'action': str, 'observation': str}`` where
        ``observation`` is the environment feedback (user/tool turn) and
        ``action`` is the agent response (assistant turn).

        Falls back to treating the whole ``inp`` as the initial context when
        no structured turn markers are found.
        """
        import re
        steps = []

        # --- ChatML / Qwen format -------------------------------------------
        user_turns = re.findall(
            r'<\|im_start\|>user\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        asst_turns = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', inp, re.DOTALL
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            # Final (failed) action has no follow-up observation
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Human / Assistant format ----------------------------------------
        user_turns = re.findall(
            r'(?:Human|User):\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        asst_turns = re.findall(
            r'Assistant:\s*(.*?)(?=(?:Human|User|Assistant):|$)',
            inp, re.DOTALL | re.IGNORECASE,
        )
        if user_turns and asst_turns:
            for obs, act in zip(user_turns, asst_turns):
                steps.append({
                    'action': act.strip()[:1500],
                    'observation': obs.strip()[:800],
                })
            steps.append({'action': out[:2000], 'observation': ''})
            return steps

        # --- Fallback: treat full inp as initial context ---------------------
        steps.append({'action': '', 'observation': inp[:3000]})
        steps.append({'action': out[:2000], 'observation': ''})
        return steps

    def _detect_task_type_from_input(self, inp: str) -> str:
        """从输入中检测任务类型"""
        inp_lower = inp.lower()
        if 'clean' in inp_lower:
            return 'clean'
        elif 'heat' in inp_lower:
            return 'heat'
        elif 'cool' in inp_lower:
            return 'cool'
        elif 'look at' in inp_lower and ('lamp' in inp_lower or 'light' in inp_lower):
            return 'look_at_obj_in_light'
        elif 'examine' in inp_lower:
            return 'examine'
        else:
            return 'pick_and_place'

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_best_checkpoint_state(self):
        """Restore the best-validation record when resuming a training run."""
        self.best_validation_score = None
        self.best_validation_step = None
        metadata_path = os.path.join(
            self.config.trainer.default_local_dir, "best_checkpoint.json"
        )
        if not os.path.isfile(metadata_path):
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.best_validation_score = float(metadata["score"])
            self.best_validation_step = int(metadata["global_step"])
            print(
                "Restored best validation checkpoint: "
                f"step={self.best_validation_step}, "
                f"score={self.best_validation_score}"
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not restore {metadata_path}: {exc}")

    def _best_checkpoint_candidate(self, val_metrics):
        """Return (score, metric_name) when this validation is a new best."""
        if not self.config.trainer.get("select_best_checkpoint", False):
            return None

        metric_name = self.config.trainer.get(
            "best_checkpoint_metric", "val/search_qa/accuracy"
        )
        if metric_name not in val_metrics:
            available = sorted(
                key for key in val_metrics if key.startswith("val/")
            )
            raise KeyError(
                f"Best-checkpoint metric {metric_name!r} was not produced. "
                f"Available validation metrics: {available}"
            )

        score = float(val_metrics[metric_name])
        mode = self.config.trainer.get("best_checkpoint_mode", "max")
        if mode not in ("max", "min"):
            raise ValueError(
                f"trainer.best_checkpoint_mode must be 'max' or 'min', got {mode!r}"
            )
        is_better = (
            self.best_validation_score is None
            or (mode == "max" and score > self.best_validation_score)
            or (mode == "min" and score < self.best_validation_score)
        )
        return (score, metric_name) if is_better else None

    def _record_best_checkpoint(self, score, metric_name):
        """Record the already-saved checkpoint selected by validation."""
        checkpoint_dir = os.path.abspath(
            os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}",
            )
        )
        self.best_validation_score = float(score)
        self.best_validation_step = int(self.global_steps)
        metadata = {
            "checkpoint": checkpoint_dir,
            "global_step": self.best_validation_step,
            "metric": metric_name,
            "score": self.best_validation_score,
            "mode": self.config.trainer.get("best_checkpoint_mode", "max"),
        }
        output_dir = self.config.trainer.default_local_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "best_checkpoint.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        with open(
            os.path.join(output_dir, "best_checkpoint.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(checkpoint_dir + "\n")
        print(
            f"New best checkpoint: {checkpoint_dir} "
            f"({metric_name}={score:.6f})"
        )

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _update_easy_with_action_ig(
        self,
        plain_output,
        easy_task_indices,
        step_task_indices,
        timing_raw,
        metrics,
        action_ig_active=True,
        action_ig_scale=1.0,
    ):
        """Update utilization tasks with information-weighted standard GRPO.

        No No-skill environment trajectory is generated. Skill and No-skill
        policies are evaluated at the same student-visited state; their bounded
        symmetric JSD supplies a positive per-action credit multiplier. The
        original GRPO advantage keeps its sign and aggregation, so information
        can emphasize useful actions but cannot replace the task objective.
        """
        easy_mask = np.array([idx in easy_task_indices for idx in step_task_indices])
        easy_idxs = np.where(easy_mask)[0]
        easy_batch = plain_output.select_idxs(easy_idxs)
        bs_easy_real = len(easy_batch)

        if action_ig_active:
            required_probe_keys = {
                'no_skill_probe_input_ids',
                'no_skill_probe_attention_mask',
                'no_skill_probe_position_ids',
            }
            missing = required_probe_keys - set(easy_batch.batch.keys())
            if missing:
                raise RuntimeError(
                    f"Action-IG is enabled but easy samples are missing no-skill prompts: {sorted(missing)}")

        easy_batch = adjust_batch(self.config, easy_batch)
        is_padding = np.zeros(len(easy_batch), dtype=bool)
        is_padding[bs_easy_real:] = True
        easy_batch.non_tensor_batch['_is_padding'] = is_padding
        easy_batch.batch['response_mask'] = compute_response_mask(easy_batch)

        if is_padding.any():
            padding_indices = torch.as_tensor(np.where(is_padding)[0], dtype=torch.long)
            easy_batch.batch['response_mask'][padding_indices] = 0.0
            if 'loss_mask' in easy_batch.batch:
                easy_batch.batch['loss_mask'][padding_indices] = 0.0

        if self.config.trainer.balance_batch:
            self._balance_batch(easy_batch, metrics=metrics)

        easy_batch.meta_info['global_token_num'] = torch.sum(
            easy_batch.batch['attention_mask'], dim=-1).tolist()

        with _timer('reward_easy', timing_raw):
            reward_tensor, reward_extra = compute_reward(easy_batch, self.reward_fn)
        easy_batch.batch['token_level_scores'] = reward_tensor
        if reward_extra:
            easy_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra.items()})

        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
            easy_batch, invalid_metrics = apply_invalid_action_penalty(
                easy_batch,
                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
            )
            metrics.update(invalid_metrics)

        if self.config.algorithm.use_kl_in_reward:
            easy_batch, kl_metrics = apply_kl_penalty(
                easy_batch, kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            easy_batch.batch['token_level_rewards'] = easy_batch.batch['token_level_scores']

        utilize_cfg = self.config.env.get('utilize', {})
        action_information = None
        if action_ig_active:
            # Two scoring forwards at the exact same student-visited state:
            # Skill and No-skill. The Skill logits are reused for old log-probs
            # and entropy, so this does not add a third model forward.
            easy_batch.meta_info['action_information_top_k'] = int(
                utilize_cfg.get('action_info_top_k', 64))
            easy_batch.meta_info['action_information_temperature'] = float(
                utilize_cfg.get('action_info_temperature', 1.0))
            with _timer('action_information_easy', timing_raw):
                action_info_output = self.actor_rollout_wg.compute_action_information(easy_batch)
            action_information = action_info_output.batch.pop('action_information')
            entropys = action_info_output.batch.pop('entropys')
            old_log_prob = action_info_output
        else:
            with _timer('old_log_prob_easy', timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(easy_batch)
            entropys = old_log_prob.batch.pop('entropys')

        # Student log-probability for the actually executed action.
        with _timer('entropy_metric_easy', timing_raw):
            response_mask = easy_batch.batch['response_mask']
            entropy_loss = agg_loss(
                loss_mat=entropys,
                loss_mask=response_mask,
                loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
            )
            metrics['actor/entropy_loss_easy'] = entropy_loss.detach().item()

        easy_batch = easy_batch.union(old_log_prob)

        if self.use_reference_policy:
            with _timer('ref_easy', timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(easy_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(easy_batch)
                easy_batch = easy_batch.union(ref_log_prob)

        norm_adv_by_std = self.config.algorithm.get('norm_adv_by_std_in_grpo', True)
        easy_batch = compute_advantage(
            easy_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.env.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std,
            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
            use_pf_ppo=self.config.algorithm.use_pf_ppo,
            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
            gigpo_mode=self.config.algorithm.gigpo.mode,
            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
            # Padding copies are excluded from the GRPO task statistics by
            # compute_advantage through the _is_padding metadata.
        )

        success_threshold = self.config.env.get('ours', {}).get('success_threshold', 0.0)
        episode_rewards = easy_batch.non_tensor_batch['episode_rewards']
        padding_rows = np.asarray(easy_batch.non_tensor_batch['_is_padding'], dtype=bool)
        valid_rows = ~padding_rows
        successful_rows = (episode_rewards > success_threshold) & valid_rows
        if action_ig_active:
            action_weights, information_modulation, action_ig_metrics = compute_information_action_weights(
                action_information=action_information,
                traj_uids=easy_batch.non_tensor_batch['traj_uid'],
                task_uids=easy_batch.non_tensor_batch['uid'],
                successful_rows=successful_rows,
                valid_rows=valid_rows,
                schedule_scale=action_ig_scale,
                max_weight_delta=utilize_cfg.get(
                    'action_info_max_weight_delta',
                    utilize_cfg.get('action_info_lambda', 0.1),
                ),
                z_clip=utilize_cfg.get('action_info_z_clip', 3.0),
            )
            effective_weight_delta = (
                float(
                    utilize_cfg.get(
                        'action_info_max_weight_delta',
                        utilize_cfg.get('action_info_lambda', 0.1),
                    )
                )
                * float(action_ig_scale)
            )
        else:
            action_weights = torch.ones(
                easy_batch.batch['advantages'].shape[0],
                dtype=easy_batch.batch['advantages'].dtype,
                device=easy_batch.batch['advantages'].device,
            )
            action_weights[torch.as_tensor(padding_rows, dtype=torch.bool)] = 0.0
            information_modulation = torch.zeros_like(action_weights)
            effective_weight_delta = 0.0
            action_ig_metrics = {
                'action_ig/jsd_mean': 0.0,
                'action_ig/jsd_max': 0.0,
                'action_ig/jsd_success_mean': 0.0,
                'action_ig/task_pass_rate_mean': 0.0,
                'action_ig/competence_gate_mean': 0.0,
                'action_ig/modulation_mean': 0.0,
                'action_ig/modulation_std': 0.0,
                'action_ig/modulation_min': 0.0,
                'action_ig/modulation_max': 0.0,
                'action_ig/action_weight_mean': 1.0,
                'action_ig/action_weight_std': 0.0,
                'action_ig/action_weight_min': 1.0,
                'action_ig/action_weight_max': 1.0,
                'action_ig/effective_weight_delta': 0.0,
                'action_ig/info_adv_mean': 0.0,
                'action_ig/info_adv_std': 0.0,
                'action_ig/info_adv_min': 0.0,
                'action_ig/info_adv_max': 0.0,
                'action_ig/trajectory_scale_mean': 1.0,
            }

        action_token_weights = action_weights.to(
            device=easy_batch.batch['advantages'].device,
            dtype=easy_batch.batch['advantages'].dtype,
        ).unsqueeze(-1)
        information_token_modulation = information_modulation.to(
            device=easy_batch.batch['advantages'].device,
            dtype=easy_batch.batch['advantages'].dtype,
        ).unsqueeze(-1)
        information_token_modulation = (
            information_token_modulation * easy_batch.batch['response_mask']
        )
        easy_batch.batch['action_information_modulation'] = (
            information_token_modulation
        )
        easy_batch.batch['action_information_weight'] = action_token_weights
        if action_information is not None:
            easy_batch.batch['action_information'] = action_information.to(
                device=easy_batch.batch['advantages'].device,
                dtype=easy_batch.batch['advantages'].dtype,
            )

        # Preserve standard GRPO exactly and use conditional information only
        # as a positive, bounded per-action credit multiplier.  No independent
        # information reward can override or reverse the environment signal.
        easy_batch.batch['advantages'] = (
            easy_batch.batch['advantages'] * action_token_weights
        )
        metrics.update(action_ig_metrics)
        metrics['action_ig/info_lambda'] = effective_weight_delta
        metrics['action_ig/update_mode_multiplicative'] = 1.0

        padding_mask = easy_batch.non_tensor_batch['_is_padding']
        if padding_mask.any():
            padding_indices = torch.as_tensor(np.where(padding_mask)[0], dtype=torch.long)
            easy_batch.batch['advantages'][padding_indices] = 0.0

        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer('update_actor_easy', timing_raw):
                easy_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
                easy_batch.meta_info['multi_turn'] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                easy_batch.meta_info['hdpo_mode'] = 'grpo'
                easy_output = self.actor_rollout_wg.update_actor(easy_batch)
            easy_metrics = reduce_metrics(easy_output.meta_info['metrics'])
            if 'actor/grad_norm' in easy_metrics:
                easy_metrics['actor/grad_norm_easy'] = easy_metrics.pop('actor/grad_norm')
            if 'actor/pg_loss' in easy_metrics:
                easy_metrics['actor/grpo_loss_easy'] = easy_metrics.pop('actor/pg_loss')
            if 'actor/kl_loss' in easy_metrics:
                easy_metrics['actor/kl_loss_easy'] = easy_metrics.pop('actor/kl_loss')
            metrics.update(easy_metrics)

        easy_update_mode = "information-weighted-grpo" if action_ig_active else "grpo"
        print(f"[Ours] Update 1 (easy/{easy_update_mode}): {bs_easy_real} real samples")
        return easy_batch, reward_extra

    def _update_filtered_grpo(
        self,
        plain_output,
        task_indices,
        step_task_indices,
        timing_raw,
        metrics,
        negative_advantage_weight=0.3,
    ):
        """Compute standard GRPO for tasks routed to the filtered quadrant."""
        source_mask = np.asarray(
            [idx in task_indices for idx in step_task_indices], dtype=bool)
        source_idxs = np.where(source_mask)[0]
        if len(source_idxs) == 0:
            return None, {}

        full_batch = plain_output.select_idxs(source_idxs)
        full_batch.batch["response_mask"] = compute_response_mask(full_batch)
        with _timer("reward_filtered_grpo", timing_raw):
            reward_tensor, reward_extra = compute_reward(full_batch, self.reward_fn)
        full_batch.batch["token_level_scores"] = reward_tensor
        if reward_extra:
            full_batch.non_tensor_batch.update(
                {key: np.asarray(value) for key, value in reward_extra.items()})

        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
            full_batch, invalid_metrics = apply_invalid_action_penalty(
                full_batch,
                invalid_action_penalty_coef=(
                    self.config.actor_rollout_ref.actor.invalid_action_penalty_coef),
            )
            metrics.update(invalid_metrics)

        if self.config.algorithm.use_kl_in_reward:
            with _timer("old_log_prob_filtered_all", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(full_batch)
                old_log_prob.batch.pop("entropys")
                full_batch = full_batch.union(old_log_prob)
            if self.use_reference_policy:
                with _timer("ref_filtered_all", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(full_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(full_batch)
                    full_batch = full_batch.union(ref_log_prob)
            full_batch, kl_metrics = apply_kl_penalty(
                full_batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            full_batch.batch["token_level_rewards"] = full_batch.batch["token_level_scores"]

        # The complete rollout group defines the GRPO baseline before the
        # negative-advantage coefficient is applied.
        with _timer("adv_filtered_grpo", timing_raw):
            full_batch = compute_advantage(
                full_batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.env.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True),
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
            )

        traj_uids = np.asarray(full_batch.non_tensor_batch["traj_uid"], dtype=object)
        unique_trajs = list(dict.fromkeys(traj_uids.tolist()))
        metrics["quadrant/filtered_source_trajectories"] = float(len(unique_trajs))
        metrics["quadrant/filtered_positive_trajectories"] = 0.0
        metrics["quadrant/filtered_kept_trajectories"] = float(len(unique_trajs))
        metrics["quadrant/filtered_negative_advantage_weight"] = 1.0
        metrics["quadrant/filtered_standard_grpo"] = 1.0

        filtered_batch = full_batch
        real_size = len(filtered_batch)
        filtered_batch = adjust_batch(self.config, filtered_batch)
        is_padding = np.zeros(len(filtered_batch), dtype=bool)
        is_padding[real_size:] = True
        filtered_batch.non_tensor_batch["_is_padding"] = is_padding
        filtered_batch.batch["response_mask"] = compute_response_mask(filtered_batch)
        if is_padding.any():
            padding_idxs = torch.as_tensor(np.where(is_padding)[0], dtype=torch.long)
            filtered_batch.batch["response_mask"][padding_idxs] = 0.0
            filtered_batch.batch["advantages"][padding_idxs] = 0.0
            if "loss_mask" in filtered_batch.batch:
                filtered_batch.batch["loss_mask"][padding_idxs] = 0.0

        valid_advantage_mask = filtered_batch.batch["response_mask"].bool()
        valid_advantage_count = valid_advantage_mask.sum().item()
        negative_advantage_mask = (
            (filtered_batch.batch["advantages"] < 0) & valid_advantage_mask)
        metrics["quadrant/filtered_negative_advantage_fraction"] = (
            float(negative_advantage_mask.sum().item() / valid_advantage_count)
            if valid_advantage_count > 0 else 0.0
        )

        if "old_log_probs" not in filtered_batch.batch:
            with _timer("old_log_prob_filtered", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(filtered_batch)
                old_log_prob.batch.pop("entropys")
                filtered_batch = filtered_batch.union(old_log_prob)
        if self.use_reference_policy and "ref_log_prob" not in filtered_batch.batch:
            with _timer("ref_filtered", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(filtered_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(filtered_batch)
                filtered_batch = filtered_batch.union(ref_log_prob)

        filtered_batch.meta_info["global_token_num"] = torch.sum(
            filtered_batch.batch["attention_mask"], dim=-1).tolist()
        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor_filtered_grpo", timing_raw):
                filtered_batch.meta_info["temperature"] = (
                    self.config.actor_rollout_ref.rollout.temperature)
                filtered_batch.meta_info["multi_turn"] = (
                    self.config.actor_rollout_ref.rollout.multi_turn.enable)
                filtered_batch.meta_info["hdpo_mode"] = "grpo"
                filtered_batch.meta_info["loss_agg_mode_override"] = (
                    "seq-mean-token-mean")
                output = self.actor_rollout_wg.update_actor(filtered_batch)
            output_metrics = reduce_metrics(output.meta_info["metrics"])
            if "actor/grad_norm" in output_metrics:
                output_metrics["actor/grad_norm_filtered_grpo"] = (
                    output_metrics.pop("actor/grad_norm"))
            if "actor/pg_loss" in output_metrics:
                output_metrics["actor/filtered_grpo_loss"] = (
                    output_metrics.pop("actor/pg_loss"))
            if "actor/kl_loss" in output_metrics:
                output_metrics["actor/kl_loss_filtered_grpo"] = (
                    output_metrics.pop("actor/kl_loss"))
            metrics.update(output_metrics)

        print(
            f"[Quadrant] Filtered quadrant uses standard GRPO on "
            f"{len(unique_trajs)} trajectories"
        )
        return filtered_batch, reward_extra

    def _update_trajectory_dpo(
        self,
        plain_output,
        task_indices,
        step_task_indices,
        timing_raw,
        metrics,
        beta=0.1,
        score_key="trajectory_webshop_task_score",
        top_k=2,
        min_score_gap=0.1,
    ):
        """Trajectory DPO using rank-paired top-k and bottom-k rollouts."""
        top_k = max(1, int(top_k))
        min_score_gap = float(min_score_gap)
        if min_score_gap < 0:
            raise ValueError(
                f"dpo_min_score_gap must be non-negative, got {min_score_gap}"
            )
        source_mask = np.asarray(
            [idx in task_indices for idx in step_task_indices], dtype=bool)
        source_idxs = np.where(source_mask)[0]
        if len(source_idxs) == 0:
            return None
        ppo_epochs = int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
        if ppo_epochs != 1:
            raise ValueError(
                "Trajectory DPO requires actor_rollout_ref.actor.ppo_epochs=1; "
                f"received {ppo_epochs}."
            )

        source = plain_output.select_idxs(source_idxs)
        task_uids = np.asarray(source.non_tensor_batch["uid"], dtype=object)
        traj_uids = np.asarray(source.non_tensor_batch["traj_uid"], dtype=object)
        if score_key in source.non_tensor_batch:
            preference_scores = np.asarray(
                source.non_tensor_batch[score_key], dtype=float)
            score_fallback = False
        else:
            # Non-WebShop environments do not expose a continuous task score.
            # Preserve their previous behavior while making the fallback
            # explicit in metrics.
            preference_scores = np.asarray(
                source.non_tensor_batch["episode_rewards"], dtype=float)
            score_fallback = True
        trajectory_rows = defaultdict(list)
        trajectory_meta = {}
        for row_idx, (task_uid, traj_uid, score) in enumerate(
            zip(task_uids, traj_uids, preference_scores)
        ):
            trajectory_rows[traj_uid].append(row_idx)
            trajectory_meta[traj_uid] = (task_uid, float(score))
        task_trajectories = defaultdict(list)
        for traj_uid, (task_uid, score) in trajectory_meta.items():
            task_trajectories[task_uid].append((score, traj_uid))

        selected_rows = []
        selected_pair_ids = []
        selected_signs = []
        pair_count = 0
        candidate_pair_count = 0
        pair_score_gaps = []
        for trajectories in task_trajectories.values():
            trajectories = sorted(trajectories, key=lambda item: item[0])
            num_rank_pairs = min(top_k, len(trajectories) // 2)
            for rank in range(num_rank_pairs):
                rejected_score, rejected_uid = trajectories[rank]
                chosen_score, chosen_uid = trajectories[-(rank + 1)]
                candidate_pair_count += 1
                score_gap = float(chosen_score - rejected_score)
                if score_gap <= min_score_gap:
                    continue
                for row_idx in trajectory_rows[chosen_uid]:
                    selected_rows.append(row_idx)
                    selected_pair_ids.append(pair_count)
                    selected_signs.append(1)
                for row_idx in trajectory_rows[rejected_uid]:
                    selected_rows.append(row_idx)
                    selected_pair_ids.append(pair_count)
                    selected_signs.append(-1)
                pair_score_gaps.append(score_gap)
                pair_count += 1
        metrics["quadrant/dpo_pairs"] = float(pair_count)
        metrics["quadrant/dpo_candidate_pairs"] = float(candidate_pair_count)
        metrics["quadrant/dpo_skipped_small_gap_pairs"] = float(
            candidate_pair_count - pair_count)
        metrics["quadrant/dpo_top_k"] = float(top_k)
        metrics["quadrant/dpo_min_score_gap"] = min_score_gap
        metrics["quadrant/dpo_score_fallback"] = float(score_fallback)
        if pair_score_gaps:
            metrics["quadrant/dpo_score_gap_mean"] = float(
                np.mean(pair_score_gaps))
            metrics["quadrant/dpo_score_gap_min"] = float(
                np.min(pair_score_gaps))
            metrics["quadrant/dpo_score_gap_max"] = float(
                np.max(pair_score_gaps))
        if pair_count == 0:
            return None

        dpo_batch = source.select_idxs(np.asarray(selected_rows, dtype=np.int64))
        dpo_batch.non_tensor_batch["dpo_pair_id"] = np.asarray(
            selected_pair_ids, dtype=np.int64)
        dpo_batch.non_tensor_batch["dpo_sign"] = np.asarray(
            selected_signs, dtype=np.int8)

        # Continuous scores construct preferences, while the common trainer
        # metrics still require canonical token reward fields on the batch
        # returned by _ours_step.
        with _timer("reward_dpo", timing_raw):
            dpo_reward_tensor, dpo_reward_extra = compute_reward(
                dpo_batch, self.reward_fn)
        dpo_batch.batch["token_level_scores"] = dpo_reward_tensor
        dpo_batch.batch["token_level_rewards"] = dpo_reward_tensor
        if dpo_reward_extra:
            dpo_batch.non_tensor_batch.update(
                {
                    key: np.asarray(value)
                    for key, value in dpo_reward_extra.items()
                }
            )

        real_size = len(dpo_batch)
        dpo_batch = adjust_batch(self.config, dpo_batch)
        is_padding = np.zeros(len(dpo_batch), dtype=bool)
        is_padding[real_size:] = True
        dpo_batch.non_tensor_batch["_is_padding"] = is_padding
        dpo_batch.batch["response_mask"] = compute_response_mask(dpo_batch)
        if is_padding.any():
            padding_idxs = torch.as_tensor(np.where(is_padding)[0], dtype=torch.long)
            dpo_batch.batch["response_mask"][padding_idxs] = 0.0
            if "loss_mask" in dpo_batch.batch:
                dpo_batch.batch["loss_mask"][padding_idxs] = 0.0

        with _timer("old_log_prob_dpo", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(dpo_batch)
            old_log_prob.batch.pop("entropys")
            dpo_batch = dpo_batch.union(old_log_prob)
        if self.use_reference_policy:
            with _timer("ref_dpo", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(dpo_batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(dpo_batch)
                dpo_batch = dpo_batch.union(ref_log_prob)
            reference_log_probs = dpo_batch.batch["ref_log_prob"]
            reference_fallback = False
        else:
            reference_log_probs = dpo_batch.batch["old_log_probs"].detach()
            reference_fallback = True

        response_length = dpo_batch.batch["responses"].shape[-1]
        if self.config.actor_rollout_ref.rollout.multi_turn.enable:
            effective_mask = dpo_batch.batch["loss_mask"][:, -response_length:]
        else:
            effective_mask = dpo_batch.batch["response_mask"]
        pair_ids = np.asarray(
            dpo_batch.non_tensor_batch["dpo_pair_id"], dtype=np.int64)
        signs = np.asarray(dpo_batch.non_tensor_batch["dpo_sign"], dtype=np.int8)
        valid_rows = ~is_padding
        advantages = torch.zeros_like(dpo_batch.batch["old_log_probs"])
        dpo_losses = []
        dpo_margins = []
        beta = float(beta)

        with torch.no_grad():
            for pair_id in range(pair_count):
                chosen_np = np.where(
                    valid_rows & (pair_ids == pair_id) & (signs == 1))[0]
                rejected_np = np.where(
                    valid_rows & (pair_ids == pair_id) & (signs == -1))[0]
                if len(chosen_np) == 0 or len(rejected_np) == 0:
                    continue
                chosen_rows = torch.as_tensor(chosen_np, dtype=torch.long)
                rejected_rows = torch.as_tensor(rejected_np, dtype=torch.long)
                chosen_mask = effective_mask[chosen_rows]
                rejected_mask = effective_mask[rejected_rows]
                chosen_tokens = chosen_mask.sum().clamp(min=1)
                rejected_tokens = rejected_mask.sum().clamp(min=1)
                policy_chosen = (
                    dpo_batch.batch["old_log_probs"][chosen_rows] * chosen_mask
                ).sum() / chosen_tokens
                policy_rejected = (
                    dpo_batch.batch["old_log_probs"][rejected_rows] * rejected_mask
                ).sum() / rejected_tokens
                ref_chosen = (
                    reference_log_probs[chosen_rows] * chosen_mask
                ).sum() / chosen_tokens
                ref_rejected = (
                    reference_log_probs[rejected_rows] * rejected_mask
                ).sum() / rejected_tokens
                margin = beta * (
                    (policy_chosen - policy_rejected)
                    - (ref_chosen - ref_rejected)
                )
                coefficient = beta * torch.sigmoid(-margin)
                advantages[chosen_rows] = coefficient / chosen_tokens * chosen_mask
                advantages[rejected_rows] = (
                    -coefficient / rejected_tokens * rejected_mask)
                dpo_losses.append(torch.nn.functional.softplus(-margin))
                dpo_margins.append(margin)

        dpo_batch.batch["advantages"] = advantages
        # GRPO's compute_advantage() normally creates both fields and, for
        # outcome GRPO, uses the same tensor for advantages and returns.
        # DPO builds its signed advantages directly, so mirror that contract
        # explicitly for the shared metric pipeline.
        dpo_batch.batch["returns"] = advantages.clone()
        dpo_batch.meta_info["global_token_num"] = torch.sum(
            dpo_batch.batch["attention_mask"], dim=-1).tolist()
        if self.config.trainer.critic_warmup <= self.global_steps:
            with _timer("update_actor_dpo", timing_raw):
                dpo_batch.meta_info["temperature"] = (
                    self.config.actor_rollout_ref.rollout.temperature)
                dpo_batch.meta_info["multi_turn"] = (
                    self.config.actor_rollout_ref.rollout.multi_turn.enable)
                dpo_batch.meta_info["hdpo_mode"] = "grpo"
                dpo_batch.meta_info["loss_agg_mode_override"] = (
                    "seq-mean-token-sum")
                dpo_batch.meta_info["disable_actor_kl_loss"] = True
                output = self.actor_rollout_wg.update_actor(dpo_batch)
            output_metrics = reduce_metrics(output.meta_info["metrics"])
            if "actor/grad_norm" in output_metrics:
                output_metrics["actor/grad_norm_dpo"] = (
                    output_metrics.pop("actor/grad_norm"))
            if "actor/pg_loss" in output_metrics:
                output_metrics["actor/dpo_surrogate_loss"] = (
                    output_metrics.pop("actor/pg_loss"))
            output_metrics.pop("actor/kl_loss", None)
            metrics.update(output_metrics)

        if dpo_losses:
            metrics["quadrant/dpo_loss"] = torch.stack(dpo_losses).mean().item()
            metrics["quadrant/dpo_margin"] = torch.stack(dpo_margins).mean().item()
        metrics["quadrant/dpo_beta"] = beta
        metrics["quadrant/dpo_reference_old_fallback"] = float(reference_fallback)
        print(
            f"[Quadrant] Trajectory DPO constructed {pair_count}/"
            f"{candidate_pair_count} top-{top_k}/bottom-{top_k} pairs "
            f"(min_score_gap={min_score_gap:.3f}, "
            f"score_fallback={score_fallback})"
        )
        return dpo_batch

    def _ours_step(self, gen_batch, timing_raw, metrics):
        """Run the adaptive router or the smooth internalize-to-utilize curriculum."""
        internalize_cfg = self.config.env.get('internalize', {})
        jsd_lambda = internalize_cfg.get('jsd_lambda', 1.0)
        action_jsd_lambda = internalize_cfg.get('action_jsd_lambda', 0.1)
        jsd_top_k = internalize_cfg.get('jsd_top_k', 64)
        jsd_temperature = internalize_cfg.get('jsd_temperature', 1.0)
        jsd_micro_batch_size_per_gpu = max(
            1,
            int(
                internalize_cfg.get(
                    'jsd_micro_batch_size_per_gpu',
                    self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                )
            ),
        )
        ours_cfg = self.config.env.get('ours', {})
        warmup_steps = ours_cfg.get('warmup_steps', 30)
        action_ig_ramp_steps = max(1, int(ours_cfg.get('action_ig_ramp_steps', 5)))
        action_ig_active = self.global_steps > warmup_steps
        action_ig_scale = (
            min(1.0, float(self.global_steps - warmup_steps) / action_ig_ramp_steps)
            if action_ig_active else 0.0
        )
        is_warmup = self.global_steps <= warmup_steps
        rollout_n = self.config.actor_rollout_ref.rollout.n
        env_rollout_n = self.config.env.rollout.n  # env workers per task (used for task_modes expansion)

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Plain rollout (specific skills only, exclude general+common)
        # ══════════════════════════════════════════════════════════════
        with _timer("gen_plain", timing_raw):
            self.envs.set_mode(plain=True)
            plain_output = self.traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                envs=self.envs,
                is_train=True,
            )

        # Keep the initial task states for the legacy full-trajectory warmup.
        reset_info = self.envs.get_last_reset_info()

        # Dump plain trajectories immediately (before balance_batch reorders)
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            plain_inputs = self.tokenizer.batch_decode(plain_output.batch["prompts"], skip_special_tokens=True)
            plain_outputs_text = self.tokenizer.batch_decode(plain_output.batch["responses"], skip_special_tokens=True)
            plain_scores = [float(x) for x in plain_output.non_tensor_batch['episode_rewards']]
            self._dump_generations(
                inputs=plain_inputs, outputs=plain_outputs_text, scores=plain_scores,
                reward_extra_infos_dict={}, dump_path=rollout_data_dir,
                traj_uids=plain_output.non_tensor_batch.get('traj_uid', None),
                uids=plain_output.non_tensor_batch.get('uid', None),
            )

        # ══════════════════════════════════════════════════════════════
        # Task-level pass rate computation (grouped by uid)
        # ══════════════════════════════════════════════════════════════
        episode_rewards = plain_output.non_tensor_batch['episode_rewards']
        continuous_score_key = str(
            ours_cfg.get(
                "continuous_score_key",
                "trajectory_webshop_task_score",
            )
        )
        if continuous_score_key in plain_output.non_tensor_batch:
            trajectory_continuous_scores = np.asarray(
                plain_output.non_tensor_batch[continuous_score_key],
                dtype=float,
            )
            continuous_score_fallback = False
        else:
            # Keep the shared trainer usable for environments without a dense
            # trajectory score. WebShop supplies the key above.
            trajectory_continuous_scores = np.asarray(
                episode_rewards, dtype=float)
            continuous_score_fallback = True
        if not np.all(np.isfinite(trajectory_continuous_scores)):
            raise ValueError(
                f"Non-finite values found in {continuous_score_key}"
            )
        group_uids = plain_output.non_tensor_batch['uid']
        traj_uids = plain_output.non_tensor_batch['traj_uid']

        # Build uid -> task_index mapping (stable ordinal based on first occurrence)
        uid_to_task_idx = {}
        for uid in group_uids:
            if uid not in uid_to_task_idx:
                uid_to_task_idx[uid] = len(uid_to_task_idx)

        # Deduplicate: get one reward per trajectory
        traj_reward = {}   # traj_uid -> reward
        traj_continuous_score = {}  # traj_uid -> dense WebShop task score
        traj_to_task = {}  # traj_uid -> task_index
        for g_uid, t_uid, r, continuous_score in zip(
            group_uids,
            traj_uids,
            episode_rewards,
            trajectory_continuous_scores,
        ):
            traj_reward[t_uid] = float(r)
            traj_continuous_score[t_uid] = float(continuous_score)
            traj_to_task[t_uid] = uid_to_task_idx[g_uid]

        # Group trajectory rewards/scores by task. Competence remains the
        # binary pass rate, while uncertainty is the population std of the
        # eight continuous WebShop task scores.
        task_rewards = defaultdict(list)
        task_continuous_scores = defaultdict(list)
        for t_uid, r in traj_reward.items():
            task_rewards[traj_to_task[t_uid]].append(r)
            task_continuous_scores[traj_to_task[t_uid]].append(
                traj_continuous_score[t_uid])

        n_total_tasks = len(task_rewards)
        task_pass_rates = {}
        task_score_stds = {}
        for tidx, rewards in task_rewards.items():
            success_flags = np.asarray([float(r > 0) for r in rewards], dtype=float)
            task_pass_rates[tidx] = float(success_flags.mean())
            continuous_scores = np.asarray(
                task_continuous_scores[tidx], dtype=float)
            task_score_stds[tidx] = float(continuous_scores.std(ddof=0))

        # Batch-level pass rate
        batch_pass_rate = np.mean(list(task_pass_rates.values())) if task_pass_rates else 0.0
        batch_std_median = (
            float(np.median(list(task_score_stds.values())))
            if task_score_stds else 0.0
        )

        # ══════════════════════════════════════════════════════════════
        # Four-quadrant routing (previous-W success mean and std median)
        # ══════════════════════════════════════════════════════════════
        # Thresholds use previous rounds only. The current batch is appended
        # after all tasks have been routed.
        success_history_len = len(self._routing_window)
        std_history_len = len(self._routing_std_window)
        if len(self._routing_window) > 0:
            self._routing_threshold = float(np.mean(list(self._routing_window)))
        else:
            self._routing_threshold = float(
                ours_cfg.get("success_threshold_default", 0.5))
        if len(self._routing_std_window) > 0:
            historical_stds = np.concatenate(list(self._routing_std_window))
            self._routing_std_threshold = float(np.median(historical_stds))
        else:
            self._routing_std_threshold = float(
                ours_cfg.get("std_threshold_default", 0.25))

        success_threshold = self._routing_threshold
        std_threshold = self._routing_std_threshold
        filtered_task_indices = {
            idx for idx in task_pass_rates
            if task_pass_rates[idx] > success_threshold
            and task_score_stds[idx] > std_threshold
        }
        dpo_task_indices = {
            idx for idx in task_pass_rates
            if task_pass_rates[idx] <= success_threshold
            and task_score_stds[idx] > std_threshold
        }
        contrastive_task_indices = {
            idx for idx in task_pass_rates
            if task_pass_rates[idx] > success_threshold
            and task_score_stds[idx] <= std_threshold
        }
        action_jsd_task_indices = {
            idx for idx in task_pass_rates
            if task_pass_rates[idx] <= success_threshold
            and task_score_stds[idx] <= std_threshold
        }

        curriculum_enabled = bool(ours_cfg.get("curriculum_enabled", False))
        curriculum_stage = "router"
        curriculum_stage_idx = -1
        curriculum_progress = 0.0
        internalize_weight = 0.0
        utilize_weight = 0.0
        curriculum_grpo_weight = 1.0
        jsd_length_guard_triggered = False
        plain_response_mask = compute_response_mask(plain_output)
        plain_response_lengths = plain_response_mask.sum(dim=-1).float()
        plain_response_capacity = max(1, int(plain_response_mask.shape[-1]))
        plain_response_mean_ratio = float(
            plain_response_lengths.mean().item() / plain_response_capacity
        )
        plain_response_clip_ratio = float(
            (plain_response_lengths >= plain_response_capacity).float().mean().item()
        )
        if curriculum_enabled:
            # The three objectives share one continuous schedule.  The
            # auxiliary objectives reach zero at their boundaries, so the
            # optimizer never sees a discontinuous JSD/information jump.
            all_task_indices = set(task_pass_rates.keys())
            total_steps = max(1, int(self.total_training_steps))
            current_step = max(1, int(self.global_steps))
            curriculum_progress = (
                float(current_step - 1) / float(max(1, total_steps - 1))
            )
            internalize_fraction = float(
                ours_cfg.get("curriculum_internalize_fraction", 0.2)
            )
            utilize_start_fraction = float(
                ours_cfg.get("curriculum_utilize_start_fraction", 0.7)
            )
            if not 0.0 < internalize_fraction < utilize_start_fraction < 1.0:
                raise ValueError(
                    "Curriculum fractions must satisfy 0 < internalize < "
                    f"utilize_start < 1, got {internalize_fraction} and "
                    f"{utilize_start_fraction}"
                )

            def _smoothstep(value):
                value = min(1.0, max(0.0, float(value)))
                return value * value * (3.0 - 2.0 * value)

            internalize_decay = 1.0 - _smoothstep(
                curriculum_progress / internalize_fraction
            )
            internalize_ramp = min(
                1.0, float(current_step) / float(action_ig_ramp_steps)
            )
            internalize_weight = internalize_ramp * internalize_decay
            if curriculum_progress > utilize_start_fraction:
                utilize_weight = _smoothstep(
                    (curriculum_progress - utilize_start_fraction)
                    / (1.0 - utilize_start_fraction)
                )
            min_aux_weight = float(
                ours_cfg.get("curriculum_min_aux_weight", 0.01)
            )
            if not 0.0 <= min_aux_weight < 1.0:
                raise ValueError(
                    "curriculum_min_aux_weight must be in [0, 1), "
                    f"got {min_aux_weight}"
                )
            # Avoid paying for Teacher/No-skill scoring when a smooth endpoint
            # has made the corresponding auxiliary update numerically trivial.
            if internalize_weight < min_aux_weight:
                internalize_weight = 0.0
            if utilize_weight < min_aux_weight:
                utilize_weight = 0.0

            grpo_floor = float(
                ours_cfg.get("curriculum_internalize_grpo_floor", 1.0)
            )
            if not 0.0 < grpo_floor <= 1.0:
                raise ValueError(
                    "curriculum_internalize_grpo_floor must be in (0, 1], "
                    f"got {grpo_floor}"
                )
            curriculum_grpo_weight = grpo_floor + (
                1.0 - grpo_floor
            ) * _smoothstep(curriculum_progress / internalize_fraction)

            # Stop the auxiliary JSD before an EOS/length failure becomes
            # self-reinforcing.  The current batch then falls back to pure
            # standard GRPO and can recover its output format.
            jsd_clip_guard = float(
                ours_cfg.get("curriculum_jsd_clip_ratio_guard", 0.05)
            )
            jsd_mean_guard = float(
                ours_cfg.get("curriculum_jsd_mean_length_ratio_guard", 0.4)
            )
            jsd_length_guard_triggered = bool(
                internalize_weight > 0.0
                and (
                    plain_response_clip_ratio >= jsd_clip_guard
                    or plain_response_mean_ratio >= jsd_mean_guard
                )
            )
            if jsd_length_guard_triggered:
                internalize_weight = 0.0

            filtered_task_indices = set()
            dpo_task_indices = set()
            action_jsd_task_indices = set()
            contrastive_task_indices = set()
            medium_task_indices = set(all_task_indices)
            is_warmup = False

            if internalize_weight > 0.0:
                curriculum_stage = "internalize"
                curriculum_stage_idx = 0
                # Internalization is competence-gated: a task that already has
                # a successful Student rollout does not keep receiving Teacher
                # JSD.  It still receives ordinary GRPO through the medium set.
                max_internalize_pass_rate = float(
                    ours_cfg.get("curriculum_internalize_max_pass_rate", 0.0)
                )
                action_jsd_task_indices = {
                    idx for idx, pass_rate in task_pass_rates.items()
                    if pass_rate <= max_internalize_pass_rate
                }
                medium_task_indices -= action_jsd_task_indices
                action_ig_active = bool(action_jsd_task_indices)
                action_ig_scale = internalize_weight
            elif utilize_weight > 0.0:
                curriculum_stage = "utilize"
                curriculum_stage_idx = 2
                contrastive_task_indices = set(all_task_indices)
                medium_task_indices = set()
                action_ig_active = True
                action_ig_scale = utilize_weight
            else:
                curriculum_stage = "grpo"
                curriculum_stage_idx = 1
                action_ig_active = False
                action_ig_scale = 0.0

        # Reuse the existing Hard/Medium/Easy update implementations as the
        # internalize/GRPO/utilize optimizers. In curriculum mode the task sets
        # above are disjoint, so every Student action receives exactly one
        # policy-gradient update; only the gated internalize set also gets JSD.
        hard_task_indices = action_jsd_task_indices
        easy_task_indices = contrastive_task_indices
        if not curriculum_enabled:
            medium_task_indices = set()
        n_hard = len(action_jsd_task_indices)
        n_medium = len(medium_task_indices)
        n_easy = len(contrastive_task_indices)

        metrics['routing/hard_ratio'] = (
            n_hard / n_total_tasks if n_total_tasks > 0 else 0.0)
        metrics['routing/medium_ratio'] = (
            n_medium / n_total_tasks if n_total_tasks > 0 else 0.0)
        metrics['routing/easy_ratio'] = (
            n_easy / n_total_tasks if n_total_tasks > 0 else 0.0)
        metrics['curriculum/enabled'] = float(curriculum_enabled)
        metrics['curriculum/stage_idx'] = float(curriculum_stage_idx)
        # Keep the old metric name as a compatibility alias for existing
        # plotting scripts, and expose the precise stage name as well.
        metrics['curriculum/action_jsd_stage'] = float(
            curriculum_stage == "internalize")
        metrics['curriculum/teacher_success_jsd_stage'] = 0.0
        metrics['curriculum/grpo_stage'] = float(curriculum_stage == "grpo")
        metrics['curriculum/action_noskill_skill_stage'] = float(
            curriculum_stage == "utilize")
        # Keep the old metric names as aliases for existing plotting scripts.
        metrics['curriculum/hard_stage'] = float(
            curriculum_stage == "internalize")
        metrics['curriculum/medium_stage'] = float(curriculum_stage == "grpo")
        metrics['curriculum/easy_stage'] = float(curriculum_stage == "utilize")
        metrics['curriculum/progress'] = float(curriculum_progress)
        metrics['curriculum/internalize_weight'] = float(internalize_weight)
        metrics['curriculum/grpo_weight'] = float(curriculum_grpo_weight)
        metrics['curriculum/utilize_weight'] = float(utilize_weight)
        metrics['curriculum/jsd_length_guard_triggered'] = float(
            jsd_length_guard_triggered
        )
        metrics['curriculum/plain_response_mean_ratio'] = float(
            plain_response_mean_ratio
        )
        metrics['curriculum/plain_response_clip_ratio'] = float(
            plain_response_clip_ratio
        )
        metrics['routing/threshold'] = success_threshold
        metrics['routing/std_threshold'] = std_threshold
        metrics['routing/score_std_threshold'] = std_threshold
        metrics['routing/batch_pass_rate'] = batch_pass_rate
        metrics['routing/batch_std_median'] = batch_std_median
        metrics['routing/batch_score_std_median'] = batch_std_median
        metrics['routing/continuous_score_fallback'] = float(
            continuous_score_fallback)
        metrics['routing/window_len'] = float(success_history_len)
        metrics['routing/std_window_len'] = float(std_history_len)
        metrics['routing/is_warmup'] = float(is_warmup)
        metrics['action_ig/active'] = float(action_ig_active)
        metrics['action_ig/ramp_scale'] = float(action_ig_scale)
        metrics['quadrant/filtered_ratio'] = (
            len(filtered_task_indices) / n_total_tasks if n_total_tasks else 0.0)
        metrics['quadrant/dpo_ratio'] = (
            len(dpo_task_indices) / n_total_tasks if n_total_tasks else 0.0)
        metrics['quadrant/contrastive_ratio'] = (
            len(contrastive_task_indices) / n_total_tasks if n_total_tasks else 0.0)
        metrics['quadrant/action_jsd_ratio'] = (
            len(action_jsd_task_indices) / n_total_tasks if n_total_tasks else 0.0)

        # Only now update the history, so round t never participates in the
        # thresholds used to route round t.
        self._routing_window.append(float(batch_pass_rate))
        self._routing_std_window.append(
            np.asarray(list(task_score_stds.values()), dtype=float))

        # Per-task pass rates sorted descending (e.g. "8/8, 7/8, 5/8, 0/8, ...")
        group_size = len(next(iter(task_rewards.values()))) if task_rewards else 8
        sorted_prs = sorted(task_pass_rates.values(), reverse=True)
        pr_str = ", ".join(f"{int(pr * group_size)}/{group_size}" for pr in sorted_prs)
        task_stat_str = ", ".join(
            f"task{idx}:p={task_pass_rates[idx]:.3f},"
            f"score_std={task_score_stds[idx]:.3f}"
            for idx in sorted(task_pass_rates)
        )

        jsd_route_label = "action_jsd"
        print(
            f"[Quadrant] Step {self.global_steps}: "
            f"curriculum={curriculum_stage}, "
            f"filtered={len(filtered_task_indices)}, dpo={len(dpo_task_indices)}, "
            f"contrastive={len(contrastive_task_indices)}, "
            f"{jsd_route_label}={len(action_jsd_task_indices)}, "
            f"medium={len(medium_task_indices)}, "
            f"success_threshold={success_threshold:.4f}, "
            f"std_threshold={std_threshold:.4f}, "
            f"history={success_history_len}/{self._routing_window_size}"
        )
        print(f"[Ours] Step {self.global_steps} pass_rates: [{pr_str}]")
        print(f"[Quadrant] Step {self.global_steps} task_stats: [{task_stat_str}]")

        # Phase 1 full-batch episode metrics (all tasks, not just medium/easy sub-batch)
        unique_traj_uids_p1, unique_idx_p1 = np.unique(plain_output.non_tensor_batch['traj_uid'], return_index=True)
        episode_rewards_p1 = plain_output.non_tensor_batch['episode_rewards'][unique_idx_p1]
        episode_lengths_p1 = plain_output.non_tensor_batch['episode_lengths'][unique_idx_p1]
        metrics['episode/reward/mean'] = float(episode_rewards_p1.mean())
        metrics['episode/reward/max'] = float(episode_rewards_p1.max())
        metrics['episode/reward/min'] = float(episode_rewards_p1.min())
        metrics['episode/length/mean'] = float(episode_lengths_p1.mean())
        metrics['episode/length/max'] = float(episode_lengths_p1.max())
        metrics['episode/length/min'] = float(episode_lengths_p1.min())
        for k, v in plain_output.non_tensor_batch.items():
            if "success_rate" in k:
                metrics[f'episode/{k}'] = float(v[0])

        # Per-step task index array (for selecting samples by tier)
        step_task_indices = np.array([uid_to_task_idx[uid] for uid in group_uids])

        # Extract unique uids (one per task) from Phase 1 for uid_base
        seen = set()
        uid_base = []
        for uid in group_uids:
            if uid not in seen:
                seen.add(uid)
                uid_base.append(uid)
        uid_base = np.array(uid_base, dtype=object)

        # ══════════════════════════════════════════════════════════════
        # The unified curriculum uses only the eight Student environment
        # rollouts. Hard Teacher supervision and Easy Skill/No-skill
        # supervision are both same-state action scoring; neither generates
        # another full environment trajectory.
        # ══════════════════════════════════════════════════════════════
        # Hard Action-JSD uses same-state Teacher probes on stored Student
        # actions and does not require Teacher environment rollouts.
        guided_r1_batch = None
        metrics['action_ig/aux_environment_rollouts'] = 0.0
        metrics['action_ig/hard_action_rows'] = float(
            np.sum([idx in hard_task_indices for idx in step_task_indices])
            if action_ig_active else 0
        )
        metrics['teacher/full_guided_rollouts'] = 0.0
        metrics['teacher/successful_guided_rollouts'] = 0.0
        metrics['teacher/successful_action_rows'] = 0.0
        easy_info_active = action_ig_active
        metrics['action_ig/easy_action_rows'] = float(
            np.sum([idx in easy_task_indices for idx in step_task_indices])
            if easy_info_active else 0
        )
        if action_ig_active and n_hard > 0:
            hard_probe_mask = np.array([idx in hard_task_indices for idx in step_task_indices])
            hard_probe_idxs = np.where(hard_probe_mask)[0]
            guided_r1_batch = plain_output.select_idxs(hard_probe_idxs)
            required_teacher_probe = {
                'teacher_probe_input_ids',
                'teacher_probe_attention_mask',
                'teacher_probe_position_ids',
            }
            missing_teacher_probe = required_teacher_probe - set(guided_r1_batch.batch.keys())
            if missing_teacher_probe:
                raise RuntimeError(
                    "Action-IG is enabled but hard samples are missing teacher prompts: "
                    f"{sorted(missing_teacher_probe)}")
        noskill_output = None
        phase2_needed = (
            (n_hard > 0 and is_warmup)
            or (n_easy > 0 and not action_ig_active)
        )
        metrics['action_ig/aux_environment_rollouts'] = float(
            n_total_tasks * env_rollout_n if phase2_needed else 0
        )
        metrics['easy/full_noskill_rollouts'] = float(
            n_easy * env_rollout_n if phase2_needed else 0)

        if phase2_needed:
            # Build per-task mode list (one entry per task)
            task_modes_base = []
            for tidx in range(n_total_tasks):
                if is_warmup and tidx in hard_task_indices:
                    task_modes_base.append('guided')
                elif tidx in easy_task_indices:
                    task_modes_base.append('noskill')
                else:
                    task_modes_base.append('plain')
            # Expand to match env batch (each task repeated env_rollout_n times, interleaved)
            task_modes = [m for m in task_modes_base for _ in range(env_rollout_n)]

            with _timer("gen_phase2", timing_raw):
                # Set mode: guide_internalize=True (needed for hard tasks to get dual text)
                self.envs.set_mode(plain=False)
                self.envs.set_per_task_mode(task_modes)
                phase2_output = self.traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=self.actor_rollout_wg,
                    envs=self.envs,
                    is_train=True,
                    reset_info=reset_info,
                    uid_base=uid_base,
                )
                self.envs.clear_per_task_mode()

            # Build uid -> task_index for Phase 2 output
            p2_group_uids = phase2_output.non_tensor_batch['uid']
            p2_traj_uids = phase2_output.non_tensor_batch['traj_uid']
            p2_rewards = phase2_output.non_tensor_batch['episode_rewards']

            p2_uid_to_task_idx = {}
            for uid in p2_group_uids:
                if uid not in p2_uid_to_task_idx:
                    p2_uid_to_task_idx[uid] = len(p2_uid_to_task_idx)

            p2_traj_reward = {}
            p2_traj_to_task = {}
            for g_uid, t_uid, r in zip(p2_group_uids, p2_traj_uids, p2_rewards):
                p2_traj_reward[t_uid] = float(r)
                p2_traj_to_task[t_uid] = p2_uid_to_task_idx[g_uid]

            # ── Extract hard task data (for JSD) ──
            if n_hard > 0 and is_warmup:
                # R=1 mask: step belongs to hard task AND trajectory reward > 0
                guided_r1_mask = np.array([
                    (p2_traj_to_task.get(t_uid, -1) in hard_task_indices) and (float(r) > 0)
                    for t_uid, r in zip(p2_traj_uids, p2_rewards)
                ])
                guided_r1_idxs = np.where(guided_r1_mask)[0]

                # Metrics
                n_guided_total = sum(1 for t_uid in p2_traj_reward
                                     if p2_traj_to_task[t_uid] in hard_task_indices)
                n_guided_pass = sum(1 for t_uid, r in p2_traj_reward.items()
                                    if p2_traj_to_task[t_uid] in hard_task_indices and r > 0)
                n_guided_r1_steps = int(guided_r1_mask.sum())
                metrics['teacher/full_guided_rollouts'] = float(n_guided_total)
                metrics['teacher/successful_guided_rollouts'] = float(n_guided_pass)
                metrics['teacher/successful_action_rows'] = float(n_guided_r1_steps)

                print(f"[Ours] Phase 2 (hard): guided_traj={n_guided_total}, "
                      f"R=1={n_guided_pass} (rate={n_guided_pass / n_guided_total if n_guided_total > 0 else 0.0:.3f}), "
                      f"jsd_token_count={n_guided_r1_steps}")

                if n_guided_pass > 0:
                    guided_r1_batch = phase2_output.select_idxs(guided_r1_idxs)
            # ── Extract easy task data (for contrastive) ──
            if n_easy > 0:
                # Select samples belonging to easy tasks
                p2_step_task_indices = np.array([p2_uid_to_task_idx[uid] for uid in p2_group_uids])
                easy_mask_p2 = np.array([idx in easy_task_indices for idx in p2_step_task_indices])
                easy_idxs_p2 = np.where(easy_mask_p2)[0]
                noskill_output = phase2_output.select_idxs(easy_idxs_p2)
                # Mark all as no_skill context_type
                noskill_output.non_tensor_batch['context_type'] = np.array(
                    ['no_skill'] * len(easy_idxs_p2), dtype=object)
                print(f"[Ours] Phase 2 (easy): noskill_output={len(noskill_output.batch['input_ids'])} samples")

            # ── Dump Phase 2 branch trajectories ──
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                p2_step_task_indices_all = np.array([p2_uid_to_task_idx[uid] for uid in p2_group_uids])
                # Dump guided (hard) trajectories
                if n_hard > 0 and is_warmup:
                    guided_mask_all = np.array([idx in hard_task_indices for idx in p2_step_task_indices_all])
                    guided_idxs_all = np.where(guided_mask_all)[0]
                    if len(guided_idxs_all) > 0:
                        guided_subset = phase2_output.select_idxs(guided_idxs_all)
                        guided_dump_path = os.path.join(rollout_data_dir, "guided")
                        guided_inputs = self.tokenizer.batch_decode(guided_subset.batch["prompts"], skip_special_tokens=True)
                        guided_outputs_text = self.tokenizer.batch_decode(guided_subset.batch["responses"], skip_special_tokens=True)
                        guided_scores = [float(x) for x in guided_subset.non_tensor_batch['episode_rewards']]
                        self._dump_generations(
                            inputs=guided_inputs,
                            outputs=guided_outputs_text,
                            scores=guided_scores,
                            reward_extra_infos_dict={},
                            dump_path=guided_dump_path,
                            traj_uids=guided_subset.non_tensor_batch['traj_uid'],
                            uids=guided_subset.non_tensor_batch['uid'],
                            extra_meta={"tier": "hard"},
                        )
                # Dump noskill (easy) trajectories
                if n_easy > 0 and noskill_output is not None:
                    noskill_dump_path = os.path.join(rollout_data_dir, "noskill")
                    noskill_inputs = self.tokenizer.batch_decode(noskill_output.batch["prompts"], skip_special_tokens=True)
                    noskill_outputs_text = self.tokenizer.batch_decode(noskill_output.batch["responses"], skip_special_tokens=True)
                    noskill_scores = [float(x) for x in noskill_output.non_tensor_batch['episode_rewards']]
                    self._dump_generations(
                        inputs=noskill_inputs,
                        outputs=noskill_outputs_text,
                        scores=noskill_scores,
                        reward_extra_infos_dict={},
                        dump_path=noskill_dump_path,
                        traj_uids=noskill_output.non_tensor_batch['traj_uid'],
                        uids=noskill_output.non_tensor_batch['uid'],
                        extra_meta={"tier": "easy"},
                    )

        # Restore env mode
        self.envs.restore_mode()

        # Keep vLLM resident across all environment turns, then release its
        # weights and KV cache before action scoring and actor updates.
        if (
            self.config.actor_rollout_ref.rollout.get("free_cache_engine", False)
            and self.config.actor_rollout_ref.rollout.get("sleep_at_rollout_end", False)
        ):
            self.actor_rollout_wg.sleep_rollout_engine()

        # ══════════════════════════════════════════════════════════════
        # Update 1: Easy tasks → Contrastive GRPO (independent step)
        # ══════════════════════════════════════════════════════════════
        reward_extra_infos_dict = {}
        batch = None  # will hold the "main" batch for return (use medium or easy)

        # High-success/high-uncertainty: compute GRPO on all rollouts, retaining
        # negative advantages with a smaller configurable coefficient.
        if filtered_task_indices:
            filtered_batch, filtered_reward_extra = self._update_filtered_grpo(
                plain_output=plain_output,
                task_indices=filtered_task_indices,
                step_task_indices=step_task_indices,
                timing_raw=timing_raw,
                metrics=metrics,
                negative_advantage_weight=float(
                    ours_cfg.get(
                        "filtered_negative_advantage_weight",
                        0.3,
                    )
                ),
            )
            if filtered_batch is not None:
                batch = filtered_batch
            if filtered_reward_extra:
                reward_extra_infos_dict = filtered_reward_extra

        # Low-success/high-uncertainty: preference optimization between
        # rank-paired top-k and bottom-k continuous-score trajectories.
        if dpo_task_indices:
            dpo_batch = self._update_trajectory_dpo(
                plain_output=plain_output,
                task_indices=dpo_task_indices,
                step_task_indices=step_task_indices,
                timing_raw=timing_raw,
                metrics=metrics,
                beta=float(ours_cfg.get("dpo_beta", 0.1)),
                score_key=continuous_score_key,
                top_k=int(ours_cfg.get("dpo_top_k", 2)),
                min_score_gap=float(
                    ours_cfg.get("dpo_min_score_gap", 0.1)),
            )
            if dpo_batch is not None:
                batch = dpo_batch

        # High-success/low-uncertainty: same-state action-information
        # Contrastive-GRPO. Skill and No-skill score the Student's executed
        # action; no additional environment rollout is generated.
        if n_easy > 0 and action_ig_active:
            easy_batch, reward_extra_easy = self._update_easy_with_action_ig(
                plain_output=plain_output,
                easy_task_indices=easy_task_indices,
                step_task_indices=step_task_indices,
                timing_raw=timing_raw,
                metrics=metrics,
                action_ig_active=True,
                action_ig_scale=action_ig_scale,
            )
            batch = easy_batch
            if reward_extra_easy:
                reward_extra_infos_dict = reward_extra_easy

        # Legacy fallback retained only for configurations that explicitly
        # disable action-level information and provide a No-skill trajectory.
        if (
            (not action_ig_active)
            and
            n_easy > 0
            and noskill_output is not None
        ):
            # Select easy task samples from plain_output (Phase 1)
            easy_mask = np.array([idx in easy_task_indices for idx in step_task_indices])
            easy_idxs = np.where(easy_mask)[0]
            easy_skill_batch = plain_output.select_idxs(easy_idxs)

            # The Phase-1 student rows may contain action-IG probe tensors,
            # while the legacy Phase-2 rows do not.  The old contrastive path
            # does not use those probes, so align both tensor and non-tensor
            # schemas before concatenating.
            common_batch_keys = set(easy_skill_batch.batch.keys()) & set(noskill_output.batch.keys())
            for proto in (easy_skill_batch, noskill_output):
                for k in list(proto.batch.keys()):
                    if k not in common_batch_keys:
                        del proto.batch[k]
            common_non_tensor_keys = (
                set(easy_skill_batch.non_tensor_batch.keys())
                & set(noskill_output.non_tensor_batch.keys())
            )
            for proto in (easy_skill_batch, noskill_output):
                for k in list(proto.non_tensor_batch.keys()):
                    if k not in common_non_tensor_keys:
                        del proto.non_tensor_batch[k]

            # Merge skill (easy from Phase 1) + noskill (Phase 2)
            merged_easy = DataProto.concat([easy_skill_batch, noskill_output])
            merged_easy = adjust_batch(self.config, merged_easy)
            merged_easy.batch["response_mask"] = compute_response_mask(merged_easy)

            # Reward on merged batch
            with _timer("reward_easy", timing_raw):
                reward_tensor_easy, reward_extra_easy = compute_reward(merged_easy, self.reward_fn)
            merged_easy.batch["token_level_scores"] = reward_tensor_easy
            if reward_extra_easy:
                merged_easy.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_easy.items()})

            # Apply invalid action penalty
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                merged_easy, _ = apply_invalid_action_penalty(
                    merged_easy,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                )

            # token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                merged_easy, _ = apply_kl_penalty(merged_easy, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
            else:
                merged_easy.batch["token_level_rewards"] = merged_easy.batch["token_level_scores"]

            # Contrastive advantage (noskill_mean as baseline)
            contrastive_context_types = merged_easy.non_tensor_batch.get('context_type', None)
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            utilize_cfg = self.config.env.get('utilize', {})
            contrastive_omega = utilize_cfg.get('omega', 1.0)
            adv2_clip = utilize_cfg.get('adv2_clip', 3.0)
            effective_rollout_n = rollout_n * 2
            # Match original Skill0.5: during warmup disable the contrastive
            # term (omega=0); afterward use the configured omega and EMA delta.
            if is_warmup:
                effective_omega = 0.0
                delta_baseline_value = None
            else:
                effective_omega = contrastive_omega
                # Use sliding window mean as delta baseline (None on first easy step → falls back to batch mode)
                delta_baseline_value = float(np.mean(list(self._delta_window))) if len(self._delta_window) > 0 else None
            merged_easy = compute_advantage(
                merged_easy,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=effective_rollout_n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                contrastive_context_types=contrastive_context_types,
                contrastive_omega=effective_omega,
                ema_delta=delta_baseline_value,
                adv2_clip=adv2_clip,
            )

            # Filter: keep only skill (top_k) samples for training
            phase1_mask = np.array([ct == 'top_k' for ct in merged_easy.non_tensor_batch['context_type']])
            phase1_idxs_easy = np.where(phase1_mask)[0]
            easy_batch = merged_easy.select_idxs(phase1_idxs_easy)
            easy_batch.meta_info = merged_easy.meta_info.copy()
            easy_batch.meta_info["global_token_num"] = torch.sum(easy_batch.batch["attention_mask"], dim=-1).tolist()

            # Pad to divisible by world_size
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            bs_easy = len(easy_batch)
            remainder = bs_easy % world_size
            if remainder != 0:
                to_add = world_size - remainder
                dup_indices = np.random.choice(bs_easy, to_add, replace=(to_add > bs_easy))
                dup_proto = easy_batch.select_idxs(dup_indices)
                dup_proto.batch["advantages"] = torch.zeros_like(dup_proto.batch["advantages"])
                dup_proto.batch["response_mask"] = torch.zeros_like(dup_proto.batch["response_mask"])
                if "loss_mask" in dup_proto.batch:
                    dup_proto.batch["loss_mask"] = torch.zeros_like(dup_proto.batch["loss_mask"])
                easy_batch = DataProto.concat([easy_batch, dup_proto])
                easy_batch.meta_info["global_token_num"] = torch.sum(easy_batch.batch["attention_mask"], dim=-1).tolist()

            # Contrastive metrics
            if contrastive_context_types is not None:
                episode_rewards_arr = merged_easy.non_tensor_batch.get('episode_rewards', None)
                bs_merged = len(merged_easy.batch['token_level_scores'])
                traj_uids_merged = merged_easy.non_tensor_batch.get('traj_uid', None)
                seen_trajs = {}
                for i in range(bs_merged):
                    uid = traj_uids_merged[i] if traj_uids_merged is not None else i
                    if uid not in seen_trajs:
                        is_succ = float(episode_rewards_arr[i]) > 0 if episode_rewards_arr is not None else False
                        seen_trajs[uid] = (contrastive_context_types[i], is_succ)
                # Per-task delta: mean(skill_pass_rate - noskill_pass_rate) across easy tasks
                # Used to update the sliding window baseline for adv2 computation
                group_uids_merged = merged_easy.non_tensor_batch.get('uid', None)
                if group_uids_merged is not None:
                    task_skill_rewards = defaultdict(list)
                    task_noskill_rewards = defaultdict(list)
                    for i, uid in enumerate(traj_uids_merged):
                        if uid in seen_trajs:
                            ct, is_succ = seen_trajs[uid]
                            g_uid = group_uids_merged[i]
                            if ct == 'top_k':
                                task_skill_rewards[g_uid].append(float(is_succ))
                            elif ct == 'no_skill':
                                task_noskill_rewards[g_uid].append(float(is_succ))

                    # Compute per-task delta and update sliding window
                    task_deltas = []
                    for g_uid in task_skill_rewards:
                        skill_pr = np.mean(task_skill_rewards[g_uid])
                        noskill_pr = np.mean(task_noskill_rewards[g_uid]) if g_uid in task_noskill_rewards else 0.0
                        task_deltas.append(skill_pr - noskill_pr)

                    if task_deltas:
                        mean_delta = float(np.mean(task_deltas))
                        self._delta_window.append(mean_delta)

            # Old log probs + ref log probs
            with _timer("old_log_prob_easy", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(easy_batch)
                entropys = old_log_prob.batch["entropys"]
                # Compute entropy metric on real samples only (exclude padding)
                entropys_real = entropys[:bs_easy]
                response_masks_real = easy_batch.batch["response_mask"][:bs_easy]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy_loss_easy"] = entropy_loss.detach().item()
                old_log_prob.batch.pop("entropys")
                easy_batch = easy_batch.union(old_log_prob)

            if self.use_reference_policy:
                with _timer("ref_easy", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(easy_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(easy_batch)
                    easy_batch = easy_batch.union(ref_log_prob)

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_easy", timing_raw):
                    easy_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    easy_batch.meta_info["hdpo_mode"] = "grpo"
                    easy_output = self.actor_rollout_wg.update_actor(easy_batch)
                easy_output_metrics = reduce_metrics(easy_output.meta_info["metrics"])
                if "actor/grad_norm" in easy_output_metrics:
                    easy_output_metrics["actor/grad_norm_easy"] = easy_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in easy_output_metrics:
                    easy_output_metrics["actor/grpo_loss_easy"] = easy_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in easy_output_metrics:
                    easy_output_metrics["actor/kl_loss_easy"] = easy_output_metrics.pop("actor/kl_loss")
                metrics.update(easy_output_metrics)

            print(f"[Ours] Update 1 (easy/utilize): {bs_easy} samples")
            batch = easy_batch  # use as return batch if no medium


        # ══════════════════════════════════════════════════════════════
        # Update 2: Medium tasks → Standard GRPO (independent step)
        # ══════════════════════════════════════════════════════════════
        if n_medium > 0:
            medium_mask = np.array([idx in medium_task_indices for idx in step_task_indices])
            medium_idxs = np.where(medium_mask)[0]
            medium_batch = plain_output.select_idxs(medium_idxs)
            bs_medium_real = len(medium_batch)

            medium_batch = adjust_batch(self.config, medium_batch)
            # Mark padding
            is_padding = np.zeros(len(medium_batch), dtype=bool)
            is_padding[bs_medium_real:] = True
            medium_batch.non_tensor_batch['_is_padding'] = is_padding
            medium_batch.batch["response_mask"] = compute_response_mask(medium_batch)

            # Zero out padding masks immediately so they never contribute to
            # entropy/kl gradients or metrics (dp_actor uses loss_mask in multi_turn)
            n_padding = int(is_padding.sum())
            if n_padding > 0:
                padding_indices = torch.tensor(np.where(is_padding)[0], dtype=torch.long)
                medium_batch.batch["response_mask"][padding_indices] = 0.0
                if "loss_mask" in medium_batch.batch:
                    medium_batch.batch["loss_mask"][padding_indices] = 0.0

            if self.config.trainer.balance_batch:
                self._balance_batch(medium_batch, metrics=metrics)

            medium_batch.meta_info["global_token_num"] = torch.sum(
                medium_batch.batch["attention_mask"], dim=-1).tolist()

            # Reward
            with _timer("reward_medium", timing_raw):
                reward_tensor_med, reward_extra_med = compute_reward(medium_batch, self.reward_fn)
            medium_batch.batch["token_level_scores"] = reward_tensor_med
            if reward_extra_med:
                medium_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_med.items()})
            if not reward_extra_infos_dict:
                reward_extra_infos_dict = reward_extra_med

            # Apply invalid action penalty
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                medium_batch, invalid_metrics = apply_invalid_action_penalty(
                    medium_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)
                # Metrics from real samples only
                padding_mask = medium_batch.non_tensor_batch['_is_padding']
                if 'valid_actions' in medium_batch.non_tensor_batch:
                    real_valid = medium_batch.non_tensor_batch['valid_actions'][~padding_mask]
                    invalid_metrics['episode/valid_action_ratio'] = float(np.mean(real_valid))
                metrics.update(invalid_metrics)

            # KL / token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                medium_batch, kl_metrics = apply_kl_penalty(
                    medium_batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                medium_batch.batch["token_level_rewards"] = medium_batch.batch["token_level_scores"]

            # Old log probs
            with _timer("old_log_prob_medium", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(medium_batch)
                entropys = old_log_prob.batch["entropys"]
                # Compute entropy metric on real samples only (exclude padding)
                real_mask_bool = ~medium_batch.non_tensor_batch['_is_padding']
                real_indices = torch.tensor(np.where(real_mask_bool)[0], dtype=torch.long)
                entropys_real = entropys[real_indices]
                response_masks_real = medium_batch.batch["response_mask"][real_indices]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys_real, loss_mask=response_masks_real, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy_loss_medium"] = entropy_loss.detach().item()
                old_log_prob.batch.pop("entropys")
                medium_batch = medium_batch.union(old_log_prob)

            # Ref log probs
            if self.use_reference_policy:
                with _timer("ref_medium", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(medium_batch)
                    else:
                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(medium_batch)
                    medium_batch = medium_batch.union(ref_log_prob)

            # Advantage
            with _timer("adv_medium", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                medium_batch = compute_advantage(
                    medium_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                    gigpo_mode=self.config.algorithm.gigpo.mode,
                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                )

            # Zero out padding advantages (response_mask/loss_mask already zeroed above)
            padding_mask = medium_batch.non_tensor_batch.get('_is_padding', np.zeros(len(medium_batch), dtype=bool))
            n_padding = int(padding_mask.sum())
            if n_padding > 0:
                padding_indices = torch.tensor(np.where(padding_mask)[0], dtype=torch.long)
                medium_batch.batch["advantages"][padding_indices] = 0.0

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_medium", timing_raw):
                    medium_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    medium_batch.meta_info["global_token_num"] = torch.sum(
                        medium_batch.batch["attention_mask"], dim=-1).tolist()
                    medium_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    medium_batch.meta_info["hdpo_mode"] = "grpo"
                    medium_output = self.actor_rollout_wg.update_actor(medium_batch)
                medium_output_metrics = reduce_metrics(medium_output.meta_info["metrics"])
                if "actor/grad_norm" in medium_output_metrics:
                    medium_output_metrics["actor/grad_norm_medium"] = medium_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in medium_output_metrics:
                    medium_output_metrics["actor/grpo_loss_medium"] = medium_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in medium_output_metrics:
                    medium_output_metrics["actor/kl_loss_medium"] = medium_output_metrics.pop("actor/kl_loss")
                metrics.update(medium_output_metrics)

            print(f"[Ours] Update 2 (medium/grpo): {bs_medium_real} real + {n_padding} padding samples")
            batch = medium_batch  # prefer medium as return batch

        # ══════════════════════════════════════════════════════════════
        # Update 3: Hard tasks → Standard GRPO (format signal)
        # Even all-fail tasks get gradient from invalid_action_penalty
        # differences (score=0 vs score=-0.1 across steps).
        # ══════════════════════════════════════════════════════════════
        hard_grpo_enabled = bool(ours_cfg.get("hard_grpo_enabled", False))
        # The unified internalization stage retains a GRPO environment/EOS
        # anchor while adding the smoothly decaying Teacher-probe JSD.
        if curriculum_enabled and curriculum_stage == "internalize":
            hard_grpo_enabled = True
        metrics["routing/hard_grpo_enabled"] = float(hard_grpo_enabled)
        metrics["quadrant/hard_grpo_skipped_tasks"] = float(
            n_hard if not hard_grpo_enabled else 0
        )
        if n_hard > 0 and hard_grpo_enabled:
            hard_mask = np.array([idx in hard_task_indices for idx in step_task_indices])
            hard_idxs = np.where(hard_mask)[0]
            hard_batch = plain_output.select_idxs(hard_idxs)
            bs_hard_real = len(hard_batch)

            hard_batch = adjust_batch(self.config, hard_batch)
            # Mark padding
            is_padding_hard = np.zeros(len(hard_batch), dtype=bool)
            is_padding_hard[bs_hard_real:] = True
            hard_batch.non_tensor_batch['_is_padding'] = is_padding_hard
            hard_batch.batch["response_mask"] = compute_response_mask(hard_batch)

            # Zero out padding masks
            n_padding_hard = int(is_padding_hard.sum())
            if n_padding_hard > 0:
                padding_indices_hard = torch.tensor(np.where(is_padding_hard)[0], dtype=torch.long)
                hard_batch.batch["response_mask"][padding_indices_hard] = 0.0
                if "loss_mask" in hard_batch.batch:
                    hard_batch.batch["loss_mask"][padding_indices_hard] = 0.0

            hard_batch.meta_info["global_token_num"] = torch.sum(
                hard_batch.batch["attention_mask"], dim=-1).tolist()

            # Reward
            with _timer("reward_hard", timing_raw):
                reward_tensor_hard, reward_extra_hard = compute_reward(hard_batch, self.reward_fn)
            hard_batch.batch["token_level_scores"] = reward_tensor_hard
            if reward_extra_hard:
                hard_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_hard.items()})

            # Apply invalid action penalty (this creates the reward variance for GRPO)
            if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                hard_batch, _ = apply_invalid_action_penalty(
                    hard_batch,
                    invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef)

            # token_level_rewards
            if self.config.algorithm.use_kl_in_reward:
                hard_batch, _ = apply_kl_penalty(
                    hard_batch, kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty)
            else:
                hard_batch.batch["token_level_rewards"] = hard_batch.batch["token_level_scores"]

            # Old log probs
            with _timer("old_log_prob_hard", timing_raw):
                old_log_prob_hard = self.actor_rollout_wg.compute_log_prob(hard_batch)
                old_log_prob_hard.batch.pop("entropys")
                hard_batch = hard_batch.union(old_log_prob_hard)

            # Ref log probs
            if self.use_reference_policy:
                with _timer("ref_hard", timing_raw):
                    if not self.ref_in_actor:
                        ref_log_prob_hard = self.ref_policy_wg.compute_ref_log_prob(hard_batch)
                    else:
                        ref_log_prob_hard = self.actor_rollout_wg.compute_ref_log_prob(hard_batch)
                    hard_batch = hard_batch.union(ref_log_prob_hard)

            # Advantage (standard GRPO, task-level z-score)
            with _timer("adv_hard", timing_raw):
                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                hard_batch = compute_advantage(
                    hard_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=rollout_n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                    gigpo_mode=self.config.algorithm.gigpo.mode,
                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                )

            # Full GRPO (weight 1) is the stable default.  The configurable
            # floor permits ablations with a weaker early task signal without
            # changing the JSD schedule or the optimizer implementation.
            if curriculum_enabled and curriculum_stage == "internalize":
                hard_batch.batch["advantages"] = (
                    hard_batch.batch["advantages"]
                    * float(curriculum_grpo_weight)
                )

            # Zero out padding advantages
            if n_padding_hard > 0:
                hard_batch.batch["advantages"][padding_indices_hard] = 0.0

            # Update actor (independent step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_hard", timing_raw):
                    hard_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    hard_batch.meta_info["global_token_num"] = torch.sum(
                        hard_batch.batch["attention_mask"], dim=-1).tolist()
                    hard_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    hard_batch.meta_info["hdpo_mode"] = "grpo"
                    hard_output = self.actor_rollout_wg.update_actor(hard_batch)
                hard_output_metrics = reduce_metrics(hard_output.meta_info["metrics"])
                if "actor/grad_norm" in hard_output_metrics:
                    hard_output_metrics["actor/grad_norm_hard"] = hard_output_metrics.pop("actor/grad_norm")
                if "actor/pg_loss" in hard_output_metrics:
                    hard_output_metrics["actor/grpo_loss_hard"] = hard_output_metrics.pop("actor/pg_loss")
                if "actor/kl_loss" in hard_output_metrics:
                    hard_output_metrics["actor/kl_loss_hard"] = hard_output_metrics.pop("actor/kl_loss")
                metrics.update(hard_output_metrics)

            print(f"[Ours] Update 3 (hard/grpo): {bs_hard_real} real + {n_padding_hard} padding samples")
            if batch is None:
                batch = hard_batch
        elif n_hard > 0:
            print(
                f"[Ours] Update 3 (hard/grpo): disabled; skipped "
                f"{n_hard} hard tasks"
            )

        # ══════════════════════════════════════════════════════════════
        # Update 4: hard-task teacher supervision. During warmup this consumes
        # successful full teacher trajectories; after warmup it consumes the
        # stored student actions with teacher counterfactual probes.
        # ══════════════════════════════════════════════════════════════
        if guided_r1_batch is not None and len(guided_r1_batch.batch['input_ids']) > 0:
            jsd_batch = guided_r1_batch
            world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
            jsd_divisor = jsd_micro_batch_size_per_gpu * world_size
            bs_jsd_real = len(jsd_batch)
            remainder = bs_jsd_real % jsd_divisor
            if remainder != 0:
                to_add = jsd_divisor - remainder
                dup_indices = np.random.choice(bs_jsd_real, to_add, replace=(to_add > bs_jsd_real))
                dup_proto = jsd_batch.select_idxs(dup_indices)
                jsd_batch = DataProto.concat([jsd_batch, dup_proto])
            jsd_batch.batch["response_mask"] = compute_response_mask(jsd_batch)
            # Zero out padding masks
            if len(jsd_batch) > bs_jsd_real:
                jsd_batch.batch["response_mask"][bs_jsd_real:] = 0.0
                if "loss_mask" in jsd_batch.batch:
                    jsd_batch.batch["loss_mask"][bs_jsd_real:] = 0.0

            # Update actor (independent JSD step)
            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer("update_actor_jsd", timing_raw):
                    jsd_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    jsd_batch.meta_info["global_token_num"] = torch.sum(
                        jsd_batch.batch["attention_mask"], dim=-1).tolist()
                    jsd_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    jsd_batch.meta_info["hdpo_mode"] = "jsd"
                    jsd_batch.meta_info["hdpo_config"] = {
                        # Full-trajectory warmup uses the original JSD weight;
                        # action-level JSD is ramped in after warmup.
                        'jsd_lambda': (
                            jsd_lambda if not action_ig_active
                            else action_jsd_lambda * action_ig_scale
                        ),
                        'jsd_top_k': jsd_top_k,
                        'jsd_temperature': jsd_temperature,
                        'jsd_micro_batch_size_per_gpu': jsd_micro_batch_size_per_gpu,
                        'use_teacher_probe': action_ig_active,
                        'action_ig_enabled': action_ig_active,
                        'action_ig_clip': internalize_cfg.get('action_ig_clip', 1.2),
                        'action_ig_beta': (
                            internalize_cfg.get('action_ig_beta', 0.2) * action_ig_scale
                        ),
                    }
                    jsd_output = self.actor_rollout_wg.update_actor(jsd_batch)
                jsd_output_metrics = reduce_metrics(jsd_output.meta_info["metrics"])
                if "actor/grad_norm" in jsd_output_metrics:
                    jsd_output_metrics["actor/grad_norm_jsd"] = jsd_output_metrics.pop("actor/grad_norm")
                metrics.update(jsd_output_metrics)

            jsd_mode = "action-IG JSD" if action_ig_active else "full-trajectory teacher JSD"
            print(f"[Ours] Update 4 (hard/{jsd_mode}): {bs_jsd_real} real samples")

        # ══════════════════════════════════════════════════════════════
        # Fallback: if no batch set (degenerate edge case)
        # ══════════════════════════════════════════════════════════════
        if batch is None:
            # Use plain_output as-is for return value (no GRPO update happened)
            print("[Ours] WARNING: No medium/easy samples. Using plain_output as fallback batch.")
            batch = plain_output
            batch = adjust_batch(self.config, batch)
            batch.batch["response_mask"] = compute_response_mask(batch)
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            with _timer("reward_fallback", timing_raw):
                reward_tensor_fb, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor_fb
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            # Fill advantages with zeros (no GRPO update happened, but fit() needs this field).
            # response_mask is an integer/bool mask, so create floating advantages explicitly.
            batch.batch["advantages"] = torch.zeros_like(
                batch.batch["response_mask"], dtype=torch.float32
            )

        # Every branch returned to fit() must satisfy the common metrics
        # contract. Keep this guard even though individual update branches
        # normally populate these fields themselves.
        if "token_level_scores" not in batch.batch:
            with _timer("reward_return_guard", timing_raw):
                reward_tensor_return, reward_extra_return = compute_reward(
                    batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor_return
            if reward_extra_return:
                batch.non_tensor_batch.update(
                    {
                        key: np.asarray(value)
                        for key, value in reward_extra_return.items()
                    }
                )
        if "token_level_rewards" not in batch.batch:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        if "advantages" not in batch.batch:
            batch.batch["advantages"] = torch.zeros_like(
                batch.batch["token_level_rewards"], dtype=torch.float32
            )
        if "returns" not in batch.batch:
            if "advantages" in batch.batch:
                batch.batch["returns"] = batch.batch["advantages"].clone()
            else:
                batch.batch["returns"] = batch.batch["token_level_rewards"].clone()
        if not torch.is_floating_point(batch.batch["advantages"]):
            batch.batch["advantages"] = batch.batch["advantages"].float()
        if not torch.is_floating_point(batch.batch["returns"]):
            batch.batch["returns"] = batch.batch["returns"].float()

        return batch, reward_extra_infos_dict


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_best_checkpoint_state()
        # breakpoint()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            if self.val_envs_ood is not None:
                val_ood_metrics = self._validate_ood()
                val_metrics.update(val_ood_metrics)
                pprint(f"Initial OOD validation metrics: {val_ood_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                # Save merged metrics (val + val_ood) to val_dump_path
                # Merge strategy: per-domain metrics (different names) go under val/
                # Aggregate metrics (success_rate, test_score) that exist in both
                # val/ and val_ood/ are kept as val/id/X and val/ood/X, with a
                # weighted val/X computed from both.
                val_dump_path = self.config.trainer.get("val_dump_path", None)
                if val_dump_path:
                    os.makedirs(val_dump_path, exist_ok=True)

                    # Identify conflicting keys (exist in both val/ and val_ood/)
                    val_keys = {k for k in val_metrics if k.startswith('val/') and not k.startswith('val_ood/')}
                    ood_keys = {k for k in val_metrics if k.startswith('val_ood/')}
                    ood_as_val = {k.replace('val_ood/', 'val/'): k for k in ood_keys}
                    conflicting = val_keys & set(ood_as_val.keys())

                    normalized = {}
                    for k, v in val_metrics.items():
                        if k.startswith('val_ood/'):
                            new_key = k.replace('val_ood/', 'val/')
                            if new_key in conflicting:
                                # Conflicting aggregate: store under val/ood/
                                normalized[k.replace('val_ood/', 'val/ood/')] = v
                            else:
                                normalized[new_key] = v
                        elif k in conflicting:
                            # ID side of conflicting aggregate: store under val/id/
                            normalized[k.replace('val/', 'val/id/')] = v
                        else:
                            normalized[k] = v

                    # Compute weighted overall success_rate / test_score
                    id_sr = val_metrics.get('val/success_rate')
                    ood_sr = val_metrics.get('val_ood/success_rate')
                    id_ts = val_metrics.get('val/text/test_score')
                    ood_ts = val_metrics.get('val_ood/text/test_score')
                    n_id = val_metrics.get('val/num_trajs', 0)
                    n_ood = val_metrics.get('val_ood/num_trajs', 0)

                    if id_sr is not None and ood_sr is not None and n_id + n_ood > 0:
                        normalized['val/success_rate'] = (id_sr * n_id + ood_sr * n_ood) / (n_id + n_ood)
                    if id_ts is not None and ood_ts is not None and n_id + n_ood > 0:
                        normalized['val/text/test_score'] = (id_ts * n_id + ood_ts * n_ood) / (n_id + n_ood)

                    metrics_file = os.path.join(val_dump_path, "metrics.json")
                    with open(metrics_file, "w") as f:
                        json.dump(normalized, f, indent=2, ensure_ascii=False)
                    print(f"Saved merged validation metrics to {metrics_file}")

                    # SearchQA-friendly report with stable names. This is
                    # written in addition to the complete flat metric dump.
                    if 'val/search_qa/accuracy' in normalized:
                        task_names = (
                            'nq',
                            'hotpotqa',
                            'triviaqa',
                            'popqa',
                            '2wikimultihopqa',
                            'musique',
                            'bamboogle',
                        )
                        accuracy_summary = {
                            "checkpoint": self.config.trainer.get(
                                "resume_from_path", None
                            ),
                            "overall_accuracy": normalized.get(
                                'val/search_qa/accuracy'
                            ),
                            "id_accuracy": normalized.get('val/id/accuracy'),
                            "ood_accuracy": normalized.get('val/ood/accuracy'),
                            "task_accuracy": {
                                task: normalized.get(f'val/{task}/accuracy')
                                for task in task_names
                            },
                            "task_num_samples": {
                                task: normalized.get(f'val/{task}/num_samples')
                                for task in task_names
                            },
                        }
                        summary_file = os.path.join(
                            val_dump_path, "accuracy_summary.json"
                        )
                        with open(summary_file, "w", encoding="utf-8") as f:
                            json.dump(
                                accuracy_summary,
                                f,
                                indent=2,
                                ensure_ascii=False,
                            )
                        print(
                            "SearchQA accuracy summary:\n"
                            + json.dumps(
                                accuracy_summary,
                                indent=2,
                                ensure_ascii=False,
                            )
                        )
                        print(f"Saved SearchQA accuracy summary to {summary_file}")
                return

        # add tqdm
        quiet_training = self.config.trainer.get("quiet_training", False)
        print_validation_metrics = self.config.trainer.get(
            "print_validation_metrics", False
        )
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
            disable=quiet_training,
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                step_wall_start = time.perf_counter()
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    batch, reward_extra_infos_dict = self._ours_step(
                        gen_batch, timing_raw, metrics)

                # Note: plain rollout dump is now done inside each _*_step method
                # (before balance_batch reorders data), so we skip the post-update dump here.

                # validate
                best_candidate = None
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    if print_validation_metrics:
                        pprint(
                            f"Validation metrics at step {self.global_steps}: "
                            f"{val_metrics}"
                        )
                    metrics.update(val_metrics)
                    # OOD validation
                    if self.val_envs_ood is not None:
                        with _timer("testing_ood", timing_raw):
                            val_ood_metrics = self._validate_ood()
                        if print_validation_metrics:
                            pprint(
                                "OOD validation metrics at step "
                                f"{self.global_steps}: {val_ood_metrics}"
                            )
                        metrics.update(val_ood_metrics)
                    best_candidate = self._best_checkpoint_candidate(metrics)

                regular_checkpoint = (
                    self.config.trainer.save_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    )
                )
                if regular_checkpoint or best_candidate is not None:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                    if best_candidate is not None:
                        self._record_best_checkpoint(*best_candidate)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # Preserve Phase 1 full-batch episode metrics (set in _ours_step) before
                # compute_data_metrics overwrites them with sub-batch values
                episode_keys_override = {k: v for k, v in metrics.items() if k.startswith("episode/")}
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                if episode_keys_override:
                    metrics.update(episode_keys_override)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                step_wall_seconds = time.perf_counter() - step_wall_start
                metrics["timing_s/train_step_wall"] = step_wall_seconds
                if self.config.trainer.get("print_step_time", False):
                    print(
                        f"[train_step] step {self.global_steps}/{self.total_training_steps}, "
                        f"time: {step_wall_seconds:.2f}s",
                        flush=True,
                    )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    if not print_validation_metrics:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        # Dual tokenization for guide internalization
        obs_plain_texts = obs.get('text_plain', None)
        if obs_plain_texts is not None:
            obs_plain_text = obs_plain_texts[item]
            plain_chat = np.array([{
                "content": obs_plain_text,
                "role": "user",
            }])
            plain_prompt_with_chat_template = self.tokenizer.apply_chat_template(
                plain_chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
            plain_input_ids, plain_attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=plain_prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            plain_position_ids = compute_position_id_with_mask(plain_attention_mask)
            row_dict['plain_input_ids'] = plain_input_ids[0]
            row_dict['plain_attention_mask'] = plain_attention_mask[0]
            row_dict['plain_position_ids'] = plain_position_ids[0]

        # Counterfactual single-action probes.  These are prompt-only here;
        # vanilla_multi_turn_loop appends the student response so the actor can
        # score the same action under teacher/no-skill context without taking
        # another environment step.
        for obs_key, field_prefix in (
            ('text_teacher_probe', 'teacher_probe'),
            ('text_no_skill_probe', 'no_skill_probe'),
        ):
            probe_texts = obs.get(obs_key, None)
            if probe_texts is None:
                continue
            probe_chat = np.array([{
                "content": probe_texts[item],
                "role": "user",
            }])
            probe_prompt = self.tokenizer.apply_chat_template(
                probe_chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
            probe_input_ids, probe_attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=probe_prompt,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            probe_position_ids = compute_position_id_with_mask(probe_attention_mask)
            row_dict[f'{field_prefix}_input_ids'] = probe_input_ids[0]
            row_dict[f'{field_prefix}_attention_mask'] = probe_attention_mask[0]
            row_dict[f'{field_prefix}_position_ids'] = probe_position_ids[0]

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # Determine context_type per episode to exclude no_skill probes from success_rate
        context_types = []
        for bs in range(batch_size):
            ct = None
            for data in total_batch_list[bs]:
                if data.get('context_type') is not None:
                    ct = data['context_type']
                    break
            context_types.append(ct)

        success_rate = {}
        for key, value in success.items():
            # If contrastive training is active, only count top_k episodes for success_rate
            if context_types[0] is not None and len(value) == batch_size:
                top_k_mask = np.array([ct == 'top_k' for ct in context_types])
                filtered = value[top_k_mask]
                success_rate[key] = np.mean(filtered) if len(filtered) > 0 else 0.0
            else:
                success_rate[key] = np.mean(value) if len(value) > 0 else 0.0

        # ``success_rate`` above intentionally stores batch-level logging
        # aggregates.  The four-quadrant WebShop router also needs the original
        # continuous score for every trajectory, so preserve it separately
        # instead of trying to recover it from the broadcast batch mean.
        webshop_task_scores = success.get(
            'webshop_task_score (not success_rate)', None)
        if webshop_task_scores is not None:
            webshop_task_scores = np.asarray(webshop_task_scores, dtype=np.float32)
            if len(webshop_task_scores) != batch_size:
                raise ValueError(
                    "Expected one WebShop task score per trajectory, got "
                    f"{len(webshop_task_scores)} scores for batch size {batch_size}"
                )
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    if webshop_task_scores is not None:
                        data['trajectory_webshop_task_score'] = float(
                            webshop_task_scores[bs])

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            uid_override: np.ndarray = None,
            reset_info: dict = None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
            uid_override (np.ndarray): If provided, use these uids instead of generating new ones (for wave-mode)
            reset_info (dict): If provided, pass to envs.reset() to replay the same task/goal (for wave-mode)

        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment.
        # Pass n=batch_size so envs that support variable-size eval batches
        # (e.g. WebshopMultiProcessEnv) know how many to reset.
        env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', None)
        if reset_info is not None:
            obs, infos = envs.reset(kwargs=env_kwargs, n=batch_size, reset_info=reset_info)
        else:
            obs, infos = envs.reset(kwargs=env_kwargs, n=batch_size)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        if uid_override is not None:
            # Wave-mode or two-phase contrastive: use the provided uids directly
            uid_batch = uid_override
        elif self.config.env.rollout.n > 0: # env grouping
            group_size = self.config.env.rollout.n
            uid_batch = []
            for i in range(batch_size):
                if i % group_size == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)

        # Extract context_type from observations (for contrastive training)
        context_type_batch = None
        if obs.get('context_type') is not None:
            context_type_batch = np.array(obs['context_type'], dtype=object)

        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            # Preserve the temporal order after later padding and token-balance
            # reordering. Action-level information reward-to-go is grouped by
            # traj_uid and sorted by this index; it never triggers another env
            # step or rollout.
            batch.non_tensor_batch['step_index'] = np.full(batch_size, _step, dtype=np.int64)
            if context_type_batch is not None:
                batch.non_tensor_batch['context_type'] = context_type_batch

            batch = batch.union(batch_output)

            # Append the executed student response to each alternate prompt.
            # The alternate policy is only scored on this one action; its
            # output is never sent to envs.step().
            responses = batch.batch['responses']
            eos_token_id = gen_batch.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
            from verl.utils.torch_functional import get_response_mask
            for field_prefix in ('plain', 'teacher_probe', 'no_skill_probe'):
                ids_key = f'{field_prefix}_input_ids'
                mask_key = f'{field_prefix}_attention_mask'
                pos_key = f'{field_prefix}_position_ids'
                if ids_key not in batch.batch.keys():
                    continue

                prompt_ids = batch.batch[ids_key]
                prompt_mask = batch.batch[mask_key]
                prompt_pos = batch.batch[pos_key]
                resp_attn_mask = get_response_mask(
                    response_id=responses,
                    eos_token=eos_token_id,
                    dtype=prompt_mask.dtype,
                )
                resp_length = responses.size(1)
                delta_pos = torch.arange(1, resp_length + 1, device=prompt_pos.device)
                delta_pos = delta_pos.unsqueeze(0).expand(prompt_ids.size(0), -1)
                resp_pos = prompt_pos[:, -1:] + delta_pos

                batch.batch[ids_key] = torch.cat([prompt_ids, responses], dim=-1)
                batch.batch[mask_key] = torch.cat([prompt_mask, resp_attn_mask], dim=-1)
                batch.batch[pos_key] = torch.cat([prompt_pos, resp_pos], dim=-1)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Log progress
            n_done = int(is_done.sum())
            print(f"[multi_turn_loop] step {_step+1}/{self.config.env.max_steps}, done: {n_done}/{batch_size}, avg_reward: {episode_rewards.mean():.4f}")

            # Break if all environments are done
            if is_done.all():
                break

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            reset_info: dict = None,
            uid_base: np.ndarray = None,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            reset_info (dict): If provided, replay the same tasks (for two-phase contrastive).
            uid_base (np.ndarray): If provided, use these uids (for two-phase contrastive).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        wave_mode = self.config.env.rollout.get('wave_mode', False) if is_train else False

        if is_train:
            # Two-phase contrastive: no doubling here; each phase runs rollout.n
            effective_group_size = self.config.env.rollout.n

            if not wave_mode:
                # Original logic: repeat batch and create group_n workers per prompt
                gen_batch = gen_batch.repeat(repeat_times=effective_group_size, interleave=True)

        if wave_mode and is_train:
            # Wave-mode: env layer has group_n=wave_batch_size workers,
            # run effective_group_size / wave_batch_size waves.
            wave_batch_size = self.config.env.rollout.get('wave_batch_size', 1)
            assert effective_group_size % wave_batch_size == 0, (
                f"effective_group_size ({effective_group_size}) must be divisible by "
                f"wave_batch_size ({wave_batch_size})")
            num_waves = effective_group_size // wave_batch_size

            # Pre-generate one uid per prompt; shared across all waves
            env_num = len(gen_batch.batch)
            uid_base = np.array([str(uuid.uuid4()) for _ in range(env_num)], dtype=object)

            all_batch_list = []
            all_episode_rewards = []
            all_episode_lengths = []
            all_success = []
            all_traj_uid = []
            all_tool_callings = []
            reset_info = None

            for wave in range(num_waves):
                print(f"[wave_mode] Starting wave {wave+1}/{num_waves} "
                      f"(wave_batch_size={wave_batch_size})")

                # Repeat gen_batch wave_batch_size times for this wave
                wave_gen_batch = gen_batch.repeat(repeat_times=wave_batch_size, interleave=True)

                # uid: repeat uid_base to match repeated batch (env_num * wave_batch_size)
                uid_override = np.repeat(uid_base, wave_batch_size)

                w_batch_list, w_rewards, w_lengths, w_success, w_traj_uid, w_tool_callings = \
                    self.vanilla_multi_turn_loop(
                        gen_batch=wave_gen_batch,
                        actor_rollout_wg=actor_rollout_wg,
                        envs=envs,
                        uid_override=uid_override,
                        reset_info=reset_info,
                    )

                # After wave 0, capture reset_info so subsequent waves replay the same tasks
                if wave == 0:
                    reset_info = envs.get_last_reset_info()

                all_batch_list.extend(w_batch_list)
                all_episode_rewards.append(w_rewards)
                all_episode_lengths.append(w_lengths)
                all_success.append(w_success)
                all_traj_uid.append(w_traj_uid)
                all_tool_callings.append(w_tool_callings)

            total_batch_list = all_batch_list
            total_episode_rewards = np.concatenate(all_episode_rewards, axis=0)
            total_episode_lengths = np.concatenate(all_episode_lengths, axis=0)
            total_success = {key: np.concatenate([s[key] for s in all_success], axis=0) for key in all_success[0].keys()}
            total_traj_uid = np.concatenate(all_traj_uid, axis=0)
            totoal_tool_callings = np.concatenate(all_tool_callings, axis=0)

        elif self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling
            # Build uid_override from uid_base if provided (two-phase contrastive)
            uid_override = None
            if uid_base is not None:
                uid_override = np.repeat(uid_base, effective_group_size)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                uid_override=uid_override,
                reset_info=reset_info,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)


        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )

        return gen_batch_output

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

from typing import List, Tuple, Dict, Union, Any, Optional
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        self.guide_internalize = config.env.get('guide_internalize', True)
        self.exclude_hints = config.env.get('exclude_hints', False)
        # Store original values for mode restoration
        self._orig_guide_internalize = self.guide_internalize
        self._orig_exclude_hints = self.exclude_hints

        # Two-phase contrastive probe
        base_group_n = config.env.rollout.n if config.env.rollout.get('n', 0) > 0 else 1
        self.group_n = base_group_n
        self._contrastive_probe_mode = False

        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
                include_general=som_cfg.get('include_general', True),
            )
            self.retrieved_memories = None
            print(f"[SearchEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')}, include_general={som_cfg.get('include_general', True)})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[SearchEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None
            self.retrieved_memories = None

        super().__init__(envs, projection_f, config)

    def set_mode(self, plain: bool):
        """Switch prompt construction mode for HDPO two-phase rollout.

        plain=True  (Phase 1): exclude_hints=True, guide_internalize=False
            → build_text_obs returns plain-only text (specific skills only)
        plain=False (Phase 2): exclude_hints=False, guide_internalize=True
            → build_text_obs returns (guided_texts, plain_texts) tuple
        """
        if plain:
            self.exclude_hints = True
            self.guide_internalize = False
        else:
            self.exclude_hints = False
            self.guide_internalize = True

    def restore_mode(self):
        """Restore original mode after HDPO rollout."""
        self.guide_internalize = self._orig_guide_internalize
        self.exclude_hints = self._orig_exclude_hints

    def set_contrastive_probe(self, probe: bool):
        """Switch contrastive probe mode for two-phase rollout.

        probe=True:  next reset() marks ALL envs as no_skill (Phase 2)
        probe=False: next reset() marks ALL envs as top_k (Phase 1, default)
        """
        self._contrastive_probe_mode = probe

    def set_per_task_mode(self, task_modes: list):
        """Set per-task mode for unified Phase 2 in ours_step.

        task_modes: list of str, one per task. Values:
            "guided"  — hard tasks: all skills, guide_internalize per-index
            "noskill" — easy tasks: no skills (contrastive probe)
            "plain"   — medium tasks (filler): specific skills only, exclude hints
        """
        self._per_task_modes = task_modes

    def clear_per_task_mode(self):
        """Clear per-task mode after Phase 2."""
        self._per_task_modes = None

    def reset(self, kwargs, n: int = None, reset_info: dict = None) -> Tuple[Dict[str, Any], List[Dict]]:
        if reset_info is not None and reset_info.get('env_kwargs') is not None:
            kwargs = reset_info['env_kwargs']
        if kwargs is None:
            raise ValueError(
                "SearchEnvironmentManager.reset requires dataset env_kwargs "
                "containing question and ground_truth"
            )
        self.kwargs = kwargs
        self._last_reset_info = {'env_kwargs': kwargs}
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))

        # Determine context_type per env for contrastive training
        batch_size = len(obs)
        self.context_types = ['top_k'] * batch_size  # default: all top-k

        # Per-task mode (ours Phase 2): override context_types per task
        per_task_modes = getattr(self, '_per_task_modes', None)
        if per_task_modes is not None:
            for idx, mode in enumerate(per_task_modes):
                if idx < batch_size:
                    if mode == 'noskill':
                        self.context_types[idx] = 'no_skill'

        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            # Two-phase contrastive: probe mode overrides all to no_skill
            if getattr(self, '_contrastive_probe_mode', False):
                self.context_types = ['no_skill'] * batch_size

            for idx, task in enumerate(self.tasks):
                if self.context_types[idx] == 'no_skill':
                    self.retrieved_memories.append(None)
                else:
                    task_type = infos[idx].get('skill_type') if idx < len(infos) else None
                    memories = self.retrieval_memory.retrieve(
                        task_description=task,
                        top_k=mem_config.get('top_k', 10),
                        similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                        max_tokens=mem_config.get('max_tokens', 2000),
                        include_examples=mem_config.get('include_examples', False),
                        task_type=task_type,
                    )
                    self.retrieved_memories.append(memories)

        text_obs_result = self.build_text_obs(obs, init=True)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                obs, init=True, counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                obs, init=True, counterfactual_mode='no_skill')

        observations = {
            "text": guided_texts,
            "text_plain": plain_texts,
            "text_teacher_probe": teacher_probe_texts,
            "text_no_skill_probe": no_skill_probe_texts,
            "image": None,
            "anchor": obs.copy(),
            "context_type": self.context_types,
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        text_obs_result = self.build_text_obs(next_obs)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                next_obs, counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                next_obs, counterfactual_mode='no_skill')

        next_observations = {
            "text": guided_texts,
            "text_plain": plain_texts,
            "text_teacher_probe": teacher_probe_texts,
            "text_no_skill_probe": no_skill_probe_texts,
            "image": None,
            "anchor": next_obs.copy()
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
        counterfactual_mode: str = None,
    ):
        postprocess_text_obs: List[str] = []
        build_plain_pair = self.guide_internalize and counterfactual_mode is None
        plain_text_obs: List[str] = [] if build_plain_pair else None

        # Per-task mode support: determine exclude_hints per index
        per_task_modes = getattr(self, '_per_task_modes', None)

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            # Per-task exclude_hints: "guided" → False, "plain"/"noskill" → True
            if per_task_modes is not None and i < len(per_task_modes):
                exclude_hints_i = (per_task_modes[i] != 'guided')
            else:
                exclude_hints_i = self.exclude_hints

            # Per-trial retrieval check: no_skill trials have retrieved_memories[i] = None
            use_retrieval_i = (self.retrieval_memory is not None and
                              self.retrieved_memories is not None and
                              self.retrieved_memories[i] is not None)
            if counterfactual_mode == 'teacher':
                exclude_hints_i = False
            elif counterfactual_mode == 'no_skill':
                use_retrieval_i = False

            if init and use_retrieval_i:
                # Step-0 with skills (all modes: train, val, eval)
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs_i = SEARCH_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                )
            elif init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            elif use_retrieval_i:
                # Format retrieved memories for prompt
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    memory_context=memory_ctx[i],
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

            # Build plain text (without hints) for guide internalization
            if build_plain_pair and use_retrieval_i:
                plain_memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=True
                )
                if init:
                    plain_i = SEARCH_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                    )
                else:
                    plain_i = SEARCH_TEMPLATE_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                        step_count=len(self.memory[i]),
                        memory_context=memory_ctx[i],
                    )
                plain_text_obs.append(plain_i)
            elif build_plain_pair:
                # No retrieval memory, plain = guided
                plain_text_obs.append(obs_i)

        if build_plain_pair:
            return postprocess_text_obs, plain_text_obs
        return postprocess_text_obs

    def get_last_reset_info(self) -> dict:
        """Return the QA rows needed to replay the same tasks in phase two."""
        return getattr(self, '_last_reset_info', {})


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.guide_internalize = True
        self.exclude_hints = config.env.get('exclude_hints', False)
        # Store original values for mode restoration
        self._orig_guide_internalize = True
        self._orig_exclude_hints = self.exclude_hints

        # Two-phase contrastive probe
        base_group_n = config.env.rollout.n if config.env.rollout.get('n', 0) > 0 else 1
        self.group_n = base_group_n
        self._contrastive_probe_mode = False

        # Add retrieval memory or skills-only memory if configured
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
                include_general=som_cfg.get('include_general', True),
            )
            self.retrieved_memories = None
            print(f"[AlfWorldEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')}, include_general={som_cfg.get('include_general', True)})")
        elif config.env.get('use_retrieval_memory', False):
            from agent_system.memory import RetrievalMemory
            self.retrieval_memory = RetrievalMemory(
                memory_json_path=config.env.retrieval_memory.json_path,
                embedding_model_name=config.env.retrieval_memory.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B'),
                device=config.env.retrieval_memory.get('device', 'cuda'),
                skills_json_path=config.env.retrieval_memory.get('skills_json_path', None)
            )
            self.retrieved_memories = None  # Store retrieved memories per episode
            print(f"[AlfWorldEnvironmentManager] Retrieval memory enabled")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    def set_mode(self, plain: bool):
        """Switch prompt construction mode for HDPO two-phase rollout.

        plain=True  (Phase 1): exclude_hints=True, guide_internalize=False
            → build_text_obs returns plain-only text (specific skills only)
        plain=False (Phase 2): exclude_hints=False, guide_internalize=True
            → build_text_obs returns (guided_texts, plain_texts) tuple
        """
        if plain:
            self.exclude_hints = True
            self.guide_internalize = False
        else:
            self.exclude_hints = False
            self.guide_internalize = True

    def restore_mode(self):
        """Restore original mode after HDPO rollout."""
        self.guide_internalize = self._orig_guide_internalize
        self.exclude_hints = self._orig_exclude_hints

    def set_contrastive_probe(self, probe: bool):
        """Switch contrastive probe mode for two-phase rollout.

        probe=True:  next reset() marks ALL envs as no_skill (Phase 2)
        probe=False: next reset() marks ALL envs as top_k (Phase 1, default)
        """
        self._contrastive_probe_mode = probe

    def set_per_task_mode(self, task_modes: list):
        """Set per-task mode for unified Phase 2 in ours_step.

        task_modes: list of str, one per task. Values:
            "guided"  — hard tasks: all skills, guide_internalize per-index
            "noskill" — easy tasks: no skills (contrastive probe)
            "plain"   — medium tasks (filler): specific skills only, exclude hints
        """
        self._per_task_modes = task_modes

    def clear_per_task_mode(self):
        """Clear per-task mode after Phase 2."""
        self._per_task_modes = None

    def reset(self, kwargs, n: int = None, reset_info: dict = None):
        game_files = reset_info.get('game_files') if reset_info else None
        text_obs, image_obs, infos = self.envs.reset(game_files=game_files)
        self.gamefile = parse_gamefile(infos)
        self._last_reset_info = {'game_files': self.gamefile}
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        # Determine context_type per env for contrastive training
        batch_size = len(text_obs)
        self.context_types = ['top_k'] * batch_size  # default: all top-k

        # Per-task mode (ours Phase 2): override context_types per task
        per_task_modes = getattr(self, '_per_task_modes', None)
        if per_task_modes is not None:
            for idx, mode in enumerate(per_task_modes):
                if idx < batch_size:
                    if mode == 'noskill':
                        self.context_types[idx] = 'no_skill'
                    # 'guided' and 'plain' keep 'top_k' (skills retrieved)
            n_noskill = sum(1 for ct in self.context_types if ct == 'no_skill')
            print(f"[EnvManager DEBUG] reset: batch_size={batch_size}, per_task_modes_len={len(per_task_modes)}, "
                  f"n_noskill_in_modes={sum(1 for m in per_task_modes if m == 'noskill')}, "
                  f"n_noskill_in_context_types={n_noskill}")

        # Retrieve memories for each task if enabled
        if self.retrieval_memory is not None:
            self.retrieved_memories = []

            # Determine which config to use
            if self.config.env.get('use_skills_only_memory', False):
                mem_config = self.config.env.skills_only_memory
            else:
                mem_config = self.config.env.retrieval_memory

            # Two-phase contrastive: probe mode overrides all to no_skill
            if getattr(self, '_contrastive_probe_mode', False):
                self.context_types = ['no_skill'] * batch_size

            for idx, task in enumerate(self.tasks):
                if self.context_types[idx] == 'no_skill':
                    self.retrieved_memories.append(None)
                else:
                    memories = self.retrieval_memory.retrieve(
                        task_description=task,
                        top_k=mem_config.get('top_k', 10),
                        similarity_threshold=mem_config.get('similarity_threshold', 0.7),
                        max_tokens=mem_config.get('max_tokens', 2000),
                        include_examples=mem_config.get('include_examples', False)
                    )
                    self.retrieved_memories.append(memories)
            n_none_mem = sum(1 for m in self.retrieved_memories if m is None)
            print(f"[EnvManager DEBUG] reset: retrieved_memories count={len(self.retrieved_memories)}, n_None={n_none_mem}")

        text_obs_result = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        # Action-IG uses the student trajectory as the only environment path.
        # During the plain/student rollout, cache counterfactual prompts for the
        # same state so hard/easy routing can score one teacher/no-skill action
        # later without executing another environment trajectory.
        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                text_obs, self.envs.get_admissible_commands, init=True,
                counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                text_obs, self.envs.get_admissible_commands, init=True,
                counterfactual_mode='no_skill')

        return {
            'text': guided_texts,
            'text_plain': plain_texts,
            'text_teacher_probe': teacher_probe_texts,
            'text_no_skill_probe': no_skill_probe_texts,
            'image': image_obs,
            'anchor': text_obs,
            'context_type': self.context_types,
        }, infos

    def get_last_reset_info(self) -> dict:
        """Return info needed to replay the same reset (for wave-mode)."""
        return getattr(self, '_last_reset_info', {})

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        text_obs_result = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                text_obs, self.envs.get_admissible_commands,
                counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                text_obs, self.envs.get_admissible_commands,
                counterfactual_mode='no_skill')

        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {
            'text': guided_texts,
            'text_plain': plain_texts,
            'text_teacher_probe': teacher_probe_texts,
            'text_no_skill_probe': no_skill_probe_texts,
            'image': image_obs,
            'anchor': text_obs,
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
        counterfactual_mode: str = None,
    ):
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        build_plain_pair = self.guide_internalize and counterfactual_mode is None
        plain_text_obs: List[str] = [] if build_plain_pair else None

        # Per-task mode support: determine exclude_hints per index
        per_task_modes = getattr(self, '_per_task_modes', None)

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            # Per-task exclude_hints: "guided" → False, "plain"/"noskill" → True
            if counterfactual_mode == 'teacher':
                exclude_hints_i = False
            elif counterfactual_mode == 'no_skill':
                exclude_hints_i = True
            elif per_task_modes is not None and i < len(per_task_modes):
                exclude_hints_i = (per_task_modes[i] != 'guided')
            else:
                exclude_hints_i = self.exclude_hints

            # Per-trial retrieval check: no_skill trials have retrieved_memories[i] = None
            use_retrieval_i = (self.retrieval_memory is not None and
                              self.retrieved_memories is not None and
                              self.retrieved_memories[i] is not None)
            if counterfactual_mode == 'no_skill':
                use_retrieval_i = False

            if init and use_retrieval_i:
                # Step-0 with skills (all modes: train, val, eval)
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            elif use_retrieval_i:
                # Step 1+ with skills
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

            # Build plain text (without hints) for guide internalization
            if build_plain_pair and use_retrieval_i:
                plain_memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=True
                )
                if init:
                    plain_obs = ALFWORLD_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
                else:
                    plain_obs = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        admissible_actions=reformatted_admissible_actions
                    )
                plain_text_obs.append(plain_obs)
            elif build_plain_pair:
                plain_text_obs.append(obs)

        if build_plain_pair:
            return postprocess_text_obs, plain_text_obs
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

    def save_episode_trajectories(self, batch_data_list, infos_list):
        """
        Save successful/failed trajectories from completed episodes to memory pool.

        Args:
            batch_idx: Index of the batch
            total_batch_list: List of batch data containing trajectories
            infos: List of info dicts containing episode metadata
        """
        if self.retrieval_memory is None:
            return

        save_dir = self.config.env.retrieval_memory.get('save_dir', None)
        if save_dir is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'new_memories.json')

        # Iterate through each environment
        for env_idx in range(len(self.tasks)):
            # Check if episode is done
            # We'll save trajectories when episodes complete
            # This will be called from the trainer after validation/training episodes
            pass  # Actual saving logic will be called from trainer


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs, n: int = None):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs, n: int = None) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.guide_internalize = config.env.get('guide_internalize', False)
        self.exclude_hints = config.env.get('exclude_hints', False)
        # Store original values for HDPO mode restoration
        self._orig_guide_internalize = self.guide_internalize
        self._orig_exclude_hints = self.exclude_hints

        # Two-phase contrastive probe (used by _utilize_step / _ours_step)
        base_group_n = config.env.rollout.n if config.env.rollout.get('n', 0) > 0 else 1
        self.group_n = base_group_n
        self._contrastive_probe_mode = False

        # Skills-only memory (same interface as AlfWorldEnvironmentManager)
        if config.env.get('use_skills_only_memory', False):
            from agent_system.memory import SkillsOnlyMemory
            som_cfg = config.env.skills_only_memory
            self.retrieval_memory = SkillsOnlyMemory(
                skills_json_path=som_cfg.skills_json_path,
                retrieval_mode=som_cfg.get('retrieval_mode', 'template'),
                embedding_model_path=som_cfg.get('embedding_model_path', None),
                task_specific_top_k=som_cfg.get('task_specific_top_k', None),
            )
            self.retrieved_memories = None
            print(f"[WebshopEnvironmentManager] Skills-only memory enabled "
                  f"(mode={som_cfg.get('retrieval_mode', 'template')})")
        else:
            self.retrieval_memory = None

        super().__init__(envs, projection_f, config)

    def set_mode(self, plain: bool):
        """Switch prompt construction mode for HDPO two-phase rollout.

        plain=True  (Phase 1): exclude_hints=True, guide_internalize=False
            → build_text_obs returns plain-only text (specific skills only)
        plain=False (Phase 2): exclude_hints=False, guide_internalize=True
            → build_text_obs returns (guided_texts, plain_texts) tuple
        """
        if plain:
            self.exclude_hints = True
            self.guide_internalize = False
        else:
            self.exclude_hints = False
            self.guide_internalize = True

    def restore_mode(self):
        """Restore original mode after HDPO rollout."""
        self.guide_internalize = self._orig_guide_internalize
        self.exclude_hints = self._orig_exclude_hints

    def set_contrastive_probe(self, probe: bool):
        """Switch contrastive probe mode for two-phase rollout.

        probe=True:  next reset() marks ALL envs as no_skill (Phase 2)
        probe=False: next reset() marks ALL envs as top_k (Phase 1, default)
        """
        self._contrastive_probe_mode = probe

    def set_per_task_mode(self, task_modes: list):
        """Set per-task mode for unified Phase 2 in ours_step.

        task_modes: list of str, one per task. Values:
            "guided"  — hard tasks: all skills, guide_internalize per-index
            "noskill" — easy tasks: no skills (contrastive probe)
            "plain"   — medium tasks (filler): specific skills only, exclude hints
        """
        self._per_task_modes = task_modes

    def clear_per_task_mode(self):
        """Clear per-task mode after Phase 2."""
        self._per_task_modes = None

    def reset_eval_cursor(self) -> None:
        """Reset the inner envs' eval cursor for a new validation round."""
        if hasattr(self.envs, 'reset_eval_cursor'):
            self.envs.reset_eval_cursor()

    def reset(self, kwargs, n: Optional[int] = None, reset_info: dict = None) -> Dict[str, Any]:
        goal_indices = reset_info.get('goal_indices') if reset_info else None
        obs, infos = self.envs.reset(n=n, goal_indices=goal_indices)
        self._last_reset_info = {'goal_indices': self.envs.get_last_goal_indices()}
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)

        # Determine context_type per env for contrastive training
        batch_size = len(obs)
        self.context_types = ['top_k'] * batch_size  # default: all top-k

        # Per-task mode (ours Phase 2): override context_types per task
        per_task_modes = getattr(self, '_per_task_modes', None)
        if per_task_modes is not None:
            for idx, mode in enumerate(per_task_modes):
                if idx < batch_size:
                    if mode == 'noskill':
                        self.context_types[idx] = 'no_skill'

        # Retrieve skills for each task if memory is configured
        if self.retrieval_memory is not None:
            mem_cfg = self.config.env.skills_only_memory
            self.retrieved_memories = []

            # Two-phase contrastive: probe mode overrides all to no_skill
            if getattr(self, '_contrastive_probe_mode', False):
                self.context_types = ['no_skill'] * batch_size

            for idx, task in enumerate(self.tasks):
                if self.context_types[idx] == 'no_skill':
                    self.retrieved_memories.append(None)
                else:
                    memories = self.retrieval_memory.retrieve(
                        task_description=task,
                        top_k=mem_cfg.get('top_k', 6),
                    )
                    self.retrieved_memories.append(memories)

        text_obs_result = self.build_text_obs(obs, infos, init=True)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                obs, infos, init=True, counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                obs, infos, init=True, counterfactual_mode='no_skill')

        observations = {'text': guided_texts,
                        'text_plain': plain_texts,
                        'text_teacher_probe': teacher_probe_texts,
                        'text_no_skill_probe': no_skill_probe_texts,
                        'image': None,
                        'anchor': obs.copy(),
                        'context_type': self.context_types,
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def get_last_reset_info(self) -> dict:
        """Return info needed to replay the same reset (for wave-mode)."""
        return getattr(self, '_last_reset_info', {})

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        text_obs_result = self.build_text_obs(next_obs, infos)
        if self.guide_internalize and isinstance(text_obs_result, tuple):
            guided_texts, plain_texts = text_obs_result
        else:
            guided_texts = text_obs_result
            plain_texts = None

        teacher_probe_texts = None
        no_skill_probe_texts = None
        action_ig_enabled = self.config.env.get('ours', {}).get('action_ig_enabled', True)
        if action_ig_enabled and self.exclude_hints and not self.guide_internalize:
            teacher_probe_texts = self.build_text_obs(
                next_obs, infos, counterfactual_mode='teacher')
            no_skill_probe_texts = self.build_text_obs(
                next_obs, infos, counterfactual_mode='no_skill')

        next_observations = {
            'text': guided_texts,
            'text_plain': plain_texts,
            'text_teacher_probe': teacher_probe_texts,
            'text_no_skill_probe': no_skill_probe_texts,
            'image': None,
            'anchor': next_obs.copy(),
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[List[str]],
        init: bool = False,
        counterfactual_mode: str = None,
    ):
        """
        Build text observation for the agent.

        When ``guide_internalize`` is True, returns ``(guided_texts, plain_texts)``
        where *guided* includes hints and *plain* excludes them
        (same pattern as AlfWorldEnvironmentManager).
        """
        postprocess_text_obs = []
        build_plain_pair = self.guide_internalize and counterfactual_mode is None
        plain_text_obs: List[str] = [] if build_plain_pair else None

        # Per-task mode support: determine exclude_hints per index
        per_task_modes = getattr(self, '_per_task_modes', None)

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")

        for i in range(len(text_obs)):

            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            # Per-task exclude_hints: "guided" → False, "plain"/"noskill" → True
            if counterfactual_mode == 'teacher':
                exclude_hints_i = False
            elif counterfactual_mode == 'no_skill':
                exclude_hints_i = True
            elif per_task_modes is not None and i < len(per_task_modes):
                exclude_hints_i = (per_task_modes[i] != 'guided')
            else:
                exclude_hints_i = self.exclude_hints

            # Per-trial retrieval check: no_skill trials have retrieved_memories[i] = None
            use_retrieval_i = (
                self.retrieval_memory is not None
                and self.retrieved_memories is not None
                and self.retrieved_memories[i] is not None
            )
            if counterfactual_mode == 'no_skill':
                use_retrieval_i = False

            if init and use_retrieval_i:
                # Step-0 with skills
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs = WEBSHOP_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            elif init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            elif use_retrieval_i:
                memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=exclude_hints_i
                )
                obs = WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                    task_description=self.tasks[i],
                    retrieved_memories=memory_context,
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )

            postprocess_text_obs.append(obs)

            # Build plain text (without hints) for guide internalization
            if build_plain_pair and use_retrieval_i:
                plain_memory_context = self.retrieval_memory.format_for_prompt(
                    self.retrieved_memories[i], exclude_hints=True
                )
                if init:
                    plain_obs = WEBSHOP_TEMPLATE_WITH_MEMORY_NO_HIS.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                else:
                    plain_obs = WEBSHOP_TEMPLATE_WITH_MEMORY.format(
                        task_description=self.tasks[i],
                        retrieved_memories=plain_memory_context,
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )
                plain_text_obs.append(plain_obs)
            elif build_plain_pair:
                # No retrieval context — plain == guided
                plain_text_obs.append(obs)

        if build_plain_pair:
            return postprocess_text_obs, plain_text_obs
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)

                # Per-category success rate (e.g. apparel_success_rate)
                if batch_idx < len(self.tasks):
                    from agent_system.environments.env_package.webshop.envs import classify_webshop_goal
                    cat = classify_webshop_goal(self.tasks[batch_idx])
                    success[f"{cat}_success_rate"].append(won_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs, n: int = None):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    wave_mode = config.env.rollout.get('wave_mode', False)
    if wave_mode:
        group_n = config.env.rollout.get('wave_batch_size', 1)
    else:
        group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from copy import deepcopy
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)

        # Val envs: disable guide_internalize, enable exclude_hints if guide mode
        config_val = deepcopy(config)
        if config.env.get('guide_internalize', False):
            OmegaConf.update(config_val, "env.guide_internalize", False, force_add=True)
            OmegaConf.update(config_val, "env.exclude_hints", True, force_add=True)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config_val)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        webshop_data_dir = os.environ.get(
            'WEBSHOP_DATA',
            os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data'),
        )
        if config.env.webshop.use_small:
            file_path = os.path.join(webshop_data_dir, 'items_shuffle_1000.json')
            attr_path = os.path.join(webshop_data_dir, 'items_ins_v2_1000.json')
        else:
            file_path = os.path.join(webshop_data_dir, 'items_shuffle.json')
            attr_path = os.path.join(webshop_data_dir, 'items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text',
                    'num_products': None,
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


def make_envs_ood(config):
    """Create train (ID), val_id, and val_ood environments for OOD experiments.

    Only supports ALFWorld. Requires config.env.alfworld_ood with:
        - id_task_types: list of task type IDs for in-distribution (e.g. [1, 3, 5])
        - ood_task_types: list of task type IDs for out-of-distribution (e.g. [2, 4, 6])
        - skills_json_path: path to OOD skill JSON file

    Returns:
        envs, val_envs, val_envs_ood, val_id_size, val_ood_size
        where val_id_size / val_ood_size are the actual game counts (for DataLoader sizing).
    """
    from copy import deepcopy
    from functools import partial

    assert "alfworld" in config.env.env_name.lower(), "make_envs_ood only supports ALFWorld"

    from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection, count_alfworld_games

    alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')

    ood_cfg = config.env.alfworld_ood
    id_task_types = list(ood_cfg.id_task_types)
    ood_task_types = list(ood_cfg.ood_task_types)
    eval_dataset = config.env.alfworld.eval_dataset

    wave_mode = config.env.rollout.get('wave_mode', False)
    if wave_mode:
        group_n = config.env.rollout.get('wave_batch_size', 1)
    else:
        group_n = config.env.rollout.n
    resources_per_worker = {k: v for k, v in config.env.resources_per_worker.items()}

    # --- Count actual game files for val sets (before creating workers) ---
    val_id_env_kwargs = {'eval_dataset': eval_dataset, 'task_types': id_task_types}
    val_ood_env_kwargs = {'eval_dataset': eval_dataset, 'task_types': ood_task_types}

    val_id_size = count_alfworld_games(alf_config_path, is_train=False, env_kwargs=val_id_env_kwargs)
    val_ood_size = count_alfworld_games(alf_config_path, is_train=False, env_kwargs=val_ood_env_kwargs)

    # Train: ID task types only
    train_env_kwargs = {'eval_dataset': eval_dataset, 'task_types': id_task_types}
    _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n,
                                is_train=True, env_kwargs=train_env_kwargs, resources_per_worker=resources_per_worker)

    # Val ID: use actual game count as env_num
    _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, val_id_size, 1,
                                    is_train=False, env_kwargs=val_id_env_kwargs, resources_per_worker=resources_per_worker)

    # Val OOD: use actual game count as env_num
    _val_envs_ood = build_alfworld_envs(alf_config_path, config.env.seed + 2000, val_ood_size, 1,
                                        is_train=False, env_kwargs=val_ood_env_kwargs, resources_per_worker=resources_per_worker)

    projection_f = partial(alfworld_projection)

    # Train uses the ID skill JSON (from config.env.skills_only_memory)
    envs = AlfWorldEnvironmentManager(_envs, projection_f, config)

    # Val envs: disable guide_internalize, exclude hints (val uses plain mode)
    config_val = deepcopy(config)
    if config.env.get('guide_internalize', False):
        OmegaConf.update(config_val, "env.guide_internalize", False, force_add=True)
        OmegaConf.update(config_val, "env.exclude_hints", True, force_add=True)
    val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config_val)

    # Val OOD uses the OOD skill JSON — deep copy config and override skills_json_path
    config_ood = deepcopy(config)
    config_ood.env.skills_only_memory.skills_json_path = ood_cfg.skills_json_path
    if config.env.get('guide_internalize', False):
        OmegaConf.update(config_ood, "env.guide_internalize", False, force_add=True)
        OmegaConf.update(config_ood, "env.exclude_hints", True, force_add=True)
    val_envs_ood = AlfWorldEnvironmentManager(_val_envs_ood, projection_f, config_ood)

    print(f"[make_envs_ood] ID task_types={id_task_types}, OOD task_types={ood_task_types}")
    print(f"[make_envs_ood] val_id_size={val_id_size}, val_ood_size={val_ood_size}")
    print(f"[make_envs_ood] Train/Val-ID skills: {config.env.skills_only_memory.skills_json_path}")
    print(f"[make_envs_ood] Val-OOD skills: {ood_cfg.skills_json_path}")

    return envs, val_envs, val_envs_ood, val_id_size, val_ood_size


def make_envs_webshop_ood(config):
    """Create train (ID), val_id, and val_ood environments for WebShop OOD experiments.

    Data split follows WebShop original convention:
        - test:  index 0-499
        - eval:  index 500-1499
        - train: index 1500+

    Supports two modes:

    1. **Precomputed splits** (preferred): set ``config.env.webshop_ood.splits_file``
       to a JSON file generated by ``preprocess_webshop_ood.py``.  Goal indices
       are loaded directly, skipping all runtime classification/FPS.

    2. **Runtime splits** (legacy): uses ``filter_categories``, ``split``,
       ``downsample_other_*``, and ``fps_cache_*`` config keys.

    All three environments use the **same seed** so that goal ordering is
    consistent and the same goal index maps to the same instruction across envs.

    Returns:
        envs, val_envs, val_envs_ood, val_id_size, val_ood_size,
        val_id_env_num, val_ood_env_num
    """
    from copy import deepcopy
    from functools import partial
    import time

    assert "webshop" in config.env.env_name.lower(), "make_envs_webshop_ood only supports WebShop"

    from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection

    ood_cfg = config.env.webshop_ood
    id_categories = list(ood_cfg.id_categories)
    ood_categories = list(ood_cfg.ood_categories)

    # Same seed for all envs so goal ordering is consistent
    env_seed = config.env.seed

    wave_mode = config.env.rollout.get('wave_mode', False)
    if wave_mode:
        group_n = config.env.rollout.get('wave_batch_size', 1)
    else:
        group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    # --- Data paths ---
    webshop_data_dir = os.environ.get(
        'WEBSHOP_DATA',
        os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data'),
    )
    if config.env.webshop.use_small:
        file_path = os.path.join(webshop_data_dir, 'items_shuffle_1000.json')
        attr_path = os.path.join(webshop_data_dir, 'items_ins_v2_1000.json')
    elif config.env.webshop.human_goals:
        # Use the human-goal subset (~50K products, ~244MB) instead of the
        # full 1.18M products (5.2GB) to keep per-worker memory manageable.
        file_path = os.path.join(webshop_data_dir, 'items_shuffle_human.json')
        attr_path = os.path.join(webshop_data_dir, 'items_ins_v2_human.json')
    else:
        file_path = os.path.join(webshop_data_dir, 'items_shuffle.json')
        attr_path = os.path.join(webshop_data_dir, 'items_ins_v2.json')
    env_kwargs = {
        'observation_mode': 'text',
        'num_products': None,
        'human_goals': config.env.webshop.human_goals,
        'file_path': file_path,
        'attr_path': attr_path,
    }

    # --- Check for precomputed splits ---
    splits_file = ood_cfg.get('splits_file', None)
    if splits_file:
        import json as _json
        with open(splits_file) as _f:
            splits = _json.load(_f)
        train_goal_idxs = splits['train']['goal_idxs']
        val_id_goal_idxs = splits['val_id']['goal_idxs']
        val_ood_goal_idxs = splits['val_ood']['goal_idxs']
        print(f"[make_envs_webshop_ood] Loaded precomputed splits from {splits_file}")
        print(f"  train: {splits['train']['count']} goals {splits['train']['by_category']}")
        print(f"  val_id: {splits['val_id']['count']} goals {splits['val_id']['by_category']}")
        print(f"  val_ood: {splits['val_ood']['count']} goals {splits['val_ood']['by_category']}")
    else:
        train_goal_idxs = None
        val_id_goal_idxs = None
        val_ood_goal_idxs = None

    # --- Downsample config (only used in legacy mode) ---
    emb_model_path = config.env.skills_only_memory.get('embedding_model_path', None)
    ds_train_n = ood_cfg.get('downsample_other_train', None)
    ds_val_n = ood_cfg.get('downsample_other_val', None)
    fps_cache_train = ood_cfg.get('fps_cache_train', None)
    fps_cache_val = ood_cfg.get('fps_cache_val', None)
    ds_train = {'n': ds_train_n, 'embedding_model_path': emb_model_path, 'cache_file': fps_cache_train} if ds_train_n and not splits_file else None
    ds_val = {'n': ds_val_n, 'embedding_model_path': emb_model_path, 'cache_file': fps_cache_val} if ds_val_n and not splits_file else None

    # --- Train: ID categories from train split (1500+) ---
    _envs = build_webshop_envs(
        seed=env_seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
        filter_categories=id_categories if not splits_file else None,
        split='train' if not splits_file else None,
        downsample_other=ds_train,
        precomputed_goal_idxs=train_goal_idxs,
    )

    # --- Val ID: ID categories from test+eval (0-1499) ---
    val_id_batch_size = config.data.val_batch_size
    _val_envs = build_webshop_envs(
        seed=env_seed,
        env_num=val_id_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
        filter_categories=id_categories if not splits_file else None,
        split='test+eval' if not splits_file else None,
        downsample_other=ds_val,
        precomputed_goal_idxs=val_id_goal_idxs,
    )

    # --- Val OOD: OOD categories from test+eval (0-1499) ---
    val_ood_batch_size = config.data.get("val_ood_batch_size", config.data.val_batch_size)
    _val_envs_ood = build_webshop_envs(
        seed=env_seed,
        env_num=val_ood_batch_size,
        group_n=1,
        is_train=False,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
        filter_categories=ood_categories if not splits_file else None,
        split='test+eval' if not splits_file else None,
        precomputed_goal_idxs=val_ood_goal_idxs,
    )

    # Full goal count for dataset sizing — the dataloader will split into
    # multiple batches of env_num each, achieving full-coverage eval.
    val_id_size = _val_envs.get_goal_count()
    val_ood_size = _val_envs_ood.get_goal_count()

    projection_f = partial(webshop_projection)

    # Train uses ID skills
    envs = WebshopEnvironmentManager(_envs, projection_f, config)

    # Val ID: disable guide_internalize, exclude hints (val uses plain mode)
    config_val = deepcopy(config)
    if config.env.get('guide_internalize', False):
        OmegaConf.update(config_val, "env.guide_internalize", False, force_add=True)
        OmegaConf.update(config_val, "env.exclude_hints", True, force_add=True)
    val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config_val)

    # Val OOD: use OOD skill JSON
    config_ood = deepcopy(config)
    config_ood.env.skills_only_memory.skills_json_path = ood_cfg.skills_json_path
    if config.env.get('guide_internalize', False):
        OmegaConf.update(config_ood, "env.guide_internalize", False, force_add=True)
        OmegaConf.update(config_ood, "env.exclude_hints", True, force_add=True)
    val_envs_ood = WebshopEnvironmentManager(_val_envs_ood, projection_f, config_ood)

    # Wait for Ray workers to be ready (use actual clamped sizes)
    total_workers = (config.data.train_batch_size * group_n + val_id_size + val_ood_size)
    time.sleep(total_workers * 0.1)

    # Worker counts (clamped) — used as dataloader batch_size
    val_id_env_num = _val_envs.env_num
    val_ood_env_num = _val_envs_ood.env_num

    print(f"[make_envs_webshop_ood] ID categories={id_categories}, OOD categories={ood_categories}")
    if splits_file:
        print(f"[make_envs_webshop_ood] Using precomputed splits: {splits_file}")
    else:
        print(f"[make_envs_webshop_ood] Train: split='train'(1500+), ID only"
              f"{f', other downsampled to {ds_train_n}' if ds_train_n else ''}")
    print(f"[make_envs_webshop_ood] val_id: {val_id_size} goals, "
          f"{val_id_env_num} workers -> {-(-val_id_size // val_id_env_num)} batches")
    print(f"[make_envs_webshop_ood] val_ood: {val_ood_size} goals, "
          f"{val_ood_env_num} workers -> {-(-val_ood_size // val_ood_env_num)} batches")
    print(f"[make_envs_webshop_ood] All envs use seed={env_seed}")
    print(f"[make_envs_webshop_ood] Train/Val-ID skills: {config.env.skills_only_memory.skills_json_path}")
    print(f"[make_envs_webshop_ood] Val-OOD skills: {ood_cfg.skills_json_path}")

    return envs, val_envs, val_envs_ood, val_id_size, val_ood_size, val_id_env_num, val_ood_env_num

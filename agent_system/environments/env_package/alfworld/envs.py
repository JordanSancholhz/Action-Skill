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

import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """

    def __init__(self, config, seed, base_env, assigned_game_file=None):
        if assigned_game_file is not None:
            # Eval mode: register only the assigned game file for deterministic coverage
            self.env = base_env.init_env_single(batch_size=1, game_file=assigned_game_file)
        else:
            # Train mode: register all game files, sample randomly
            self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos

    def reset_to_game(self, game_file):
        """Reset to a specific game file (for wave-mode replay)."""
        tw_env = self.env  # TextworldBatchGymEnv instance
        if tw_env.batch_env is not None:
            tw_env.batch_env.close()
        tw_env.batch_env.load([game_file])
        tw_env.last_commands = [None] * tw_env.batch_size
        tw_env.obs, infos = tw_env.batch_env.reset()
        infos['observation_text'] = tw_env.obs
        return tw_env.obs, infos

    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        # Allow env_kwargs to override task_types for OOD experiments
        task_types_override = env_kwargs.get('task_types', None)
        if task_types_override is not None:
            config['env']['task_types'] = task_types_override
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []

        if not is_train:
            # Eval mode: assign each worker a unique game file for full coverage
            game_files = base_env.game_files
            assert self.num_processes <= len(game_files), (
                f"num_processes={self.num_processes} > num_games={len(game_files)}. "
                f"Cannot assign unique games to each worker.")
            for i in range(self.num_processes):
                game_idx = i // group_n  # within a group, share the same game
                worker = env_worker.remote(config, seed + i, base_env,
                                           assigned_game_file=game_files[game_idx])
                self.workers.append(worker)
        else:
            # Train mode: each worker samples from the full game pool
            for i in range(self.num_processes):
                worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
                self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, game_files=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.

        Args:
            game_files: List of game file paths for wave-mode replay.
                When provided, each worker is reset to the specified game
                instead of advancing to the next game in its shuffled cycle.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        if game_files is not None:
            for worker, gf in zip(self.workers, game_files):
                futures.append(worker.reset_to_game.remote(gf))
        else:
            for worker in self.workers:
                futures.append(worker.reset.remote())

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)


def count_alfworld_games(alf_config_path, is_train=True, env_kwargs={}):
    """Count how many game files exist after task_type filtering, without creating workers.

    Returns the number of games that would be loaded by AlfworldEnvs with the
    same parameters.  Used to dynamically determine val batch sizes.
    """
    eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
    config = load_config_file(alf_config_path)
    task_types_override = env_kwargs.get('task_types', None)
    if task_types_override is not None:
        config['env']['task_types'] = task_types_override
    env_type = config['env']['type']
    base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
    num_games = base_env.num_games
    print(f"[count_alfworld_games] is_train={is_train}, task_types={config['env']['task_types']}, "
          f"num_games={num_games}")
    return num_games
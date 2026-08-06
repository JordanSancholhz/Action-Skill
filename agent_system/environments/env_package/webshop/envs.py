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
import ray
import gymnasium as gym
import numpy as np
from typing import List, Optional, Dict


# -----------------------------------------------------------------------------
# Goal classification utility -------------------------------------------------
# -----------------------------------------------------------------------------

def classify_webshop_goal(instruction_text: str) -> str:
    """Classify a WebShop goal into one of 7 categories using keyword matching.

    Mirrors the logic in ``SkillsOnlyMemory._detect_task_type`` for WebShop.
    """
    goal = instruction_text.lower()
    if any(kw in goal for kw in [
        'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
        'blouse', 'clothing', 'clothes', 't-shirt',
    ]):
        return 'apparel'
    elif any(kw in goal for kw in [
        'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper', 'footwear',
    ]):
        return 'footwear'
    elif any(kw in goal for kw in [
        'laptop', 'phone', 'computer', 'tablet', 'charger',
        'cable', 'headphone', 'speaker', 'camera', 'electronic',
    ]):
        return 'electronics'
    elif any(kw in goal for kw in [
        'necklace', 'ring', 'bracelet', 'earring', 'watch',
        'jewelry', 'bag', 'purse', 'wallet',
    ]):
        return 'accessories'
    elif any(kw in goal for kw in [
        'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
        'decor', 'candle', 'vase', 'rug',
    ]):
        return 'home_decor'
    elif any(kw in goal for kw in [
        'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
        'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
    ]):
        return 'beauty_health'
    else:
        return 'other'


# -----------------------------------------------------------------------------
# Farthest-point sampling (FPS) for diversity-based downsampling --------------
# -----------------------------------------------------------------------------

def farthest_point_sampling(
    texts: List[str],
    n_samples: int,
    embedding_model_path: str,
    seed: int = 42,
) -> List[int]:
    """Select *n_samples* indices from *texts* via farthest-point sampling.

    Uses a sentence-transformer model to embed all texts, picks a random seed
    point, then iteratively selects the point farthest from the already-chosen
    set (greedy FPS on cosine distance).

    Returns a list of selected indices into the original *texts* list.
    """
    from sentence_transformers import SentenceTransformer

    if n_samples >= len(texts):
        return list(range(len(texts)))

    model = SentenceTransformer(embedding_model_path)
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=False, batch_size=256)
    # embeddings: (N, D), L2-normalised → cosine sim = dot product
    embeddings = np.asarray(embeddings, dtype=np.float32)

    rng = np.random.RandomState(seed)
    N = len(texts)
    selected: List[int] = [int(rng.randint(N))]
    # min_dist[i] = min cosine distance from i to any selected point
    min_dist = np.full(N, np.inf, dtype=np.float32)

    for _ in range(n_samples - 1):
        last = selected[-1]
        # cosine distance = 1 - dot(a, b) for unit vectors
        dist_to_last = 1.0 - embeddings @ embeddings[last]
        np.minimum(min_dist, dist_to_last, out=min_dist)
        # mask already-selected
        min_dist[selected] = -1.0
        selected.append(int(np.argmax(min_dist)))

    del model
    return selected


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', disable_env_checker=True, **env_kwargs).unwrapped
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        filter_categories: Optional[List[str]] = None,
        split: Optional[str] = None,
        downsample_other: Optional[Dict] = None,
        precomputed_goal_idxs: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            split: Data split to use.  When provided, overrides ``is_train``:
                - ``'train'``     → goal indices 1500+  (WebShop original train)
                - ``'test+eval'`` → goal indices 0-1499 (WebShop test + eval)
                - ``'all'``       → all goal indices
                If *None* (default), falls back to legacy behaviour:
                is_train=True → 500+, is_train=False → 0-499.
            downsample_other: Dict with keys ``n`` (target count) and
                ``embedding_model_path``.  When set, the 'other' category
                goals are downsampled to *n* via farthest-point sampling.
            precomputed_goal_idxs: Pre-computed list of goal indices to use
                directly, skipping all split/filter/FPS logic.  Generated by
                ``examples/data_preprocess/preprocess_webshop_ood.py``.
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # ---------- Create Ray workers ----------
        env_worker_cls = ray.remote(**resources_per_worker)(WebshopWorker)
        probe_worker = env_worker_cls.remote(seed, self._env_kwargs)

        # ---------- Determine goal indices ----------
        if precomputed_goal_idxs is not None:
            # Use pre-computed indices directly, skip all runtime logic
            self.goal_idxs = list(precomputed_goal_idxs)
            print(f"[WebshopEnv] Loaded precomputed goal indices: "
                  f"{len(self.goal_idxs)} goals, is_train={is_train}")
        else:
            goals = ray.get(probe_worker.get_goals.remote())

            # ---------- Determine base index range ----------
            if split is not None:
                if split == 'train':
                    base_idxs = list(range(1500, len(goals)))
                elif split == 'test+eval':
                    base_idxs = list(range(0, 1500))
                elif split == 'all':
                    base_idxs = list(range(len(goals)))
                else:
                    raise ValueError(f"Unknown split: {split!r}")
            else:
                # Legacy behaviour
                if not self.is_train:
                    base_idxs = list(range(500))
                else:
                    base_idxs = list(range(500, len(goals)))

            # Optional category-based filtering for OOD experiments
            if filter_categories is not None:
                filter_set = set(filter_categories)
                filtered = []
                other_idxs = []
                category_counts = {}
                for idx in base_idxs:
                    instruction = goals[idx].get('instruction_text', '')
                    cat = classify_webshop_goal(instruction)
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    if cat in filter_set:
                        if cat == 'other':
                            other_idxs.append(idx)
                        else:
                            filtered.append(idx)

                # Downsample 'other' via farthest-point sampling if requested
                if downsample_other is not None and len(other_idxs) > downsample_other['n']:
                    n_target = downsample_other['n']
                    cache_file = downsample_other.get('cache_file', None)
                    loaded_from_cache = False

                    if cache_file and os.path.isfile(cache_file):
                        import json as _json
                        with open(cache_file) as _f:
                            cache = _json.load(_f)
                        cached_set = set(cache['selected_goal_idxs'])
                        other_idxs = [i for i in other_idxs if i in cached_set]
                        loaded_from_cache = True
                        print(f"[WebshopEnv] Loaded FPS cache: {cache_file} "
                              f"({len(cached_set)} cached, {len(other_idxs)} matched)")
                    else:
                        emb_path = downsample_other['embedding_model_path']
                        other_texts = [goals[i]['instruction_text'] for i in other_idxs]
                        print(f"[WebshopEnv] FPS downsampling 'other': {len(other_idxs)} -> {n_target} "
                              f"(embedding_model={emb_path})")
                        selected_local = farthest_point_sampling(
                            other_texts, n_target, emb_path, seed=seed)
                        other_idxs = [other_idxs[j] for j in selected_local]

                filtered.extend(other_idxs)
                self.goal_idxs = filtered
                print(f"[WebshopEnv] split={split}, filter_categories={filter_categories}, "
                      f"is_train={is_train}, "
                      f"category distribution (before filter): {category_counts}, "
                      f"goals after filter: {len(filtered)}/{len(base_idxs)}"
                      f"{f', other downsampled to {len(other_idxs)}' if downsample_other else ''}")
            else:
                self.goal_idxs = base_idxs

        # In eval mode, cap worker count at a reasonable limit to avoid
        # exhausting OS threads/memory.  Each WebshopWorker is a heavy
        # process (Python + BM25 index + product data).
        MAX_EVAL_WORKERS = 64
        if not is_train:
            effective_max = min(len(self.goal_idxs), MAX_EVAL_WORKERS)
            if env_num > effective_max:
                print(f"[WebshopEnv] Eval mode: clamping env_num {env_num} -> {effective_max} "
                      f"(max_workers={MAX_EVAL_WORKERS}, available_goals={len(self.goal_idxs)})")
                env_num = effective_max

        self.env_num = env_num
        self.num_processes = env_num * group_n

        # Eval-mode sequential cursor for full-coverage evaluation.
        # Each reset() call advances the cursor by the requested batch size,
        # cycling back to 0 after all goals have been visited.
        self._eval_cursor = 0

        # ---------- Create Ray workers (reuse probe as worker 0) ----------
        # All workers use the same seed so that goals are shuffled identically
        # and precomputed goal indices map to the same goals across workers.
        self._workers = [probe_worker]
        for i in range(1, self.num_processes):
            worker = env_worker_cls.remote(seed, self._env_kwargs)
            self._workers.append(worker)

        print(f"[WebshopEnv] goal_idxs: {len(self.goal_idxs)} goals, "
              f"env_num: {self.env_num}, num_processes: {self.num_processes}")

    def get_goal_count(self) -> int:
        """Return the number of goals available after filtering."""
        return len(self.goal_idxs)

    def reset_eval_cursor(self) -> None:
        """Reset the eval cursor to 0 for a new validation round."""
        self._eval_cursor = 0

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        n_actions = len(actions)
        if n_actions > self.num_processes:
            raise ValueError(
                f'Too many actions: got {n_actions}, max {self.num_processes}',
            )

        # Send step commands to the first n_actions workers
        futures = []
        for worker, action in zip(self._workers[:n_actions], actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self, n: Optional[int] = None, goal_indices: Optional[List[int]] = None):
        """Reset environments with goal indices.

        Args:
            n: Number of environments to reset. Only used in eval mode to
               support variable-size last batches. Ignored in train mode.
               Must be <= self.env_num.
            goal_indices: Pre-determined goal indices for wave-mode replay.
               When provided, these are used directly instead of random sampling.
        """
        if self.is_train:
            if goal_indices is not None:
                idx = np.array(goal_indices)
            else:
                idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
            self._last_goal_indices = idx.tolist()
            use_n = self.env_num
        else:
            # Eval mode: sequential iteration for full-coverage evaluation.
            use_n = n if n is not None else self.env_num
            assert use_n <= self.env_num, (
                f"Requested n={use_n} exceeds env_num={self.env_num}")
            start = self._eval_cursor
            end = start + use_n
            total = len(self.goal_idxs)
            if end <= total:
                idx = self.goal_idxs[start:end]
            else:
                # Wrap around (shouldn't happen if dataloader sizes are correct,
                # but handle gracefully).
                idx = self.goal_idxs[start:] + self.goal_idxs[:end - total]
            self._eval_cursor = end % total
            idx = np.array(idx)

        idx = np.repeat(idx, self.group_n).tolist()
        num_to_reset = use_n * self.group_n

        # Send reset commands to workers (only first num_to_reset workers)
        futures = []
        for worker, i in zip(self._workers[:num_to_reset], idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def get_last_goal_indices(self) -> Optional[List[int]]:
        """Return goal indices used by the last train-mode reset (for wave replay)."""
        return getattr(self, '_last_goal_indices', None)

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    filter_categories: Optional[List[str]] = None,
    split: Optional[str] = None,
    downsample_other: Optional[Dict] = None,
    precomputed_goal_idxs: Optional[List[int]] = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        filter_categories=filter_categories,
        split=split,
        downsample_other=downsample_other,
        precomputed_goal_idxs=precomputed_goal_idxs,
    )
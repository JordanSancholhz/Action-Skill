# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
    valid_rows: np.ndarray = None,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if valid_rows is not None and not bool(valid_rows[i]):
                continue
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                group_scores = torch.stack(id2score[idx])
                id2mean[idx] = group_scores.mean()
                id2std[idx] = group_scores.std()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if valid_rows is not None and not bool(valid_rows[i]):
                scores[i] = 0.0
                continue
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores



def compute_grpo_decomposed_contrastive_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    contrastive_context_types: np.ndarray,
    omega: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
    ema_delta: float = None,
    adv2_clip: float = 3.0,
):
    """
    Decomposed contrastive advantage: adv1 (task-level GRPO) + omega * adv2 (cross-task skill delta).

    Within each group (same uid/index), trajectories are split into:
      - skill (context_type == 'top_k'): advantage = adv1 + omega * adv2
      - no_skill (context_type == 'no_skill'): probe only, advantage = 0

    adv1: standard within-group GRPO advantage for skill trajectories
      adv1_i = (R_i - mean(R_skill_task)) / (std(R_skill_task) + eps)

    adv2: cross-task skill utilization advantage (same for all trajectories in a task)
      If ema_delta is None (batch mode):
        adv2 = (delta_task - mean(all deltas)) / (std(all deltas) + eps)
      If ema_delta is provided (EMA mode):
        adv2 = clip((delta_task - ema_delta) / (std(all deltas) + eps), -adv2_clip, adv2_clip)

    When only 1 task or std(delta)=0, adv2 degrades to 0 gracefully.
    When all skill trajectories in a task have same reward (std=0), adv1=0 (no intra-group signal).
    """
    scores = token_level_rewards.sum(dim=-1)

    id2skill_scores = defaultdict(list)
    id2skill_indices = defaultdict(list)
    id2noskill_scores = defaultdict(list)
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            if contrastive_context_types[i] == 'top_k':
                id2skill_scores[index[i]].append(scores[i])
                id2skill_indices[index[i]].append(i)
            else:
                id2noskill_scores[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        # Compute per-task statistics
        id2skill_mean = {}
        id2skill_std = {}
        id2noskill_mean = {}
        id2delta = {}

        all_task_ids = list(id2skill_scores.keys())

        for idx in all_task_ids:
            skill_scores = torch.stack(id2skill_scores[idx])
            id2skill_mean[idx] = torch.mean(skill_scores)
            id2skill_std[idx] = torch.std(skill_scores) if len(skill_scores) > 1 else torch.tensor(0.0)

            noskill_scores = id2noskill_scores.get(idx, [])
            if len(noskill_scores) > 0:
                id2noskill_mean[idx] = torch.mean(torch.stack(noskill_scores))
            else:
                id2noskill_mean[idx] = torch.tensor(0.0)

            id2delta[idx] = id2skill_mean[idx] - id2noskill_mean[idx]

        # Compute adv2: cross-task delta normalization
        deltas = torch.stack([id2delta[idx] for idx in all_task_ids]) if len(all_task_ids) > 0 else torch.zeros(1)
        delta_std = torch.std(deltas) if len(deltas) > 1 else torch.tensor(0.0)

        # Determine baseline for adv2
        if ema_delta is not None:
            # EMA mode: use historical EMA as baseline
            delta_baseline = torch.tensor(ema_delta, dtype=deltas.dtype)
        else:
            # Batch mode: use current batch mean as baseline (zero-centered)
            delta_baseline = torch.mean(deltas) if len(deltas) > 1 else torch.tensor(0.0)

        id2adv2 = {}
        for idx in all_task_ids:
            if delta_std > epsilon:
                raw_adv2 = (id2delta[idx] - delta_baseline) / (delta_std + epsilon)
                # Clip to prevent extreme values (especially in EMA mode where mean != baseline)
                id2adv2[idx] = torch.clamp(raw_adv2, -adv2_clip, adv2_clip)
            else:
                id2adv2[idx] = torch.tensor(0.0)

        # Compute final advantages
        advantages = torch.zeros_like(scores)
        for i in range(bsz):
            if contrastive_context_types[i] != 'top_k':
                advantages[i] = 0.0  # no_skill probe: no gradient
            else:
                idx = index[i]
                # adv1: task-level GRPO
                if norm_adv_by_std_in_grpo and id2skill_std[idx] > epsilon:
                    adv1 = (scores[i] - id2skill_mean[idx]) / (id2skill_std[idx] + epsilon)
                else:
                    adv1 = scores[i] - id2skill_mean[idx]
                # adv2: cross-task skill delta
                adv2 = id2adv2[idx]
                advantages[i] = adv1 + omega * adv2

        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        token_count = torch.sum(loss_mask, dim=-1)
        valid_sequences = token_count > 0
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / token_count.clamp(min=1)  # token-mean
        # Padding rows have a zero response mask. Excluding them here avoids
        # both NaNs and accidental dilution of the real action losses.
        loss = seq_losses[valid_sequences].mean() if valid_sequences.any() else seq_losses.sum() * 0.0
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_symmetric_topk_jsd_per_action(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    response_mask: torch.Tensor,
    top_k: int = 64,
    temperature: float = 1.0,
):
    """Return a bounded, symmetric JSD score for each environment action.

    Each batch row is one multi-turn environment action.  For every response
    token, the support is the union of the two policies' top-k tokens plus one
    aggregate tail category.  JSD on this coarse-grained support is symmetric,
    bounded by ``log(2)``, and avoids materializing two full-vocabulary
    probability tensors.  Token scores are averaged within an action so action
    length does not change the information scale.

    Args:
        logits_a: Skill-conditioned response logits, ``(bs, response_len, V)``.
        logits_b: No-skill response logits with the same shape.
        response_mask: Valid response-token mask, ``(bs, response_len)``.
        top_k: Number of high-probability tokens retained from each policy.
        temperature: Softmax temperature used for the information measure.

    Returns:
        action_jsd: One detached-compatible scalar per batch row, ``(bs,)``.
        token_jsd: Per-token JSD, ``(bs, response_len)``.
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(f"JSD logits must have identical shapes, got {logits_a.shape} and {logits_b.shape}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    vocab_size = logits_a.size(-1)
    k = max(1, min(int(top_k), vocab_size))

    # Ranking is invariant to positive temperature scaling, so top-k can be
    # selected from the raw logits.  Only the gathered support and log
    # normalizers are converted to fp32.
    top_a_idx = logits_a.topk(k, dim=-1).indices
    top_b_idx = logits_b.topk(k, dim=-1).indices
    support_idx = torch.cat([top_a_idx, top_b_idx], dim=-1)

    # The second half may repeat tokens already selected by policy A. Masking
    # those duplicates makes the concatenated support a true union.
    duplicate_b = (top_b_idx.unsqueeze(-1) == top_a_idx.unsqueeze(-2)).any(dim=-1)
    support_valid = torch.cat([torch.ones_like(top_a_idx, dtype=torch.bool), ~duplicate_b], dim=-1)

    if temperature == 1.0:
        log_z_a = torch.logsumexp(logits_a.float(), dim=-1, keepdim=True)
        log_z_b = torch.logsumexp(logits_b.float(), dim=-1, keepdim=True)
        support_logits_a = logits_a.gather(-1, support_idx).float()
        support_logits_b = logits_b.gather(-1, support_idx).float()
    else:
        log_z_a = torch.logsumexp(logits_a.float() / temperature, dim=-1, keepdim=True)
        log_z_b = torch.logsumexp(logits_b.float() / temperature, dim=-1, keepdim=True)
        support_logits_a = logits_a.gather(-1, support_idx).float() / temperature
        support_logits_b = logits_b.gather(-1, support_idx).float() / temperature

    p = torch.exp(support_logits_a - log_z_a) * support_valid
    q = torch.exp(support_logits_b - log_z_b) * support_valid
    p_tail = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)
    q_tail = (1.0 - q.sum(dim=-1, keepdim=True)).clamp(min=0.0, max=1.0)
    p = torch.cat([p, p_tail], dim=-1)
    q = torch.cat([q, q_tail], dim=-1)

    eps = 1e-12
    m = 0.5 * (p + q)
    kl_pm = torch.where(p > 0, p * (p.clamp(min=eps).log() - m.clamp(min=eps).log()), 0.0).sum(dim=-1)
    kl_qm = torch.where(q > 0, q * (q.clamp(min=eps).log() - m.clamp(min=eps).log()), 0.0).sum(dim=-1)
    token_jsd = (0.5 * (kl_pm + kl_qm)).clamp(min=0.0, max=math.log(2.0))

    token_count = response_mask.sum(dim=-1)
    action_jsd = (token_jsd * response_mask).sum(dim=-1) / token_count.clamp(min=1)
    action_jsd = torch.where(token_count > 0, action_jsd, torch.zeros_like(action_jsd))
    return action_jsd, token_jsd


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _compute_per_seq_quantile(ratio: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    """Compute quantile of ratio values per sequence.

    Args:
        ratio: shape (batch_size, seq_len)
        mask: shape (batch_size, seq_len)
        quantile: float in [0, 1], e.g. 0.9 for P90

    Returns:
        Tensor of shape (batch_size,) with one quantile value per sequence.
    """
    batch_size = ratio.shape[0]
    quantiles = []
    for i in range(batch_size):
        valid = ratio[i][mask[i].bool()]
        if len(valid) == 0:
            quantiles.append(torch.tensor(1.0, device=ratio.device, dtype=ratio.dtype))
        else:
            quantiles.append(torch.quantile(valid.float(), quantile).to(ratio.dtype))
    return torch.stack(quantiles)



def compute_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    response_mask: torch.Tensor,
    top_k: int = 64,
    temperature: float = 1.0,
    loss_agg_mode: str = "token-mean",
    action_ig_enabled: bool = False,
    action_ig_clip: float = 1.2,
    action_ig_beta: float = 0.2,
):
    """
    Compute HDPO-style top-k JSD loss between teacher and student distributions.

    Teacher logits should be detached (no gradient). Student logits carry gradient.
    Uses top-k truncation with exact tail correction for efficiency.

    Args:
        teacher_logits: (bs, response_length, vocab_size) — detached, no gradient
        student_logits: (bs, response_length, vocab_size) — with gradient
        response_mask: (bs, response_length) — valid token mask
        top_k: number of teacher top tokens for truncated JSD (default 64)
        temperature: softmax temperature (default 1.0)
        loss_agg_mode: aggregation mode for agg_loss
        action_ig_enabled: aggregate JSD per environment action (one batch
            row), then apply a bounded detached information-gain weight
        action_ig_clip: safety upper bound for the action weight
        action_ig_beta: maximum deviation around weight 1.0 (default 0.2)

    Returns:
        jsd_loss: scalar loss value
        jsd_metrics: dict with diagnostic metrics
    """
    # Teacher: softmax → top-k truncation → renormalize
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  # (bs, resp_len, V)
    topk_teacher_probs, topk_indices = teacher_probs.topk(top_k, dim=-1)  # (bs, resp_len, k)
    # Renormalize teacher top-k to sum to 1
    P = topk_teacher_probs / topk_teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Student: gather logits at top-k positions → softmax over k positions
    student_logits_topk = student_logits.gather(-1, topk_indices)  # (bs, resp_len, k)
    Q = F.softmax(student_logits_topk / temperature, dim=-1)  # (bs, resp_len, k)

    # Student rest mass for tail correction
    student_probs = F.softmax(student_logits / temperature, dim=-1)  # (bs, resp_len, V)
    student_topk_mass = student_probs.gather(-1, topk_indices).sum(dim=-1)  # (bs, resp_len)
    P_rest = (1.0 - student_topk_mass).clamp(min=0.0)  # (bs, resp_len)

    # JSD over top-k support
    M = 0.5 * (P + Q)  # (bs, resp_len, k)
    # Clamp to avoid log(0)
    eps = 1e-8
    M = M.clamp(min=eps)
    P_safe = P.clamp(min=eps)
    Q_safe = Q.clamp(min=eps)

    kl_pm = (P_safe * (P_safe / M).log()).sum(dim=-1)  # (bs, resp_len)
    kl_qm = (Q_safe * (Q_safe / M).log()).sum(dim=-1)  # (bs, resp_len)
    jsd_topk = 0.5 * kl_pm + 0.5 * kl_qm  # (bs, resp_len)

    # Tail correction: 0.5 * P_rest * ln(2)
    tail_correction = 0.5 * P_rest * math.log(2)
    jsd_per_token = jsd_topk + tail_correction  # (bs, resp_len)

    # Each row is one environment action in the multi-turn collector. First
    # aggregate token JSD into one loss per action so long actions do not get
    # more weight merely because they contain more tokens. Then use a bounded,
    # mean-one detached weight to make information gain a mild router rather
    # than multiplying JSD by a normalized copy of itself (which behaves like
    # a squared loss on high-disagreement actions).
    token_count = response_mask.sum(dim=-1)
    valid_actions = token_count > 0
    action_loss = (jsd_per_token * response_mask).sum(dim=-1) / token_count.clamp(min=1)
    action_weights = torch.ones(
        jsd_per_token.shape[0], device=jsd_per_token.device, dtype=jsd_per_token.dtype)
    if action_ig_enabled and valid_actions.any():
        with torch.no_grad():
            valid_gain = action_loss.detach()[valid_actions]
            centered_score = torch.zeros_like(valid_gain)
            if valid_gain.numel() > 1:
                gain_std = valid_gain.std(unbiased=False)
                if gain_std > 1e-8:
                    z_score = (valid_gain - valid_gain.mean()) / gain_std.clamp(min=1e-8)
                    centered_score = torch.tanh(z_score)
                    centered_score = centered_score - centered_score.mean()
                    max_abs = centered_score.abs().max()
                    if max_abs > 1e-8:
                        centered_score = centered_score / max_abs

            beta = max(0.0, min(float(action_ig_beta), 0.95))
            if action_ig_clip is not None:
                beta = min(beta, max(0.0, float(action_ig_clip) - 1.0))
            action_weights[valid_actions] = 1.0 + beta * centered_score
            action_weights[~valid_actions] = 0.0

        # The valid action weights have mean exactly one and lie in
        # [1-beta, 1+beta], so the JSD loss scale remains comparable to the
        # original full-trajectory objective.
        jsd_loss = (action_loss[valid_actions] * action_weights[valid_actions]).mean()
    else:
        jsd_loss = agg_loss(
            loss_mat=jsd_per_token,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
        )

    # Diagnostic metrics
    valid_jsd = jsd_per_token[response_mask.bool()]
    jsd_metrics = {
        "jsd/mean_per_token": valid_jsd.mean().item() if valid_jsd.numel() > 0 else 0.0,
        "jsd/max_per_token": valid_jsd.max().item() if valid_jsd.numel() > 0 else 0.0,
        "jsd/tail_mass_mean": P_rest[response_mask.bool()].mean().item() if valid_jsd.numel() > 0 else 0.0,
        "jsd/token_count": int(response_mask.sum().item()),
        "jsd/action_ig_mean": action_weights[action_weights > 0].mean().item()
        if (action_weights > 0).any() else 0.0,
        "jsd/action_ig_max": action_weights.max().item() if action_weights.numel() > 0 else 0.0,
        "jsd/action_loss_mean": action_loss[valid_actions].mean().item()
        if valid_actions.any() else 0.0,
        "jsd/action_weight_min": action_weights[valid_actions].min().item()
        if valid_actions.any() else 0.0,
    }

    return jsd_loss, jsd_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data

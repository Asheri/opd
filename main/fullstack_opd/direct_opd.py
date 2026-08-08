"""★ Direct-OPD：迁移对象 = RL 诱导的策略偏移 Δ_T，作为密集隐式奖励。

消除「迁移终态」限制：不直接蒸馏 post-RL teacher 的*最终策略*（那会混入小模型
自身的局限），而是蒸馏 teacher 的 *RL 诱导策略偏移*：
    Δ_T(x, a) = log π_T^RL(a|x) − log π_T^ref(a|x)

真实代码：
- Direct-OPD/verl/verl/workers/actor/dp_actor.py::_compute_delta_opd_rm_scores
      delta = teacher_rl_logp - teacher_ref_logp   (在 student top-k 上)
      weights = softmax(student_topk_logp)
      rm_scores = (weights * delta).sum(-1)
- Direct-OPD/verl/verl/trainer/ppo/core_algos.py::compute_token_reward_direct_advantage
      advantages = token_level_rewards * response_mask   (token_reward_direct)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_delta_transfer(teacher_rl_dists: torch.Tensor,
                           teacher_ref_dists: torch.Tensor) -> torch.Tensor:
    """Δ_T = logπ_T^RL − logπ_T^ref，形状 (T, V)。"""
    return teacher_rl_dists - teacher_ref_dists


def delta_opd_reward_topk(student_topk_logp: torch.Tensor,
                          teacher_rl_topk: torch.Tensor,
                          teacher_ref_topk: torch.Tensor,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """与 dp_actor::_compute_delta_opd_rm_scores 完全一致的 top-k 加权形式。

    student_topk_logp: (T, K)  student 在其 top-k action 上的 logp
    teacher_rl_topk / teacher_ref_topk: (T, K)  teacher 在同样 top-k 上的 logp
    返回 (T,) 的 token 级密集奖励。
    """
    delta = teacher_rl_topk - teacher_ref_topk            # (T, K)
    weights = F.softmax(student_topk_logp, dim=-1)        # (T, K)
    rm = (weights * delta).sum(-1)                        # (T,)
    if mask is not None:
        rm = rm * mask
    return rm


def delta_opd_reward_expected(student_dists: torch.Tensor,
                              teacher_rl_dists: torch.Tensor,
                              teacher_ref_dists: torch.Tensor,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    """全词表期望形式（demo 用，因缓存了完整 (T,V) 分布）。

    rm[t] = E_{π_student(·|s_t)}[ logπ_rl − logπ_ref ]_t
          = Σ_v π_student(v|s_t) · Δ_T(v|s_t)
    这正是把 teacher 的 RL 策略偏移按 student 当前分布加权平均，施加于 student 自身
    的 on-policy 状态 —— 即 Direct-OPD 的「密集隐式奖励」。
    """
    delta = teacher_rl_dists - teacher_ref_dists          # (T, V)
    probs = student_dists.exp()                           # (T, V)
    rm = (probs * delta).sum(-1)                          # (T,)
    if mask is not None:
        rm = rm * mask
    return rm


def build_advantage(reward: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """Direct-OPD：token 级奖励直接作为 advantage（ADV_ESTIMATOR=token_reward_direct）。"""
    adv = reward.clone()
    if mask is not None:
        adv = adv * mask
    return adv

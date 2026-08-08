"""v2 损失层：批量版（算法内核与 v1 审阅修复后完全一致，仅加 batch 维度）。

正确性依据（v1 审阅结论，勿回退）：
- PG 必须按行为策略 π_old 加权：−Σ_v π_old(v)·min(ratio(v)·Δ(v), clip(ratio(v))·Δ(v))，
  ratio=1 时精确等于 Direct-OPD 目标 −E_{π_cur}[Δ_T]，一阶梯度非零。
  （等权 mean 形式目标错误；token 级标量 adv 形式一阶梯度恒为 0 —— 均已实测验证。）
- KL 正则用 k3 逐点估计量在 π_student 下取期望，分布形式下恒等真 KL(π||π_ref)。
"""

from __future__ import annotations

import torch


def pg_loss(s_cur: torch.Tensor, s_old: torch.Tensor, delta: torch.Tensor,
            mask: torch.Tensor | None = None, clip_eps: float = 0.2) -> torch.Tensor:
    """Direct-OPD 的 PG 损失 + AsyncOPD 陈旧截断（批量版）。

    s_cur / s_old: (B, T, V) log-softmax（cur 带梯度，old 为 rollout 时刻快照）
    delta:         (B, T, V) Δ_T = logπ_rl − logπ_ref（离线缓存，常量）
    """
    ratio = (s_cur - s_old).exp()                              # (B, T, V)
    unclipped = ratio * delta
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * delta
    pointwise = torch.min(unclipped, clipped)                  # 悲观下界
    pg = -(s_old.exp() * pointwise).sum(-1)                    # E_{π_old}[·], (B, T)
    if mask is not None:
        return (pg * mask).sum() / (mask.sum() + 1e-8)
    return pg.mean()


def low_var_kl(s: torch.Tensor, ref: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    """KL(π_student || π_ref) 的低方差估计（k3 在 π_student 下取期望）。

    s / ref: (B, T, V) log-softmax。
    """
    log_r = ref - s
    k3 = log_r.exp() - log_r - 1.0
    kl = (s.exp() * k3).sum(-1)                                # (B, T)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def low_var_kl_support(s_topk_logp: torch.Tensor, ref_logp_at_support: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
    """稀疏 KL 锚点版本：仅在 student 的 top-K 支撑上做 k3 期望（与 low_var_kl 同内核）。

    真实词表下 ref（student 初始分布）不便存稠密 (N,T,V)，只存其 top-K；
    训练时按当前 student 的 top-K 支撑取回 ref logp（ref_logp_at_support）。
    ⚠️ 精度说明：这【不是】对真 KL 的恒等式——真 KL=Σ_v π_s·k3(v) 需对全词表求和，
    本函数只对 top-K 支撑求和，省略了支撑外的非负尾部（k3≥0），故系统性【略低估】
    真 KL（正则偏弱、方向安全）；对支撑内但不在 ref top-K 的 token 已填 ref_tail_logp
    极负值，给出强漂移惩罚。这是稀疏锚点的有界近似，不是恒等替换。

    s_topk_logp:        (B, T, Ks) 当前 student 在自身 top-K 上的 logp
    ref_logp_at_support:(B, T, Ks) 初始 student 在同一支撑上的 logp（支撑外已填极负值）
    """
    x = ref_logp_at_support - s_topk_logp
    k3 = x.exp() - x - 1.0
    kl = (s_topk_logp.exp() * k3).sum(-1)                      # (B, T)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def expected_reward(dists: torch.Tensor, delta: torch.Tensor,
                    mask: torch.Tensor | None = None) -> torch.Tensor:
    """E_{π_dists}[Δ_T]：(B,T,V),(B,T,V) -> (B,T)。监控用（不反传）。"""
    rm = (dists.exp() * delta).sum(-1)
    if mask is not None:
        rm = rm * mask
    return rm

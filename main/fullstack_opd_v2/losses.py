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
            mask: torch.Tensor | None = None, clip_eps: float = 0.2,
            p_old: torch.Tensor | None = None,
            log_ratio_max: float | None = None,
            renormalize_support: bool = False) -> torch.Tensor:
    """Direct-OPD 的 PG 损失 + AsyncOPD 陈旧截断（批量版）。

    s_cur / s_old: (B, T, V) log-softmax（cur 带梯度，old 为 rollout 时刻快照）
    delta:         (B, T, V) Δ_T = logπ_rl − logπ_ref（离线缓存，常量）
    p_old:         (B, T, V) π_old = s_old.exp() 的预计算版（调用方可缓存省一次 exp）。
                   None 时内部用 s_old.exp() 现算，逐位等价。
    log_ratio_max: 可选纵深防御——对 s_old 低于 -log_ratio_max 的位置（支撑外 log0 近似，
                   如 rollout_vllm._LOG_ZERO=-30，π_old≈0）做【失配屏蔽】：该处贡献强制为 0。
                   否则支撑失配且 delta≠0 时，ratio=exp(s_cur-s_old) 放大到天文数字、
                   与 p_old=exp(-30) 抵消后残留符号相关伪梯度（负 delta 有值、正 delta 为 0）。
                   默认 None 走原路径，正常输入下逐位不变（正常 s_old 最小值 ≈ -ln V > -log_ratio_max）。
    renormalize_support: 稀疏支撑重归一化（对齐原始 Direct-OPD）。稀疏缓存下 delta 只在
                   student top-K 支撑上有值（其余=0），此时把 π_old 在该支撑上重归一
                   （除以其支撑内概率和 Z），使 pg = −E_{π_old^renorm}[min(ratio·Δ, clip·Δ)]
                   成为**条件期望**——否则 top-K 尾部质量被直接丢弃、E_{π_old}[·] 系统性
                   低估 Z 倍（原始 Direct-OPD 用 softmax(student_topk_logp) 做同一件事）。
                   支撑判定 = `delta != 0`（稀疏展开的支撑外恒为 0；支撑内 delta 恰为 0 的
                   几率在连续 logp 下可忽略）。dense 模式（demo 默认）无 top-K，不应开。
    """
    logr = s_cur - s_old
    ratio = logr.exp()                                         # (B, T, V)
    unclipped = ratio * delta
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * delta
    pointwise = torch.min(unclipped, clipped)                  # 悲观下界
    if log_ratio_max is not None:
        # 失配屏蔽：π_old≈0 处（s_old 是 log0 近似）贡献应为 0，避免 ratio 放大→伪梯度/NaN
        pointwise = pointwise.masked_fill(s_old < -log_ratio_max, 0.0)
    if p_old is not None:
        p = p_old
    else:
        p = s_old.exp()                                        # π_old, (B, T, V)
    if renormalize_support:
        # 稀疏支撑重归一：π_old 限定在 delta≠0 支撑上并除以支撑内概率和（条件期望）。
        # 支撑判定用 delta（常量、无梯度），p 保持 detach 语义（p_old 本就 detach）。
        support = (delta != 0)
        p = p * support
        z = p.sum(-1, keepdim=True) + 1e-8
        p = p / z
    pg = -(p * pointwise).sum(-1)                              # E_{π_old}[·], (B, T)
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
                       mask: torch.Tensor | None = None,
                       renormalize_support: bool = False) -> torch.Tensor:
    """稀疏 KL 锚点版本：仅在 student 的 top-K 支撑上做 k3 期望（与 low_var_kl 同内核）。

    真实词表下 ref（student 初始分布）不便存稠密 (N,T,V)，只存其 top-K；
    训练时按当前 student 的 top-K 支撑取回 ref logp（ref_logp_at_support）。
    ⚠️ 精度说明：这【不是】对真 KL 的恒等式——真 KL=Σ_v π_s·k3(v) 需对全词表求和，
    本函数只对 top-K 支撑求和，省略了支撑外的非负尾部（k3≥0），故系统性【略低估】
    真 KL（正则偏弱、方向安全）；对支撑内但不在 ref top-K 的 token 已填 ref_tail_logp
    极负值，给出强漂移惩罚。这是稀疏锚点的有界近似，不是恒等替换。

    renormalize_support: 把 π_cur 在 top-K 支撑上重归一（除以支撑内概率和），得到
    支撑上的【条件 KL 期望】——与 pg_loss 的 renormalize_support 同源（对齐原始
    Direct-OPD 对支撑做 softmax 归一）。否则 Σ_{top-K} π_cur·k3 因尾部质量缺失而
    低估 Z 倍。⚠️ pg_loss 与 low_var_kl_support 的归一开关应【同步】开/关，否则
    λ_kl 的相对权衡会漂。

    s_topk_logp:        (B, T, Ks) 当前 student 在自身 top-K 上的 logp
    ref_logp_at_support:(B, T, Ks) 初始 student 在同一支撑上的 logp（支撑外已填极负值）
    """
    x = ref_logp_at_support - s_topk_logp
    k3 = x.exp() - x - 1.0
    p = s_topk_logp.exp()                                      # π_cur 在 top-K 上, (B,T,Ks)
    if renormalize_support:
        p = p / (p.sum(-1, keepdim=True) + 1e-8)               # 条件期望（支撑内归一）
    kl = (p * k3).sum(-1)                                      # (B, T)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def expected_reward(dists: torch.Tensor, delta: torch.Tensor,
                    mask: torch.Tensor | None = None,
                    p_dists: torch.Tensor | None = None) -> torch.Tensor:
    """E_{π_dists}[Δ_T]：(B,T,V),(B,T,V) -> (B,T)。监控用（不反传）。

    p_dists: 可选，= dists.exp() 的预计算值；调用方已缓存时复用，省一次 (B,T,V) exp。
            默认 None 时内部自算 dists.exp()（语义与原版一致）。
    """
    rm = (p_dists if p_dists is not None else dists.exp()) * delta
    rm = rm.sum(-1)
    if mask is not None:
        rm = rm * mask
    return rm

"""losses.py 内核单测：π_old 加权 PG、k3 KL 恒等式、clip 行为。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fullstack_opd_v2.losses import (
    pg_loss,
    low_var_kl,
    low_var_kl_support,
    expected_reward,
)


def _logp(B, T, V, seed):
    g = torch.Generator().manual_seed(seed)
    return F.log_softmax(torch.randn(B, T, V, generator=g), dim=-1)


def test_pg_loss_onpolicy_equals_neg_expected_delta():
    """ratio=1（s_cur==s_old）时 pg_loss = -E_{π_old}[Δ_T]（Direct-OPD 目标）。"""
    s = _logp(4, 6, 32, seed=0)
    delta = torch.randn(4, 6, 32, generator=torch.Generator().manual_seed(1))
    loss = pg_loss(s, s, delta)
    expected = -(s.exp() * delta).sum(-1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_pg_loss_has_nonzero_gradient():
    """一阶梯度必须非零（回归：v1 曾把 PG 写成 token 级标量 adv，梯度恒为 0）。"""
    s_old = _logp(2, 4, 16, seed=2)
    logits = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(3),
                         requires_grad=True)
    s_cur = F.log_softmax(logits, dim=-1)
    delta = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(4))
    loss = pg_loss(s_cur, s_old, delta)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_pg_loss_clipping_bounds_ratio():
    """clip：ratio 超出 [1-eps,1+eps] 时应被裁剪（悲观下界）。

    构造：s_old 均匀、s_cur 集中到 token0（其 ratio≫1+eps）、delta 仅在 token0=+1。
    此时仅 token0 有贡献，且被 clip 到 1+eps。"""
    import math
    V = 8
    eps = 0.2
    s_old = torch.full((1, 1, V), -math.log(V))            # 均匀分布
    logits = torch.zeros(1, 1, V)
    logits[0, 0, 0] = 10.0
    s_cur = F.log_softmax(logits, dim=-1)                   # 集中 token0 → ratio_0≈8≫1.2
    delta = torch.zeros(1, 1, V)
    delta[0, 0, 0] = 1.0                                    # 仅 token0 有正 advantage
    loss = pg_loss(s_cur, s_old, delta, clip_eps=eps)
    # token0: min(ratio,1+eps)*1 = 1+eps；其余 delta=0。s_old.exp()[0]=1/V
    expected = torch.tensor(-(1.0 / V) * (1 + eps))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_low_var_kl_equals_true_kl():
    """k3 在 π_s 下取期望恒等真 KL(π_s||π_ref)（稠密、全词表求和时）。"""
    s = _logp(3, 5, 24, seed=6)
    ref = _logp(3, 5, 24, seed=7)
    est = low_var_kl(s, ref)
    p_s, p_ref = s.exp(), ref.exp()
    true_kl = (p_s * (s - ref)).sum(-1).mean()
    assert torch.allclose(est, true_kl, atol=1e-5)


def test_low_var_kl_zero_when_identical():
    s = _logp(2, 4, 16, seed=8)
    assert torch.allclose(low_var_kl(s, s), torch.tensor(0.0), atol=1e-7)


def test_low_var_kl_support_leq_true_kl():
    """稀疏版只对 top-K 支撑求和 → 系统性【略低估】真 KL（有界近似，非恒等）。"""
    V, K = 64, 8
    s = _logp(2, 4, V, seed=9)
    ref = _logp(2, 4, V, seed=10)
    s_topk = s.topk(K, dim=-1).values            # (B,T,K)
    ref_at = ref.gather(-1, s.topk(K, dim=-1).indices)  # ref 在同一支撑上的 logp
    sparse = low_var_kl_support(s_topk, ref_at)
    p_s = s.exp()
    true_kl = (p_s * (s - ref)).sum(-1).mean()
    assert sparse <= true_kl + 1e-5
    assert sparse >= 0  # k3 ≥ 0，支撑内仍非负


def test_pg_loss_renormalize_support_conditional_expectation():
    """稀疏支撑重归一化（对齐原始 Direct-OPD）：π_old 在 Δ≠0 支撑上重归一 → 条件期望。

    非归一版 = Σ_{v∈支撑} π_old(v)·Δ(v)（尾部质量被丢，低估 Z 倍）；
    归一版   = Σ_{v∈支撑} (π_old(v)/Z)·Δ(v) = E_{π_old^renorm}[Δ]，ratio=1 时逐位 =
    非归一版 / 该位置支撑内 π_old 和 Z_t。
    """
    B, T, V = 3, 5, 32
    s = _logp(B, T, V, seed=20)
    delta = torch.zeros(B, T, V)
    # 稀疏支撑：每个 (b,t) 随机 8 个位置有 Δ（模拟 student top-K 支撑）
    g = torch.Generator().manual_seed(21)
    ids = torch.randperm(V, generator=g)[None, None, :8].expand(B, T, 8)
    delta.scatter_(-1, ids, torch.randn(B, T, 8, generator=g) * 0.5)

    loss_plain = pg_loss(s, s, delta)                              # 非归一（ratio=1）
    loss_renorm = pg_loss(s, s, delta, renormalize_support=True)   # 归一
    # 手动条件期望：π_old 限定在支撑上再归一
    p = s.exp()
    support = (delta != 0)
    p_sup = p * support
    z = p_sup.sum(-1, keepdim=True) + 1e-8
    manual = -((p_sup / z) * delta).sum(-1).mean()                 # ratio=1 → pointwise=Δ
    assert torch.allclose(loss_renorm, manual, atol=1e-6)
    # 归一版 ≠ 非归一版（支撑内 π_old 和 < 1 时归一放大）
    assert not torch.allclose(loss_renorm, loss_plain, atol=1e-6)
    # 逐位置等价：renorm(b,t) = plain(b,t) / Z_t（Z_t<1 → 归一版每位置贡献更大）
    plain_bt = -(p_sup * delta).sum(-1)
    renorm_bt = -((p_sup / z) * delta).sum(-1)
    assert torch.allclose(renorm_bt, plain_bt / z.squeeze(-1), atol=1e-6)


def test_pg_loss_renormalize_dense_noop():
    """dense delta（无 0）时 renormalize 应为恒等（支撑=全词表，Z=1）——文档说不应开，但语义上不破坏。"""
    s = _logp(2, 4, 24, seed=22)
    delta = torch.randn(2, 4, 24, generator=torch.Generator().manual_seed(23))
    plain = pg_loss(s, s, delta)
    renorm = pg_loss(s, s, delta, renormalize_support=True)
    assert torch.allclose(plain, renorm, atol=1e-6)


def test_pg_loss_renormalize_over_explicit_support():
    """P2-G（二次审查）：显式 support 掩码（完整 student top-K）覆盖 delta!=0（交集）。

    delta 只在 student∩teacher 交集有值；显式 support = 更宽的 student top-K 时，
    归一化分母用显式支撑（与 low_var_kl_support 同源），未覆盖 token Δ=0 贡献 0、
    但计入分母。验证 support 参数优先于 delta!=0。
    """
    B, T, V, Ks = 3, 5, 32, 8
    s = _logp(B, T, V, seed=30)
    delta = torch.zeros(B, T, V)
    # 交集：仅少量 token 有 Δ
    g = torch.Generator().manual_seed(31)
    inter = torch.randperm(V, generator=g)[None, None, :3].expand(B, T, 3)
    delta.scatter_(-1, inter, torch.randn(B, T, 3, generator=g) * 0.5)
    # 显式支撑 = 更宽的 student top-K
    topk_ids = s.topk(Ks, dim=-1).indices          # (B,T,Ks) ⊇ inter
    support = torch.zeros(B, T, V, dtype=torch.bool)
    support.scatter_(-1, topk_ids, True)

    p = s.exp()
    z_exp = (p * support).sum(-1, keepdim=True) + 1e-8
    manual = -((p * support / z_exp) * delta).sum(-1).mean()
    loss = pg_loss(s, s, delta, renormalize_support=True, support=support)
    assert torch.allclose(loss, manual, atol=1e-6)
    # 与交集（delta!=0）归一结果不同（分母更宽 → 每 token 权重更小 → 损失更小）
    loss_inter = pg_loss(s, s, delta, renormalize_support=True)
    assert not torch.allclose(loss, loss_inter, atol=1e-6)


def test_pg_loss_delta_clip_bounds():
    """部署实测 P1：Δ_T 数值护栏——clamp 到 ±delta_clip 使大 Δ_T 下 loss 有界（防梯度爆炸）。"""
    B, T, V = 2, 4, 16
    s = _logp(B, T, V, seed=40)
    delta = torch.zeros(B, T, V)
    delta[0, 0, 0] = 10.0        # 极端 Δ_T（真实教师对可达 ±10）
    delta[0, 0, 1] = -9.0
    unclipped = pg_loss(s, s, delta)                      # 无护栏：loss 无界大
    clipped = pg_loss(s, s, delta, delta_clip=2.0)        # 护栏：clamp 到 ±2
    assert clipped < unclipped
    assert abs(clipped) <= 2.0 + 1e-6                     # loss 有界
    # ratio=1 时 clipped = -E_{π_old}[clamp(Δ)]，逐位置验证
    expected = -((s.exp() * torch.clamp(delta, -2.0, 2.0)).sum(-1)).mean()
    assert torch.allclose(clipped, expected, atol=1e-6)


def test_low_var_kl_support_renormalized():
    """稀疏 KL 归一版：π_cur 在 top-K 上重归一 → 条件 KL 期望，≥ 非归一版（k3≥0、Z<1）。"""
    V, K = 64, 8
    s = _logp(2, 4, V, seed=24)
    ref = _logp(2, 4, V, seed=25)
    s_topk = s.topk(K, dim=-1).values
    ref_at = ref.gather(-1, s.topk(K, dim=-1).indices)

    plain = low_var_kl_support(s_topk, ref_at)
    renorm = low_var_kl_support(s_topk, ref_at, renormalize_support=True)

    p = s_topk.exp()
    z = p.sum(-1, keepdim=True) + 1e-8
    k3 = (ref_at - s_topk).exp() - (ref_at - s_topk) - 1.0
    manual = ((p / z) * k3).sum(-1).mean()
    assert torch.allclose(renorm, manual, atol=1e-6)
    # 归一 ≥ 非归一（π_cur/Z ≥ π_cur，k3≥0）
    assert renorm >= plain - 1e-6


def test_expected_reward():
    dists = _logp(2, 4, 16, seed=11)
    delta = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(12))
    rm = expected_reward(dists, delta)
    assert rm.shape == (2, 4)
    assert torch.allclose(rm, (dists.exp() * delta).sum(-1), atol=1e-6)


def test_pg_loss_mask_averaging():
    """mask 应屏蔽 padding 位置（只对 mask=1 取均值）。"""
    s = _logp(2, 4, 16, seed=13)
    delta = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(14))
    mask = torch.ones(2, 4)
    mask[0, 2:] = 0  # 第 0 条只留前 2 位
    loss_masked = pg_loss(s, s, delta, mask=mask)
    pg = -(s.exp() * delta).sum(-1)
    expected = (pg * mask).sum() / mask.sum()
    assert torch.allclose(loss_masked, expected, atol=1e-6)

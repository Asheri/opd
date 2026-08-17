# -*- coding: utf-8 -*-
"""P0：base 池稀疏 PG（_train_step）与旧稠密路径的数值等价性测试。

稠密版：delta 仅在 student top-K 支撑非零 + 在同一支撑掩码上重归一；
稀疏版：直接把支撑值（s_topk.values / s_old_at / delta_at）喂 pg_loss。
两者应在同一支撑、同一 clip/重归一设置下逐位一致。
"""
import torch

from fullstack_opd_v2.losses import pg_loss


def _mk(B=2, T=3, V=16, K=4, seed=0):
    torch.manual_seed(seed)
    s_cur = torch.randn(B, T, V).log_softmax(-1)
    s_old = torch.randn(B, T, V).log_softmax(-1)
    topk = torch.topk(s_cur, K, dim=-1)
    # 教师 Δ_T：只在 student top-K 支撑上有值（模拟 delta_for_student_topk 的展开）
    delta = torch.zeros(B, T, V)
    delta.scatter_(-1, topk.indices, torch.randn(B, T, K))
    return s_cur, s_old, delta, topk


def test_dense_vs_sparse_pg_equivalent_basic():
    s_cur, s_old, delta_d, topk = _mk()
    clip_eps, log_ratio_max, log_ratio_clip, delta_clip = 0.2, 20.0, 10.0, 2.0

    # 稠密路径（_train_step 旧逻辑）
    pg_support = torch.zeros_like(delta_d, dtype=torch.bool)
    pg_support.scatter_(-1, topk.indices, True)
    loss_dense = pg_loss(s_cur, s_old, delta_d, None, clip_eps,
                         p_old=s_old.exp(), log_ratio_max=log_ratio_max,
                         log_ratio_clip=log_ratio_clip,
                         renormalize_support=True, support=pg_support,
                         delta_clip=delta_clip)

    # 稀疏路径（P0：_train_step 新逻辑）
    s_topk_v = topk.values
    s_old_at = s_old.gather(-1, topk.indices)
    delta_at = delta_d.gather(-1, topk.indices)
    sup = torch.ones_like(s_topk_v, dtype=torch.bool)
    loss_sparse = pg_loss(s_topk_v, s_old_at, delta_at, None, clip_eps,
                          p_old=s_old_at.exp(), log_ratio_max=log_ratio_max,
                          log_ratio_clip=log_ratio_clip,
                          renormalize_support=True, support=sup,
                          delta_clip=delta_clip)

    assert torch.allclose(loss_dense, loss_sparse, atol=1e-6), (
        loss_dense.item(), loss_sparse.item())


def test_dense_vs_sparse_pg_equivalent_no_renorm():
    """renormalize 关闭时的等价性（默认配置可能关，需同样成立）。"""
    s_cur, s_old, delta_d, topk = _mk(seed=1)
    clip_eps, log_ratio_max, log_ratio_clip, delta_clip = 0.2, 20.0, None, None

    pg_support = torch.zeros_like(delta_d, dtype=torch.bool)
    pg_support.scatter_(-1, topk.indices, True)
    loss_dense = pg_loss(s_cur, s_old, delta_d, None, clip_eps,
                         p_old=s_old.exp(), log_ratio_max=log_ratio_max,
                         log_ratio_clip=log_ratio_clip,
                         renormalize_support=False, support=pg_support,
                         delta_clip=delta_clip)

    s_old_at = s_old.gather(-1, topk.indices)
    delta_at = delta_d.gather(-1, topk.indices)
    loss_sparse = pg_loss(topk.values, s_old_at, delta_at, None, clip_eps,
                          p_old=s_old_at.exp(), log_ratio_max=log_ratio_max,
                          log_ratio_clip=log_ratio_clip,
                          renormalize_support=False, support=None,
                          delta_clip=delta_clip)

    assert torch.allclose(loss_dense, loss_sparse, atol=1e-6), (
        loss_dense.item(), loss_sparse.item())


def test_sparse_delta_cache_interface():
    """delta_at_student_topk 与 delta_for_student_topk 在支撑上取值一致。"""
    from fullstack_opd_v2.cache import (expand_student_topk_delta,
                                        expand_student_topk_delta_sparse)
    B, T, Kt, Ks = 2, 3, 5, 4
    torch.manual_seed(2)
    ids = torch.randint(0, 16, (B, T, Kt)).sort(-1).values
    delta_k = torch.randn(B, T, Kt)
    s_ids = torch.randint(0, 16, (B, T, Ks))
    dense = expand_student_topk_delta(ids, delta_k, s_ids, vocab=16)
    sparse = expand_student_topk_delta_sparse(ids, delta_k, s_ids)
    got = dense.gather(-1, s_ids)
    assert torch.allclose(got, sparse, atol=1e-7), "cache sparse != dense.gather"

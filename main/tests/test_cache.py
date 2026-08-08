"""cache.py 单测：dense/topk 构建、稀疏展开形状回归、save/load roundtrip、一致性校验。"""
from __future__ import annotations

import torch
import pytest

from fullstack_opd_v2.model import CausalToyLM, response_dists
from fullstack_opd_v2.cache import TensorTeacherCache, TeacherConsistencyError


def _make(N=6, P=4, T=5, V=24, d=16, L=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, V, (N, P), generator=g)
    responses = torch.randint(0, V, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    teacher_ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    return prompts, responses, teacher_rl, teacher_ref


def test_dense_build_delta_shape_and_value():
    prompts, responses, rl, ref = _make()
    cache = TensorTeacherCache(enforce_consistency=True, top_k=0)
    cache.build(prompts, responses, rl, ref, batch_size=4)
    N, T = prompts.size(0), responses.size(1)
    V = rl.vocab
    assert cache.delta.shape == (N, T, V)
    expected = response_dists(rl, prompts, responses) - response_dists(ref, prompts, responses)
    assert torch.allclose(cache.delta, expected, atol=1e-5)


def test_dense_get_delta_indexing():
    prompts, responses, rl, ref = _make()
    cache = TensorTeacherCache(True, 0).build(prompts, responses, rl, ref)
    idxs = torch.tensor([0, 2, 5])
    d = cache.get_delta(idxs)
    assert d.shape == (3, responses.size(1), rl.vocab)
    assert torch.allclose(d, cache.delta[idxs])


def test_topk_build_shapes_and_delta():
    prompts, responses, rl, ref = _make()
    K = 7
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref, batch_size=4)
    N, T = prompts.size(0), responses.size(1)
    assert cache.ids.shape == (N, T, K)
    assert cache.rl_k.shape == (N, T, K)
    assert cache.ref_k.shape == (N, T, K)
    assert torch.allclose(cache.delta_k, cache.rl_k - cache.ref_k, atol=1e-6)


def test_delta_for_student_topk_regression_1d_idxs():
    """回归：旧版 `B,T = idxs.shape` 对一维 (B,) 索引解包会 ValueError。
    idxs 是一维批次索引，B/T 必须从支撑张量取。"""
    prompts, responses, rl, ref = _make()
    K = 7
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    B, T = 3, responses.size(1)
    Ks = 5
    idxs = torch.tensor([0, 1, 2])                       # 一维 (B,)
    student_topk_ids = torch.randint(0, rl.vocab, (B, T, Ks))
    out = cache.delta_for_student_topk(idxs, student_topk_ids)
    assert out.shape == (B, T, rl.vocab)                 # 展开回 dense (B,T,V)


def test_delta_for_student_topk_support_only():
    """支撑外应填 fill(0)；匹配到 teacher top-K 的 token 应取真实 Δ。"""
    prompts, responses, rl, ref = _make(N=4, T=3, V=20)
    K = 6
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    B, T = 2, responses.size(1)
    idxs = torch.tensor([0, 1])
    # 直接用 teacher top-K 的 id 作为 student 支撑 → 全部应匹配上
    teacher_ids = cache.ids[idxs]                        # (B,T,K)
    out = cache.delta_for_student_topk(idxs, teacher_ids)
    # 支撑上取值应等于 delta_k
    gathered = out.gather(-1, teacher_ids)
    assert torch.allclose(gathered, cache.delta_k[idxs], atol=1e-5)
    # 支撑外应为 0（随机挑一个不在 teacher_ids 里的 token）
    in_support = torch.zeros(B, T, rl.vocab, dtype=torch.bool)
    in_support.scatter_(-1, teacher_ids, True)
    assert torch.allclose(out[~in_support], torch.zeros_like(out[~in_support]), atol=1e-7)


def test_save_load_roundtrip_dense(tmp_path):
    prompts, responses, rl, ref = _make()
    cache = TensorTeacherCache(True, 0).build(prompts, responses, rl, ref)
    p = tmp_path / "c.pt"
    cache.save(str(p))
    loaded = TensorTeacherCache.load(str(p))
    assert loaded.mode == "dense"
    assert torch.allclose(loaded.delta, cache.delta, atol=1e-6)
    assert loaded.vocab == cache.vocab


def test_save_load_roundtrip_topk(tmp_path):
    prompts, responses, rl, ref = _make()
    cache = TensorTeacherCache(True, top_k=5).build(prompts, responses, rl, ref)
    p = tmp_path / "c.pt"
    cache.save(str(p))
    loaded = TensorTeacherCache.load(str(p))
    assert loaded.mode == "topk"
    assert torch.equal(loaded.ids, cache.ids)
    assert torch.allclose(loaded.delta_k, cache.delta_k, atol=1e-6)
    # P1-1：sorted 支撑字段须随 roundtrip 持久化且逐位一致
    assert loaded.ids_sorted is not None
    assert loaded.delta_k_sorted is not None
    assert torch.equal(loaded.ids_sorted, cache.ids_sorted)
    assert torch.equal(loaded.delta_k_sorted, cache.delta_k_sorted)


def test_teacher_consistency_raises_on_mismatch():
    prompts, responses, rl, ref = _make(V=24)
    bad_ref = CausalToyLM(vocab=99, d_model=16, n_layers=1)   # 词表不一致
    cache = TensorTeacherCache(enforce_consistency=True, top_k=0)
    with pytest.raises(TeacherConsistencyError):
        cache.build(prompts, responses, rl, bad_ref)

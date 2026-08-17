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


def test_dense_build_per_teacher_prompts():
    """C3（2026-08-18）：prompts_rl/prompts_ref 时两教师各用自己的格式评分。
    delta = response_dists(rl, prl, R) - response_dists(ref, pref, R)。"""
    prompts, responses, rl, ref = _make()
    prl = torch.randint(0, rl.vocab, (responses.size(0), 6))      # 独立教师格式 prompt
    pref = torch.randint(0, rl.vocab, (responses.size(0), 5))
    cache = TensorTeacherCache(True, 0)
    cache.build(prompts, responses, rl, ref, batch_size=4,
                prompts_rl=prl, prompts_ref=pref)
    expected = response_dists(rl, prl, responses) - response_dists(ref, pref, responses)
    assert cache.delta.shape == (prompts.size(0), responses.size(1), rl.vocab)
    assert torch.allclose(cache.delta, expected, atol=1e-5)


def test_build_per_teacher_prompt_row_mismatch_raises():
    """C3：prompts_rl/ref 行数 ≠ responses → 显式报错（防错位）。"""
    prompts, responses, rl, ref = _make()
    prl = torch.randint(0, rl.vocab, (responses.size(0) - 1, 4))
    pref = torch.randint(0, rl.vocab, (responses.size(0), 4))
    with pytest.raises(ValueError):
        TensorTeacherCache(True, 0).build(prompts, responses, rl, ref,
                                          prompts_rl=prl, prompts_ref=pref)


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


def test_save_load_with_fat_context(tmp_path):
    """模块2：save 携带 fat prompts/responses → load 后可直接索引（多学生 load_cache 路径）。"""
    prompts, responses, rl, ref = _make()
    cache = TensorTeacherCache(True, 0).build(prompts, responses, rl, ref)
    p = tmp_path / "c.pt"
    cache.save(str(p), prompts=prompts, responses=responses)
    loaded = TensorTeacherCache.load(str(p))
    assert torch.equal(loaded.prompts, prompts)
    assert torch.equal(loaded.responses, responses)
    # 旧格式（不传 fat）向后兼容：load 后为 None
    p2 = tmp_path / "old.pt"
    TensorTeacherCache(True, 0).build(prompts, responses, rl, ref).save(str(p2))
    loaded2 = TensorTeacherCache.load(str(p2))
    assert loaded2.prompts is None and loaded2.responses is None


def test_teacher_consistency_raises_on_mismatch():
    prompts, responses, rl, ref = _make(V=24)
    bad_ref = CausalToyLM(vocab=99, d_model=16, n_layers=1)   # 词表不一致
    cache = TensorTeacherCache(enforce_consistency=True, top_k=0)
    with pytest.raises(TeacherConsistencyError):
        cache.build(prompts, responses, rl, bad_ref)


# --------------------------- 方案 A：跨词表 top-K 支撑（对齐 Direct-OPD 论文） ---------------------------
def _make_cross_vocab(N=6, P=4, T=5, Vt=24, Vs=32, Kt=8, seed=0):
    """teacher vocab=Vt(24)、student vocab=Vs(32) 的跨词表场景（模拟 7B：152064 vs 151936）。

    student top-K 含 ≥Vt 的 id（如 24..31）——这些在 teacher 词表外，未命中 → Δ=0。
    """
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(0, Vt, (N, P), generator=g)
    responses = torch.randint(0, Vt, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=Vt, d_model=16, n_layers=1)
    teacher_ref = CausalToyLM(vocab=Vt, d_model=16, n_layers=1)
    cache = TensorTeacherCache(True, top_k=Kt).build(prompts, responses, teacher_rl, teacher_ref)
    return cache, prompts, responses, teacher_rl, teacher_ref, Vt, Vs


def test_delta_for_student_topk_cross_vocab_expands():
    """方案 A：vocab_out(32) > teacher vocab(24) → out 维度 (B,T,32)，
    student 超出 teacher 词的 id（24..31）未命中 → Δ=0；命中 id 取 teacher delta。"""
    cache, prompts, responses, rl, ref, Vt, Vs = _make_cross_vocab()
    B, T = 4, responses.size(1)
    Ks = 6
    # 构造 student top-K：含超出 teacher 词表的 id（24..29）与命中 id（0..5）
    g = torch.Generator().manual_seed(1)
    ids_low = torch.randint(0, Vt, (B, T, Ks // 2), generator=g)   # 命中区 [0,24)
    ids_high = torch.randint(Vt, Vs, (B, T, Ks - Ks // 2), generator=g)  # 超出区 [24,32)
    student_topk = torch.cat([ids_low, ids_high], dim=-1)
    out = cache.delta_for_student_topk(torch.arange(B), student_topk,
                                       vocab_out=Vs)
    assert out.shape == (B, T, Vs)                     # 按 student 词表展开
    # 超出区全部为 0（teacher 词表外无 Δ_T）
    assert (out.gather(-1, ids_high) == 0).all()
    # 命中区：若 student id 落在 teacher top-K 内则取 teacher delta，否则 0
    # （不能全断言非零——teacher top-K 只覆盖部分 id；只验证维度与不越界）
    assert torch.isfinite(out).all()


def test_delta_for_student_topk_default_vocab_from_ids():
    """vocab_out=None → 用 max(student_topk_ids)+1 推断（7B 默认路径）。"""
    cache, prompts, responses, rl, ref, Vt, Vs = _make_cross_vocab()
    B, T = 4, responses.size(1)
    Ks = 6
    g = torch.Generator().manual_seed(2)
    ids_high = torch.randint(Vt, Vs, (B, T, Ks), generator=g)
    out = cache.delta_for_student_topk(torch.arange(B), ids_high)  # 无 vocab_out
    assert out.shape == (B, T, Vs)                     # 推断自 max id+1
    assert (out == 0).all()                            # 全超出 → 全 0


def test_delta_for_student_topk_vocab_out_smaller_raises():
    """vocab_out < teacher vocab → 显式报错（展开维度不能小于缓存词表）。"""
    cache, prompts, responses, rl, ref, Vt, Vs = _make_cross_vocab()
    B, T = 2, responses.size(1)
    student_topk = torch.randint(0, Vt, (B, T, 4))
    with pytest.raises(ValueError):
        cache.delta_for_student_topk(torch.arange(B), student_topk, vocab_out=Vt - 1)


# ---- S1-3：expand_student_topk_delta 纯函数（in-memory / 磁盘共用；DiskTeacherCache 会复用）----

def test_expand_pure_matches_inmemory_bitwise():
    """纯函数 expand_student_topk_delta 与 TensorTeacherCache.delta_for_student_topk
    在【同输入同输出】下逐位一致——保证磁盘路径复用后训练数字不变。"""
    from fullstack_opd_v2.cache import expand_student_topk_delta
    prompts, responses, rl, ref = _make(N=6, T=5, V=24)
    K = 7
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    B, T = 3, responses.size(1)
    Ks = 5
    idxs = torch.tensor([0, 1, 2])
    student_topk = torch.randint(0, rl.vocab, (B, T, Ks))
    # in-memory 路径（内部取 sorted 切片 → 调纯函数）
    ref_out = cache.delta_for_student_topk(idxs, student_topk)
    # 磁盘路径等价：直接取 sorted 切片喂纯函数
    pure_out = expand_student_topk_delta(cache.ids_sorted[idxs], cache.delta_k_sorted[idxs],
                                         student_topk, cache.vocab)
    assert pure_out.shape == ref_out.shape
    assert torch.equal(pure_out, ref_out)


def test_expand_pure_cross_vocab_expands():
    """纯函数跨词表：student vocab > teacher vocab 时扩展展开维度。"""
    from fullstack_opd_v2.cache import expand_student_topk_delta
    cache, prompts, responses, rl, ref, Vt, Vs = _make_cross_vocab()
    B, T = 2, responses.size(1)
    student_topk = torch.randint(Vt, Vs, (B, T, 4))          # 全在 [Vt, Vs) 高 id 区
    out = expand_student_topk_delta(cache.ids_sorted[torch.arange(B)],
                                    cache.delta_k_sorted[torch.arange(B)],
                                    student_topk, cache.vocab)
    assert out.shape[2] == Vs                               # 扩展到 student 词表
    # 高 id 区不在 teacher 支撑 → 全部 fill(0)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_expand_pure_respects_mask():
    """S1-5 变长：mask 置 0 的 (B,T) 位置 Δ 全为 0（padding 不参与统计）。"""
    from fullstack_opd_v2.cache import expand_student_topk_delta
    prompts, responses, rl, ref = _make(N=4, T=5, V=24)
    K = 7
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    B, T = 2, responses.size(1)
    idxs = torch.tensor([0, 1])
    teacher_ids = cache.ids[idxs]                            # 全匹配 teacher 支撑
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 0] = True                                        # 只留第 0 个位置有效
    out = expand_student_topk_delta(cache.ids_sorted[idxs], cache.delta_k_sorted[idxs],
                                    teacher_ids, cache.vocab, mask=mask)
    # 无 mask 基线（对照）：有效位置上 Δ 与 unmasked 一致；无效位置全 0
    base = expand_student_topk_delta(cache.ids_sorted[idxs], cache.delta_k_sorted[idxs],
                                     teacher_ids, cache.vocab)
    assert torch.allclose(out[:, 0:1], base[:, 0:1], atol=1e-7)   # 有效位置不变
    assert torch.allclose(out[:, 1:], torch.zeros_like(out[:, 1:]), atol=1e-7)  # padding 排除
    assert base[:, 0].abs().sum() > 0                        # 有效位置确有 teacher Δ（非恒 0）

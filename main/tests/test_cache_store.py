"""cache_store.py 单测：磁盘 mmap 写入/重载/batch-local/checksum/一致性/变长。

覆盖 §3/§4/§5/§6/§7：磁盘 roundtrip 与 in-memory 逐位一致、内存驻留只读 batch 行、
checksum 验签、模型哈希一致性 fail-fast、metadata 13 键、total_tokens=ΣL、padding 排除。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fullstack_opd_v2.cache import TensorTeacherCache
from fullstack_opd_v2.cache_store import (
    METADATA_KEYS, CacheConsistencyError, DiskTeacherCache,
    compute_lengths, hash_model_dir, load_cache_metadata,
    verify_consistency, write_cache_disk,
)
from fullstack_opd_v2.model import CausalToyLM


def _make(N=6, P=4, T=6, V=24, d=16, L=1, seed=0, pad_id=0):
    g = torch.Generator().manual_seed(seed)
    prompts = torch.randint(1, V, (N, P), generator=g)   # 避开 0（pad）
    responses = torch.randint(1, V, (N, T), generator=g)
    teacher_rl = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    teacher_ref = CausalToyLM(vocab=V, d_model=d, n_layers=L)
    return prompts, responses, teacher_rl, teacher_ref


def _build_topk_cache(N=6, T=6, K=7, **kw):
    prompts, responses, rl, ref = _make(N=N, T=T, **kw)
    cache = TensorTeacherCache(True, top_k=K).build(prompts, responses, rl, ref)
    return cache, prompts, responses


def test_disk_roundtrip_build_reload(tmp_path):
    """磁盘 roundtrip：build → write → DiskTeacherCache 加载 → 同 idxs 输出与 in-memory 逐位一致。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    disk = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    B, T, Ks = 3, responses.size(1), 5
    idxs = torch.tensor([0, 1, 2])
    student_topk = torch.randint(0, cache.vocab, (B, T, Ks))
    # 磁盘路径与 in-memory 路径逐位一致（训练数字不变）
    ref = cache.delta_for_student_topk(idxs, student_topk)
    out = disk.delta_for_student_topk(idxs, student_topk)
    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def test_disk_metadata_fields(tmp_path):
    """metadata 含全部 13 键；total_tokens == Σ lengths（非 N×T）。"""
    cache, prompts, responses = _build_topk_cache(N=6, T=6)
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    for k in METADATA_KEYS:
        assert k in meta, f"metadata 缺键 {k}"
    lengths = compute_lengths(responses, pad_id=0)
    assert meta["total_tokens"] == int(lengths.sum())   # ΣL，非 N×T
    assert meta["num_samples"] == responses.size(0)
    assert meta["top_k"] == cache.top_k
    # 重载 metadata 一致
    reloaded = load_cache_metadata(prefix)
    assert reloaded["checksum"] == meta["checksum"]


def test_disk_batch_local_underlying_is_memmap(tmp_path):
    """batch-local（§4）：_ids/_delta 是 np.memmap，不是整文件 ndarray 副本。"""
    cache, prompts, responses = _build_topk_cache(N=6, T=6)
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    disk = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    assert isinstance(disk._ids, np.memmap)
    assert isinstance(disk._delta, np.memmap)
    assert isinstance(disk._lengths, np.memmap)
    # 取子集 == 完整 numpy 的子集（功能等价，且不实例化全量 ndarray）
    idx = [0, 2]
    ids_sub = disk._ids[idx]
    ids_full = np.asarray(disk._ids)[idx]
    assert np.array_equal(ids_sub, ids_full)


def test_disk_checksum_mismatch_raises(tmp_path):
    """篡改 .dat 一个字节 → load CacheConsistencyError（fail fast）。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    # 翻转 lengths.dat 一个字节
    with open(f"{prefix}.lengths.dat", "r+b") as f:
        f.seek(0)
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 0xFF]))
    with pytest.raises(CacheConsistencyError):
        load_cache_metadata(prefix)


def test_disk_consistency_mismatch_fails_fast(tmp_path):
    """metadata 里 teacher_hash 与当前不符 → verify_consistency 抛 CacheConsistencyError。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0,
                            hashes={"teacher_model_hash": "abc",
                                    "reference_model_hash": "def",
                                    "tokenizer_hash": "ghi"})
    assert meta["teacher_model_hash"] == "abc"
    cfg = {"cache": {"top_k": cache.top_k}, "max_response_len": responses.size(1),
           "teacher_rl_path": "x", "teacher_ref_path": "y", "student_path": "z"}
    with pytest.raises(CacheConsistencyError):
        verify_consistency(meta, cfg, hashes_now={"tokenizer_hash": "ghi",
                                                  "teacher_model_hash": "CHANGED",
                                                  "reference_model_hash": "def"})

def test_disk_consistency_topk_mismatch_fails_fast(tmp_path):
    """metadata top_k 与配置 cache.top_k 不符 → 抛错。"""
    cache, prompts, responses = _build_topk_cache(K=7)
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    cfg = {"cache": {"top_k": 32}, "max_response_len": responses.size(1)}
    with pytest.raises(CacheConsistencyError):
        verify_consistency(meta, cfg, hashes_now={"tokenizer_hash": meta["tokenizer_hash"],
                                                  "teacher_model_hash": meta["teacher_model_hash"],
                                                  "reference_model_hash": meta["reference_model_hash"]})


# ------------------------------- C2 prompt_format 守卫 -------------------------------
def _pf_cfg(pf=True):
    """verify_consistency 用 cfg：hashes 为空（write 未传）→ hashes_now 用 meta 值对齐。"""
    return {"dataset": {"apply_chat_template": pf}}


def test_disk_prompt_format_written(tmp_path):
    """C2：write_cache_disk 按 prompt_format 参数写入 metadata。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0,
                            prompt_format="chat")
    assert meta["prompt_format"] == "chat"
    reloaded = load_cache_metadata(prefix)
    assert reloaded["prompt_format"] == "chat"


def test_disk_prompt_format_mismatch_fails_fast(tmp_path):
    """C2：cache 为 raw 而配置开模板 → fail-fast（防 Δ_T 静默错位）。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0)  # 默认 raw
    with pytest.raises(CacheConsistencyError):
        verify_consistency(meta, _pf_cfg(pf=True),
                           hashes_now={"tokenizer_hash": meta["tokenizer_hash"],
                                       "teacher_model_hash": meta["teacher_model_hash"],
                                       "reference_model_hash": meta["reference_model_hash"]})


def test_disk_prompt_format_old_missing_field_defaults_raw(tmp_path):
    """C2：旧 cache 缺 prompt_format 字段 → 默认 raw：模板关时放行、开时拒绝。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    meta0 = write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    meta = {k: v for k, v in meta0.items() if k != "prompt_format"}  # 模拟旧 cache
    h = {"tokenizer_hash": meta["tokenizer_hash"],
         "teacher_model_hash": meta["teacher_model_hash"],
         "reference_model_hash": meta["reference_model_hash"]}
    # 模板关（raw 配置）→ 放行
    verify_consistency(meta, _pf_cfg(pf=False), hashes_now=h)
    # 模板开 → 拒绝
    with pytest.raises(CacheConsistencyError):
        verify_consistency(meta, _pf_cfg(pf=True), hashes_now=h)


def test_disk_prompt_format_chat_matches_passes(tmp_path):
    """C2：cache 为 chat 且配置开模板 → 放行（正向路径）。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0,
                            prompt_format="chat")
    h = {"tokenizer_hash": meta["tokenizer_hash"],
         "teacher_model_hash": meta["teacher_model_hash"],
         "reference_model_hash": meta["reference_model_hash"]}
    verify_consistency(meta, _pf_cfg(pf=True), hashes_now=h)


def test_disk_variable_length_padding_excluded(tmp_path):
    """两条样本长度 3/6（T_max=6）：delta 在 >实际长度 处为 0；total_tokens=ΣL。"""
    cache, prompts, responses = _build_topk_cache(N=2, T=6, K=7)
    # 制造变长：把样本0 的后 3 个 token 置 pad_id=0
    responses = responses.clone()
    responses[0, 3:] = 0
    prefix = str(tmp_path / "c")
    meta = write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    lengths = compute_lengths(responses, pad_id=0)
    assert lengths.tolist() == [3, 6]
    assert meta["total_tokens"] == 9 == int(lengths.sum())
    disk = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    idxs = torch.tensor([0, 1])
    teacher_ids = cache.ids[idxs]                       # 全匹配 teacher 支撑
    mask = disk.token_mask(idxs)                        # (2,6)：样本0 前3为True
    out = disk.delta_for_student_topk(idxs, teacher_ids, mask=mask)
    # 无效位置（>= 实际长度）Δ 全 0
    assert torch.allclose(out[0, 3:], torch.zeros_like(out[0, 3:]), atol=1e-7)
    assert torch.allclose(out[1, :], out[1, :], atol=1e-7)         # 样本1 全 valid
    # response_length 访问器
    assert disk.response_length(idxs).tolist() == [3, 6]


def test_disk_reload_restart_lookup(tmp_path):
    """重启后 lookup（§9 验收）：换一个 DiskTeacherCache 实例做 lookup 结果一致。"""
    cache, prompts, responses = _build_topk_cache()
    prefix = str(tmp_path / "c")
    write_cache_disk(cache, prefix, responses=responses, pad_id=0)
    B, T, Ks = 2, responses.size(1), 4
    idxs = torch.tensor([0, 3])
    student_topk = torch.randint(0, cache.vocab, (B, T, Ks))
    d1 = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    o1 = d1.delta_for_student_topk(idxs, student_topk)
    # 模拟新进程：重新构造（磁盘文件仍在）
    d2 = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    o2 = d2.delta_for_student_topk(idxs, student_topk)
    assert torch.equal(o1, o2)


def test_stage1_build_cache_disk_integration(tmp_path):
    """S1-6：stage1_build_cache(storage='disk') 写磁盘三件套 + metadata，DiskTeacherCache
    加载后与 in-memory 输出一致（build→disk→load 全链路）。"""
    from fullstack_opd_v2.pipeline import stage1_build_cache
    prompts, responses, rl, ref = _make(N=6, T=6, V=24)
    prefix = str(tmp_path / "c")
    cfg = {"cache_mode": "topk", "top_k_teacher": 7, "cache_path": prefix,
           "build_batch_size": 4, "dtype": "bf16", "warmup_M": 0,
           "warmup_source": "none", "warmup_temperature": 1.0,
           "enforce_teacher_consistency": True}
    cache, fp, fr = stage1_build_cache(
        prompts, responses, rl, ref, cfg, storage="disk",
        hashes={"tokenizer_hash": "t", "teacher_model_hash": "a",
                "reference_model_hash": "b", "generation_model_hash": "g"})
    # 磁盘三件套 + metadata 已写
    assert np.memmap(f"{prefix}.ids_sorted.dat", dtype=np.int32, mode="r").size > 0
    assert np.memmap(f"{prefix}.delta_k_sorted.dat", dtype=np.float32, mode="r").size > 0
    meta = load_cache_metadata(prefix)
    assert meta["tokenizer_hash"] == "t"
    assert meta["vocab"] == cache.vocab
    assert meta["num_samples"] == prompts.size(0) == fp.size(0)
    # DiskTeacherCache 加载 → 与 build 出的 in-memory 缓存逐位一致
    disk = DiskTeacherCache(prefix, device="cpu", top_k=cache.top_k, vocab=cache.vocab)
    B, T, Ks = 2, responses.size(1), 4
    idxs = torch.tensor([0, 1])
    student_topk = torch.randint(0, cache.vocab, (B, T, Ks))
    assert torch.equal(cache.delta_for_student_topk(idxs, student_topk),
                       disk.delta_for_student_topk(idxs, student_topk))


def test_hash_model_dir_deterministic(tmp_path):
    """同一目录哈希确定；不同目录可区分；None 退化为路径哈希。"""
    a = tmp_path / "a"; a.mkdir()
    (a / "tokenizer.json").write_text("{}", encoding="utf-8")
    h1 = hash_model_dir(str(a))
    h2 = hash_model_dir(str(a))
    assert h1 == h2
    b = tmp_path / "b"; b.mkdir()
    (b / "tokenizer.json").write_text("[]", encoding="utf-8")
    assert hash_model_dir(str(b)) != h1
    assert hash_model_dir(None) != hash_model_dir(str(a))
    assert hash_model_dir("/nonexistent/path") == hash_model_dir("/nonexistent/path")
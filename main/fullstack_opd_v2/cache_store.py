"""Stage 1 磁盘 mmap 教师缓存存储（§3/§4/§5/§6/§7）。

解决 50K×8192 的 cache memory wall：不再把全部 (N,T,K) 张量一次性 cat 进 GPU/RAM。
- **磁盘 mmap 驻留**（§3）：`ids_sorted`/`delta_k_sorted`/`lengths` 三份 `.dat` 用
  `np.memmap` 文件驻留，GPU/RAM 只驻当前 batch 行（§4 batch-local）。
- **最小 sufficient statistics**（§2）：只持久化训练热路径真需要的
  `ids_sorted`(int32) + `delta_k_sorted`(fp32) + `lengths`(uint32) + `vocab`，
  其余 `ids/rl_k/ref_k/delta_k` 全是 build 中间量可舍（~4× 缩减）。
- **变长**（§7）：dense(T_max)+mask，`lengths` 给出每样本真实长度，padding 排除出统计。
- **metadata + 一致性**（§5/§6）：`<prefix>.metadata.json` 记 13 键（含 tokenizer/教师/
  reference/generation 模型哈希 + checksum）；加载前校验，不匹配 fail fast。

训练期 `DiskTeacherCache` 与 `TensorTeacherCache` 同接口（mode/top_k/vocab/
get_delta/delta_for_student_topk/topk/to/response_length），scheduler 零改动。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .cache import expand_student_topk_delta


class CacheConsistencyError(Exception):
    """加载磁盘缓存时 metadata 与当前 config 不一致（tokenizer/教师/top_k/长度等）。"""


# ---------------------------------------------------------------------------
# §5 metadata schema
# ---------------------------------------------------------------------------
METADATA_KEYS = (
    "dataset_size", "max_prompt_len", "max_response_len", "top_k", "vocab",
    "dtype", "num_samples", "total_tokens", "format_version",
    "tokenizer_hash", "teacher_model_hash", "reference_model_hash",
    "generation_model_hash", "checksum",
)
FORMAT_VERSION = "stage1-disk-1"


# ---------------------------------------------------------------------------
# §6 哈希工具（一致性校验）
# ---------------------------------------------------------------------------
def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_string(s: str) -> str:
    return hash_bytes(s.encode("utf-8"))


def hash_model_dir(path: str | None, seed: str = "") -> str:
    """对模型目录关键文件（tokenizer.json/config.json/…）做确定性 sha256。

    目录缺失 / 为 None 时（toy 场景）退化为对路径字符串哈希——保证同一模型路径
    得到同一哈希，便于一致性校验，但可区分不同路径。
    """
    if path and os.path.isdir(path):
        h = hashlib.sha256()
        h.update(seed.encode("utf-8"))
        for name in sorted(os.listdir(path)):
            fp = os.path.join(path, name)
            if os.path.isfile(fp):
                try:
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                except OSError:
                    pass
        return h.hexdigest()
    return hash_string(f"{seed}:{path or 'none'}")


def file_sha256(paths) -> str:
    """对一批文件按给定顺序串接流式求 sha256（§9 checksum 验签）。"""
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# §7 变长：每样本真实长度
# ---------------------------------------------------------------------------
def compute_lengths(responses: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(N,) 每样本非 padding token 数（不假设所有 sample 等长 T）。"""
    return (responses != pad_id).sum(dim=-1)


def _mask_from_lengths(lengths: torch.Tensor, T: int, device) -> torch.Tensor:
    """(N,) 长度 → (N, T) 有效 token 掩码（< len 为 True）。"""
    arange = torch.arange(T, device=device)
    return arange.unsqueeze(0) < lengths.to(torch.long).unsqueeze(-1)


# ---------------------------------------------------------------------------
# §3/§4 磁盘写入 + DiskTeacherCache
# ---------------------------------------------------------------------------
def _write_memmap(path: str, data: np.ndarray, chunk: int) -> None:
    """逐 chunk 直写 memmap（不整文件进 RAM：只驻 chunk 行）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mm = np.memmap(path, dtype=data.dtype, mode="w+", shape=data.shape)
    try:
        for i in range(0, data.shape[0], chunk):
            mm[i:i + chunk] = data[i:i + chunk]
        mm.flush()
    finally:
        del mm


def write_cache_disk(cache, prefix: str, responses: torch.Tensor | None = None,
                     pad_id: int = 0, hashes: dict | None = None,
                     max_response_len: int = 8192, max_prompt_len: int = 0,
                     dtype: str = "bf16", dataset_size: int = 0,
                     chunk: int = 256) -> dict:
    """把已 build 的 TensorTeacherCache（top-K）写为磁盘 mmap 三件套 + metadata。

    只落盘最小 sufficient statistics：ids_sorted(int32) + delta_k_sorted(fp32) +
    lengths(uint32)。逐 chunk 直写 memmap，GPU/RAM 只驻 chunk 行。

    返回 metadata dict（供调用方落 metadata.json）。
    """
    if cache.mode != "topk":
        raise ValueError("磁盘存储仅支持 top-K 模式（dense 忽略 storage）")
    N, T, K = cache.ids_sorted.shape

    ids_np = cache.ids_sorted.cpu().numpy().astype(np.int32)
    delta_np = cache.delta_k_sorted.float().cpu().numpy().astype(np.float32)
    # 注：delta_k_sorted 源自 teacher bf16 前向 → bf16 张量，numpy 不支持 bf16，
    # 必须先 .float() 再 numpy（否则 `Got unsupported ScalarType BFloat16`）。
    # 逐 chunk 直写（避免整文件 numpy 副本驻留 RAM）
    _write_memmap(f"{prefix}.ids_sorted.dat", ids_np, chunk)
    _write_memmap(f"{prefix}.delta_k_sorted.dat", delta_np, chunk)

    if responses is not None:
        lengths_np = compute_lengths(responses, pad_id).cpu().numpy().astype(np.uint32)
    else:
        lengths_np = np.full(N, T, dtype=np.uint32)          # 无响应数据 → 全 valid
    _write_memmap(f"{prefix}.lengths.dat", lengths_np, chunk)

    checksum = file_sha256([
        f"{prefix}.ids_sorted.dat", f"{prefix}.delta_k_sorted.dat",
        f"{prefix}.lengths.dat"])
    hashes = hashes or {}
    metadata = {
        "dataset_size": int(dataset_size) if dataset_size else int(N),
        "max_prompt_len": int(max_prompt_len),
        "max_response_len": int(max_response_len),
        "top_k": int(cache.top_k),
        "vocab": int(cache.vocab),
        "dtype": str(dtype),
        "num_samples": int(N),
        "total_tokens": int(lengths_np.sum()),
        "format_version": FORMAT_VERSION,
        "tokenizer_hash": hashes.get("tokenizer_hash", ""),
        "teacher_model_hash": hashes.get("teacher_model_hash", ""),
        "reference_model_hash": hashes.get("reference_model_hash", ""),
        "generation_model_hash": hashes.get("generation_model_hash", ""),
        "checksum": checksum,
    }
    with open(f"{prefix}.metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def load_cache_metadata(prefix: str) -> dict:
    """读 metadata.json，校验存在性 + format_version + checksum 与文件匹配。"""
    meta_path = f"{prefix}.metadata.json"
    if not os.path.exists(meta_path):
        raise CacheConsistencyError(f"缓存 metadata 缺失：{meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format_version") != FORMAT_VERSION:
        raise CacheConsistencyError(
            f"缓存 format_version={meta.get('format_version')} != 期望 {FORMAT_VERSION}")
    for k in METADATA_KEYS:
        if k not in meta:
            raise CacheConsistencyError(f"metadata 缺键 {k}")
    actual = file_sha256([
        f"{prefix}.ids_sorted.dat", f"{prefix}.delta_k_sorted.dat",
        f"{prefix}.lengths.dat"])
    if actual != meta["checksum"]:
        raise CacheConsistencyError(
            f"缓存 checksum 不匹配（文件被篡改或写坏）：期望 {meta['checksum']}，实际 {actual}")
    return meta


def hash_models_from_cfg(cfg: dict) -> dict:
    """从 config 提取模型路径并算一致性哈希（缺失/None 时退化为路径哈希）。"""
    return {
        "tokenizer_hash": hash_model_dir(cfg.get("tokenizer_path") or cfg.get("student_path"), "tok"),
        "teacher_model_hash": hash_model_dir(cfg.get("teacher_rl_path"), "tea"),
        "reference_model_hash": hash_model_dir(cfg.get("teacher_ref_path"), "ref"),
        "generation_model_hash": hash_model_dir(cfg.get("student_path"), "gen"),
    }


def verify_consistency(meta: dict, cfg: dict, hashes_now: dict | None = None) -> None:
    """比对 metadata 与当前 config 的 tokenizer/教师/ref/top_k/max_response_len。

    不匹配抛 CacheConsistencyError（fail fast，不静默继续）。hashes_now 缺省时用
    cfg 现场重算。
    """
    hashes_now = hashes_now or hash_models_from_cfg(cfg)
    checks = [
        ("tokenizer_hash", meta.get("tokenizer_hash"), hashes_now["tokenizer_hash"]),
        ("teacher_model_hash", meta.get("teacher_model_hash"), hashes_now["teacher_model_hash"]),
        ("reference_model_hash", meta.get("reference_model_hash"), hashes_now["reference_model_hash"]),
    ]
    for name, got, want in checks:
        if got != want:
            raise CacheConsistencyError(
                f"缓存 {name}（{got}）与当前配置（{want}）不一致，不准加载")
    top_k = meta.get("top_k")
    cfg_top_k = (cfg.get("cache") or {}).get("top_k")
    if cfg_top_k and top_k != int(cfg_top_k):
        raise CacheConsistencyError(
            f"缓存 top_k={top_k} 与配置 cache.top_k={cfg_top_k} 不一致，不准加载")
    # max_response_len 读取【dataset 块】（data loader 实际用键）而非顶层——此前误读顶层
    # 导致 cache(T=2048) vs dataset(T=4096) 错配不报错，直到训练中 searchsorted 维度
    # 崩溃（2026-08-17 实测 [1,2048,256] vs [1,4096,256]）。缓存与数据必须同长。
    _data_cfg = cfg.get("dataset") or {}
    _data_len = _data_cfg.get("max_response_len")
    meta_len = meta.get("max_response_len")
    if meta_len and _data_len and meta_len != int(_data_len):
        raise CacheConsistencyError(
            f"缓存 max_response_len={meta_len} 与 dataset.max_response_len={_data_len} "
            "不一致（缓存用旧配置建则需对齐 dataset.max_response_len 或重建缓存，"
            "否则 teacher top-K 支撑与训练响应维度错配 → searchsorted 崩溃）。")


class DiskTeacherCache:
    """磁盘 mmap 驻留教师缓存，与 TensorTeacherCache 同接口（scheduler 零改动）。

    - `_ids`/`_delta`/`_lengths` 为 `np.memmap` 只读视图，GPU/RAM 不驻留全量。
    - `delta_for_student_topk` 只读本 batch 行（§4 batch-local）：`mmap[idxs]` → GPU。
    - `get_delta`/`topk` 与内存缓存一致（top-K 模式）。
    """

    def __init__(self, prefix: str, device: torch.device | str = "cpu",
                 top_k: int = 0, vocab: int = 0, mode: str = "topk"):
        self.prefix = prefix
        self.device = torch.device(device)
        self.top_k = top_k
        self.vocab = vocab
        self.mode = mode
        ids = np.memmap(f"{prefix}.ids_sorted.dat", dtype=np.int32, mode="r")
        # 形状从文件字节数反推：(N*T*K) 一维，reshape 回三维
        n_total = ids.size
        with open(f"{prefix}.metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_samples = int(meta["num_samples"])
        self.T = n_total // (self.num_samples * top_k) if top_k else 0
        self.K = top_k
        self._ids = ids.reshape((self.num_samples, self.T, self.K))
        self._delta = np.memmap(f"{prefix}.delta_k_sorted.dat", dtype=np.float32,
                                mode="r").reshape((self.num_samples, self.T, self.K))
        self._lengths = np.memmap(f"{prefix}.lengths.dat", dtype=np.uint32,
                                  mode="r").reshape((self.num_samples,))

    @torch.no_grad()
    def _fetch(self, idxs: torch.Tensor):
        """batch-local：只读本 batch 的行 → GPU 张量（绝不整文件进 GPU/RAM）。"""
        idx = idxs.cpu().tolist()
        ids = torch.from_numpy(self._ids[idx]).to(self.device)
        delta = torch.from_numpy(self._delta[idx]).to(self.device)
        return ids, delta

    def delta_for_student_topk(self, idxs, student_topk_ids, vocab_out=None,
                               fill=0.0, mask=None):
        ids_bs, delta_bs = self._fetch(idxs)
        # 支撑张量随 batch 行已移到 self.device；student 支撑须同设备，否则 searchsorted 报
        # "Expected all tensors to be on the same device"（部署实测：缓存 cuda、student 支撑 cpu）。
        if student_topk_ids.device != ids_bs.device:
            student_topk_ids = student_topk_ids.to(ids_bs.device)
        return expand_student_topk_delta(ids_bs, delta_bs, student_topk_ids,
                                         self.vocab, vocab_out, fill, mask)

    def delta_at_student_topk(self, idxs, student_topk_ids, device=None):
        """P0：稀疏 Δ_T 展开到 s_cur top-K 支撑（(B,T,Ks)），不建稠密 (B,T,V)。

        与 delta_for_student_topk 同 searchsorted 语义；磁盘 batch-local 拉取。
        """
        ids_bs, delta_bs = self._fetch(idxs)
        if student_topk_ids.device != ids_bs.device:
            student_topk_ids = student_topk_ids.to(ids_bs.device)
        from .cache import expand_student_topk_delta_sparse
        out = expand_student_topk_delta_sparse(ids_bs, delta_bs, student_topk_ids)
        return out if device is None else out.to(device)

    def get_delta(self, idxs):
        if self.mode != "dense":
            raise RuntimeError("稀疏缓存请用 delta_for_student_topk()（磁盘存储仅 top-K）")
        raise NotImplementedError("磁盘存储仅支持 top-K 模式（dense 忽略 storage）")

    def topk(self, idxs):
        ids_bs, delta_bs = self._fetch(idxs)
        return ids_bs, delta_bs

    def response_length(self, idxs) -> torch.Tensor:
        return torch.from_numpy(self._lengths[idxs.cpu().tolist()]).to(self.device)

    def token_mask(self, idxs) -> torch.Tensor:
        lens = self.response_length(idxs)                      # (B,)
        return _mask_from_lengths(lens, self.T, self.device)   # (B, T)

    def to(self, device) -> "DiskTeacherCache":
        self.device = torch.device(device)                     # memmap 无需搬移，仅换目标设备
        return self
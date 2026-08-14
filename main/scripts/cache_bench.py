"""Stage 1 §8 性能 benchmark：K∈{32,64,128,256} 磁盘缓存存储性能对比。

测量（对每个 K）：cache_size_disk / RAM_peak / GPU_peak / write_time / lookup_latency /
throughput / I/O bandwidth。用【合成缓存】而非真实教师前向——本基准测的是存储架构
（磁盘写 + mmap + batch-local 查找）的性能，与 Δ_T 语义内容无关，故无需加载真实模型，
CPU 即可跑。

用法（服务器）：
  python scripts/cache_bench.py --N 2000 --max-len 2048 --k 32,64,128,256 \
    --batch 8,16,32 --out /root/autodl-tmp/eval/cache_bench.json

--N/--max-len 为单 K 的 cache 行数 × 响应长度（把盘体控制在分钟级实测）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import torch

# 脚本位于 main/scripts/ 下：把仓库根（main/）加进 sys.path 以便 import fullstack_opd_v2
_MAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from fullstack_opd_v2.cache_store import write_cache_disk


def _synthetic_cache(N: int, T: int, K: int, vocab: int, seed: int = 0):
    """构造一个『已 build 语义等价』的 top-K 缓存最小对象（ids_sorted/delta_k_sorted/
    vocab/top_k/mode），供 write_cache_disk 落盘。ids_sorted 每行升序（searchsorted 前提）。"""
    g = torch.Generator().manual_seed(seed + K)
    ids = torch.randint(0, vocab, (N, T, K), generator=g)
    ids_sorted, _ = ids.sort(dim=-1)                       # 升序（对齐真实 build）
    delta = torch.randn(N, T, K, dtype=torch.float32)
    return type("SynthCache", (), {
        "mode": "topk", "top_k": K, "vocab": vocab,
        "ids_sorted": ids_sorted, "delta_k_sorted": delta,
    })()


def _peak_ram_mb() -> float:
    """当前进程已分配的峰值 RAM（MB）。"""
    cur, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def _bench_one(N: int, T: int, K: int, batches, out_dir: str) -> dict:
    vocab = 512
    cache = _synthetic_cache(N, T, K, vocab)
    prefix = os.path.join(out_dir, f"cache_K{K}")
    # ---- GPU / RAM 峰值：写盘过程（torch 量化 GPU，tracemalloc 量化本进程 RAM）----
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    write_cache_disk(cache, prefix, responses=None, pad_id=0,
                     hashes=None, max_response_len=T, max_prompt_len=T,
                     dtype="bf16", dataset_size=N)
    write_s = time.perf_counter() - t0
    row = {
        "top_k": K, "N": N, "T": T,
        "cache_size_disk_bytes": int(sum(
            os.path.getsize(f"{prefix}{suf}")
            for suf in (".ids_sorted.dat", ".delta_k_sorted.dat", ".lengths.dat"))),
        "cache_size_disk_GB": 0.0,
        "ids_sorted_dtype": "int32", "delta_k_sorted_dtype": "float32",
        "write_time_s": round(write_s, 3),
        "RAM_peak_MB": round(_peak_ram_mb(), 1),
        "GPU_peak_MB": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) \
            if torch.cuda.is_available() else 0.0,
    }
    row["cache_size_disk_GB"] = round(row["cache_size_disk_bytes"] / (1024 ** 3), 3)
    tracemalloc.stop()

    # ---- lookup 延迟 / 吞吐 / I/O 带宽（batch-local mmap 读）----
    from fullstack_opd_v2.cache_store import DiskTeacherCache
    disk = DiskTeacherCache(prefix, device="cpu", top_k=K, vocab=vocab)
    idxs = torch.arange(0, N)
    student_topk = _synthetic_cache(1, T, K, vocab).ids_sorted
    lookup = {}
    for b in batches:
        bi = idxs[:b]
        n_iter = max(1, 200 // b)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            disk.delta_for_student_topk(bi, student_topk.expand(b, -1, -1))
        dt = (time.perf_counter() - t0) / n_iter
        tokens = b * T
        lookup[b] = {
            "lookup_latency_ms": round(dt * 1e3, 3),
            "throughput_tok_s": round(tokens / dt, 1),
            # I/O 带宽 ≈ 读入字节 / 耗时（batch 行 × 张量字节）
            "io_bandwidth_GB_s": round((b * T * K * (4 + 4)) / dt / (1024 ** 3), 3),
        }
    row["lookup"] = lookup
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--k", default="32,64,128,256")
    ap.add_argument("--batch", default="8,16,32")
    ap.add_argument("--out", default="cache_bench.json")
    ap.add_argument("--workdir", default=None, help="临时盘体目录（默认系统临时）")
    args = ap.parse_args()

    Ks = [int(x) for x in args.k.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    out_dir = args.workdir or os.path.join(os.environ.get("TEMP", "/tmp"), "cache_bench")
    os.makedirs(out_dir, exist_ok=True)

    rows = [_bench_one(args.N, args.max_len, K, batches, out_dir) for K in Ks]
    report = {"N": args.N, "max_len": args.max_len, "results": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ASCII 对比表（Windows/GBK 安全，不用 ✅/⚠️）
    print(f"[OK] 结果写至 {args.out}")
    print(f"{'K':>5} {'disk_GB':>8} {'RAM_MB':>8} {'GPU_MB':>8} {'write_s':>8} "
          f"{'lat_ms':>8} {'thr_tok/s':>10} {'I/O_GB/s':>9}")
    for r in rows:
        lat = r["lookup"][batches[0]]["lookup_latency_ms"]
        thr = r["lookup"][batches[0]]["throughput_tok_s"]
        io = r["lookup"][batches[0]]["io_bandwidth_GB_s"]
        print(f"{r['top_k']:>5} {r['cache_size_disk_GB']:>8.3f} {r['RAM_peak_MB']:>8.1f} "
              f"{r['GPU_peak_MB']:>8.1f} {r['write_time_s']:>8.3f} {lat:>8.3f} "
              f"{thr:>10.0f} {io:>9.3f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""D3：Δ_T 信号体检（inspect_delta_cache.py）

纯离线、CPU 可跑。加载磁盘 mmap 教师缓存（ids_sorted/delta_k_sorted/lengths），
在 teacher top-K=256 支撑上统计 Δ_T 分布，判定信号有效性（H4：chat cache 的
Δ_T 是否有方向/量级/密度，排除「教师对在模板下分歧弱」导致训练信号无效）。

判据（写死，不许放宽）：
- 通过：正 Δ token 占比 ≥ 15% 且 |均值| ≤ 1.0
- 不通过：正比例 < 5%，或均值 < -1.0
- 边界：5% ≤ 正比例 < 15%，或 -1.0 ≤ 均值 < -0.5 → 记录并标注风险

用法：
    python scripts/inspect_delta_cache.py --prefix /root/autodl-tmp/cache_skywork_chat
        [--max-samples 100] [--out report.json] [--student-topk-json path] [--seed 0]

输出：打印统计表 + 写 JSON（--out 指定，默认 {prefix}.delta_inspect.json）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.cache_store import load_cache_metadata  # noqa: E402

# 判据常量（写死）
PASS_POS_RATIO = 0.15
FAIL_POS_RATIO = 0.05
PASS_ABS_MEAN = 1.0
FAIL_MEAN = -1.0
BOUNDARY_MEAN = -0.5
DELTA_CLIP = 2.0


def _effective_mask(lengths: np.ndarray, T: int) -> np.ndarray:
    """(N,) lengths → (N, T) 有效 token 掩码（padding 排除）。"""
    N = lengths.shape[0]
    t = np.arange(T)[None, :].repeat(N, axis=0)
    return t < lengths[:, None]


def _pos_frac(lengths: np.ndarray, T: int) -> np.ndarray:
    """(N,) lengths → (N, T) 每个有效 token 的相对位置分数（0~1）；padding 为 -1。"""
    N = lengths.shape[0]
    t = np.arange(T, dtype=np.float64)[None, :].repeat(N, axis=0)
    denom = np.maximum(lengths[:, None].astype(np.float64) - 1.0, 1.0)
    frac = t / denom
    frac[~_effective_mask(lengths, T)] = -1.0
    return frac


def _quantiles(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "std": float(vals.std()),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
    }


def analyze_delta_distribution(delta: np.ndarray, mask: np.ndarray | None = None,
                               clip: float = DELTA_CLIP) -> dict:
    """在给定支撑上统计 Δ_T 分布（核心纯函数，便于单测）。

    delta: (N, T, K) fp32；mask: (N, T) 有效 token 掩码（None=全有效）。
    """
    vals = delta[mask] if mask is not None else delta.reshape(-1)
    if vals.size == 0:
        return {"n": 0, "pos_ratio": None, "pos_ratio_gt05": None,
                "pos_ratio_gt1": None, "clip_ratio": None, "dist": _quantiles(vals)}
    dist = _quantiles(vals)
    absv = np.abs(vals)
    return {
        "n": int(vals.size),
        "pos_ratio": float((vals > 0).mean()),
        "pos_ratio_gt05": float((vals > 0.5).mean()),
        "pos_ratio_gt1": float((vals > 1.0).mean()),
        "clip_ratio": float((absv > clip).mean()),
        "dist": dist,
    }


def analyze_position_breakdown(delta: np.ndarray, lengths: np.ndarray,
                               T: int) -> dict:
    """按 response 有效长度分三段：前 25% / 中 50% / 后 25%。"""
    frac = _pos_frac(lengths, T)
    mask = frac >= 0
    out = {}
    segs = {
        "early_0_25": (frac >= 0.0) & (frac < 0.25),
        "mid_25_75": (frac >= 0.25) & (frac < 0.75),
        "late_75_1": (frac >= 0.75),   # 最后一段含 frac==1.0 的末 token（mask 已排除 padding）
    }
    for name, seg in segs.items():
        seg_delta = delta[seg & mask]
        if seg_delta.size == 0:
            out[name] = {"n": 0, "pos_ratio": None, "mean": None}
        else:
            out[name] = {"n": int(seg_delta.size),
                         "pos_ratio": float((seg_delta > 0).mean()),
                         "mean": float(seg_delta.mean())}
    return out


def load_cache(prefix: str):
    """加载磁盘 mmap cache → (ids, delta, lengths, meta)。"""
    meta = load_cache_metadata(prefix)
    ids = np.memmap(f"{prefix}.ids_sorted.dat", dtype=np.int32, mode="r")
    delta = np.memmap(f"{prefix}.delta_k_sorted.dat", dtype=np.float32, mode="r")
    lengths = np.memmap(f"{prefix}.lengths.dat", dtype=np.uint32, mode="r")
    N = int(meta["num_samples"])
    K = int(meta["top_k"])
    T = int(ids.size // (N * K))
    return (ids.reshape(N, T, K), delta.reshape(N, T, K),
            lengths.reshape(N), meta)


def judge(result: dict) -> dict:
    """按写死判据判定 D3 是否通过。result 为 analyze_delta_distribution 全量输出。"""
    pos = result["pos_ratio"]
    mean = result["dist"]["mean"]
    if pos is None or mean is None:
        return {"passed": False, "verdict": "FAIL", "reason": "无有效 token"}
    if pos >= PASS_POS_RATIO and abs(mean) <= PASS_ABS_MEAN:
        return {"passed": True, "verdict": "PASS", "reason":
                f"正Δ占比 {pos:.3f}≥{PASS_POS_RATIO} 且 |均值| {abs(mean):.3f}≤{PASS_ABS_MEAN}"}
    if pos < FAIL_POS_RATIO or mean < FAIL_MEAN:
        return {"passed": False, "verdict": "FAIL", "reason":
                f"正Δ占比 {pos:.3f}<{FAIL_POS_RATIO} 或 均值 {mean:.3f}<{FAIL_MEAN}（信号弱/方向可疑）"}
    return {"passed": False, "verdict": "BOUNDARY", "reason":
            f"边界：正Δ占比 {pos:.3f}∈[{FAIL_POS_RATIO},{PASS_POS_RATIO}) 或 均值 {mean:.3f}∈[{FAIL_MEAN},{BOUNDARY_MEAN})"}


def compute_student_overlap(teacher_ids: np.ndarray, student_topk: dict) -> dict:
    """teacher top-K ids 与外部 student top-K 的重合率（searchsorted 语义近似）。

    student_topk: {sample_idx: list[int]} 每条样本的 student top-K token ids（不要求同 K）。
    """
    if not student_topk:
        return {"status": "no-data"}
    hit_total = 0
    pos_total = 0
    per_sample = []
    for idx_s, ids_list in student_topk.items():
        s_ids = np.asarray(ids_list, dtype=np.int64)
        t_ids = teacher_ids[int(idx_s)].reshape(-1).astype(np.int64)
        # teacher ids_sorted 是排序的 → searchsorted 命中
        t_sorted = np.sort(t_ids)
        hit = np.isin(s_ids, t_sorted)
        hit_total += int(hit.sum())
        pos_total += int(s_ids.size)
        per_sample.append({"sample": int(idx_s), "hit_ratio": float(hit.mean())})
    return {"status": "ok", "overall_hit_ratio": hit_total / max(pos_total, 1),
            "n_samples": len(per_sample), "per_sample": per_sample}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D3 Δ_T 信号体检")
    ap.add_argument("--prefix", required=True, help="cache 前缀（如 /root/autodl-tmp/cache_skywork_chat）")
    ap.add_argument("--max-samples", type=int, default=0, help="抽样条数（0=全量）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="输出 JSON 路径（默认 {prefix}.delta_inspect.json）")
    ap.add_argument("--student-topk-json", default="", help="可选：外部 student top-K JSON（{idx:[ids]}）")
    args = ap.parse_args(argv)

    ids, delta, lengths, meta = load_cache(args.prefix)
    N, T, K = ids.shape
    if args.max_samples and args.max_samples < N:
        rng = np.random.default_rng(args.seed)
        idxs = np.sort(rng.choice(N, size=args.max_samples, replace=False))
        ids, delta, lengths = ids[idxs], delta[idxs], lengths[idxs]
        N = idxs.size

    mask = _effective_mask(lengths, T)
    full = analyze_delta_distribution(delta, mask)
    by_pos = analyze_position_breakdown(delta, lengths, T)
    clip_info = {"clip": DELTA_CLIP,
                 "clip_ratio_full": full["clip_ratio"],
                 "clip_ratio_pos": float((np.abs(delta[mask]) > DELTA_CLIP).mean())}

    student_overlap = {"status": "not-provided"}
    if args.student_topk_json:
        with open(args.student_topk_json, encoding="utf-8") as f:
            student_topk = json.load(f)
        student_overlap = compute_student_overlap(ids, student_topk)

    j = judge(full)
    report = {
        "cache_prefix": args.prefix,
        "meta": {"num_samples": N, "T": T, "top_k": K,
                 "prompt_format": meta.get("prompt_format", "raw"),
                 "dataset_size": meta.get("dataset_size"),
                 "max_response_len": meta.get("max_response_len"),
                 "dtype": meta.get("dtype")},
        "full_distribution": full,
        "position_breakdown": by_pos,
        "clip_interaction": clip_info,
        "student_topk_overlap": student_overlap,
        "verdict": j,
    }

    out = args.out or f"{args.prefix}.delta_inspect.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印
    print(f"[D3] cache={args.prefix} samples={N} T={T} K={K} prompt_format={report['meta']['prompt_format']}")
    d = full["dist"]
    print(f"[D3] Δ 分布: n={d['n']} mean={d['mean']:.4f} median={d['median']:.4f} std={d['std']:.4f}")
    print(f"[D3]   分位 P10/P25/P50/P75/P90 = {d['p10']:.3f}/{d['p25']:.3f}/{d['p50']:.3f}/{d['p75']:.3f}/{d['p90']:.3f}")
    print(f"[D3]   正Δ占比={full['pos_ratio']:.4f}  Δ>0.5={full['pos_ratio_gt05']:.4f}  Δ>1.0={full['pos_ratio_gt1']:.4f}")
    print(f"[D3]   clip(|Δ|>{DELTA_CLIP})={clip_info['clip_ratio_pos']:.4f}")
    print("[D3] 位置分解:")
    for k, v in by_pos.items():
        print(f"   {k}: n={v['n']} 正比例={v['pos_ratio']} 均值={v['mean']}")
    if student_overlap.get("status") == "ok":
        print(f"[D3] student top-K 重合率: {student_overlap['overall_hit_ratio']:.4f} (n={student_overlap['n_samples']})")
    print(f"[D3] 判据: {j['verdict']} —— {j['reason']}")
    print(f"[D3] 报告已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

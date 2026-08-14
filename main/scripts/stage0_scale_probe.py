#!/usr/bin/env python3
"""Stage 0 多阶段扩展 + 规模决策：读 gen_benchmark 实测 JSON，外推各规模 wall time，
打印测量表 + 推荐规模（docs《Stage 0 提示词》§4/§6）。

核心口径：成本用实测 E[L]（非 max_len），T = N × min(E[L], max_len)；
外推不属于生成线程墙钟，而是 Σ 各阶段独立生成时间。

CLI：
  python scripts/stage0_scale_probe.py \
    --benchmark /root/autodl-tmp/eval/gen_benchmark.json \
    --out /root/autodl-tmp/eval/scale_probe_report.json

推荐逻辑（§6 验收）：
  - 依据实测 tok/s 与长度分布，对 pilot/scale-1/scale-2/full 各算 1 卡与 2 卡 hours
  - 若 full(50K×8192) 超现实阈值（默认 72h）→ 建议降规模 / 升 batch / 2 卡并行
  - 依据 P(L>2048)/P(L>4096) 判断是否值得保持 max_response_len=8192
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 多阶段扩展（docs 提示词 §4 / 计划任务 4）
STAGES = [
    ("pilot",   5000,  2048),
    ("scale-1", 10000, 4096),
    ("scale-2", 25000, 8192),
    ("full",    50000, 8192),
]

# 超现实阈值：单阶段单卡超过该小时数认为不现实（约 3 天）
UNREALISTIC_HOURS = 72.0


def extrapolate(stage_n: int, stage_max_len: int, mean_len: float,
                measured_tok_s: float, n_gpus: int) -> dict:
    """预计 token = stage_n × min(mean_len, stage_max_len)（用实测 E[L]，非 max_len）
    预计 hours = 预计 token / (measured_tok_s × n_gpus) / 3600。
    mean_len 超过 max_len 时按 max_len 计（不超生成上限）。"""
    eff_len = min(mean_len, stage_max_len)
    total_tokens = stage_n * eff_len
    total_hours = total_tokens / (measured_tok_s * n_gpus) / 3600
    return {"n": stage_n, "max_len": stage_max_len, "mean_len": eff_len,
            "tok_s": measured_tok_s, "n_gpus": n_gpus,
            "total_tokens": int(total_tokens), "total_hours": total_hours}


def load_benchmark(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def best_group(bench: dict, max_len: int) -> dict | None:
    """从 benchmark JSON 里挑与某 max_new 档匹配的组（tok/s 最高者）作为外推依据。"""
    cands = [v for v in bench.values()
             if isinstance(v, dict) and v.get("max_new_tokens") == max_len
             and v.get("tok_per_s", 0) > 0]
    if not cands:
        return None
    return max(cands, key=lambda v: v["tok_per_s"])


def build_rows(bench: dict, n_gpus: int) -> list[dict]:
    """对每个 stage，用对应 max_len 档的实测 mean_len + tok/s 外推。"""
    rows = []
    for name, n, max_len in STAGES:
        g = best_group(bench, max_len)
        if g is None:
            rows.append({"stage": name, "n": n, "max_len": max_len,
                         "mean_len": None, "tok_s": None,
                         "total_hours": None, "n_gpus": n_gpus,
                         "missing_benchmark": True})
            continue
        rows.append({"stage": name, "n": n, "max_len": max_len,
                     **extrapolate(n, max_len, g["mean_len"], g["tok_per_s"], n_gpus)})
    return rows


def print_table(rows_1: list[dict], rows_2: list[dict]) -> None:
    print("=" * 78)
    print(f"{'stage':<9}{'N':>7}{'max_len':>9}{'mean_len':>9}{'tok/s':>7}"
          f"{'hours(1卡)':>12}{'hours(2卡)':>12}")
    print("-" * 78)
    for r1, r2 in zip(rows_1, rows_2):
        if r1.get("missing_benchmark"):
            print(f"{r1['stage']:<9}{r1['n']:>7,}{r1['max_len']:>9}  [缺该档 benchmark 实测]")
            continue
        h1 = r1["total_hours"]
        h2 = r2["total_hours"]
        flag = "  <-- 超现实" if h1 > UNREALISTIC_HOURS else ""
        print(f"{r1['stage']:<9}{r1['n']:>7,}{r1['max_len']:>9,}"
              f"{r1['mean_len']:>9.0f}{r1['tok_s']:>7.1f}"
              f"{h1:>12.1f}{h2:>12.1f}{flag}")
    print("=" * 78)


def recommend(rows_1: list[dict], rows_2: list[dict], bench: dict,
              unrealistic_hours: float = UNREALISTIC_HOURS) -> list[str]:
    """§6 验收输出：推荐规模 / 是否 2 卡 / 是否保持 8192 / Base Pool 推荐。"""
    rec: list[str] = []
    full_1 = next((r for r in rows_1 if r["stage"] == "full"), None)
    full_2 = next((r for r in rows_2 if r["stage"] == "full"), None)

    # 1) 保持 8192？看长度分布里 P(L>4096) 的实测
    g8192 = best_group(bench, 8192)
    if g8192:
        p_gt4096 = g8192.get("P_L_gt_4096", 0.0)
        if p_gt4096 < 0.05:
            rec.append(f"长度分布：P(L>4096)={p_gt4096:.2f}，极少数样本超 4096 → "
                       f"max_response_len=8192 可保留，但实际 E[L]={g8192['mean_len']:.0f} 远低于上限")
        else:
            rec.append(f"长度分布：P(L>4096)={p_gt4096:.2f}，确有长尾 → 8192 上限合理")

    # 2) full 规模是否现实 + 是否 2 卡
    if full_1 and not full_1.get("missing_benchmark"):
        h1, h2 = full_1["total_hours"], full_2["total_hours"]
        if h1 <= unrealistic_hours:
            rec.append(f"full(50K,8192) 单卡 {h1:.0f}h 现实 → 保持 50K×8192，单卡即可")
        elif h2 <= unrealistic_hours:
            rec.append(f"full(50K,8192) 单卡 {h1:.0f}h 超现实，但 2 卡 {h2:.0f}h 达标 → 需 2 卡并行生成")
        else:
            rec.append(f"full(50K,8192) 2 卡 {h2:.0f}h 仍超现实 → 建议降规模"
                       f"（见下方推荐 N）")
    # 3) 推荐 materialized 规模（静态锚点）
    rec.append("Base Pool 推荐：materialized 静态锚点实际只需 E[L] 长度，"
               "建议 5K~10K 起步，其余 prompt 留空待 L2 在线 refresh")
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, type=Path, help="gen_benchmark.py 输出 JSON")
    p.add_argument("--out", type=Path, default=Path("scale_probe_report.json"),
                   help="决策报告输出 JSON")
    p.add_argument("--unrealistic-hours", type=float, default=UNREALISTIC_HOURS,
                   help="单阶段单卡超现实阈值（默认 72h）")
    args = p.parse_args()
    if not args.benchmark.is_file():
        raise SystemExit(f"benchmark JSON 不存在: {args.benchmark}")

    unrealistic = args.unrealistic_hours
    bench = load_benchmark(args.benchmark)
    rows_1 = build_rows(bench, n_gpus=1)
    rows_2 = build_rows(bench, n_gpus=2)
    print(f"实测 benchmark 组：{len(bench)} 个\n")
    print_table(rows_1, rows_2)
    print("\n推荐（§6 验收结论）：")
    for i, r in enumerate(recommend(rows_1, rows_2, bench, unrealistic), 1):
        print(f"  {i}. {r}")

    report = {
        "stages": STAGES,
        "one_gpu": rows_1,
        "two_gpu": rows_2,
        "recommendations": recommend(rows_1, rows_2, bench, unrealistic),
        "unrealistic_hours_threshold": unrealistic,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 决策报告已存 {args.out}")


if __name__ == "__main__":
    main()
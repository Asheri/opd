#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-0d（归因分析 §5 0d）：on-policy（refresh）占比核实。

目的：量化 F3 的推断——E2 run 的 metrics.csv 里 base（500 条静态重放）与 refresh
（on-policy 补充）的步数比。若 refresh 占比远小于 base，则"固定 D 重放"主导训练，
强化 RC1（固定 500 条 base 轨迹偏离论文 on-policy）的论证；若 refresh 占比可观，
则需重新审视"静态重放"的主导性。

metrics.csv 的相位列名不确定（pool/phase/stage/type），脚本自动探测可用列；
找不到则原样报告列名（不伪造占比）。

用法（服务器）：
    /root/miniconda3/bin/python -u scripts/onpolicy_share.py \
        --run-dir /root/autodl-tmp/runs_s2_fix/S2_E2_opd1024 \
        --metrics metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# 已知表示"刷新/on-policy"的相位值（大小写不敏感）
_REFRESH_VALUES = {"refresh", "s2_refresh", "refresh_phase", "rollout_refresh"}
# 已知表示"base/静态重放"的相位值
_BASE_VALUES = {"base", "s2_base", "base_phase", "train"}


def _detect_phase_col(headers: list[str]) -> str | None:
    """从表头探测相位列：优先 pool/phase/stage/type，否则 None。"""
    for cand in ("pool", "phase", "stage", "type", "phase_type"):
        if cand in headers:
            return cand
    return None


def pool_share(rows: list[dict], phase_col: str | None) -> dict:
    """按相位列统计 base/refresh/其他 的步数占比。

    rows: csv.DictReader 行；phase_col None → 返回仅含列名信息的结构（不伪造）。
    """
    if phase_col is None:
        return {"phase_col": None, "available_cols": sorted(rows[0].keys()) if rows else [],
                "base_steps": 0, "refresh_steps": 0, "other_steps": len(rows),
                "base_share": None, "refresh_share": None, "note": "无相位列，无法统计（不伪造）"}
    base = refresh = other = 0
    refresh_vals: list[str] = []
    for r in rows:
        v = (r.get(phase_col) or "").strip().lower()
        if v in _REFRESH_VALUES or any(tok in v for tok in ("refresh",)):
            refresh += 1
            refresh_vals.append(v)
        elif v in _BASE_VALUES or any(tok in v for tok in ("base",)):
            base += 1
        else:
            other += 1
    n = max(1, len(rows))
    return {"phase_col": phase_col,
            "available_cols": sorted(rows[0].keys()) if rows else [],
            "base_steps": base, "refresh_steps": refresh, "other_steps": other,
            "base_share": round(base / n, 4), "refresh_share": round(refresh / n, 4),
            "other_share": round(other / n, 4),
            "refresh_values_seen": sorted(set(refresh_vals))[:10]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="训练 run 目录（含 metrics.csv）")
    p.add_argument("--metrics", default="metrics.csv", help="metrics 文件名（默认 metrics.csv）")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    path = os.path.join(args.run_dir, args.metrics)
    if not os.path.isfile(path):
        print(f"[onpolicy] metrics 不存在: {path}", flush=True)
        sys.exit(2)
    with open(path, encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        print(f"[onpolicy] metrics 空: {path}", flush=True)
        sys.exit(2)
    phase_col = _detect_phase_col(list(reader[0].keys()))
    res = pool_share(reader, phase_col)
    import json
    print(json.dumps(res, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

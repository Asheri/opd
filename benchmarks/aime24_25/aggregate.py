"""AIME24/25 蒸馏效果结果汇总：老师基线 / 学生蒸馏前 / 学生蒸馏后 → 对比表。

读 eval CLI 写出的每样本 jsonl（每行含 "correct": true/false），按目录结构归类：
  results/teacher/*.jsonl                          → 教师基线（JustRL-1.5B）
  results/student_baseline/<combo>/*.jsonl         → 学生蒸馏前
  results/student_post/<combo>/<step_N>/*.jsonl    → 学生蒸馏后（watch 产出）

用法: python aggregate.py <results_dir>
"""
from __future__ import annotations

import glob
import json
import os
import sys

# 文件名 → 年份（AIME24 / AIME25）
# 兼容两种命名：旧 benchmark（Maxwell-Jia_AIME_2024.jsonl）与 eval-aime（AIME24.jsonl / AIME25.jsonl）
def _year(filename: str) -> str:
    if "2024" in filename or "AIME24" in filename or "aime_2024" in filename:
        return "AIME24"
    if "2025" in filename or "AIME25" in filename or "aime_2025" in filename:
        return "AIME25"
    return "?"


def _accuracy(path: str) -> float:
    n = correct = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                      # R4：跳过损坏行，不整表崩溃
            n += 1
            correct += int(row.get("correct", False))
    return (correct / n * 100.0) if n else float("nan"), correct, n


def _collect(results_dir: str) -> dict:
    """返回 {阶段: {模型: {年份: 准确率}}}。阶段 ∈ {teacher, student_pre, student_post}。"""
    out = {"teacher": {}, "student_pre": {}, "student_post": {}}
    for path in glob.glob(os.path.join(results_dir, "**", "*.jsonl"), recursive=True):
        rel = os.path.relpath(path, results_dir)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue
        stage_dir, rest = parts[0], parts[1:]
        year = _year(os.path.basename(path))
        if year == "?":
            continue
        acc, correct, n = _accuracy(path)
        if stage_dir == "teacher":
            out["teacher"].setdefault("JustRL-1.5B", {})[year] = (acc, correct, n)
        elif stage_dir == "student_baseline":
            combo = rest[0] if rest else "?"
            out["student_pre"].setdefault(_combo_label(combo), {})[year] = (acc, correct, n)
        elif stage_dir == "student_post":
            # path: student_post/<combo>/[step_N/]<dataset>.jsonl
            combo = rest[0] if rest else "?"
            step = rest[1] if len(rest) > 1 and rest[1].startswith("step_") else rest[1] if len(rest) > 1 else "final"
            key = f"{_combo_label(combo)}@{step}"
            out["student_post"].setdefault(key, {})[year] = (acc, correct, n)
    return out


def _combo_label(combo: str) -> str:
    return {
        "combo1_qwen3_1p7b": "组1 Qwen3-1.7B",
        "combo2_qwen3_4b": "组2 Qwen3-4B",
        "combo3_r1_distill_7b": "组3 R1-Distill-7B",
    }.get(combo, combo)


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    acc, c, n = v
    return f"{acc:6.2f}% ({c}/{n})"


def main() -> None:
    # 中文/± 输出在 GBK 终端（Windows）下炸，强制 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    data = _collect(results_dir)

    print("=" * 78)
    print("全栈 OPD · 弱到强蒸馏 · AIME24/25 对比")
    print("=" * 78)

    # 阶段行列表：(阶段, 模型, 是 post 则记录 Δ)
    rows = []
    for model, years in data["teacher"].items():
        rows.append(("教师基线", model, years, None))
    for model, years in data["student_pre"].items():
        rows.append(("学生 蒸馏前", model, years, None))
    for model, years in data["student_post"].items():
        rows.append(("学生 蒸馏后", model, years, None))

    if not rows:
        print("（无结果。先跑 bash run_benchmark.sh teacher / student_baseline，蒸馏后 watch_student.sh）")
        return

    hdr = f"{'阶段':<12}{'模型':<18}{'AIME24':<16}{'AIME25':<16}{'ΔAIME24':<10}{'ΔAIME25':<10}"
    print(hdr)
    print("-" * 78)

    # 每个组的 post 与其 pre 做 Δ
    pre_map = dict(data["student_pre"].items())
    for stage, model, years, _ in rows:
        if stage != "学生 蒸馏后":
            print(f"{stage:<12}{model:<18}{_fmt(years.get('AIME24'))}{_fmt(years.get('AIME25'))}")
            continue
        # post 行：找同组 pre 算 Δ
        base = model.split("@")[0]
        pre_years = pre_map.get(base, {})
        d24 = d25 = ""
        if "AIME24" in years and "AIME24" in pre_years:
            d24 = f"{years['AIME24'][0] - pre_years['AIME24'][0]:+.2f}"
        if "AIME25" in years and "AIME25" in pre_years:
            d25 = f"{years['AIME25'][0] - pre_years['AIME25'][0]:+.2f}"
        print(f"{stage:<12}{model:<18}{_fmt(years.get('AIME24'))}{_fmt(years.get('AIME25'))}{d24:<10}{d25:<10}")

    print("-" * 78)
    print("注：Δ = 蒸馏后 - 蒸馏前（弱到强蒸馏收益）。格式 准确率%(对题/总题)。")


if __name__ == "__main__":
    main()
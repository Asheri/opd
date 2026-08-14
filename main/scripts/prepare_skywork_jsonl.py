#!/usr/bin/env python3
"""Skywork DAPO parquet -> jsonl + 10K 子集采样（阶段 0.3）。

把 Direct-OPD/scripts/prepare_skywork_math.py 产出的 DAPO parquet（105,055 行）
转为 JsonLinesDataLoader 可读的 jsonl，并随机采样 10K 子集。

输入 parquet schema（prepare_skywork_math.py 输出）：
  - prompt: list[struct{content, role}]  （单条 user 消息，content 已是 DAPO 模板文本）
  - reward_model: struct{ground_truth, style}

输出 jsonl 每行：
  {"prompt": <DAPO 文本>, "response": "", "ground_truth": <答案>}
  response 暂空--由 prepare_skywork_responses.py（阶段 0.4）用初始 student 生成后回填。

用法：
  python prepare_skywork_jsonl.py \
    --input /root/autodl-tmp/datasets/skywork-or1-math-dapo.parquet \
    --output /root/autodl-tmp/datasets/skywork_10k.jsonl \
    --n 10000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="DAPO parquet（prepare_skywork_math.py 输出）")
    p.add_argument("--output", required=True, type=Path, help="输出 jsonl 路径")
    p.add_argument("--n", type=int, default=10000, help="子集大小（默认 10000）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"输入 parquet 不存在: {args.input}")

    table = pq.read_table(args.input)
    n_total = table.num_rows
    print(f"读入 {n_total:,} 行（{args.input}）")

    # 提取 prompt 文本 + ground_truth
    rows = []
    bad = 0
    for i, row in enumerate(table.to_pylist()):
        prompt_msgs = row.get("prompt")
        reward_model = row.get("reward_model")
        # prompt 必须是单条 user 消息
        if (not isinstance(prompt_msgs, list) or len(prompt_msgs) != 1
                or not isinstance(prompt_msgs[0], dict)
                or prompt_msgs[0].get("role") != "user"):
            bad += 1
            continue
        content = prompt_msgs[0].get("content")
        gt = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if not content or gt is None:
            bad += 1
            continue
        rows.append({"prompt": content, "response": "", "ground_truth": str(gt)})
    if bad:
        print(f"⚠️ 跳过 {bad} 行格式异常（prompt/ground_truth 缺失或结构不符）")
    print(f"有效 {len(rows):,} 行")

    # 随机采样子集
    n = min(args.n, len(rows))
    rng = random.Random(args.seed)
    subset = rng.sample(rows, n)
    print(f"采样 {n:,} 条子集（seed={args.seed}）")

    # 写 jsonl
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 校验
    written = sum(1 for _ in open(args.output, encoding="utf-8"))
    print(f"✅ 写入 {written:,} 行 -> {args.output}")
    # 抽样打印第一条
    first = json.loads(open(args.output, encoding="utf-8").readline())
    print(f"样例 prompt 前 120 字符: {first['prompt'][:120]!r}")
    print(f"样例 ground_truth: {first['ground_truth']!r}")


if __name__ == "__main__":
    main()

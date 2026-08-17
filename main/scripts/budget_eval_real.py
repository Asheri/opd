#!/usr/bin/env python3
"""IMP-4 双卡 Budget-Aware 评估脚本（真实 GPU）。

在统一 reasoning budget 下公平比较 Base/L0/L2 模型，产出逐样本 jsonl + 聚合结果，
供 budget_curve 指标与 write_report 生成决策报告（Q2/Q4）。双卡并行：每进程一个
--device，models 子集分到不同卡（如 cuda:0=Base+E1、cuda:1=E2），并行评估。

用法（双卡并行）：
  python budget_eval_real.py --device cuda:0 --models "Base=/path/Qwen3-1.7B,E1=/path/e1" \
      --budgets 256,512,1024 --dataset MATH500 --n-limit 100 --out-dir <dir_a>
  python budget_eval_real.py --device cuda:1 --models "E2=/path/e2" \
      --budgets 256,512,1024 --dataset MATH500 --n-limit 100 --out-dir <dir_b>

复用 BudgetEvaluator（预算感知逐位 EOS 判定 + Accuracy@B/PrefixAccuracy@B + token 记账），
加 n_limit 快速子集；不改评估内核、不改训练。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.budget_eval import BudgetEvaluator, DEFAULT_BUDGETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--models", required=True, help="Label=path[,Label2=path2...]")
    p.add_argument("--budgets", default="256,512,1024", help="逗号分隔 reasoning budget")
    p.add_argument("--dataset", default="MATH500")
    p.add_argument("--n-limit", type=int, default=None, help="只评估前 N 条（快速子集）")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--scoring", default="sympy")
    p.add_argument("--prompt-style", default="boxed")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn", default=None, help="attention 实现（flash_attention_2 加速长序列生成）")
    return p.parse_args()


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    models = [(lab, path) for lab, path in
              (kv.split("=", 1) for kv in args.models.split(",") if "=" in kv)]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[budget-eval-real] device={args.device} models={[l for l,_ in models]} "
          f"budgets={budgets} dataset={args.dataset} n_limit={args.n_limit}", flush=True)
    all_results = []
    for label, path in models:
        if not path or not os.path.isdir(path):
            print(f"[budget-eval-real] {label}: 路径无效/缺失，跳过: {path}", flush=True)
            continue
        with BudgetEvaluator(
                path, device=args.device, batch_size=args.batch_size,
                temperature=args.temperature, scoring=args.scoring,
                prompt_style=args.prompt_style, dtype=args.dtype,
                attn_implementation=args.attn) as ev:
            for B in budgets:
                res = ev.evaluate_budget(args.dataset, B, n_limit=args.n_limit)
                res["label"] = label
                all_results.append(res)
                rows = res.pop("rows", [])
                out_path = os.path.join(args.out_dir, f"{label}__{args.dataset}__B{B}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                summary = {k: v for k, v in res.items()}
                print(f"  {label} @B{B}: acc={res['accuracy']:.3f} prefix={res['prefix_accuracy']} "
                      f"eos={res['eos_rate']:.3f} budget_stop={res['budget_stop_rate']:.3f} "
                      f"avg_rt={res['avg_reasoning_tokens']:.0f} n={res['n']}", flush=True)
    # 聚合落盘（供 budget_curve / write_report 消费）
    agg_path = os.path.join(args.out_dir, "all_results.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[budget-eval-real] 聚合 {len(all_results)} 条 -> {agg_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

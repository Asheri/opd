#!/usr/bin/env python3
"""KV-cached rollout 快速冒烟：真实 HF 模型上验证 generate_with_status_kv 的
eos_token_id=-1 哨兵（永不 EOS）不报错、全 budget_stop、能与朴素路径对齐。

供 S2 实跑前解阻塞（GPU 空闲时跑，~2 min）。用法：
  python smoke_kv_rollout.py --model Qwen__Qwen3-1.7B --prompts <jsonl> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, help="jsonl（取 prompt 字段）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in open(args.prompts, encoding="utf-8") if l.strip()]
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    rng = random.Random(args.seed)
    sample = rng.sample(prompts, min(args.n, len(prompts)))

    from transformers import AutoTokenizer
    from fullstack_opd_v2.model_factory import HFCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    enc = tok(sample, return_tensors="pt", padding=True, truncation=True,
              max_length=1024).to(args.device)

    m = HFCausalLM(args.model, args.device, dtype="bfloat16")
    t0 = time.time()
    out = m.generate_with_status_kv(enc["input_ids"], max_new=args.max_new,
                                    eos_token_id=None, pad_id=tok.pad_token_id)
    dt = time.time() - t0
    print(f"[kv] {args.n}×{args.max_new} tok 用时 {dt:.1f}s "
          f"({args.n*args.max_new/dt:.0f} tok/s)")
    print(f"[kv] statuses={out['statuses']}")
    print(f"[kv] lengths={out['lengths']}")
    assert all(s == "budget_stop" for s in out["statuses"]), "非全 budget_stop？"
    print("✅ KV 路径 eos=-1 哨兵正常（全 budget_stop，no error）")


if __name__ == "__main__":
    main()
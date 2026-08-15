#!/usr/bin/env python3
"""Stage 2 校准：真实 HF 模型 eos_token_id + loop_periods（任务 2）。

在真实 HF 模型（默认 Qwen3-1.7B）上做短 rollout，产出三项真实数值：
  1. tokenizer.eos_token_id —— l2.rollout.eos_token_id 取值。
  2. 尾部周期自相关：对每条 rollout 的【新生成 token 序列】尾部，按
     detect_loop 语义（末 p 段 == 倒数第二 p 段）统计各周期 p∈{2..8}
     的命中率 —— l2.rollout.loop_periods 取值依据。
  3. 状态分布：eos_token_id=None（永不判 EOS）下 budget_stop/loop 占比
     （loop 由尾部周期性判定，证明真实模型是否退化出循环尾部）。

用法：
  python calibrate_rollout.py --model <HF模型> --jsonl <prompt jsonl> \
      --device cuda:0 --n 32 --max-new 512 [--eos-id <int>]

默认 eos-id=None → 采样时不让 HF 停 EOS（do_sample + max_new 截断），观察纯预算
行为；若显式给 --eos-id，则同时开出 eos 状态占比。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="HF 模型路径")
    p.add_argument("--jsonl", required=True, type=Path, help="prompt jsonl（取 prompt 字段）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=32, help="采样条数")
    p.add_argument("--max-new", type=int, default=512, help="每 rollout 生成上限")
    p.add_argument("--eos-id", type=int, default=None, help="显式 eos 采样（None=不判 EOS）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tail_is_loop(seq: list[int], p: int, min_len: int = 16) -> bool:
    """detect_loop 语义：有效长 >= max(2p, min_len) 且末 p 段 == 倒数第二 p 段。"""
    L = len(seq)
    return L >= 2 * p and L >= min_len and seq[-p:] == seq[-2 * p:-p]


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    if not prompts:
        raise SystemExit("jsonl 无 prompt")
    rng = random.Random(args.seed)
    sample = rng.sample(prompts, min(args.n, len(prompts)))
    print(f"采样 {len(sample)} 条 prompt（共 {len(prompts)}）", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model))
    eos = args.eos_id if args.eos_id is not None else tok.eos_token_id
    print(f"tokenizer.eos_token_id = {tok.eos_token_id!r} "
          f"(eos_token={tok.eos_token!r})；本采样 eos-id = {eos!r}", flush=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"加载模型 {args.model} (flash_attention_2, bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()

    all_new: list[list[int]] = []
    t0 = time.time()
    for start in range(0, len(sample), args.batch_size):
        bs = sample[start:start + args.batch_size]
        enc = tok(bs, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(args.device)
        seq_len = enc["input_ids"].size(1)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True,
                temperature=1.0, top_p=0.95, num_return_sequences=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos)          # eos=None → HF 用模型默认 eos（会自然停）
        for j in range(out.size(0)):
            new = out[j][seq_len:].tolist()
            # 去 pad（budget 撞满时尾部是 pad_token）
            while new and new[-1] == tok.pad_token_id:
                new.pop()
            all_new.append(new)
        print(f"  {min(start+args.batch_size, len(sample))}/{len(sample)} "
              f"完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 分析 ----
    lens = [len(x) for x in all_new]
    print(f"\n新生成 token 长度：min={min(lens)} max={max(lens)} "
          f"E[L]={sum(lens)/len(lens):.0f}", flush=True)

    # eos 命中：new 序列是否含 tok.eos_token_id（若 eos=None 用模型默认仍在序列里）
    eos_tok = tok.eos_token_id
    n_eos = sum(1 for x in all_new if eos_tok in x)
    print(f"new 序列含 tokenizer EOS({eos_tok}) 条数：{n_eos}/{len(all_new)}", flush=True)

    # 尾部周期自相关（p=2..8）
    print("\n尾部周期自相关（detect_loop 语义，min_len=16）：")
    print(f"{'p':>3} | {'loop 命中率':>10} | {'命中/总数':>8}")
    for p in range(2, 9):
        hit = sum(1 for x in all_new if tail_is_loop(x, p))
        print(f"{p:>3} | {hit/len(all_new):>10.3f} | {hit:>4}/{len(all_new)}", flush=True)

    # 建议 loop_periods：命中率 > 5% 的周期
    periods = [p for p in range(2, 9)
               if sum(1 for x in all_new if tail_is_loop(x, p)) / len(all_new) > 0.05]
    print(f"\n建议 loop_periods = {tuple(periods)}（命中率>5% 的周期）", flush=True)
    print(f"建议 eos_token_id = {eos}（l2.rollout.eos_token_id）", flush=True)


if __name__ == "__main__":
    main()
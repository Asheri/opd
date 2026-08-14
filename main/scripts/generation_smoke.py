#!/usr/bin/env python3
"""Stage 0 最终 smoke test：并行生成 N 条真实 response（cap=4096），精确测量长度特性。

重点不是"能不能生成"，而是确认 max_new_tokens=4096 下真实数据的：
  E[L]        —— 平均 response token 长度（总体 & 仅 EOS 终止的条件均值）
  P(L=4096)   —— 撞 cap 截断的比例（= 非 EOS 终止）
  EOS rate    —— EOS 自然终止比例（= 1 − P(L=4096)）
  tok/s       —— 生成吞吐（聚合，含 batch 并行）

协议（对齐数据生成）：Qwen3-1.7B 初始权重，temperature=1.0, top_p=0.95, do_sample=True,
flash_attention_2, bf16。逐位置用 transformers 原生 generate 精确判 EOS：
  - 新 token 序列含 eos_token_id → EOS 终止，长度=eos 位置
  - 否则长度=max_new（截断）
response 写回 jsonl（--out），供后续 build 复用。支持 --shard-rank/--num-shards 双卡并行。

用法（双卡并行，500 条 → 每卡 250）：
  卡0: python scripts/generation_smoke.py --jsonl skywork_smoke_in.jsonl \
        --model .../Qwen3-1.7B --device cuda:0 --max-new-tokens 4096 \
        --shard-rank 0 --num-shards 2 --out smoke_shard0.jsonl --report smoke_shard0.json
  卡1: 同上 --device cuda:1 --shard-rank 1 --out smoke_shard1.jsonl --report smoke_shard1.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, help="输入 jsonl（response 空）")
    p.add_argument("--model", required=True, help="生成模型（Qwen3-1.7B）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-rank", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out", required=True, help="生成的 jsonl")
    p.add_argument("--report", required=True, help="指标 JSON 输出")
    return p.parse_args()


def load_rows(path, max_samples):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(args.jsonl, args.max_samples)
    # 按位置分片（shard-rank/num-shards）
    todo = [i for i in range(len(rows)) if i % args.num_shards == args.shard_rank]
    print(f"[INFO] 共 {len(rows)} 行；本 shard({args.shard_rank}/{args.num_shards}) "
          f"待生成 {len(todo)} 行，max_new_tokens={args.max_new_tokens}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 批量生成必须左填充，否则 pad 污染 hidden state
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()
    eos = tok.eos_token_id
    print(f"[INFO] 模型加载完成 EOS={eos}", flush=True)

    lengths, eos_flags = [], []
    total_tokens = 0
    t0 = time.perf_counter()
    out_rows = []
    for start in range(0, len(todo), args.batch_size):
        idxs = todo[start:start + args.batch_size]
        batch = [rows[i]["prompt"] for i in idxs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_prompt_len)
        input_ids = enc.input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-6),
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id, eos_token_id=eos)
        new = out[:, input_ids.size(1):]                 # (b, n_new)
        total_tokens += new.numel()
        for j, row_ in enumerate(new.tolist()):
            if eos in row_:
                eos_flags.append(True)
                lengths.append(row_.index(eos))          # 不含 eos 的 response 长度
            else:
                eos_flags.append(False)
                lengths.append(len(row_))                # == max_new（截断）
            # response 文本 = 去掉 eos 后的 token 序列解码
            resp_tokens = row_[:row_.index(eos)] if eos in row_ else row_
            resp_text = tok.decode(resp_tokens, skip_special_tokens=True)
            r = dict(rows[idxs[j]])
            r["response"] = resp_text
            out_rows.append(r)
        done = min(start + args.batch_size, len(todo))
        el = sum(lengths[-args.batch_size:]) / len(lengths[-args.batch_size:])
        trunc = [l for l, f in zip(lengths, eos_flags) if not f]
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(todo)} 完成 | "
              f"本批 {new.numel()} tok / {time.perf_counter()-t0:.0f}s | "
              f"E[L]={el:.0f} P(L={args.max_new_tokens})={len(trunc)}", flush=True)

    wall = time.perf_counter() - t0
    tok_s = total_tokens / wall
    n = len(lengths)
    eos_n = sum(eos_flags)
    trunc_n = n - eos_n
    eos_lens = [l for l, f in zip(lengths, eos_flags) if f]
    report = {
        "n": n, "max_new_tokens": args.max_new_tokens,
        "E[L]_overall": round(sum(lengths) / n, 1),
        "E[L]_eos_only": round(sum(eos_lens) / len(eos_lens), 1) if eos_lens else None,
        "P(L=max)_truncated": round(trunc_n / n, 4),
        "EOS_rate": round(eos_n / n, 4),
        "tok_per_s": round(tok_s, 1),
        "wall_s": round(wall, 1),
        "total_tokens": int(total_tokens),
        "max_len_observed": max(lengths),
        "min_len_observed": min(lengths),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] response 写至 {args.out}")
    print(f"[OK] 指标写至 {args.report}")
    print(f"[REPORT] n={n} E[L]={report['E[L]_overall']} "
          f"(eos-only={report['E[L]_eos_only']}) "
          f"P(L={args.max_new_tokens})={report['P(L=max)_truncated']} "
          f"EOS_rate={report['EOS_rate']} tok/s={report['tok_per_s']} "
          f"wall={report['wall_s']}s")


if __name__ == "__main__":
    main()
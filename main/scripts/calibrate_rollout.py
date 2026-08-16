#!/usr/bin/env python3
"""Stage 2 校准：真实 HF 模型 eos_token_id + loop_periods（任务 2 / IMP-1b）。

在真实 HF 模型（默认 Qwen3-1.7B）上做短 rollout，产出三项真实数值：
  1. tokenizer.eos_token_id —— l2.rollout.eos_token_id 取值。
  2. 尾部周期自相关：对每条 rollout 的【新生成 token 序列】尾部，按
     detect_loop 语义（末 p 段 == 倒数第二 p 段）统计各周期 p∈{2..8}
     的命中率 —— l2.rollout.loop_periods 取值依据。
  3. 状态分布：eos_token_id=None（永不判 EOS）下 budget_stop/loop 占比
     （loop 由尾部周期性判定，证明真实模型是否退化出循环尾部）。

产出双通道：
  - stdout：人读报告（现有行为，不回退）；
  - --output <yaml>：把建议的 l2.rollout 配置（loop_periods + eos_token_id）
    写成 YAML，可直接作为 configs/*.yaml 的 l2.rollout 段覆盖（IMP-1b）。

用法：
  python calibrate_rollout.py --model <HF模型> --jsonl <prompt jsonl> \
      --device cuda:0 --n 32 --max-new 512 [--eos-id <int>] \
      [--output <建议配置.yaml>]

默认 eos-id=None → 采样时不让 HF 停 EOS（do_sample + max_new 截断），观察纯预算
行为；若显式给 --eos-id，则同时开出 eos 状态占比。

本脚本核心分析（tail_is_loop / analyze_rollouts / write_yaml）为纯函数，可在无
GPU / 无 HF 模型 / 无联网环境下直接单测；HF 模型加载只发生在 main() 内。
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
    p.add_argument("--output", type=Path, default=None,
                   help="把建议的 l2.rollout 配置（loop_periods/eos_token_id）写成 YAML（IMP-1b）")
    return p.parse_args()


def tail_is_loop(seq: list[int], p: int, min_len: int = 16) -> bool:
    """detect_loop 语义：有效长 >= max(2p, min_len) 且末 p 段 == 倒数第二 p 段。"""
    L = len(seq)
    return L >= 2 * p and L >= min_len and seq[-p:] == seq[-2 * p:-p]


def analyze_rollouts(all_new: list[list[int]], eos_tok: int,
                     eos_used: int | None, min_len: int = 16) -> dict:
    """纯函数：从真实 rollout 的新生成 token 序列统计 loop 周期与 eos 命中。

    all_new: list[list[int]]，每条为一条 rollout 的新生成 token 序列（已去 pad）。
    eos_tok : tokenizer.eos_token_id（统计序列里是否含该 id）。
    eos_used: 本次采样实际用的 eos_token_id（None=未判 EOS）。
    min_len : tail_is_loop 的最小有效长度（与 detect_loop 语义对齐）。

    返回报告 dict（供 stdout 打印与 write_yaml 落盘）：
      n, lens_min, lens_max, lens_mean, n_eos, loop_rate_by_period,
      suggested_loop_periods, suggested_eos_token_id, min_len。
    """
    n = len(all_new)
    lens = [len(x) for x in all_new]
    n_eos = sum(1 for x in all_new if eos_tok in x)
    rates = {}
    for p in range(2, 9):
        hit = sum(1 for x in all_new if tail_is_loop(x, p, min_len=min_len))
        rates[p] = hit / n if n else 0.0
    # 建议 loop_periods：命中率 > 5% 的周期（IMP-1b，替代原硬编码 (2,3,4)）
    periods = [p for p in range(2, 9) if rates[p] > 0.05]
    return {
        "n": n,
        "lens_min": min(lens) if lens else 0,
        "lens_max": max(lens) if lens else 0,
        "lens_mean": sum(lens) / len(lens) if lens else 0.0,
        "n_eos": n_eos,
        "eos_tok": eos_tok,
        "loop_rate_by_period": rates,
        "suggested_loop_periods": tuple(periods),
        "suggested_eos_token_id": eos_used,
        "min_len": min_len,
    }


def write_yaml(report: dict, out_path: Path | str) -> str:
    """把建议的 l2.rollout 配置写成 YAML 片段（可直接并入 configs/*.yaml）。返回内容。"""
    import yaml
    out_path = Path(out_path)   # 兼容 str / Path 调用（测试与 CLI 两用）
    body = {
        "#": "scripts/calibrate_rollout.py 校准产物——建议覆盖 l2.rollout（IMP-1b）",
        "l2": {
            "rollout": {
                "loop_periods": list(report["suggested_loop_periods"]),
                "eos_token_id": report["suggested_eos_token_id"],
            }
        },
    }
    text = yaml.safe_dump(body, allow_unicode=True, sort_keys=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl 不存在：{args.jsonl}（请给真实 prompt jsonl）")
    if not args.model.exists():
        raise SystemExit(f"HF 模型路径不存在：{args.model}")
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

    # ---- 分析（纯函数，可单测）----
    report = analyze_rollouts(all_new, eos_tok=tok.eos_token_id, eos_used=eos)
    rates = report["loop_rate_by_period"]
    print(f"\n新生成 token 长度：min={report['lens_min']} max={report['lens_max']} "
          f"E[L]={report['lens_mean']:.0f}", flush=True)

    print(f"new 序列含 tokenizer EOS({report['eos_tok']}) 条数："
          f"{report['n_eos']}/{report['n']}", flush=True)

    # 尾部周期自相关（p=2..8）
    print("\n尾部周期自相关（detect_loop 语义，min_len=16）：")
    print(f"{'p':>3} | {'loop 命中率':>10} | {'命中/总数':>8}")
    for p in range(2, 9):
        print(f"{p:>3} | {rates[p]:>10.3f} | "
              f"{int(rates[p] * report['n']):>4}/{report['n']}", flush=True)

    periods = report["suggested_loop_periods"]
    print(f"\n建议 loop_periods = {periods}（命中率>5% 的周期）", flush=True)
    print(f"建议 eos_token_id = {report['suggested_eos_token_id']}（l2.rollout.eos_token_id）", flush=True)

    # IMP-1b：可选 YAML 落盘（不只 stdout）
    if args.output is not None:
        text = write_yaml(report, args.output)
        print(f"\n✅ 建议配置写入 {args.output}：", flush=True)
        print(text, end="")


if __name__ == "__main__":
    main()

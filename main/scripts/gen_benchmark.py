#!/usr/bin/env python3
"""Stage 0 generation benchmark：实测吞吐 + 长度分布，消除 50K×8192 时间墙的理论不确定性。

目的（docs《Stage 0 提示词》§1/§2）：不盲信理论估算，在真实 GPU 上实测
  - 不同 (N, max_new_tokens, batch_size) 组合下的真实 decode 吞吐（tok/s）
  - 真实 response 长度分布（E[L]、P(L>2048)、P(L>4096)、P(L=8192)）
  - 成本只用 T_actual = Σ L_i（非 N × max_new_tokens）

CLI：
  # 跑矩阵（多组）
  python scripts/gen_benchmark.py --model <Qwen3-1.7B> \
    --prompts /root/autodl-tmp/datasets/skywork_50k.jsonl \
    --matrix "32,2048,1 32,4096,2 128,2048,2 128,8192,2 512,8192,2" \
    --max-time 120 --out /root/autodl-tmp/eval/gen_benchmark.json

  # 单组直跑
  python scripts/gen_benchmark.py --model <Qwen3-1.7B> \
    --prompts /root/autodl-tmp/datasets/skywork_50k.jsonl \
    --max-samples 128 --max-new-tokens 8192 --batch-size 2 --max-time 120

策略：长序列组（max_new=8192）用 --max-time 限时测真实 decode tok/s（TimeLimitCriteria）；
     短序列组（max_new≤2048）可 --max-time 0 完整生成测真实长度分布。
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import StoppingCriteria, StoppingCriteriaList


# ============================================================================
# 统计聚合（纯函数，可单测，不依赖模型）
# ============================================================================

def aggregate_stats(samples: list[dict]) -> dict:
    """samples: [{prompt_len, gen_len, ended_eos, truncated, wall_s}, ...]
    返回：T_actual=ΣL_i、mean/p50/p90/p95/max_len、eos_rate、truncation_rate、
          E[L]、P(L>2048)、P(L>4096)、P(L=8192)、tok_per_s、samples_per_s、GPU mem peak。
    成本口径 T_actual=ΣL_i（非 N×max_new_tokens）。
    """
    n = len(samples)
    if n == 0:
        return {"prompt_count": 0, "generated_tokens": 0, "mean_len": 0.0,
                "p50_len": 0.0, "p90_len": 0.0, "p95_len": 0.0, "max_len": 0,
                "eos_rate": 0.0, "truncation_rate": 0.0,
                "P_L_eq_8192": 0.0, "P_L_gt_2048": 0.0, "P_L_gt_4096": 0.0,
                "tok_per_s": 0.0, "samples_per_s": 0.0,
                "wall_time_s": 0.0, "gpu_mem_peak_gb": 0.0}
    lens = [s["gen_len"] for s in samples]
    total_tokens = sum(lens)                       # T_actual = ΣL_i
    total_wall = sum(s["wall_s"] for s in samples)  # 逐样本 wall 累加（非并发的组墙钟）
    eos = sum(1 for s in samples if s.get("ended_eos"))
    trunc = sum(1 for s in samples if s.get("truncated"))
    mem_peak = max((s.get("gpu_mem_peak_gb", 0.0) for s in samples), default=0.0)
    return {
        "prompt_count": n,
        "generated_tokens": total_tokens,          # T_actual（决策唯一权威口径）
        "mean_len": total_tokens / n,              # E[L]
        "p50_len": statistics.median(lens),
        "p90_len": statistics.quantiles(lens, n=10)[-1],
        "p95_len": statistics.quantiles(lens, n=20)[-1],
        "max_len": max(lens),
        "eos_rate": eos / n,
        "truncation_rate": trunc / n,
        "P_L_eq_8192": sum(1 for L in lens if L == 8192) / n,   # 截断到上限的比例
        "P_L_gt_2048": sum(1 for L in lens if L > 2048) / n,
        "P_L_gt_4096": sum(1 for L in lens if L > 4096) / n,
        "tok_per_s": total_tokens / total_wall if total_wall > 0 else 0.0,
        "samples_per_s": n / total_wall if total_wall > 0 else 0.0,
        "wall_time_s": total_wall,
        "gpu_mem_peak_gb": mem_peak,
    }


# ============================================================================
# 限时生成（兜底防长序列跑飞，复用 timing_flash_10min.py 模式）
# ============================================================================

class TimeLimitCriteria(StoppingCriteria):
    """到时间上限即停（每步检查，最多多生成 1 token）。"""
    def __init__(self, limit_s):
        self.t0 = time.time()
        self.limit = limit_s

    def __call__(self, input_ids, scores, **kwargs):
        return (time.time() - self.t0) >= self.limit


# ============================================================================
# 单组 benchmark
# ============================================================================

def run_one(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int,
            max_time_s: float, device: str, seed: int) -> dict:
    """跑一组 (N=max_new_tokens, batch_size) 的 benchmark，返回 aggregate_stats。

    参数：
      max_time_s: 该组墙钟上限（long 序列用限时测真实 tok/s；0=不限时完整生成）
      prompts:    待生成样本（已取好 max_samples 前 N 条）
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    samples: list[dict] = []
    group_t0 = time.time()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(device)
        seq_len = enc["input_ids"].size(1)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=1.0, top_p=0.95, num_return_sequences=1,
                pad_token_id=tok.pad_token_id, generator=g,
                stopping_criteria=StoppingCriteriaList([TimeLimitCriteria(max_time_s)])
                if max_time_s > 0 else None)
        dt = time.time() - t0
        gen_lens = out.shape[1] - seq_len
        per_sample_wall = dt / len(batch)   # batch 内摊还，使 Σwall_s ≈ 真实组墙钟
        for j in range(len(batch)):
            L = gen_lens
            samples.append({
                "prompt_len": seq_len,
                "gen_len": L,
                "ended_eos": L < max_new_tokens,   # 比上限短 => 自然 EOS（或时间截断）
                "truncated": L >= max_new_tokens,  # 长度到上限 => 截断
                "wall_s": per_sample_wall,
                "gpu_mem_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
            })
        # 限时组：已达墙钟上限就停（避免整组超时太多）
        if max_time_s > 0 and (time.time() - group_t0) >= max_time_s:
            break
    return aggregate_stats(samples)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="HF 模型路径（如 Qwen3-1.7B）")
    p.add_argument("--prompts", required=True, type=Path, help="prompt jsonl（取 prompt 字段）")
    p.add_argument("--prompt-key", default="prompt", help="jsonl 中 prompt 字段名")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-time", type=float, default=120.0,
                   help="每组墙钟上限秒（0=不限时完整生成）")
    p.add_argument("--out", type=Path, default=Path("gen_benchmark.json"),
                   help="输出 JSON 路径（同目录另写 gen_benchmark.csv 汇总）")
    # 单组直跑
    p.add_argument("--max-samples", type=int, default=None, help="单组：取前 N 条 prompt")
    p.add_argument("--max-new-tokens", type=int, default=None, help="单组：生成长度")
    p.add_argument("--batch-size", type=int, default=None, help="单组：batch")
    # 矩阵："N,MAX_NEW,BATCH  N,MAX_NEW,BATCH ..."
    p.add_argument("--matrix", type=str, default=None,
                   help="矩阵：'32,2048,1 32,4096,2 128,2048,2 ...'（空格分隔三元组）")
    return p.parse_args()


def load_prompts(path: Path, key: str, max_samples: int | None) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(str(json.loads(line)[key]))
            except Exception:
                continue
    if max_samples is not None:
        prompts = prompts[:max_samples]
    return prompts


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"模型路径不存在: {args.model}")
    if not args.prompts.is_file():
        raise SystemExit(f"prompts jsonl 不存在: {args.prompts}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{time.strftime('%H:%M:%S')}] 加载模型 {args.model} (flash_attention_2, bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(str(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 解析跑哪些组
    if args.matrix:
        groups = []
        for tok_spec in args.matrix.split():
            n, mn, bs = (int(x) for x in tok_spec.split(","))
            groups.append((n, mn, bs))
    elif args.max_new_tokens is not None:
        groups = [(args.max_samples, args.max_new_tokens,
                   args.batch_size if args.batch_size else 1)]
    else:
        raise SystemExit("必须给 --matrix 或 --max-new-tokens（单组）")

    results = {}
    for n, max_new, bs in groups:
        prompts = load_prompts(args.prompts, args.prompt_key, n)
        print(f"[{time.strftime('%H:%M:%S')}] 组 (N={len(prompts)}, max_new={max_new}, bs={bs}) 开始...", flush=True)
        st = run_one(model, tok, prompts, max_new, bs, args.max_time,
                     args.device, args.seed)
        st["n_requested"] = n
        st["max_new_tokens"] = max_new
        st["batch_size"] = bs
        key = f"N{len(prompts)}_L{max_new}_bs{bs}"
        results[key] = st
        print(f"  -> {st['prompt_count']} 条 / {st['generated_tokens']} tok / "
              f"E[L]={st['mean_len']:.0f} / {st['tok_per_s']:.1f} tok/s / "
              f"P(L>2048)={st['P_L_gt_2048']:.2f} / P(L=8192)={st['P_L_eq_8192']:.2f}", flush=True)

    # 输出 JSON + CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    csv_path = args.out.with_name("gen_benchmark.csv")
    cols = ["group", "prompt_count", "generated_tokens", "mean_len", "p50_len",
            "p90_len", "p95_len", "max_len", "eos_rate", "truncation_rate",
            "P_L_gt_2048", "P_L_gt_4096", "P_L_eq_8192", "tok_per_s",
            "samples_per_s", "wall_time_s", "gpu_mem_peak_gb"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k, st in results.items():
            row = {"group": k}
            for c in cols[1:]:
                row[c] = st.get(c, "")
            w.writerow(row)
    print(f"✅ 结果已存 {args.out}（汇总 CSV: {csv_path}）", flush=True)


if __name__ == "__main__":
    main()
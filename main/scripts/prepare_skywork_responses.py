#!/usr/bin/env python3
"""Skywork response 预生成（阶段 0.4）--绕过 HFCausalLM 无 generate_batch 的限制。

问题：JsonLinesDataLoader 要求 prompt+response 都非空，但 Skywork 原始只有 prompt+gt。
      而 pipeline 内部 warmup（Stage1 student_init）用 generate_batch，HFCausalLM 没有此方法
      -> HF 路径下无法用 pipeline 内部机制生成 response。
解法：本脚本用 transformers 原生 generate（非 pipeline 内部），给每条 prompt 用
      **初始 Qwen3-1.7B**（未训练）生成 1 条 on-policy response，回填 jsonl。

协议（对齐论文 warmup 语义）：
  - max_new_tokens=2048（= 论文 MAX_RESP_LENGTH，训练 response 长度）
  - temperature=1.0、do_sample=True（on-policy 初始分布采样）
  - 启用 flash_attention_2（长生成提速）

容错：逐题生成 + 即时落盘 + resume（中断后重跑跳过已完成的题）。
      用临时输出文件 progressive 追加，完成后原子替换原 jsonl。

用法：
  python prepare_skywork_responses.py \
    --jsonl /root/autodl-tmp/datasets/skywork_10k.jsonl \
    --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
    --device cuda:0 --batch-size 8 --max-new-tokens 2048 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, type=Path, help="输入 jsonl（prompt 已填、response 空）")
    p.add_argument("--model", required=True, type=Path, help="初始 student HF 模型路径（如 Qwen3-1.7B）")
    p.add_argument("--device", default="cuda:0", help="设备（默认 cuda:0）")
    p.add_argument("--batch-size", type=int, default=8, help="生成 batch（控显存）")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="生成长度（=论文 MAX_RESP_LENGTH）")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度（on-policy 初始分布）")
    p.add_argument("--top-p", type=float, default=0.95, help="top-p 采样")
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({})  # 损坏行占位
    return rows


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl 不存在: {args.jsonl}")

    rows = load_rows(args.jsonl)
    n = len(rows)
    # 待生成：response 为空/缺失的行
    todo = [i for i, r in enumerate(rows) if not r.get("response")]
    print(f"[{time.strftime('%H:%M:%S')}] 共 {n} 行，待生成 response {len(todo)} 行", flush=True)
    if not todo:
        print("所有 response 已生成，无需处理")
        return

    # 临时输出文件（progressive 追加），完成后原子替换原 jsonl
    tmp_path = args.jsonl.with_suffix(".jsonl.tmp")
    # resume：tmp 已有的行 index 集合（按原 index 标记）
    done: set[int] = set()
    if tmp_path.is_file():
        for line in open(tmp_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r.get("_idx"), int):
                    done.add(r["_idx"])
            except Exception:
                continue
        print(f"resume：tmp 已完成 {len(done)} 行，跳过", flush=True)

    # 加载模型（flash_attn）
    print(f"[{time.strftime('%H:%M:%S')}] 加载模型 {args.model} (flash_attention_2, bf16)...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(str(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 批量生成必须左填充

    # 逐批生成 + 追加写 tmp
    remaining = [i for i in todo if i not in done]
    bs = args.batch_size
    with open(tmp_path, "a", encoding="utf-8") as f:
        for start in range(0, len(remaining), bs):
            batch_idx = remaining[start:start + bs]
            batch_prompts = [rows[i]["prompt"] for i in batch_idx]
            enc = tok(batch_prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(args.device)
            seq_len = enc["input_ids"].size(1)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=1, pad_token_id=tok.pad_token_id)
            dt = time.time() - t0
            gen_tokens = out.shape[1] - seq_len
            # 逐题写回（带 _idx 供 resume）
            for j, idx in enumerate(batch_idx):
                resp = tok.decode(out[j][seq_len:], skip_special_tokens=True)
                rows[idx]["response"] = resp
                # tmp 行：带 _idx 标记 + 完整 row（去 _idx 后即最终行）
                row_with_idx = dict(rows[idx])
                row_with_idx["_idx"] = idx
                f.write(json.dumps(row_with_idx, ensure_ascii=False) + "\n")
            f.flush()
            done_now = len(done) + start + len(batch_idx)
            print(f"[{time.strftime('%H:%M:%S')}] {done_now}/{len(todo)} 完成 "
                  f"(本批 {gen_tokens} tok / {dt:.1f}s = {gen_tokens/dt:.0f} tok/s)", flush=True)

    # 全部完成：从 tmp 重建有序 jsonl（按原顺序，去 _idx），原子替换
    print(f"[{time.strftime('%H:%M:%S')}] 重建有序 jsonl 并替换原文件...", flush=True)
    by_idx = {}
    for line in open(tmp_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        idx = r.pop("_idx", None)
        if isinstance(idx, int):
            by_idx[idx] = r
    # 合并：原 rows 中 response 非空的保留，空的用 by_idx 填
    for idx, r in by_idx.items():
        rows[idx] = r
    # 写临时有序文件再原子替换
    fd, final_tmp = tempfile.mkstemp(prefix=".skywork_final.", suffix=".jsonl",
                                     dir=str(args.jsonl.parent))
    os.close(fd)
    with open(final_tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(final_tmp, args.jsonl)
    tmp_path.unlink(missing_ok=True)

    # 校验
    empty = sum(1 for r in rows if not r.get("response"))
    print(f"✅ 完成：{len(rows)} 行，response 非空 {len(rows)-empty}，仍空 {empty}", flush=True)
    if empty:
        print(f"⚠️ {empty} 行 response 仍空（生成失败？），可重跑本脚本 resume 补齐", flush=True)


if __name__ == "__main__":
    main()

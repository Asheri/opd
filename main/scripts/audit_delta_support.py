#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 补充审计：student 支撑口径的 Δ 统计（对比训练 reward 量级）。

D3 FAIL 用的是 teacher top-K 支撑上的 Δ 均值（-1.159），但训练实际用的 Δ 是
cache.delta_at_student_topk 在【student 支撑】上展开。本脚本加载学生模型 + 磁盘 cache，
对样本在 student top-K 支撑上统计 Δ，与训练 reward（-0.45~-0.5）直接对比，
验证 D3 指标的口径偏置。

2026-08-25 修复：
- Bug 1：rows = [] 初始化缺失（NameError）→ 抽出 _load_jsonl_rows；
- Bug 2：cache 行与 jsonl 行索引对齐校验（student tokenizer 编码长度 vs cache.lengths），
  匹配率 < 90% 直接退出，不输出任何 Δ 结论；
- Bug 3：--max-prompt-len 默认 0 → 从 cache metadata 读取实际 max_prompt_len（不硬编码）；
- Bug 4：student/teacher 均值均按 cache.lengths 掩码，只统计有效 token；
- Bug 5：报告 student top-K 与 teacher top-K 的命中率（mean/min/max）；
- Bug 6：长度序列校验同时作为 response 版本一致性证据（记录匹配率）。

用法（服务器）：
    /root/miniconda3/bin/python -u scripts/audit_delta_support.py \
        --n-samples 20 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.cache_store import DiskTeacherCache  # noqa: E402
from fullstack_opd_v2.model_factory import HFCausalLM  # noqa: E402

STUDENT = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B"
CACHE = "/root/autodl-tmp/cache_skywork_chat.pt"
DATA = "/root/autodl-tmp/datasets/skywork_50k.jsonl"


def _load_jsonl_rows(data_path: str, n_samples: int) -> list[dict]:
    """取 jsonl 前 n_samples 条有效行（prompt+response 都存在）。"""
    rows: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prompt") and row.get("response"):
                rows.append(row)
            if len(rows) >= n_samples:
                break
    return rows


def _read_metadata(cache_path: str):
    meta_path = f"{cache_path}.metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f), meta_path


def _align_lengths(tok, rows: list[dict], cache: DiskTeacherCache,
                   max_response_len: int, max_print: int = 20):
    """cache 第 i 行长度 vs jsonl 第 i 条 response 编码长度（student tokenizer）。

    返回 (match_rate, mismatches)；mismatches 元素为 (i, jsonl_len, cache_len)。
    长度匹配是「response 版本一致」的必要条件：长度都对不上说明数据源错位。
    """
    mismatches: list[tuple[int, int, int]] = []
    for i, r in enumerate(rows):
        r_ids = tok.encode(r["response"], add_special_tokens=False,
                           truncation=True, max_length=max_response_len)
        jsonl_len = len(r_ids)
        cache_len = int(cache.response_length(torch.tensor([i]))[0])
        if jsonl_len != cache_len:
            mismatches.append((i, jsonl_len, cache_len))
    n = len(rows)
    match_rate = 1.0 if n == 0 else (n - len(mismatches)) / n
    if mismatches:
        print(f"[C3v2][ALIGN] mismatch 明细（前 {min(max_print, len(mismatches))} 条）：",
              flush=True)
        for i, jl, cl in mismatches[:max_print]:
            print(f"  i={i} jsonl_len={jl} cache_len={cl}", flush=True)
    return match_rate, mismatches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    # Bug 3：默认 0 = 从 cache metadata 读真实 max_prompt_len，禁止硬编码 1024。
    ap.add_argument("--max-prompt-len", type=int, default=0)
    ap.add_argument("--max-response-len", type=int, default=0)
    ap.add_argument("--topk", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)

    meta, meta_path = _read_metadata(CACHE)
    cache = DiskTeacherCache(CACHE, device="cpu", top_k=args.topk,
                             vocab=int(meta.get("vocab") or 151936))
    if not args.max_response_len:
        args.max_response_len = int(cache.T)   # 必须与 cache T 对齐（searchsorted 边界）
    if not args.max_prompt_len:
        args.max_prompt_len = int(meta.get("max_prompt_len") or 0)
    if args.max_prompt_len <= 0:
        print("[C3v2][FAIL] metadata 缺少 max_prompt_len，且未显式 --max-prompt-len")
        return 1

    print(f"[C3v2] cache samples={cache.num_samples} T={cache.T} K={args.topk} "
          f"resp_len={args.max_response_len}")
    print(f"[C3v2] 采用 max_prompt_len={args.max_prompt_len} "
          f"（来自 {os.path.basename(meta_path)}）")

    # 文件时间供人工核对版本
    try:
        dt_data = datetime.fromtimestamp(os.path.getmtime(DATA)).isoformat()
    except OSError:
        dt_data = "N/A"
    try:
        dt_meta = datetime.fromtimestamp(os.path.getmtime(meta_path)).isoformat()
    except OSError:
        dt_meta = "N/A"
    print(f"[C3v2][ALIGN] jsonl mtime={dt_data}  cache metadata mtime={dt_meta}")
    print(f"[C3v2][ALIGN] jsonl path={DATA}  cache={CACHE}")

    rows = _load_jsonl_rows(DATA, args.n_samples)
    if len(rows) < args.n_samples:
        print(f"[C3v2][FAIL] 样本不足：需要 {args.n_samples}，仅有 {len(rows)}")
        return 1

    # Bug 2/6：索引对齐校验（fail-fast，不通过则不输出任何 Δ 结论）
    match_rate, mismatches = _align_lengths(tok, rows, cache, args.max_response_len)
    if match_rate < 0.90:
        print(f"[C3v2][ALIGN][FAIL] 长度匹配率 {match_rate:.3f} < 0.90 —— "
              "jsonl 与 cache 非同源/索引错位，审计无效。")
        print("[C3v2][ALIGN] 请先确认 cache build 时的真实数据文件（materialized "
              "response 来源），修正 DATA/CACHE 后重跑。")
        return 1
    print(f"[C3v2][ALIGN][PASS] 长度匹配率 {match_rate:.3f}（>=0.90）—— "
          "cache 行与 jsonl 行索引一致（response 版本同源的必要条件）")

    student = HFCausalLM(STUDENT, args.device, dtype="bf16")
    student.eval()

    # 学生 top-K 支撑上的 Δ（训练口径）
    deltas_student: list[float] = []
    deltas_teacher: list[float] = []
    hit_rates: list[float] = []
    per_sample: list[dict] = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                        tokenize=False, add_generation_prompt=True)
            p_ids = tok.encode(p, add_special_tokens=False, truncation=True,
                               max_length=args.max_prompt_len)
            pad = tok.pad_token_id
            p_t = torch.tensor([p_ids + [pad] * (args.max_prompt_len - len(p_ids))],
                               dtype=torch.long, device=args.device)
            r_ids = tok.encode(r["response"], add_special_tokens=False,
                               truncation=True, max_length=args.max_response_len)
            r_t = torch.tensor([r_ids + [pad] * (args.max_response_len - len(r_ids))],
                               dtype=torch.long, device=args.device)
            s_cur = student.response_dists(p_t, r_t, dtype=torch.bfloat16)  # (1,T,V)
            s_topk = torch.topk(s_cur.float(), args.topk, dim=-1)
            idxs = torch.tensor([i])
            # Bug 3：student 支撑必须与训练/cache build 同 prompt 长度（metadata 值）
            delta_at = cache.delta_at_student_topk(idxs, s_topk.indices, "cpu")
            # student 支撑上的 E[Δ_T]（训练 reward 口径）
            rew = (s_topk.values.float().cpu().exp() * delta_at).sum(-1)   # (1,T)

            # Bug 4：按 cache.lengths 只统计有效 token
            clen = int(cache.response_length(idxs)[0])
            valid = torch.arange(cache.T) < clen
            if valid.any():
                rew_valid = float((rew[0] * valid).sum() / valid.sum())
            else:
                rew_valid = float("nan")
            deltas_student.append(rew_valid)

            # teacher top-K 支撑上的 Δ（D3 口径，同样本、有效 token）
            tid, tdelta = cache.topk(idxs)
            tdelta_valid = float(tdelta[0][valid].mean()) if valid.any() else float("nan")
            deltas_teacher.append(tdelta_valid)

            # Bug 5：student top-K 与 teacher top-K 命中率
            t_ids0 = tid[0]                                   # (T,Kt) 已按 id 排序
            s_idx = s_topk.indices[0].cpu()                   # (T,Ks)
            pos = torch.searchsorted(t_ids0, s_idx).clamp(max=t_ids0.size(-1) - 1)
            hit = t_ids0.gather(1, pos) == s_idx
            hit_rate = float(hit.float().mean())
            hit_rates.append(hit_rate)

            per_sample.append({"i": i, "student_E_delta": rew_valid,
                               "teacher_delta_mean": tdelta_valid,
                               "hit_rate": hit_rate})
            if i < 5:
                print(f"[C3v2] s{i}: student支撑E[Δ]={rew_valid:.3f} "
                      f"teacher支撑Δ均值={tdelta_valid:.3f} 命中率={hit_rate:.3f}",
                      flush=True)

    ms = statistics.mean(deltas_student)
    mt = statistics.mean(deltas_teacher)
    hr_mean = statistics.mean(hit_rates)
    hr_min = min(hit_rates)
    hr_max = max(hit_rates)
    print(f"\n[C3v2] 样本 {args.n_samples} 条（idx 0-{args.n_samples - 1}）")
    print(f"[C3v2] 对齐校验匹配率 = {match_rate:.3f}（PASS，长度同源证据）")
    print(f"[C3v2] 采用 max_prompt_len = {args.max_prompt_len}")
    print(f"[C3v2] student支撑 E[Δ_T] 均值 = {ms:+.3f}  （训练 reward 口径，有效 token）")
    print(f"[C3v2] teacher支撑 Δ 均值     = {mt:+.3f}  （D3 口径，有效 token）")
    print(f"[C3v2] top-K 命中率: mean={hr_mean:.3f} min={hr_min:.3f} max={hr_max:.3f}")
    print(f"[C3v2] 对比：训练实测 reward ≈ -0.45~-0.5；D3 支撑均值 = -1.159")
    result = {
        "n": args.n_samples,
        "alignment_match_rate": match_rate,
        "alignment_pass": True,
        "jsonl_mtime": dt_data,
        "cache_metadata_mtime": dt_meta,
        "max_prompt_len_used": args.max_prompt_len,
        "max_response_len_used": args.max_response_len,
        "student_support_E_delta_mean": ms,
        "teacher_support_delta_mean": mt,
        "hit_rate_mean": hr_mean,
        "hit_rate_min": hr_min,
        "hit_rate_max": hr_max,
        "length_mismatches": mismatches[:20],
        "per_sample": per_sample,
    }
    with open("/tmp/c3v2_delta_support.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[C3v2] 结果已写 /tmp/c3v2_delta_support.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

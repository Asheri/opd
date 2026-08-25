#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 审计：教师模板一致性 + 词表对齐 + Δ 方向实证（服务器 GPU 跑）。

D3 FAIL（Δ 均值 -1.159< -1.0）后按任务回 C3 审计。本脚本在服务器上：
1. 词表对比：Qwen3 学生 / JustRL(teacher_rl) / R1-Distill(teacher_ref) 三个 tokenizer
   vocab 大小与 sample token 越界检查。
2. 教师 logp 四组合（各 N 条样本）：
   a. teacher_rl(student_tpl_prompt, student_response_ids)
   b. teacher_rl(rl_tpl_prompt,     rl_response_ids)
   c. teacher_ref(student_tpl_prompt, student_response_ids)
   d. teacher_ref(ref_tpl_prompt,   ref_response_ids)
3. Δ 对比：
   Δ_student = logp(a) - logp(c)   （cache build 若教师共用学生模板时的口径）
   Δ_teacher = logp(b) - logp(d)   （C3 教师各自模板时的口径）
4. 判定：
   - Δ_teacher 明显 > Δ_student 且方向正常 → cache build 教师模板未生效 → 需重建 cache
   - 两者都深度负 → 教师对本身负向（角色/模型/数据）→ 更深审计
   - response 词表跨模型越界 → 指出 Δ 计算本身有词表错位风险

用法（服务器）：
    /root/miniconda3/bin/python -u scripts/audit_teacher_templates.py \
        --n-samples 8 --device cuda:0 [--start-idx 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.model_factory import HFCausalLM  # noqa: E402

STUDENT = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B"
RL = "/root/autodl-tmp/models/JustRL-DeepSeek-1.5B"
REF = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"
DATA = "/root/autodl-tmp/datasets/skywork_50k.jsonl"


def _encode_text(tok, text: str, max_len: int, tpl: bool) -> list[int]:
    if tpl:
        text = tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False, truncation=True, max_length=max_len)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    return ids[:max_len] + [pad] * max(0, max_len - len(ids))


def _next_token_logp(log_softmax: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B,T,V) log-softmax + (B,T) target → (B,T) next-token logp（第 t 行预测 target[t]）。"""
    B, T, V = log_softmax.shape
    return log_softmax.gather(-1, target.clamp(0, V - 1).unsqueeze(-1)).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description="C3 教师模板审计")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--start-idx", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-response-len", type=int, default=512)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok_s = AutoTokenizer.from_pretrained(STUDENT)
    tok_rl = AutoTokenizer.from_pretrained(RL)
    tok_ref = AutoTokenizer.from_pretrained(REF)
    print(f"[C3] tokenizer vocab: student={tok_s.vocab_size} rl={tok_rl.vocab_size} "
          f"ref={tok_ref.vocab_size}")

    # 取样本
    rows = []
    with open(DATA, encoding="utf-8") as f:
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
            if len(rows) >= args.start_idx + args.n_samples:
                break
    sel = rows[args.start_idx: args.start_idx + args.n_samples]
    if len(sel) < args.n_samples:
        print(f"[C3][FAIL] 样本不足：需要 {args.n_samples}，仅有 {len(sel)}")
        return 1
    print(f"[C3] 样本 {args.n_samples} 条（idx {args.start_idx}-{args.start_idx + args.n_samples - 1}）")

    # 教师模型（bf16, GPU）
    rl = HFCausalLM(RL, args.device, dtype="bf16")
    ref = HFCausalLM(REF, args.device, dtype="bf16")
    rl.eval()
    ref.eval()

    # 词表越界检查（学生 response token 喂教师）
    over_rl = over_ref = 0
    for r in sel:
        r_ids = _encode_text(tok_s, r["response"], args.max_response_len, tpl=False)
        over_rl += sum(1 for x in r_ids if x >= rl.vocab)
        over_ref += sum(1 for x in r_ids if x >= ref.vocab)
    print(f"[C3] 学生 response token 越界（>=教师词表）: rl={over_rl} ref={over_ref} "
          f"(共 {args.n_samples * args.max_response_len})")

    # 四组合 logp
    sums = {"a_rl_student": 0.0, "b_rl_own": 0.0,
            "c_ref_student": 0.0, "d_ref_own": 0.0}
    per_sample = []
    with torch.no_grad():
        for i, r in enumerate(sel):
            p_stu = torch.tensor([_encode_text(tok_s, r["prompt"], args.max_prompt_len, True)],
                                 dtype=torch.long, device=args.device)
            p_rl = torch.tensor([_encode_text(tok_rl, r["prompt"], args.max_prompt_len, True)],
                                dtype=torch.long, device=args.device)
            p_ref = torch.tensor([_encode_text(tok_ref, r["prompt"], args.max_prompt_len, True)],
                                 dtype=torch.long, device=args.device)
            r_stu = torch.tensor([_encode_text(tok_s, r["response"], args.max_response_len, False)],
                                 dtype=torch.long, device=args.device)
            r_rl = torch.tensor([_encode_text(tok_rl, r["response"], args.max_response_len, False)],
                                dtype=torch.long, device=args.device)
            r_ref = torch.tensor([_encode_text(tok_ref, r["response"], args.max_response_len, False)],
                                 dtype=torch.long, device=args.device)
            # a: rl 学生模板
            # Bug 7：按各自 tokenizer 的有效长度（去除 pad）求均值，避免 pad 位置稀释 logp。
            valid_stu = max(1, len(tok_s.encode(r["response"], add_special_tokens=False,
                                                truncation=True, max_length=args.max_response_len)))
            valid_rl = max(1, len(tok_rl.encode(r["response"], add_special_tokens=False,
                                                truncation=True, max_length=args.max_response_len)))
            valid_ref = max(1, len(tok_ref.encode(r["response"], add_special_tokens=False,
                                                  truncation=True, max_length=args.max_response_len)))
            la = _next_token_logp(rl.response_dists(p_stu, r_stu), r_stu)[0, :valid_stu].mean().item()
            lb = _next_token_logp(rl.response_dists(p_rl, r_rl), r_rl)[0, :valid_rl].mean().item()
            lc = _next_token_logp(ref.response_dists(p_stu, r_stu), r_stu)[0, :valid_stu].mean().item()
            ld = _next_token_logp(ref.response_dists(p_ref, r_ref), r_ref)[0, :valid_ref].mean().item()
            for k, v in (("a_rl_student", la), ("b_rl_own", lb),
                         ("c_ref_student", lc), ("d_ref_own", ld)):
                sums[k] += v
            per_sample.append({"i": args.start_idx + i,
                               "rl_student": la, "rl_own": lb,
                               "ref_student": lc, "ref_own": ld,
                               "delta_student": la - lc, "delta_teacher": lb - ld})
            print(f"[C3] s{i}: rl_stu={la:.3f} rl_own={lb:.3f} | ref_stu={lc:.3f} ref_own={ld:.3f} "
                  f"| Δ_student={la - lc:+.3f} Δ_teacher={lb - ld:+.3f}", flush=True)

    n = len(sel)
    avg = {k: v / n for k, v in sums.items()}
    d_stu = avg["a_rl_student"] - avg["c_ref_student"]
    d_tea = avg["b_rl_own"] - avg["d_ref_own"]
    print(f"\n[C3] 均值: rl_stu={avg['a_rl_student']:.3f} rl_own={avg['b_rl_own']:.3f} "
          f"ref_stu={avg['c_ref_student']:.3f} ref_own={avg['d_ref_own']:.3f}")
    print(f"[C3] Δ_student(共用学生模板)={d_stu:+.3f}  Δ_teacher(教师各自模板)={d_tea:+.3f}")

    # 判定
    print("\n[C3] 判定:")
    if d_tea - d_stu > 0.3:
        print(f"[C3]   Δ_teacher - Δ_student = {d_tea - d_stu:+.3f} > 0.3 → 教师模板显著影响 Δ 方向；"
              "若 cache build 未传教师模板则需重建 cache")
    else:
        print(f"[C3]   模板影响小（Δ差 {d_tea - d_stu:+.3f}）→ 教师对方向与模板关系不大")
    if d_tea < -0.5:
        print(f"[C3]   教师各自模板下 Δ 仍深度负（{d_tea:+.3f}）→ 教师对本身方向负（角色/模型/数据）")
    if over_rl or over_ref:
        print(f"[C3]   ⚠️ 学生 response token 越教师词表（rl={over_rl} ref={over_ref}）"
              "→ Δ 计算存在跨词表错位风险")
    with open("/tmp/c3_audit_result.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer": {"student": tok_s.vocab_size, "rl": tok_rl.vocab_size,
                                 "ref": tok_ref.vocab_size},
                   "overlap_oob": {"rl": over_rl, "ref": over_ref},
                   "avg": avg, "delta_student": d_stu, "delta_teacher": d_tea,
                   "per_sample": per_sample}, f, ensure_ascii=False, indent=2)
    print("[C3] 结果已写 /tmp/c3_audit_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

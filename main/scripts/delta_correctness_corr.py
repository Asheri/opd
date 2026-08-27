#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-1b（决定性实验）：Δ_T ↔ 答案正确性相关性——判定 RC1（固定 D）vs RC4（Δ 语义无效）。

分析依据：docs/reports/2026-08-26-opd-failure-analysis.md §5 E-1b。
目的：验证 Δ_T 的"改进方向"与答案正确性是否相关——若 Δ>0 与 correct 显著正相关，
说明信号含正确性成分、OPD 失败是 RC1（固定 500 条 base 轨迹偏离 on-policy）；若不相关，
说明 Δ 是风格成分主导（RC4），任何 KL/数据/on-policy 调整都救不了，须换教师对。

设计：MATH500 抽 N 题 × student（Base）采样 M 条/题（T=1.0, chat, B2048）
→ 两教师（rl=JustRL / ref=R1-Distill）各一次 forward 算完整序列 per-token logp
（vLLM prompt_logprobs）→ 序列级 Δ = Σ_response (logp_rl − logp_ref)（及 per-token 均值）
→ sympy 判分 → Spearman(Δ_seq, correct) + AUC。

判据（写死，来自归因分析 §5 E-1b）：
  ρ ≥ 0.2        → 信号有效，RC1 主犯 → 分支 A（on-policy 化）
  0.05 ≤ ρ < 0.2 → 弱信号          → 分支 B2（信号改造）
  ρ < 0.05       → 信号无效，RC4 主犯 → 分支 B1（换教师对）

三阶段（支持双卡流水并行，最大化 GPU 利用）：
  --stage sample    : student 采样生成 → samples.jsonl（GPU0；n_problems×n_samples 条）
  --stage logp      : 对 samples 的 full_seq 用指定教师算 per-token logp → logp_{rl|ref}.jsonl
                      （GPU0=rl、GPU1=ref 并行；脚本内逐条 generate，天然可断点续）
  --stage correlate : 判分 + 序列级 Δ + Spearman/AUC → report.json（CPU）

用法（服务器，双卡并行）：
  # GPU0：采样 200 题 × 4 条
  /root/miniconda3/bin/python -u scripts/delta_correctness_corr.py \
      --stage sample --dataset MATH500 --n-problems 200 --n-samples 4 \
      --student /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
      --budget 2048 --temperature 1.0 --chat-template --device cuda:0 \
      --out /root/autodl-tmp/delta_corr
  # GPU0=rl、GPU1=ref 并行（samples 就绪后）
  /root/miniconda3/bin/python -u scripts/delta_correctness_corr.py \
      --stage logp --teacher rl --device cuda:0 --out /root/autodl-tmp/delta_corr
  /root/miniconda3/bin/python -u scripts/delta_correctness_corr.py \
      --stage logp --teacher ref --device cuda:1 --out /root/autodl-tmp/delta_corr
  # 任一卡（CPU）：相关性
  /root/miniconda3/bin/python -u scripts/delta_correctness_corr.py \
      --stage correlate --out /root/autodl-tmp/delta_corr
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.budget_eval import BudgetEvaluator, extract_final_answer, format_prompt  # noqa: E402
from fullstack_opd_v2.eval_aime import _grade_answer_sympy  # noqa: E402

DEFAULT_STUDENT = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B"
DEFAULT_TEACHER_RL = "/root/autodl-tmp/models/JustRL-DeepSeek-1.5B"
DEFAULT_TEACHER_REF = "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B"


# ----------------------------- 纯函数（可单测，不依赖 vLLM/GPU） -----------------------------

def _extract_prompt_logprobs(prompt_logprobs) -> list[float | None]:
    """vLLM prompt_logprobs → 每 token 的条件 logprob 列表。

    每位置为 {token_id: Logprob}（key 即该位置的 prompt token），取该 token 的 logprob；
    该值 = P(token_i | token_0..i-1)，即序列级 logp 的逐位展开。
    """
    out: list[float | None] = []
    for pl in prompt_logprobs:
        if not pl:
            out.append(None)
            continue
        toks = list(pl.keys())
        out.append(pl[toks[0]].logprob if toks else None)
    return out


def compute_delta(rl_logps: list[float | None], ref_logps: list[float | None],
                  start_idx: int) -> dict:
    """response token 上的序列级 Δ = Σ(logp_rl − logp_ref)（及 per-token 均值）。

    rl/ref_logps 为完整序列（wrapped prompt + response）的 per-token logp；
    start_idx 为 response 起始位置（wrapped prompt 的 token 数）。None/NaN 跳过。
    返回 delta_sum（序列级，对应论文 Σ Δ）、delta_mean（归一化，防长序列主导）、n_tokens。
    """
    n_tokens = 0
    acc = 0.0
    hi = max(len(rl_logps), len(ref_logps))
    for i in range(start_idx, hi):
        a = rl_logps[i] if i < len(rl_logps) else None
        b = ref_logps[i] if i < len(ref_logps) else None
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            continue
        acc += a - b
        n_tokens += 1
    return {"delta_sum": round(acc, 6),
            "delta_mean": round(acc / n_tokens, 6) if n_tokens else 0.0,
            "n_tokens": n_tokens}


def _auc_pos_neg(deltas: list[float], corrects: list[bool]) -> float:
    """正确 vs 错误样本 Δ 分布的 AUC（Mann-Whitney U / (n_pos*n_neg)）。"""
    pos = [d for d, c in zip(deltas, corrects) if c]
    neg = [d for d, c in zip(deltas, corrects) if not c]
    if not pos or not neg:
        return 0.5
    u = 0.0
    for p in pos:
        for n in neg:
            u += 1 if p > n else (0.5 if p == n else 0)
    return u / (len(pos) * len(neg))


def _spearman(x: list[float], y: list[float]) -> tuple[float, float | None]:
    """手写 Spearman 秩相关（纯 Python，不依赖 scipy；并列取平均秩）。

    返回 (rho, p)：p 在 scipy 可用时给出，否则 None（判据只用 rho）。
    2026-08-26 实测：服务器缺 scipy，scipy.spearmanr 抛 ImportError 被旧实现
    吞成 rho=0.0/p=1.0 的假阴性——故改手写。
    """
    n = len(x)
    if n < 3:
        return 0.0, None

    def _rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0          # 平均秩（1-based）
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    mx = my = sum(rx) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    denom = (vx * vy) ** 0.5
    rho = cov / denom if denom else 0.0
    p = None
    try:
        from scipy.stats import spearmanr
        _, p = spearmanr(x, y)
        p = float(p)
    except Exception:
        p = None
    return rho, p


def correlate(deltas: list[float], corrects: list[bool]) -> dict:
    """Spearman ρ + AUC + 基础统计（判据由调用方对照 §5 E-1b 表执行）。"""
    n = len(deltas)
    if n < 3:
        return {"n": n, "spearman_rho": 0.0, "spearman_p": None,
                "auc": 0.5, "mean_delta": 0.0, "acc": 0.0}
    rho, p = _spearman(deltas, corrects)
    return {"n": n, "spearman_rho": round(rho, 6), "spearman_p": p,
            "auc": round(_auc_pos_neg(deltas, corrects), 6),
            "mean_delta": round(sum(deltas) / n, 6),
            "acc": round(sum(corrects) / n, 6)}


def judge_response(response: str, ground_truth: str) -> bool:
    """extract_final_answer + sympy 判分（与 vllm_budget_eval 同协议）。"""
    fa = extract_final_answer(response)
    if fa is None:
        return False
    try:
        return bool(_grade_answer_sympy(fa, ground_truth))
    except Exception:
        return False


# ----------------------------- 文件辅助 -----------------------------

def _count_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _append_rows(path: str, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _load_done_ids(path: str) -> set[int]:
    return {r["sample_idx"] for r in _load_jsonl(path)}


# ----------------------------- 阶段：sample -----------------------------

def _load_problems(dataset_ref: str, n_problems: int | None) -> list[tuple[str, str]]:
    ev = object.__new__(BudgetEvaluator)
    problems = ev.load_problems(dataset_ref)
    if n_problems is not None:
        problems = problems[:n_problems]
    return problems


def _sample_stage(args) -> None:
    """student 采样生成 → samples.jsonl（每行：problem_id, gt, sample_idx, response, full_seq, prompt_token_len）。"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    problems = _load_problems(args.dataset, args.n_problems)
    tok = AutoTokenizer.from_pretrained(args.student)
    # R2：dapo 模板先 format_prompt 再包 chat（对齐训练/评估的 DAPO 协议）；
    # boxed（默认零回归）保持旧行为：problem 原文直接包 chat。
    if args.prompt_style == "dapo":
        problems = [(format_prompt(p, "dapo"), gt) for p, gt in problems]
    wrapped = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
               for p, _ in problems]
    prompt_lens = [len(tok.encode(w)) for w in wrapped]   # 同族词表，教师侧一致

    llm = LLM(model=args.student, tensor_parallel_size=1,
              gpu_memory_utilization=0.9, max_model_len=args.max_model_len,
              enforce_eager=False)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.budget,
                        n=args.n_samples)
    out_path = os.path.join(args.out, "samples.jsonl")
    rows: list[dict] = []
    for i in range(0, len(wrapped), args.chunk):
        batch = wrapped[i:i + args.chunk]
        outs = llm.generate(batch, sampling_params=sp)
        for j, o in enumerate(outs):
            pid = i + j
            for k, comp in enumerate(o.outputs):
                rows.append({
                    "problem_id": pid,
                    "ground_truth": problems[pid][1],
                    "sample_idx": pid * args.n_samples + k,
                    "response": comp.text,
                    "full_seq": batch[j] + comp.text,
                    "prompt_token_len": prompt_lens[pid],
                })
        if len(rows) >= args.chunk * args.n_samples:
            _append_rows(out_path, rows)
            rows = []
    if rows:
        _append_rows(out_path, rows)
    print(f"[delta-corr] sample 完成: {_count_rows(out_path)} 条 → {out_path}", flush=True)


# ----------------------------- 阶段：logp -----------------------------

def _logp_stage(args) -> None:
    """对 samples 的 full_seq 用指定教师算 per-token logp → logp_{teacher}.jsonl（逐条可断点续）。"""
    from vllm import LLM, SamplingParams

    teacher_path = args.teacher_rl if args.teacher == "rl" else args.teacher_ref
    src = os.path.join(args.out, "samples.jsonl")
    samples = _load_jsonl(src)
    if not samples:
        print(f"[delta-corr] {src} 空或不存在，先跑 --stage sample", flush=True)
        sys.exit(2)
    llm = LLM(model=teacher_path, tensor_parallel_size=1,
              gpu_memory_utilization=0.9, max_model_len=args.max_model_len,
              enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    out_path = os.path.join(args.out, f"logp_{args.teacher}.jsonl")
    done = _load_done_ids(out_path)
    todo = [(s["sample_idx"], s) for s in samples if s["sample_idx"] not in done]
    rows: list[dict] = []
    for idx, (sid, s) in enumerate(todo):
        out = llm.generate([s["full_seq"]], sampling_params=sp)[0]
        # vLLM API：prompt_logprobs 在 RequestOutput 顶层（SamplingParams.prompt_logprobs>0 时
        # 填充），不在 CompletionOutput 上（后者无此属性——2026-08-26 实测 AttributeError）。
        pl = _extract_prompt_logprobs(out.prompt_logprobs)
        rows.append({"sample_idx": sid, "problem_id": s["problem_id"], "logps": pl})
        if len(rows) >= args.chunk:
            _append_rows(out_path, rows)
            rows = []
    if rows:
        _append_rows(out_path, rows)
    print(f"[delta-corr] logp({args.teacher}) 完成: {_count_rows(out_path)} 条 → {out_path}", flush=True)


# ----------------------------- 阶段：correlate -----------------------------

def _correlate_stage(args) -> None:
    """判分 + 序列级 Δ + Spearman/AUC → report.json（判据写死，对照 §5 E-1b 表）。"""
    samples = {s["sample_idx"]: s for s in _load_jsonl(os.path.join(args.out, "samples.jsonl"))}
    rl = {r["sample_idx"]: r["logps"] for r in _load_jsonl(os.path.join(args.out, "logp_rl.jsonl"))}
    ref = {r["sample_idx"]: r["logps"] for r in _load_jsonl(os.path.join(args.out, "logp_ref.jsonl"))}
    common = sorted(set(samples) & set(rl) & set(ref))
    if not common:
        print("[delta-corr] 无共同样本（缺 samples/logp_rl/logp_ref 之一），退出", flush=True)
        sys.exit(2)
    deltas: list[float] = []
    corrects: list[bool] = []
    rows: list[dict] = []
    for i in common:
        s = samples[i]
        d = compute_delta(rl[i], ref[i], int(s.get("prompt_token_len") or 0))
        ok = judge_response(s["response"], s["ground_truth"])
        deltas.append(d["delta_mean"])
        corrects.append(ok)
        rows.append({"sample_idx": i, "problem_id": s["problem_id"],
                     "delta_sum": d["delta_sum"], "delta_mean": d["delta_mean"],
                     "n_tokens": d["n_tokens"], "correct": ok})
    stats = correlate(deltas, corrects)
    report = {
        "stage": "correlate",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spec": {"dataset": args.dataset, "n_problems": args.n_problems,
                 "n_samples": args.n_samples, "budget": args.budget,
                 "temperature": args.temperature,
                 "teacher_rl": args.teacher_rl, "teacher_ref": args.teacher_ref},
        "verdict_table": {
            "rho>=0.2": "信号有效 → 分支 A（on-policy 化）",
            "0.05<=rho<0.2": "弱信号 → 分支 B2（信号改造）",
            "rho<0.05": "信号无效 → 分支 B1（换教师对）",
        },
        "stats": stats,
        "rows": rows,
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"[delta-corr] report → {os.path.join(args.out, 'report.json')}", flush=True)


# ----------------------------- CLI -----------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["sample", "logp", "correlate"], required=True,
                   help="三阶段：sample 采样 → logp 教师 forward（rl/ref 双卡并行）→ correlate 判分+相关性")
    p.add_argument("--out", required=True, help="输出目录（samples.jsonl / logp_*.jsonl / report.json）")
    # sample 参数
    p.add_argument("--student", default=DEFAULT_STUDENT, help="student 模型路径（采样生成用）")
    p.add_argument("--dataset", default="MATH500")
    p.add_argument("--n-problems", type=int, default=200, help="抽题数（MATH500 前 N 题）")
    p.add_argument("--n-samples", type=int, default=4, help="每题采样条数（T=1.0）")
    p.add_argument("--budget", type=int, default=2048, help="采样 max_tokens（对齐训练 B2048）")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--chat-template", action="store_true",
                   help="用 student chat template 包裹（对齐训练 apply_chat_template=true）")
    p.add_argument("--prompt-style", choices=["boxed", "dapo"], default="boxed",
                   help="R2（2026-08-27 数据质量审阅）：sample 阶段 prompt 模板，dapo 时"
                        "先 format_prompt(p, 'dapo') 再包 chat template--使 E-1b' 相关性"
                        "域与训练/评估 DAPO 模板一致（默认 boxed 零回归）")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=20, help="每批题数（sample）/ 每批条数（logp）")
    # logp 参数
    p.add_argument("--teacher", choices=["rl", "ref"], default="rl",
                   help="logp 阶段指定教师（rl=JustRL / ref=R1-Distill）")
    p.add_argument("--teacher-rl", default=DEFAULT_TEACHER_RL)
    p.add_argument("--teacher-ref", default=DEFAULT_TEACHER_REF)
    p.add_argument("--device", default="cuda:0", help="显式选卡（cuda:i；vLLM 经 CUDA_VISIBLE_DEVICES）")
    return p.parse_args(argv)


def _apply_cuda_visible(device: str | None) -> str | None:
    """--device cuda:i → CUDA_VISIBLE_DEVICES=i（vLLM 选卡唯一途径，双卡并行前提）。"""
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return idx
    return None


def main() -> None:
    args = parse_args()
    _apply_cuda_visible(args.device)
    if args.stage == "sample":
        _sample_stage(args)
    elif args.stage == "logp":
        _logp_stage(args)
    else:
        _correlate_stage(args)


if __name__ == "__main__":
    main()

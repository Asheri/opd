#!/usr/bin/env python3
"""Stage 1.5 K Calibration：在真实 teacher 上给 K∈{32,64,128,256} 校准四个指标。

目标：给『K 取多少』提供数据依据，替代『凭显存压力降到 16』的拍脑袋。
在真实 Skywork prompt/response 上，对每个 response 位置取 teacher_rl 的 full-vocab
分布，比较 K-truncated 稀疏 Δ_T 相对 K=256 参考的保真度：

  M_K   = Σ_{i∈TopK(rl)} p_i        —— teacher_rl 分布在 top-K 上覆盖的概率质量
  C_K   = P( y_t ∈ TopK(rl) )       —— 真实 chosen token 落在 teacher top-K 内的概率
  ρ_K   = Corr( Δ_T^(K), Δ_T^(256) )—— 在 256 支撑上，K 截断 Δ 与 256 参考 Δ 的 Pearson 相关
  MAE_K = E[ |Δ_T^(K) − Δ_T^(256)| ]—— 同支撑上的平均绝对误差

口径（与 cache 存储语义一致）：
  - Δ_T(v) = log p_rl(v) − log p_ref(v)，只在 teacher top-K(rl) 支撑上有值，其余 = 0。
  - 支撑 = teacher_rl 逐位置 top-K。Δ_T^(K) 保留前 K 个 delta，K..255 置 0。
  - ρ_K / MAE_K 在 **256 支撑**上 pooled（只统计 S_256 内元素，避免 V 上万全 0/0 稀释）。
  - 参考基准 = K=256（ρ_256=1, MAE_256=0 自洽）。

另含 generation throughput 校准：对 teacher_rl 做一次受控 generate，测 tok/s。

用法（服务器，teacher 在 GPU）：
  python scripts/k_calibration.py \
    --jsonl /root/autodl-tmp/datasets/skywork_calib_1000.jsonl \
    --teacher-rl /root/autodl-tmp/models/JustRL-DeepSeek-1.5B \
    --teacher-ref /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --device cuda:0 --batch-size 4 --max-len 1024 \
    --out /root/autodl-tmp/eval/k_calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# 把仓库根（main/）加进 sys.path 以便 import fullstack_opd_v2
_MAIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MAIN_ROOT not in sys.path:
    sys.path.insert(0, _MAIN_ROOT)

from fullstack_opd_v2.model import response_dists


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, help="Skywork prompt/response jsonl")
    p.add_argument("--teacher-rl", required=True, help="teacher_rl HF 模型路径")
    p.add_argument("--teacher-ref", required=True, help="teacher_ref HF 模型路径")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4, help="前向 batch（控 full-vocab 显存）")
    p.add_argument("--max-len", type=int, default=1024, help="response 截断长度（校准只需前缀）")
    p.add_argument("--max-prompt-len", type=int, default=512, help="prompt 截断长度")
    p.add_argument("--k", default="32,64,128,256", help="要校准的 K 集合（含参考 256）")
    p.add_argument("--max-samples", type=int, default=None, help="只取前 N 条（限时）")
    p.add_argument("--gen-tokens", type=int, default=256, help="生成吞吐校准的 token 数")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
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


def tokenize(tok, prompts, responses, P, T):
    """str 列表 -> (P_ids[], R_ids[])，截断 + 定长 pad 到 P/T。"""
    p_ids, r_ids = [], []
    for pr, rs in zip(prompts, responses):
        p = tok.encode(pr, add_special_tokens=True)
        r = tok.encode(rs, add_special_tokens=False)
        p = p[:P]
        r = r[:T]
        p_ids.append(p + [tok.pad_token_id] * (P - len(p)))
        r_ids.append(r + [tok.pad_token_id] * (T - len(r)))
    return (torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(r_ids, dtype=torch.long))


@torch.no_grad()
def calibrate(models, tok, prompts, responses, Ks, device, batch_size, P, T,
              gen_tokens):
    teacher_rl, teacher_ref = models
    top_k_ref = max(Ks)                       # 参考支撑 = 最大 K
    slot_sum = torch.zeros(top_k_ref, device=device)   # Σ_p delta_256[p, i]
    slot_sq = torch.zeros(top_k_ref, device=device)    # Σ_p delta_256[p, i]^2
    slot_abs = torch.zeros(top_k_ref, device=device)   # Σ_p |delta_256[p, i]|
    mass = {k: torch.zeros(0, device=device) for k in Ks}  # 每 K 的 M_K 逐位值（聚合用）
    n_pos_total = 0
    n_sample = 0
    chosen = torch.zeros(0, device=device)    # 每条样本每位置 chosen 的 rank（<256 或 256）

    N = len(prompts)
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        pr, rs = prompts[sl], responses[sl]
        p_t, r_t = tokenize(tok, pr, rs, P, T)          # (c,P) (c,T)
        p_t, r_t = p_t.to(device), r_t.to(device)
        rl_logp = response_dists(teacher_rl, p_t, r_t)  # (c,T,V) log-softmax
        ref_logp = response_dists(teacher_ref, p_t, r_t)
        c, TT, V = rl_logp.shape
        delta = rl_logp - ref_logp
        # teacher top-K(rl) 支撑
        tk = torch.topk(rl_logp, top_k_ref, dim=-1)     # values (c,T,256), indices
        delta_256 = delta.gather(-1, tk.indices)        # (c,T,256)
        # 有效位（非 pad 的 response 位置）
        valid = (r_t != tok.pad_token_id)               # (c,T)
        # ---- pooled 支撑统计（只统计有效位）----
        dv = delta_256[valid]                           # (n_valid, 256)
        slot_sum += dv.sum(0)
        slot_sq += (dv * dv).sum(0)
        slot_abs += dv.abs().sum(0)
        n_pos_total += valid.sum().item()
        # ---- M_K：teacher_rl 在 top-K 覆盖的概率质量 ----
        p_topk = torch.exp(tk.values)                   # (c,T,256) softmax 概率
        for k in Ks:
            mass[k] = torch.cat([mass[k], p_topk[..., :k][valid].sum(-1)])
        # ---- C_K：chosen token 落在 top-K 内的概率 ----
        y = r_t                                # (c,T) chosen token id
        rank = torch.where(
            (tk.indices == y.unsqueeze(-1)).any(-1),        # 是否在 top-256
            (tk.indices == y.unsqueeze(-1)).int().argmax(-1),  # 在 → 找 rank
            torch.full_like(y, top_k_ref, dtype=torch.long).to(y.device),
        )                                        # 不在 → 256
        chosen = torch.cat([chosen, rank[valid]])
        n_sample += valid.sum().item()
        del rl_logp, ref_logp, delta, delta_256, p_t, r_t
        torch.cuda.empty_cache()

    # ---- 吞吐校准：teacher_rl 受控生成 ----
    gen_res = {"tok_per_s": 0.0, "n_prompt_batch": 0, "gen_tokens": 0}
    if gen_tokens > 0:
        b = min(batch_size, N)
        p_t, _ = tokenize(tok, prompts[:b], [""]*b, P, 0)
        p_t = p_t.to(device)
        t0 = time.perf_counter()
        teacher_rl.model.generate(p_t, max_new_tokens=gen_tokens, do_sample=True,
                                  temperature=1.0, top_p=0.95)
        dt = time.perf_counter() - t0
        gen_res = {"tok_per_s": round(b * gen_tokens / dt, 1),
                   "n_prompt_batch": b, "gen_tokens": gen_tokens, "wall_s": round(dt, 2)}

    # ---- 汇总指标 ----
    n_pool = n_pos_total * top_k_ref
    results = {}
    for k in sorted(Ks):
        # M_K
        m_k = (mass[k].sum() / n_pos_total).item() if n_pos_total else 0.0
        # C_K
        c_k = (chosen < k).float().mean().item() if n_sample else 0.0
        # ρ_K / MAE_K（在 256 支撑上 pooled）
        if k == top_k_ref:
            rho = 1.0
            mae = 0.0
            mae_tail = 0.0
        else:
            x_sum = slot_sum[:k].sum()
            y_sum = slot_sum.sum()
            xy = slot_sq[:k].sum()          # Σ x·y = Σ_{i<k} delta²（x=y=delta on i<k）
            x_sq = slot_sq[:k].sum()
            y_sq = slot_sq.sum()
            n_k = n_pool
            rho_num = n_k * xy - x_sum * y_sum
            denom = torch.sqrt((n_k * x_sq - x_sum * x_sum) *
                               (n_k * y_sq - y_sum * y_sum))
            rho = (rho_num / denom).item() if denom.item() > 0 else 0.0
            tail_abs = slot_abs[k:].sum().item()
            # 标准口径：MAE 对【整个 256 支撑】求均（含 0 差部分），反映截断引入的
            # 平均绝对误差；随 K 增大单调递减。tail 均值（被丢弃 token 的平均量级）
            # 另存 mae_tail 供参考。
            mae = tail_abs / n_pool if n_pool else 0.0
            mae_tail = tail_abs / (n_pos_total * (top_k_ref - k)) if n_pos_total else 0.0
        results[k] = {
            "M_K": round(m_k, 6),
            "C_K": round(c_k, 6),
            "rho_K_vs_256": round(rho, 6),
            "MAE_K_vs_256": round(mae, 8),
            "MAE_tail_avg": round(mae_tail, 8),
        }
    out = {
        "n_samples": n_sample, "n_positions": n_pos_total,
        "top_k_reference": top_k_ref, "vocab": V,
        "results": results, "generation": gen_res, "k": [int(x) for x in Ks],
    }
    return out


def main():
    args = parse_args()
    Ks = sorted({int(x) for x in args.k.split(",")})
    if max(Ks) not in (64, 128, 256):
        raise ValueError("参考 K 须含 256（当前 max(K) 应为 256）")

    from transformers import AutoTokenizer
    from fullstack_opd_v2.model_factory import HFCausalLM

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.teacher_rl)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[INFO] 加载 teacher_rl={args.teacher_rl} ...", flush=True)
    teacher_rl = HFCausalLM(args.teacher_rl, dev, dtype="bf16")
    print(f"[INFO] 加载 teacher_ref={args.teacher_ref} ...", flush=True)
    teacher_ref = HFCausalLM(args.teacher_ref, dev, dtype="bf16")
    V = teacher_rl.vocab
    print(f"[INFO] vocab={V} K={Ks} device={dev}", flush=True)

    rows = load_rows(args.jsonl, args.max_samples)
    R = [r["response"] for r in rows]
    Pq = [r["prompt"] for r in rows]
    # 只保留有 response 的样本
    alive = [(p, rr) for p, rr in zip(Pq, R) if rr]
    print(f"[INFO] 有 response 的样本 {len(alive)}/{len(rows)}", flush=True)
    if not alive:
        raise SystemExit("没有可用 response（先跑 prepare_skywork_responses.py）")
    prompts = [a[0] for a in alive]
    responses = [a[1] for a in alive]

    out = calibrate((teacher_rl, teacher_ref), tok, prompts, responses,
                    Ks, dev, args.batch_size, args.max_prompt_len, args.max_len,
                    args.gen_tokens)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 校准结果写至 {args.out}")
    print(f"     n_samples={out['n_samples']} n_positions={out['n_positions']} "
          f"vocab={out['vocab']}")
    print(f"     generation: {out['generation']}")
    print(f"\n{'K':>5} {'M_K':>10} {'C_K':>10} {'rho_K':>10} {'MAE_K':>12} {'tail':>10}")
    for k in sorted(out["results"]):
        r = out["results"][k]
        print(f"{k:>5} {r['M_K']:>10.6f} {r['C_K']:>10.6f} "
              f"{r['rho_K_vs_256']:>10.6f} {r['MAE_K_vs_256']:>12.8f} "
              f"{r['MAE_tail_avg']:>10.6f}")


if __name__ == "__main__":
    main()
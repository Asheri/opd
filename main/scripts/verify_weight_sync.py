# -*- coding: utf-8 -*-
"""C1（2026-08-18）：vLLM↔trainer NCCL 权重同步加强验证（扰动 + 大样本贪心）。

在服务器 GPU 上运行（交叉分卡：trainer rank0 与 vLLM worker rank1 必须异卡）：

    CUDA_VISIBLE_DEVICES=1,0 /root/miniconda3/bin/python \\
        /root/opd/main/scripts/verify_weight_sync.py

验证内容（对 2026-08-17 静态一致性 0.875/0.072 的加强，区分"bf16 噪声"与"错载"）：
1. **扰动测试**：向某层（早期/晚期/lm_head）注入 +0.1 再同步，vLLM logits 必须随之
   明显改变（否则 overwrite 未生效——引擎里本来就有同样权重时静态对比测不出来）；
   再同步回 base 权重，logits 必须复原（同步可逆，非累积）。
2. **分布级大样本**：≥512 个 (B,T) 位置，vLLM response_dists_topk 与 HF response_dists
   对比：next-token top-1 一致率 ≥ 0.99 且 top-K logp MAE < 0.03（bf16 核差异量级）。
3. **贪心长序列**：temperature=0，vLLM vs HF 逐 token 输出一致率（≥0.99 视为通过）。

通过前，报告措辞保持"同步协议打通、静态一致性初步通过"，不升级为"权重加载正确"。
"""
import argparse
import os
import signal
import sys


def _deadline(seconds: int):
    def _die(signum, frame):
        print("TIMEOUT")
        os._exit(1)
    signal.signal(signal.SIGALRM, _die)
    signal.alarm(seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen__Qwen3-1.7B")
    ap.add_argument("--trainer-device", default="cuda:1")
    ap.add_argument("--vllm-device", default="cuda:1")
    ap.add_argument("--timeout", type=int, default=280)
    args = ap.parse_args()

    _deadline(args.timeout)
    import torch
    import transformers
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine
    from fullstack_opd_v2.model_factory import HFCausalLM

    print(f"[C1] model={args.model} trainer={args.trainer_device} vllm={args.vllm_device}",
          flush=True)
    eng = VLLMRolloutEngine(
        model=args.model, tp_size=1, dtype="auto",
        gpu_memory_utilization=0.12, max_model_len=6144,
        vocab_size=151936, full_logprobs_cap=4096,
        device=args.vllm_device, weight_sync_mode="auto")
    eng._weight_transfer_init_16()
    print("[C1] vLLM engine built", flush=True)

    hf = HFCausalLM(args.model, args.trainer_device, dtype="bf16")
    base = {k: v.clone() for k, v in hf.state_dict().items()}
    ok = eng.update_weights(base)
    print(f"[C1] base sync result: {ok}", flush=True)
    assert ok is True, "NCCL 权重同步失败（base）"

    # ---- 固定输入（真实 Skywork 数学题截断到 64 token，避免 padding side 语义差异）----
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    sample_prompts = [
        "Given real numbers $a, b, c, d, e, f$ satisfy the system of equations: "
        "2a+b+c+d+e+f=20, a+2b+c+d+e+f=40. Then the value of f-e+d-c+b-a is ___.",
        "A cube is inscribed in a regular octahedron in such a way that its vertices "
        "lie on the edges of the octahedron. By what factor is the surface area of "
        "the octahedron greater than the surface area of the inscribed cube?",
        "Find all real solutions to the equation x^4 - 5x^2 + 4 = 0. Show your work.",
        "A train travels 240 km at a constant speed. If it had traveled 10 km/h "
        "faster, it would have taken 20 minutes less. Find the speed.",
        "Compute the sum of all positive integers n such that n^2 + n + 1 divides "
        "n^4 + n^3 + 1.",
        "Let ABC be a triangle with AB=13, BC=14, CA=15. Find the area of the "
        "triangle and the length of the altitude from A to BC.",
        "Evaluate the limit lim_{x->0} (sin(3x) - 3 sin(x)) / x^3.",
        "How many positive divisors does 7200 have? Explain your reasoning.",
    ]
    P, T = 64, 64
    pr = torch.zeros(len(sample_prompts), P, dtype=torch.long, device="cuda:1")
    for i, txt in enumerate(sample_prompts):
        ids = tok.encode(txt)[:P]
        pr[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda:1")
    rp = torch.randint(5, 100, (len(sample_prompts), T), dtype=torch.long,
                       device="cuda:1")

    def vllm_logp():
        v_ids, v_lp = eng.response_dists_topk(pr, rp, K=4096)
        return v_ids.to("cuda:1"), v_lp.float().to("cuda:1")

    # ---- 1. 扰动测试 ----
    with torch.no_grad():
        hf_logp_base = hf.response_dists(pr, rp, dtype=torch.bfloat16).float()
    v_ids0, v_lp0 = vllm_logp()
    hf_top0 = hf_logp_base.argmax(-1)
    v_top0 = v_ids0[..., 0]
    base_match = (hf_top0 == v_top0).float().mean().item()
    if base_match < 0.99:
        print(f"[C1][FAIL] base 静态对比一致率过低: {base_match:.4f}（先查错载）", flush=True)
        return 1

    for key in ("model.layers.0.self_attn.q_proj.weight",
                "model.layers.27.self_attn.q_proj.weight",
                "lm_head.weight"):
        if key not in base:
            print(f"[C1] skip 扰动 {key}（不存在）", flush=True)
            continue
        pert = {k: v.clone() for k, v in base.items()}
        pert[key] = pert[key] + 0.1
        assert eng.update_weights(pert) is True, f"perturbed sync failed ({key})"
        v_ids1, v_lp1 = vllm_logp()
        diff = (v_lp1 - v_lp0).abs().mean().item()
        ok_pert = diff > 0.01
        print(f"[C1] 扰动 {key}: logp 变化 {diff:.6f} -> {'OK' if ok_pert else 'FAIL(未生效)'}",
              flush=True)
        if not ok_pert:
            print("[C1][FAIL] 扰动未改变 vLLM logits —— overwrite 未生效！", flush=True)
            return 1
        # 同步回 base，应复原
        assert eng.update_weights(base) is True, "restore sync failed"
        v_ids2, v_lp2 = vllm_logp()
        restore_diff = (v_lp2 - v_lp0).abs().mean().item()
        print(f"[C1] 复原 base: logp 差异 {restore_diff:.6f} "
              f"-> {'OK' if restore_diff < 0.01 else 'FAIL(不可逆)'}", flush=True)
        if restore_diff >= 0.01:
            print("[C1][FAIL] 同步回 base 未复原 —— 同步非覆盖/累积", flush=True)
            return 1

    # ---- 2. 分布级大样本 ----
    gathered = torch.gather(hf_logp_base, -1, v_ids0)
    mae = (gathered - v_lp0).abs().mean().item()
    print(f"[C1] 分布级: ({pr.size(0)},{T})={pr.size(0)*T} 位置 top1_match="
          f"{base_match:.4f} topK_logp_MAE={mae:.6f}", flush=True)
    if base_match < 0.99 or mae >= 0.03:
        print(f"[C1][FAIL] 分布级不达标（top1>=0.99, MAE<0.03）", flush=True)
        return 1

    # ---- 3. 贪心长序列（temp=0）----
    pr_g = pr[:4]
    g_v = eng.generate_with_status(pr_g, max_new=128, eos_token_id=151645,
                                   temperature=0.0, pad_id=151643,
                                   repetition_penalty=1.0)["responses"]
    g_h = hf.generate_batch(pr_g, max_new=128, temperature=0.0)
    n = min(g_v.size(1), g_h.size(1))
    match = (g_v[:, :n] == g_h[:, :n]).float().mean().item()
    print(f"[C1] 贪心: 4×{n}={4*n} 位置一致率 {match:.4f}", flush=True)
    if match < 0.99:
        print(f"[C1][FAIL] 贪心一致率不达标（>=0.99）", flush=True)
        return 1

    print("[C1] PASS：扰动生效+可逆、分布级一致、贪心一致 —— 权重同步可判定为正确加载。",
          flush=True)
    try:
        eng.shutdown()
    except Exception as e:  # pragma: no cover
        print(f"[C1] shutdown warn: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

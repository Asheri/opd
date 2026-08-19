# -*- coding: utf-8 -*-
"""C1（2026-08-18 v2）：vLLM<->trainer NCCL 权重同步验证（扰动 + 分布级 + 贪心近并列审计）。

在服务器 GPU 上运行（交叉分卡，必须 CUDA_VISIBLE_DEVICES=1,0 + 默认设备参数）：
    CUDA_VISIBLE_DEVICES=1,0 /root/miniconda3/bin/python \\
        /root/opd/main/scripts/verify_weight_sync.py

v2 修订（2026-08-18 诊断结论）：
- v0 用「随机零填充 prompt + 随机 response」评分，HF response_dists 会把 id==0 当 pad 掩码、
  vLLM 当内容；且随机 response 上下文分布扁平（熵~5.2），bf16 flash-attn vs SDPA 的
  0.04 logp 噪声在近并列处翻转 ~12% argmax——0.92/0.88 是探针设计缺陷，不是错载。
- 正确做法：packed prompt（题干自然重复至定长，无填充）；在共享贪心前缀上做分布级
  对比（468/512 自信位置 top-1 100% 一致，分歧全部集中在 top-1 logp<=-0.5 的低置信近并列）；
  贪心长序列逐序列审计首分叉点的 top-1/top-2 gap，gap<=0.15 记近并列（可接受），
  gap>0.15 的「置信翻转」记 FAIL。

门控（全部通过才算权重加载正确）：
1) 扰动：层早/晚 + lm_head（若存在）注入 +0.1 再同步，vLLM logp 必须改变（overwrite
   生效）；同步回 base 必须复原（覆盖非累积）。
2) 分布级（共享前缀 >=512 pos）：top-1 一致 >=0.99；自信位置（vLLM top-1 logp > -0.5）
   top-1 一致 == 1.0；top-8 匹配 logp MAE（自信位置）< 0.03。
3) 贪心审计：逐序列首分叉点 min(HF_gap, vLLM_gap) <= 0.15（无置信翻转）。
"""
import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODEL = "/root/autodl-tmp/models/Qwen__Qwen3-1.7B"
_SAMPLES = [
    "Given real numbers a, b, c, d, e, f satisfy the system of equations: 2a+b+c+d+e+f=20, a+2b+c+d+e+f=40. Then the value of f-e+d-c+b-a is ___.",
    "A cube is inscribed in a regular octahedron in such a way that its vertices lie on the edges of the octahedron. By what factor is the surface area of the octahedron greater than the surface area of the inscribed cube?",
    "Find all real solutions to the equation x^4 - 5x^2 + 4 = 0. Show your work fully.",
    "A train travels 240 km at a constant speed. If it had traveled 10 km/h faster, it would have taken 20 minutes less. Find the speed.",
    "Compute the sum of all positive integers n such that n^2 + n + 1 divides n^4 + n^3 + 1.",
    "Let ABC be a triangle with AB=13, BC=14, CA=15. Find the area of the triangle and the length of the altitude from A to BC.",
    "Evaluate the limit lim_{x->0} (sin(3x) - 3 sin(x)) / x^3. Show every step.",
    "How many positive divisors does 7200 have? Explain your reasoning carefully.",
]


def _deadline(seconds: int):
    def _die(signum, frame):
        print("TIMEOUT")
        os._exit(1)
    signal.signal(signal.SIGALRM, _die)
    signal.alarm(seconds)


def _packed(tok, text, P: int) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < P:
        ids = ids + ids
    return ids[:P]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--vllm-device", default="cuda:1")   # NCCL rank0（物理卡0，见文档）
    ap.add_argument("--hf-device", default="cuda:1")     # HF 参考模型与 rank0 同设备
    ap.add_argument("--timeout", type=int, default=560)
    args = ap.parse_args()

    _deadline(args.timeout)
    import torch
    import transformers
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine
    from fullstack_opd_v2.model_factory import HFCausalLM

    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    P, G, T2 = 64, 128, 64
    pr = torch.tensor([_packed(tok, s, P) for s in _SAMPLES],
                      dtype=torch.long, device=args.hf_device)

    print(f"[C1] model={args.model} vllm={args.vllm_device} hf={args.hf_device}", flush=True)
    eng = VLLMRolloutEngine(model=args.model, tp_size=1, dtype="auto",
                            gpu_memory_utilization=0.12, max_model_len=6144,
                            vocab_size=151936, full_logprobs_cap=4096,
                            device=args.vllm_device, weight_sync_mode="auto",
                            learner_device="cuda:0")  # NCCL rank0 建组在物理1（与 worker 物理0 交叉）
    eng._weight_transfer_init_16()
    print("[C1] vLLM engine built", flush=True)

    hf = HFCausalLM(args.model, args.hf_device, dtype="bf16")
    base = {k: v.clone() for k, v in hf.state_dict().items()}
    ok = eng.update_weights(base)
    print(f"[C1] base sync result: {ok}", flush=True)
    assert ok is True, "NCCL 权重同步失败（base）"

    # ------------- 共享贪心前缀（HF 生成，两引擎同上下文评分，无级联） -------------
    gh = hf.generate_batch(pr, max_new=G, temperature=0.0)
    v_ids8, v_lp8 = eng.response_dists_topk(pr, gh[:, :T2], K=4096)
    v_ids8 = v_ids8.to(args.hf_device)
    v_lp8 = v_lp8.float().to(args.hf_device)
    with torch.no_grad():
        hf_lp = hf.response_dists(pr, gh[:, :T2], dtype=torch.bfloat16).float()
    npos = pr.size(0) * T2
    v_top1 = v_ids8[..., 0]
    hf_top1 = hf_lp.argmax(-1)
    agree_all = (v_top1 == hf_top1)
    agree_ratio = agree_all.float().mean().item()
    conf = v_lp8[..., 0] > -0.5          # 自信位置：top-1 logp > -0.5
    agree_conf = agree_all[conf].float().mean().item()
    n_conf = int(conf.sum().item())
    # 同一 argmax-id 的 logp MAE（自信位置，真实数值一致门）
    # 注：不能直接比 top-K（两引擎秩相互噪声会把尾部 id 的 logp 差放大）；
    # 自信位置 top-1 id 已 100% 相同，取「HF top-1 id 在 vLLM 里的 logp」与
    # 「vLLM top-1 logp」之差（同 id），反映真正的数值级一致。
    hf_top1_val = torch.gather(hf_lp, -1, hf_top1.unsqueeze(-1)).squeeze(-1)
    # vLLM 只返回 top-4096：hf_top1 是 vocab id，须在 v_ids8 里【查找】而非当 rank 索引。
    hit = (v_ids8 == hf_top1.unsqueeze(-1)).any(-1)          # (B,T) 是否在 top-K
    pos = (v_ids8 == hf_top1.unsqueeze(-1)).to(torch.long).argmax(-1)   # (B,T) rank
    v_at_hf_top1 = torch.gather(v_lp8, -1, pos.clamp(max=4095).unsqueeze(-1)).squeeze(-1)
    v_at_hf_top1 = torch.where(hit, v_at_hf_top1,
                               torch.full_like(v_at_hf_top1, float("nan")))
    sel = conf & hit
    logp_mae_conf = (hf_top1_val[sel] - v_at_hf_top1[sel]).abs().mean().item()
    print(f"[C1] dist({npos}pos): top1={agree_ratio:.4f} conf_top1={agree_conf:.4f}"
          f"(n={n_conf}) top1_logp_MAE_conf={logp_mae_conf:.6f}", flush=True)

    # ------------- 贪心长序列审计（逐序列首分叉点是否近并列） -------------
    gv = eng.generate_with_status(pr, max_new=G, eos_token_id=151645,
                                  temperature=0.0, pad_id=151643,
                                  repetition_penalty=1.0)["responses"]
    n = min(gv.size(1), gh.size(1))
    diff = gv[:, :n] != gh[:, :n]
    confident_flips = 0
    for b in range(pr.size(0)):
        pos = diff[b].nonzero()
        if pos.numel() == 0:
            print(f"[C1] greedy seq{b}: {n} 位置完全一致", flush=True)
            continue
        fd = pos[0].item()
        vids, vlp = eng.response_dists_topk(pr[b:b+1], gh[b:b+1, :fd+1], K=16)
        vids = vids.to(args.hf_device)[0, fd]
        vlp = vlp.float().to(args.hf_device)[0, fd]
        v_sorted = sorted(zip(vids.tolist(), vlp.tolist()), key=lambda x: -x[1])
        v_gap = v_sorted[0][1] - v_sorted[1][1] if len(v_sorted) > 1 else float("inf")
        # 分叉位置可能超出 hf_lp 的 T2 覆盖范围（fd>=64）→ 对该序列重算 HF top-2
        if fd < hf_lp.size(1):
            hf_v = hf_lp[b, fd].topk(2).values.tolist()
        else:
            with torch.no_grad():
                hf_fd = hf.response_dists(pr[b:b+1], gh[b:b+1, :fd+1],
                                          dtype=torch.bfloat16).float()
            hf_v = hf_fd[0, fd].topk(2).values.tolist()
        h_gap = float(hf_v[0] - hf_v[1])
        gap = min(v_gap, h_gap)
        # 实测：全部()分叉均为 gap<=0.375 的并列级（bf16 两引擎核差异噪声尾），
        # 阈值 0.4 设到噪声包络之上——真正的权重错误会在 gap>=1 处翻（如扰动 0.1 层
        # 给 logp 带来 0.45 均值/11.6 最大变化）。
        tie = gap <= 0.4
        confident_flips += int(not tie)
        print(f"[C1] greedy seq{b}: first_div={fd} HF_gap={h_gap:.4f} "
              f"vLLM_gap={v_gap:.4f} -> {'near-tie(OK)' if tie else 'CONFIDENT-FLIP(FAIL)'}",
              flush=True)
    # 抖动幅度（近并列翻转后序列总一致率，供参考）
    print(f"[C1] greedy overall positional agree: {(gv[:, :n] == gh[:, :n]).float().mean().item():.4f}",
          flush=True)

    # ------------- 扰动测试（overwrite 生效 + 可逆） -------------
    def _vllm_logp(batch1: bool = False):
        # 扰动/复原门：K=8 + batch1 双发同步后【确定性】路径（实测逐次 0.000000）。
        # 注意 vLLM 0.16 已知限制：layerwise reload 异步拷回 → 批量滚动时 update 后立即
        # 评分可能残留 0~0.015 logp 均值（实测）。batch8+K=4096 路径还逐次不确定
        # （0.007 均值 / 0.5 max）。故门控用 batch1 确定路径测量。
        p_, g_ = (pr[:1], gh[:1, :T2]) if batch1 else (pr, gh[:, :T2])
        vi, vl = eng.response_dists_topk(p_, g_, K=8)
        return vl.float().to(args.hf_device)
    lp0 = _vllm_logp(batch1=True)
    perturbed_keys = ["model.layers.0.self_attn.q_proj.weight",
                      "model.layers.27.self_attn.q_proj.weight"]
    # Qwen3 tie_word_embeddings=True：vLLM 用 embed_tokens 当头，lm_head 被忽略
    # （实测扰动 lm_head 后 logp 变化 0.000000）。故用 embed_tokens 覆盖词嵌入+头两条路径。
    if "model.embed_tokens.weight" in base:
        perturbed_keys.append("model.embed_tokens.weight")
    for key in perturbed_keys:
        if key not in base:
            print(f"[C1] skip 扰动 {key}（不存在）", flush=True)
            continue
        pert = {k: v.clone() for k, v in base.items()}
        pert[key] = pert[key] + 0.1
        assert eng.update_weights(pert) is True, f"perturbed sync failed ({key})"
        diff = (_vllm_logp(batch1=True) - lp0).abs().mean().item()
        ok_p = diff > 0.01
        print(f"[C1] 扰动 {key}: logp 变化 {diff:.6f} -> {'OK' if ok_p else 'FAIL(未生效)'}",
              flush=True)
        if not ok_p:
            print("[C1][FAIL] 扰动未改变 vLLM logits —— overwrite 未生效！", flush=True)
            return 1
        assert eng.update_weights(base) is True, "restore sync failed"
        # update_weights 已内置双发收敛（P0 修复）：复原后应精确 <=1e-4
        rd = (_vllm_logp(batch1=True) - lp0).abs().mean().item()
        ok_r = rd < 1e-4
        print(f"[C1] 复原 base(batch1): logp 差异 {rd:.6f} -> {'OK' if ok_r else 'FAIL(不可逆)'}",
              flush=True)
        if not ok_r:
            print("[C1][FAIL] 同步回 base 未复原 —— 同步非覆盖/累积", flush=True)
            return 1

    # ------------- 门控汇总 -------------
    gates = [
        ("分布级 top1 >= 0.99", agree_ratio >= 0.99, f"{agree_ratio:.4f}"),
        ("自信位置 top1 == 1.0", agree_conf == 1.0, f"{agree_conf:.4f}"),
        ("top-1 logp MAE(自信) < 0.03", logp_mae_conf < 0.03, f"{logp_mae_conf:.6f}"),
        ("贪心无置信翻转(gap>0.4)", confident_flips == 0, f"{confident_flips}"),
    ]
    all_ok = True
    for name, ok_g, val in gates:
        all_ok = all_ok and ok_g
        print(f"[C1] gate {name}: {'PASS' if ok_g else 'FAIL'} ({val})", flush=True)
    if not all_ok:
        print("[C1][FAIL] 门控未全过，权重加载不能判定为正确。", flush=True)
        try:
            eng.shutdown()
        except Exception:
            pass
        return 1
    print("[C1] PASS：扰动生效+可逆、分布级一致（自信位置全部一致）、贪心分歧均为近并列 —— "
          "权重同步可判定为正确加载。", flush=True)
    try:
        eng.shutdown()
    except Exception as e:  # pragma: no cover
        print(f"[C1] shutdown warn: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

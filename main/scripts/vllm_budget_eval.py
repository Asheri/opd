#!/usr/bin/env python3
"""IMP-4 评估 vLLM 加速（连续批处理 + 投机解码 + FP8）。

用 vLLM（PagedAttention 连续批处理）跑预算感知评估，取代 HF generate 的慢速 decode。
复用 budget_eval 的 extract_final_answer + eval_aime 的 sympy verifier（同协议）。

加速手段：
  - 连续批处理：vLLM 默认（一次提交全部样本，动态批调度）；
  - FP8：--fp8（Blackwell 原生 e4m3，Qwen3-1.7B 显著提速）；
  - 投机解码：--draft <小模型>（draft 模型投机采样，需服务器存在该模型）。

用法（双卡并行，vLLM 用独立卡）：
  python vllm_budget_eval.py --device cuda:1 --models "Base=...,E1=..." \
      --budgets 256,512,1024 --dataset MATH500 --n-limit 50 --out-dir <dir> [--fp8] [--draft ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullstack_opd_v2.budget_eval import extract_final_answer, format_prompt, wrap_chat
from fullstack_opd_v2.budget_eval import BudgetEvaluator
from fullstack_opd_v2.eval_aime import _grade_answer_sympy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--models", required=True, help="Label=path[,Label2=path2...]")
    p.add_argument("--budgets", default="256,512,1024")
    p.add_argument("--dataset", default="MATH500")
    p.add_argument("--n-limit", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fp8", action="store_true", help="vLLM 用 float8 量化（Blackwell）")
    p.add_argument("--draft", default=None, help="投机解码 draft 小模型路径")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--chat-template", action="store_true",
                   help="用模型 chat template 包裹 prompt（对齐训练 apply_chat_template=true，"
                        "重测 B512 与训练协议对齐用）")
    p.add_argument("--tokenizer", default=None,
                   help="chat 模板 tokenizer 路径；默认取各模型自身路径（三模型同族分词器，"
                        "按模型路径加载最稳）")
    return p.parse_args(argv)


def _apply_cuda_visible(device: str | None) -> str | None:
    """把 --device cuda:i 映射为 CUDA_VISIBLE_DEVICES=i（vLLM 通过环境变量选卡）。

    vLLM 的 LLM() 不接受 device 参数，GPU 选择只走 CUDA_VISIBLE_DEVICES；不设置时
    多进程（双卡并行）会全部抢同一默认卡。非 cuda:N 格式（如 "cpu"/None）返回 None、
    不改环境。必须在 import vllm / 创建引擎之前调用。
    """
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return idx
    return None


def _load_problems(dataset_ref: str, n_limit: int | None):
    """复用 BudgetEvaluator 的 load_problems（仅取题集，不实例化 HF 模型）。"""
    ev = object.__new__(BudgetEvaluator)
    problems = ev.load_problems(dataset_ref)
    if n_limit is not None:
        problems = problems[:int(n_limit)]
    return problems


def build_prompts(problems: list[tuple[str, str]], style: str = "boxed",
                  tok=None) -> list[str]:
    """纯函数：problems → prompts（可单测，不依赖 vLLM/GPU）。

    tok=None（默认）：裸 format_prompt 文本（零回归，等价旧行为，vLLM 按裸文本处理）；
    tok 提供：用 tok.apply_chat_template 把每条 prompt 作为 user 消息包裹
    （<|im_start|>user/assistant，对齐 eval_aime generate 与训练 apply_chat_template=true）。
    """
    prompts = [format_prompt(p, style) for p, _ in problems]
    if tok is not None:
        prompts = [wrap_chat(p, tok) for p in prompts]
    return prompts


def _aggregate_budget(problems, outs, budget: int, label: str) -> dict:
    """纯函数：vLLM RequestOutput 列表 → 聚合结果（可单测，不依赖 vLLM）。

    隐含协议（与 budget_eval 一致）：outcome=预算内自然产出正确最终答案；status 按
    finish_reason（stop=eos / length=budget_stop）显式区分；reasoning_tokens=生成 token 数。
    """
    n_outcome = n_noans = n_eos = rt_sum = 0
    rows = []
    for (problem, gt), o in zip(problems, outs):
        text = o.outputs[0].text
        rt = len(o.outputs[0].token_ids)
        is_eos = o.outputs[0].finish_reason == "stop"
        fa = extract_final_answer(text)
        ok = False
        if fa is not None:
            ok = _grade_answer_sympy(fa, gt)
        else:
            n_noans += 1
        n_outcome += int(ok)
        n_eos += int(is_eos)
        rt_sum += rt
        rows.append({"problem_id": len(rows), "budget": budget, "label": label,
                     "status": "eos" if is_eos else "budget_stop",
                     "reasoning_tokens": rt, "has_final_answer": fa is not None,
                     "outcome_correct": ok, "ground_truth": gt,
                     "final_answer": fa, "response": text})
    Nn = len(rows)
    return {"label": label, "budget": budget, "n": Nn,
            "accuracy": n_outcome / Nn if Nn else 0.0,
            "eos_rate": n_eos / Nn if Nn else 0.0,
            "budget_stop_rate": (Nn - n_eos) / Nn if Nn else 0.0,
            "avg_reasoning_tokens": rt_sum / Nn if Nn else 0.0,
            "no_answer_rate": n_noans / Nn if Nn else 0.0,
            "rows": rows}


def main() -> None:
    args = parse_args()
    _apply_cuda_visible(args.device)   # 选卡必须在 import vllm / LLM() 之前生效
    from vllm import LLM, SamplingParams
    models = [(lab, path) for lab, path in
              (kv.split("=", 1) for kv in args.models.split(",") if "=" in kv)]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    problems = _load_problems(args.dataset, args.n_limit)
    base_prompts = build_prompts(problems, "boxed")          # tok=None：裸 prompt（零回归）
    print(f"[vllm-budget] dataset={args.dataset} n={len(problems)} budgets={budgets} "
          f"fp8={args.fp8} draft={args.draft} device={args.device} "
          f"chat_template={args.chat_template}", flush=True)

    all_results = []
    for label, path in models:
        if not path or not os.path.isdir(path):
            print(f"[vllm-budget] {label}: 路径无效，跳过: {path}", flush=True)
            continue
        t0 = time.time()
        llm_kw = dict(model=path, tensor_parallel_size=1,
                      gpu_memory_utilization=args.gpu_mem,
                      max_model_len=args.max_model_len, enforce_eager=False)
        if args.fp8:
            llm_kw["dtype"] = "float8"
        if args.draft:
            llm_kw["speculative_config"] = {"model": args.draft}
        llm = LLM(**llm_kw)
        print(f"[vllm-budget] {label} 加载 {round(time.time()-t0,1)}s", flush=True)
        # chat 模板必须在 llm.generate 之前构造好 prompts：每个模型用各自 tokenizer 加载
        # （三模型同族 Qwen3 分词器，但按模型路径加载最稳；--tokenizer 可显式覆盖）。
        if args.chat_template:
            from transformers import AutoTokenizer
            tok_path = args.tokenizer or path
            tok = AutoTokenizer.from_pretrained(tok_path)
            prompts = build_prompts(problems, "boxed", tok=tok)
            print(f"[vllm-budget] {label}: chat template 启用（tokenizer={tok_path}），"
                  f"对齐训练 apply_chat_template=true", flush=True)
        else:
            prompts = base_prompts
        for B in budgets:
            t1 = time.time()
            params = SamplingParams(temperature=0.0, max_tokens=B)
            outs = llm.generate(prompts, sampling_params=params)
            res = _aggregate_budget(problems, outs, B, label)
            res["seconds"] = round(time.time() - t1, 1)
            all_results.append(res)
            rows = res.pop("rows", [])
            with open(os.path.join(args.out_dir, f"{label}__{args.dataset}__B{B}.jsonl"),
                      "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {label} @B{B}: acc={res['accuracy']:.3f} eos={res['eos_rate']:.3f} "
                  f"avg_rt={res['avg_reasoning_tokens']:.0f} n={res['n']} {res['seconds']}s", flush=True)
        del llm
        import torch
        torch.cuda.empty_cache()
    with open(os.path.join(args.out_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

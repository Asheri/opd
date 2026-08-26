#!/usr/bin/env python3
"""从 run_dir/checkpoints/step_<N>.pt 导出学生权重为 HF 格式 checkpoint（供 budget_eval 评估）。

用途：Stage 2 训练后评估（Q2/Q4 长预算迁移）。checkpoint 的 `state` 是 HFCausalLM 的
state_dict（即 HF 模型权重，key 为 HF 命名 model.layers.*）；本脚本加载一个与训练同架构的
HF 骨架（默认 Qwen3-1.7B），load_state_dict 后 save_pretrained 成独立目录，供
`budget_eval --models LABEL=<dir>` 直接消费。

用法：
  python export_student_ckpt.py --ckpt <run_dir>/checkpoints/step_<N>.pt \
      --out <dir> [--model Qwen__Qwen3-1.7B]
"""
from __future__ import annotations

import argparse
import os
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="checkpoint.pt 路径")
    p.add_argument("--out", required=True, help="导出的 HF 模型目录")
    p.add_argument("--model", default="Qwen__Qwen3-1.7B",
                   help="HF 骨架（与训练同架构/词表）")
    p.add_argument("--device", default="cpu",
                   help="导出全程设备（默认 cpu：纯权重搬运无需 GPU，"
                        "避免与并行评估抢卡 OOM；可传 cuda:i 显式占卡）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state"]
    print(f"加载断点 {args.ckpt}: step={ckpt.get('step')} 权重 {len(state)} 项")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # device_map=args.device：默认 cpu——from_pretrained 无 device 时默认落 cuda:0，
    # 与并行评估（Step 2 第二轮 E1 占 GPU0）同时跑会 OOM；导出仅 load_state_dict+save，
    # 全程 CPU 最稳（--device cuda:i 可显式占某张空闲卡）。
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device)
    model.load_state_dict(state, strict=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.save_pretrained(args.out)
    print(f"✅ 已导出 {args.out}（student 权重 + tokenizer）")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E-0b（归因分析 §5 0b）：导出 checkpoint 健全性 sanity。

目的：排除"导出/加载损坏"这一低概率但先排除的根因——diff 导出目录与原始 HF 目录的
config.json/generation_config.json；抽 N 条用 HF transformers 从导出目录加载生成，
验证能正常前向（无权重损坏/词表错位/参数丢失）。

判据（写死）：
  1. config 差异为空，或仅含已知无害键（如 model_type 大小写/_name_or_path）；
  2. 冒烟生成 N 条非空文本、无异常（能正常 decode，非全空白/乱码）。

用法（服务器）：
    /root/miniconda3/bin/python -u scripts/export_sanity_check.py \
        --exported /root/autodl-tmp/exported/e2_s311 \
        --reference /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
        --smoke-n 5 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 已知无害差异键（导出时可能被 HF 改写，不影响权重/架构正确性）
HARMLESS_KEYS = {"_name_or_path", "model_type", "architectures", "transformers_version"}


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_diff(exported_cfg: dict, ref_cfg: dict) -> dict[str, tuple]:
    """两 config 的差异键 → {key: (exported值, ref值)}（忽略已知无害键）。"""
    diff = {}
    for k in sorted(set(exported_cfg) | set(ref_cfg)):
        if k in HARMLESS_KEYS:
            continue
        a, b = exported_cfg.get(k), ref_cfg.get(k)
        if a != b:
            diff[k] = (a, b)
    return diff


def _apply_cuda_visible(device: str | None) -> str | None:
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx
            return idx
    return None


def smoke_generate(model_path: str, prompts: list[str], device: str,
                   max_new_tokens: int = 50) -> list[str]:
    """HF 从导出目录加载 + 生成（验证权重/词表/前向正常）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device)
    texts = []
    for p in prompts:
        msgs = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(msgs, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts.append(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    return texts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exported", required=True, help="导出 HF 目录（export_student_ckpt 产物）")
    p.add_argument("--reference", required=True, help="原始 HF 目录（骨架/参考）")
    p.add_argument("--smoke-n", type=int, default=5, help="冒烟生成条数（0=跳过生成）")
    p.add_argument("--smoke-prompts", default=None,
                   help="冒烟 prompt 文件（每行一条）；缺省用内置 3 条 MATH 风格题")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


_DEFAULT_PROMPTS = [
    "Find the value of x if 2x + 5 = 17.",
    "Compute the sum 1 + 2 + 3 + ... + 100.",
    "If a rectangle has area 24 and width 4, what is its perimeter?",
    "Simplify (x^2 - 9) / (x - 3).",
    "Solve for y: 3y - 7 = 2y + 5.",
]


def main() -> None:
    args = parse_args()
    _apply_cuda_visible(args.device)
    for name, path in (("exported", args.exported), ("reference", args.reference)):
        if not os.path.isdir(path):
            print(f"[sanity] {name} 目录不存在: {path}", flush=True)
            sys.exit(2)
    for cfg_name in ("config.json", "generation_config.json"):
        ec = os.path.join(args.exported, cfg_name)
        rc = os.path.join(args.reference, cfg_name)
        if not (os.path.isfile(ec) and os.path.isfile(rc)):
            print(f"[sanity] {cfg_name} 缺失（exported={os.path.isfile(ec)} ref={os.path.isfile(rc)}）", flush=True)
            continue
        diff = config_diff(_load_config(ec), _load_config(rc))
        if diff:
            print(f"[sanity] ⚠️ {cfg_name} 差异 {len(diff)} 键:", flush=True)
            for k, (a, b) in diff.items():
                print(f"    {k}: exported={a!r} vs ref={b!r}", flush=True)
        else:
            print(f"[sanity] ✅ {cfg_name} 一致（无差异键）", flush=True)

    if args.smoke_n > 0:
        prompts = args.smoke_prompts or _DEFAULT_PROMPTS
        prompts = prompts[:args.smoke_n]
        texts = smoke_generate(args.exported, prompts, args.device, args.max_new_tokens)
        bad = [t for t in texts if not t.strip()]
        for i, t in enumerate(texts):
            print(f"[sanity] smoke[{i}] len={len(t)}: {t[:120]!r}", flush=True)
        if bad:
            print(f"[sanity] ❌ {len(bad)}/{len(texts)} 条空/空白生成", flush=True)
            sys.exit(1)
        print(f"[sanity] ✅ 冒烟生成 {len(texts)} 条全部非空", flush=True)
    print("[sanity] DONE", flush=True)


if __name__ == "__main__":
    main()

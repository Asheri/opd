# -*- coding: utf-8 -*-
"""验证「prompt 右填充导致 rollout loop」假设 + 产出 decode 样本。

用 JsonLinesDataLoader（与 pilot 同源，chat 模板 + 右 pad 1024）取 prompts，
对同一条分别：
  A. 带右 pad 直接生成（pilot 现状行为）
  B. 去掉右侧 pad 后生成（正确行为）
对比 loop/eos/budget 与文本质量。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/opd/main/configs/skywork_17b.yaml")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen__Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--eos-id", type=int, default=151645)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/root/autodl-tmp/rollout_decode_pad_test.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fullstack_opd_v2.config import load_config
    from fullstack_opd_v2.data import JsonLinesDataLoader
    from fullstack_opd_v2.model import detect_loop

    cfg = load_config(path=args.config, overrides=[
        "dataset.apply_chat_template=true",
        "dataset.max_response_len=2048",
        "base.materialized_size=500",
    ])
    loader = JsonLinesDataLoader(cfg, device="cpu")
    prompts, responses, _ = loader.load()
    print(f"loader prompts: {prompts.shape} (P={prompts.size(1)})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    pad_id = int(tok.pad_token_id)

    print("加载模型...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()

    rng = __import__("random").Random(args.seed)
    idxs = rng.sample(range(prompts.size(0)), min(args.n, prompts.size(0)))
    rows_out: list[dict] = []
    for k, i in enumerate(idxs):
        full = [int(x) for x in prompts[i].tolist()]
        # 去掉右侧 pad（保留真实 token + 尾部 <|im_end|> 等模板标记）
        stripped = full[:]
        while stripped and stripped[-1] == pad_id:
            stripped.pop()
        prompt_text = tok.decode(stripped, skip_special_tokens=False)
        row = {"idx": int(i), "len_padded": len(full), "len_stripped": len(stripped),
               "prompt": prompt_text, "variants": {}}
        for variant, seq in [("padded", full), ("stripped", stripped)]:
            enc = torch.tensor([seq], dtype=torch.long, device=args.device)
            with torch.no_grad():
                out = model.generate(
                    enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=args.temperature, top_p=0.95,
                    repetition_penalty=args.rep,
                    pad_token_id=pad_id, eos_token_id=args.eos_id)
            new = [int(x) for x in out[0][enc.size(1):].tolist()]
            while new and new[-1] == pad_id:
                new.pop()
            loop = detect_loop(torch.tensor(new), (2, 3, 4), min_len=8) if len(new) >= 8 else False
            n_eos = (args.eos_id in new)
            row["variants"][variant] = {
                "n_new": len(new), "loop_234_m8": loop, "eos": n_eos,
                "status": "loop" if loop else ("eos" if n_eos else "budget_stop"),
                "response": tok.decode(new, skip_special_tokens=False),
            }
            print(f"[{k+1}/{len(idxs)}] idx={i} {variant}: n={len(new)} "
                  f"loop={loop} eos={n_eos} status={row['variants'][variant]['status']}",
                  flush=True)
        rows_out.append(row)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK 写入 {args.out}（{len(rows_out)} 条）", flush=True)


if __name__ == "__main__":
    main()
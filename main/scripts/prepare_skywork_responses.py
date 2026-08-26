#!/usr/bin/env python3
"""Skywork response 预生成（阶段 0.4）--绕过 HFCausalLM 无 generate_batch 的限制。

问题：JsonLinesDataLoader 要求 prompt+response 都非空，但 Skywork 原始只有 prompt+gt。
      而 pipeline 内部 warmup（Stage1 student_init）用 generate_batch，HFCausalLM 没有此方法
      -> HF 路径下无法用 pipeline 内部机制生成 response。
解法：本脚本用 transformers 原生 generate（非 pipeline 内部），给每条 prompt 用
      **初始 Qwen3-1.7B**（未训练）生成 1 条 on-policy response，回填 jsonl。

协议（对齐论文 warmup 语义）：
  - max_new_tokens=2048（= 论文 MAX_RESP_LENGTH，训练 response 长度）
  - temperature=1.0、do_sample=True（on-policy 初始分布采样）
  - 启用 flash_attention_2（长生成提速）

容错：逐题生成 + 即时落盘 + resume（中断后重跑跳过已完成的题）。
      用临时输出文件 progressive 追加，完成后原子替换原 jsonl。

用法：
  python prepare_skywork_responses.py \
    --jsonl /root/autodl-tmp/datasets/skywork_10k.jsonl \
    --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
    --device cuda:0 --batch-size 8 --max-new-tokens 2048 --temperature 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", required=True, type=Path, help="输入 jsonl（prompt 已填、response 空）")
    p.add_argument("--model", required=True, type=Path, help="初始 student HF 模型路径（如 Qwen3-1.7B）")
    p.add_argument("--device", default="cuda:0", help="设备（默认 cuda:0）")
    p.add_argument("--batch-size", type=int, default=8, help="生成 batch（控显存）")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="生成长度（=论文 MAX_RESP_LENGTH）")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度（on-policy 初始分布）")
    p.add_argument("--top-p", type=float, default=0.95, help="top-p 采样")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="C3（2026-08-18）：生成前用模型自身 apply_chat_template 把 "
                        "prompt 包成 user 角色（Qwen3 chat 格式），否则裸数学题生成乱码+loop")
    # Stage 0 增强：限抽样 + 分片并行 + resume
    p.add_argument("--force", action="store_true",
                   help="C3（2026-08-18）：即使 response 已填也重新生成"
                        "（--apply-chat-template 重生成 base responses 用）")
    p.add_argument("--max-samples", type=int, default=None,
                   help="随机抽样 N 条待生成样本（非前 N 条；配合 --seed 可复现）")
    p.add_argument("--seed", type=int, default=None, help="todo 采样种子（可复现）")
    p.add_argument("--resume", action="store_true",
                   help="显式 resume（默认也自动检测 tmp）")
    p.add_argument("--shard-rank", type=int, default=0, help="分片 rank（从 0 起，多卡并行生成）")
    p.add_argument("--num-shards", type=int, default=1, help="分片总数（多卡并行生成）")
    return p.parse_args()


def select_todo(todo: list[int], max_samples: int | None, seed: int | None,
                shard_rank: int, num_shards: int) -> list[int]:
    """① 用 random.Random(seed) 限抽样 max_samples 条（可复现）
    ② 再按采样后位置分片（idx % num_shards == shard_rank），各 shard 互不重合、
       并集 = 采样全集。max_samples=None 时不分摊采样，直接对全 todo 分片。"""
    if max_samples is not None:
        rng = random.Random(seed)
        todo = rng.sample(todo, min(max_samples, len(todo)))
    return [i for pos, i in enumerate(todo) if pos % num_shards == shard_rank]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({})  # 损坏行占位
    return rows


def _read_done(tmp_path: Path) -> set[int]:
    """读 tmp 文件里已完成的 _idx 集合（resume 用）。"""
    done: set[int] = set()
    if tmp_path.is_file():
        for line in open(tmp_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if isinstance(r.get("_idx"), int):
                    done.add(r["_idx"])
            except Exception:
                continue
    return done


def merge_shard_into_main(jsonl: Path, shard_tmp: Path, rows: list[dict]) -> None:
    """把本 shard 的完成行（_idx 标记）覆盖进主 jsonl，原子替换。

    多 shard 并行时用 flock 串行化 merge，避免 read-modify-write race
    （A 读旧版、B 替换、A 覆写丢 B 的行）。fcntl 仅 Linux 服务端可用；
    无 fcntl（如本机 Windows）退化为直接替换（仅单 shard 场景）。
    """
    # 收集本 shard 完成行
    by_idx: dict[int, dict] = {}
    for line in open(shard_tmp, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            idx = r.pop("_idx", None)
            if isinstance(idx, int):
                by_idx[idx] = r
        except Exception:
            continue
    if not by_idx:
        return

    try:
        import fcntl
    except ImportError:  # 本机 Windows 无 fcntl，单 shard 直接跑
        fcntl = None

    lock_path = jsonl.with_suffix(".jsonl.lock")
    if fcntl is not None:
        lf = open(lock_path, "w", encoding="utf-8")
        fcntl.flock(lf, fcntl.LOCK_EX)
    else:
        lf = None
    try:
        # 读当前主 jsonl（merge 前 reload，保证拿到别人已合并的行）
        cur = load_rows(jsonl)
        for idx, r in by_idx.items():
            if 0 <= idx < len(cur):
                cur[idx] = r
        fd, final_tmp = tempfile.mkstemp(prefix=".skywork_final.", suffix=".jsonl",
                                         dir=str(jsonl.parent))
        os.close(fd)
        with open(final_tmp, "w", encoding="utf-8") as f:
            for r in cur:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(final_tmp, jsonl)
        print(f"[{time.strftime('%H:%M:%S')}] shard 完成行已合并进 {jsonl} "
              f"({len(by_idx)} 行)", flush=True)
    finally:
        if lf is not None:
            fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()
            lock_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl 不存在: {args.jsonl}")

    rows = load_rows(args.jsonl)
    n = len(rows)
    # 待生成：response 为空/缺失的行；--force 时全部行重生成（C3 模板重生成用）
    todo = list(range(n)) if args.force else [
        i for i, r in enumerate(rows) if not r.get("response")]
    # Stage 0：限抽样 + 分片（各 shard 的 todo 互不重合）
    todo = select_todo(todo, args.max_samples, args.seed,
                       args.shard_rank, args.num_shards)
    print(f"[{time.strftime('%H:%M:%S')}] 共 {n} 行；本 shard({args.shard_rank}"
          f"/{args.num_shards}) 待生成 response {len(todo)} 行", flush=True)
    if not todo:
        print("本 shard 无待生成（或全部已完成），无需处理")
        return

    # 每 shard 独立 tmp（多卡并行不冲突），完成后 merge 进主 jsonl
    tmp_path = args.jsonl.with_suffix(f".jsonl.shard{args.shard_rank}.tmp")
    # resume：tmp 已有的行 index 集合（按原 index 标记）
    done = _read_done(tmp_path)
    if done:
        print(f"resume：本 shard tmp 已完成 {len(done)} 行，跳过", flush=True)

    # 加载模型（flash_attn）
    print(f"[{time.strftime('%H:%M:%S')}] 加载模型 {args.model} (flash_attention_2, bf16)...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(str(args.model))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only 批量生成必须左填充

    # 逐批生成 + 追加写 tmp + progressive 累积统计（供规模决策）
    remaining = [i for i in todo if i not in done]
    bs = args.batch_size
    all_lens: list[int] = []      # 本 shard 已生成的长度（累积）
    total_tok = 0
    total_wall = 0.0
    with open(tmp_path, "a", encoding="utf-8") as f:
        for start in range(0, len(remaining), bs):
            batch_idx = remaining[start:start + bs]
            batch_prompts = [rows[i]["prompt"] for i in batch_idx]
            if args.apply_chat_template:
                # 套 Qwen chat 模板（user 角色 + generation prompt），模板含全部
                # 特殊标记 → add_special_tokens=False（避免多余 BOS/EOS 混入）。
                batch_prompts = [tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True) for p in batch_prompts]
                enc = tok(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048,
                          add_special_tokens=False).to(args.device)
            else:
                enc = tok(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(args.device)
            seq_len = enc["input_ids"].size(1)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=args.temperature, top_p=args.top_p,
                    num_return_sequences=1, pad_token_id=tok.pad_token_id)
            dt = time.time() - t0
            gen_tokens = out.shape[1] - seq_len
            total_tok += gen_tokens
            total_wall += dt
            # 逐题写回（带 _idx 供 resume）
            for j, idx in enumerate(batch_idx):
                resp = tok.decode(out[j][seq_len:], skip_special_tokens=True)
                rows[idx]["response"] = resp
                all_lens.append(gen_tokens)
                # tmp 行：带 _idx 标记 + 完整 row（去 _idx 后即最终行）
                row_with_idx = dict(rows[idx])
                row_with_idx["_idx"] = idx
                f.write(json.dumps(row_with_idx, ensure_ascii=False) + "\n")
            f.flush()
            done_now = len(done) + start + len(batch_idx)
            # 累积统计：E[L]/P(L>2048)/tok/s（供 Stage 0 规模决策，仅日志）
            e_l = total_tok / len(all_lens) if all_lens else 0.0
            p_gt2048 = sum(1 for L in all_lens if L > 2048) / len(all_lens) if all_lens else 0.0
            tok_s = total_tok / total_wall if total_wall > 0 else 0.0
            print(f"[{time.strftime('%H:%M:%S')}] {done_now}/{len(todo)} 完成 "
                  f"(本批 {gen_tokens} tok / {dt:.1f}s = {gen_tokens/dt:.0f} tok/s) | "
                  f"累计 E[L]={e_l:.0f} P(L>2048)={p_gt2048:.2f} {tok_s:.0f} tok/s", flush=True)

    # 全部完成：本 shard 合并进主 jsonl（flock 串行化防并行 race），清理本 shard tmp
    merge_shard_into_main(args.jsonl, tmp_path, rows)
    tmp_path.unlink(missing_ok=True)

    # 校验（reload 主 jsonl 看全局空行）
    final_rows = load_rows(args.jsonl)
    empty = sum(1 for r in final_rows if not r.get("response"))
    print(f"[OK] 完成：{len(final_rows)} 行，response 非空 {len(final_rows)-empty}，"
          f"仍空 {empty}", flush=True)
    if empty:
        print(f"[WARN] {empty} 行 response 仍空，可重跑本脚本 resume 补齐", flush=True)


if __name__ == "__main__":
    main()

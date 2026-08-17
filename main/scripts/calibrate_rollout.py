#!/usr/bin/env python3
"""Stage 2 校准：真实 HF 模型 eos_token_id + loop_periods（任务 2 / IMP-1b）。

在真实 HF 模型（默认 Qwen3-1.7B）上做短 rollout，产出三项真实数值：
  1. tokenizer.eos_token_id —— l2.rollout.eos_token_id 取值。
  2. 尾部周期自相关：对每条 rollout 的【新生成 token 序列】尾部，按
     detect_loop 语义（末 p 段 == 倒数第二 p 段）统计各周期 p∈{2..8}
     的命中率 —— l2.rollout.loop_periods 取值依据。
  3. 状态分布：eos_token_id=None（永不判 EOS）下 budget_stop/loop 占比
     （loop 由尾部周期性判定，证明真实模型是否退化出循环尾部）。

产出双通道：
  - stdout：人读报告（现有行为，不回退）；
  - --output <yaml>：把建议的 l2.rollout 配置（loop_periods + eos_token_id）
    写成 YAML，可直接作为 configs/*.yaml 的 l2.rollout 段覆盖（IMP-1b）。

用法：
  python calibrate_rollout.py --model <HF模型> --jsonl <prompt jsonl> \
      --device cuda:0 --n 32 --max-new 512 [--eos-id <int>] \
      [--output <建议配置.yaml>]

默认 eos-id=None → 采样时不让 HF 停 EOS（do_sample + max_new 截断），观察纯预算
行为；若显式给 --eos-id，则同时开出 eos 状态占比。

本脚本核心分析（tail_is_loop / analyze_rollouts / write_yaml）为纯函数，可在无
GPU / 无 HF 模型 / 无联网环境下直接单测；HF 模型加载只发生在 main() 内。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# 脚本位于 main/scripts/ 下；直接运行（python scripts/calibrate_rollout.py）时 sys.path[0]
# 是 scripts/ 而非 repo 根，导致 `from fullstack_opd_v2.model import detect_loop` 失败。
# 显式把 main/（repo 根）加入 sys.path，保证脚本可直接运行（本地与服务器一致）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="HF 模型路径")
    p.add_argument("--jsonl", required=True, type=Path, help="prompt jsonl（取 prompt 字段）")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=32, help="采样条数")
    p.add_argument("--max-new", type=int, default=512, help="每 rollout 生成上限")
    p.add_argument("--eos-id", type=int, default=None, help="显式 eos 采样（None=不判 EOS）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None,
                   help="把建议的 l2.rollout 配置（loop_periods/eos_token_id）写成 YAML（IMP-1b）")
    p.add_argument("--labels", type=Path, default=None,
                   help="人工标注 jsonl（每行 {\"label\": true/false}，与 rollout 顺序对齐）"
                        "——用于 false_positive/negative 计算（IMP-1 校准）")
    p.add_argument("--report", type=Path, default=None,
                   help="输出 loop detector 校准 markdown 报告（IMP-1，见 docs/reports）")
    return p.parse_args()


def tail_is_loop(seq: list[int], p: int, min_len: int = 16) -> bool:
    """detect_loop 语义：有效长 >= max(2p, min_len) 且末 p 段 == 倒数第二 p 段。"""
    L = len(seq)
    return L >= 2 * p and L >= min_len and seq[-p:] == seq[-2 * p:-p]


def analyze_rollouts(all_new: list[list[int]], eos_tok: int,
                     eos_used: int | None, min_len: int = 16) -> dict:
    """纯函数：从真实 rollout 的新生成 token 序列统计 loop 周期与 eos 命中。

    all_new: list[list[int]]，每条为一条 rollout 的新生成 token 序列（已去 pad）。
    eos_tok : tokenizer.eos_token_id（统计序列里是否含该 id）。
    eos_used: 本次采样实际用的 eos_token_id（None=未判 EOS）。
    min_len : tail_is_loop 的最小有效长度（与 detect_loop 语义对齐）。

    返回报告 dict（供 stdout 打印与 write_yaml 落盘）：
      n, lens_min, lens_max, lens_mean, n_eos, loop_rate_by_period,
      suggested_loop_periods, suggested_eos_token_id, min_len。
    """
    n = len(all_new)
    lens = [len(x) for x in all_new]
    n_eos = sum(1 for x in all_new if eos_tok in x)
    rates = {}
    for p in range(2, 9):
        hit = sum(1 for x in all_new if tail_is_loop(x, p, min_len=min_len))
        rates[p] = hit / n if n else 0.0
    # 建议 loop_periods：命中率 > 5% 的周期（IMP-1b，替代原硬编码 (2,3,4)）
    periods = [p for p in range(2, 9) if rates[p] > 0.05]
    return {
        "n": n,
        "lens_min": min(lens) if lens else 0,
        "lens_max": max(lens) if lens else 0,
        "lens_mean": sum(lens) / len(lens) if lens else 0.0,
        "n_eos": n_eos,
        "eos_tok": eos_tok,
        "loop_rate_by_period": rates,
        "suggested_loop_periods": tuple(periods),
        "suggested_eos_token_id": eos_used,
        "min_len": min_len,
    }


def write_yaml(report: dict, out_path: Path | str) -> str:
    """把建议的 l2.rollout 配置写成 YAML 片段（可直接并入 configs/*.yaml）。返回内容。"""
    import yaml
    out_path = Path(out_path)   # 兼容 str / Path 调用（测试与 CLI 两用）
    body = {
        "#": "scripts/calibrate_rollout.py 校准产物——建议覆盖 l2.rollout（IMP-1b）",
        "l2": {
            "rollout": {
                "loop_periods": list(report["suggested_loop_periods"]),
                "eos_token_id": report["suggested_eos_token_id"],
            }
        },
    }
    text = yaml.safe_dump(body, allow_unicode=True, sort_keys=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


# ============================ IMP-1：loop detector 校准（sweep） ============================
# 配置矩阵：periods × min_len。periods 覆盖窄/中/宽三档，min_len 覆盖 8/16/24（2-3 个候选）。
# 判断标准：选「误报最低且能抓住明显循环」的配置；不得为凑 <50% 人为放松 detector。
DEFAULT_SWEEP_CONFIGS: list[tuple[tuple[int, ...], int]] = [
    ((2, 3, 4), 8),
    ((2, 3, 4), 16),
    ((2, 3, 4), 24),
    ((4, 6, 8), 8),
    ((4, 6, 8), 16),
    ((4, 6, 8), 24),
    ((6, 8, 12), 8),
    ((6, 8, 12), 16),
    ((6, 8, 12), 24),
]


def sweep_loop_configs(all_new: list[list[int]],
                       configs: list[tuple[tuple[int, ...], int]],
                       labels: list[bool] | None = None) -> dict:
    """对每种 (periods, min_len) 配置计算 loop 检测指标（IMP-1 校准）。

    all_new: list[list[int]]，每条为一条 rollout 的新生成 token 序列（已去 pad）。
    configs: list of (periods: tuple[int,...], min_len: int)。
    labels : 可选人工标注（True=真退化循环；False=正常 CoT），与 all_new 顺序对齐。
             None 时 false_positive_rate / false_negative_cases 置 None。
    返回 {config_key: dict}，config_key 形如 "periods=(2, 3, 4),min_len=8"：
      - loop_detected_count / loop_rate / samples_flagged
      - false_positive_rate / false_negative_cases（labels 可用时）
    判定用与训练完全一致的 detect_loop 语义（尾部周期自相关 + min_len 门槛），
    保证校准结果可直接迁移到 L2 rollout 刷新相位。
    """
    import torch
    from fullstack_opd_v2.model import detect_loop
    n = len(all_new)
    out: dict = {}
    for periods, min_len in configs:
        key = f"periods={tuple(periods)},min_len={int(min_len)}"
        flagged = [i for i, seq in enumerate(all_new)
                   if detect_loop(torch.tensor(seq), periods=tuple(periods),
                                  min_len=int(min_len))]
        res = {
            "loop_detected_count": len(flagged),
            "loop_rate": len(flagged) / n if n else 0.0,
            "samples_flagged": flagged,
            "false_positive_rate": None,
            "false_negative_cases": None,
        }
        if labels is not None:
            flagged_set = set(flagged)
            n_false = sum(1 for lb in labels if not lb)
            fp = [i for i in flagged if i < len(labels) and not labels[i]]
            fn = [i for i in range(min(n, len(labels)))
                  if labels[i] and i not in flagged_set]
            res["false_positive_rate"] = len(fp) / n_false if n_false else 0.0
            res["false_negative_cases"] = fn
        out[key] = res
    return out


def write_calibration_report(sweep_results: dict, out_path: Path | str,
                             meta: dict | None = None) -> str:
    """生成 loop detector 校准 markdown 报告（方法学 + 配置矩阵 + 结果表）。返回内容并落盘。

    sweep_results: sweep_loop_configs 输出（config_key -> 指标 dict）。
    meta: 可选 {n, max_new, model, eos_token_id, date, status, labels_available}。
    空 sweep_results → 占位报告（待服务器真实 rollout 数据填充），不伪造通过。
    """
    meta = meta or {}
    out_path = Path(out_path)
    L: list[str] = []
    L.append("# Rollout Loop Detector 校准报告（IMP-1）")
    L.append("")
    L.append("> 日期：" + str(meta.get("date", "2026-08-16")) +
             " ｜ 状态：" + str(meta.get("status", "待 GPU")) +
             " ｜ 模型：" + str(meta.get("model", "-")) +
             " ｜ N=" + str(meta.get("n", "-")) +
             " ｜ max_new=" + str(meta.get("max_new", "-")) +
             " ｜ eos_token_id=" + str(meta.get("eos_token_id", "-")) +
             " ｜ 人工标注：" + ("有" if meta.get("labels_available") else "无"))
    L.append("")
    L.append("## 背景与目标")
    L.append("")
    L.append("- 当前 loop 检测存在**误杀风险**：真实 Qwen3-1.7B + Skywork 短 rollout 循环退化率")
    L.append("  75-87%，默认 `(2,3,4)` 可能把正常长 CoT 误判为 loop（误报）。")
    L.append("- 目标：在配置矩阵（periods × min_len）中选择「**误报最低且能抓住明显循环**」的配置。")
    L.append("- 硬约束：**不得为凑 `<50%` 人为放松 detector**——若最低误报配置仍误杀正常 CoT，")
    L.append("  应记录并转向采样侧（temperature / repetition_penalty）治理，而非放宽检测。")
    L.append("")
    L.append("## 方法学")
    L.append("")
    L.append("1. 真实 rollout N 条（temperature=1.0，短预算 `max_new`，新生成 token 序列去 pad）。")
    L.append("2. 对每条 rollout，用与训练完全一致的 `detect_loop`（尾部周期自相关 + min_len 门槛）判定。")
    L.append("3. 对每种 (periods, min_len) 配置统计：`loop_detected_count` / `loop_rate` /")
    L.append("   `samples_flagged`；有**人工标注**时另算 `false_positive_rate` / `false_negative_cases`。")
    L.append("4. 人工抽样检查：正常长 CoT 是否被误杀；`Final Answer` 重复是否被捕获；")
    L.append("   真正 token-level repetition 是否被捕获。")
    L.append("")
    L.append("## 配置矩阵")
    L.append("")
    L.append("| periods | min_len |")
    L.append("|---|---|")
    for cfg_key in sweep_results:
        periods, min_len = cfg_key.split(",min_len=")
        L.append(f"| `{periods.split('=')[1]}` | {min_len} |")
    if not sweep_results:
        L.append("| （待 GPU 填充） | |")
    L.append("")
    L.append("## 结果")
    L.append("")
    if not sweep_results:
        L.append("> **（无数据）**：真实 rollout 100 条需服务器 GPU，本机无 GPU 未伪造通过；")
        L.append("> 此表为占位，待服务器 `calibrate_rollout.py --n 100 --report ...` 填充。")
        L.append("")
    L.append("| config | loop_detected_count | loop_rate | FP rate | FN cases | samples_flagged |")
    L.append("|---|---:|---:|---:|---|---|")
    for cfg_key, r in sweep_results.items():
        fn = ",".join(str(i) for i in (r["false_negative_cases"] or [])) or "-"
        fp = ("-" if r["false_positive_rate"] is None
              else f"{r['false_positive_rate']:.3f}")
        flags = ",".join(str(i) for i in r["samples_flagged"]) or "-"
        L.append(f"| `{cfg_key}` | {r['loop_detected_count']} | "
                 f"{r['loop_rate']:.3f} | {fp} | {fn} | {flags} |")
    L.append("")
    L.append("## 人工抽样检查清单（需 GPU + 人工）")
    L.append("")
    L.append("- [ ] 正常长 CoT 是否被误杀（抽查 flagged 样本看内容）")
    L.append("- [ ] `Final Answer` 标记重复是否被捕获")
    L.append("- [ ] 真正 token-level repetition 是否被捕获")
    L.append("- [ ] 误报最低配置下的误报样本内容（是否为真实退化或误判）")
    L.append("")
    L.append("## 决策")
    L.append("")
    L.append("- 选「误报最低且能抓住明显循环」的配置作为 `l2.rollout.loop_periods` / `loop_min_len`。")
    L.append("- 若最低误报配置仍误杀正常 CoT：不放松 detector，转向采样侧治理并记录。")
    L.append("")
    L.append("## GPU 验证状态")
    L.append("")
    L.append("- 真实 rollout 100 条：**待服务器 GPU**（本机无 GPU，不伪造通过）。")
    L.append("- 人工标注与抽样检查：**待**。")
    L.append("- 结果表填充后，本报告才可视为校准结论。")
    L.append("")
    text = "\n".join(L)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = parse_args()
    if not args.jsonl.is_file():
        raise SystemExit(f"jsonl 不存在：{args.jsonl}（请给真实 prompt jsonl）")
    if not args.model.exists():
        raise SystemExit(f"HF 模型路径不存在：{args.model}")
    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    prompts = [r["prompt"] for r in rows if r.get("prompt")]
    if not prompts:
        raise SystemExit("jsonl 无 prompt")
    rng = random.Random(args.seed)
    sample = rng.sample(prompts, min(args.n, len(prompts)))
    print(f"采样 {len(sample)} 条 prompt（共 {len(prompts)}）", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model))
    eos = args.eos_id if args.eos_id is not None else tok.eos_token_id
    print(f"tokenizer.eos_token_id = {tok.eos_token_id!r} "
          f"(eos_token={tok.eos_token!r})；本采样 eos-id = {eos!r}", flush=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"加载模型 {args.model} (flash_attention_2, bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(args.device).eval()

    all_new: list[list[int]] = []
    t0 = time.time()
    for start in range(0, len(sample), args.batch_size):
        bs = sample[start:start + args.batch_size]
        enc = tok(bs, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(args.device)
        seq_len = enc["input_ids"].size(1)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True,
                temperature=1.0, top_p=0.95, num_return_sequences=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos)          # eos=None → HF 用模型默认 eos（会自然停）
        for j in range(out.size(0)):
            new = out[j][seq_len:].tolist()
            # 去 pad（budget 撞满时尾部是 pad_token）
            while new and new[-1] == tok.pad_token_id:
                new.pop()
            all_new.append(new)
        print(f"  {min(start+args.batch_size, len(sample))}/{len(sample)} "
              f"完成 ({time.time()-t0:.0f}s)", flush=True)

    # ---- 分析（纯函数，可单测）----
    report = analyze_rollouts(all_new, eos_tok=tok.eos_token_id, eos_used=eos)
    rates = report["loop_rate_by_period"]
    print(f"\n新生成 token 长度：min={report['lens_min']} max={report['lens_max']} "
          f"E[L]={report['lens_mean']:.0f}", flush=True)

    print(f"new 序列含 tokenizer EOS({report['eos_tok']}) 条数："
          f"{report['n_eos']}/{report['n']}", flush=True)

    # 尾部周期自相关（p=2..8）
    print("\n尾部周期自相关（detect_loop 语义，min_len=16）：")
    print(f"{'p':>3} | {'loop 命中率':>10} | {'命中/总数':>8}")
    for p in range(2, 9):
        print(f"{p:>3} | {rates[p]:>10.3f} | "
              f"{int(rates[p] * report['n']):>4}/{report['n']}", flush=True)

    periods = report["suggested_loop_periods"]
    print(f"\n建议 loop_periods = {periods}（命中率>5% 的周期）", flush=True)
    print(f"建议 eos_token_id = {report['suggested_eos_token_id']}（l2.rollout.eos_token_id）", flush=True)

    # ---- IMP-1：loop detector 校准（sweep periods × min_len）----
    labels = None
    if args.labels is not None:
        labels = [bool(json.loads(l)["label"])
                  for l in open(args.labels, encoding="utf-8") if l.strip()]
    sweep = sweep_loop_configs(all_new, DEFAULT_SWEEP_CONFIGS, labels)
    print("\n=== loop detector 校准（sweep） ===", flush=True)
    for k, r in sweep.items():
        fp = "-" if r["false_positive_rate"] is None else f"{r['false_positive_rate']:.3f}"
        fn = len(r["false_negative_cases"]) if r["false_negative_cases"] else 0
        print(f"  {k}: loop={r['loop_detected_count']}/{report['n']} "
              f"rate={r['loop_rate']:.3f} FP_rate={fp} FN_cases={fn}", flush=True)
    if args.report is not None:
        meta = {"n": report["n"], "max_new": args.max_new, "model": str(args.model),
                "eos_token_id": report["eos_tok"], "date": "2026-08-16",
                "status": ("已填充" if all_new else "待服务器真实 rollout 数据填充"),
                "labels_available": labels is not None}
        text = write_calibration_report(sweep, args.report, meta)
        print(f"\n✅ 校准报告写入 {args.report}", flush=True)

    # IMP-1b：可选 YAML 落盘（不只 stdout）
    if args.output is not None:
        text = write_yaml(report, args.output)
        print(f"\n✅ 建议配置写入 {args.output}：", flush=True)
        print(text, end="")


if __name__ == "__main__":
    main()

"""工程化 CLI：`python -m fullstack_opd_v2` 的子命令入口。

子命令：
  train  跑全栈流水线（Stage 0/1/2），落盘 run 目录（config/日志/checkpoint/metrics/计时）
  cache  只建 Lightning 离线缓存（Stage 1）
  eval   评 checkpoint 的健康信号（E[Δ_T] 趋势 / loss / staleness）
  info   打印解析后的完整配置（校验 YAML 与覆盖是否合法）

示例：
  python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml --run-dir runs/exp1
  python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml --run-dir runs/exp1 --resume
  python -m fullstack_opd_v2 info --config configs/fullstack_opd.yaml --set stage2.n_steps=50
"""

from __future__ import annotations

import argparse
import os

from pydantic import ValidationError

from .exceptions import ConfigError, CheckpointError, DataError, OPDError


def _device_arg(args) -> str:
    import torch
    return args.device or ("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(path, overrides):
    from .config import load_config
    return load_config(path=path, overrides=overrides)


# --------------------------- 子命令 ---------------------------
def _cmd_train(args) -> int:
    from .checkpoint import CheckpointManager
    from .pipeline import FullStackOPDv2

    run_dir = args.run_dir
    resume = None
    if args.resume:
        if not run_dir:
            raise ConfigError("--resume 需要 --run-dir（指定要续跑的 run 目录）")
        ck = CheckpointManager(run_dir).resume()
        if ck is None:
            raise CheckpointError(f"run_dir 无断点可续: {run_dir}")
        resume = ck
        print(f"[train] resume: 从 {run_dir} 的 version={ck['version']} 续跑")

    cfg = _load_cfg(args.config, args.set)
    if run_dir:
        cfg.setdefault("run", {})["run_dir"] = run_dir
    device = _device_arg(args)
    opd = FullStackOPDv2(cfg, device=device)
    out = opd.run(run_dir=run_dir, resume=resume)
    print(f"[train] 完成: {len(out['metrics'])} 步, 总耗时 {out['timings']['total']:.2f}s")
    print(f"[train] run 目录: {out['run_dir']}")
    print(f"[train] 计时: {out['timings']}")
    if out["metrics"]:
        last = out["metrics"][-1]
        print(f"[train] 末步: E[Δ_T]={last['reward']:+.4f} age={last['age']} "
              f"loss={last['loss']:.4f}")
    return 0


def _cmd_cache(args) -> int:
    from .model_factory import build_model
    from .pipeline import FullStackOPDv2, stage1_build_cache

    device = _device_arg(args)
    cfg = _load_cfg(args.config, args.set)
    opd = FullStackOPDv2(cfg, device=device)     # 加载数据 + Stage 0 教师
    teacher_rl, teacher_ref = opd._stage0_teachers()
    s1cfg = dict(cfg["stage1"])     # 部署键下渗已在 load_config 完成（config.py 校验前）
    if args.out:
        s1cfg["cache_path"] = args.out
    # L1：warmup 需要初始 student（student_init 采样）；toy 下即初始 CausalToyLM
    warmup_student = build_model(cfg, device, role="student")
    cache, _, _ = stage1_build_cache(
        opd.prompts, opd.responses, teacher_rl, teacher_ref, s1cfg,
        warmup_student=warmup_student)
    print(f"[cache] Δ_T 缓存已构建: {s1cfg['cache_path']} "
          f"mode={cfg['cache_mode']} top_k={cfg['top_k_teacher']}")
    return 0


def _cmd_eval(args) -> int:
    import torch
    from .checkpoint import CheckpointManager
    from .model import CausalToyLM

    device = _device_arg(args)
    cfg = _load_cfg(args.config, args.set)
    ck = CheckpointManager(".", checkpoint_dir=os.path.dirname(args.checkpoint)).load(args.checkpoint)
    student = CausalToyLM(vocab=cfg["vocab_size"], d_model=cfg["d_model"],
                          n_layers=cfg["n_layers"]).to(device)
    student.load_state_dict(ck["state"])
    print(f"[eval] checkpoint: {args.checkpoint}  step={ck['step']}  version={ck['version']}")
    print(f"[eval] 该断点学生已就绪；AIME 蒸馏后评估见 benchmarks/aime24_25/")
    return 0


def _cmd_info(args) -> int:
    import yaml
    cfg = _load_cfg(args.config, args.set)
    print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return 0


# --------------------------- 入口 ---------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fullstack-opd-v2",
                                 description="全栈 OPD 工程化 CLI（train/cache/eval/info）")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="跑全栈流水线（Stage 0/1/2）")
    p.add_argument("--config", default=None)
    p.add_argument("--run-dir", default=None, help="run 目录（默认自动时间戳）")
    p.add_argument("--resume", action="store_true", help="从 run-dir 最新断点续跑")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--device", default=None)

    p = sub.add_parser("cache", help="只建 Lightning 离线缓存（Stage 1）")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="缓存输出路径")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--device", default=None)

    p = sub.add_parser("eval", help="评 checkpoint 健康信号")
    p.add_argument("--config", default=None)
    p.add_argument("--checkpoint", required=True, help="checkpoint 路径")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--device", default=None)

    p = sub.add_parser("info", help="打印解析后配置")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])

    p = sub.add_parser("eval-aime", help="真实模型 AIME24/25 评估（main/ 自包含）")
    p.add_argument("--model", default=None, help="HF 模型路径 / id（与 --run-dir 二选一）")
    p.add_argument("--run-dir", default=None, help="run 目录（读 config.yaml 的 eval.model_path）")
    p.add_argument("--checkpoint", default=None, help="（预留）checkpoint 路径")
    p.add_argument("--datasets", nargs="+", default=None, help="如 AIME24 AIME25（默认两者）")
    p.add_argument("--out", default=None, help="输出目录（默认 run-dir/aime 或 results/aime）")
    p.add_argument("--max-new-tokens", type=int, default=None)  # None → 回退 run-eval cfg（P1）
    p.add_argument("--n-samples", type=int, default=None)          # None → 回退 run-eval cfg（P1）
    p.add_argument("--temperature", type=float, default=None)      # None → 回退 run-eval cfg（P2）
    p.add_argument("--top-p", type=float, default=None)            # 论文评估协议 0.95
    p.add_argument("--metric", default=None, help="pass1（默认）| ave（论文 ave@32 平均正确率）")
    p.add_argument("--prompt-style", default=None, help="boxed（默认）| dapo（论文 DAPO 模板）")
    p.add_argument("--scoring", default=None,
                   help="int（默认，整数精确匹配）| sympy（论文数学等价判定 grade_answer_mathd/sympy）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="生成 batch（默认 8）。长生成（max_new_tokens 数万）必须调小——峰值显存随 "
                        "batch 线性涨，单卡 97GB 下论文 32768 长生成建议 1-2，否则 OOM")
    p.add_argument("--chat-template", action="store_true", default=None,
                   help="用模型 chat template 包裹 prompt（对齐论文 verl 验证的 apply_chat_template）")
    p.add_argument("--dtype", default=None, help="fp32 | bf16 | float16 | auto（默认 auto）")
    p.add_argument("--device", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "cache":
            return _cmd_cache(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "eval-aime":
            return _cmd_eval_aime(args)
        raise ConfigError(f"未知子命令: {args.command}")
    except (OPDError, ValidationError) as e:
        print(f"[error] {type(e).__name__}: {e}")
        return 2
    except KeyboardInterrupt:
        print("[error] 训练被中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


def _cmd_eval_aime(args) -> int:
    """AIME 评估：--model 直评 / --run-dir 桥接（读 config.yaml 的 eval.model_path）。"""
    import os
    import yaml
    from .eval_aime import AimeEvaluator, DEFAULT_DATASETS

    if args.model:
        model_path = args.model
        run_eval_cfg = {}
    elif args.run_dir:
        cfg_path = os.path.join(args.run_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise ConfigError(f"run 目录缺 config.yaml: {args.run_dir}")
        with open(cfg_path, encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ConfigError(f"run 目录 config.yaml 解析失败：{e}") from e
        ecfg = cfg.get("eval") or {}
        mp = ecfg.get("model_path")
        if not mp:
            raise DataError(
                f"run 目录 {args.run_dir!r} 未配置 eval.model_path——toy run 目录无法跑真实 AIME；"
                "请用真实模型的 run 目录（config.yaml 的 eval.model_path 指向 HF 模型路径/id）")
        model_path = mp
        run_eval_cfg = ecfg    # R1：run 目录的 eval.max_new_tokens/n_samples/temperature 一并生效
    else:
        raise ConfigError("eval-aime 需要 --model 或 --run-dir")

    datasets = args.datasets or list(DEFAULT_DATASETS)
    device = _device_arg(args)
    out_dir = args.out or (os.path.join(args.run_dir, "aime") if args.run_dir else "results/aime")
    with AimeEvaluator(
            model_path, device=device,
            max_new_tokens=(args.max_new_tokens if args.max_new_tokens is not None
                            else run_eval_cfg.get("max_new_tokens", 2048)),
            n_samples=(args.n_samples if args.n_samples is not None
                       else run_eval_cfg.get("n_samples", 1)),
            temperature=(args.temperature if args.temperature is not None
                         else run_eval_cfg.get("temperature", 0.0)),
            top_p=(args.top_p if args.top_p is not None
                   else run_eval_cfg.get("top_p")),
            metric=(args.metric if args.metric is not None
                    else run_eval_cfg.get("metric", "pass1")),
            prompt_style=(args.prompt_style if args.prompt_style is not None
                          else run_eval_cfg.get("prompt_style", "boxed")),
            scoring=(args.scoring if args.scoring is not None
                     else run_eval_cfg.get("scoring", "int")),
            batch_size=(args.batch_size if args.batch_size is not None
                        else run_eval_cfg.get("batch_size", 8)),
            chat_template=(args.chat_template if args.chat_template
                           else run_eval_cfg.get("chat_template", False)),
            dtype=(args.dtype if args.dtype is not None
                   else run_eval_cfg.get("dtype", "auto"))) as ev:
        print(f"[eval-aime] model={model_path}  datasets={datasets}  device={device}")
        print(f"[eval-aime] 输出目录: {out_dir}")
        for ds in datasets:
            out_path = os.path.join(out_dir, f"{ds}.jsonl")
            res = ev.evaluate_to_jsonl(ds, out_path)
            # 二次审阅修复 #1：分口径标注——pass1 与 ave 混排会误导
            # （"27/30 = 56.25%" 里 27/30 是 pass@1、56.25% 是 ave@32，两码事）。
            if ev.metric == "ave" and res.ave_accuracy is not None:
                print(f"[eval-aime] {ds}: pass@1 {res.correct}/{res.total} "
                      f"({100 * res.correct / res.total:.2f}%)  "
                      f"ave@32 {res.ave_accuracy * 100:.2f}%  → {out_path}")
            else:
                print(f"[eval-aime] {ds}: {res.correct}/{res.total} "
                      f"= {res.accuracy * 100:.2f}%  → {out_path}")
    print("[eval-aime] 汇总：teacher 基线 / 学生蒸馏前后对比见 benchmarks/aime24_25/aggregate.py")
    return 0

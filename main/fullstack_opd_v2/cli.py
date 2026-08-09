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

from .exceptions import ConfigError, CheckpointError, OPDError


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
    from .pipeline import FullStackOPDv2, stage1_build_cache

    device = _device_arg(args)
    cfg = _load_cfg(args.config, args.set)
    opd = FullStackOPDv2(cfg, device=device)     # 加载数据 + Stage 0 教师
    teacher_rl, teacher_ref = opd._stage0_teachers()
    s1cfg = dict(cfg["stage1"])
    for _k in ("cache_mode", "top_k_teacher"):
        if _k not in s1cfg and _k in cfg:
            s1cfg[_k] = cfg[_k]
    if args.out:
        s1cfg["cache_path"] = args.out
    cache, _, _ = stage1_build_cache(
        opd.prompts, opd.responses, teacher_rl, teacher_ref, s1cfg)
    print(f"[cache] Δ_T 缓存已构建: {s1cfg['cache_path']} "
          f"mode={cfg['cache_mode']} top_k={cfg['top_k_teacher']}")
    return 0


def _cmd_eval(args) -> int:
    import torch
    from .checkpoint import CheckpointManager
    from .model import CausalToyLM

    device = _device_arg(args)
    cfg = _load_cfg(args.config, args.set)
    ck = CheckpointManager(".", checkpoint_dir=args.checkpoint).load(args.checkpoint)
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
        raise ConfigError(f"未知子命令: {args.command}")
    except OPDError as e:
        print(f"[error] {type(e).__name__}: {e}")
        return 2
    except KeyboardInterrupt:
        print("[error] 训练被中断")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

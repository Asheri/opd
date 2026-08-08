"""全栈 OPD 叠加 demo v2 的命令行入口（打包后 `python -m fullstack_opd_v2` 可调）。

支持 YAML 配置 + 命令行覆盖（见 config.py 的 schema 校验）：

    python -m fullstack_opd_v2                              # 用内置 DEFAULT_CONFIG_V2
    python -m fullstack_opd_v2 --config configs/fullstack_opd.yaml
    python -m fullstack_opd_v2 --set stage2.n_steps=50 --set stage1.warmup_source=mix
"""
from __future__ import annotations

import argparse

from .pipeline import FullStackOPDv2
from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="全栈 OPD 叠加 demo v2")
    ap.add_argument("--config", default=None,
                    help="YAML 配置路径（缺省用内置 DEFAULT_CONFIG_V2）")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="点分覆盖，可多次，如 --set stage2.n_steps=50 --set n_prompts=32")
    ap.add_argument("--device", default=None, help="cpu | cuda（默认自动检测）")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(path=args.config, overrides=args.overrides)
    print(f"device = {device}   batch_size = {cfg['stage2']['batch_size']}   "
          f"warmup = {cfg['stage1']['warmup_source']}×{cfg['stage1']['warmup_M']}\n")

    opd = FullStackOPDv2(cfg, device=device)
    out = opd.run()
    metrics = out["metrics"]
    timings = out["timings"]

    print("\n================ 全栈叠加结果（v2）================")
    print(f"训练步数            : {len(metrics)}  (每步 batch={metrics[-1]['batch'] if metrics else '-'})")
    if metrics:
        last = metrics[-1]
        print(f"最后一步 version    : {last['version']}")
        print(f"最后一步 staleness  : age={last['age']} "
              f"(>0 ⇒ 调度器确实在消费陈旧样本)")
        print(f"最后一步 loss       : {last['loss']:.4f} "
              f"(pg={last['pg_loss']:.4f}, kl={last['kl_loss']:.4f})")
        print(f"最后一步 E[Δ_T]     : {last['reward']:+.4f} "
              f"(当前 student 的期望 Direct-OPD 奖励，随训练上升)")
        print(f"最后一步 adv_mean   : {last['adv_mean']:+.4f} "
              f"(rollout 时刻的批次奖励均值)")

    print("\n----------- 逐 stage 耗时 -----------")
    for k, v in timings.items():
        print(f"  {k:14s}: {v:6.2f}s")

    print("\n----------- 三重限制突破对照 -----------")
    print("  [常驻教师] Lightning 张量缓存 Δ_T 零拷贝索引，训练循环内无 teacher 前向")
    print("  [同步等待] AsyncOPD 四阶段异步调度，批次在队列中流动，陈旧批次 learner 时刻重算")
    print("  [迁移终态] Direct-OPD 迁移对象 = RL 策略偏移 Δ_T，作用于 student 自身 on-policy 状态")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()

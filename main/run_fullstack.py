"""全栈 OPD 叠加 demo 入口。

运行：
    cd C:/Users/12062/OneDrive/Desktop/opd/main
    python run_fullstack.py

仅依赖 torch（CPU + 极小词表即可端到端跑通，证明三篇论文的全栈叠加逻辑正确）。
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fullstack_opd.pipeline import FullStackOPD, DEFAULT_CONFIG


def main():
    # ★ 修复：原先 torch 只在 __main__ 块 import，而 main() 函数体直接引用 torch ——
    # 一旦被当作模块调用（from run_fullstack import main）即 NameError。
    # 且包顶层 pipeline 已链式 import torch，「无 torch 仅阅读」本就不成立。
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}\n")
    opd = FullStackOPD(dict(DEFAULT_CONFIG), device=device)
    out = opd.run()
    metrics = out["metrics"]

    print("\n================ 全栈叠加结果 ================")
    print(f"训练步数            : {len(metrics)}")
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
              f"(rollout 时刻的 token 级奖励，随样本流转)")

    print("\n----------- 三重限制突破对照 -----------")
    print("  [常驻教师] Lightning 离线缓存 Δ_T，训练循环内不出现任何 teacher 前向")
    print("  [同步等待] AsyncOPD 异步调度器解耦 rollout 与 learner，陈旧样本 learner 时刻重算")
    print("  [迁移终态] Direct-OPD 迁移对象 = RL 策略偏移 Δ_T，作用于更强 student 自身 on-policy 状态")
    print("-------------------------------------------")


if __name__ == "__main__":
    main()

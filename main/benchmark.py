"""v1（逐样本）vs v2（批量化重构）基准对比。

运行：
    cd C:/Users/12062/OneDrive/Desktop/opd/main
    python benchmark.py

对比维度：总墙钟时间 / stage2 样本吞吐量 / E[Δ_T] 收敛。
注意：v2 每步处理 batch_size 个样本（默认 8），30 步 = 240 个样本更新；
v1 每步 1 个样本，30 步 = 30 个样本更新。
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fullstack_opd.pipeline import FullStackOPD, DEFAULT_CONFIG
from fullstack_opd_v2.pipeline import FullStackOPDv2, DEFAULT_CONFIG_V2


def main():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = DEFAULT_CONFIG_V2["stage2"]["batch_size"]
    print(f"device = {device}\n")

    print("=== 跑 v1（逐样本执行底座）===")
    t0 = time.perf_counter()
    out1 = FullStackOPD(dict(DEFAULT_CONFIG), device=device).run()
    t_v1 = time.perf_counter() - t0
    m1 = out1["metrics"]

    print("\n=== 跑 v2（批量化重构底座）===")
    t0 = time.perf_counter()
    out2 = FullStackOPDv2(dict(DEFAULT_CONFIG_V2), device=device).run()
    t_v2 = time.perf_counter() - t0
    m2 = out2["metrics"]

    n1 = len(m1)          # v1 样本数 = 步数 × 1
    n2 = len(m2) * batch  # v2 样本数 = 步数 × batch
    tput1 = n1 / max(out1["timings"]["stage2_train"], 1e-9)
    tput2 = n2 / max(out2["timings"]["stage2_train"], 1e-9)

    print("\n==================== 基准对比 ====================")
    print(f"{'指标':<22}{'v1 (逐样本)':>14}{'v2 (批量化)':>14}")
    print("-" * 52)
    print(f"{'总墙钟时间 (s)':<22}{t_v1:>14.2f}{t_v2:>14.2f}")
    print(f"{'stage2 耗时 (s)':<22}{out1['timings']['stage2_train']:>14.2f}{out2['timings']['stage2_train']:>14.2f}")
    print(f"{'stage2 样本数':<22}{n1:>14}{n2:>14}")
    print(f"{'吞吐 (样本/s, 全程)':<22}{n1 / t_v1:>14.1f}{n2 / t_v2:>14.1f}")
    print(f"{'吞吐 (样本/s, stage2)':<22}{tput1:>14.1f}{tput2:>14.1f}")
    print(f"{'最终 E[Δ_T]':<22}{m1[-1]['reward']:>+14.4f}{m2[-1]['reward']:>+14.4f}")
    print(f"{'最终 staleness age':<22}{m1[-1]['age']:>14}{m2[-1]['age']:>14}")
    print("-" * 52)
    print(f"总时间加速比            : {t_v1 / t_v2:.2f}x")
    print(f"stage2 吞吐加速比       : {tput2 / tput1:.2f}x")
    print("\nv2 逐 stage 耗时："
          + ", ".join(f"{k}={v:.2f}s" for k, v in out2["timings"].items()))


if __name__ == "__main__":
    main()

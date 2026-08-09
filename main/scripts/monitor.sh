#!/bin/bash
# 监控：训练期/训练后查看健康信号。用法: bash scripts/monitor.sh <logfile 可选>
# 无参数时对最近一次 demo 日志做健康检查；有参数时 tail 该文件。
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=${1:-/root/autodl-tmp/opd-train.log}

if [[ -f "$LOG" ]]; then
  echo "=== 实时日志（Ctrl-C 退出）：$LOG ==="
  echo "  健康信号：E[Δ_T] 单调上升、staleness age>0、无 teacher 前向。"
  tail -f "$LOG"
else
  echo "⚠️  未找到日志 $LOG（还没跑训练？先跑 03_gpu_demo / 04_distributed_2gpu）。"
  echo ""
  echo "=== 健康信号检查表 ==="
  echo "  [ ] E[Δ_T] 随训练单调上升（修复后 −0.18 → +0.72）"
  echo "  [ ] staleness age > 0（双截断在工作，异步在消费陈旧样本）"
  echo "  [ ] 训练循环无任何 teacher 前向（Lightning 缓存使命达成）"
  echo "  [ ] GPU 利用率 / 显存合理（2×96GB 不 OOM）"
fi
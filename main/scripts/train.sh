#!/bin/bash
# 全栈 OPD · main/ v2 GPU 骨架训练总入口
# 连上服务器后一键跑：环境自检 → CPU 基线 → 单卡 GPU → 2 卡分布式。
# 用法: bash scripts/train.sh [smoke|full]
#   smoke - 每级少量步数快速冒烟（默认，验证全链路跑通）
#   full  - 完整训练（单卡 30 步 + 分布式 30 步）
set -euo pipefail
cd "$(dirname "$0")/.."          # → main/
MODE=${1:-smoke}

echo "################################################################"
echo "# 全栈 OPD · main/ v2 GPU 骨架 · 训练总入口（mode=$MODE）"
echo "################################################################"

echo ""
echo "=== [1/4] 环境自检 ==="
bash scripts/01_env_check.sh

echo ""
echo "=== [2/4] CPU 基线（58 测试 + CPU demo）==="
if [[ "$MODE" == "full" ]]; then
  bash scripts/02_cpu_demo.sh
else
  # smoke 模式跳过完整测试套件，只跑 4 个关键测试快速验证
  python -m pytest tests/test_losses.py tests/test_scheduler.py -q
fi

echo ""
echo "=== [3/4] 单卡 GPU 骨架 demo ==="
if [[ "$MODE" == "full" ]]; then
  bash scripts/03_gpu_demo.sh 30
else
  bash scripts/03_gpu_demo.sh 5
fi

echo ""
echo "=== [4/4] 2 卡分布式骨架 ==="
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$NGPU" -ge 2 ]]; then
  if [[ "$MODE" == "full" ]]; then
    bash scripts/04_distributed_2gpu.sh full
  else
    bash scripts/04_distributed_2gpu.sh smoke
  fi
else
  echo "⚠️  当前仅 $NGPU 卡，跳过分布式骨架（需 ≥2 卡）。"
fi

echo ""
echo "################################################################"
echo "# 全部完成。健康信号检查："
echo "#   bash scripts/monitor.sh <日志>"
echo "# 单步重跑："
echo "#   bash scripts/02_cpu_demo.sh / 03_gpu_demo.sh / 04_distributed_2gpu.sh"
echo "################################################################"
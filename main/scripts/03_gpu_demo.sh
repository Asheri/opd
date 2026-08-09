#!/bin/bash
# 03 · 单卡 GPU 骨架 demo：bf16 + 稀疏 topk 缓存 + 异步调度（线程版）
# 用法: bash scripts/03_gpu_demo.sh [n_steps]
#   可选第一个参数 = 训练步数（默认 30）
set -euo pipefail
cd "$(dirname "$0")/.."          # → main/
PY=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
N_STEPS=${1:-30}
GITDIR="$(cd "$(dirname "$0")/../.." && pwd)"   # 仓库根（opd/）

echo "=== 单卡 GPU 骨架 demo（device=$DEVICE · n_steps=$N_STEPS）==="
echo "  配置: configs/gpu_skeleton_2gpu.yaml（bf16 + topk 缓存 + offload）"
echo "  ⚠️  模型是 CausalToyLM（toy），跑通 GPU 路径，非真实 7B。"
echo ""

# 校验 GPU 可用
$PY -c "import torch, sys; assert torch.cuda.is_available(), '无 GPU 可用'; print('GPU OK:', torch.cuda.get_device_name(0))"

$PY -m fullstack_opd_v2 \
  --device "$DEVICE" \
  --config configs/gpu_skeleton_2gpu.yaml \
  --set stage2.distributed=false \
  --set stage2.n_steps="$N_STEPS"

echo ""
echo "=== 单卡 GPU demo 完成 ==="
echo "  日志若需保留：bash scripts/03_gpu_demo.sh 2>&1 | tee /root/autodl-tmp/gpu-demo.log"
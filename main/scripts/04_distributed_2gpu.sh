#!/bin/bash
# 04 · 2 卡分布式骨架（DistAsyncScheduler + Ray + NCCL 权重广播）
# 用法: bash scripts/04_distributed_2gpu.sh [smoke|full]
#   smoke - 3 步，快速验证分布式路径（OOM/配置/NCCL 组）
#   full  - 30 步（读 gpu_skeleton_2gpu.yaml 的 stage2.n_steps）
set -euo pipefail
cd "$(dirname "$0")/.."          # → main/
PY=${PYTHON:-python}
MODE=${1:-smoke}

echo "=== 前置校验：ray + torch.distributed + CUDA≥2 ==="
$PY - <<'PYEOF'
import torch, importlib.util
assert torch.cuda.is_available(), "需要 CUDA"
assert torch.cuda.device_count() >= 2, f"需 ≥2 GPU，当前 {torch.cuda.device_count()}"
assert torch.distributed.is_available(), "需要 torch.distributed(NCCL)"
assert importlib.util.find_spec("ray") is not None, "需要 ray（pip install ray）"
print("  前置 OK：CUDA≥2 + torch.distributed + ray")
PYEOF

if [[ "$MODE" == "smoke" ]]; then
  echo ""
  echo "=== 分布式骨架 SMOKE（3 步）==="
  $PY scripts/launch_v2_distributed.py --set stage2.n_steps=3
else
  echo ""
  echo "=== 分布式骨架 FULL（30 步）==="
  $PY scripts/launch_v2_distributed.py
fi

echo ""
echo "=== 2 卡分布式骨架完成 ==="
echo "  保留日志：bash scripts/04_distributed_2gpu.sh full 2>&1 | tee /root/autodl-tmp/dist-2gpu.log"
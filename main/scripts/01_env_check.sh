#!/bin/bash
# 01 · 环境自检：GPU 拓扑 / torch·CUDA / 可选 GPU 依赖
# 用法: bash scripts/01_env_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # → main/
PY=${PYTHON:-python}

echo "=== [1/3] GPU 拓扑 ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "⚠️  未找到 nvidia-smi —— 非 GPU 节点？仅能跑 02_cpu_demo 与单卡 fallback。"
fi

echo ""
echo "=== [2/3] torch / CUDA ==="
$PY - <<'PYEOF'
import torch
print(f"torch = {torch.__version__}")
print(f"cuda_available = {torch.cuda.is_available()}")
print(f"ngpu = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")
print(f"tf32_matmul_allowed = {torch.backends.cuda.matmul.allow_tf32}")
PYEOF

echo ""
echo "=== [3/3] 可选 GPU 依赖（缺失不影响 CPU demo / 单卡 toy）==="
$PY - <<'PYEOF'
import importlib.util
for m in ("vllm", "ray", "megatron.core"):
    ok = importlib.util.find_spec(m) is not None
    print(f"  {m:14s}: {'OK' if ok else 'MISSING'}")
PYEOF

echo ""
echo "=== 自检结论 ==="
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
echo "  可跑路径："
echo "    - 总是可跑 : 02_cpu_demo（58 测试 + CPU demo）"
echo "    - NGPU≥1  : 03_gpu_demo（单卡 bf16 + 稀疏缓存）"
echo "    - NGPU≥2  : 04_distributed_2gpu（2 卡分布式骨架；需 ray + torch.distributed）"
echo "    - vllm 缺失→ rollout_engine 保持 toy；ray 缺失→ 04 会报 L5 需要 ray"
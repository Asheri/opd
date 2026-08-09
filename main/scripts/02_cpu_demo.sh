#!/bin/bash
# 02 · CPU 基线：58 个回归测试 + 默认全栈 demo（验证内核没被云环境破坏）
# 用法: bash scripts/02_cpu_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."          # → main/
PY=${PYTHON:-python}

echo "=== [1/2] 回归测试（58 个，算法内核正确性）==="
$PY -m pytest tests/ -q

echo ""
echo "=== [2/2] CPU 全栈 demo（默认 toy 配置，30 步）==="
$PY -m fullstack_opd_v2

echo ""
echo "=== CPU 基线完成 ==="
echo "  健康信号应满足：E[Δ_T] 随训练单调上升、staleness age>0、训练循环无 teacher 前向。"
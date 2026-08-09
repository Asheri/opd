#!/bin/bash
# AIME24/25 蒸馏效果基准 harness —— main/ 自包含（opd eval-aime，无 async-opd 依赖）
#
# 验证「异步 + 预加载教师 + 弱到强蒸馏」，三组组合：
#   1. JustRL-1.5B → Qwen3-1.7B     2. JustRL-1.5B → Qwen3-4B     3. JustRL-1.5B → R1-Distill-7B
#
# 用法:
#   bash run_benchmark.sh teacher          # 教师（JustRL-1.5B = TEACHER_RL_PATH）AIME24/25 起点
#   bash run_benchmark.sh student_baseline # 三组学生蒸馏前 AIME24/25
#   bash run_benchmark.sh all              # teacher + student_baseline + 汇总表
#   bash run_benchmark.sh aggregate        # 只汇总已产出的结果
#
# 前置：main/ 已 pip install -e（提供 fullstack_opd_v2.eval_aime）；模型在 models.env。
#       AIME 数据集需可访问 huggingface datasets（服务器用 source /etc/network_turbo）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"
MAIN="$(dirname "$HERE")/main"
PY=${PYTHON:-python}

source "$HERE/models.env"
MODE=${1:-all}

run_eval() {
  # $1=模型路径 $2=outdir；其余透传给 eval-aime
  local mpath="$1" out="$2"; shift 2
  mkdir -p "$out"
  echo ""
  echo ">>> eval-aime  model=$mpath  out=$out"
  (cd "$MAIN" && $PY -m fullstack_opd_v2 eval-aime \
    --model "$mpath" --datasets AIME24 AIME25 --out "$out" "$@")
}

eval_teacher() {
  # 教师基线：π_RL（JustRL-1.5B）AIME24/25 起点
  run_eval "$TEACHER_RL_PATH" "$RES/teacher"
}

eval_student_baseline() {
  run_eval "$STUDENT_COMBO1" "$RES/student_baseline/combo1_qwen3_1p7b"
  run_eval "$STUDENT_COMBO2" "$RES/student_baseline/combo2_qwen3_4b"
  run_eval "$STUDENT_COMBO3" "$RES/student_baseline/combo3_r1_distill_7b"
}

case "$MODE" in
  teacher) eval_teacher ;;
  student_baseline) eval_student_baseline ;;
  all) eval_teacher; eval_student_baseline ;;
  aggregate) : ;;
  *) echo "未知模式: $MODE（用 teacher|student_baseline|all|aggregate）"; exit 1 ;;
esac

echo ""
echo "=== 汇总 ==="
$PY "$HERE/aggregate.py" "$RES"
echo ""
echo "蒸馏后学生评估：bash watch_student.sh <combo> <run-dir>（见 README）"
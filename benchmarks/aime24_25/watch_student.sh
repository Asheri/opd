#!/bin/bash
# watch 蒸馏产出的学生 checkpoint 并评估 AIME24/25（学生「蒸馏后」得分）
#
# main/ 自包含（opd eval-aime，无 async-opd 依赖）。真实蒸馏训练产出的学生
# checkpoint（HF 格式模型目录）用 --model 直评；run 目录桥接（config.yaml 配了
# eval.model_path）用 --run-dir。
#
# 用法:
#   bash watch_student.sh <combo:1|2|3> <checkpoint 模型路径或 run-dir>
#   bash watch_student.sh 1 /root/autodl-tmp/outputs/combo1/step_100
#   bash watch_student.sh 1 /root/autodl-tmp/runs/combo1      # 读 config.yaml 的 eval.model_path
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RES="$HERE/results"
MAIN="$(dirname "$HERE")/main"
PY=${PYTHON:-python}

source "$HERE/models.env"
COMBO=${1:?需要 combo:1|2|3}
TARGET=${2:?需要 checkpoint 模型路径 或 run-dir}

case "$COMBO" in
  1) OUT="$RES/student_post/combo1_qwen3_1p7b";;
  2) OUT="$RES/student_post/combo2_qwen3_4b";;
  3) OUT="$RES/student_post/combo3_r1_distill_7b";;
  *) echo "combo 需 1|2|3"; exit 1;;
esac

mkdir -p "$OUT"
echo "=== 学生蒸馏后 AIME24/25 评估（combo$COMBO · $TARGET）==="

if [[ -f "$TARGET/config.yaml" ]]; then
  # run 目录桥接：读 config.yaml 的 eval.model_path
  (cd "$MAIN" && $PY -m fullstack_opd_v2 eval-aime --run-dir "$TARGET" \
    --datasets AIME24 AIME25 --out "$OUT")
else
  # 直接评 checkpoint 模型目录（HF 格式）
  (cd "$MAIN" && $PY -m fullstack_opd_v2 eval-aime --model "$TARGET" \
    --datasets AIME24 AIME25 --out "$OUT")
fi

echo ""
echo "=== 汇总（含学生 post）==="
$PY "$HERE/aggregate.py" "$RES"
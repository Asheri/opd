#!/bin/bash
# watch 蒸馏产出的学生 checkpoint 并评估 AIME24/25（学生「蒸馏后」得分）
#
# async-opd 蒸馏训练边训边存 checkpoint（run-dir/checkpoints/step_NNN/），本脚本
# 用 opd.cli.eval --watch 自动发现新 checkpoint 并逐个在 AIME24/25 上评分，
# 结果写到 results/student_post/<combo>/。
#
# 用法:
#   bash watch_student.sh <combo:1|2|3> <蒸馏训练run-dir> [watch-timeout分钟]
#   bash watch_student.sh 1 /root/autodl-tmp/opd-run-combo1 120
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/models.env"
cd "$(dirname "$HERE")/../async-opd"          # → async-opd/（eval CLI 所在）

COMBO=${1:?需要 combo:1|2|3}
RUNDIR=${2:?需要蒸馏训练 run-dir（含 checkpoints/step_*/）}
TIMEOUT=${3:-120}

case "$COMBO" in
  1) CFG="$HERE/configs/combo1_qwen3_1p7b.yaml";     MODEL="$STUDENT_COMBO1"; GPUS="1";  TP="1";;
  2) CFG="$HERE/configs/combo2_qwen3_4b.yaml";       MODEL="$STUDENT_COMBO2"; GPUS="1";  TP="1";;
  3) CFG="$HERE/configs/combo3_r1_distill_7b.yaml";  MODEL="$STUDENT_COMBO3"; GPUS="0,1"; TP="2";;
  *) echo "combo 需 1|2|3"; exit 1;;
esac

OUT="$HERE/results/student_post/$(basename "$CFG" .yaml)"
mkdir -p "$OUT"
echo "=== watch 学生蒸馏后 checkpoint（combo$COMBO · $MODEL）==="
echo "  run-dir: $RUNDIR"
echo "  输出  : $OUT"
echo "  TP    : $TP  gpus=$GPUS"

python -m opd.cli.eval --watch \
  --config "$CFG" --model student \
  --gpus "$GPUS" --tp "$TP" \
  --datasets "$AIME24" "$AIME25" \
  --eval-n-samples "$EVAL_N_SAMPLES" --eval-temperature "$EVAL_TEMP" \
  --set teacher.path="$TEACHER_PATH" --set model.path="$MODEL" \
  --run-dir "$RUNDIR" --output-dir "$OUT" --watch-timeout "$TIMEOUT"

echo ""
echo "=== 汇总（含学生 post）==="
python "$HERE/aggregate.py" "$HERE/results"
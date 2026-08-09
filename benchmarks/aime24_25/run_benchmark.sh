#!/bin/bash
# AIME24/25 蒸馏效果基准 harness（教师基线 + 学生蒸馏前基线 + 学生蒸馏后 watch）
#
# 验证「异步 + 预加载教师 + 弱到强蒸馏」，三组组合：
#   1. JustRL-1.5B → Qwen3-1.7B     2. JustRL-1.5B → Qwen3-4B     3. JustRL-1.5B → R1-Distill-7B
#
# 用法:
#   bash run_benchmark.sh teacher          # 教师 JustRL-1.5B 的 AIME24/25 起点（跑一次）
#   bash run_benchmark.sh student_baseline # 三组学生蒸馏前 AIME24/25
#   bash run_benchmark.sh all              # teacher + student_baseline + 汇总表
#   bash run_benchmark.sh aggregate        # 只汇总已产出的结果
#   bash bench_watch_student.sh            # watch 蒸馏产出的 checkpoint（另见 watch_student.sh）
#
# 前置：async-opd 已 pip install -e（提供 opd.cli.eval）；HF 加速 source /etc/network_turbo。
set -euo pipefail

# ---- 路径 ----
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$HERE/configs"
RES="$HERE/results"
PY=${PYTHON:-python}
cd "$(dirname "$HERE")/../async-opd"          # → async-opd/（eval CLI 所在）

# ---- 模型/数据/评估参数（一次填好，放 models.env）----
source "$HERE/models.env"

MODE=${1:-all}
DATASETS=("$AIME24" "$AIME25")

run_eval() {
  # $1=config $2=model(teacher|student) $3=模型路径(覆盖) $4=gpus $5=tp $6=outdir
  local cfg="$1" model="$2" mpath="$3" gpus="$4" tp="$5" out="$6"
  mkdir -p "$out"
  echo ""
  echo ">>> eval  model=$model($mpath)  gpus=$gpus tp=$tp  out=$out"
  $PY -m opd.cli.eval \
    --config "$cfg" --model "$model" --gpus "$gpus" --tp "$tp" \
    --datasets "${DATASETS[@]}" \
    --eval-n-samples "$EVAL_N_SAMPLES" --eval-temperature "$EVAL_TEMP" \
    --set teacher.path="$TEACHER_PATH" \
    --set "model.path=$mpath" \
    --output-dir "$out" --output-name "$model.jsonl"
}

eval_teacher() {
  # 教师基线：JustRL-1.5B，三组共享，跑一次（用 combo1 的 config；teacher.path 均=TEACHER_PATH）
  run_eval "$CFG/combo1_qwen3_1p7b.yaml" teacher "$TEACHER_PATH" 0 1 "$RES/teacher"
}

eval_student_baseline() {
  # 三组学生：<config> <学生模型id> <gpus> <tp>（7B 用 TP=2）
  local cfg m gpus tp
  while read -r cfg m gpus tp; do
    [ -z "$cfg" ] && continue
    run_eval "$cfg" student "$m" "$gpus" "$tp" \
      "$RES/student_baseline/$(basename "$cfg" .yaml)"
  done <<COMBOS
$CFG/combo1_qwen3_1p7b.yaml      $STUDENT_COMBO1 1 1
$CFG/combo2_qwen3_4b.yaml        $STUDENT_COMBO2 1 1
$CFG/combo3_r1_distill_7b.yaml   $STUDENT_COMBO3 0,1 2
COMBOS
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
echo "蒸馏后学生评估：bash watch_student.sh（watch 训练产出的 checkpoint，见 README）"
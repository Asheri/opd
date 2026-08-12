#!/bin/bash
# 单学生训练（run_all.sh 并发调用）：载入预建缓存（跳过 Stage 0/1）+ Stage 2 训练。
# 用法：bash train_one.sh <student: 7b|4b|17b> <config> <mode: real|smoke>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$(cd "$(dirname "$(dirname "$HERE")")" && pwd)"   # scripts/multistudent/ -> main/
PY=${PYTHON:-python}
S="$1"; CONFIG="$2"; MODE="$3"
source "$HERE/students.env"

DEV="${S_DEV[$S]}"
[ "$MODE" = "smoke" ] && DEV="cpu"
mkdir -p "${S_RUN[$S]}"

echo "[$S] train  start device=$DEV batch=${S_BATCH[$S]} -> ${S_RUN[$S]}"

# real: 传模型路径（hf 用）+ 每档步数；smoke: toy 忽略路径、步数走 config（5）
EXTRA=()
if [ "$MODE" = "real" ]; then
  EXTRA=(--set stage2.n_steps="${S_NSTEPS[$S]}"
         --set student_path="${S_PATH[$S]}"
         --set teacher_rl_path="$TEACHER_RL_PATH"
         --set teacher_ref_path="$TEACHER_REF_PATH"
         --set eval.model_path="${S_PATH[$S]}")
fi

cd "$MAIN"
$PY -m fullstack_opd_v2 train --config "$CONFIG" --run-dir "${S_RUN[$S]}" --device "$DEV" \
  --set stage1.cache_path="${S_CACHE[$S]}" --set stage1.load_cache=true \
  --set stage2.batch_size="${S_BATCH[$S]}" \
  "${EXTRA[@]}"

echo "[$S] train done -> ${S_RUN[$S]}"

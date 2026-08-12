#!/bin/bash
# 训练期 AIME 评估 watcher：周期性地把最新学生 checkpoint 在空闲卡后台评 AIME24/25。
# AIME 极轻（~60 题，7B 贪心 ~1-3 分钟）→ 快照 + 独立进程评估，不碰训练权重。
#
# ⚠️ 骨架约束（GPU_MEMORY_AND_PARALLEL_PLAN §7.3）：
#   - 4B/1.7B 快照：随时在 cuda:1 气泡跑（进程隔离，零扰动）；
#   - 7B 快照：rank0 训练已占 ~91GB，同卡评估会 OOM——真实跑时应在 checkpoint 边界
#     短暂让出（或把 7B 评估留到训练结束），本 watcher 对 7B 仅打日志不抢跑。
# 用法：bash eval_watch.sh   （由 run_all.sh real 模式启动；INTERVAL 秒一轮）
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$(cd "$(dirname "$(dirname "$HERE")")" && pwd)"   # scripts/multistudent/ -> main/
PY=${PYTHON:-python}
source "$HERE/students.env"

INTERVAL="${EVAL_INTERVAL:-300}"          # 每轮间隔（秒）
[ "${1:-}" = "once" ] && INTERVAL=0

while true; do
  for s in $STUDENTS; do
    # 该学生最新 checkpoint（存在且比上次评估新才评）
    latest=$(ls -t "${S_RUN[$s]}"/checkpoints/step_*.pt 2>/dev/null | head -1 || true)
    [ -n "$latest" ] || continue
    stamp="${S_EVALOUT[$s]}/.last_eval"
    # 已评过这个快照（latest 不比 stamp 新）→ 跳过
    [ -f "$stamp" ] && [ ! "$latest" -nt "$stamp" ] && continue

    # 7B 与训练同卡（cuda:0 已满）→ 只记录，避免 OOM 干扰训练（真实跑法见头注释）
    if [ "$s" = "7b" ]; then
      echo "[eval] 7b 快照 $latest 待 checkpoint 边界评估（同卡训练占满，跳过本轮）"
      continue
    fi

    out="${S_EVALOUT[$s]}"
    mkdir -p "$out"
    echo "[eval] $s 评估快照 $latest -> $out"
    # --run-dir 桥接：读 run 目录 config.yaml 的 eval.model_path（train_one real 模式已写）
    (cd "$MAIN" && $PY -m fullstack_opd_v2 eval-aime --run-dir "${S_RUN[$s]}" \
        --datasets AIME24 AIME25 --out "$out" --device "${S_DEV[$s]}") \
        >> "$WORK_ROOT/eval_$s.log" 2>&1 \
      && touch "$stamp" \
      || echo "[eval] $s 评估失败（见 $WORK_ROOT/eval_$s.log）"
  done
  [ "$INTERVAL" -eq 0 ] && break
  sleep "$INTERVAL"
done

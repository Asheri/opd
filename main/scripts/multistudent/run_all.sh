#!/bin/bash
# 多学生并发训练总编排（GPU_MEMORY_AND_PARALLEL_PLAN.md §7 最终方案）
#
# 阶段：
#   Phase 1  并行建 3 份缓存（opd cache，各学生默认 student_init → 缓存各建 §7.0；
#            7B→cuda:0，4B/1.7B→cuda:1 并行）
#   Phase 2  并发起 3 个训练（opd train --set stage1.load_cache=true，跳过 Stage 0/1）
#   Phase 3  训练期 AIME 评估 watcher（real 模式；smoke 跳过——toy 无法跑真实 AIME）
#
# 用法：
#   bash run_all.sh real     # 真实 HF + GPU（默认；需已下载模型，见 students.env）
#   bash run_all.sh smoke    # toy + CPU 本地冒烟：验证编排逻辑（缓存并行建 + 训练并行跑）
#
# smoke 前台等待训练结束（确定性验证）；real 后台启动训练并返回（运维用 nohup/log 监控）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MAIN="$(cd "$(dirname "$(dirname "$HERE")")" && pwd)"   # scripts/multistudent/ -> main/
PY=${PYTHON:-python}
MODE="${1:-real}"
source "$HERE/students.env"

case "$MODE" in
  real)  CONFIG="$HERE/student_real.yaml" ;;
  smoke) CONFIG="$HERE/student_smoke.yaml" ;;
  *) echo "用法: bash run_all.sh [real|smoke]"; exit 1 ;;
esac

echo "================================================================"
echo "  多学生并发训练 · $MODE 模式 · 工作根目录 $WORK_ROOT"
echo "  打包（§7.2）: 7B→${S_DEV[7b]}  4B+1.7B→${S_DEV[4b]}/${S_DEV[17b]}"
echo "================================================================"
mkdir -p "$WORK_ROOT"

# ---- Phase 1：并发建 3 份缓存 ----
echo "--- Phase 1: 并行建缓存（${S_CACHE[7b]##*/} / ${S_CACHE[4b]##*/} / ${S_CACHE[17b]##*/}）---"
pids=()
for s in $STUDENTS; do
  (
    DEV="${S_DEV[$s]}"; [ "$MODE" = "smoke" ] && DEV="cpu"
    EXTRA=()
    if [ "$MODE" = "real" ]; then
      EXTRA=(--set student_path="${S_PATH[$s]}"
             --set teacher_rl_path="$TEACHER_RL_PATH"
             --set teacher_ref_path="$TEACHER_REF_PATH")
    fi
    if ! cd "$MAIN" && $PY -m fullstack_opd_v2 cache --config "$CONFIG" \
        --out "${S_CACHE[$s]}" --device "$DEV" \
        --set stage1.cache_path="${S_CACHE[$s]}" \
        --set stage2.batch_size="${S_BATCH[$s]}" \
        "${EXTRA[@]}" > "$WORK_ROOT/cache_$s.log" 2>&1; then
      echo "  [cache] $s 失败（见 $WORK_ROOT/cache_$s.log）" >&2
      exit 1
    fi
    echo "  [cache] $s -> ${S_CACHE[$s]}"
  ) &
  pids+=($!)
done
# P2（二次审查）：wait 单失败即被 set -e abort 会留下孤儿进程（GPU 上占显存）。
# 逐 p 收集失败码；任一失败 → 报错退出（子进程已各自结束或由 trap 清理）。
FAIL=0
for p in "${pids[@]}"; do wait "$p" || FAIL=1; done
if [ "$FAIL" -ne 0 ]; then
  echo "ERROR: 缓存构建失败（见 $WORK_ROOT/cache_*.log）；中止多学生流程" >&2
  exit 1
fi
echo "--- 缓存构建完成（3 份并行）---"

# ---- Phase 2：并发起 3 个训练 ----
echo "--- Phase 2: 并发训练（load_cache=true 跳过 Stage 0/1）---"
PIDS=()
for s in $STUDENTS; do
  if [ "$MODE" = "smoke" ]; then
    bash "$HERE/train_one.sh" "$s" "$CONFIG" "$MODE" > "$WORK_ROOT/train_$s.log" 2>&1 &
  else
    # P3（二次审查）：real 模式用 > 截断而非 >> 追加——重复 run_all（如失败重跑）时
    # 日志与新输出混排无从分辨。
    nohup bash "$HERE/train_one.sh" "$s" "$CONFIG" "$MODE" \
        > "$WORK_ROOT/train_$s.log" 2>&1 &
  fi
  PIDS+=($!)
  echo "  [train] $s pid=$! -> ${S_RUN[$s]}  (log: $WORK_ROOT/train_$s.log)"
done

# ---- Phase 3：训练期评估 watcher（real 模式）----
if [ "$MODE" = "real" ]; then
  nohup bash "$HERE/eval_watch.sh" >> "$WORK_ROOT/eval_watch.log" 2>&1 &
  echo "  [eval]   watcher pid=$! （训练期评估，见 §7.3）"
fi

# smoke：前台等待训练结束（确定性验证编排）；real：返回，训练后台跑
if [ "$MODE" = "smoke" ]; then
  for p in "${PIDS[@]}"; do wait "$p"; done
  echo "================================================================"
  echo "  冒烟完成 ✓"
  for s in $STUDENTS; do
    echo "  $s: run_dir=${S_RUN[$s]}  metrics=$(wc -l < "${S_RUN[$s]}"/metrics.csv 2>/dev/null || echo 0) 行"
  done
  echo "  产物: 缓存 ${S_CACHE[@]}  +  3 个 run 目录"
else
  echo "================================================================"
  echo "  已并发启动 3 个训练 + 评估 watcher（后台）"
  echo "  监控: tail -f $WORK_ROOT/train_*.log   nvidia-smi"
  echo "  训练结束评估汇总: $PY $MAIN/../benchmarks/aime24_25/aggregate.py $WORK_ROOT/eval"
fi

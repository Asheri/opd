#!/bin/bash
# E3 完整重跑：batch=2（batch=4×2048 超显存卡死），n_steps=20，与 E1/E2 对齐
set -e
cd /root/opd/main
export PYTHONPATH=/root/opd/main:/root/opd/main/scripts
echo "=== E3 完整重跑(2048,batch2,n20) 启动 $(date +%H:%M:%S) ==="
timeout 900 /root/miniconda3/bin/python /root/opd/main/scripts/run_s2_real.py \
  --config /root/opd/main/configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_fix \
  --names S2_E3_opd2048 \
  --n-steps 20 --device cuda:0 --eos-id 151645 --materialized 500 \
  --load-cache --cache-path /root/autodl-tmp/cache_skywork_17b.pt \
  --batch-size 2
rc=$?
if [ $rc -eq 124 ]; then echo "❌ E3 完整重跑再次卡死(timeout 900)"; else echo "✅ E3 完整重跑完成 rc=$rc (0=正常)"; fi
echo "=== E3 完整重跑结束 $(date +%H:%M:%S) ==="

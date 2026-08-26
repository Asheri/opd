#!/bin/bash
# E3 重跑 v3：batch=1（batch2 仍 OOM），expandable_segments 防碎片，n_steps=20
set -e
cd /root/opd/main
export PYTHONPATH=/root/opd/main:/root/opd/main/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== E3 重跑v3(batch1,expandable,n20) 启动 $(date +%H:%M:%S) ==="
timeout 900 /root/miniconda3/bin/python /root/opd/main/scripts/run_s2_real.py \
  --config /root/opd/main/configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_fix \
  --names S2_E3_opd2048 \
  --n-steps 20 --device cuda:0 --eos-id 151645 --materialized 500 \
  --load-cache --cache-path /root/autodl-tmp/cache_skywork_17b.pt \
  --batch-size 1
rc=$?
if [ $rc -eq 124 ]; then echo "❌ E3 batch1 再次卡死(timeout 900)"; else echo "✅ E3 batch1 完成 rc=$rc (0=正常)"; fi
echo "=== E3 batch1 结束 $(date +%H:%M:%S) ==="

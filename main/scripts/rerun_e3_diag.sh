#!/bin/bash
# E3 卡死诊断：batch2 + n_steps5 复现，timeout 600s 防再一次无限卡死
set -e
cd /root/opd/main
export PYTHONPATH=/root/opd/main:/root/opd/main/scripts
echo "=== E3 诊断启动 $(date +%H:%M:%S) ==="
timeout 600 /root/miniconda3/bin/python /root/opd/main/scripts/run_s2_real.py \
  --config /root/opd/main/configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_diag \
  --names S2_E3_opd2048 \
  --n-steps 5 --device cuda:0 --eos-id 151645 --materialized 500 \
  --load-cache --cache-path /root/autodl-tmp/cache_skywork_17b.pt \
  --batch-size 2
rc=$?
if [ $rc -eq 124 ]; then
  echo "❌ E3 诊断 【再次卡死】 timeout 600s 触发 (exit 124)"
else
  echo "✅ E3 诊断完成 rc=$rc (0=正常, 非0=异常)"
fi
echo "=== E3 诊断结束 $(date +%H:%M:%S) ==="

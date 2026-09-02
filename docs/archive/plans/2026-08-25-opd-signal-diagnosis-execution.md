# 2026-08-25：OPD 信号诊断执行序列（D3→D1→D2→正式训练）服务器命令

> 状态：本地代码完成并提交 d30e401；服务器 SSH 持续不可达（35318 端口 refused），
> 恢复后按本节顺序执行。硬约束：判据门控、不改核心训练目标、GPU≥2 双卡并行、
> 不伪造结果、报错追加 training-errors.md。

## 0. 同步代码（本地 → 服务器）

```bash
# 本地（PowerShell/任意 shell）：
scp -P 35318 main/fullstack_opd_v2/cache.py main/fullstack_opd_v2/cache_store.py \
  main/fullstack_opd_v2/config.py main/fullstack_opd_v2/scheduler.py \
  main/fullstack_opd_v2/pipeline.py main/scripts/inspect_delta_cache.py \
  main/tests/test_inspect_delta_cache.py main/tests/test_eval_holdout.py \
  root@connect.westd.seetacloud.com:/root/opd/main/fullstack_opd_v2/  # 注意分目录
# 实际按目标目录逐个 scp：fullstack_opd_v2/ → 5 个文件；scripts/ → 1 个；tests/ → 2 个
```

## 1. 服务器回归

```bash
cd /root/opd/main
/root/miniconda3/bin/python -m pytest tests/ -q   # 期望 523 passed
```

## 2. D3：Δ_T 信号体检（0.5h，纯离线零训练，CPU）

```bash
cd /root/opd/main
/root/miniconda3/bin/python -u scripts/inspect_delta_cache.py \
  --prefix /root/autodl-tmp/cache_skywork_chat \
  --out /root/autodl-tmp/d3_report.json
```

判据（写死）：正 Δ 占比 ≥15% 且 |均值|≤1.0 → PASS 进 D1；
<5% 或 均值<-1.0 → FAIL 停一切训练，回 C3 审计教师模板；
5%-15% 或 -1.0≤均值<-0.5 → BOUNDARY 记录风险仍进 D1。

## 3. D1：固定评估集 80 步探针（D3 PASS 后，单实验 E2 口径）

```bash
cd /root/opd/main
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup /root/miniconda3/bin/python -u scripts/run_s2_real.py \
  --config configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_d1_probe \
  --names S2_E2_opd1024 --n-steps 80 \
  --eos-id 151645 --materialized 500 --load-cache \
  --batch-size 2 --refresh-size 64 \
  --set stage2.eval_holdout_size=64 --set stage2.eval_every=10 \
  --set stage2.gradient_checkpointing=true --set stage2.teacher_offload=true \
  --set stage2.refresh_chunk=2 --set stage2.rollout_engine=vllm \
  --set stage2.rollout_gpu_mem=0.12 --set stage2.rollout_max_model_len=2048 \
  --set stage2.rollout_max_num_seqs=8 --set stage2.offload_to_cpu=true \
  --set stage2.queue_size=2 --set stage2.staleness_queue_min=2 \
  --set dataset.apply_chat_template=true --set dataset.max_response_len=2048 \
  --set stage1.cache_path=/root/autodl-tmp/cache_skywork_chat.pt \
  --set l2.rollout.repetition_penalty=1.0 \
  > /tmp/d1_probe.log 2>&1 &
```

判据：metrics.csv 中 eval_reward 末段（step 60-80 均值）> 首段（step 0-20 均值）+0.05 → PASS 进正式训练（跳过 D2）；
否则 → D2。

## 4. D2：KL 消融三档（仅 D1 不过时，三组各 40 步同一 holdout）

- 组 A：`--set stage2.kl_reg_coef=0.5`（现状）
- 组 B：`--set stage2.kl_reg_coef=0.1`
- 组 C：`--set stage2.kl_reg_coef=0.02`
- 其余配置同 D1（batch=2、v16 工程链路、eval_every=10、同一 eval_holdout_size=64）
- 任一档末段>首段+0.05 → 该档进正式训练；三档全不升 → 停训练输出算法审计报告
- 双卡并行：三组可 2 并行 + 1 串行（--parallel 或 shell 双 nohup）

## 5. 正式训练（D1/D2 通过后，300 步）

见任务原文「正式训练配置模板」：batch4 需先单步首验显存（<90GB），OOM 回退 batch2+n_steps=600；
kl_reg_coef 取 D2 最优（默认 0.1）；eval_every=20；checkpoint_every=20；
配套评估：step 0/100/200/300 跑 MATH500 B512、最优 checkpoint 跑 AIME24@B4096。

## 6. 验收（全部满足才算成功）

1. 固定集 eval_reward：末 100 步均值 > 首 20 步均值 + 0.05
2. MATH500 B512 acc ≥ 0.086（base 水平）
3. AIME24 ≥ base 水平（不重演 S2_E3=0.000）
4. 全程无 OOM/死锁/孤儿引擎/权重同步失败
5. metrics.csv 完整（300 步 + refresh + eval_reward 列）

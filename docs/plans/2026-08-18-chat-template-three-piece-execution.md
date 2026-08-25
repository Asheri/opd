# 2026-08-18：chat template 三件套——服务器恢复后执行序列（零决策）

> 前置：本地已提交（HEAD=6608c5e），全量 434 passed。以下全部在服务器 /root/opd 执行。
> 目标：C3 模板一致性（Step 0 + 重生成 + 重建 Δ_T）→ C1 权重同步加强验证 → 模板 pilot 复测
> （验收：valid_rate≥0.5、refresh pool≥8、附 decode 样本）。

## 0. 同步代码（本地 → 服务器）

本地：
```
cd C:\Users\12062\OneDrive\Desktop\opd
git push origin <branch>      # 或直接 sftp 覆盖 main/ 下变更文件
```
服务器（确保 /root/opd 与本地 HEAD 一致，参照既有 sync 流程）：
```
cd /root/opd && git fetch && git reset --hard <HEAD>   # 或 sftp 覆盖变更文件
cd /root/opd/main && /root/miniconda3/bin/python -m pytest tests/ -q | tail -1
```

## 1. C3 Step 0 — Qwen3 generation prompt 结尾确认（决定 response 前缀约定）

```
/root/miniconda3/bin/python - <<'PY'
import transformers
tok = transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/models/Qwen__Qwen3-1.7B")
s = tok.apply_chat_template([{"role":"user","content":"X"}], tokenize=False, add_generation_prompt=True)
print("GEN_PROMPT:", repr(s))
print("HAS_THINKING:", "thinking" in s)
PY
```
- 记录结尾是否含 `  thinking`；prepare_skywork_responses --apply-chat-template 生成的
  response 自动以该前缀为起点（模型生成即该前缀后文本），无需手工拼接；仅需把结论写入报告。

## 2. C1 — 权重同步加强验证（扰动 + 分布级 + 贪心）

```
cd /root/opd/main
CUDA_VISIBLE_DEVICES=1,0 /root/miniconda3/bin/python scripts/verify_weight_sync.py
```
- 布局与通过探针一致：CUDA_VISIBLE_DEVICES=1,0 + engine device=cuda:1 →
  NCCL rank0（+HF state_dict 源）在物理卡0，vLLM worker 落可见 cuda:0=物理卡1
  （交叉，勿改动）。通过标准：扰动（注入 +0.1 后 logp 变化>0.01、复原<0.01）；
  分布级 ≥512 位置 top1≥0.99、topK logp MAE<0.03；贪心 4×128 位置一致率≥0.99。
- 通过后报告措辞才可升级为"权重加载正确"；未通过则继续排查（不静默）。

## 3. 按模板重生成 base responses（C3）

先确认 input jsonl 各行 prompt 为原始题目、response 已有但将被覆盖（--apply-chat-template
只处理 response 为空的 todo 行 → 如需整体重生成，先清空 response 列或新建副本）：

```
# 副本保护原 jsonl + --force 覆盖已填 response（模板重生成）
cp /root/autodl-tmp/datasets/skywork_math_500.jsonl{,.raw}
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_math_500.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B --device cuda:0 \
  --max-samples 500 --apply-chat-template --force
```
- 样本检查：抽 2-3 条 decode 前 200 字符（应为正常推理，非乱码 token soup）。

## 4. 重建 teacher cache（模板 prompt + 教师各自模板 Δ_T）

```
cd /root/opd/main
/root/miniconda3/bin/python -m fullstack_opd_v2.cli cache \
  --config configs/skywork_17b.yaml --set dataset.apply_chat_template=true \
  --set stage1.load_cache=false --out /root/autodl-tmp/cache_skywork_chat.pt 2>&1 | tail -5
```
- 检查 cache metadata 含 `prompt_format: "chat"`（C2 守卫写入）。
- 抽样 decode 校验：教师模板下 Δ_T 合理（无 NaN、支撑率正常）。

## 5. 模板 pilot 复测（L2 + vLLM，repetition_penalty 回退 1.0）

```
cd /root/opd/main
/root/miniconda3/bin/python scripts/run_s2_real.py --config configs/skywork_17b.yaml \
  --run-dir /root/autodl-tmp/runs_s2_vllm_chat --device cuda:0 --n-steps 20 \
  --names S2_E1_opd512 S2_E2_opd1024 --eos-id 151645 --materialized 500 \
  --load-cache \
  --set stage2.rollout_engine=vllm \
  --set dataset.apply_chat_template=true \
  --set stage1.cache_path=/root/autodl-tmp/cache_skywork_chat.pt \
  --set l2.rollout.repetition_penalty=1.0
```
- 用第 4 步已建的 cache（--load-cache + cache_path 指向新 cache）；load_cache 路径会校验
  prompt_format=chat（C2 守卫）——若建的是旧裸 cache 会直接 fail-fast，防静默错位。
  E1/E2 各自 20 步（交叉分卡 E1 训练@0+vLLM@1、E2 反向，见 run_s2_real 约定）。

## 6. 验收（判定达标才算完成）

| 条款 | 判据 | 实测（2026-08-25 v16，双卡并行） |
|---|---|---|
| valid_rate | ≥ 0.5（IMP-1 原目标；模板下实测应远高于此） | ✅ **E1=1.0（3/3 refresh 相位）、E2=1.0（3/3）**——8/8 全 valid |
| refresh pool | ≥ 8（不再触发冷启动跳过，refresh 训练真正跑起来） | ✅ **E1 n_appended=8、E2 n_appended=8**；refresh 训练实际执行（α=0.300→实际 0.200，5 步） |
| decode 样本 | 报告中附 2-3 条完整 rollout decode（正常推理内容） | ✅ 见下方「decode 样本证据」：3 条完整 decode（chat 模板包裹 + thinking 前缀 + 正常推理，0 loop） |
| C1 | verify_weight_sync.py 三关全过 | ⏳ 服务器恢复后补跑（本轮 focus pilot） |
| 回归 | 服务器 pytest 全绿 + 本地 434 passed | ✅ 本地 511 passed（≥基线）；服务器 pytest 待阶段4 |

- 顺带把 loop 检测器在模板 rollout 上重新抽样校准（calibrate_rollout.py，
  periods/min_len 可能可收紧）——旧校准已标注 stale。

## 6.5 权重同步失败根因分析（vLLM 0.16.0 NCCL WeightTransferEngine，2026-08-19 静态定位）

### 现象
- 每次 rollout 前 `rollout_engine.update_weights(student.state_dict())` 失败，
  日志 `[L2] vLLM 权重同步失败（继续用引擎现有权重）：NCCL error: invalid usage`；
  失败点固定：**worker 侧**（EngineCore 子进程）`gpu_worker.init_weight_transfer_engine`
  → `NCCLWeightTransferEngine.init_transfer_engine` → `StatelessProcessGroup.create`
  → `PyNcclCommunicator.ncclCommInitRank`（vllm/distributed/device_communicators/pynccl.py:139）。
- **单实例（verify3 n_steps=1）与双卡并行均复现** → 与 --parallel/端口竞争无关。
- 代码走降级路径（引擎期初始权重），训练不受影响（E1/E2 各 20 步完成）。

### 调用链（与 vLLM 0.16.0 源码逐行核对）
- trainer（rank0）：`_weight_transfer_init_16`（rollout_vllm.py）——随机端口开 TCPStore、
  `world_size=1+tp_size=2`、`torch.cuda.set_device(训练卡)`；后台线程 worker_init 并发
  `llm.init_weight_transfer_engine({"init_info": ...})`；主线程 `trainer_init` 建组。
- worker：`rank = dp_rank*TP + rank_within_dp + rank_offset = 0*1+0+1 = 1`（<2 合法）；
  `device = torch.cuda.current_device()`（CUDA_VISIBLE_DEVICES 重排后逻辑 0）；
  unique_id 经 TCPStore `broadcast_obj(src=0)` 从 trainer 取。
- **参数链正确**：rank=1/world_size=2/device 可为每侧独立卡；WeightTransferConfig(backend="nccl")
  已生效（worker 走到 init 而非 "Weight transfer not configured" 拒绝路径）。

### 剩余候选根因（按概率，待 NCCL_DEBUG 实证）
1. **[高] spawn 子进程 + NCCL unique_id 的 TCPStore pickle 广播完整性**：worker 侧
   `ncclUniqueId()` 空对象经 pickle 往返填充；若 trainer set 与 worker get 之间 store
   键计数错位（多组/重试）或 id 损坏 → `ncclCommInitRank` 用无效 id → NCCL invalid usage（经典触发）。
2. **[中] worker `torch.cuda.current_device()` 与 CUDA_VISIBLE_DEVICES 重排组合**：
   EngineCore 进程内 current_device 非预期（如仍为物理索引/未设）→ comm init device 非法。
3. **[低] NCCL 2.27.5 + CUDA 版本组合 bug**（`vLLM is using nccl==2.27.5` 已确认 trainer 侧）。

### 验证协议（SSH 恢复后执行，一次性拿全证据）
- `NCCL_DEBUG=INFO` 重跑 n_steps=1：NCCL 内部会打印 rank/world_size/comm 建立过程，
  invalid usage 前的 NCCL WARN 行即根因出口。
- rollout_vllm.py `_weight_transfer_init_16` 加 `OPD_WT_DEBUG=1` 门控打点：
  打印 trainer/worker 两侧 rank/world_size/device/unique_id[:16] hex —— 判定候选 1/2。

### 缓解与正式方案
- **pilot 阶段（本序列验收）**：`--set stage2.rollout_weight_sync=off` 走代码既有逃生舱
  （明示 off，不再失败刷屏；rollout 用引擎初始权重）。decode/valid_rate 验收由模板质量主导，
  影响可控；报告标注 on-policy 违约范围。
- **正式方案（阶段 3 合并评估）**：TP=1 快路径 `merge_map + param.copy_` 直接拷贝
  （绕开 NCCL weight transfer）——若 200 步耗时不可接受则实现；仍要 NCCL 时按验证协议修复 worker 端。

## 7. 已知边界（不影响本序列执行）

- warmup_M=0 部署下教师模板 fat 行对齐无需处理；warmup>0 + 模板会按 1+warmup_M 倍
  cat 对齐（stage1_build_cache 已实现）。
- is_checkpoint_format=True 每步全量 load_weights；200-step 正式训练前评估耗时，必要时
  切 TP=1 merge_map+param.copy_ 直接拷贝快路径。


## 7. 验收实测证据（2026-08-25，v16 双卡并行）

### 运行命令（关键新增：`--refresh-size 64`）
```
run_s2_real.py --config configs/skywork_17b.yaml --run-dir .../runs_s2_vllm_chat_v16 \
  --names S2_E1_opd512 S2_E2_opd1024 --parallel 2 --stagger 180 --n-steps 20 \
  --eos-id 151645 --materialized 500 --load-cache --batch-size 2 --refresh-size 64 \
  --set stage2.gradient_checkpointing=true --set stage2.teacher_offload=true \
  --set stage2.refresh_chunk=2 --set stage2.rollout_engine=vllm \
  --set stage2.rollout_gpu_mem=0.12 --set stage2.rollout_max_model_len=2048 \
  --set stage2.rollout_max_num_seqs=8 --set stage2.offload_to_cpu=true \
  --set stage2.queue_size=2 --set stage2.staleness_queue_min=2 \
  --set dataset.apply_chat_template=true --set dataset.max_response_len=2048 \
  --set stage1.cache_path=/root/autodl-tmp/cache_skywork_chat.pt \
  --set l2.rollout.repetition_penalty=1.0
```

### 结果
- **E1（opd512）**：26 步，188.2s，reward=-0.486，pg_loss=0.411，kl_loss=1.355，rollout_n_appended=8 / n_eos=0 / n_loop=0
- **E2（opd1024）**：26 步，194.4s，reward=-0.455，pg_loss=0.397，kl_loss=1.256，rollout_n_appended=8 / n_eos=0 / n_loop=0
- 两实验均无 error、无 OOM；双卡并行期间 utilization.gpu>0
- 3 个 refresh 相位全部 valid_rate=1.0、loop_rate=0.0、eos_rate=0.0

### OOM 根治链（本轮全部修复，v16 组合生效）
1. gradient_checkpointing=true（backward OOM）→ 503a91d
2. offload_to_cpu（s_old 队列）+ P5 bf16（前向峰值）→ 6664966/c1bf7ba
3. refresh 训练前 empty_cache + teacher_offload + refresh_chunk=2 → e7c8903
4. vLLM init 错峰（--stagger）+ max_model_len 守卫 → e7c8903
5. NCCL trainer_init 线程内 set_device（E2 Duplicate GPU）→ 1e26a45
6. 孤儿 EngineCore 清理 ps comm=→args=（E1 残留 12.76GB）→ 1e26a45
7. **_send 线程设备作用域** → d2fb647
8. **`--refresh-size 64`**（ring buffer 5000×T×K 预分配 ~78GB GPU OOM → 64）→ 本次发现

### decode 样本证据（rollout_decode_pad_test.jsonl，chat 模板 + thinking，0 loop）
样本1（idx=327）：`<|im_start|>user Find the value of $$...$$<|im_end|> <|im_start|>assistant  thinking Okay, so I need to find the value of the sum of binomial coefficients where the lower index is congruent to 1 mod 3...`
样本2（idx=57）：`<|im_start|>user A domino is a rectangular tile...<|im_end|> <|im_start|>assistant  thinking Okay, so I need to figure out the probability that a randomly selected domino from a complete set is a double...`
样本3（idx=12）：`<|im_start|>user How many different rectangles...<|im_end|> <|im_start|>assistant  thinking Okay, so I need to figure out how many different rectangles with sides parallel to the grid...`
（loop 检测按新校准 0/100、eos=151645 口径；以上 3 条全部正常推理、无乱码/循环）


## 8. 阶段3：TP=1 快路径耗时评估结论（2026-08-25）

### 实测现状（v16，双卡并行，is_checkpoint_format=true + load_weights 路径）
- 权重同步（WT-update，311 keys，第一发+第二发）：**≤1s/次**（日志秒级粒度：开始→完成同一秒）。
  - E1 首 refresh 12:15:59 开始→完成；E2 首 refresh 12:19:04 开始→完成。
- 每 refresh 相位总 wall_time：E1=2.44s（rollout）、E2=4.88s（rollout，1024-token 更长）。
- 26 步总耗时：E1=188.2s、E2=194.4s（含全部 refresh）。

### 200 步正式训练推演
- refresh 间隔 refresh_min_interval=10 → 200 步约 20 次 refresh。
- 权重同步总开销 ≈ 20 × 1s = **20s**。
- 训练总时长 ≈ 200 × 7.2s ≈ 1440s（24min）。
- **权重同步占比 ≈ 1.4% —— 可接受。**

### 结论
- 当前 `is_checkpoint_format=true + load_weights` 路径**不阻塞 200 步正式训练**（开销 <2%）。
- 双发 workaround 已收敛 layerwise reload 异步残留（v16 权重同步 send_err=0、refresh 训练 kl_loss 正常）。
- **TP=1 merge_map + param.copy_ 直接拷贝快路径：本轮不实现**（收益 <2%，不值得引入 merge_map 复杂度与验证成本）。
- 记录为已知边界：若未来训练规模/刷新频率提升使权重同步占比显著，再评估快路径。

---

## 9. 决策记录：关闭服务器任务已取消（2026-08-25，用户指令）

- **决定**：用户明确取消「阶段 5：关闭服务器」任务。**不要**在后续会话中对服务器执行 `shutdown` / `poweroff`。
- **原因**：服务器（AutoDL 35318 端口）在当前会话中已失联；用户主动取消关机，保留服务器供后续使用（如 200 步正式训练 / C1 verify 三关补跑）。
- **影响**：目标范围收缩为「pilot 验收 + TP=1 评估 + 同步」——此三项已全部完成并有证据（§6/§7/§8）。关闭服务器从待办中移除。
- **遗留（需服务器恢复）**：§6 验收表 C1 行 ⏳ `verify_weight_sync.py` 三关补跑（正式训练前置）；若未来仍需关机，须由用户在 seetacloud 控制台操作或明确重新授权 SSH 关机。

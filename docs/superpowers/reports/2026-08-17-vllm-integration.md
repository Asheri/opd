# IMP-2/P0 报告：训练管线 vLLM 提速接入

> 日期：2026-08-17，更新 2026-08-18 ｜ 状态：**代码完成 + 单测全绿；GPU 端到端验证已跑通
> （NCCL 权重同步 + E1/E2 20 步 pilot 完成，无权重同步 warning）；三件套（C2 守卫已落地 /
> C3 教师模板一致性代码已落地 / C1 加强验证探针待服务器恢复）**
> 服务器：2x RTX PRO 6000 Blackwell（vLLM 0.16.0，FP8 sm_120 已确认）

## 1. 已修改文件

| 文件 | 修改 |
|---|---|
| main/fullstack_opd_v2/rollout_vllm.py | response_dists_topk 加 K 参数 + 按 logprob 降序排序修复（vLLM dict 迭代顺序无序，原实现截断 top-K/searchsorted 会错乱） |
| main/fullstack_opd_v2/adaptive_cache.py | run_refresh_phase 加 dist_engines 参数；新增 _dist_topk_cached / _gather_support / _dist_rl_ref_delta（vLLM 引擎 vs HF per-chunk 回落） |
| main/fullstack_opd_v2/pipeline.py | rollout_engine=vllm 时按 l2.rollout.dist_engines（默认 false）构造 4 个分布引擎（s_old/rl/ref/ref_anchor），各低 gpu_memory_utilization 共存；rollout 相位前 s_old 引擎 update_weights（on-policy）；run_refresh_phase(dist_engines=...) |
| main/tests/test_vllm_dist.py | P0 纯函数单测（7 用例） |

## 2. 修改目的

训练侧 VLLMRolloutEngine 已有 generate_with_status / response_dists_topk / update_weights，
但 rollout 相位的 4 个分布前向（s_old/rl/ref/ref_anchor）仍是 HF per-chunk。本阶段把分布
前向切 vLLM（连续批处理 + FP8 + 稀疏 top-K），rollout 不再是训练瓶颈。

结构边界：_train_step / _train_step_refresh 的 s_cur 带梯度（backward 必需），vLLM 无梯度
不可替换 → 保持 HF，未改（符合计划）。

## 3. 数据流变化

- rollout 生成：vLLM generate_with_status（已有，独立卡 cuda:1）——前一提交已接。
- rollout 相位分布（本次）：
  - s_old = dist_engines[s_old].response_dists_topk(p_b_v, resp_v, K=Ks)（rollout 前同步 student 权重）
  - rl/ref delta = 双引擎各自 response_dists_topk，_gather_support（searchsorted）取 ref 在
    rl top-K 支撑的 logp → delta = rl_k - ref_at_rl（与 HF _rl_ref_delta_k 数值对齐）
  - ref_anchor = dist_engines[ref_anchor].response_dists_topk（初始 student，不同步）
- 任一引擎缺失 → 该前向回落 HF per-chunk（dist_engines=None 全 HF，零回归）。

## 4. 新增 config

| 键 | 默认 | 说明 |
|---|---|---|
| l2.rollout.dist_engines | false | 启用 rollout 相位分布前向走 vLLM（可禁用） |
| l2.rollout.dist_engine_gpu_mem | 0.25 | 每个分布引擎 gpu_memory_utilization（4x~0.25<=1.0 共存 rollout_device） |
| stage2.rollout_device | cuda:1 | vLLM 引擎独立卡（训练 cuda:0） |
| stage2.rollout_model / rollout_dtype | -/auto | 生成引擎模型/精度（dtype=fp8 可提速） |

## 5. Tests

- test_vllm_dist.py：7 passed（_gather_support 命中/未命中/multi-T、_dist_topk_cached 引擎优先、
  None 走 HF、_dist_rl_ref_delta 双引擎 delta 数值、引擎缺失回落 HF）。
- 回归：test_l2_rollout + test_adaptive_cache + test_pipeline 92 passed（dist_engines 默认 None 零回归）。
- 上一提交 test_budget_eval / curve 44 passed（评估批量修复）。

## 6. GPU validation（2026-08-17 已完成，2026-08-18 复核）

- **权重同步打通**：vLLM 0.16 NCCL WeightTransferEngine（`weight_transfer_config` 引擎启动注入 +
  交叉分卡 trainer@GPU0/vLLM@GPU1）。探针 INIT + UPDATE-1/2 均 `True`；同步后 vLLM vs HF
  logits `top1_match=0.875`、`topk_logp_absdiff=0.072`（**静态一致性初步通过**；扰动/贪心
  加强验证见 §7 风险①，待服务器恢复执行）。
- **E1/E2 pilot 完成**：各 20 步，reward≈-0.19 / pg_loss≈0.22-0.23 / kl_loss≈0.84-0.89，
  **无权重同步 warning**。
- **rollout 质量根因（2026-08-18 实测）**：生成内容为乱码+loop，根因是 prompt 未套 Qwen
  chat template（裸 prompt `*. 202951173.` vs 套模板正常推理、0 loop）→ C2/C3 修复。
- **验收条款（C3 模板生效后判定）**：valid_rate ≥ 0.5（IMP-1 原目标）；refresh pool ≥ 8
  （不再触发冷启动跳过）；报告中附 2-3 条完整 decode 的 rollout 样本供人工检查。

## 7. Remaining risks（2026-08-18 复核后）

① **权重同步验证仅为静态一致性**（同步协议打通、logits 初步一致）；"扰动测试 + ≥512 位置
   greedy 逐 token 对比（一致率≥0.99、logit MAE<0.03）" 尚未执行（C1，待服务器恢复）——通过前
   报告措辞不升级为"权重加载正确"。
② **base response / 教师模板一致性未验证**：C3 代码已落地（prepare_skywork_responses
   --apply-chat-template、教师各自模板 Δ_T、cache metadata prompt_format 守卫 C2），
   但 Qwen3 generation prompt 结尾确认（是否含 thinking 前缀）+ 按模板重生成 base responses +
   教师模板 Δ_T 重建 + 抽样 decode 校验需服务器恢复后执行。
③ **单卡共置布局从此不可用**：NCCL 权重同步要求 trainer(rank0) 与 vLLM worker(rank1) 异卡
   （Duplicate GPU detected），单卡共置必然卡死；4 卡/DP 布局规划须预留互斥物理卡对。
④ **is_checkpoint_format=True 走完整 load_weights**：TP=1 下官方快路径是 merge_map +
   param.copy_ 直接拷贝；当前实现每次 update 都调 model.load_weights（is_checkpoint_format），
   200-step 正式训练前应评估每步同步耗时，必要时切 direct-copy。

## 8. 是否允许进入下一阶段

本地代码层放行（单测 + 回归绿）；GPU 端到端验证已跑通（权重同步 + E1/E2 pilot，无 warning）。
**放行条件（C3 模板启用后）**：C2 守卫已生效（fail-fast）；C3 Step 0（Qwen3 模板结尾确认）+
  按模板重生成 base responses + 教师各自模板重建 Δ_T + 抽样 decode 校验；C1 扰动/贪心验证通过；
  模板 pilot 复测满足验收条款（valid_rate≥0.5 / refresh pool≥8 / 附 decode 样本）。
  本次会话累计 24 笔提交（含本次），关键 3 笔：NCCL 权重同步打通、top_p 对齐、chat template 根因。

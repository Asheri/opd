# IMP-2/P0 报告：训练管线 vLLM 提速接入

> 日期：2026-08-17 ｜ 状态：**代码完成 + 本地单测通过；P1 GPU 端到端验证待服务器恢复**
> 服务器：2x RTX PRO 6000 Blackwell（vLLM 0.16.0，FP8 sm_120 已确认）｜ 服务器当前不可达

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

## 6. GPU validation requirement（P1，待服务器恢复）

计划：
- run_s2_real.py --set stage2.rollout_engine=vllm 双卡（cuda:0/1）并行 S2_E1/E2 20 步 pilot。
- 验收：rollout valid_rate~1.0 / n_loop~0（与 toy 一致）；refresh kl_loss~1.5-1.8（不回旧锚点错位量级）；无 OOM。
- vLLM 权重同步 update_weights 若在本机版本不工作 → 明确版本适配说明（>=0.6 用 LLM.update_weights），不伪造通过。
- 顺带验证 vllm_budget_eval.py（FP8 / 投机解码吞吐对比）。

当前阻塞：服务器 connect.westd.seetacloud.com:35318 持续不可达（NoValidConnectionsError），
P1 无法执行。恢复后立即按上述验收跑。

## 7. Remaining risks

- 4 个 vLLM 引擎共存显存/启动：低 gpu_memory_utilization 方案需 GPU 实测（fp8 小模型 ~2GB/个，
  预计 OK，但以实测为准）。
- update_weights 版本兼容：vLLM 0.16 的 LLM.update_weights 签名待实测；失败会 warning 降级
  （引擎用初始权重，破坏 on-policy——报告会明确标注，不静默）。
- teacher 分布前向 vLLM 数值对齐：_gather_support 的 searchsorted 与 HF gather 等价性
  需 P1 端到端核对（delta_k 一致性）。
- compute_disagreement=True 分支的 token_logprobs 仍 HF（计划标注待办，未改）。

## 8. 是否允许进入下一阶段

本地代码层放行（单测 + 回归绿）；GPU 端到端验证待服务器恢复后执行 P1，届时按验收
项核对后正式放行。若服务器长期不可达，需用户确认是否换地址/恢复。

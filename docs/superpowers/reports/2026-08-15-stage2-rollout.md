# Stage 2 短 Rollout OPD 训练协议报告

> **状态：占位/合成。** 本报告如实标注：训练端真实数值待服务器实跑（toy/CPU 端
> 已实现并验收协议抽象，见 §验证）。评估端协议见 `budget_eval`
> （B∈{256..4096}）。

## 训练矩阵（S2_E0-E3）

| 实验 | reward_mean | pg_loss_mean | kl_loss_mean | n_steps |
| --- | --- | --- | --- | --- |
| S2_E0_static | — | — | — | — |
| S2_E1_opd512 | — | — | — | — |
| S2_E2_opd1024 | — | — | — | — |
| S2_E3_opd2048 | — | — | — | — |

## 长预算评估矩阵

（无数据）

## Q1 · 短 rollout 能否稳定产生有效 OPD learning signal？

**待服务器实跑。** 验收代理：S2_E1/E2/E3 的 `reward_mean`/`pg_loss_mean` 相对
`S2_E0_static` 有区分度；`rollout/*` 状态分布中 n_loop 占比有限（短预算内 loop
拦截不削弱信号）。toy 端已验收：`run_refresh_phase` 在 budget=8/16 下
`mean_disagreement > 0` 且 `n_loop` 有限（§验证）。

## Q2 · 1024 训练预算能否提升长预算（4096）评估？

**待服务器实跑。** 对比 `S2_E2_opd1024` @eval B=4096 vs `S2_E0_static` 基线。

## Q3 · 训练预算的边际收益（512→1024→2048）如何？

**待服务器实跑。** 对比 S2_E1/E2/E3，找边际递减拐点。

## Q4 · 训练短预算、评估长预算的迁移是否存在？

**待服务器实跑。** 验证「训练短、评估长」协议；若成立，训练吞吐大幅提升。

---

## 实现状态（已落地，本地全绿）

- `config.py`：`L2RolloutCfg` + `l2.rollout`（max_new_tokens=1024 / allow_budget_stop /
  eos_token_id=None / loop_detection / pad_id）。
- `model.py`：`generate_with_status`（toy 逐 token + EOS 提前停 + alive 掩码）、
  `build_length_mask`（长度式，不扫 pad）、`detect_loop`（尾部周期）。`generate_batch` 不动。
- `rollout_vllm.py`：`generate_with_status` + 纯函数 `parse_vllm_outputs`（CPU 可测）。
- `adaptive_cache.py`：`run_refresh_phase` 注入 rollout_generator + status 判定 +
  loop/invalid 跳过 teacher 前向与 append；`RefreshRingBuffer` 存 status（state_dict 往返）。
- `pipeline.py`：消费 `l2.rollout.max_new_tokens`（fallback cache.max_response_length）、
  注入 vLLM generator、`rollout/*` 指标落 CSV。
- `experiment.py`：`STAGE2_ROLLOUT_MATRIX`（S2_E0-E3）+ build_config/run_matrix 泛化 matrix。
- `report_stage2.py`：Q1-Q4 报告（无数据降级）。

## 验证

`cd main && python -m pytest tests/ -q` 全绿（含新增 test_l2_rollout.py 21 项：
config 默认/覆盖/拒未知键；detect_loop 真假/短序列；generate_with_status 无 eos 全
budget_stop / eos 提前停 mask / loop；generate_batch 回归；parse_vllm_outputs eos/budget/loop
/loop-disabled；run_refresh_phase 注入 + status 往返 + 全 loop 跳过；pipeline 消费 max_new
+ fallback + 落盘；S2 矩阵建配置/跑通；报告 Q1-Q4 占位与带数据）。

## 服务器待执行项

1. GPU 真实模型跑 S2_E1/E2/E3（真实 512/1024/2048 rollout）+ `budget_eval` B=4096
   长预算评估，产出 Q2-Q4 数值。
2. 校准 `l2.rollout.eos_token_id`（真实词表取真实 EOS）与 `loop_periods`（真实模型
   退火尾部周期）。
3. 更新本报告数值 + `TECHNICAL_REPORT.md` §5/§8 短 rollout 协议说明。
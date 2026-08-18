# IMP-1 修复报告：Rollout Loop 根因（pad_id）+ 显存 + 双卡并行

> ⚠️ **修正注记（2026-08-18，C3）**：本报告把 pad_id 定位为「校准 0% vs 训练 75%」的唯一根因，
> **不完整**。2026-08-18 GPU 实测补充：训练/rollout 的 prompt **一直没套 Qwen chat template**——
> 裸数学题 prompt 下 Qwen3 生成乱码 token soup（`*. 202951173.`）且 6/8~7/8 loop；套模板后
> 生成正常推理且 0 loop。即 pad_id 只解释了同一乱码分布的「训练 vs 校准」差异，模板缺失才是
> rollout 质量问题的更根本原因。修复为三件套：C2 metadata 守卫 / C3 教师各自模板 Δ_T /
> C1 权重同步加强验证（详见 2026-08-18 会话计划与实现）。

> 日期：2026-08-17 ｜ 状态：**完成（GPU 双卡实测验证）** ｜ 服务器：2×RTX PRO 6000（95GB×2）

## 1. 已修改文件

| 文件 | 修改 |
|---|---|
| main/fullstack_opd_v2/model_factory.py | HFCausalLM.pad_token_id 在 config 无 pad 时回落到 AutoTokenizer.pad_token_id（Qwen3=151643） |
| main/fullstack_opd_v2/adaptive_cache.py | rollout 前 mpty_cache()；rollout 分布计算 per-chunk（_response_dists_topk/_rl_ref_delta_k），不再驻留完整 (M,T,V) |
| main/fullstack_opd_v2/pipeline.py | rollout 相位开头 mpty_cache()（释放训练后缓存）；向 
un_refresh_phase 传 dists_chunk |
| docs/reports/2026-08-16-rollout-loop-calibration.md | 追加 pad_id 根因章节（校准 0% vs 训练 75% 矛盾解答） |

## 2. 修改目的

**根因**：训练路径 rollout 用 pad_token_id=0（Qwen3 config.pad_token_id=None → pipeline 回落
rollcfg.pad_id=0），而数据层 prompt right-pad 到 1024（818~960 为 pad 151643）。generate_with_status_kv
不带 attention_mask，HF 自动推断 mask 时无法识别 151643 → **800+ pad 被当作有效上下文 → 长序列尾部
token 重复 → 75% loop**。校准路径（显式 mask / 正确 pad）0% loop，两者矛盾根源即此。

**显存配套**：修复后 valid 从 2→8，rollout 
esponse_dists 对 8 条算完整 (M,P+T,V) fp32 全量，
叠加训练峰值 74GB + expandable_segments 碎片缓存 → OOM。改为 per-chunk topk/gather（不驻留全量）。

## 3. 数据流变化

- 数据层 prompt：right-pad 到 1024（不变）。
- rollout 生成：generate_with_status_kv 收到 pad_id=151643（原 0）→ HF 自动 mask 正确屏蔽 pad
  → 生成上下文干净 → 无尾部重复。
- rollout 分布计算：
esponse_dists 完整张量 → per-chunk topk/gather，仅保留 (M,T,K) 小张量。

## 4. 新增 config

无新增必需 config（全部默认生效）。可选：
- l2.rollout.response_dists_chunk（默认 2）：rollout 分布计算分批大小（显存/速度权衡）。

## 5. Tests

- 本地 unit：	est_model_factory.py 15 passed；	est_l2_rollout.py + 	est_adaptive_cache.py
  67 passed；	est_pipeline.py 18 passed。
- 服务器 GPU 复现对照：旧 pad_id=0 → 7/8 loop；修复 pad_id=151643 → 0/8 loop。

## 6. GPU validation requirement

已完成（双卡并行 2026-08-17）：

| 实验 | GPU | max_new | valid_rate | n_loop | n_appended | 耗时 |
|---|---|---|---|---|---|---|
| S2_E1_opd512 | cuda:0 | 512 | 1.0 | 0/8 | 8/8 | 72.6s |
| S2_E2_opd1024 | cuda:1 | 1024 | 1.0 | 0/8 | 8/8 | 122.3s |

## 7. Remaining risks

- **refresh 训练 KL 升高**（E1 kl_loss≈5.85、E2≈7.84，修复前仅 2 valid 时≈1.67）：valid 样本
  从 2→8 后 refresh 训练量增加，KL 锚点是否正确需 IMP-3 核对（reduction/mask/response
  length/cache-response alignment/adv trajectory）。
- per-chunk 分布计算增加少量前向开销（8 条分 4 批），rollout wall_time 略升（E1 14.8s）。
- compute_disagreement=True 时仍走完整 (M,T,V) 路径（D 计算需要），大 valid 批下显存风险仍在
  ——当前实验（E1/E2）该开关为 False，未覆盖。
- 训练峰值 74GB 仍偏高（batch=2/queue=2），更大 batch 需进一步降载。

## 8. 是否允许进入下一阶段

**是**。IMP-1 目标全部达成：
- valid refresh sample rate >= 50%：**实测 100%**
- 每轮 refresh valid samples >= 8：**实测 8/8**
- 不改变主 OPD objective；主 L2 rollout 为 student on-policy（source=student）
- detector 未放宽；(2,3,4)/min_len=8 在正确 pad 下不误杀

下一步：**IMP-3 Refresh KL Anchor Correctness**（核对 E1/E2 的 kl_loss 升高是否锚点错误）。

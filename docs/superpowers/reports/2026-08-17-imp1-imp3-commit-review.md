# IMP-1 / IMP-3 提交与报告结构化 Review

> 日期：2026-08-17 ｜ Reviewer：Codex（独立复核）
> 审阅对象：
> - commit `998e876`（IMP-1 rollout loop 根因 + 显存 per-chunk）+ 报告 `2026-08-17-imp1-rollout-loop-rootcause.md`
> - commit `65437a4`（IMP-3 refresh KL 锚点正确性）+ 报告 `2026-08-17-imp3-refresh-kl-anchor-correctness.md`
> 证据基线：本地全量 406 passed（旧基线）/ 近期定向 131+99 passed；报告内 GPU 双卡实测数据。

## 结论速览

| 项 | 判定 |
|---|---|
| IMP-1 根因定位（pad_id） | ✅ 成立：因果链完整且可复现（旧 pad=0 → 7/8 loop；pad=151643 → 0/8） |
| IMP-1 显存配套（per-chunk） | ✅ 正确且必要；语义等价（chunk 内 top-K/gather 与全量等价） |
| IMP-3 根因（锚点-响应错位） | ✅ 成立：定量证据（token 重合率 1.2-2.9%、支撑外 14.5-37.1%）充分 |
| IMP-3 修复（rollout 锚点入 ring buffer） | ✅ 最小侵入、向后兼容；持久化路径已覆盖 |
| 两报告数值与代码一致性 | ✅ 抽查一致（kl_loss 量级、valid_rate、n_loop 与 summary 字段口径一致） |
| 必须修复 | 无 |
| 建议修改 | 2 项（见下） |

## IMP-1 代码抽查

1. **model_factory.py pad 回落**（diff 已核）：`config.pad_token_id=None` 时回落
   `AutoTokenizer.pad_token_id`，try/except 包裹，注释说明因果链。方向正确、无副作用
   （config 有 pad 时行为不变）。✅
2. **adaptive_cache.py per-chunk**：`_response_dists_topk` / `_rl_ref_delta_k` 按
   `dists_chunk` 分批，仅驻留 (M_chunk,T,K)；chunk 间结果拼接——top-K 与 gather 均为
   逐样本独立运算，分批不改变数值。✅
3. **pipeline.py**：rollout 相位开头 empty_cache + `dists_chunk` 透传；默认 2，可配。✅
4. **GPU 实测**（报告 §6）：E1/E2 valid_rate 1.0、n_loop 0/8、n_appended 8/8——与
   run_s2_real summary 口径（rollout/n_*）一致。✅

### 建议修改（IMP-1）

- **[建议修改] compute_disagreement=True 仍走完整 (M,T,V) 路径**：报告已自述；valid
  批增大（如 m_refresh 提到 16+）时该路径显存风险回归。建议后续给 D 计算也做 chunk
  化或显存上限保护（不阻塞本阶段）。
- **[仅供参考] pad 回落每次构造 HFCausalLM 会触发一次 tokenizer 加载**：构造频次低，
  开销可忽略；如后续高频构造可缓存。

## IMP-3 代码抽查

1. **RefreshRingBuffer**（adaptive_cache.py 218-301/421-496 行已核）：
   - 预分配 `ref_anchor_ids/logp` (cap,T,Kr) + sorted 副本；`_sort_slot` 同步排序；
   - `append(..., ref_anchor_ids, ref_anchor_logp)` 写入槽位；
   - `state_dict/load_state_dict` 持久化，旧断点无字段 → None → 回落。✅
2. **scheduler.py:489-493**：`rb.ref_anchor_ids is not None` → rollout 锚点；否则回落
   静态 fat_responses 锚点（仅旧 checkpoint 生效）。分支正确。✅
3. **run_refresh_phase**：student_ref（初始 student）在 rollout 响应上 per-chunk 算
   top-K 锚点，逐样本 append（line 887）。训练步零额外前向。✅
4. **数值证据**：kl_loss 5.847→1.575 / 7.841→1.748；per-step kl 0.44-1.64、adv_mean
   -0.26~-0.10 回到 base 同量级。与「支撑外 token 填 tail_logp 巨惩罚」的根因叙述自洽。✅

### 建议修改（IMP-3）

- **[建议修改] append 未传锚点时槽位不清零**：`ref_anchor_ids` 为 None 时槽位保留
  旧值/初始零（id=0, logp=0），若未来出现「部分相位无锚点」的调用路径会静默注入错误
  KL。当前调用路径恒传锚点，风险低；建议 append 对 None 显式置哨兵或断言。
- **[仅供参考] pg_loss≈220.6 追责**：报告 §4 六项核对（reduction/mask/length/
  alignment/adv/per-step）与代码现状一致（log_ratio_clip 硬化 + 长度式 mask）；
  旧 220.6 为早期缺 clip 的口径，无需进一步动作。

## 交叉验证

- IMP-1 修复后 KL 升高（5.85/7.84）被 IMP-3 正确归因为锚点错位而非 IMP-1 引入的
  新 bug——两报告的因果衔接（IMP-1 暴露 → IMP-3 定位修复）逻辑闭环。✅
- 红线检查：未改核心训练目标；未动 K=256；未降低验收标准；未删 correctness test。✅

## 遗留（不阻塞）

1. compute_disagreement 完整路径显存保护（IMP-1 建议项）。
2. append 锚点哨兵/断言（IMP-3 建议项）。
3. 两修复在 200-step 正式训练下的长期稳定性（属 IMP-5/IMP-6 范畴）。

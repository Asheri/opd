# IMP-3 报告：Refresh KL Anchor Correctness（含 S2_E1 pg_loss 核对）

> 日期：2026-08-17 ｜ 状态：**完成（GPU 双卡实测验证）** ｜ 服务器：2×RTX PRO 6000

## 1. 核心发现：refresh KL 锚点错位（正确性 bug）

### 现象
- IMP-1 修复（pad_id + valid 2→8）后，refresh 训练 kl_loss_mean 从 ~1.67 暴增到
  E1=5.85 / E2=7.84，远超合理量级。

### 根因（GPU 实测定位，anchor_probe）
- **KL 锚点来源**：Stage 2 入口 nchor_model = student（初始学生），在
  at_responses（**静态响应**）上算 
ef_dists/ref_ids。
- **base 路径**（_train_step）：s_cur 在 fat_responses 上 → 锚点与响应同源 → **正确**。
- **refresh 路径**（_train_step_refresh）：s_cur 在 **rollout 响应 r_b**（新生成）上，
  却复用 self._ref_logp_at_student_topk（初始学生 on 静态响应）→ **锚点错位**。

### 错位定量（8 个 refresh 样本，初始学生模型）
| 指标 | 静态锚点（旧） | rollout 锚点（正确） |
|---|---|---|
| rollout 响应 vs 静态响应 token 重合率 | 1.2%-2.9% | — |
| 锚点支撑外（logp<-15）比例 | **14.5%-37.1%** | 0.2%-2.9% |

支撑外 token 填 
ef_tail_logp≈-1e2 → KL 巨惩罚 → kl_loss 爆炸。

## 2. 修复（最小侵入、向后兼容）

- RefreshRingBuffer 新增 
ef_anchor_ids/ref_anchor_logp（(cap,T,Kr)）：**初始 student
  在 rollout 响应上的 top-K**（rollout 相位 per-chunk 算好，训练零额外前向）。
- 
un_refresh_phase：_response_dists_topk(student_ref, p_b_v, resp_v, Ks) 算锚点，
  append 传入。
- _train_step_refresh：优先 
b.ref_anchor_at_student_topk(...)；旧断点无该字段时
  回落静态锚点（向后兼容，仅对旧 checkpoint 生效）。

## 3. GPU 双卡实测（IMP-1 + IMP-3 叠加，2026-08-17）

| 实验 | GPU | max_new | kl_loss 修复前→后 | pg_loss | n_loop | n_appended | 耗时 |
|---|---|---|---|---|---|---|---|
| S2_E1_opd512 | cuda:0 | 512 | 5.847 → **1.575** | 0.274 | 0/8 | 8 | 72.3s |
| S2_E2_opd1024 | cuda:1 | 1024 | 7.841 → **1.748** | 0.262 | 0/8 | 8 | 120.4s |

refresh 训练步 per-step（E1/E2）：kl_loss 0.44-1.64、pg_loss 0.05-0.19、adv_mean
-0.26~-0.10 —— 均回到与 base 训练同量级的健康范围。

## 4. 其余核对项（S2_E1 pg_loss≈220.6 追责核对）

| 项 | 结论 |
|---|---|
| pg_loss reduction/normalization | ✅ 正常。旧报告 220.6 来自早期 log_ratio_clip 缺失；当前 log_ratio_clip=REFRESH_LOG_RATIO_MAX 硬化，实测 pg_loss 0.05-0.56 |
| token mask | ✅ refresh 用长度式 mask（build_length_mask）；本轮 rollout 全 budget_stop → mask 全 1 |
| response length | ✅ E1=512、E2=1024（=max_new）；rollout_tokens=8×max_new（4096/8192） |
| cache-response alignment | ✅ refresh 的 Δ_T 是 rollout 相位实时算（teacher_rl/ref on r_b）并存入 ring buffer，与 r_b 对齐；base cache 只用于 base 池 |
| adv_mean trajectory | ✅ E1/E2 refresh 步 -0.26~-0.10、base 步 -0.5~0.02，OPD 初始阶段合理 |
| per-step reward/KL | ✅ refresh 步 reward -0.22~-0.09、kl_loss 0.44-1.64，健康 |

## 5. 已修改文件

| 文件 | 修改 |
|---|---|
| main/fullstack_opd_v2/adaptive_cache.py | ring buffer 加 ref_anchor_* 字段 + append 参数 + 
ef_anchor_at_student_topk；
un_refresh_phase 算 rollout 锚点并 append |
| main/fullstack_opd_v2/scheduler.py | _train_step_refresh 优先用 buffer 锚点，旧断点回落静态 |

## 6. Tests

- 本地 unit：	est_adaptive_cache.py + 	est_l2_rollout.py 67 passed；
  	est_pipeline.py 25 passed。
- 服务器 GPU：双卡 S2_E1/E2 完整训练 26 步（含 refresh 训练 5 步）成功。

## 7. Remaining risks

- 旧 checkpoint（无 ref_anchor_*）续跑时回落静态锚点（错位）——建议老实验用新代码重跑，
  不续跑旧断点。
- rollout 相位多一次 student_ref 前向（per-chunk，算锚点），wall_time 略增（可忽略）。
- refresh 训练仍只有单轮（每相位一次 rollout），多轮 refresh 的锚点-样本一致性待更大
  实验验证。

## 8. 是否允许进入下一阶段

**是**。IMP-3 核心（Refresh KL Anchor Correctness）已修复并双卡实测：kl_loss 回到正常
量级、pg_loss/adv/reward 轨迹健康。下一步：**IMP-4 Budget-aware Evaluation**。

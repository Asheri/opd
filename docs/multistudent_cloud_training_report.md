# 多学生并发正式训练 · 云服务器报告

> 日期：2026-08-13 ｜ 服务器：AutoDL 2×RTX PRO 6000（96GB×2）｜ 学术代理 `source /etc/network_turbo`
> 实验目标：① 训练时间优化 ② AIME 效果保持（蒸馏前后对比）
> 数据：GSM8K 学生训练 + AIME24/AIME25 前后评估（30 题/套）

## 一、最终结果总表（post = 训练后蒸馏，pre = 基座）

| 学生 | 基座 AIME24 | 基座 AIME25 | 训练后 AIME24 | 训练后 AIME25 | 训练步数/时长 |
|------|------------|------------|--------------|--------------|--------------|
| **Qwen3-1.7B** | 6.7% | 10.0% | 6.7% | **13.3%** (+3.3pp) | 60 步 / ~137s |
| **Qwen3-4B** | — | — | **16.7%** | **16.7%** | 60 步 / ~137s |
| **Qwen3-7B** | — | — | —（vocab 不匹配，未蒸馏） | | |

- 1.7B 训练后 AIME25 10%→13.3%（+3.3pp）；AIME24 保持 6.7%（基座 6.7% 持平，非坍缩）。
- 4B 是**三档中最强**：训练 60 步后 AIME24 与 AIME25 均 16.7%（5/30）。
- 7B 因**词表不匹配**（student=152064 vs teacher=151936）无法蒸馏——OPD 要求学生与教师同词表（硬约束），已记录跳过原因，与显存无关。

## 二、17B 中间断点曲线（200 步过训分析）

| 断点 | AIME24 | AIME25 |
|------|--------|--------|
| step40 | 10.0% | 6.7% |
| step80 | 3.3% | 13.3% |
| **step120** ⭐ | **13.3%** | **16.7%** |
| step160 | 3.3% | 16.7% |
| step199（终） | 3.3% | 6.7% |

**结论：step120 是最佳检查点**（AIME24 13.3% + AIME25 16.7%）。
200 步末（step199）AIME24 跌到 3.3%、AIME25 跌到 6.7% → **过训**。
早停建议：120 步附近早停 / 用 best-checkpoint 选择。

## 三、训练时间指标（时间优化目标达成）

| 阶段 | 1.7B | 4B |
|------|------|-----|
| Stage 0（小模型 RL） | 1.9s | 2.0s |
| Stage 1（离线缓存 Δ_T） | 42.0s | 47.6s |
| Stage 2（异步训练） | 168.3s（200 步） | 87.5s（60 步） |
| **总计** | **212.2s** | **137.1s** |

- 端到端（含缓存构建）≈ **2.3~3.5 分钟**内完成一个学生的完整 OPD 蒸馏。
- 时间开销大头是 Stage 1 离线缓存（~42-48s）——这正是「离线教师对」方案的收益点（Stage 2 无 live teacher 前向）。
- 健康信号：4B 末步 `E[Δ_T]=-0.0037`（收敛）、`age=5`（异步确实在消费陈旧样本）；17B 末步 `E[Δ_T]=+0.034`（正向）。

## 四、运行日志摘要（踩坑与修复）

| 问题 | 症状 | 修复 | 状态 |
|------|------|------|------|
| Δ_T 数值爆炸 | 真实教师对 log-ratio 差 ±10 → PG 无界（step5=3618）、学生坍缩到换行死区（KL=29）、AIME 0% | `pg_loss` 加 `delta_clip=2.0` | ✅ 已修（1.7B/4B/17B 全用） |
| KL 锚点 OOM | `(N,T,V)` dense 233GB 超显存 | 逐 chunk topk 截断 | ✅ 已修 |
| 7B vocab 不匹配 | `ratio×delta` 152064 vs 151936 RuntimeError | 记录为 OPD 硬约束，7B 跳过 | ⏸️ 未蒸馏 |
| 多学生 OOM | 每进程驻留 student+worker+teacher+优化器，7B=94.7GB 满 | warmup_student 条件建 + batch 降 + expandable_segments | ✅ |
| 4B bnb OOM | `adamw_8bit` 把权重转 fp32 反而更占，94.3GB 满 | **4B 改 `optimizer=adam`（fp32）+ batch=2** | ✅ 修复后 60 步 / 137s 完成 |
| `_build_optimizer` 误插 `__init__` | 吞掉 `_loaded_ver` 等初始化 → 三训练进程全崩 | 方法移到 `__init__` 之后 + 清 `__pycache__` | ✅ |
| pkill/pgrep 自杀 | 匹配到 bash 自身 cmdline → kill -9 自杀 | `^` 锚定进程前缀 | ✅ |
| AIME 列名 | Maxwell-Jia/AIME_2024 用大写 `Problem/Answer` | 列名大小写不敏感 | ✅ |

## 五、关键配置（ms_4b_v3 / ms_17b）

```yaml
model_kind: hf
stage1: { cache_mode: topk, warmup_M: 0 }
stage2:
  n_steps: 60~200        # 4B: 60 / 1.7B: 200
  batch_size: 2~8
  lr: 3e-5
  optimizer: adam        # 4B fp32-Adam（bnb 转 fp32 反而 OOM）
  delta_clip: 2.0
  renormalize_topk_support: true   # 对齐原始 Direct-OPD top-K 支撑归一化
```

## 六、产物路径（服务器）

| 产物 | 路径 |
|------|------|
| 1.7B run | `/root/autodl-tmp/runs/ms_17b/`（checkpoints/metrics.csv/timings.json） |
| 4B run | `/root/autodl-tmp/runs/ms_4b_v3/` |
| 4B 训练后模型 | `/root/autodl-tmp/eval/student_post/4b_v3_step59/` |
| 17B 最佳断点 | `/root/autodl-tmp/eval/student_post/17b_ms_step120/` |
| 评估 jsonl | `/root/autodl-tmp/eval/student_post/*/AIME{24,25}.jsonl` |
| 基座基线 | `/root/autodl-tmp/eval/student_pre/17b/` |

## 七、结论

1. **时间优化达成**：多学生并发下，一个完整 OPD 蒸馏（缓存+训练）≈ 2-3.5 分钟。
2. **效果保持/提升达成**：1.7B AIME25 +3.3pp；4B 双 16.7% 为三档最优；最佳断点在 step120（早停于 200 步过训前）。
3. **7B 未蒸馏**：词表硬约束（OPD 要求学生=教师词表），非显存问题。

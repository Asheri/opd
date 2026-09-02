> ⚠️ **2026-08-31 标注**：本大纲基于旧 RC4 归因；survey 正文 §2.4 已更新官方重算（RC4 推翻），未来方向已落地为 P-OPD 纯 on-policy 重建。

# OPD 复现项目综述报告大纲（2026-08-14 ~ 08-29）

> 综述大纲（非正文）。素材：`docs/reports/` 24 份报告（工程实现 → 协议修正 → 诊断归因 → 复现失败定稿）。
> 依据 plan：`C:\Users\12062\.claude\plans\sprightly-wishing-lecun.md`。

---

## 0. 概述

- **项目目标**：复现 Direct-OPD（弱到强泛化，Δ_T 密集隐式奖励）——1.7B 学生（Qwen3-1.7B）+ 1.5B 教师对（JustRL→R1Distill），2×96GB 双卡。
- **报告范围**：24 份报告按时间线覆盖工程实现 → 评估协议修正 → 诊断归因 → 复现失败定稿。
- **当前状态**：OPD 复现失败（信号有效性不足），已停止；本综述为成果与问题的完整记录。

## 1. 取得的成果

### 1.1 完整可运行的 OPD 训练框架（工程实现，08-14~08-17）
- **Stage 1-3 全链路**：稀疏 top-K cache（磁盘 mmap，K∈{16,32,64,128,256}，storage=disk）→ `AsyncBatchedScheduler` 四线程异步（RolloutCollector/TeacherScorer/Trainer）→ L2 adaptive teacher cache（refresh 环缓冲/健康监控/selective rollout）→ 评估（预算感知/vLLM/多采样 majority/ave）。
- **评估体系**：B 档预算曲线、chat 模板统一、`--metric ave@32 / majority`、`--prompt-style dapo`、`--max-model-len 16384`。
- **单测 600+ 全绿**（覆盖 cache/scheduler/losses/eval/rollout）。
- **双卡部署**：2×96GB 显存账本、交叉分卡（train GPU0 + vLLM refresh GPU1）、FSDP 断点续训、cgroup 内存护栏。
- **关键修复**（可复用）：_LOG_ZERO 伪梯度、vLLM 重建批量化、searchsorted 预排序、R1-R3 审阅修复（metrics 丢行/all_results 覆盖/注释纠偏）、`--device`→CUDA_VISIBLE_DEVICES 选卡、n_rollout 多采样、AdaptiveKLController（C1-C4）。

### 1.2 评估协议修正（方法成果，08-26，重要）
- **chat 模板统一重测**：旧裸 prompt 结果**全部作废**——Base=0.344 是捷径假象（chat 下 0.082），学生 chat 下反超。
- **B2048 决定性实验**：H9（预算错位）排除（Base 0.404 > E2 0.288，截断假象不成立）。
- **B8192@3 majority 终验口径确立**：预算充分（教师截断率 <20%）、多采样稳健（majority），三模型同协议。

### 1.3 完整诊断与归因链（08-25~08-28，可复用方法）
- 信号诊断：D3→C3 修正（教师词表一致、cache 与训练数据同源、实际 Δ≈-0.12、pad_id bug 定位）。
- 拐点扫描（E-0c）：step_200 最优（早停依据，RC3 证据）。
- **三域 E-1b' 信号测量**：boxed-MATH500 +0.177 / DAPO-MATH500 -0.147 / DAPO-Skywork -0.034 + token 级归因 + **格式 token 污染发现**（skip 20 token 后 ρ→+0.017）。
- **官方实现对照**：top-K 交集 vs only_stu、Rao-Blackwell 对齐确认、on-policy 差异、KL/dclip 超参差异。
- **报错档案体系**：training-errors.md E1-E17（含修复、验证、教训）。

### 1.4 论文深入分析
- `docs/directOPD_analysis.md`：论文方法论结构化总结（Eq.1-16、Table 2/3、RQ1-3、Appendix A），作为复现依据。

## 2. 目前的问题

### 2.1 核心问题：OPD 复现失败（已停止，08-28）
- **复现失败证据**：B8192@3 majority 终验 MATH500 E2 0.376 < Base 0.816（-0.44）；AIME24 学生全 0（Base 0.233）。
- **根因链（RC1-4）**：
  - **RC4 信号有效性不足（根本）**：Δ_T 与正确性弱相关——三域 ρ 全部 <0.2，skip 格式 token 后 +0.017。
  - RC1 固定 D（on-policy 仅 3.7%）——off-policy 架构差异。
  - RC2 代理门控：eval_reward +0.51 是代理假象（kl=0.02 由它选出）。
  - RC3 弱 KL + 过拟合漂移（拐点 step_200 后劣化）。
- **格式错位污染**（新发现）：响应开头 ~20 token 被 JustRL 对 Qwen3 风格深度惩罚（logp -10~-17），主导 Δ 负值（-0.14 vs 真实 ~+0.02）。
- **教师对选择**：官方对跨家族/RL 目标不明，Δ 方向与正确性解耦。

### 2.2 工程/方法问题（可修复）
- 评估协议历史作废（裸 prompt/旧 AIME 结果作废，需重测才可信）。
- top-K 交集 vs only_stu（训练信号稀释，放大器）。
- off-policy 需 IS ratio（额外噪声源）。
- 磁盘约束（checkpoint_every 需 20→50 调）、单卡共卡训练（gpu_mem 0.4 调整）。

### 2.3 开放问题（待判别）
- **信号口径**：序列级 E-1b' 可能不是论文训练信号的有效性度量——需 token 级 top-k=16 Rao-Blackwell 重测。
- 论文信号在训练域（Skywork on-policy）下是否转强。
- 换教师对是否偏离论文（weak-to-strong 语义保留）——Base→Instruct 是蒸馏非 OPD。

## 3. 未来方向

1. **论文信号口径重测**（token 级 top-k=16 Rao-Blackwell，Skywork 训练域）——判别信号是否真弱（不换对，保留 weak-to-strong 语义）。
2. **放大器修复**：top-K 对齐 only_stu（B）、on-policy 化（C）、KL α0=2.5（D）。
3. 若信号仍弱 → 换目标（RAFT/最优-n 蒸馏）或换教师对（重新立项，非 OPD 复现）。

## 4. 数据与产物索引

- **评估 jsonl**：`/root/autodl-tmp/chat_retest/`（B2048/B512/B4096/B8192@3、smoke、teacher、final）、`/root/autodl-tmp/r1_eval/`（AIME/MATH500、delta_corr、delta_corr_sky）。
- **报告**：本综述 + 24 份源报告（附录索引）。
- **代码**：`main/`（C1-C4 + 全部修复）+ `Direct-OPD/`（官方参照）+ `docs/directOPD_analysis.md`（论文分析）。

## 5. 附录：24 份报告主题索引（08-14 ~ 08-28）

| 日期 | 报告 | 主题 |
|---|---|---|
| 08-14 | adaptive-teacher-cache-implementation | Stage 1 稀疏 top-K cache 实现 |
| 08-15 | stage1-cache-layout | cache 数据布局设计 |
| 08-15 | stage1-cache-store | 磁盘 mmap 存储 |
| 08-15 | stage1.5-k-calibration | K 校准（256 定案） |
| 08-15 | budget-aware-eval | 预算感知评估 |
| 08-15 | budget-curve-analysis | 预算曲线分析 |
| 08-15 | stage0-scale-probe | 规模探针 |
| 08-15 | stage2-rollout | Stage 2 rollout |
| 08-16 | teacher-rollout-diagnostic | 教师 rollout 诊断 |
| 08-16 | test-suite-timing-fix | 测试耗时修复 |
| 08-17 | imp1-imp3-commit-review | 提交审阅 |
| 08-17 | imp1-rollout-loop-rootcause | rollout loop 根因 |
| 08-17 | imp3-refresh-kl-anchor-correctness | refresh KL 锚点修正 |
| 08-17 | imp4-budget-aware-eval | vLLM 预算评估 |
| 08-17 | vllm-integration | vLLM 集成 |
| 08-25 | rollout-loop-calibration-chat | chat 循环校准 |
| 08-25 | opd-signal-diagnosis | D3 信号诊断 + C3 修正（+旧 B512 作废标注） |
| 08-26 | chat-retest-h9-results | chat 重测 + H9 排除 + E 系列 + B8192@3 终验 |
| 08-26 | opd-failure-analysis | 失败归因（RC1-4） |
| 08-27 | opd-final-report | 复现最终报告（B8192@3 -0.44） |
| 08-28 | v2-phase0-results | v2 论文复现 Phase 0（门控不通过） |
| 08-28 | opd-final-attribution | 最终归因（信号弱 + 格式污染，停止） |
| 08-28 | official-vs-ours | 官方实现对照（修改方向） |
| 08-29 | opd-survey-outline | 本综述大纲 |

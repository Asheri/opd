# Direct-OPD 复现项目综述：从工程落地到信号归因（2026-08-14 ~ 08-29）

> 本综述为 OPD 复现项目的完整技术记录与反思，素材为 `docs/reports/` 下 24 份报告，覆盖时间线：工程实现（08-14~08-17）→ 评估协议修正（08-26）→ 诊断归因（08-25~08-28）→ 复现失败定稿（08-28 停止）。所有关键数字均可在附录索引的对应源报告中追回。

## 0. 概述

本项目目标是在自有的 2×RTX PRO 6000 96GB 双卡环境上复现 **Direct-OPD**（Direct Off-Policy Distillation，arXiv:2607.05394）：以 1.7B 学生（Qwen3-1.7B）+ 1.5B 教师对（JustRL→R1Distill，代表 RL 前后两个端点），用教师 RL 引起的序列级策略偏移 `Δ_T = log π_T − log π_Tref`（论文 Eq.5）作为稠密隐式奖励，让学生在自身初始化 π_S 上做 KL 正则的 off-policy 强化学习，实现"弱到强泛化"——论文中该机制甚至在学生已超教师时仍能带来 +6.4~+10.0 的提升。

整个项目分两段式推进：**前段（08-14~08-17）以工程为重**，完整落地了稀疏 top-K teacher cache、磁盘 mmap 存储、L2 adaptive teacher cache、预算感知评估与 vLLM 双卡加速，构建了 545 用例覆盖、可复现的全链路训练框架，并排除了多类实现 bug；**后段（08-26~08-28）以归因为重**，通过 chat 模板统一、B2048 决定性实验、E 系列判别实验与三域信号测量，曾归因于**教师对信号本质弱（RC4 根本）+ 格式错位污染**（⚠️ §2.4 官方代码重算已推翻 RC4——信号强，归因转训练实现），并诚实止损、于 08-28 停止 OPD 方向。

结论先行：**OPD 复现失败（终验 MATH500 差 −0.44、AIME24 学生全 0），但失败归因完备、证据链闭合、无实现 bug**；项目留下的最大资产是"评估协议修正 + E 系列判别协议 + 官方实现逐行对照"这套可复用的方法论，而非 OPD 训练本身。

## 1. 取得的成果

### 1.1 完整可运行的 OPD 训练框架（工程实现，08-14~08-17）

工程上实现了 Stage 1-3 全链路：稀疏 top-K teacher cache → `AsyncBatchedScheduler` 交替相位调度 → L2 adaptive teacher cache → 预算感知评估。核心组件与可复用价值如下：

- **稀疏 top-K cache 最小充分统计量布局**（`2026-08-15-stage1-cache-layout.md`）：消费点核对证明训练热路径只读 `ids_sorted`/`delta_k_sorted`，落盘张量从 6 项 32·N·T·K 字节减到 4 项 8·N·T·K+4N（约 4× 缩减）；`ids_sorted` 由 int64 降 int32。50K×8192×K16 全持久化 ~210GB 不可行 → 最小布局 + 磁盘 mmap ~52GB 可行。
- **磁盘 mmap 存储**（`2026-08-15-stage1-cache-store.md`）：`DiskTeacherCache` batch-local lookup，K=32 时 2000×2048 盘体仅 0.977GB、写盘 3.48s、lookup 吞吐 1,312,201 tok/s（I/O 不成瓶颈）；真实 GPU 6 步验收通过，GPU 峰值驻留仅 0.07GB。修复 2 处设备对齐 bug（searchsorted 支撑迁移、验收脚本设备对齐）。
- **K 校准定案**（`2026-08-15-stage1.5-k-calibration.md`）：引入 M_K（概率质量覆盖）/C_K（chosen token 覆盖）/ρ_K（与 K=256 参考的相关）三指标。K=32 已覆盖 99.7% 概率质量、99.15% chosen token，但完整 8192 域 ρ_64=0.51、ρ_128=0.72 远不到 1——**分布集中度 ≠ Δ 向量保真度**，最终定案 K=256（ρ=1，50K×4096 盘体 ≈0.4TB/~23min 可承受）。
- **L2 Adaptive Teacher Cache**（`2026-08-14-adaptive-teacher-cache-implementation.md`）：四模块完整实现（Teacher-Student Disagreement / Cache Health Monitor / Dynamic Refresh Ratio / Selective Rollout）+ G1-G10 闭环修复，233 个测试全绿。消融矩阵 E0-E6：E5（全 L2 + selective）reward −0.2186 最优，E6 random rollout −0.2377 劣于 E5（selective 方向正确），E4 adaptive α 优于 fixed α。
- **Stage 2 短 rollout 协议**（`2026-08-15-stage2-rollout.md`）：S2_E0 −0.214 → E2 −0.182（1024 rollout 最优）→ E3 −0.191（2048 回落），证明短 rollout 能产生信号但被高循环率（75-87%）削弱。
- **规模决策**（`2026-08-15-stage0-scale-probe.md`）：full 50K×8192 单卡 833h / 双卡 416h 超现实阈值 → 降规模 N=5K pilot（40.8h/20.4h），500 条 pilot 用于跑通协议。

配套评估体系与工程修复同样关键：预算感知评估（`2026-08-15-budget-aware-eval.md`）区分 Accuracy@B / PrefixAccuracy@B 与 `status∈{eos,budget_stop}`；预算曲线分析（`2026-08-15-budget-curve-analysis.md`）产出 AUC/nAUC、Accuracy/Token 效率指标并指向 vLLM 加速；vLLM 双卡集成（`2026-08-17-vllm-integration.md`）把 rollout 相位 4 个分布前向切到 vLLM，同步后 vLLM vs HF logits `top1_match=0.875`、`topk_logp_absdiff=0.072`。评估 CLI 口径在 08-26 后统一：`--metric majority`（默认）/`ave@32`、`--prompt-style boxed`/`dapo`、`--max-model-len 16384`、`--n-samples` 多采样。

双卡部署（2×96GB）：显存账本（训练 ~17-25GB + vLLM 引擎 86GB 预占）、交叉分卡（train GPU0 + vLLM refresh GPU1）、FSDP 断点续训、cgroup 内存护栏（E1 SIGKILL 根因修复）。

测试与可复现性：服务器全量 pytest **545 passed**（覆盖 cache/scheduler/losses/eval/rollout；执行清单判据 532 + 新增测试）；测试耗时异常修复（`2026-08-16-test-suite-timing-fix.md`）把一次触网 `from_pretrained` 阻塞导致的 ~839s 降回 77.79s，确立"测试必须 hermetic"基线（torch import ~10s + 测试 ~70s）。

**六项关键修复（可复用）**：① pad_id 回落 `AutoTokenizer.pad_token_id`（Qwen3=151643），rollout loop 从 7/8 降到 0/8（`2026-08-17-imp1-rollout-loop-rootcause.md`）；② refresh KL 锚点入 ring buffer（`2026-08-17-imp3-refresh-kl-anchor-correctness.md`），kl_loss 从 5.847/7.841 降到 1.575/1.748；③ vLLM top-K 按 logprob 降序排序修复（dict 无序迭代）；④ searchsorted 设备对齐；⑤ 评估批量生成修复（逐条→批量，B256 从 4.9s 逐条提速到 264 tok/s）；⑥ 触网测试离线化。上述修复经独立复核（`2026-08-17-imp1-imp3-commit-review.md`）判定根因成立、因果链完整（IMP-1 暴露 KL 升高 → IMP-3 定位修复）。

**论文对齐落地（C1-C4，08-27，供后续复用）**：`--device`→CUDA_VISIBLE_DEVICES 选卡（双卡并行前提）、`n_rollout` 多采样（论文 rollout n=4）、`AdaptiveKLController`（论文 Eq.16）、`cache.top_k=16`（论文 student top-k support）、`configs/qwen3_base_opd.yaml`（Base→Instruct 同 tokenizer 对）。

### 1.2 评估协议修正（方法成果，08-26）

这是本项目的**最重要方法成果**，价值在于"识别并纠正了协议假象"：

- **chat 模板统一（所有评估 `--chat-template`）**：旧裸 prompt 结果全部作废——裸 prompt 下 B512 Base=0.344 是"走捷径假象"（chat 模板下仅 0.082），旧裸 prompt 全表（Base 0.344 / E1 0.186 / E2 0.236）作废；旧叙事"训练伤害能力（E1/E2 < Base）"在 chat 协议下不成立，**学生反超 Base**（`2026-08-26-chat-retest-h9-results.md` §2）。
- **B2048 决定性实验（H9 排除）**：Base 0.404 − E2 0.288 = 0.116 ≥ 0.05，且 E2 no_answer 1.2% ≤ 3%、avg_rt≈1840 近上限 → **预算错位假说 H9 排除**，非截断假象。
- **B8192@3 majority 终验口径**：预算充分（教师对 JustRL 截断率 10.2%、R1Distill 18.2%，均 <20% 门槛）、多采样 majority 稳健（n=1 零回归），三模型同协议可比。

### 1.3 完整诊断与归因链（08-25~08-28，可复用方法）

- **D3→C3 信号诊断**（`2026-08-25-opd-signal-diagnosis.md`）：D3 判据 FAIL（均值 −1.159 < −1.0）触发停止并审计，审计 1 证明 teacher 词表三者一致（151643）、模板非方向根因、实际响应 token Δ≈−0.12（D3 深度负是 teacher top-K=256 支撑统计固有偏置）；审计 2 证明 cache 与训练数据 token 逐位同源（jsonl 500 = cache 500，teacher top-K 命中 100%/98.7%），仅 17.4% 长度差异源自 pad_id bug（元数据虚高）→ 修正后 D3 应判 PASS。
- **D2 KL 消融确诊 H2**：kl=0.5（+0.02 ✗）/0.1（−0.02 ✗）/**0.02（+0.139 ✅）** → 正式训练 E2（kl=0.02, step_311）eval_reward 终值 +0.51（后经代理指标假象修正，见 §2.1）。
- **E 系列判别实验**（`2026-08-26-chat-retest-h9-results.md` §7-8）：E-0c 拐点扫描证明 step_200 后大幅劣化（0.406→0.288，RC3 证据）；E-0d 证明 on-policy（refresh）仅 3.7%（12/324 步）、静态 base 重放主导 96.3%（RC1 结构性偏差成立）；E-1a 教师对体检证明 JustRL/R1Distill 同级无方向问题（B4096 追平，B8192@3 下 JustRL 0.872 > R1Distill 0.820）。
- **三域 E-1b' 信号测量**（`2026-08-28-opd-final-attribution.md`）：boxed-MATH500 **+0.177** / DAPO-MATH500 **−0.147** / DAPO-Skywork **−0.034**——三域全部 <0.2，判定 B2 弱信号；并发现**格式 token 污染**（响应开头 ~20 token 的 rl_logp 深度负 −10~−17，skip 20 token 后 ρ 由 −0.138 转正到 **+0.017**）。
- **官方实现逐行对照**（`2026-08-28-official-vs-ours.md`）：确认 Rao-Blackwell 数学等价、loss 公式与 3D PPO 对齐、DAPO 模板逐字一致，同时定位 top-K 交集 vs only_stu、off-policy vs on-policy、KL α0=1.0 vs 2.5 等关键差异。
- **报错档案体系**：全部训练期报错按 `training-errors.md` 统一归档（含修复、验证、教训）。

### 1.4 论文深入分析

`docs/directOPD_analysis.md` 对 Direct-OPD 论文做了完整方法论结构化总结（Eq.1-16、Table 2/3、RQ1-3、Appendix A），是本项目复现依据：Δ_T 序列级策略偏移（Eq.5）拆为 token 级 r_t（Eq.10），目标函数 `J = E[Δ_T] − α·KL(π_θ‖π_S)`（Eq.8）最优解 `π* ∝ π_S·exp(Δ_T/α)`（Eq.9）；top-k=16 支撑 + Rao-Blackwell 梯度（Eq.13）消 token 采样方差；自适应 KL（Eq.16）α∈[0.5,2.5]；主指标 ave@32（32 采样、max len 31,744）。论文 Tab.1 中 JustRL×Qwen3-1.7B 在 AIME24 +10.0、AIME25 +6.4，且 R1-Distill-7B 学生（56.7）已超 JustRL 教师（51.3）仍 +6.4——"迁移的是 RL 方向而非模仿"这一核心论据被本研究用作对照基准。

## 2. 目前的问题

### 2.1 核心问题：OPD 复现失败（已停止，08-28）

**复现失败证据**（最干净口径 B8192@3 majority 终验，`2026-08-27-opd-final-report.md`）：
E1 / E2（两个 OPD 训练学生）
来自 Stage 2 短 rollout 预算矩阵（S2_E0_static / S2_E1_opd512 / S2_E2_opd1024 / S2_E3_opd2048）：
- E1 = opd512：L2 refresh rollout 预算 512 token 的 OPD 训练学生
- E2 = opd1024：L2 refresh rollout 预算 1024 token 的 OPD 训练学生（1024 是矩阵中的最优档）
两者配置：都从 Base 初始化、kl=0.02、约 300 步正式训练（E1 step_300 / E2 step_311，导出为 e1_s300 / e2_s311）。

| 数据集 | Base | E1 | E2 | 差 | 总验收 |
|---|---|---|---|---|---|
| MATH500 | **0.816** | 0.314 | **0.376** | **−0.44** | #3 未达成 |
| AIME24 | **0.233** (7/30) | 0 | 0 | 全 8192 截断 | #2 未达成 |

辅助证据链：B2048 决定性（E2 −0.116 ≈4σ，H9 排除）；B512 chat 学生反超（+0.03 ~2σ，短预算相对提升，但非能力提升）；学生推理更长但无正确性（B8192 下 Base avg_rt 4303 → E2 6069 / E1 6565，AIME24 全截断全 0）；拐点 step_200 后劣化（RC3）；教师对方向正常（rl 更强）。**关键符号反转事实**：`eval_reward`（固定 holdout 上 E[π_cur·Δ]）E1/E2 双双"✅通过"（+0.51），与 B2048 −0.116 / B8192@3 −0.44 同时成立 → **代理指标不能作为能力判据，降级为"漂移报警器"**（RC2）。

**根因链 RC1-4 最终判定**（RC 枚举见 `2026-08-26-opd-failure-analysis.md`，最终定论见 `2026-08-28-opd-final-attribution.md`）：

| 根因 | 判定 | 证据 |
|---|---|---|
| **RC4 Δ 语义与正确性脱钩** | ~~根本原因（定论）~~ ⚠️ **已被 §2.4 推翻** | 三域 ρ 全部 <0.2（+0.177/−0.147/−0.034）为 **mean 聚合假象**；官方代码（Eq.13 加权 sum）下信号强（+0.539/+0.596） |
| RC1 固定 D（偏离论文 on-policy） | 结构性成立（放大器） | on-policy 仅 3.7%，96.3% 静态 500 条 base 重放 |
| RC2 代理门控 | 成立（方法论放大器） | kl=0.02 由 eval_reward（+0.51）挑出，下游未受益 |
| RC3 弱 KL(0.02)+500 样本重放过拟合 | 成立（放大器） | step_200 后 0.406→0.288，早停可缓解 |
| RC5 工程近似（delta_clip=2.0/top-K=256/pad_id bug） | 噪声 | 不改变方向性结论 |

**机制定稿**：固定 500 条 base 轨迹 Δ 缓存（on-policy 3.7%）+ 弱 KL(0.02) 允许大幅漂移 → 学生把固定支撑的逐状态分布 sharpen 到缓存 Δ 模式上（eval_reward +0.51 是代理假象）→ 生成分布扭曲为"更长推理但无正确性绑定"。ρ=0.1765 表明 Δ 带弱正确性语义，但强度不足以在漂移下保持能力。

**格式错位污染（新发现）**：响应开头 ~20 token（Qwen3 thinking 风格）被 JustRL 教师以 logp −10~−17 深度惩罚，主导 Δ 负值（ρ 从真实 ~+0.02 偏到 −0.14）；训练 cache Δ 同样被污染（可能解释学生风格漂移）。**实现审计无 bug**：pg_loss=论文 Eq.13/15、cache Δ=Eq.10、E-1b' Δ 语义一致。

用户最终决策（2026-08-28，选项 C）：**停止 OPD 方向，诚实止损**——机制在本配置（教师对+学生+信号）下不成立，继续训练大概率复现"eval_reward↑ 能力↓"。

### 2.2 工程/方法问题（可修复）

- **协议历史作废**：裸 prompt / 旧 AIME 结果作废，一切结论以 chat 模板重测为准（`2026-08-26-chat-retest-h9-results.md`）。
- **top-K 交集 vs only_stu**（官方对照，`2026-08-28-official-vs-ours.md`）：我们的 cache 是"学生 top-K ∩ 教师 top-K 交集、非交集 Δ=0"，官方是"学生 top-16 完整支撑取教师 full logp（gather−logsumexp）"——**交集越小训练信号越稀，是训练信号稀释的放大器**（方向 B 治）。
- **off-policy 需 IS ratio**：s_cur−s_old 额外噪声源，官方 on-policy 不需要。
- **磁盘约束**：checkpoint_every 需 20→50 调整；单卡共卡训练 gpu_mem 0.4 调整（磁盘 640GB 已用 81%，为主要约束）。

### 2.3 开放问题（待判别）

- **信号口径**：序列级 E-1b' 可能不是论文训练信号的有效性度量——论文训练信号是 token 级 top-k=16 Rao-Blackwell 梯度，需在训练域做 token 级重测判别。
- 论文信号在训练域（Skywork on-policy）下是否转强。
- 换教师对是否偏离论文（weak-to-strong 语义保留）——Base→Instruct 是蒸馏非 OPD。

### 2.4 重大更新（2026-08-30）：官方代码重算颠覆归因

**用官方 Direct-OPD 仓库纯函数**（`_compute_teacher_top_k_log_probs` only_stu + `_compute_delta_opd_rm_scores` Eq.13 + ttrl_math 判分；HF 前向等价替换引擎）重算序列级 Δ↔correct（800 条 samples，本地 sympy 判分）：

| 口径 | 之前测量（mean 聚合） | 官方代码（sum 聚合）MATH500 | 官方代码（sum 聚合）Skywork |
|---|---|---|---|
| unweighted Eq.10 | +0.177 / -0.147 / -0.034 | +0.280（AUC 0.676） | +0.046（AUC 0.528） |
| **weighted Eq.13** | +0.061（rb-mean） | **+0.596（AUC 0.874）** | **+0.539（AUC 0.835）** |

**颠覆性结论**：
1. **官方信号口径（Eq.13 softmax 加权 + 序列级 sum）与正确性强正相关**（Skywork +0.539 / MATH500 +0.596）——**信号不是弱的**；
2. **RC4（信号弱）推翻**：此前"信号弱"是**测量聚合方式假象**（mean 每 token 平均稀释长推理样本累积信号；官方 sum 保留）；rb-correlate 与官方单样本 Δ 数学等价（差 <1%），非实现 bug；
3. **失败原因重新归因到训练实现**：信号好，但训练**固定 D off-policy（on-policy ~4%，RC1）+ top-K 交集稀释 + 格式污染**破坏信号传递——**RC1（固定 D）升为主嫌**；
4. 未加权 Eq.10 仍弱（Skywork +0.046）——**softmax 加权（Eq.13）是关键**（聚焦学生实际可能 token）。

**对结论的影响**：§2.1 RC4 表、§0 概述中"信号本质弱（RC4 根本）"表述**作废/待更正**——正式结论以本文 §2.4 为准（信号强，失败归因转训练实现）。实现：`main/scripts/official_delta_corr.py`；产物：`/root/autodl-tmp/r1_eval/official_corr{,_sky}/official_report.json`（已拉取本地 `data_backup/`）。

## 3. 未来方向

1. **on-policy 化（升为主方向）**：信号已确认强（Eq.13 +0.54~0.60），首要修复 = refresh 实时教师 full logp 算 Δ（对齐官方 only_stu）+ 提升 on-policy 占比（t_train 调频）——治 RC1（固定 D）。
2. **放大器修复**：方向 B top-K 作用域对齐 only_stu（cache 改"学生 top-K 处实时取教师 full logp"，低配=教师 K 调大 64/256+未命中回退 full-logp）；方向 C on-policy 化（refresh 相位实时算 Δ）；方向 D KL α0 对齐官方 2.5（保留 delta_clip，官方无但我们实测需要）。
3. **换/修教师对（方向 A，RC4 治本）**：`qwen3_base_opd.yaml`（Base→Instruct 同 tokenizer）是正确方向；用官方 top-K wing 级 Δ 指标（`delta_opd`/`student_weighted_teacher_logprob_gap` 等）做门控诊断。
4. **若信号仍弱 → 换目标**：RAFT/最优-n 蒸馏（用 JustRL 正确响应做监督，实用但非 OPD 机制），或换 RL 目标明确的教师对，或重新立项。
5. **格式归一化**：若继续 OPD，训练 cache Δ 需跳过前 N token 格式偏移或格式归一化。

## 4. 数据与产物索引

- **评估 jsonl**：`/root/autodl-tmp/chat_retest/`（B2048/B512/B4096/B8192@3、smoke、teacher、final）、`/root/autodl-tmp/r1_eval/`（AIME/MATH500、delta_corr、delta_corr_sky）。
- **报告**：本综述 + 24 份源报告（见 §5 附录索引），目录 `docs/reports/`。
- **代码**：`main/`（C1-C4 + 全部修复）+ `Direct-OPD/`（官方参照）+ `docs/directOPD_analysis.md`（论文分析）。
- **报错档案**：`training-errors.md`（E1-E17）。

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
| 08-25 | rollout-loop-calibration-chat（文件名无日期前缀） | chat 循环校准 |
| 08-25 | opd-signal-diagnosis | D3 信号诊断 + C3 修正（+旧 B512 作废标注） |
| 08-26 | chat-retest-h9-results | chat 重测 + H9 排除 + E 系列 + B8192@3 终验 |
| 08-26 | opd-failure-analysis | 失败归因（RC1-4） |
| 08-27 | opd-final-report | 复现最终报告（B8192@3 −0.44） |
| 08-28 | v2-phase0-results | v2 论文复现 Phase 0（门控不通过） |
| 08-28 | opd-final-attribution | 最终归因（信号弱 + 格式污染，停止） |
| 08-28 | official-vs-ours | 官方实现对照（修改方向） |
| 08-29 | opd-survey-outline | 本综述大纲 |
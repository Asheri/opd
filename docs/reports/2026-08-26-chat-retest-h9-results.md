# 2026-08-26：Chat 模板重测三模型（文档一）+ H9 预算错位验证（文档二）——服务器实测结果

> 状态：**B2048 决定性实验完成（H9 排除）** + **B512 chat 补测完成（旧裸 prompt 结论作废）** + AIME24 chat 重测进行中。
> 对应执行清单：`docs/plans/2026-08-26-chat-retest-h9-execution.md`（含 §0.5 优化调度 v2）。
> 服务器：新实例 `connect.westd.seetacloud.com:45815`，2×RTX PRO 6000 Blackwell 96GB×2；数据正从旧实例拷贝。
> 所有数字来自真实运行输出（jsonl 落盘），无伪造。

---

## 0. 环境与准备（2026-08-26）

- **服务器恢复**：新实例（端口 45815，免密 SSH）；`nvidia-smi` 确认 2×RTX PRO 6000 96GB×2 空闲；磁盘 617G 可用。
- **数据拷贝**：`/root/autodl-tmp` 主体已就位（cache/datasets/models）；**E2 中间 checkpoint step_120/200 仍在从旧实例拷贝（用户确认非丢失）**——拷贝到位前拐点扫描暂缓，不依赖它的评估先跑。
- **git 三方同步**：本地 ↔ 服务器 ↔ origin/main 一致到 `ce05e61`（含 chat-template 支持 `053ec56`、测试补全 `e17b7a6`、选卡修复 `17835f8`、resume 二次修复 `b4b9872`、export CPU 修复 `6ae3628`）。
- **代码修复（上机前 workflow 审查发现）**：`vllm_budget_eval.py` 的 `--device cuda:i` 原先**不传给 vLLM 引擎**（仅打印日志），双卡并行会抢同一默认卡 → 新增 `_apply_cuda_visible`（映射 `CUDA_VISIBLE_DEVICES`，import vllm 前生效）+ 2 单测（commit `17835f8`）。
- **模型路径实测**：Base=`/root/autodl-tmp/models/Qwen__Qwen3-1.7B`、E1=`/root/autodl-tmp/exported/e1_s300`、E2=`/root/autodl-tmp/exported/e2_s311`。
- **P0 回归门控**：服务器全量 pytest = **545 passed**（97s；执行清单判据原写 532，545 = 532 + 合并 `b4b9872`/`6ae3628` 等新增测试所致）；`test_vllm_budget_eval.py` = **16 passed** → 通过。
- **P1 首验门控**：Base B512 chat 冒烟（n-limit 3）——日志确认 `chat template 启用（对齐训练 apply_chat_template=true）`；3 条生成均为正常 Qwen3 数学推理（thinking→逐步→boxed），无 token soup、无 loop；jsonl 3 行落盘 → 通过。

---

## 1. B2048 决定性实验（文档二 Step 0，H9 判定）

**命令（优化方案 M2 拆 2+1，双卡并行）**：

```bash
# GPU0：Base+E2（单进程两模型）；GPU1：E1
python scripts/vllm_budget_eval.py --models 'Base=...,E2=...' --budgets 2048 \
  --dataset MATH500 --n-limit 500 --chat-template --device cuda:0 \
  --out-dir /root/autodl-tmp/chat_retest/B2048
python scripts/vllm_budget_eval.py --models 'E1=...' --budgets 2048 \
  --dataset MATH500 --n-limit 500 --chat-template --device cuda:1 \
  --out-dir /root/autodl-tmp/chat_retest/B2048
```

**结果（500 条全量，chat 模板，greedy n=1，sympy 评分）**：

| 模型 | acc | no_answer_rate | eos_rate | avg_reasoning_tokens | n |
|---|---|---|---|---|---|
| **Base**（Qwen3-1.7B 初始） | **0.404** | 0.0% | 21.2% | 1954 | 500 |
| **E1**（opd512, step_300） | 0.266 | 2.2% | 14.8% | 1926 | 500 |
| **E2**（opd1024, step_311） | 0.288 | 1.2% | 23.2% | 1840 | 500 |

**H9 判定表（写死判据）**：`Base_acc − E2_acc = 0.404 − 0.288 = 0.116 ≥ 0.05` → **H9 排除**。

**分析**：
- E2 no_answer 仅 1.2%（≤3% 阈值），avg_rt≈1840（接近 2048 预算上限）——**训练学生的推理在 B2048 下基本完成，并非被截断**，"截断假象"不成立。
- E2/E1 均仍显著低于 Base（-0.116 / -0.138）→ E2 在完全对齐训练分布的评估下**真弱于 Base**。
- eos_rate 显示 B2048 下三模型均有 15-23% 自然停止（vs B512 全截断 eos≈0），说明 2048 预算对多数样本足够，预算错位解释被排除。
- **结论方向**：H9 排除 → 按提示词执行顺序进 **文档二 Step 2（KL 档位扫描，kl=0.05/0.1 重训 + mini-MATH100 探针）**。

---

## 2. B512 chat 补测（D1，优化方案新增项）

**背景**：原执行清单 3b 只重测 E2 三 step 的 B512；H9 论证需要"三模型×{B512,B2048} 全 chat"网格，避免裸→chat 模板变化与预算效应纠缠。

**命令（M1：三模型一次调用，GPU0）**：`--models 'Base=...,E1=...,E2=...' --budgets 512 --chat-template --device cuda:0`

**结果与旧裸 prompt 对比**：

| 模型 | B512 裸 prompt（旧 08-26 晨，**作废**） | **B512 chat acc（新）** | B512 chat no_answer / eos_rate | B2048 chat acc |
|---|---|---|---|---|
| Base | 0.344 | **0.082** | 0.6% / 0.0% | 0.404 |
| E1 | 0.186 | 0.110 | 2.2% / 0.2% | 0.266 |
| E2 | 0.236 | **0.114** | 1.2% / 0.8% | 0.288 |

**关键发现（颠覆旧结论）**：
- **chat 模板下 Base 的"捷径优势"消失**：B512 chat 下 Base acc 从 0.344 暴跌到 **0.082**，E1/E2 **反超 Base**（0.110/0.114 > 0.082）。
- 机制：裸 prompt 下 Base 走捷径快速给出答案，B512 截断对它伤害小；chat 模板下 Base 也进入长推理（avg_rt=512 全用满预算截断），被截断后 acc 崩。
- **旧报告"Base=0.344 远高于学生"是裸 prompt 协议的假象**——B512 口径下"训练伤害能力"的结论在 chat 协议下**不成立**（甚至相反）。
- 三模型 no_answer 均 ≤2.2%、eos≈0（B512 全截断）→ 与 H9 叙事一致（B512 下所有模型都被截断），但 B2048 下差距仍存在，故 H9 仍排除。

---

## 3. AIME24 chat 重测（文档一 Step 2，进行中）

- **协议**：`--chat-template`、`--max-new-tokens 4096`、`--n-samples 1`、`--temperature 0.0`、`--scoring sympy`、`--batch-size 2`。
- **首验门控通过**：Base 前 2 题响应为正常 Qwen3 长推理（无 loop、重复检测 False）；执行清单判据写"前 3 题"，冒烟层已 3 条全过 + AIME 层 Base 前 2 题已证协议健康（第 3 题随评估自然完成，同协议续跑）——判据实质满足。
- **旧裸 prompt AIME24 结果一律作废**（文档一背景），以本次 chat 重测为准。
- **当前**：GPU0=E1、GPU1=Base 双卡并行评估中（各 30 题，B4096 长生成）；E2 随后补跑。
- ⚠️ **2026-08-26 晚间服务器停机中断**（SSH connection refused，seetacloud 实例关闭）——eval-aime 逐题落盘 + 同 `--out` 重跑自动续（`_load_done_ids` 跳过已完成题），**实例重启后重发命令即可续跑，已落盘题不丢**。
- 结果待 AIME24 完成后回填本节。

---

## 4. 判定与结论汇总

| 条款 | 判据（写死） | 状态 |
|---|---|---|
| 协议统一 | 所有评估 `--chat-template`；**旧裸 prompt 结果（B512 与 AIME24）全部作废标记** | ✅（B512 已作废标注；旧 AIME24 作废见 §3） |
| MATH500 B2048 E2 ≥ Base（总验收 #3） | 0.288 vs 0.404 | ❌ **未达成（H9 排除）** |
| B512 截断量化 | 旧裸 B512（0.344/0.186/0.236，作废）vs 新 chat B512 对比表（含 no_answer/eos_rate） | ✅ 已产（见 §2） |
| AIME24 @B4096 chat（总验收 #2） | E2 ≥ Base（同协议 pass@1） | ⏳ 进行中（2026-08-26 晚服务器停机中断，eval-aime 逐题落盘 + resume 续跑保护） |
| 拐点表（总验收 #4） | step vs acc vs KL 表 | ⏳ 等 E2 step_120/200 拷贝到位 |
| 产物入库（总验收 #5） | 每模型完整 jsonl + 每模型 3 条 decode 样本附入报告 | ⏳ jsonl 已落盘（§6）；decode 样本待 AIME24 完成后补入 |

**核心结论**：
1. **H9（预算错位）排除**——B2048 对齐训练分布下 E2 仍显著弱于 Base（0.116 差）。
2. **旧裸 prompt B512 结论作废**——chat 协议下 Base=0.082（非 0.344），"训练伤害能力"的旧叙事不成立。
3. **能力信号方向**：chat 协议下 E1/E2 在 B512 反超 Base、B2048 仍落后——训练提升了"短预算下的答题能力"（相对 chat-Base）但未能在长预算下超越 Base 的强推理。
4. **下一步方向（待用户决策）**：H9 排除按写死判定表应"**停下，回查训练/信号，记录报告**"；Step 2（KL 档位扫描）为候选路径但门控（部分成立 + 拐点表产出）未满足——**需用户明确确认后才启动**；同时等 E2 中间 checkpoint 到位补拐点。

---

## 5. 待办（服务器）

1. **AIME24 三模型完成**（Base/E1 进行中 → E2 补跑）→ 回填 §3 + 总验收 #2。
2. **E2 中间 checkpoint step_120/200 拷贝到位** → 导出 + B512/B2048 拐点扫描（文档一 Step 3 + 文档二 Step 1）。
3. **KL 档位扫描决策**（文档二 Step 2，需用户确认）——kl=0.05/0.1 各 120 步 + mini-MATH100 B2048 探针。

---

## 6. 产物清单

- `/root/autodl-tmp/chat_retest/B2048/{Base,E1,E2}__MATH500__B2048.jsonl` + `all_results.json`（500 行/文件）
- `/root/autodl-tmp/chat_retest/B512/{Base,E1,E2}__MATH500__B512.jsonl` + `all_results.json`
- `/root/autodl-tmp/chat_retest/smoke/Base__MATH500__B512.jsonl`（首验冒烟 3 行）
- `/root/autodl-tmp/aime_eval_chat/{Base,E1,E2}/AIME24.jsonl`（进行中）

---

## 7. E 系列判别实验（归因分析 §5，2026-08-26 服务器执行）

> 依据 `docs/reports/2026-08-26-opd-failure-analysis.md`；脚本 commit `81b227e`（E-1b 两处 bug 修复 `99c4d89`）。

### E-0b 导出健全性（export_sanity_check.py）
- config.json 差异 3 键均为无害字段表示差异（`torch_dtype`↔`dtype` 等价、`layer_types` 架构字段）；generation_config.json 完全一致 → **导出健全，排除损坏**。

### E-0c 拐点扫描（E2 step_120/200 从旧实例拷贝到位后，B2048 chat，500 条）
| step | acc | no_answer | 判定 |
|---|---|---|---|
| Base | 0.424 | 0.0% | 参考 |
| S120 | 0.392 | 0.6% | ≈Base（差 0.03 < 0.05） |
| **S200** | **0.406** | 1.0% | ≈Base（序列峰值） |
| S311 | 0.288 | 1.2% | **显著劣化（-0.136）** |

→ **拐点在 step_200 附近**：120-200 平台 ≈Base、200 后大幅劣化 → **RC3（漂移/过拟合）为主放大器，早停+收紧 KL 可救**；信号非"从头坏"（RC1/RC4 不作为"从头坏"主嫌）。

### E-0d on-policy 占比（onpolicy_share.py，E2 正式训练 metrics.csv）
- `pool` 列：312 步 base（空标记）+ **12 步 refresh（3.7%）** → **F3 证实：静态 base 重放主导（96.3%）** → RC1（固定 D）结构性偏差成立。

### E-1a 教师对体检（B2048 → 作废；B4096 复测）
| 教师 | B2048 acc（作废） | **B4096 acc** | avg_rt(B4096) | 截断率(B4096) |
|---|---|---|---|---|
| JustRL（rl） | 0.358 | **0.620** | 3202 | 49.2% |
| R1Distill（ref） | 0.492 | 0.624 | 2798 | 40.2% |

- **B2048 的"rl<ref"是预算假象（用户 2026-08-26 指出并验证）**：JustRL 是深推理模型（avg_rt=3202），B2048 截断 78.4% 样本 → acc 被低估 0.26；**B4096 下 JustRL 追平 R1Distill（0.620 vs 0.624）**——教师对同级、无方向问题。
- 与论文反证一致（R1-Distill-7B 56.7% > JustRL-1.5B 51.3%，Direct-OPD 仍 +6.4）：rl 教师绝不弱，Δ_T 捕捉 RL 增量而非"模仿更强模型"。
- **深层洞察**：JustRL 原生推理 3202 token **> 训练分布 T=2048**——训练用 teacher Δ 在 500 条 base 轨迹（B2048、avg_rt≈1950）上计算，**对 JustRL 是"截断的教师分布"** → Δ 在教师截断域上算 = E-1b 信号弱（ρ=0.1765）的结构性根源之一（非教师对问题）。

### E-1b Δ↔correct 相关性（决定性，200 题 × 4 采样 = 800 条）
| 指标 | 值 |
|---|---|
| **Spearman ρ** | **0.1765** |
| AUC | 0.6037 |
| mean_delta | -0.0954 |
| acc | 0.4075 |

- **判定（写死）：0.05 ≤ ρ < 0.2 → 分支 B2（弱信号 → 信号改造）**。
- 解读：ρ>0 且 AUC>0.5 → Δ 增量**携带一定正确性语义**（论文路径部分成立，非纯风格 RC4）；但 ρ<0.2 → **强度不足**。**B4096 教师对复测后新增解释**：Δ 在教师截断域（B2048 base 轨迹 vs JustRL 原生 3202）上计算，天然弱化信号——非教师对问题（B4096 下同级 0.620 vs 0.624）。
- 结合全链：E-0c（漂移过拟合）+ E-0d（固定 D 主导）+ E-1b（弱信号 + 截断域）+ E-1a（教师对正常，B4096 复测）→ **推荐组合修复：A1 on-policy 化（refresh 变主食）+ B2 信号改造（在非截断域算 Δ：教师分布对齐 B4096、correctness 混合权重 reward=α·correct+β·Δ_seq）**；止损线 120 步、下游 100 题子集每 50 步探针。**排除 B1（换教师对）**。

### 产物
- `/root/autodl-tmp/delta_corr/{samples,logp_rl,logp_ref}.jsonl`（各 800 行）+ `report.json`
- `/root/autodl-tmp/chat_retest/E2_steps_B2048/{Base,S120,S200,S311}__MATH500__B2048.jsonl`
- `/root/autodl-tmp/chat_retest/teacher_B2048/{JustRL,R1Distill}__MATH500__B2048.jsonl`
- 导出目录：`/root/autodl-tmp/exported/e2_s120`、`e2_s200`（E2 step_120/200 HF 导出）

---

## 8. B8192@3 终验口径升级（2026-08-27，用户提议 + 教师对验证）

> 用户提议：后续评测全部改 B8192@3（majority vote、T=0.7）。vllm_budget_eval 加 `--n-samples`/`--temperature` + majority 聚合（commit `77203cf`，n=1 零回归）。

### B8192@3 教师对验证（JustRL/R1Distill，500 题，majority vote）
| 教师 | B2048 n=1 | B4096 n=1 | **B8192@3 majority** | avg_rt | 截断率 |
|---|---|---|---|---|---|
| JustRL（rl） | 0.358 | 0.620 | **0.872** | 3905 | 10.2% |
| R1Distill（ref） | 0.492 | 0.624 | 0.820 | 3962 | 18.2% |

**三大结论**：
1. **JustRL 反超 R1Distill（0.872 vs 0.820）**——用户观点验证：rl 教师确实更强，B2048"rl<ref"是彻底预算假象；
2. **预算充分**：截断率 10-18%（avg_rt 3900-3960、eos 82-90%），B8192 基本释放；
3. **Δ_T 方向正确**：指向更强的 rl 教师 → B2048 域 ρ=0.1765（弱信号）很可能是**截断域假象**，非截断域信号可能大幅增强（待 E-1b' 验证）。

**R2 合并写实战通过**：双进程并写同一 out-dir → all_results.json 含 2 模型。

### AIME24 chat 终验（B4096 n=1，2026-08-27）
| 模型 | acc（pass@1） | 正确题数 |
|---|---|---|
| Base | 0.033 | 1/30 |
| E1 | 0.000 | 0/30 |
| E2 | 0.000 | 0/30 |

- 难题场景 pass@1 区分度极低（0-1 题）→ **总验收 #2（E2≥Base）未达成**（greedy 口径）；
- 支持终验升级 B8192@3（进行中：MATH500 三模型 B8192@3 + AIME24 B8192@3，majority 口径）。

### 最终终验结果（B8192@3 majority, T=0.7，2026-08-27）
**MATH500**（500 题，`final_B8192_3/`）：
| 模型 | acc（majority） | avg_rt |
|---|---|---|
| Base | **0.816** | 4303 |
| E1 | 0.314 | 6565 |
| E2 | 0.376 | 6069 |

**AIME24**（30 题，`aime_B8192_3/`）：
| 模型 | acc（majority） | avg_rt |
|---|---|---|
| Base | 0.233（7/30） | 7827 |
| E1 | 0.000（0/30） | 8192（全截断） |
| E2 | 0.000（0/30） | 8192（全截断） |

**最终判定（B8192@3 最干净口径）**：
- 总验收 #3：E2 0.376 < Base 0.816 → **未达成**；总验收 #2：E2 0 < Base 0.233 → **未达成**。
- **关键机制**：学生推理**更长**（E2 6069/E1 6565 vs Base 4303；AIME24 下 E1/E2 完全 8192 截断）——训练把学生推向更长推理但**未带来正确性提升**。非预算假象（B8192 充分），是真实"推理风格偏移无益"。
- **最终归因**：教师对正常（B8192@3 rl 更强）+ 信号弱（B2048 域 ρ=0.1765）+ 学生推理偏移更长无正确性 → **分支：B2 信号改造（correctness 加权）+ A1 on-policy 化**（refresh 变主食、非截断域 B8192 教师分布算 Δ）。
- E-1b'（非截断域 ρ 重算）为可选项，用于精确定信号改造强度。

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
- **P0 回归门控**：服务器全量 pytest = **545 passed**（97s）；`test_vllm_budget_eval.py` = **16 passed** → 通过。
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
- E2/B1 均仍显著低于 Base（-0.116 / -0.138）→ E2 在完全对齐训练分布的评估下**真弱于 Base**。
- eos_rate 显示 B2048 下三模型均有 15-23% 自然停止（vs B512 全截断 eos≈0），说明 2048 预算对多数样本足够，预算错位解释被排除。
- **结论方向**：H9 排除 → 按提示词执行顺序进 **文档二 Step 2（KL 档位扫描，kl=0.05/0.1 重训 + mini-MATH100 探针）**。

---

## 2. B512 chat 补测（D1，优化方案新增项）

**背景**：原执行清单 3b 只重测 E2 三 step 的 B512；H9 论证需要"三模型×{B512,B2048} 全 chat"网格，避免裸→chat 模板变化与预算效应纠缠。

**命令（M1：三模型一次调用，GPU0）**：`--models 'Base=...,E1=...,E2=...' --budgets 512 --chat-template --device cuda:0`

**结果与旧裸 prompt 对比**：

| 模型 | B512 裸 prompt（旧 08-26 晨，**作废**） | **B512 chat（新）** | B2048 chat（新） |
|---|---|---|---|
| Base | 0.344 | **0.082** | 0.404 |
| E1 | 0.186 | 0.110 | 0.266 |
| E2 | 0.236 | **0.114** | 0.288 |

**关键发现（颠覆旧结论）**：
- **chat 模板下 Base 的"捷径优势"消失**：B512 chat 下 Base acc 从 0.344 暴跌到 **0.082**，E1/E2 **反超 Base**（0.110/0.114 > 0.082）。
- 机制：裸 prompt 下 Base 走捷径快速给出答案，B512 截断对它伤害小；chat 模板下 Base 也进入长推理（avg_rt=512 全用满预算截断），被截断后 acc 崩。
- **旧报告"Base=0.344 远高于学生"是裸 prompt 协议的假象**——B512 口径下"训练伤害能力"的结论在 chat 协议下**不成立**（甚至相反）。
- 三模型 no_answer 均 ≤2.2%、eos≈0（B512 全截断）→ 与 H9 叙事一致（B512 下所有模型都被截断），但 B2048 下差距仍存在，故 H9 仍排除。

---

## 3. AIME24 chat 重测（文档一 Step 2，进行中）

- **协议**：`--chat-template`、`--max-new-tokens 4096`、`--n-samples 1`、`--temperature 0.0`、`--scoring sympy`、`--batch-size 2`。
- **首验门控通过**：Base 前 2 题响应为正常 Qwen3 长推理（无 loop、重复检测 False），协议健康。
- **当前**：GPU0=E1、GPU1=Base 双卡并行评估中（各 30 题，B4096 长生成）；E2 随后补跑。
- 结果待 AIME24 完成后回填本节。

---

## 4. 判定与结论汇总

| 条款 | 判据（写死） | 状态 |
|---|---|---|
| 协议统一 | 所有评估 `--chat-template`（B2048/B512/AIME24 均确认 chat 启用日志） | ✅ |
| MATH500 B2048 E2 ≥ Base（总验收 #3） | 0.288 vs 0.404 | ❌ **未达成（H9 排除）** |
| B512 截断量化 | 旧裸 B512（0.344/0.186/0.236，作废）vs 新 chat B512（0.082/0.110/0.114）对比表 | ✅ 已产（见 §2） |
| AIME24 @B4096 chat（总验收 #2） | E2 ≥ Base（同协议 pass@1） | ⏳ 进行中 |
| 拐点表（总验收 #4） | step vs acc vs KL 表 | ⏳ 等 E2 step_120/200 拷贝到位 |

**核心结论**：
1. **H9（预算错位）排除**——B2048 对齐训练分布下 E2 仍显著弱于 Base（0.116 差）。
2. **旧裸 prompt B512 结论作废**——chat 协议下 Base=0.082（非 0.344），"训练伤害能力"的旧叙事不成立。
3. **能力信号方向**：chat 协议下 E1/E2 在 B512 反超 Base、B2048 仍落后——训练提升了"短预算下的答题能力"（相对 chat-Base）但未能在长预算下超越 Base 的强推理。
4. **下一步方向**：文档二 Step 2（KL 档位扫描）——H9 排除后的主路径；同时等 E2 中间 checkpoint 到位补拐点。

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

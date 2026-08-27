# 修改版方案 v2：论文原样复现（撤回换对，R1Distill→JustRL + Qwen3-1.7B）

> 依据：用户 2026-08-27 决策"**严禁更换教师对**，仿照原始论文进行实验设计；
> 教师对 R1-Distill-1.5B（ref）→ JustRL-1.5B（post-RL）；学生 Qwen3-1.7B"。
> 本文档 = v1（`docs/plans/2026-08-27-newpair-mechanism-validation.md`，换对路线）的修改版 v2，
> 撤回换对 + 评估基准改论文口径 + 配方完全对齐论文 Table 2/3。
> 论文依据：`docs/directOPD_analysis.md`（Table 2/3、Tab.1、Appendix A、§4.2/4.3）。

---

## 0. 风险修正（v2.2，2026-08-27 数据质量审阅后）

> 数据质量二次审阅结论：**整体合格**（数据源/模板/学生自身 rollout/top_k/T/lr 六项核心
> 对齐论文 + R-E2 门控），但识别出 R1-R7 潜在风险。以下为逐项修正方案（R3 为新增关键修正）。

| # | 风险 | 修正方案 | 文件/动作 |
|---|---|---|---|
| **R1** | **prompt 截断不一致**：prepare 生成时 `max_length=2048` 硬编码、训练 loader 用 `P=1024` → 长题（套模板后 >1024 token）response 依赖被截断上下文，训练/teacher 打分错位 | ① prepare 加 `--max-prompt-len`（默认 1024 对齐 loader）；② 生成命令显式 `--max-prompt-len 1024`；③ Phase 0 用 `samples.jsonl` 的 `prompt_token_len` 统计 max（E-1b 已有该字段，零成本）判据：`max(prompt_token_len) < 1024` 则消解 | `prepare_skywork_responses.py` |
| **R2** | **E-1b 相关性域 ≠ DAPO 域**：`delta_corr` sample 阶段直接 `apply_chat_template` 包 problem 原文（未 format_prompt dapo）→ ρ 判据与训练模板错位 | `delta_correctness_corr.py` `_sample_stage` 加 `--prompt-style`（默认 boxed 零回归；dapo 时 `format_prompt(p, "dapo")` 再包）；E-1b' 命令加 `--prompt-style dapo` | `delta_correctness_corr.py` |
| **R3** | **on-policy 占比回退**：v2 计划漏设 `l2.t_train`（默认 100）→ refresh 相位 300/100=3 次 × 32 条 = 96 条 on-policy vs 5000 静态 = **~2%**（等于旧失败） | **训练命令加 `--set l2.t_train=2`**：300/2=150 相位 × 32 = 4800 条 on-policy vs 5000 静态 ≈ **49%**；`refresh_min_interval=1` 保持；Phase 1 用 `onpolicy_share.py` 复核 pool 占比 ≥40% | `run_s2_real.py` 命令（配置不改默认） |
| **R4** | 数据无内容级去重（重复题→过拟合） | Skywork-OR1 数学子集来源已去重（论文同源）；补充：Phase 0 统计 `n_unique_prompt`（hash 前 100 字符），判据 >99%；不达标则 `prepare_skywork_jsonl --n 5000 --seed 43` 重采样 | 命令层，无代码 |
| **R5** | prepare 无 loop 检测（Qwen3-1.7B 已知裸 prompt 会 loop，chat 模板下大幅缓解但未消除） | 复用 `model.detect_loop`（`model.py:118`）：生成后对 response 做 `detect_loop`，loop 条目标记并计数；R-E2 门控增加 `loop_rate ≤5%` 判据；不达标停查 | `prepare_skywork_responses.py` |
| **R6** | top_p=0.95 vs 论文 1.0 | prepare 生成命令显式 `--top-p 1.0`（论文 Table 3 training sampling） | 命令层 |
| **R7** | decode `skip_special_tokens=True` → 训练序列无 eos token | 低风险（base 池 mask 全 1 假设）；记录在案，不改 | — |

**R3 是本次审阅新发现的关键修正**：旧实验 refresh 3.7% 失败，v2 若沿用 `t_train=100`
默认值会重蹈覆辙（on-policy 仍 ~2%）。**训练命令必须显式 `--set l2.t_train=2`**，
否则 on-policy 化（RC1 修复）名存实亡。

---


---

## 1. v1 → v2 变更总览（核心差异）

| 维度 | v1（换对，已废弃） | **v2（论文原样复现，本文档）** |
|---|---|---|
| 教师对 | Qwen3-1.7B-Base ↔ Qwen3-1.7B-Instruct（需下载 Base） | **R1-Distill-1.5B（ref）→ JustRL-1.5B（rl）**（服务器已有，无需下载/模板补丁） |
| 学生 | Qwen3-1.7B-Base（需下载） | **Qwen3-1.7B**（服务器已有；论文 Tab.1 主学生） |
| 主判据 | MATH500 B8192@3 majority | **AIME24/25 ave@32**（论文 Table 2）+ MATH500 B2048 辅助 |
| Prompt 模板 | boxed（评估侧硬编码） | **DAPO**（论文 Appendix A：训练与评估同用；数据 prompt 已是 DAPO） |
| 训练配方 | 120 步止损 | **300 步**（论文 Table 3）+ on-policy（refresh 主食） |
| 新增代码 | — | 评估侧：`--metric ave`、`--prompt-style dapo`、`--max-model-len 32768` |

**撤回换对的理由**：我们失败的主因是**配方层 + 评估基准**，不是教师对/学生——论文用
本组合证明 Qwen3-1.7B 在 AIME24 +10.0（Tab.1）；E-1a 已证教师对方向正确（B8192@3 下
rl>ref）；E-1b 的 ρ=0.1765 是在 **B2048 域 + boxed 模板**下测的，而论文信号在
**DAPO 模板 + AIME 难题域**下更强（论文 A 附录明说 DAPO 迁移略好）。故 v2 回到原对、
对齐全部配方，而不是靠换对绕开信号。

---

## 2. 已具备（v1/C1-C4 + 本会话修复，无需重做）

- ✅ **C1** top_k=16 允许（`config.py` validator 加 16，cache 重建提示齐全）
- ✅ **C2** n_rollout（`adaptive_cache.py` 实现 + 本会话 2 个 bug 修复：
  展平行序配对错位改 `repeat`、pipeline 透传 `n_rollout`；测试 12 passed）
- ✅ **C3** AdaptiveKLController（Eq.16 实现 + 双训练步接入 + 8 单测；默认 false 零回归）
- ✅ **C4** 配方骨架：lr=1e-6、kl_adaptive=true、n_rollout=4、T=2048、materialized 5000、
  top_k=16（`configs/qwen3_base_opd.yaml` 存在，v2 只需改模型路径 → 新文件名）
- ✅ R-E1 eos 定档、R-E2 D 质量门控（≤30%）、R-S1 词表校验、R-S2 sample 长度、
  R-S3 快评-全量一致性（v1 审阅修正，全部保留——v2 学生是 Qwen3-1.7B，
  eos 直接 151645，仍跑一次 R-E1 冒烟确认）
- ✅ 全量回归 593 passed（本会话基线）

---

## 3. 需要新增/修改的代码（本地，全部带测试）

### 3.0 风险修正配套代码（v2.2，R1/R2/R5）

1. **`scripts/prepare_skywork_responses.py`**（R1/R5）：
   - 加 `--max-prompt-len`（默认 1024，对齐训练 loader 的 `P=max_prompt_len`）：
     `enc = tok(batch_prompts, ..., max_length=args.max_prompt_len)`（当前硬编码 2048）。
   - 加 `--loop-check`（默认 on）：生成后对每条 response 调 `model.detect_loop`
     （`fullstack_opd_v2/model.py:118`，纯函数），loop 条目标记并在日志统计
     `loop_rate`（供 R-E2 门控 ≤5% 判据）。
2. **`scripts/delta_correctness_corr.py`**（R2）：`_sample_stage` 加 `--prompt-style`
   （默认 boxed 零回归）：`format_prompt(p, args.prompt_style)` 后再
   `apply_chat_template`——E-1b' 用 `--prompt-style dapo` 使相关性域与训练模板一致。
3. 测试：prepare 的 max-prompt-len 截断、loop 标记；delta_corr dapo 包装各 1-2 例。

### 3.1 `scripts/vllm_budget_eval.py`（评估侧，3 处）

1. **`--metric`**：`Literal["majority","ave"]` 默认 `majority`（零回归）。
   - `ave` = 论文 ave@32：每题 n 采样中答对比例的平均，再全部题目平均
     （`samps` 已逐条判分 `outcome_correct`；`fracs.append(mean(correct)/n)` 聚合）。
2. **`--prompt-style`**：`Literal["boxed","dapo"]` 默认 `boxed`（零回归）；透传给
   `build_prompts(problems, style, tok)`（`build_prompts` 已有 style 参数，只需 main
   从 `args.prompt_style` 读而非硬编码 `"boxed"`）。
   - dapo 模式答案提取用 `eval_aime.extract_answer(text, "dapo")`（已存在）
     + `_grade_answer_sympy`（同 boxed 路径）。
3. **`--max-model-len` 默认 12288 → 32768**（覆盖 31744+1024 论文协议；12288 兼容旧
   B8192/B2048 不变，显存 96GB 下 1.7B 无压力——KV 32768 可行，待 P0 实测确认）。

### 3.2 `fullstack_opd_v2/eval_aime.py`（80% 守卫与论文 31744 冲突）

- 论文 max generation length 31744 > 0.8×32768=26214（Qwen3-1.7B
  `max_position_embeddings=32768`）→ 当前 `eval_aime.py:408` 会 `ConfigError`。
- 处置：**AIME 主评估走 vllm_budget_eval（--metric ave --prompt-style dapo
  --max-model-len 32768 --n-samples 32）**，eval-aime 仅作 sanity 对照（--n-samples 1）；
  守卫不改（保持 P2 修复 #5 的保护语义），若确需 eval-aime 跑 ave@32 再单独立项放宽。

### 3.3 测试（新增）

- `tests/test_vllm_budget_eval.py`：`--metric ave` 聚合（3 采样 2 对 1 错 → 0.667）、
  `--prompt-style dapo` 提取（Answer: 行）、`--max-model-len 32768` 默认值断言。
- 全量回归 0 failed（基线 593）。

---

## 4. 实验设计（论文原样，Phase 0/1/2）

### 4.0 前置（服务器恢复后，~15min）

```bash
cd /root/opd && git pull && git log --oneline -1   # 到含 C1-C4+修复+评估改动的 commit
grep -n 'metrics = out' main/scripts/run_s2_real.py   # 恰 1 行（R1）
PYTHONIOENCODING=utf-8 /root/miniconda3/bin/python -m pytest main/tests/test_vllm_budget_eval.py -q | tail -1  # ≥26
# 模型路径确认（全部已有，无下载）：
ls /root/autodl-tmp/models/Qwen__Qwen3-1.7B            # 学生 = rl 教师
ls /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B  # ref 教师
ls /root/autodl-tmp/models/JustRL-DeepSeek-1.5B           # rl 教师（=论文 JustRL-1.5B）
```

### 4.1 学生基线 + 教师对体检（论文 Tab.1 口径，AIME24/25 ave@32）

```bash
# 双卡并行：GPU0 学生基线，GPU1 教师对
for M in BaseS JustRL R1Distill; do ... done   # 各自：
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "$M=...路径..." --budgets 8192 --n-samples 32 --temperature 0.7 \
  --dataset AIME24 --n-limit 30 --chat-template --prompt-style dapo \
  --metric ave --max-model-len 32768 --device cuda:0 --out-dir /root/autodl-tmp/r1_eval/AIME24
# AIME25 同法；MATH500 B2048（--n-samples 1, --metric majority）辅助
```

- 论文参考值（Tab.1，ave@32）：Qwen3-1.7B init AIME24=48.3 / AIME25=36.8；
  JustRL=51.3、R1Distill=28.5（AIME24）。
- **门控（写死）**：学生基线 AIME24 ave@32 ∈ [20,60]（论文 48.3 附近 ±；
  <20 说明 dapo 提取/协议有问题，停查）。

### 4.2 E-1b'：Δ↔correct 相关性（**DAPO 域**重测，原对）

> 此前 E-1b 在 B2048+boxed 域得 ρ=0.1765；论文信号在 DAPO 模板下更强——
> 用 DAPO 域重测，作为训练前信号门控。

```bash
O=/root/autodl-tmp/r1_eval/delta_corr
# sample：DAPO 模板（delta_correctness_corr.py 目前直接 apply_chat_template 包 problem
# 原文，未 format_prompt——需确认/补 dapo 包装，见 §3 附注）
# logp rl/ref 双卡并行（原对路径覆盖）→ correlate
```
- **门控（写死）**：ρ ≥ 0.2 进训练；0.05-0.2 用户决策；<0.05 停（机制层存疑）。

### 4.3 数据 D（on-policy 学生自身 rollout，论文语义）

- **V7 仍适用**：`cp skywork_50k.jsonl skywork_50k_r1.jsonl` + 清空 response（500 条旧
  response 是 Qwen3-1.7B 生成的，但为干净起见清空重生成 5000 条，避免与新 rollout 混）。
- 生成（R1/R4/R5/R6 修正后）：`prepare_skywork_responses.py --model Qwen__Qwen3-1.7B
  --apply-chat-template --max-samples 5000 --seed 43 --max-new-tokens 2048
  --max-prompt-len 1024 --top-p 1.0`（双卡分片）+ R-E2 质量门控（含新增 loop_rate ≤5%）。
- **R4**：Phase 0 统计 `n_unique_prompt`（hash 前 100 字符）判据 >99%；不达标重采样
  `prepare_skywork_jsonl --n 5000 --seed 43`。
- **R-E1**：eos 冒烟——Qwen3-1.7B 是 instruct，`--eos-id 151645`（im_end）；
  冒烟确认后再定（流程保留）。

### 4.35 设计决策记录：训练数据 max_response_len=2048 的依据（2026-08-28 追加）

> 用户疑问："为什么训练数据是 2048 的 max len？不会太少了吗？训练数据的长度对训练是否有影响？"
> 以下为对照原始论文（`docs/directOPD_analysis.md`）的论证记录——**2048 是论文实验筛选出的
> 最优短视界值，不是拍脑袋**，且在本方案中照抄论文 Table 3 属于正确选择。

**论文原文依据（`[原文明确]`）**：
1. Table 3：`Max response length = 2,048（short-horizon；§4.2 证明可泛化到长 rollout）`。
2. §4.2：**2k response length 在 Qwen3-1.7B 与 R1-Distill-7B 上验证最稳（Fig.7）**；
   40 步 2k 训练后 actor 在 ~16k 长 rollout 上已朝教师偏移方向移动（Fig.8）——**训练短视界、
   评估长 rollout 可泛化**；**6k 在晚期不可靠前缀上过驱动、验证反而更差（45.6 vs 2k 的 48.8）**
   ——**训练长度加长不是单调有利的**。

**为什么短视界训练反而更稳（机制）**：Direct-OPD 的 Δ_T 是 per-token 稠密奖励（Eq.10），
每个 token 位置都有信号，不需要完整 response 就能学偏移方向。推理"开启方式"在前段决定、
前段 token 的 Δ 更可靠；6k 训练把"晚期不可靠前缀"也纳入梯度反而过驱动。配合
Rao-Blackwell 化（Eq.13）+ top-k=16 支撑，短序列每个 token 的学习效率更高。short-horizon
本身就是泛化性来源（论文 §4.2 核心主张）。

**我们实测对照**：Qwen3-1.7B 在 MATH500 上 B2048 avg_rt=1954（接近满预算）、B8192 avg_rt=4303
——自然推理长度确实超 2048，训练 D 中会有截断样本。但这正是论文用同款学生验证过的设定
（Fig.7 的 2k 档同样大量截断仍最稳）：Δ_T 是 per-token 信号，截断的 response 前 2048 token
依然携带有效梯度。

**v2.2 已为截断质量上了三重保险**：R-E2 门控（no_answer ≤30%、过短 ≤20%）+ R5 loop 检测
（loop_rate ≤5%）+ R-S2 sample 长度检查（Phase 0 实测）——若 Skywork 题比 MATH500 更长、
截断率过高，由门控拦下；确实不达标才考虑 D 生成预算升档（`--max-new-tokens 4096`），
但**不盲提高**（6k 档教训）。

**结论**：照抄论文 2048 是正确且经过验证的选择；真正需监控的是截断导致的 no_answer/loop
质量，v2.2 门控已覆盖。

### 4.4 新配置 `configs/qwen3_r1_opd.yaml`（复制 qwen3_base_opd.yaml 改模型路径）

```yaml
student_path:     /root/autodl-tmp/models/Qwen__Qwen3-1.7B
teacher_rl_path:  /root/autodl-tmp/models/JustRL-DeepSeek-1.5B
teacher_ref_path: /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-1.5B
dataset:
  path: /root/autodl-tmp/datasets/skywork_50k_r1.jsonl
  max_response_len: 2048        # 论文 Table 3（短视界）
stage1:
  cache_path: /root/autodl-tmp/cache_qwen3_r1_t16.pt
  top_k: 16                     # 论文 student top-k support（C1）
  load_cache: false             # 首次建缓存；训练置 true
stage2:
  lr: 1.0e-6                    # 论文 Table 3
  kl_reg_coef: 1.0              # adaptive KL 初始值 α0
  kl_adaptive: true             # 论文 §2.4 Eq.16（C3）
  n_steps: 300                  # 论文 Table 3（止损线后移，拐点扫描保留）
  batch_size: 4
  gradient_checkpointing: true  # 经 --set
l2:
  rollout:
    max_new_tokens: 2048
    n_rollout: 4                # 论文 rollout n=4（C2）
    temperature: 1.0
    repetition_penalty: 1.0
    loop_detection: true
    loop_periods: [2, 3, 4]
    rollout_source: student
run:
  checkpoint_every: 20          # step20/.../300 共 15 断点（拐点扫描）
eval:
  model_path: /root/autodl-tmp/models/Qwen__Qwen3-1.7B
  n_samples: 32                 # 论文主指标 ave@32（辅助口径）
  temperature: 0.7
  max_new_tokens: 8192
  top_p: 0.95
  metric: ave
  prompt_style: dapo            # 论文模板（对齐 DAPO）
  scoring: sympy
  chat_template: true
  attn_implementation: flash_attention_2
```

### 4.5 训练（on-policy 化，论文 verl 语义）

```bash
python scripts/run_s2_real.py \
  --config configs/qwen3_r1_opd.yaml \
  --run-dir /root/autodl-tmp/runs_r1/e1_opd2048 \
  --names S2_E2_opd1024 --device cuda:0 --n-steps 300 \
  --eos-id 151645 \
  --set stage2.rollout_engine=vllm \
  --set stage2.gradient_checkpointing=true \
  --set l2.t_train=2 \
  --set l2.cache.refresh_min_interval=1 \
  --set l2.m_refresh=8 \
  --set l2.rollout.max_new_tokens=2048 \
  --set l2.rollout.n_rollout=4 \
  --set l2.rollout.repetition_penalty=1.0
```

- **on-policy（A1 核心，R3 修正）**：`--set l2.t_train=2`（**关键**：默认 100 会让
  refresh 相位仅 300/100=3 次 → on-policy ~2% 重蹈旧失败）+ `refresh_min_interval=1`
  使 refresh 相位与训练相位交替频繁——300/2=150 相位 × m_refresh(8) × n_rollout(4)=32
  条 = 4800 条 on-policy vs 5000 静态 ≈ **49%**。Phase 1 用 `onpolicy_share.py` 复核
  pool 占比 ≥40%，不达标停查（RC1 未修复）。
- **止损（E-0c 教训）**：60/120/200/300 快评（AIME 100 题子集无 AIME 子集——
  用 MATH500 100 题子集 B2048 作快评代理），下游不升即停（保留 120 步决策点）。

### 4.6 判定（写死）

| 结果（AIME24 ave@32 全量 30 题） | 判定 |
|---|---|
| ≥ 学生基线 + 5 点（论文 +10） | **积极结果：论文复现成功** |
| +2 ~ +5 点 | 弱阳性：200-300 步/refresh 提高再判 |
| < +2 点 | 失败：信号/实现层审计（E-1b' ρ 复核） |

---

## 5. 风险表

| 风险 | 缓解 |
|---|---|
| AIME 仅 30 题，ave@32 采样均方差大（±2-3 题 ≈ ±7-10 点） | 主判据 ave@32 已平均化；AIME24+AIME25 双基准交叉确认 |
| 学生基线门控范围 [20,60] 若论文 48.3 复现失败 | 先测基线（4.1），超范围停查 dapo 提取/协议 |
| `max-model-len 16384` 显存 | 96GB 1.7B 无压力（KV 16k 减半，P0 nvidia-smi 实测确认） |
| 80% 守卫拦 31744（eval-aime） | AIME 主走 vllm_budget_eval；eval-aime 仅 sanity（--n-samples 1） |
| on-policy 化后 refresh 频率高 → 训练变慢 | refresh_min_interval=1 起步，慢则 2-3 折中 |
| 旧 500 条 response 污染 | V7 清空重生成 5000 条（4.3） |
| delta_corr 未 dapo 包装（E-1b' 前需确认） | §3 附注：sample 阶段补 `format_prompt(p,"dapo")` 或说明沿用原文（数据 prompt 已是 DAPO） |

---

## 6. 验证（判据写死）

```bash
cd main
PYTHONIOENCODING=utf-8 python -m pytest tests/test_vllm_budget_eval.py -q   # 含 ave/dapo/maxlen 新测试
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q                           # 全量 0 failed（基线 593）
grep -n "prompt-style\|metric\|32768" scripts/vllm_budget_eval.py | head    # 判据：新参数存在
# 服务器：Phase 0 门控通过才训练；全部数字来自真实运行（不伪造）
```

## 7. 提交纪律

- 评估代码改动（§3）独立提交（`feat(eval): --metric ave / --prompt-style dapo / max-model-len 32768`）。
- 新配置 `configs/qwen3_r1_opd.yaml` + 计划文档 v2 各自提交；推 worktree 分支 + main。
- 服务器不实跑（本任务纯本地代码+文档）；实跑按 v2 计划执行。

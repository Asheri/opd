# 对齐 Direct-OPD 论文实验设计 —— 设计规格

> 日期：2026-08-13 ｜ 状态：用户已批准方案 A（跨词表 top-K 支撑）
> 目标：数据集、教师对、词表策略完全对齐 Direct-OPD 论文（arXiv:2607.05394），
> 修复 7B 训练失败 + 纠正 4B/17B 的"语义错位伪成功"

## 一、背景与关键发现

### 1.1 词表事实（服务器实测 + HF config 验证）

| 模型 | 家族 | config vocab | tokenizer 实际 | 与 Qwen2.5-1.5B 教师兼容 |
|------|------|-------------|---------------|------------------------|
| DeepSeek-R1-Distill-Qwen-**1.5B**（pre-RL 教师） | Qwen2.5 | 151936 | 151665 | — |
| JustRL-**DeepSeek-1.5B**（post-RL 教师） | Qwen2.5 | 151936 | 151665 | 同源 ✅ |
| DeepSeek-R1-Distill-Qwen-**7B**（7B 学生） | Qwen2.5 | **152064** | 151665 | **27/27 id 一致** ✅ |
| Qwen3-**1.7B**（学生） | Qwen3 | 151936 | 151669 | **0/26** ❌ |
| Qwen3-**4B**（学生） | Qwen3 | 151936 | 151669 | 0/26 ❌ |
| JustRL-**Qwen3-1.7B**（post-RL Qwen3 教师） | Qwen3 | 151936 | 151669 | 与 Qwen3 学生 26/26 ✅ |

**两个决定性事实**：
1. **Qwen2.5 家族内不同规模 vocab 不同**（1.5B=151936、7B=152064），但 tokenizer **逐 id 兼容**（27/27）。
2. **Qwen3 与 Qwen2.5 是不同 tokenizer**（同文本 0/26）——**不能跨家族 gather Δ_T**。

### 1.2 现有问题的根因（被本次发现纠正）

| 现象 | 真实根因 |
|------|---------|
| 7B 训练崩（152064 vs 151936） | student config vocab(152064) > teacher(151936)，`ratio*delta` 维度崩。但 tokenizer 兼容 → **只需跨词表展开，不是换教师对** |
| 4B/17B "成功"但分数低（13.3% vs 论文 58.3%） | 用错教师对：**Qwen2.5 家族教师对 + Qwen3 学生**，维度碰巧相等不崩，但 **token 语义错位**（0/26），Δ_T 施加到错误 token → 伪成功 |

**修正**：Qwen3 学生必须用 **Qwen3 家族教师对**（`JustRL-Qwen3-1.7B` = post-RL，Qwen3 基座 = pre-RL ref）。

## 二、设计决策（用户已确认）

1. **对齐深度**：完全对齐论文（数据集 + 教师对 + 跨词表机制）
2. **数据集**：Skywork-OR1-RL-Data math split（105,055 条）全量下载，**子集训练**（如 10K 条）
3. **教师对策略**：**分家族配对**——Qwen2.5 家族（7B）用 R1-Distill/JustRL-1.5B 对；Qwen3 家族（1.7B/4B）用 JustRL-Qwen3 对
4. **7B 修复**：方案 A 跨词表 top-K 展开（student 超出 teacher 的 128 个 id 未命中置 0）
5. **4B 教师对**：尝试下载 JustRL-Qwen3-4B，失败复用 JustRL-Qwen3-1.7B（同 Qwen3 tokenizer 兼容）

## 三、目标模型矩阵

| 学生 | 家族/vocab | post-RL 教师 | pre-RL 教师(ref) | 需跨词表 |
|------|-----------|-------------|-----------------|---------|
| 7B（R1-Distill-7B） | Qwen2.5 / 152064 | JustRL-DeepSeek-1.5B (151936) | R1-Distill-Qwen-1.5B (151936) | ✅ 方案 A |
| 1.7B（Qwen3-1.7B） | Qwen3 / 151936 | JustRL-Qwen3-1.7B | Qwen3-1.7B 基座 | ❌ 同词表 |
| 4B（Qwen3-4B） | Qwen3 / 151936 | JustRL-Qwen3-4B（或复用 1.7B 对） | Qwen3-4B 基座 | ❌ 同词表 |

## 四、数据管道设计

1. **下载**：Skywork-OR1-RL-Data `data/math-00000-of-00001.parquet`（105,055 条，服务器学术代理）
2. **转换**：跑 `Direct-OPD/scripts/prepare_skywork_math.py` → `skywork-or1-math-dapo-original.parquet`
   （DAPO 模板 prompt + ground_truth）
3. **子集采样**：从 10.5 万条随机采样 10K 条 → 转 jsonl（`{"prompt": <DAPO 包装>, "response": <student 生成>}`）
4. **response 标签**：用**初始 student `generate_batch`** 对每个 DAPO prompt 生成响应
   （on-policy，最贴近论文；复用 `stage1.warmup_source=student_init` 机制）
5. **加载**：现有 `JsonLinesDataLoader`（jsonl → tokenizer 编码定长张量）

**规模注意**：10K 条 × prompt 右 pad 到 `max_prompt_len`（论文 1024）+ response 到 `max_response_len`
（论文 2048）。内存/显存按子集控制；缓存 build 逐条 teacher 前向，10K 条在 2×96GB 上可行。

## 五、方案 A：跨词表 top-K 支撑（仅 7B 需要）

### 5.1 核心改动（全在 `main/fullstack_opd_v2/`）

**`cache.py` — `delta_for_student_topk`**：
- 现状：`out = torch.full((B, T, self.vocab), fill)` 按 **teacher** vocab(151936) 建 →
  7B 的 `student_topk_ids`（含 ≥151936 的 id）`scatter_` 越界。
- 改：`out` 的 vocab 维度按 **student** vocab 建——从 `student_topk_ids.max()+1` 推断
  （或显式传 `student_vocab`），保证 `(B,T,152064)` 与 `ratio`(152064) 对齐。
- 超出 teacher 词表的 student id：`searchsorted` 未命中 → `found=False` → `matched=0`（已正确）。
- 新增参数：`vocab_out: int | None = None`（None → 用 max(student_topk_ids)+1）。

**`cache.py` — `TensorTeacherCache.build` 校验放宽**：
- 现状：`teacher_rl.vocab == teacher_ref.vocab`（教师对内部，保留 ✅）
- 保持：`d_model/max_len` 一致性校验（学生与教师同架构，仍需）
- 移除/放宽：无 student 侧约束（build 阶段本就不校验 student；训练期由 `delta_for_student_topk`
  的 vocab_out 兜底）——**确认现状 build 不校验 student，无需改**。

**`scheduler.py`**：
- `use_topk` 分支已把 `s_topk.indices`（student top-K ids）传给 `delta_for_student_topk`——
  核心逻辑已就位；仅需确认调用处传 `vocab_out=student.vocab`（或依赖默认推断）。

**`losses.py`**：`pg_loss`/`low_var_kl_support` 已支持 `renormalize_support`（对齐论文）——无需改。

### 5.2 测试（本地可测，toy 构造不同 vocab）

- `test_cache_delta_for_student_topk_cross_vocab`：teacher vocab=24、student vocab=32，
  student top-K 含 ≥24 的 id → 断言 out 维度 (B,T,32)、超出 id 的 Δ=0、命中 id 的 Δ=teacher 值。
- `test_scheduler_7b_style_cross_vocab_train`：7B 风格（student vocab > teacher vocab）端到端
  训练跑通、loss 有限、（新增 student 超出 id 的 top-K 支撑）Δ_T=0。
- `test_build_teacher_pair_consistency_kept`：教师对内部 vocab 不一致仍抛 `TeacherConsistencyError`。

## 六、评估协议（已实现，commit 2b0b268）

- `eval-aime --metric ave --n-samples 32 --temperature 0.7 --top-p 0.95 --prompt-style dapo`
  对齐论文 Table 2（ave@32）。
- 注意：论文评估用 DAPO 模板；本项目评估 prompt 用 DAPO 风格对齐。

## 七、多学生编排（复用现有 `scripts/multistudent/`）

- 每学生独立缓存/run（`student_init` warmup，Δ_T 缓存不共享——各家族教师对不同）。
- 打包不变：7B→cuda:0，4B+1.7B→cuda:1。
- `students.env` 更新教师对映射（按 §三 矩阵）。

## 八、需服务器验证项（部署时）

1. 下载 `JustRL-Qwen3-4B` 是否存在（不存在则复用 1.7B 教师对）。
2. Skywork parquet 下载 + `prepare_skywork_math.py` 转换成功（105,055 行）。
3. 10K 子集采样 + student 生成响应（`warmup_M` 机制）。
4. 7B 用方案 A 跑通（跨词表展开不越界）。
5. ave@32 评估三档学生（对齐论文数字量级）。

## 九、范围与不做

- **不做**：L2/L3 在线刷新；vLLM 引擎（保留 toy scorer）；Megatron。
- **不做**：把 Qwen3 学生换成 Qwen2.5 家族（偏离论文表 1 学生设置）。
- **7B 显存**：词表修复后仍需面对（fp32-Adam ≈117GB > 96GB）→ `adamw_8bit` 或 `offload_to_cpu`。

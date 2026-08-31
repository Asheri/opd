# 2026-08-28：Direct-OPD 官方实现 vs 我们实现——对照分析与修改方向

> 依据：官方仓库 `Direct-OPD/`（patched verl）逐行对照 `main/fullstack_opd_v2/`。找出信号弱之外的关键差异与后续修改方向。

## 一、核心结论

1. **E-1b' ρ 弱（+0.177/-0.147/-0.034）= 教师对本身信号弱（RC4）**——ρ 用 vLLM full logp 直接算（`delta_correctness_corr.py`），不经过我们的 cache/交集；三域交叉说明 Δ 被风格/格式主导。**不是实现 bug**。
2. **top-K 交集（我们）vs only_stu 完整学生 top-K（官方）是显著差异**——训练信号被稀释（放大器，非根因）。
3. **Rao-Blackwell 已实现**（`renormalize_topk_support` 与官方 `softmax(student_topk)` 数学等价）。
4. **DAPO 数据域一致**（官方 parquet 转 jsonl，模板逐字等价）。
5. **on-policy vs off-policy 是架构级差异**（官方每步 fresh rollout + 教师实时 forward；我们 base 池固定 + cache 离线 Δ）。

## 二、关键差异表

| 维度 | 官方 | 我们 | 差异影响 |
|---|---|---|---|
| **Δ 支撑** | 学生 top-16 **完整**支撑，取教师 full logp（`gather(logits, student_ids) − logsumexp`） | 学生 top-K ∩ **教师 top-K 交集**，非交集 Δ=0 | 交集越小训练信号越稀（放大器） |
| **on-policy** | 每步 fresh rollout n=4 + 教师实时 forward | base 池固定 + cache 离线 Δ（RC1） | 信号跟不上学生演化 |
| **IS ratio** | 不需要（on-policy） | 必须（off-policy，s_cur−s_old） | 额外噪声源 |
| **KL α0** | 2.5 | 1.0 | 需对齐 |
| **dual clip** | clip_ratio_c=3.0 | 无 | 护栏差异 |
| **batch** | 128/8卡 | 4/2卡 | — |
| Rao-Blackwell | softmax(student_topk) | π_old^renorm | ✅ 等价 |
| DAPO 模板 | 官方 DAPO | 逐字一致 | ✅ |
| 训练不用 correct/format reward | 是（format_reward=False） | 是（只用 Δ） | ✅ |

## 三、修改方向（按优先级）

### A. 换/修教师对（治本，RC4 方向）——最高优先级
- 官方实验 teacher=JustRL、student=Qwen3 是**跨 tokenizer 对**（信号强否未证）；我们新配置 `qwen3_base_opd.yaml`（Base→Instruct **同 tokenizer**）是正确方向，**应优先跑**。
- **用官方训练信号做门控**：官方有 `delta_opd/student_weighted_teacher_logprob_gap`、`student_weighted_pos_frac` 等 **top-K wing 级指标**（ray_trainer.py），比序列级 E-1b' 更贴近训练目标——建议加同样的 top-K wing 级 Δ 相关性诊断。

### B. top-K 作用域对齐 only_stu（治"训练信号稀"）
- cache 从"存教师 top-K + 交集"改成"**学生 top-K 处实时取教师 full logp**"（官方 `gather(logits, student_ids) − logsumexp`）。
- 最低成本变体：cache 教师 K 调大（64/256）+ 未命中不置 0，回退教师 full-logp 近似。

### C. 数据 on-policy 化（治 RC1）
- refresh 相位改为对 fresh rollout **实时用教师 full logp 算 Δ**（对齐官方），而非查缓存交集。

### D. KL 与护栏
- KL α0 对齐官方 2.5；保留 delta_clip（官方没有，真实教师对 Δ 可达 ±10，我们实测需要）。

### E. 不要改的（已对齐）
- loss 公式（renormalize 后与官方 3D PPO 数学等价）、DAPO 模板、训练不用 correct/format reward。

## 四、关键证据

官方：`fsdp_workers.py L1874-1978`（only_stu）、`dp_actor.py L48-87`（student_p）、`core_algos.py L854-880/L1118-1144`（token_reward_direct/3D PPO）、`train_justrl_qwen.sh`（超参）、`prepare_skywork_math.py`（DAPO）、`ttrl_math/__init__.py`（boxed+sympy）
我们：`cache.py L67-83/L168-180`（交集）、`losses.py L15-93/L80-89`（pg_loss/renorm）、`scheduler.py L477-508`（base 稀疏 PG）、`qwen3_base_opd.yaml`（新对）

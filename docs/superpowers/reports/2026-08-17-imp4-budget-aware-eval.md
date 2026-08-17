# Stage 1.6/IMP-4 Budget-Aware Evaluation 进度报告（2026-08-17）

> 状态：评估性能 bug 已修复（批量生成），vLLM 加速集成代码已完成并单测通过；
> **GPU 实测被服务器不可达阻塞**（see `connect.westd.seetacloud.com:35318` 已重试 15+ 次）。
> 本报告记录已确认事实 + 待 GPU 验证清单，恢复后按清单执行即可。

## 1. 已确认（GPU/本地实测）

### 1.1 评估性能 bug（已修复，commit 18fc032）
- **根因**：BudgetEvaluator.evaluate_budget 逐条调 generate_budget([prompts[i]], budget)
  → atch_size 从未生效（每次 HF generate 只处理 1 条，巨大固定调用开销）。
- **验证数据**（速度探针，Qwen3-1.7B / cuda:0 / greedy）：
  - 批量 5 条 B256：4.9s → **264 tok/s**；B512：8.6s → **298 tok/s**。
- **修复**：n<=1（greedy）一次传全部 prompts 批量生成；n>1 保留逐条归组。
  - 效果：50 条 B256 从「13+ 分钟未完成」降到「~3-4 分钟」（batch=8，加载后），
    且 batch=50 实测出数（50 条一次生成）。
- **batch 上界的取舍**（实测）：152k 词表 lm_head 使 decode 每步计算量随 batch 线性增长，
  batch=50 每步 50×151936 logits → 每步慢 ~6×，相对 batch=8 只快 ~14%-2×（视 token 数）。
  **最优 batch≈8-16**；更大规模评估走 vLLM 连续批处理。

### 1.2 vLLM 环境（服务器最后可达时确认）
- vLLM **0.16.0**、torch 2.9.1+cu128、2×RTX PRO 6000 Blackwell（sm_120，**FP8 可用**）。
- 仓库已有 VLLMRolloutEngine（TP/FP8/PagedAttention/update_weights/response_dists 齐备），
  parse_vllm_outputs 已有单测（eos/budget_stop/loop/loop_disabled）。

## 2. 已完成（本地，单测通过）

| 改动 | commit | 测试 |
|---|---|---|
| evaluate_budget 批量修复 + n_limit；测试 fake 契约更新 | 18fc032 | 44 passed |
| pipeline vLLM rollout 集成（独立 rollout_device + 权重同步 + 跨卡 responses） | 18fc032 | 85+67 passed |
| budget_eval_real.py 双卡预算评估脚本 | 18fc032 | — |
| vllm_budget_eval.py（连续批处理 + FP8 + 投机解码） | 18fc032/0bc2ec3 | 5 passed |
| 提取 _aggregate_budget 纯函数 + 单测 | 0bc2ec3 | 5 passed |

## 3. 已拿到的初步 budget 数据（GPU 不可达前，n=50 MATH500 前 50 条）

| Model | Budget | acc | status | 备注 |
|---|---|---:|---|---|
| Base | 256 | 0.160 | 全 budget_stop（avg_rt=256） | 前 50 条子集 |
| E2(L2@1024) | 256 | 0.100 | 全 budget_stop | 26 步训练后 |

> 说明：样本少（n=50）+ 训练步数少（24-26 步），仅作方向参考；正式矩阵待 vLLM 评估补全
> B{256,512,1024} 全量 + Base/E1/E2 三模型 + 双卡并行。

## 4. 待 GPU 验证清单（服务器恢复后按序执行）

1. **vLLM 加载可行性**：LLM(model=Qwen3-1.7B, tensor_parallel_size=1, dtype=float8
   |auto, gpu_memory_utilization=0.9) + 批量生成 + LLM.update_weights（vLLM 0.16 API）。
2. **评估加速**：llm_budget_eval.py --device cuda:1 --models Base=...,E1=...,E2=...
   --budgets 256,512,1024 --dataset MATH500 --n-limit 50 --out-dir ... [--fp8] [--draft ...]；
   对比 HF 吞吐（264 tok/s 基线）。
   - 投机解码：先 ls /root/autodl-tmp/models 确认有无小模型作 draft（如 Qwen3-0.6B）。
3. **训练 rollout vLLM 端到端**：S2_E1/E2 配置 
ollout_engine: vllm,
   
ollout_device: cuda:1, 
ollout_model: /root/autodl-tmp/models/Qwen__Qwen3-1.7B,
   
ollout_dtype: fp8；验证 on-policy（update_weights 同步）+ 生成加速。
4. **IMP-4 正式矩阵**：Base/L0/L2 × B{256,512,1024} 曲线 + budget_curve 指标（AUC/nAUC/
   GainPerToken/ΔA）+ write_report 4 图。

## 5. 关键设计点（vLLM rollout 集成）
- 双卡分工：训练 cuda:0（student/teacher HF），rollout vLLM cuda:1（
ollout_device），
  避免 vLLM gpu_memory_utilization=0.9 挤掉训练显存。
- on-policy：每次 rollout 相位前 
ollout_engine.update_weights(student.state_dict())；
  toy（None）路径零回归。
- 跨卡：vLLM 生成 responses（cuda:1）在 run_refresh_phase 转回训练 device（cuda:0）。

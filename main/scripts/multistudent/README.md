# 多学生并发训练编排（最终方案落地）

把 1.7B / 4B / 7B 三个学生**同时**在 2×RTX PRO 6000（96GB×2 + NVLink）上训练，
并在训练期并行跑 AIME 评估，最大化 GPU 利用。方案依据
`docs/GPU_MEMORY_AND_PARALLEL_PLAN.md` §7（二次审阅定稿）。

## 核心机制

| 阶段 | 做法 | 复用 |
|---|---|---|
| Stage 0 | 教师对（预下载）只建一次语义上共享；**每学生缓存各建**（默认 `student_init`，胖 D 含各学生初始采样块 → Δ_T 缓存不共享，§7.0） | `opd cache` |
| Stage 1 | 3 份缓存**并行** build（7B→cuda:0，4B/1.7B→cuda:1） | `opd cache --out` |
| Stage 2 | 3 个训练**并发**（每学生一线程版调度器 + 单 `--device`，`load_cache=true` 跳过 Stage 0/1） | `opd train` |
| 评估 | checkpoint 快照 + 空闲卡后台 `opd eval-aime`（4B/1.7B 随气泡评；7B 留 checkpoint 边界，§7.3） | `opd eval-aime --run-dir` |

**显存打包（§7.2）**：rank0 = 7B（8-bit-Adam 76GB + scorer 15GB ≈ 91GB）；
rank1 = 4B（8-bit-Adam 39GB）+ 1.7B（fp32-Adam 28GB）≈ 78GB。两卡满载。

**关键开关**：`stage2.renormalize_topk_support: true`（稀疏支撑重归一化，对齐原始 Direct-OPD 条件期望，模块 1）。

## 用法

```bash
# 真实 HF + GPU（需已下载模型，改 students.env 的 *_PATH）
bash run_all.sh real

# 本地 toy + CPU 冒烟（验证编排逻辑：缓存并行建 + 训练并行跑 + run 目录产出）
bash run_all.sh smoke
```

smoke 前台等待训练结束（确定性验证）；real 后台启动 3 训练 + 评估 watcher 后返回。

## 文件

| 文件 | 作用 |
|---|---|
| `students.env` | 三档学生定义（模型路径 / device / 缓存 / run 目录 / batch）|
| `student_real.yaml` | 真实 HF 训练基准（bf16 / topk / 归一化开）|
| `student_smoke.yaml` | toy CPU 冒烟配置（小步数）|
| `run_all.sh` | 总编排（Phase 1 建缓存 → Phase 2 并发训练 → Phase 3 评估 watcher）|
| `train_one.sh` | 单学生训练（load_cache=true）|
| `eval_watch.sh` | 训练期评估 watcher |

## ⚠️ 需 GPU/真实模型验证（骨架标注）

- 模型接入：`model_factory.HFCausalLM`（transformers 适配器）——本地无法实测；
  HF 路径输入为已 tokenize 的等长 id 张量（无 padding mask）、未实现 KV cache。
- `students.env` 的模型路径是占位，按服务器实际下载改。
- 7B 训练期评估：rank0 已占 ~91GB，同卡 eval 会 OOM → 见 `eval_watch.sh` 头注释的
  checkpoint 边界策略。
- 8-bit-Adam：`train` 目前用 fp32 Adam（`scheduler.py:opt = torch.optim.Adam`）；
  7B 档需换 `adamw_8bit`（或 Megatron/FSDP 8-bit 优化器）才装进 96GB——这是代码层
  剩余骨架缺口，落地时按 `docs/GPU_MEMORY_AND_PARALLEL_PLAN.md` §7.4 补。

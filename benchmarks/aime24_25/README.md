# AIME24/25 蒸馏效果基准（异步 + 预加载教师 + 弱到强蒸馏）

验证三组「弱教师 → 强学生」蒸馏在 AIME24/AIME25 上的效果：**教师自身起点**、
**学生蒸馏前**、**学生蒸馏后**。**main/ 自包含**（`opd eval-aime`），不依赖 async-opd。

| 组 | 教师（弱） | 学生（强，被蒸馏） |
|---|---|---|
| 1 | JustRL-1.5B（π_RL） | Qwen3-1.7B |
| 2 | JustRL-1.5B（π_RL） | Qwen3-4B |
| 3 | JustRL-1.5B（π_RL） | R1-Distill-7B |

## 评估后端

`python -m fullstack_opd_v2 eval-aime`（`main/fullstack_opd_v2/eval_aime.py`）：
- 模型：transformers（本地路径 / HF id）
- 数据：`AIME24`=`Maxwell-Jia/AIME_2024`、`AIME25`=`yentinglin/aime_2025`
- 提示：`{problem}\n...answer within \boxed{}.`；答案提取 `\boxed{}` → 整数；精确匹配

## 跑法

```bash
cd benchmarks/aime24_25
vi models.env                      # 填/覆写模型路径（服务器上指向 /root/autodl-tmp/models/）

bash run_benchmark.sh teacher          # (1) 教师 JustRL-1.5B 的 AIME24/25 起点
bash run_benchmark.sh student_baseline # (2) 三组学生蒸馏前 AIME24/25
bash run_benchmark.sh all              # (1)+(2)+汇总表
bash run_benchmark.sh aggregate        # 只汇总已有结果

# 蒸馏后学生（HF checkpoint 或 run 目录）：
bash watch_student.sh <1|2|3> <checkpoint模型目录或run-dir>
```

## 输出

- 每样本 jsonl：`results/teacher/`、`results/student_baseline/<combo>/`、`results/student_post/<combo>/`（`AIME24.jsonl` / `AIME25.jsonl`）
- 汇总表（`aggregate.py`，含 Δ = 蒸馏后 − 蒸馏前）：

```
阶段          模型             AIME24            AIME25            ΔAIME24   ΔAIME25
教师基线      JustRL-1.5B      12.50% (3/24)     8.33% (2/24)
学生 蒸馏前   组1 Qwen3-1.7B   20.83% (5/24)     16.67% (4/24)
学生 蒸馏后   组1 Qwen3-1.7B@step_100  25.00% (6/24)  20.83% (5/24)  +4.17    +4.16
```

## run 目录桥接

真实蒸馏训练产出的 run 目录若在 `config.yaml` 配了 `eval.model_path`（真实 HF 模型路径），
`opd eval-aime --run-dir <dir>` 会读它评估——toy run 目录（main/ v2 无 model_path）会报
`DataError` 明确提示。

## 说明

- **组3 学生基座** = `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`（论文的 "R1-Distill-7B"）；
  `models.env` 默认给的是论文参考 post 模型 `JustRL-R1-7B`，服务器上请改为基座。
- AIME 数据走 huggingface datasets（服务器 `source /etc/network_turbo` 加速）。
- 答案提取/评分在 `main/fullstack_opd_v2/eval_aime.py`（纯函数可单测）。
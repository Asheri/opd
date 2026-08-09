# AIME24/25 蒸馏效果基准（异步 + 预加载教师 + 弱到强蒸馏）

验证三组「弱教师 → 强学生」蒸馏在 AIME24/AIME25 上的效果，测三个量：
**教师自身起点**、**学生蒸馏前**、**学生蒸馏后**。

| 组 | 教师（弱） | 学生（强，被蒸馏） |
|---|---|---|
| 1 | JustRL-1.5B | Qwen3-1.7B |
| 2 | JustRL-1.5B | Qwen3-4B |
| 3 | JustRL-1.5B | R1-Distill-7B |

## 前置（一次性）

在 AutoDL 服务器上：

```bash
source /etc/network_turbo          # 学术加速（HF 下载快）
conda activate <你的 opd 训练环境>   # 含 async-opd（pip install -e ./async-opd）
# 填一次模型 ID：
vi benchmarks/aime24_25/models.env  # TEACHER_PATH=JustRL-1.5B 的 HF 模型 ID（必填）
```

> ⚠️ **`models.env` 里的 `TEACHER_PATH` 是占位符**，我无法联网查证 JustRL-1.5B 的准确 HF ID，
> 请改成真实 ID（如 `org/JustRL-1.5B`）。学生 Qwen 系 ID 已按标准填写。

## 跑法

```bash
cd benchmarks/aime24_25

bash run_benchmark.sh teacher          # (1) 教师 JustRL-1.5B 的 AIME24/25 起点（跑一次）
bash run_benchmark.sh student_baseline # (2) 三组学生蒸馏前 AIME24/25
bash run_benchmark.sh all              # (1)+(2)+汇总表
bash run_benchmark.sh aggregate        # 只汇总已有结果

# 蒸馏进行中/完成后，watch 学生 checkpoint 逐个评 AIME24/25：
bash watch_student.sh <1|2|3> <蒸馏训练run-dir> [watch-timeout分钟]
```

## 输出

- 每样本 jsonl：`results/teacher/`、`results/student_baseline/<combo>/`、`results/student_post/<combo>/`
- 汇总表（`aggregate.py`）：

```
阶段          模型             AIME24            AIME25            ΔAIME24   ΔAIME25
教师基线      JustRL-1.5B      12.50% (3/24)     8.33% (2/24)
学生 蒸馏前   组1 Qwen3-1.7B   20.83% (5/24)     16.67% (4/24)
学生 蒸馏后   组1 Qwen3-1.7B@step_100  25.00% (6/24)     20.83% (5/24)   +4.17    +4.16
```

## 关键参数（`models.env`）

| 变量 | 意义 | 默认 |
|---|---|---|
| `TEACHER_PATH` | 教师 HF ID（**必填**） | 占位符 |
| `STUDENT_COMBO1/2/3` | 学生 HF ID | Qwen3-1.7B/4B/R1-Distill-7B |
| `AIME24` / `AIME25` | 数据集 | `hf:Maxwell-Jia/AIME_2024` / `hf:yentinglin/aime_2025` |
| `EVAL_N_SAMPLES` | 1=greedy；Avg@32 更稳但慢 | 1 |
| `EVAL_TEMP` | 采样温度 | 0.0 |

## 说明

- **7B 学生（组3）** 用 `--gpus 0,1 --tp 2`（2 卡），1.7B/4B 用单卡。
- **推理模型**（R1-Distill）若需 `thinking` 标签，加 `--enable-thinking` 或改 `models.env` 外的
  `run_benchmark.sh` 里 eval 调用。
- **蒸馏后 checkpoint** 由 async-opd 蒸馏训练产出（`<run-dir>/checkpoints/step_NNN/`），
  `watch_student.sh` 用 `--watch` 自动发现；`--output-dir` 指向本目录便于汇总。
- AIME 答案提取用 eval CLI 内置的 `\boxed{}` 级联 + 数值匹配（`opd.utils.eval`），无需自写。
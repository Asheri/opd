# 2026-08-26：服务器恢复后执行清单——Chat 模板重测三模型（文档一）+ H9 预算错位验证（文档二）

> **状态：服务器恢复后执行**（服务器已关闭、SSH 暂缓；本清单为恢复后唯一执行依据，零决策、按序门控）。
>
> 前置事实（本地已完成的实现与审计，勿在服务器重做）：
> - vllm_budget_eval 已加 `--chat-template`/`--tokenizer` + `build_prompts` 纯函数（parse_args 支持 argv 注入）+ **`--device`→`CUDA_VISIBLE_DEVICES` 选卡修复**（原缺陷：--device 仅打印不生效，双卡并行会抢卡）；budget_eval 已加 `wrap_chat`；单测 **16 例通过**（CLI 默认零回归 / tokenizer 覆盖 / chat 顺序 / eos·budget_stop 与 no_answer 解耦对照 / CUDA_VISIBLE_DEVICES 映射）。
> - export_student_ckpt.py 支持任意中间 step 手动导出（纯路径驱动、服务器可直接用），批量导出为可选增强（本清单用三次单步导出即可，不做）。
> - prepare_skywork_responses.py 500→2000 补生成无需任何代码改动（`--max-samples 1500 --seed S --apply-chat-template` 即可），重建 cache 自动按 jsonl 非空行数走。
> - 全量回归 532 passed（本地）。
>
> 硬约束（全程适用）：每步判据写死、GPU≥2 必须双卡并行分配、**不伪造结果**（数字必须来自真实运行输出）、任何报错追加 `C:\Users\12062\OneDrive\Desktop\items\training-errors.md`（本机本地文件）。

---

## 0. 同步代码（本地 → 服务器）+ 服务器回归

### 0.1 本轮改动文件（务必全部同步）

| 本地文件 | 服务器目标 |
|---|---|
| `main/fullstack_opd_v2/budget_eval.py`（新增 `wrap_chat` 纯函数 + `__all__`） | `/root/opd/main/fullstack_opd_v2/budget_eval.py` |
| `main/scripts/vllm_budget_eval.py`（新增 `--chat-template`/`--tokenizer`/`build_prompts`） | `/root/opd/main/scripts/vllm_budget_eval.py` |
| `main/tests/test_vllm_budget_eval.py`（14 例） | `/root/opd/main/tests/test_vllm_budget_eval.py` |

```bash
# 本地（在 C:\Users\12062\OneDrive\Desktop\opd 下执行）：
scp -P 35318 main/fullstack_opd_v2/budget_eval.py \
  root@connect.westd.seetacloud.com:/root/opd/main/fullstack_opd_v2/
scp -P 35318 main/scripts/vllm_budget_eval.py \
  root@connect.westd.seetacloud.com:/root/opd/main/scripts/
scp -P 35318 main/tests/test_vllm_budget_eval.py \
  root@connect.westd.seetacloud.com:/root/opd/main/tests/
```

> 注：`main/scripts/run_s2_real.py` 的 `--resume` 注册修复（把 `return p.parse_args()` 移到 `add_argument` 之后，否则续跑报未知参数）**已随 053ec56 提交**——同步走 `git push` + 服务器 `git pull` 即可，无需单独 scp。

### 0.2 服务器全量回归

```bash
cd /root/opd/main
/root/miniconda3/bin/python -m pytest tests/ -q | tail -1
# 期望：532 passed（全量）；至少确认本清单相关文件：
/root/miniconda3/bin/python -m pytest tests/test_vllm_budget_eval.py -q | tail -1
# 期望：16 passed（test_wrap_chat_format / test_build_prompts_bare / test_build_prompts_chat / test_aggregate_* / test_apply_cuda_visible_*）
```

- **判据（写死）**：全量 `532 passed, 0 failed` 且 `test_vllm_budget_eval.py` 16 例全过 → 才允许进入 Step 1；任一失败先修后进，失败记录追加 training-errors.md。

---

---

# 0.5 优化调度 v2（2026-08-26 workflow 审查结论，替代上文 0-3 的串行编排）

> 7-agent workflow 审查（扫描 vllm_budget_eval/eval-aime/导出补生成 + 设计 + 对抗验证）。判据与协议不变，只优化"怎么跑"。服务器硬件：2×RTX PRO 6000 96GB×2（本地文档，恢复后 nvidia-smi 核验）。

## 0.5.1 前置修复（已 commit，本地完成）

- **致命 FAIL 已修**：`vllm_budget_eval.py` 原 `--device` 不传给 vLLM 引擎（仅打印），双卡并行会抢同一默认卡。已加 `_apply_cuda_visible`（`--device cuda:i` → `CUDA_VISIBLE_DEVICES=i`，import vllm 前生效）+ 2 单测（共 16 例）。**服务器同步走 git pull 即可。**

## 0.5.2 实验设计补充（判据外，建议必做）

- **D1（必改）**：补跑 Base/E1/E2 的 **B512 chat** 重测（原 3b 只跑 E2 三 step）——H9"截断假象"叙事不能混入裸→chat 模板变化，需"三模型×{B512,B2048} 全 chat"网格才可辩护。M1 一次调用顺手完成（+0.75h）。
- **D2（强烈建议）**：Base 的 **B1024** 对照——验证"Base 也随预算升"（截断曲线 vs 两档跳跃），合并进 M3（+0.5h）。

## 0.5.3 Phase 编排（双卡满载）

| Phase | GPU0（vLLM 轨道） | GPU1（eval-aime 轨道） | CPU/其他 | 墙钟 | 门控 |
|---|---|---|---|---|---|
| P0 同步回归 | — | — | git pull + pytest 全量 532 + chat 16 例 | ~10 min | 通过才进 P1 |
| P1 首验+导出 | 冒烟：Base B512 n=3 chat（~3min） | 空闲（首验安全闸保留） | 并行 CPU 导出 E2 step120/200/311 + 确认 E1 step311/Base 目录存在 | ~15 min | 冒烟无 loop + 导出完整 |
| P2 主战役 | ① B2048 `Base+E2`（2h）→② **B512 六模型一次调用**（1.5h）→③ B1024 扫描（判定后，1.5-2h） | ① E1 B2048（1h）→② AIME24 队列：Base→S120→S200→S311→E1→E2（3-6h） | 拐点表 CPU 聚合 | 5.5-7h | **判定点 @2h**；B1024 门控于判定∈{确诊,部分成立} |
| P3 分流 | 确诊→进 Step 1 + 提前 Step 2 KL 训练；排除→停下游回查训练 | KL 训练 / 数据扩展补生成双卡 shard | cache 重建独占一卡（或 CPU 随时） | 小时级 | Step 0 判定 + 拐点表 |

## 0.5.4 命令级合并（M1-M6）

| # | 合并 | 命令要点 |
|---|---|---|
| M1 | B512 六模型一次调用（含 D1 补测） | `vllm_budget_eval.py --models "Base=...,E1=...,E2=...,S120=...,S200=...,S311=..." --budgets 512 --dataset MATH500 --chat-template --device cuda:0 --out-dir <dir>/B512` |
| M2 | B2048 拆 2+1 | GPU0 `--models "Base=...,E2=..." --budgets 2048 --device cuda:0`；GPU1 `--models "E1=..." --budgets 2048 --device cuda:1` |
| M3 | B1024 判定后一次跑（含 D2 Base） | `--models "S120=...,S200=...,S311=...,Base=..." --budgets 1024 --device cuda:0` |
| M4 | AIME24 队列入 GPU1 | 6 次 `eval-aime --model <p> --datasets AIME24 --max-new-tokens 4096 --n-samples 1 --temperature 0.0 --scoring sympy --chat-template --device cuda:1 --batch-size 2 --out <dir>/aime_eval_chat/<label>`（顺序排队，同 --out 自动续跑） |
| M5 | 导出提前 P1 与冒烟并行 | `export_student_ckpt.py --ckpt <run>/checkpoints/step_<N>.pt --model Qwen__Qwen3-1.7B --out <dir>/models/student_e2_step<N>` ×3（纯 CPU） |
| M6 | vLLM 断点续跑薄包装 | `nohup ... > log 2>&1 &`；完成判据=对应 jsonl 存在且 500 行；崩了只补缺失 (label,budget)（缩小 --models/--budgets 重跑） |

> GPU1 AIME24 顺序取 Base→S120→S200→S311→E1→E2：满足 Step 2 首验门控 + 让拐点表/KL 门控尽早（~4h）；E2 的 AIME24 验收（总验收 #2）延后到 ~6-7h。

## 0.5.5 风险与降级

1. **显存**：vLLM `--gpu-mem 0.9` 预占 86GB，**绝不与任何 GPU 任务共卡**（eval-aime 走 GPU1 异卡已规避）；OOM 降 `--gpu-mem 0.8` 或 `--max-model-len 6144`。
2. **失败续跑**：eval-aime 同 --out 自动续；vLLM 按 M6 补缺口（all_results.json 缺失用各 jsonl 现场聚合）；prepare 有 --resume+同 seed；export/cache 幂等。
3. **双卡不均衡**：任卡空闲取"最长未启动任务"；优先级 = B2048 未完成 → AIME24 未跑 → B512 → B1024(判定后)；vLLM 调用可按模型粒度拆分迁移。
4. **B4096 应急**（任一模型 no_answer>3%）：抢占 vLLM 队列，GPU0=Base+E2、GPU1=E1（复用 M2 分法），判定重走表。
5. **硬件核验**：恢复后先 `nvidia-smi`；若 <96GB 或单卡，P2 退化为 GPU0 串行 + GPU1 轻载（AIME24），B1024/Base-B1024 降级为可选。

## 0.5.6 墙钟对比（依据：~280 tok/s、AIME24 20-60min/模型）

- 直接压缩：慢锚点同证据集 10.2h → **7.0h（-31%）**；快锚点 6.6h → **5.5h（-16%）**。
- **决策延迟（真正杠杆）**：H9 判定 ~7.4h → **~2h**（3.7×）；拐点表/KL 门控 ~8.9h → **~4-5h**（2×）——下游 KL 训练/补生成/cache 重建整条链级联提前。

---

# 文档一：Chat 模板重测三模型（P0）

> 背景：eval-aime 默认 `chat_template=False`（裸 prompt），Base 循环退化、旧 AIME24 结果作废；MATH500 B512 的 vllm_budget_eval 此前也是裸 prompt（现补 `--chat-template` 重测）。训练事实：`apply_chat_template=true`、`eos=151645`、chat 校准 0/100 loop、`repetition_penalty=1.0`。

**模型路径约定（2026-08-26 服务器实测修正）**：
- Base = `/root/autodl-tmp/models/Qwen__Qwen3-1.7B`（存在 ✅）
- E1 = `/root/autodl-tmp/exported/e1_s300`（存在 ✅，即 E1 最终导出，对应清单旧写 `student_e1_step311`）
- E2 = `/root/autodl-tmp/exported/e2_s311`（存在 ✅，E2 最终导出，对应清单旧写 `student_e2_step311`）
- ⏳ **E2 中间 checkpoint step_120/200 拷贝进行中**（2026-08-26 用户确认：数据正从旧实例拷贝；当前 `models/student_17b_ms_step120` 为空壳、全盘无 `step_*.pt` 是拷贝未完成所致，**非丢失**）——拐点扫描排到数据同步完成后确认；在 checkpoint 到位前先跑不依赖它的评估（B2048/AIME24/B512），拐点表 120/200 待数据到位后补
- 骨架：`--model /root/autodl-tmp/models/Qwen__Qwen3-1.7B`（服务器 HF 缓存已有该 id，给全路径最稳）

## 文档一 Step 1：MATH500 脚本 chat 支持确认（5 分钟）

**1a. CLI 参数确认**：

```bash
cd /root/opd/main
/root/miniconda3/bin/python scripts/vllm_budget_eval.py --help | grep -A2 -- "--chat-template"
/root/miniconda3/bin/python scripts/vllm_budget_eval.py --help | grep -A2 -- "--tokenizer"
```

- 判据：`--chat-template`（store_true，默认 False 零回归）与 `--tokenizer`（默认取各模型自身路径）均出现在 help 中。

**1b. 首验门控——Base 抽 3 条生成验证无 loop**：

```bash
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "Base=/root/autodl-tmp/models/Qwen__Qwen3-1.7B" \
  --budgets 512 --dataset MATH500 --n-limit 3 --chat-template \
  --device cuda:0 --out-dir /root/autodl-tmp/chat_retest/smoke
```

- 判据（写死）：日志出现 `chat template 启用（tokenizer=/root/autodl-tmp/models/Qwen__Qwen3-1.7B），对齐训练 apply_chat_template=true`；3 条生成文本为正常中文/数学推理（含 `\boxed{}` 或可判答案），**无 token soup、无无限重复刷屏（loop）**；`Base__MATH500__B512.jsonl` 中 3 行均落盘。
- 通过 → 进 Step 2；失败（乱码/loop）→ 停下查 C3 模板一致性与 tokenizer 路径，报错记录 training-errors.md，不继续。

## 文档一 Step 2：AIME24 chat 模板重测（双卡并行）

**协议（与训练对齐）**：`--chat-template`、`--max-new-tokens 4096`（覆盖训练响应长度）、`--n-samples 1`、`--temperature 0.0`、`--scoring sympy`、`--batch-size 2`（长生成控显存）。

**GPU 分配（GPU≥2 硬约束）**：GPU0 = Base、GPU1 = E2（第一轮并行）；E1 放第二轮（Base 已完成释放 GPU0 → GPU0 = E1，GPU1 空闲可并行跑 Step 3 的导出/评估）。

```bash
cd /root/opd/main
# ===== 第一轮：GPU0=Base、GPU1=E2 并行 =====
/root/miniconda3/bin/python -m fullstack_opd_v2 eval-aime \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --datasets AIME24 --max-new-tokens 4096 --n-samples 1 \
  --temperature 0.0 --scoring sympy --chat-template \
  --device cuda:0 --batch-size 2 \
  --out /root/autodl-tmp/aime_eval_chat/Base &
/root/miniconda3/bin/python -m fullstack_opd_v2 eval-aime \
  --model /root/autodl-tmp/models/student_e2_step311 \
  --datasets AIME24 --max-new-tokens 4096 --n-samples 1 \
  --temperature 0.0 --scoring sympy --chat-template \
  --device cuda:1 --batch-size 2 \
  --out /root/autodl-tmp/aime_eval_chat/E2 &
wait

# ===== 第二轮：GPU0=E1（另一卡空闲）=====
/root/miniconda3/bin/python -m fullstack_opd_v2 eval-aime \
  --model /root/autodl-tmp/models/student_e1_step311 \
  --datasets AIME24 --max-new-tokens 4096 --n-samples 1 \
  --temperature 0.0 --scoring sympy --chat-template \
  --device cuda:0 --batch-size 2 \
  --out /root/autodl-tmp/aime_eval_chat/E1 &
```

**首验门控（写死）**：先只启动 Base（GPU0）首 3 题生成——日志前 3 条输出无 loop（正常推理、非重复刷屏）才允许启动 E2/E1 全量；有 loop 立即中断并回查模板。eval-aime 逐题落盘 + resume（同 `--out` 重跑自动续跑），中断不丢数据。

- 判据（写死）：三模型均跑完 AIME24 全 30 题，输出 pass@1（chat 协议）；记录到报告，**不与旧裸 prompt 结果混比**。

## 文档一 Step 3：中间 checkpoint 导出 + 三档跑 chat（拐点表）

**3a. 导出 E2 中间 checkpoint → HF 目录**（`export_student_ckpt.py` 纯路径驱动，服务器直接可用；每次单步导出、骨架重复加载属预期，不做批量增强）：

```bash
cd /root/opd/main
# <E2_RUN_DIR> = E2 正式训练 run 目录（服务器实际路径，如 /root/autodl-tmp/runs_s2_e2_final）
/root/miniconda3/bin/python scripts/export_student_ckpt.py \
  --ckpt /root/autodl-tmp/<E2_RUN_DIR>/checkpoints/step_120.pt \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --out /root/autodl-tmp/models/student_e2_step120
# 同法导出：
#   step_200.pt  → /root/autodl-tmp/models/student_e2_step200
#   step_311.pt  → /root/autodl-tmp/models/student_e2_step311   （最终步）
```

- 判据：每个 `--out` 目录含 `config.json` + 权重 + `tokenizer*`（HF 完整目录）；日志 `step=N` 与预期一致、`已导出 ...`。

**3b. 三档跑 chat AIME24 + MATH500 B512**（三个 step × 两种评估，按双卡轮转；每卡同时只跑一个模型）：

- AIME24 chat：命令同 Step 2 协议（`--model <student_e2_stepXXX>`、`--out /root/autodl-tmp/aime_eval_chat/E2_step<XXX>`）
- MATH500 B512 chat：

```bash
# GPU0：step_120；GPU1：step_200（并行），随后 step_311 再占空闲卡
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "S120=/root/autodl-tmp/models/student_e2_step120" \
  --budgets 512 --dataset MATH500 --chat-template \
  --device cuda:0 --out-dir /root/autodl-tmp/chat_retest/E2_steps &
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "S200=/root/autodl-tmp/models/student_e2_step200" \
  --budgets 512 --dataset MATH500 --chat-template \
  --device cuda:1 --out-dir /root/autodl-tmp/chat_retest/E2_steps &
wait
# 再跑 step_311（GPU0 空闲后）
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "S311=/root/autodl-tmp/models/student_e2_step311" \
  --budgets 512 --dataset MATH500 --chat-template \
  --device cuda:0 --out-dir /root/autodl-tmp/chat_retest/E2_steps &
```

**3c. 产出拐点表（step vs acc vs KL）**：

| step | KL（训练 metrics 对应 step 的 kl 锚点） | AIME24 acc (chat) | MATH500 B512 acc (chat) | no_answer% | eos_rate |
|---|---|---|---|---|---|
| 120 | 取自 run metrics csv | 实跑值 | 实跑值 | 实跑值 | 实跑值 |
| 200 | 同上 | 实跑值 | 实跑值 | 实跑值 | 实跑值 |
| 311 | 同上 | 实跑值 | 实跑值 | 实跑值 | 实跑值 |

- 判据：表内所有数字来自真实运行输出（jsonl/日志），KL 无该字段则标 N/A（不算伪造）；标注 acc 峰值 step（拐点）。

---

# 文档二：H9 预算错位验证

> 背景：MATH500 B512 下 E1=0.186/E2=0.236 < Base=0.344 疑为截断假象（训练 response T=2048，B512 截断深推理、no_answer 6-8%、eos_rate=0）。B2048（对齐训练分布）验证 E2 ≥ Base 则 OPD 复现成功。

## 文档二 Step 0：MATH500 B2048 决定性实验（P0）

**GPU 分配**：GPU0 = Base + E2（脚本内一个模型接一个模型顺序生成，卡内串行但两模型一次调用）；GPU1 = E1。

```bash
cd /root/opd/main
# GPU0：Base、E2
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "Base=/root/autodl-tmp/models/Qwen__Qwen3-1.7B,E2=/root/autodl-tmp/models/student_e2_step311" \
  --budgets 2048 --dataset MATH500 --n-limit 500 --chat-template \
  --device cuda:0 --out-dir /root/autodl-tmp/chat_retest/B2048 &
# GPU1：E1
/root/miniconda3/bin/python scripts/vllm_budget_eval.py \
  --models "E1=/root/autodl-tmp/models/student_e1_step311" \
  --budgets 2048 --dataset MATH500 --n-limit 500 --chat-template \
  --device cuda:1 --out-dir /root/autodl-tmp/chat_retest/B2048 &
wait
```

- 判据：三模型日志均出现 `chat template 启用`；`Base__MATH500__B2048.jsonl` / `E1__MATH500__B2048.jsonl` / `E2__MATH500__B2048.jsonl` + `all_results.json` 落盘；对比指标（accuracy / no_answer_rate / eos_rate / avg_reasoning_tokens）。

**H9 判定表（写死，按此分流）**：

> **2026-08-26 已执行（用户提供）：Base − E2 = 0.116 ≥ 0.05 -> 「排除」行命中。**
> no_answer 仅 1.2%（截断假象不成立）。按该行原定动作"停下，回查训练/信号"--
> 归因分析与后续方案已写入 `docs/reports/2026-08-26-opd-failure-analysis.md`，
> **下方文档二 Step 1-4 原门控全部冻结**，改按该文档 §5（E-0 拐点扫描 ->
> E-1 教师对体检 + Δ↔correctness 相关性 -> 分支 A on-policy 化 / B1 换教师对 /
> B2 信号改造）执行。

| 条件 | 判定 | 后续动作 |
|---|---|---|
| `E2_acc ≥ Base_acc` 且 `E2_no_answer ≤ 3%` | **确诊**：B512 为预算错位假象，OPD 复现成功 | 进文档二 Step 1（拐点稳健性） |
| `0 ≤ Base_acc − E2_acc < 0.05` | **部分成立**：错位解释大部分，残差待查 | 进 Step 1 + Step 2（KL 档位） |
| `Base_acc − E2_acc ≥ 0.05` | **排除**：H9 不成立，E2 真弱于 Base | 停下，回查训练/信号，记录报告 |
| 任一模型 `no_answer > 3%` | 深度推理仍被截断，**补 B4096** 再判 | 同命令 `--budgets 4096` 重跑后回判定表 |

## 文档二 Step 1：拐点扫描（门控：Step 0 确诊或部分成立）

- 目的：确认 step vs acc 拐点在双预算下稳健（B512/B1024 chat 均扫 E2 的 step_120/200/311）。
- 命令要点：复用 Step 3b 命令，`--budgets 512,1024` 两档一次跑；AIME24 复用 Step 2 协议。
- 判据：三档（120/200/311）× 双预算 acc 单调或峰值一致 → 记录拐点 step 作为后续导出基准；不一致则如实记录、不选优。

## 文档二 Step 2：KL 档位（门控：Step 0 部分成立且拐点表产出后）

- 目的：残差归因 KL 压制（D2 已证 kl=0.5/0.1 不升、kl=0.02 显著升）。
- 命令要点：新训练用 `--set stage2.kl_reg_coef=0.02`（D2 通过档），正式训练命令同 E2 300 步口径（`run_s2_real.py --config configs/skywork_17b.yaml` + `--set dataset.apply_chat_template=true --set dataset.max_response_len=2048` + `--set l2.rollout.repetition_penalty=1.0` + `--eos-id 151645`），训练后再跑 Step 2/Step 0 同协议评估。
- 判据：新模型固定集 eval_reward 相对 Base 转正（提升 ≥ +0.05），且 MATH500 B2048 chat 下 E2 ≥ Base → 通过。

## 文档二 Step 3：数据扩展 500 → 2000（门控：Step 2 通过后需要更大数据时）

**补生成（双卡分片并行，同 seed 保证跨 shard/resume 确定）**：

```bash
# 备份原 jsonl（500 行有 response）
cp /root/autodl-tmp/datasets/skywork_50k.jsonl{,.bak}
# GPU0：shard 0；GPU1：shard 1（并集 = 1500 条新行，与已有 500 不重叠）
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_50k.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --apply-chat-template --max-samples 1500 --seed 42 \
  --device cuda:0 --batch-size 8 --max-new-tokens 2048 \
  --temperature 1.0 --shard-rank 0 --num-shards 2 --resume &
/root/miniconda3/bin/python scripts/prepare_skywork_responses.py \
  --jsonl /root/autodl-tmp/datasets/skywork_50k.jsonl \
  --model /root/autodl-tmp/models/Qwen__Qwen3-1.7B \
  --apply-chat-template --max-samples 1500 --seed 42 \
  --device cuda:1 --batch-size 8 --max-new-tokens 2048 \
  --temperature 1.0 --shard-rank 1 --num-shards 2 --resume &
wait
```

> 用法铁律（审计结论，写死）：**必须** `--apply-chat-template`（否则新 1500 条分布与旧 500 不一致，C3 根因）；**必须** `--max-samples 1500 --seed <S>`（否则会全量生成 49,500 空行）；`--max-samples` 语义是随机抽样（非"前 N 条"）；resume 期间保持同 seed。

**重建 cache（与原始 cache 同数据配置，C2 守卫防错配）**：

```bash
cd /root/opd/main
/root/miniconda3/bin/python -m fullstack_opd_v2 cache \
  --config configs/skywork_17b.yaml \
  --set dataset.apply_chat_template=true \
  --set dataset.max_response_len=2048 \
  --set stage1.load_cache=false \
  --device cuda:0 --out /root/autodl-tmp/cache_skywork_chat_2000.pt
```

- 判据：jsonl 中 response 非空行数 = 2000；cache metadata `prompt_format=chat`、T=2048（与原始 cache 对齐，`verify_consistency` fail-fast 守卫）；抽样 decode 2-3 条无乱码；cache 规模自动按 2000 走，无需改代码/传规模参数（`--materialized` 仅为声明可选）。

## 文档二 Step 4：信号改造（门控：Step 2/3 数据就绪后，视残差方向定）

- 目的：按 Step 0-2 残差方向调整训练信号（如 Δ_T 阈值、refresh 参数），命令 = `run_s2_real.py` 正式训练命令（同 Step 2 要点，可用 `--resume` 续跑；⚠️ 需已同步 0.1 注明的 run_s2_real.py 一行修复）。
- 判据：新训练固定集 eval_reward 转正（≥ +0.05），且复测 AIME24 chat / MATH500 B2048 chat 下 E2 ≥ Base → OPD 复现成功，收尾。

---

# 总验收表（文档一 + 文档二全部完成后逐项打勾）

| # | 验收项 | 判据（写死） | 状态 |
|---|---|---|---|
| 1 | 协议统一 | 文档一/二所有评估均 `--chat-template`（eval-aime 与 vllm_budget_eval 日志均确认 chat 启用），旧裸 prompt 结果全部作废标记 | ☐ |
| 2 | AIME24 E2 ≥ Base | 文档一 Step 2：`E2_acc ≥ Base_acc`（同 chat 协议 pass@1） | ☐ |
| 3 | MATH500 B2048 E2 ≥ Base | 文档二 Step 0：`E2_acc ≥ Base_acc` 且 `no_answer ≤ 3%` | ☐ |
| 4 | 拐点表 | 文档一 Step 3 产出 step vs acc vs KL 表（120/200/311），标明峰值 step | ☐ |
| 5 | 产物入库 | 每个评估的 jsonl（每行含 response / final_answer / ground_truth / status）+ 每模型 3 条 decode 样本附入报告 | ☐ |

---

# 执行纪律（贯穿全清单）

1. **判据写死**：每步"判据"栏达标才算完成；不达标不进入下一步，不静默跳过。
2. **GPU≥2 双卡并行**：每个多任务步骤按上文 GPU 分配并行（Step 2 双卡两模型、Step 0 双卡分模型、Step 3 双卡分片），脚本均显式 `--device cuda:i`，禁止串行排空。
3. **不伪造结果**：所有数字来自真实运行输出（jsonl / 日志 / metrics csv）；缺数据标注 N/A。
4. **报错档案**：服务器端任何训练/评估报错（含堆栈片段、根因、修复、验证）追加到本地 `C:\Users\12062\OneDrive\Desktop\items\training-errors.md`，完成后回复用户"已追加到 training-errors.md"。
5. **报告更新**：每步完成后把实跑数字回填到对应报告（文档一/文档二结论、与旧 Base=0.344 / E1=0.186 / E2=0.236 B512 数字对比），不改结论只填证据。

---

# E 系列判别实验（归因分析 §5，2026-08-26 新增）

> 依据：`docs/reports/2026-08-26-opd-failure-analysis.md`。B2048 后 H9 排除 → 按判定表"停下回查训练/信号" = 本系列。
> 已本地实现的脚本（commit `81b227e`，23 单测全过，服务器 git pull 即得）：
> - `scripts/delta_correctness_corr.py`（E-1b 决定性，三阶段 sample/logp/correlate 双卡流水）
> - `scripts/export_sanity_check.py`（E-0b config diff + HF 冒烟）
> - `scripts/onpolicy_share.py`（E-0d metrics pool 占比）

## GPU 优化编排（双卡满载，替代顺序执行）

| 步骤 | GPU0 | GPU1 | 命令要点 |
|---|---|---|---|
| E-0b 导出健全性 | —（CPU） | — | `export_sanity_check.py --exported <dir> --reference /root/autodl-tmp/models/Qwen__Qwen3-1.7B`（可随时穿插） |
| E-0c 拐点扫描 | vLLM：S120/S200/S311（+E1 序列若有）一次调用 | 空闲→接 E-1a | `vllm_budget_eval.py --models "S120=...,S200=...,S311=..." --budgets 2048 --chat-template --device cuda:0`（等 step_120/200 拷贝到位） |
| E-1a 教师对体检 | — | vLLM：JustRL+R1Distill 一次调用 | `vllm_budget_eval.py --models "JustRL=...,R1Distill=..." --budgets 2048 --n-limit 500 --chat-template --device cuda:1`（与 E-0c 同窗并行） |
| E-0d on-policy 占比 | —（CPU） | — | `onpolicy_share.py --run-dir <E2 run 目录>` |
| E-1b 采样 | vLLM sample（200×4=800 条） | — | `delta_correctness_corr.py --stage sample --dataset MATH500 --n-problems 200 --n-samples 4 --budget 2048 --temperature 1.0 --chat-template --device cuda:0 --out <dir>` |
| E-1b logp | 教师 rl forward | 教师 ref forward | `--stage logp --teacher rl --device cuda:0` / `--teacher ref --device cuda:1`（双卡并行） |
| E-1b correlate | —（CPU） | — | `--stage correlate --out <dir>` → report.json |

**判据（写死，归因分析 §5）**：
- E-0c：step_20/60 已 ≤ Base−0.05 → RC1/RC4 主导；≈Base 且随 step 单调劣化 → RC3；中途峰值 → 混合
- E-1a：`teacher_rl < teacher_ref` → Δ_T 方向存疑 → RC4 加权 + 查 JustRL 论文 RL 目标
- E-1b：Spearman ρ ≥ 0.2 → **分支 A**（on-policy 化）；0.05 ≤ ρ < 0.2 → **分支 B2**（信号改造）；ρ < 0.05 → **分支 B1**（换教师对 Qwen3 同族）

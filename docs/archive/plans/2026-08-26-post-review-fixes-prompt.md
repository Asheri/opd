# 提示词：实验代码审阅修复（R1 / R2 / R3 + 可选 E-1b 工具）

> 生成于 2026-08-26，依据：实验代码二次审阅（纯审阅报告，见会话记录）。
> 性质：**本地代码修复任务**，无 GPU、无训练、无评估实跑。
> 仓库：opd（Windows checkout；若在 `.claude/worktrees/` 下工作遵守该目录纪律）。

---

## 0. 背景（30 秒）

2026-08-26 对实验代码的纯审阅发现 2 个 bug + 1 处注释错误，均属评估/训练基础设施，不涉及算法：

- **R1（P0，阻断所有训练类实验）**：`main/scripts/run_s2_real.py` 的 `_run_experiment` 中
  `metrics = out["metrics"]` 赋值行在 commit `8d1411c` 改写 `.run()` 调用区时丢失
  （`b4b9872` 修了同提交引入的重复 `.run()` 调用，但漏了这个丢行）。后果：训练完成
  300 步后进入汇总阶段抛 `NameError: metrics`，被外层 except 捕获 -> 每个实验的
  summary 记为 error（`l2_experiment_summary.json` 全 error）。训练产物本身完好，
  但汇总全毁且极具误导性。正式 E1/E2（早于 8d1411c）未受影响；**KL 档位扫描若从
  当前 main 启动必中此雷**。
- **R2（P1，产物完整性）**：`main/scripts/vllm_budget_eval.py` 的 `all_results.json`
  以覆盖模式 `"w"` 写。两个丢数据场景：① 同一 out-dir 多次调用（E-0c 拐点扫描对
  S120/S200/S311 发起 3 次调用）只剩最后一次；② 两进程并发写同一 out-dir
  （B2048 的 2+1 分卡模式）后完成者覆盖先完成者--服务器上
  `/root/autodl-tmp/chat_retest/B2048/all_results.json` 很可能只含 2 个模型。
- **R3（P2，注释纠偏）**：commit `6ae3628` 给 `export_student_ckpt.py` 加 `--device`
  默认 cpu 的理由写错了（"from_pretrained 无 device 默认落 cuda:0" 不成立--transformers
  默认就在 CPU 加载）。**行为保留**（显式 `device_map="cpu"` 仍值得要：把设备选择从
  隐式默认变成显式契约），只纠正代码注释与 help 文案中的错误论证。

## 0.5 硬约束（全程适用）

1. **只改下文明确列出的文件与位置**；不顺手重构、不改无关行为、不动算法。
2. 遵守 AGENTS.md「训练产物不可再生约束」：本任务只改代码/测试/文档，
   **不触碰任何 run-dir、metrics.csv、checkpoint、评估 jsonl**。
3. 每项修复必须附带指定测试；全量回归零失败才算完成（判据写死，见各任务）。
4. 提交规范：conventional commit + 中文详细说明；任务 A/B/C **各自独立提交**
   （回滚粒度）；任务 D（若做）单独提交。
5. Windows 下所有 pytest 命令前缀 `PYTHONIOENCODING=utf-8`（GBK 控制台）。
6. 不伪造：测试数字如实汇报；任何一步失败停下记录，不静默绕过、不缩小判据。

---

## 任务 A（P0）：修复 run_s2_real.py 的 metrics NameError

### A1. 缺陷定位

文件 `main/scripts/run_s2_real.py`，函数 `_run_experiment`（当前约 L212 起）：

```python
        out = FullStackOPDv2(cfg, device=args.device).run(run_dir=d, resume=resume)
        # M3：均值只统计【含该键】的训练步 metric--rollout 相位 metric 缺键时
        # 旧实现 m.get(k, 0.0) 会往 reward/pg/kl 均值里混入大量 0，污染口径。
        def _keyed_mean(key):
            vals = [m[key] for m in metrics
                    if isinstance(m, dict) and key in m]
```

`metrics` 在该函数内 3 处使用（`_keyed_mean`、summary 的 `n_steps` 统计、rollout
状态计数扫描），全文件无赋值（`grep -n "metrics = " main/scripts/run_s2_real.py`
仅应命中你即将加回的这一行）。

### A2. 修改内容（恰好一处）

在 `out = FullStackOPDv2(...).run(...)` 行之后、`# M3` 注释之前插入：

```python
        metrics = out["metrics"]   # 8d1411c 重写时丢行（b4b9872 漏修）：缺此行汇总 NameError
```

不改其它任何内容（重复 `.run()` 已由 b4b9872 修掉，勿动）。

### A3. 新增测试（防再次漏网；现有测试全是纯函数测试，未覆盖此路径）

文件：`main/tests/test_run_s2_real_parallel.py`（追加 1 例，不动已有 31 例）。

`test_run_experiment_summary_no_nameerror`：
- `monkeypatch.setattr(rs2, "load_config", fake_load_config)`，返回
  `{"run": {"checkpoint_every": 10}}`（跳过真实 YAML）；
- `monkeypatch.setattr("fullstack_opd_v2.pipeline.FullStackOPDv2", FakePipeline)`
  （注意：`_run_experiment` 内部是**函数内 import**，monkeypatch 必须打在
  `fullstack_opd_v2.pipeline` 模块属性上，import 语句每次执行都取当前属性）；
  `FakePipeline.__init__(self, cfg, device=None)` 为 no-op；
  `run(self, run_dir=None, resume=None)` 返回
  `{"metrics": [{"step": 0, "reward": 1.0, "pg_loss": 0.5, "kl_loss": 0.1,
                 "phase": "train"}],
    "timings": {"total": 1.0}}`；
- 构造 argparse.Namespace 形式的 args，**必须含** `_build_overrides` 与
  `_run_experiment` 用到的全部字段：`config, run_dir, device, n_steps, materialized,
  m_refresh, refresh_min, cache_path, load_cache, batch_size, refresh_size, eos_id,
  extra_sets=[], resume=False`（实验名用 `"S2_E0_static"`--矩阵内合法名）；
- 调 `rs2._run_experiment(args, "S2_E0_static", load_cache=True, prefix=None)`；
- 断言：返回 dict 的 summary **无 error 键**、`summary["reward_mean"] == 1.0`、
  `summary["n_steps"] == 1`。

（原理：若 NameError 存在，外层 except 会把它吞成 `summary["error"]`，断言
`"error" not in summary` 即失败--测试天然捕雷。）

### A4. 判据（写死）

```bash
cd main
PYTHONIOENCODING=utf-8 python -m pytest tests/test_run_s2_real_parallel.py -q
# 期望：32 passed（31 旧 + 1 新）
grep -n "metrics = out" scripts/run_s2_real.py        # 期望恰好 1 行
grep -c "FullStackOPDv2(cfg" scripts/run_s2_real.py   # 期望 1（防 .run() 重复回归）
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q     # 期望全量 0 failed
```

提交：`fix(s2): run_s2_real 补回 8d1411c 丢失的 metrics = out["metrics"]（汇总 NameError 根因）+ _run_experiment 集成测试`

---

## 任务 B（P1）：vllm_budget_eval.py 的 all_results.json 改合并写

### B1. 缺陷定位

文件 `main/scripts/vllm_budget_eval.py`，main() 末尾（当前 L183-185）：

```python
    with open(os.path.join(args.out_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("DONE", flush=True)
```

### B2. 修改内容

（1）在 `_aggregate_budget` 之后新增纯函数：

```python
def _merge_all_results(out_dir: str, new_results: list[dict]) -> list[dict]:
    """读已有 all_results.json 按 (label, budget) 合并新结果后返回（合并写，不覆盖）。

    - 同 (label, budget) 旧条目被新结果替换（重跑幂等）；
    - 其它 label/budget 的旧条目保留：同 out-dir 多次调用（拐点扫描逐模型）与
      多进程分模型并写（B2048 式 2+1 分卡）都不再互相覆盖；
    - 已有文件不存在/损坏/非 list -> 视为空（不因旧产物损坏崩评估）。
    输出按 (label, budget) 排序，保证确定性。
    """
    path = os.path.join(out_dir, "all_results.json")
    merged: dict[tuple, dict] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, list):
                for r in old:
                    if isinstance(r, dict) and "label" in r and "budget" in r:
                        merged[(r["label"], r["budget"])] = r
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass                       # 旧产物损坏：视为空，不崩
    for r in new_results:
        merged[(r["label"], r["budget"])] = r
    return [merged[k] for k in sorted(merged)]
```

（2）main() 末尾改为原子合并写：

```python
    merged = _merge_all_results(args.out_dir, all_results)
    tmp = os.path.join(args.out_dir, "all_results.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(args.out_dir, "all_results.json"))  # 原子替换防半写
    print("DONE", flush=True)
```

（3）模块 docstring 用法段追加一行：同一 `--out-dir` 可多次调用/多进程分模型并写，
`all_results.json` 按 (label, budget) 合并、重跑幂等。

说明：并发**同时**落笔的毫秒级窗口仍存在（读-改-写非锁保护），但两进程完成时间
天然错开，实践中够用；文档化推荐模式仍是一进程一 out-dir 的严格版。

### B3. 新增测试（3 例）

文件 `main/tests/test_vllm_budget_eval.py`（当前 16 例 -> 19 例；import 行追加
`_merge_all_results`）：

```python
def test_merge_all_results_preserves_other_labels(tmp_path):
    """已有 S120/B512，新写 S200/B512 -> 两条共存（顺序多次调用不丢旧结果）。"""
def test_merge_all_results_same_key_overwrite(tmp_path):
    """已有 S120/B512 acc=0.1，重跑同键 acc=0.2 -> 只剩 0.2（幂等）。"""
def test_merge_all_results_corrupted_file(tmp_path):
    """已有文件非法 JSON -> 静默视为空，返回仅含新结果（不崩评估）。"""
```

### B4. 判据（写死）

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_vllm_budget_eval.py -q
# 期望：19 passed
PYTHONIOENCODING=utf-8 python -m pytest tests/ -q     # 期望全量 0 failed
```

提交：`fix(eval): vllm_budget_eval all_results.json 改 (label,budget) 合并写 + 原子替换--同 out-dir 多次调用/多进程并写不再互相覆盖（E-0c 拐点扫描与 B2048 分卡场景）+ 3 测试`

---

## 任务 C（P2）：export_student_ckpt.py 注释纠偏

commit `6ae3628` 的论证错误（"from_pretrained 无 device 默认落 cuda:0" 不成立）。
**行为与默认值一律保留**，只改文字：

- `--device` 的 help：删除"避免与并行评估抢卡 OOM"的错误理由，改为
  `"导出设备（默认 cpu：显式契约，纯权重搬运无需 GPU；可传 cuda:i）"`；
- main() 内块注释：改写为真实动机（device_map 显式化--把设备选择从隐式默认变成
  显式契约、可选上卡），删除"默认落 cuda:0 会 OOM"的错误表述。

判据：`grep -rn "默认落 cuda:0" main/scripts/` 无输出；`--help` 显示新文案；
无任何行为变化（无需新增测试，现有测试零回归）。

提交：`docs(export): export_student_ckpt 注释纠偏--6ae3628 的 cuda:0 论证不成立（from_pretrained 默认即 CPU），改为显式契约表述；行为不变`

---

## 任务 D（可选、独立提交、时间盒）：E-1b 判别实验脚本

> **2026-08-26 更新：本任务已由原始会话 commit 81b227e 实现**--
> （三阶段 sample/logp/correlate +
> Spearman/AUC + 23 单测）+ （E-0b）+ 
> （E-0d）均已入库。**执行本提示词时跳过任务 D**，仅当核对上述脚本与下述规格
> 有实质缺口时才补差。

**目的**：量化「序列级 Δ_T 与答案正确性的相关性」--OPD 失败归因的决定性实验
（`docs/reports/2026-08-26-opd-failure-analysis.md` §5 E-1b）。

**新文件**：`main/scripts/delta_correctness_corr.py`

CLI（写死）：
```
--problems MATH500 --n-problems 200 --n-samples 4
--student <Base 路径> --teacher-rl <JustRL 路径> --teacher-ref <R1-Distill 路径>
--temperature 1.0 --max-new-tokens 2048 --chat-template --seed 42
--device cuda:i --out <out.json>（逐样本 jsonl 同目录）
```

流程（四段）：
1. **采样**：vLLM 起 student（复用 `_apply_cuda_visible` 选卡 + `build_prompts`
   chat 包裹），T=1.0、n=4，200 题 -> 800 条 response 文本；
2. **判分**：复用 `extract_final_answer` + `_grade_answer_sympy` 逐条判 correct；
3. **教师打分**：两个教师引擎**顺序加载**（del + empty_cache 换下一个），
   每条序列算 `Δ_seq = Σ_t (logπ_rl(tok_t) − logπ_ref(tok_t))`，求和域 =
   response 段 token。实现口径：把 chat prompt + response 拼成完整文本，
   `SamplingParams(max_tokens=1, prompt_logprobs=0)` 对完整文本打分，
   从 prompt_token_ids 长度处切出 response 段的逐 token logprob。
   两教师同词表 151643（C3 审计已证）-> 同一 tokenization、位置天然对齐；
   **注意 BOS 一致性**：拼接时 `add_special_tokens=False`，先 smoke 一条
   decode 校验边界（30 秒）；
4. **汇总**：Spearman(Δ_seq, correct)（逐样本级）+ 按题聚合级 Spearman + AUC
   + 逐样本 jsonl（problem_id/sample_idx/response/final_answer/correct/Δ_seq）。

判定阈值（写死，抄自归因文档，脚本只算数不判分支）：ρ ≥ 0.2 / 0.05-0.2 / < 0.05。

测试：Δ 求和、题级聚合、判分连接等**纯函数**各 1-2 例（不起 vLLM/GPU）；
CLI parse_args 支持 argv 注入（沿用 vllm_budget_eval 的可测模式）。

提交：`feat(diag): delta_correctness_corr--Δ_T↔正确性相关性判别实验（E-1b，OPD 失败归因决定性实验）`

---

## 执行顺序与总验收

顺序：A -> B -> C ->（可选 D）。每任务完成即独立 commit；最后跑一次全量回归。

| # | 验收项 | 判据（写死） |
|---|---|---|
| 1 | R1 修复 | `metrics = out["metrics"]` 恰好 1 行；`FullStackOPDv2(cfg` 恰好 1 处；新集成测试过 |
| 2 | R1 测试 | test_run_s2_real_parallel.py = 32 passed |
| 3 | R2 修复 | all_results.json 合并写 + 原子替换；docstring 更新 |
| 4 | R2 测试 | test_vllm_budget_eval.py = 19 passed |
| 5 | R3 纠偏 | `grep -rn "默认落 cuda:0" main/scripts/` 空；行为零变化 |
| 6 | 全量回归 | `pytest tests/ -q` 0 failed（总数相对当前基线只增不减） |
| 7 | 提交纪律 | A/B/C(/D) 独立提交，conventional commit + 中文说明 |

## 后续（不在本提示词范围，仅备忘）

- 修复 push 到 main 后，**服务器训练类实验（KL 档位扫描）开跑前必须 `git pull`**
  （服务器当前停在 `ce05e61`，同样带 R1 雷）；评估类实验（AIME24 续跑/E-0c/E-1a）
  不受 R1 影响。
- 服务器新实例端口 **45815**（旧 35318 已废）；ssh config 的 `Host opd` 需更新。
- B2048 已产出的 `all_results.json` 若只有 2 个模型：从三个 jsonl 用
  `_aggregate_budget` 语义重算聚合补全（jsonl 是唯一事实来源）。

## 汇报要求

完成后逐项汇报：每任务 diff 摘要、测试数变化（A: 31->32；B: 16->19）、总验收表逐条
打勾、commit 哈希。任何失败：停下、原样贴出现象（含堆栈），不静默绕过。

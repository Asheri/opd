# 工程化二次审查修复设计（A 正确性 / B 一致性 / C 性能 / D 测试）

> 基于 2026-08-09 六维 ultracode 审查（47 条确认发现），对 `main/fullstack_opd_v2`
> 工程化改造（T1-T12）做第二轮修复。**算法内核（π_old PG / k3 KL / causal mask /
> staleness 双截断 / teacher 一致性）一行不动**——审查已确认当前未被破坏。

## 背景

第一轮工程化（demo → 工程项目，T1-T12）引入了 run 目录 / 结构化日志 / checkpoint /
metrics / CLI。二次审查发现 47 条问题，归为四主题：A 正确性（7 P1）、B 工程一致性、
C 性能、D 测试。用户批准单设计 + 单实现计划全做。

---

## A · 正确性（P1 全修）

### A1 修复 distributed 分支 UnboundLocalError
- **问题**：`pipeline.run()` 的 distributed 分支只赋 `metrics`，后续 `_on_step`/`cm` 引用
  未定义的 `scheduler` → 崩。
- **修法**：`_on_step` 闭包只在非分布式（线程）分支定义；分布式分支不注册 per-step 钩子，
  末步 force 保存照常。分布式骨架本就无 per-step metrics/checkpoint。

### A2 修复 `cli eval` FileExistsError
- **问题**：`_cmd_eval` 用 `CheckpointManager(".", checkpoint_dir=断点文件路径)` →
  `os.makedirs` 在文件上抛 `FileExistsError`（实测复现）。
- **修法**：`CheckpointManager` 构造传**目录**（如 `os.path.dirname(checkpoint)`），
  `load()` 收**文件路径**。`_cmd_eval` 构造参数修正。

### A3 resume 的 KL 锚点污染（关键）
- **问题**：resume 后 `student.load_state_dict(断点)`，Stage 2 的 KL 锚点用**已训练** student
  的 `response_dists` → 破坏「KL 锚点 = 初始 student 分布」不变式。
- **修法**：**checkpoint 存 ref 锚点**（初始 student 在 fat D 上的 `ref_dists`/`ref_ids`/
  `ref_logp`）。resume 时若 checkpoint 带锚点则恢复，否则（旧断点）重算初始 student 锚点。
  `CheckpointManager.save/load` 扩展 `ref_*` 字段。

### A4+A5 config 快照≠运行时配置 + 下渗与 extra="forbid" 矛盾
- **问题**：下渗（顶层部署键注入 stage 子 dict）发生在 run() 内、config.yaml 快照**之后**且
  不回写 → 快照非有效配置；且 `stage2.dtype` 等在 `Stage2Cfg` schema 无合法位置。
- **修法**：**下渗逻辑移到 `load_config`**（config 层完成顶层键→stage 子 dict 注入），
  快照即有效配置；pipeline 与 cli 不再各自下渗（DRY，见 B4）。

### A6 scheduling_mode 未实现枚举
- **问题**：`Literal["fully_async","n_step_off","fused_hybrid_sync"]` 后两者未实现，
  通过校验后静默按 fully_async 跑。
- **修法**：收窄为 `Literal["fully_async"]`（唯一已实现），请求其它抛 `ConfigError`。

### A7 异常路径资源泄漏
- **问题**：`run()` 中 `mr.close()`/`close_logging()` 只在成功路径执行。
- **修法**：`try/finally` 包裹，保证无论成败都释放 MetricsRecorder 与 FileHandler。

## B · 工程一致性

### B1 异常体系收敛 OPDError
- `pydantic.ValidationError` / `--set` 缺 `=` → `ConfigError`
- warmup_source 非法 → `DataError`
- state_dict 键不匹配 → `CheckpointError`（resume/eval 捕获并友好报错）
- CLI `main()` 捕 `OPDError` + `ValidationError` 统一友好退出

### B2 build_model 接入训练路径
- pipeline 的 student / teacher 构造改走 `model_factory.build_model`（toy 默认行为不变）。
- `stage0_small_rl` 内部模型仍用 `CausalToyLM`（toy RL 阶段），或也接 build_model（二选一，
  默认接主 student/teacher）。

### B3 清理入口分裂
- 删 `demo.py`（死代码），`run_fullstack_v2.py` 指向 `cli.main`。
- README / CLAUDE.md 更新：`python -m fullstack_opd_v2` 需子命令，`train` 为推荐入口。

### B4 下渗逻辑 DRY
- 下渗集中到 `config.py` 一个函数（如 `_apply_deployment_keys(cfg)`），pipeline 与 cli 复用。

### B5 metrics.csv_path 配置键接入
- pipeline 构造 `MetricsRecorder` 时读 `cfg["metrics"]["csv_path"]`（此前被静默忽略）。

## C · 性能优化

### C1 on_step 异步化
- checkpoint + metrics 写进**后台线程队列**，不阻塞训练线程。
- 实现：pipeline 里起一个 daemon 消费线程 + `queue.Queue`；`_on_step` 只 `put`；
  训练结束 `join`。`metrics.record` 线程安全化（加锁）。

### C2 MetricsRecorder flush 节流
- 每 N 步 flush（默认 10），`record()` 不再每步 flush；`close()` 时终刷。

### C3 热路径去重
- metrics 先收集成 list，末尾统一 `.item()`（避免每步 5 次设备同步）。
- `scheduler._train_step` 里 reward/adv 的 `.item()` 推迟到收集后。

### C4 ToyDataLoader 复用
- pipeline `__init__` 只 load 一次；`cache` 命令复用同一 pipeline 实例的数据。

### C5 RunManager.create 幂等
- 目录已存在时不重复 mkdir；config.yaml 已存在时覆盖更新（仍快照最新 cfg）。

## D · 测试补强

- **D1** `eval`/`cache` 子命令端到端测试（随 A2/B3 同步加）。
- **D2** 修 metrics 假绿：断言 CSV 内容逐列正确 + 缺失字段补齐行为真实。
- **D3** checkpoint 边界：force 保存、空目录 resume→None、坏文件→`CheckpointError`。
- **D4** resume 强断言：版本从 step N 继续、KL 锚点恢复（A3）、E[Δ_T] 继续上升。
- **D5** `backend='none'` 路径 + seed 配置生效（A4 后顶层 seed 真正生效）。

---

## 文件改动映射

| 文件 | A | B | C | D |
|---|---|---|---|---|
| `config.py` | A4/A5/A6 | B4 | | D5 |
| `pipeline.py` | A1/A4/A7 | B2 | C1 | |
| `checkpoint.py` | A3 | | | D3 |
| `cli.py` | A2 | B1/B3 | | D1 |
| `metrics.py` | | | C2 | D2 |
| `scheduler.py` | | | C3 | |
| `logging.py` | A7 | | | |
| `data.py` | | | C4 | |
| `run.py` | | | C5 | |
| `demo.py`/`run_fullstack_v2.py`/README/CLAUDE.md | | B3 | | |
| `tests/*` | | | | D1-D5 |

## 验证

- 全量 `pytest tests/ -q`（94 旧 + 新增全绿）。
- 端到端：`train` → `--resume`（版本续跑 + KL 锚点恢复 + E[Δ_T] 上升）。
- `cli eval`/`cache` 端到端不崩。
- 性能：on_step 异步后训练循环不再同步写盘。
- 算法内核未动：E[Δ_T]↑ + staleness age>0 保持。
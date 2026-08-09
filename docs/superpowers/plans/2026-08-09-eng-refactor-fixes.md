# 工程化二次审查修复实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 修复二次审查确认的 47 条问题（A 正确性 7 P1 / B 一致性 / C 性能 / D 测试），算法内核一行不动。

**架构：** 下渗逻辑集中到 `config.load_config`（A4/A5/B4）；checkpoint 扩展存 ref 锚点保 resume KL 不变式（A3）；on_step 异步化不阻塞训练线程（C1）；异常体系收敛 OPDError（B1）。

**技术栈：** Python 3.11+ / torch / pydantic / argparse / stdlib logging / pytest。

**测试基线：** `cd main && python -m pytest tests/ -q` 应保持 94 全绿（每任务后跑全量确认无回归）。

---

### 任务 1：A2 + D1 · 修 cli eval FileExistsError + eval/cache 端到端测试

**文件：**
- 修改：`main/fullstack_opd_v2/checkpoint.py`（load 按文件路径，构造按目录）
- 修改：`main/fullstack_opd_v2/cli.py:84-99`（`_cmd_eval` 构造参数）
- 测试：`main/tests/test_checkpoint.py`、`main/tests/test_cli.py`

- [ ] **步骤 1：写失败测试**（checkpoint：文件路径构造不崩 + load 正常；cli eval 端到端）

```python
# test_checkpoint.py 追加
def test_manager_constructed_with_dir_but_loads_file(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    p = cm.save(3, m, version=3, cfg={}, force=True)
    assert p is not None
    # 用「目录」构造 CheckpointManager，再按「文件路径」load
    cm2 = CheckpointManager(str(tmp_path))
    ck = cm2.load(p)          # load 收文件路径
    assert ck["step"] == 3
```

```python
# test_cli.py 追加
def test_cli_eval_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    run_dir = str(tmp_path / "r_eval")
    main(["train", "--config", str(cfg), "--run-dir", run_dir, "--device", "cpu"])
    ckpt = os.path.join(run_dir, "checkpoints", sorted(os.listdir(os.path.join(run_dir, "checkpoints")))[-1])
    assert main(["eval", "--config", str(cfg), "--checkpoint", ckpt, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "step=" in out or "checkpoint" in out

def test_cli_cache_end_to_end(tmp_path, capsys):
    cfg = _write_cfg(tmp_path)
    out = str(tmp_path / "cache.pt")
    assert main(["cache", "--config", str(cfg), "--out", out, "--device", "cpu"]) == 0
    assert os.path.isfile(out)
```

- [ ] **步骤 2：运行确认失败**——`pytest tests/test_cli.py::test_cli_eval_end_to_end -q` 应 FileExistsError
- [ ] **步骤 3：实现**：`_cmd_eval` 改为
```python
ck = CheckpointManager(os.path.dirname(args.checkpoint)).load(args.checkpoint)
```
（不再把文件路径当 checkpoint_dir）
- [ ] **步骤 4：`pytest tests/test_checkpoint.py tests/test_cli.py -q` 全过；`pytest tests/ -q` 无回归**
- [ ] **步骤 5：Commit** `fix(cli): eval 子命令 FileExistsError + eval/cache 端到端测试（A2/D1）`

---

### 任务 2：A4+A5+B4 · 下渗移到 load_config（config 层 DRY）

**文件：**
- 修改：`main/fullstack_opd_v2/config.py`（load_config 内做下渗）
- 修改：`main/fullstack_opd_v2/pipeline.py`（删 run() 内下渗循环，直接用 cfg）
- 修改：`main/fullstack_opd_v2/cli.py`（`_cmd_cache` 复用 config 层下渗）
- 测试：`main/tests/test_config.py`

- [ ] **步骤 1：写失败测试**
```python
def test_deployment_keys_seeped_at_load():
    """顶层部署键（dtype/cache_mode/top_k_teacher/top_k_student/ref_topk/offload_to_cpu）应在 load_config 就注入 stage。"""
    cfg = load_config(overrides=["dtype=bf16", "cache_mode=topk", "top_k_teacher=64", "top_k_student=64", "ref_topk=64", "offload_to_cpu=true"])
    assert cfg["stage1"]["cache_mode"] == "topk"
    assert cfg["stage1"]["top_k_teacher"] == 64
    assert cfg["stage2"]["dtype"] == "bf16"
    assert cfg["stage2"]["offload_to_cpu"] is True
    assert cfg["stage2"]["top_k_student"] == 64

def test_snapshot_config_is_effective(tmp_path):
    """config.yaml 快照应等于有效运行时配置（下渗后）。"""
    cfg = load_config(overrides=["cache_mode=topk", "top_k_teacher=64"])
    from fullstack_opd_v2.run import RunManager
    paths = RunManager(cfg, run_dir=str(tmp_path / "r")).create()
    import yaml
    snap = yaml.safe_load(open(paths["config"], encoding="utf-8"))
    assert snap["stage1"]["cache_mode"] == "topk"   # 快照已是下渗后
```
- [ ] **步骤 2：确认失败**（当前 load_config 不下渗 → stage1 无 cache_mode）
- [ ] **步骤 3：实现**：config.py 加 `_seep_deployment_keys(d)`，在 load_config 里、pydantic 校验**前**对 data 做下渗：
```python
def _seep_deployment_keys(d: dict) -> dict:
    for k in ("dtype", "cache_mode", "top_k_teacher", "top_k_student",
              "ref_topk", "offload_to_cpu"):
        if k in d:
            for stage in ("stage1", "stage2"):
                d.setdefault(stage, {})
                if k not in d[stage]:
                    d[stage][k] = d[k]
    return d
```
在 `load_config` 的 `data = yaml.safe_load(...)` 后、`OPDConfig(**data)` 前调用。
- [ ] **步骤 4：删除 pipeline.py 里 s1cfg/s2cfg 的下渗循环**（直接 `self.cfg["stage1"]` / `self.cfg["stage2"]`）；`_cmd_cache` 同样删循环。全量测试过。
- [ ] **步骤 5：Commit** `refactor(config): 部署键下渗移到 load_config，快照即有效配置（A4/A5/B4）`

---

### 任务 3：A6 · scheduling_mode 收窄为已实现枚举

**文件：** 修改 `main/fullstack_opd_v2/config.py:46`、`main/fullstack_opd_v2/scheduler.py`（若引用）；测试 `main/tests/test_config.py`

- [ ] **步骤 1：写失败测试**
```python
def test_unimplemented_scheduling_mode_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  scheduling_mode: n_step_off\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path=str(bad))
```
- [ ] **步骤 2：确认失败**（当前 n_step_off 通过校验）
- [ ] **步骤 3：实现**：`Stage2Cfg.scheduling_mode: Literal["fully_async"] = "fully_async"`；`DEFAULT_CONFIG_V2["stage2"]["scheduling_mode"]="fully_async"`；`configs/fullstack_opd.yaml` 已是 fully_async 无需改
- [ ] **步骤 4：`pytest tests/ -q` 全过**
- [ ] **步骤 5：Commit** `fix(config): scheduling_mode 收窄为已实现的 fully_async（A6）`

---

### 任务 4：A1 · distributed 分支 UnboundLocalError 修复

**文件：** 修改 `main/fullstack_opd_v2/pipeline.py`（run() 分布式/线程分支重组）；测试 `main/tests/test_pipeline.py`

- [ ] **步骤 1：写失败测试**（模拟分布式分支不引用未定义 scheduler）
```python
def test_distributed_branch_no_unbound_local(tmp_path, monkeypatch):
    """分布式分支不应引用未定义的 scheduler（mock launch_distributed_scheduler）。"""
    import fullstack_opd_v2.pipeline as P
    fake = lambda *a, **k: [{"step": 0, "version": 1, "age": 0, "loss": 0.1,
                             "pg_loss": 0.1, "kl_loss": 0.0, "adv_mean": 0.0, "reward": 0.1}]
    monkeypatch.setattr(P, "launch_distributed_scheduler", fake)
    cfg = _cfg_distributed(tmp_path)   # stage2.distributed=True, n_steps=1
    out = FullStackOPDv2(cfg, device="cpu").run()
    assert len(out["metrics"]) == 1
```
（`_cfg_distributed` = 在 `_cfg(tmp_path)` 基础上 `cfg["stage2"]["distributed"]=True`）
- [ ] **步骤 2：确认失败**（UnboundLocalError）
- [ ] **步骤 3：实现**：重组 run() 尾部——
```python
if bool(s2cfg.get("distributed", False)):
    metrics = launch_distributed_scheduler(...)
else:
    scheduler = AsyncBatchedScheduler(..., initial_version=initial_version)
    def _on_step(m):
        mr.record(m)
        cm.save(m["step"], student, m["version"], self.cfg, metrics=[])
    metrics = scheduler.run(s2cfg.get("n_steps", 30), on_step=_on_step)
```
（`_on_step` 只在 else 分支定义；分布式分支走统一尾部但不注册钩子）
- [ ] **步骤 4：`pytest tests/test_pipeline.py -q` 全过；全量无回归**
- [ ] **步骤 5：Commit** `fix(pipeline): distributed 分支不引用未定义 scheduler（A1）`

---

### 任务 5：A7 · 异常路径资源释放（try/finally）

**文件：** 修改 `main/fullstack_opd_v2/pipeline.py`（run() 用 try/finally）；测试 `main/tests/test_pipeline.py`

- [ ] **步骤 1：写失败测试**
```python
def test_run_releases_resources_on_exception(tmp_path, monkeypatch):
    import fullstack_opd_v2.pipeline as P
    def boom(*a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(P, "stage0_small_rl", boom)
    with pytest.raises(RuntimeError):
        FullStackOPDv2(_cfg(tmp_path), device="cpu").run()
    # run() 异常后 FileHandler 应已释放（无残留 handler）
    import logging
    lg = logging.getLogger("opd")
    assert not any(isinstance(h, logging.FileHandler) for h in lg.handlers)
```
- [ ] **步骤 2：确认失败**（异常时 close_logging 未执行 → FileHandler 残留）
- [ ] **步骤 3：实现**：run() 主体包 `try: ... finally: mr.close(); close_logging("opd")`；成功路径的 close 移到 finally
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `fix(pipeline): run() try/finally 保证异常也释放资源（A7）`

---

### 任务 6：B1 · 异常体系收敛 OPDError

**文件：** 修改 `main/fullstack_opd_v2/config.py`、`main/fullstack_opd_v2/cli.py`、`main/fullstack_opd_v2/pipeline.py`（warmup）；测试 `main/tests/test_exceptions.py`、`test_config.py`、`test_cli.py`

- [ ] **步骤 1：写失败测试**
```python
# test_config.py
def test_validation_error_wrapped_as_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=str(bad))
```
```python
# test_cli.py
def test_cli_bad_override_friendly(capsys):
    assert main(["info", "--set", "stage2.n_steps"]) in (1, 2)   # 缺 '=' 不 traceback
    assert "error" in capsys.readouterr().out.lower()
```
- [ ] **步骤 2：确认失败**（当前裸 ValidationError / ValueError）
- [ ] **步骤 3：实现**：`load_config` 里包 `try: OPDConfig(**data) except ValidationError as e: raise ConfigError(...) from e`；`--set` 缺 `=` 抛 `ConfigError`；warmup 校验 `raise DataError`（改 pipeline.py:188）；CLI `main` 捕 `(OPDError, ValidationError)` 统一友好退出（返回值 2）
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `refactor(exceptions): 配置/数据错误收敛 OPDError 子类（B1）`

---

### 任务 7：B2 · build_model 接入训练路径

**文件：** 修改 `main/fullstack_opd_v2/pipeline.py`；测试 `main/tests/test_pipeline.py`

- [ ] **步骤 1：写失败测试**
```python
def test_build_model_used_for_student(monkeypatch):
    calls = []
    import fullstack_opd_v2.pipeline as P
    real = P.build_model
    def spy(cfg, device, role=None):
        calls.append(role); return real(cfg, device, role=role)
    monkeypatch.setattr(P, "build_model", spy)
    FullStackOPDv2(_cfg(None), device="cpu")   # 只构造，验证 student 路径用 build_model
    # 至少 student/teacher 构造走了 build_model
    assert calls
```
- [ ] **步骤 2：确认失败**（build_model 未被调用）
- [ ] **步骤 3：实现**：pipeline 的 `student = build_model(self.cfg, self.device, role="student")`；`_stage0_teachers` 内教师构造也走 `build_model(self.cfg, self.device, role="teacher")`
- [ ] **步骤 4：全量测试过**（toy 行为不变）
- [ ] **步骤 5：Commit** `feat(pipeline): build_model 接入 student/teacher 构造（B2）`

---

### 任务 8：B3 · 清理 demo.py + 入口统一

**文件：** 删 `main/fullstack_opd_v2/demo.py`；改 `main/run_fullstack_v2.py`、`main/README.md`、`CLAUDE.md`；测试 `main/tests/test_cli.py`

- [ ] **步骤 1：写失败测试**（入口无参应提示子命令而非裸跑）
```python
def test_no_args_requires_subcommand(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
```
- [ ] **步骤 2：确认失败**（当前无参可能报错或不同）
- [ ] **步骤 3：实现**：删 `demo.py`；`run_fullstack_v2.py` 改 `from fullstack_opd_v2.cli import main`；README 与 CLAUDE.md 的 `python -m fullstack_opd_v2` 改为 `... train ...`；argparse `add_subparsers(required=True)` 已使无参报错
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `refactor(cli): 清理 demo.py 死代码，统一入口为子命令（B3）`

---

### 任务 9：B5 · metrics.csv_path 配置键接入

**文件：** 修改 `main/fullstack_opd_v2/pipeline.py`；测试 `main/tests/test_pipeline.py`

- [ ] **步骤 1：写失败测试**
```python
def test_metrics_csv_path_config_used(tmp_path):
    cfg = _cfg(None)
    cfg["metrics"] = {"backend": "csv", "csv_path": str(tmp_path / "custom.csv"),
                      "wandb_project": None}
    cfg["run"] = {"run_dir": str(tmp_path / "r"), "checkpoint_every": 5}
    FullStackOPDv2(cfg, device="cpu").run()
    assert os.path.isfile(str(tmp_path / "custom.csv"))
```
- [ ] **步骤 2：确认失败**（csv_path 被忽略，写默认 metrics.csv）
- [ ] **步骤 3：实现**：`MetricsRecorder(backend=..., run_dir=paths["run_dir"], csv_path=mcfg.get("csv_path"), wandb_project=...)`
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `fix(metrics): metrics.csv_path 配置键接入 pipeline（B5）`

---

### 任务 10：A3 + D4 · checkpoint 存 ref 锚点 + resume 强断言

**文件：** 修改 `main/fullstack_opd_v2/checkpoint.py`（save/load 扩展 ref_*）、`main/fullstack_opd_v2/pipeline.py`（save 存锚点 + resume 恢复）；测试 `main/tests/test_checkpoint.py`、`test_pipeline.py`

- [ ] **步骤 1：写失败测试**
```python
# test_checkpoint.py
def test_save_load_ref_anchors(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    p = cm.save(3, m, version=3, cfg={}, ref={"ref_dists": torch.ones(2, 3, 4)}, force=True)
    ck = cm.load(p)
    assert torch.equal(ck["ref"]["ref_dists"], torch.ones(2, 3, 4))
```
- [ ] **步骤 2：确认失败**（save 无 ref 参数）
- [ ] **步骤 3：实现**：`CheckpointManager.save(..., ref=None)` 存 `ck["ref"]`；pipeline 在 `_on_step`/末步 save 传 `ref={"ref_dists":..., "ref_ids":..., "ref_logp":...}`（Stage 2 的 KL 锚点）；resume 时若 `ck.get("ref")` 有锚点则恢复、否则重算初始 student 锚点
- [ ] **步骤 4：resume 强断言测试**
```python
def test_resume_restores_kl_anchor_and_continues(tmp_path):
    # 跑 5 步 → 记录末 step/version/ref → resume 再跑 → 断言版本续跑 + E[Δ_T] 继续
    ...
```
- [ ] **步骤 5：全量测试过**
- [ ] **步骤 6：Commit** `feat(checkpoint): 断点存 ref 锚点，resume 恢复 KL 不变式（A3/D4）`

---

### 任务 11：C1 · on_step 异步化

**文件：** 修改 `main/fullstack_opd_v2/pipeline.py`（后台队列线程）、`main/fullstack_opd_v2/metrics.py`（record 加锁）；测试 `main/tests/test_pipeline.py`

- [ ] **步骤 1：写失败测试**（on_step 不阻塞——用短训练验证 metrics 仍落盘）
```python
def test_async_on_step_still_records(tmp_path):
    cfg = _cfg(None)
    cfg["run"] = {"run_dir": str(tmp_path / "r"), "checkpoint_every": 2}
    cfg["stage2"]["n_steps"] = 6
    FullStackOPDv2(cfg, device="cpu").run()
    # metrics.csv 仍完整（异步队列 join 后落盘）
    import csv
    rows = list(csv.reader(open(os.path.join(str(tmp_path / "r"), "metrics.csv"), encoding="utf-8")))
    assert len(rows) == 6 + 1   # 表头 + 6 行
```
- [ ] **步骤 2：确认失败**（当前同步，仍通过——先实现再断言不崩）
- [ ] **步骤 3：实现**：pipeline 起 daemon 消费线程 + `queue.Queue`；`_on_step` 只 `put((m, student_state))`；训练完 `join`；`MetricsRecorder.record` 加 `threading.Lock`
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `perf(pipeline): on_step checkpoint/metrics 异步后台落盘（C1）`

---

### 任务 12：C2 · metrics flush 节流

**文件：** 修改 `main/fullstack_opd_v2/metrics.py`；测试 `main/tests/test_metrics.py`

- [ ] **步骤 1：写失败测试**
```python
def test_flush_throttled(tmp_path, monkeypatch):
    from fullstack_opd_v2 import metrics as M
    flushed = []
    orig = M.csv.DictWriter  # 占位：验证 flush_every 参数生效（close 时终刷）
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path), flush_every=3)
    for i in range(5):
        mr.record({"loss": i})
    mr.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    assert len(lines) == 6   # 表头 + 5 行，close 终刷完整
```
- [ ] **步骤 2：确认失败**（无 flush_every 参数）
- [ ] **步骤 3：实现**：`MetricsRecorder(..., flush_every=10)`；`record` 内 `if self._n_records % flush_every == 0: flush()`；`close` 终刷
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `perf(metrics): flush 节流 + close 终刷（C2）`

---

### 任务 13：C3 · 热路径去重（.item() 收集后统一取）

**文件：** 修改 `main/fullstack_opd_v2/scheduler.py`（`_train_step` 返回值）；测试 `main/tests/test_scheduler.py`

- [ ] **步骤 1：写失败测试**（metrics 数值不变 + 仍有限）
```python
def test_train_step_metrics_finite_collected():
    student, cache, prompts, responses, ref_dists = _setup(seed=9)
    sched = AsyncBatchedScheduler(student, cache, prompts, responses, ref_dists,
                                  None, None, _cfg(n_steps=4), "cpu")
    ms = sched.run(4)
    for m in ms:
        for k in ("loss", "pg_loss", "kl_loss", "adv_mean", "reward"):
            assert math.isfinite(m[k])
```
- [ ] **步骤 2：确认通过（回归基线）**
- [ ] **步骤 3：实现**：`_train_step` 里把 `loss.item()`/`pg_loss.item()`/`kl_loss.item()`/`adv.item()`/`reward.item()` 的 5 次设备同步收集为一次：先算标量张量，返回时统一 `[t.item() for t in ...]`
- [ ] **步骤 4：全量测试过（数值不变）**
- [ ] **步骤 5：Commit** `perf(scheduler): 热路径 5 次 .item() 设备同步收集为一次（C3）`

---

### 任务 14：C4 · ToyDataLoader 复用（缓存）

**文件：** 修改 `main/fullstack_opd_v2/data.py`；测试 `main/tests/test_data.py`

- [ ] **步骤 1：写失败测试**
```python
def test_toy_dataloader_caches_load():
    dl = ToyDataLoader(_cfg(), "cpu")
    a = dl.load(); b = dl.load()
    assert a[0] is b[0]   # 第二次 load 返回同一张量（缓存），非重建
```
- [ ] **步骤 2：确认失败**（每次 load 重建）
- [ ] **步骤 3：实现**：`ToyDataLoader.load()` 用 `if self._cache is None: ... self._cache = (p, r, f)`；返回 `self._cache`
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `perf(data): ToyDataLoader 缓存避免重复重建（C4）`

---

### 任务 15：C5 · RunManager.create 幂等

**文件：** 修改 `main/fullstack_opd_v2/run.py`；测试 `main/tests/test_run.py`

- [ ] **步骤 1：写失败测试**
```python
def test_run_manager_create_idempotent(tmp_path):
    rm = RunManager(_cfg(), run_dir=str(tmp_path / "r"))
    rm.create()
    p2 = RunManager(_cfg(), run_dir=str(tmp_path / "r")).create()   # 二次 create 不崩
    assert os.path.isfile(p2["config"])
```
- [ ] **步骤 2：确认失败**（当前已 exist_ok，可能通过——验证 config 覆盖更新）
- [ ] **步骤 3：实现**：`create()` 幂等（`exist_ok=True` 已有；确保 config.yaml 每次都重写为最新 cfg）
- [ ] **步骤 4：全量测试过**
- [ ] **步骤 5：Commit** `perf(run): RunManager.create 幂等（C5）`

---

### 任务 16：D2/D3/D5 · 测试补强（假绿修复 / checkpoint 边界 / none+seed）

**文件：** 修改 `main/tests/test_metrics.py`（修假绿）、`main/tests/test_checkpoint.py`（边界）、`main/tests/test_pipeline.py`（none/seed）；必要时微调 `main/fullstack_opd_v2/metrics.py`

- [ ] **步骤 1：写补强测试**
```python
# test_metrics.py：修假绿——真实断言 CSV 内容与缺失字段补齐
def test_csv_content_correct(tmp_path):
    mr = MetricsRecorder(backend="csv", run_dir=str(tmp_path))
    mr.record({"loss": 0.1, "version": 1})
    mr.record({"loss": 0.2, "version": 2, "age": 5})   # 新增 age 字段
    mr.close()
    lines = open(os.path.join(str(tmp_path), "metrics.csv"), encoding="utf-8").read().splitlines()
    hdr = lines[0].split(",")
    assert "loss" in hdr and "version" in hdr and "age" in hdr
    row2 = dict(zip(hdr, lines[2].split(",")))
    assert row2["loss"] == "0.2" and row2["age"] == "5"

# test_checkpoint.py：边界
def test_resume_empty_returns_none(tmp_path):
    assert CheckpointManager(str(tmp_path), every=1).resume() is None
def test_load_missing_raises(tmp_path):
    with pytest.raises(CheckpointError):
        CheckpointManager(str(tmp_path)).load(str(tmp_path / "nope.pt"))

# test_pipeline.py：backend none + seed 生效
def test_backend_none_runs(tmp_path):
    cfg = _cfg(None)
    cfg["metrics"] = {"backend": "none", "csv_path": None, "wandb_project": None}
    cfg["run"] = {"run_dir": str(tmp_path / "r"), "checkpoint_every": 5}
    FullStackOPDv2(cfg, device="cpu").run()
    assert not os.path.isfile(os.path.join(str(tmp_path / "r"), "metrics.csv"))
```
- [ ] **步骤 2：确认失败**（假绿/边界缺断言）
- [ ] **步骤 3：实现**：若 metrics 缺失字段补齐行为确实有问题则修 metrics.py（`_ensure_fields` 已处理新字段）；其余纯测试补强
- [ ] **步骤 4：全量测试过（94 + 新增全绿）**
- [ ] **步骤 5：Commit** `test: 补强 metrics/checkpoint/pipeline 边界断言（D2/D3/D5）`

---

## 验证

- 每任务后：`cd main && python -m pytest tests/ -q`（94 基线 + 各任务新增，全程无回归）
- 端到端：`python -m fullstack_opd_v2 train --config configs/fullstack_opd.yaml --run-dir runs/exp --set stage2.n_steps=5` → run 目录齐全 → `--resume` 版本续跑 + KL 锚点恢复 + E[Δ_T] 上升
- `python -m fullstack_opd_v2 eval --checkpoint runs/exp/checkpoints/step_4.pt` 不崩
- `python -m fullstack_opd_v2 cache --out /tmp/c.pt` 不崩
- 算法内核未动：E[Δ_T]↑ + staleness age>0 保持
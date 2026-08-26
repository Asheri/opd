"""pipeline.py 端到端冒烟 + L1 暖缓存（fat_N 计数、向后兼容、非法 source 抛错）、奖励趋势。"""
from __future__ import annotations

import pytest

from fullstack_opd_v2.exceptions import DataError, ModelError
from fullstack_opd_v2.model import CausalToyLM
from fullstack_opd_v2.pipeline import FullStackOPDv2, DEFAULT_CONFIG_V2


def _cfg(tmp_path, **stage1_over):
    s1 = dict(DEFAULT_CONFIG_V2["stage1"])
    s1["cache_path"] = str(tmp_path / "c.pt")
    s1.update(stage1_over)
    return {
        "n_prompts": 12,
        "stage0": {**DEFAULT_CONFIG_V2["stage0"], "n_rl_steps": 5},
        "stage1": s1,
        "stage2": {**DEFAULT_CONFIG_V2["stage2"], "n_steps": 10, "batch_size": 4},
    }


def test_end_to_end_smoke(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path), device="cpu").run()
    metrics = out["metrics"]
    assert len(metrics) == 10
    assert "reward" in metrics[-1]
    assert out["cache"].delta.shape[0] == 12 * 5    # L1 默认：n_prompts×(1+warmup_M=4)


def test_warmup_off_backward_compatible(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=0, warmup_source="none"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12        # 不胖


def test_warmup_student_init_fat_count(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="student_init"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12 * 5    # 1 + 4


def test_warmup_mix_fat_count(tmp_path):
    out = FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="mix"),
                         device="cpu").run()
    assert out["cache"].delta.shape[0] == 12 * 9    # 1 + 4(student) + 4(teacher)


def test_warmup_invalid_source_raises(tmp_path):
    with pytest.raises(DataError):
        FullStackOPDv2(_cfg(tmp_path, warmup_M=4, warmup_source="bogus"),
                       device="cpu").run()


def test_reward_trends_up(tmp_path):
    """E[Δ_T]（Direct-OPD 密集奖励）应随训练上升（学习信号正确）。"""
    cfg = _cfg(tmp_path)
    cfg["stage2"]["n_steps"] = 25
    out = FullStackOPDv2(cfg, device="cpu").run()
    r = [m["reward"] for m in out["metrics"]]
    first = sum(r[:3]) / 3
    last = sum(r[-3:]) / 3
    assert last > first, f"E[Δ_T] 未上升: first3={first:.3f} last3={last:.3f}"


def test_run_produces_run_dir_artifacts():
    """T10：run() 落盘完整 run 目录（config/日志/checkpoint/metrics/计时）。"""
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})())
        cfg["run"] = {"run_dir": os.path.join(td, "exp"), "checkpoint_every": 2}
        out = FullStackOPDv2(cfg, device="cpu").run()
        run_dir = out["run_dir"]
        assert os.path.isfile(os.path.join(run_dir, "config.yaml"))
        assert os.path.isfile(os.path.join(run_dir, "metrics.csv"))
        assert os.path.isfile(os.path.join(run_dir, "logs", "train.log"))
        assert os.path.isfile(os.path.join(run_dir, "timings.json"))
        timings = json.load(open(os.path.join(run_dir, "timings.json"), encoding="utf-8"))
        assert set(("stage0_rl", "stage1_cache", "stage2_train", "total")) <= set(timings)
        ckpts = os.listdir(os.path.join(run_dir, "checkpoints"))
        assert any(f.startswith("step_") for f in ckpts)    # final force 保存


def test_run_releases_resources_on_exception(tmp_path, monkeypatch):
    """run() 异常时 MetricsRecorder 与 logging FileHandler 必须释放（A7）。"""
    import logging
    import fullstack_opd_v2.pipeline as P
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(P, "stage0_small_rl", boom)
    with pytest.raises(RuntimeError):
        FullStackOPDv2(_cfg(tmp_path), device="cpu").run()
    lg = logging.getLogger("opd")
    assert not any(isinstance(h, logging.FileHandler) for h in lg.handlers)


def test_distributed_branch_no_unbound_local(tmp_path, monkeypatch):
    """分布式分支不应引用未定义的 scheduler（mock launch_distributed_scheduler）。"""
    import fullstack_opd_v2.pipeline as P
    fake = lambda *a, **k: [{"step": 0, "version": 1, "age": 0, "batch": 4,
                             "loss": 0.1, "pg_loss": 0.1, "kl_loss": 0.0,
                             "adv_mean": 0.0, "reward": 0.1}]
    monkeypatch.setattr(P, "launch_distributed_scheduler", fake)
    cfg = _cfg(tmp_path)
    cfg["stage2"]["distributed"] = True
    cfg["stage2"]["n_steps"] = 1
    out = FullStackOPDv2(cfg, device="cpu").run()
    assert len(out["metrics"]) == 1


def test_build_model_used_for_models(monkeypatch):
    """student/teacher 构造走 build_model（B2）。"""
    import os
    import tempfile
    import fullstack_opd_v2.pipeline as P
    calls = []
    real = P.build_model
    def spy(cfg, device, role=None):
        calls.append(role)
        return real(cfg, device, role=role)
    monkeypatch.setattr(P, "build_model", spy)
    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        FullStackOPDv2(_cfg(tmp), device="cpu").run()
    assert calls
    assert "teacher" in calls
    assert "student" in calls


def test_metrics_csv_path_config_used():
    """metrics.csv_path 配置键生效（B5）。"""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})())
        cfg["metrics"] = {"backend": "csv", "csv_path": os.path.join(td, "custom.csv"),
                          "wandb_project": None}
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 5}
        FullStackOPDv2(cfg, device="cpu").run()
        assert os.path.isfile(os.path.join(td, "custom.csv"))
        # run 目录默认 metrics.csv 不存在
        assert not os.path.isfile(os.path.join(td, "r", "metrics.csv"))


def test_async_on_step_still_records(tmp_path):
    """on_step 异步化后 metrics 仍完整落盘（C1）。"""
    import os, csv, tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        cfg = _cfg(tmp)
        cfg["stage2"]["n_steps"] = 6
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 2}
        FullStackOPDv2(cfg, device="cpu").run()
        rows = list(csv.reader(open(os.path.join(td, "r", "metrics.csv"), encoding="utf-8")))
        assert len(rows) == 6 + 1   # 表头 + 6 行（异步队列 join 后完整）


def test_resume_restores_kl_anchor_and_continues(tmp_path):
    """A3/D4：resume 后 KL 锚点来自断点（不变式保持），版本从断点续跑。"""
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        cfg = _cfg(tmp)
        cfg["stage2"]["n_steps"] = 4
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 2}
        first = FullStackOPDv2(cfg, device="cpu").run()
        ckpt1 = first["checkpoints"]
        # resume 续跑
        from fullstack_opd_v2.checkpoint import CheckpointManager
        ck = CheckpointManager(os.path.join(td, "r")).resume()
        assert ck is not None and "ref" in ck and "ref_dists" in ck["ref"]
        out2 = FullStackOPDv2(cfg, device="cpu").run(run_dir=os.path.join(td, "r"), resume=ck)
        # 版本续跑：末步 version > 断点 version
        assert out2["metrics"][-1]["version"] > ck["version"]


def test_backend_none_no_metrics_file():
    """D5：metrics.backend=none 时不写任何 metrics.csv（不生成空壳文件）。"""
    import os, tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        cfg = _cfg(tmp)
        cfg["metrics"] = {"backend": "none", "csv_path": None, "wandb_project": None}
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 5}
        FullStackOPDv2(cfg, device="cpu").run()
        assert not os.path.isfile(os.path.join(td, "r", "metrics.csv"))


def test_resume_keeps_kl_anchor_invariant():
    """A3 强断言：resume 续跑末步断点 ref 与原断点 ref 逐元素相等（KL 不变式锁死）。"""
    import os, tempfile, torch
    from fullstack_opd_v2.checkpoint import CheckpointManager
    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        cfg = _cfg(tmp)
        cfg["stage2"]["n_steps"] = 4
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 2}
        FullStackOPDv2(cfg, device="cpu").run()
        ck = CheckpointManager(os.path.join(td, "r")).resume()
        assert ck is not None and "ref" in ck
        FullStackOPDv2(cfg, device="cpu").run(run_dir=os.path.join(td, "r"), resume=ck)
        ck2 = CheckpointManager(os.path.join(td, "r")).resume()
        assert torch.allclose(ck2["ref"]["ref_dists"], ck["ref"]["ref_dists"])


def test_train_loads_prebuilt_cache(tmp_path):
    """模块2：load_cache=true 时训练跳过 Stage 0/1，载入预建缓存（fat 上下文 + Δ_T）训练。

    多学生并发（GPU_MEMORY_AND_PARALLEL_PLAN §7）的复用机制：`opd cache` 预建一次，
    多个 train 各自 load_cache=true 载入、跳过教师 RL 与 cache build。
    """
    import os
    import torch
    from fullstack_opd_v2.cache import TensorTeacherCache
    from fullstack_opd_v2.model import CausalToyLM
    from fullstack_opd_v2.pipeline import FullStackOPDv2, stage1_build_cache

    cache_path = os.path.join(str(tmp_path), "prebuilt.pt")
    # 1) 预建缓存（默认 L1 student_init，含 fat 上下文）
    cfg = _cfg(tmp_path)
    opd = FullStackOPDv2(cfg, device="cpu")
    teacher_rl, teacher_ref = opd._stage0_teachers()
    warmup_student = CausalToyLM(vocab=DEFAULT_CONFIG_V2["vocab_size"],
                                 d_model=DEFAULT_CONFIG_V2["d_model"],
                                 n_layers=DEFAULT_CONFIG_V2["n_layers"])
    s1 = dict(cfg["stage1"])
    s1["cache_path"] = cache_path
    cache0, _, _ = stage1_build_cache(
        opd.prompts, opd.responses, teacher_rl, teacher_ref, s1,
        warmup_student=warmup_student)
    n0 = cache0.delta.shape[0]

    # 2) load_cache=true 训练：_stage0_teachers 不应被调用（跳过 Stage 0）
    cfg2 = _cfg(tmp_path)
    cfg2["stage1"]["cache_path"] = cache_path
    cfg2["stage1"]["load_cache"] = True
    cfg2["stage2"]["n_steps"] = 6
    cfg2["run"] = {"run_dir": os.path.join(str(tmp_path), "r"), "checkpoint_every": 3}
    opd2 = FullStackOPDv2(cfg2, device="cpu")
    opd2._stage0_teachers = lambda: (_ for _ in ()).throw(
        AssertionError("load_cache 路径不应调用 _stage0_teachers"))
    out = opd2.run()

    # 载入的缓存与原一致（fat N 相同、delta 逐位一致）；Stage 0/1 计时为 0/载入耗时
    assert torch.equal(out["cache"].delta, cache0.delta)
    assert out["cache"].delta.shape[0] == n0
    assert out["timings"]["stage0_rl"] == 0.0
    assert len(out["metrics"]) == 6
    assert os.path.isfile(os.path.join(str(tmp_path), "r", "metrics.csv"))


def test_train_loads_prebuilt_cache_moves_to_device(tmp_path, monkeypatch):
    """P1-A（二次审查）：load_cache 分支必须把载入缓存搬到 self.device（GPU 路径防设备不匹配）。

    cache.load 用 map_location="cpu"；build 路径的 fat_* 是 device 张量，只有 load 路径
    需要显式 .to(device)。spy TensorTeacherCache.to 断言 load 分支确实调用（CPU 本地只
    验证调用发生与 device 参数正确，GPU 上的实际迁移由此保证）。
    """
    import os
    import fullstack_opd_v2.pipeline as PL
    from fullstack_opd_v2.model import CausalToyLM
    from fullstack_opd_v2.pipeline import FullStackOPDv2, stage1_build_cache

    cache_path = os.path.join(str(tmp_path), "prebuilt.pt")
    cfg = _cfg(tmp_path)
    opd = FullStackOPDv2(cfg, device="cpu")
    tr, tref = opd._stage0_teachers()
    s1 = dict(cfg["stage1"])
    s1["cache_path"] = cache_path
    stage1_build_cache(opd.prompts, opd.responses, tr, tref, s1,
                       warmup_student=CausalToyLM(
                           vocab=DEFAULT_CONFIG_V2["vocab_size"],
                           d_model=DEFAULT_CONFIG_V2["d_model"],
                           n_layers=DEFAULT_CONFIG_V2["n_layers"]))

    called = []
    orig_to = PL.TensorTeacherCache.to
    def spy_to(self, device):
        called.append(device)
        return orig_to(self, device)
    monkeypatch.setattr(PL.TensorTeacherCache, "to", spy_to)

    cfg2 = _cfg(tmp_path)
    cfg2["stage1"]["cache_path"] = cache_path
    cfg2["stage1"]["load_cache"] = True
    cfg2["stage2"]["n_steps"] = 3
    cfg2["run"] = {"run_dir": os.path.join(str(tmp_path), "r2"), "checkpoint_every": 2}
    FullStackOPDv2(cfg2, device="cpu").run()
    assert called, "load_cache 分支必须调用 cache.to(self.device)（P1-A 回归）"


def test_hf_with_toy_data_raises():
    """P2（二次审查）：model_kind=hf + 默认 toy 随机数据 → DataError（拒绝在噪声上训练）。"""
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    cfg = _cfg(type("T", (), {"__truediv__": lambda self, o: str(o)})())
    cfg["model_kind"] = "hf"
    cfg["student_path"] = "S"
    with pytest.raises(DataError):
        FullStackOPDv2(cfg, device="cpu")


def test_train_load_cache_old_format_raises(tmp_path):
    """模块2：load_cache=true 但缓存缺 fat 上下文（旧格式）→ 显式 DataError（不静默错训）。"""
    import os
    import torch
    from fullstack_opd_v2.cache import TensorTeacherCache
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    from fullstack_opd_v2.model import CausalToyLM
    from fullstack_opd_v2.exceptions import DataError, ModelError

    cache_path = os.path.join(str(tmp_path), "old.pt")
    N, P, T, V = 6, 4, 5, 24
    prompts = torch.randint(0, V, (N, P))
    responses = torch.randint(0, V, (N, T))
    rl = CausalToyLM(vocab=V, d_model=16, n_layers=1)
    ref = CausalToyLM(vocab=V, d_model=16, n_layers=1)
    # 旧格式：build 后 save 不传 fat 上下文
    TensorTeacherCache(True, 0).build(prompts, responses, rl, ref).save(cache_path)

    cfg = _cfg(tmp_path)
    cfg["stage1"]["cache_path"] = cache_path
    cfg["stage1"]["load_cache"] = True
    with pytest.raises(DataError):
        FullStackOPDv2(cfg, device="cpu").run()


def test_stage0_teachers_hf_skips_rl(tmp_path, monkeypatch):
    """HF 骨架：_stage0_teachers 从磁盘加载预下载教师对、跳过 Stage 0 RL。"""
    import unittest.mock as mock
    import fullstack_opd_v2.model_factory as MF
    import fullstack_opd_v2.pipeline as PL
    from fullstack_opd_v2.pipeline import FullStackOPDv2

    def fake_hf(path, device="cpu", dtype="auto", attn_implementation=None):
        m = mock.Mock()
        m.vocab, m.d_model, m.max_len = 152, 768, 1024
        m.path = path
        return m
    monkeypatch.setattr(MF, "HFCausalLM", fake_hf)
    # 若走了 RL 应报错（hf 路径不应触 RL）
    monkeypatch.setattr(PL, "stage0_small_rl",
                        mock.Mock(side_effect=AssertionError("hf 不应跑 Stage 0 RL")))

    cfg = _cfg(tmp_path)
    cfg["model_kind"] = "hf"
    cfg["teacher_rl_path"] = "RL_PATH"
    cfg["teacher_ref_path"] = "REF_PATH"
    # __init__ 有 hf+toy 数据守卫（P2-F），此处只测 _stage0_teachers → object.__new__ 绕过
    opd = object.__new__(FullStackOPDv2)
    opd.cfg = {**DEFAULT_CONFIG_V2, **cfg}
    opd.device = "cpu"
    teacher_rl, teacher_ref = opd._stage0_teachers()
    assert teacher_rl.path == "RL_PATH"
    assert teacher_ref.path == "REF_PATH"
    PL.stage0_small_rl.assert_not_called()


def test_stage0_teachers_hf_missing_ref_raises(monkeypatch):
    """HF 骨架：缺 teacher_ref_path → 显式 ModelError（hermetic：mock build_model 不触网）。"""
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    import fullstack_opd_v2.pipeline as PL
    cfg = _cfg(type("T", (), {"__truediv__": lambda self, o: str(o)})())
    cfg["model_kind"] = "hf"
    cfg["teacher_rl_path"] = "RL"
    cfg["teacher_ref_path"] = None
    opd = object.__new__(FullStackOPDv2)
    opd.cfg = {**DEFAULT_CONFIG_V2, **cfg}
    opd.device = "cpu"
    # 耗时修复：_stage0_teachers 会先 build_model(teacher_rl)→from_pretrained("RL") 触网
    # （hf-mirror 不可达时超时 7-50s/839s）。mock build_model 使本测试秒回、且仍验证
    # 「缺 teacher_ref_path → ModelError」这一被测行为。
    monkeypatch.setattr(PL, "build_model", lambda *a, **k: object())
    with pytest.raises(ModelError):
        opd._stage0_teachers()


def test_resume_warmup_uses_fresh_initial_student(tmp_path, monkeypatch):
    """P1-4：resume 时 warmup_student 是独立初始 student（非续跑学生权重）。

    run() 开头 torch.manual_seed(seed) → 两次 run 的 RNG 同流；warmup_student 在
    resume load_state_dict 之前创建 → 首跑与 resume 的 warmup 权重逐元素相等。
    且 resume 的 warmup 权重 ≠ 断点学生权重（未被续跑分布污染，否则曝光偏差不变式破）。
    """
    import os, tempfile, torch
    import fullstack_opd_v2.pipeline as PL
    from fullstack_opd_v2.checkpoint import CheckpointManager

    captured = []
    real = PL.stage1_build_cache
    def spy(*args, **kwargs):
        captured.append(kwargs.get("warmup_student"))
        return real(*args, **kwargs)
    monkeypatch.setattr(PL, "stage1_build_cache", spy)

    with tempfile.TemporaryDirectory() as td:
        tmp = type("T", (), {"__truediv__": lambda self, o: os.path.join(td, o)})()
        cfg = _cfg(tmp)
        cfg["stage2"]["n_steps"] = 4
        cfg["run"] = {"run_dir": os.path.join(td, "r"), "checkpoint_every": 2}
        FullStackOPDv2(cfg, device="cpu").run()
        ck = CheckpointManager(os.path.join(td, "r")).resume()
        assert ck is not None
        FullStackOPDv2(cfg, device="cpu").run(run_dir=os.path.join(td, "r"), resume=ck)

    w_first, w_resume = captured[0], captured[1]
    assert w_first is not None and w_resume is not None
    # 1) warmup 分布跨 resume 不变（同一初始 student 源）
    for k in w_first.state_dict():
        assert torch.allclose(w_first.state_dict()[k], w_resume.state_dict()[k]), k
    # 2) resume 的 warmup ≠ 断点学生权重（未被续跑分布污染）
    diffs = [k for k in ck["state"]
             if not torch.allclose(w_resume.state_dict()[k], ck["state"][k])]
    assert diffs, "warmup 权重与断点学生权重逐项相等——warmup 被续跑分布污染（P1-4 回归）"


def test_cloud_config_l1_and_seepage():
    """L2P3：CLOUD_CONFIG 结构性 L1 默认 + 顶层部署键下渗后 schema 合法。

    CLOUD_CONFIG 是脚本直传的云预设 dict（不经 load_config），两条防线：
    1) 自带 L1 翻转（warmup_M=4/student_init），否则云部署退回 L0 曝光偏差最大档；
    2) 顶层部署键（dtype/cache_mode/top_k_*/offload_to_cpu）经 _seep_deployment_keys
       分流后必须通过 OPDConfig 校验且落进 stage 子 dict——新增顶层键未同步分流表
       的 P0 bug 类型在此显式报错而非静默忽略。
    """
    from fullstack_opd_v2.config import OPDConfig, _seep_deployment_keys
    from fullstack_opd_v2.pipeline import CLOUD_CONFIG

    # 1) L1 结构性默认
    assert CLOUD_CONFIG["stage1"]["warmup_M"] == 4
    assert CLOUD_CONFIG["stage1"]["warmup_source"] == "student_init"

    # 2) 下渗 + 校验（extra="forbid" 下非法键会炸）
    seeped = _seep_deployment_keys(dict(CLOUD_CONFIG))
    OPDConfig(**seeped)                 # 不抛 = 云预设 schema 合法
    for k in ("cache_mode", "top_k_teacher"):
        assert seeped["stage1"][k] == CLOUD_CONFIG[k], k
    for k in ("dtype", "top_k_student", "offload_to_cpu"):
        assert seeped["stage2"][k] == CLOUD_CONFIG[k], k
    # ref_topk 保持纯顶层（不下渗、stage 无槽位）
    assert seeped["ref_topk"] == CLOUD_CONFIG["ref_topk"]
    assert "ref_topk" not in seeped["stage1"] and "ref_topk" not in seeped["stage2"]


def test_warmup_requires_student_raises_dataerror():
    """B1 收尾：warmup_source=student_init 但未传 warmup_student 时抛 DataError（非裸 ValueError）。"""
    import pytest
    from fullstack_opd_v2.exceptions import DataError, ModelError
    from fullstack_opd_v2.pipeline import stage1_build_cache
    import torch
    p = torch.zeros(4, 3, dtype=torch.long)
    r = torch.zeros(4, 2, dtype=torch.long)
    tr = CausalToyLM(vocab=16, d_model=8, n_layers=1)
    t = CausalToyLM(vocab=16, d_model=8, n_layers=1)
    cfg = {"cache_mode": "dense", "top_k_teacher": 0, "warmup_M": 2,
           "warmup_source": "student_init", "warmup_temperature": 1.0,
           "enforce_teacher_consistency": True,
           "cache_path": "x.pt", "build_batch_size": 4}
    with pytest.raises(DataError):
        stage1_build_cache(p, r, tr, t, cfg, warmup_student=None)


def test_l2_load_cache_loads_teachers_for_refresh(tmp_path):
    """回归：L2 启用 + load_cache=true 时，教师对必须补加载供刷新相位（否则 NoneType 崩溃）。

    load_cache 路径原本跳过 Stage 0、teacher_rl/ref=None；L2 刷新相位需要教师算
    disagreement（4 chosen logp），本测试验证补丁：跑通且有 rollout 相位行。
    """
    import os
    from fullstack_opd_v2.cache import TensorTeacherCache
    from fullstack_opd_v2.model import CausalToyLM
    from fullstack_opd_v2.pipeline import FullStackOPDv2, stage1_build_cache

    cache_path = os.path.join(str(tmp_path), "prebuilt.pt")
    cfg = _cfg(tmp_path)
    opd = FullStackOPDv2(cfg, device="cpu")
    teacher_rl, teacher_ref = opd._stage0_teachers()
    warmup_student = CausalToyLM(vocab=DEFAULT_CONFIG_V2["vocab_size"],
                                 d_model=DEFAULT_CONFIG_V2["d_model"],
                                 n_layers=DEFAULT_CONFIG_V2["n_layers"])
    s1 = dict(cfg["stage1"])
    s1["cache_path"] = cache_path
    stage1_build_cache(opd.prompts, opd.responses, teacher_rl, teacher_ref, s1,
                       warmup_student=warmup_student)

    cfg2 = _cfg(tmp_path)
    cfg2["stage1"]["cache_path"] = cache_path
    cfg2["stage1"]["load_cache"] = True
    cfg2["stage2"]["n_steps"] = 6
    cfg2["l2"] = {"enabled": True, "t_train": 3, "m_refresh": 4,
                  "cache": {"refresh_size": 8, "max_response_length": 4,
                            "refresh_min_interval": 2, "refresh_max_interval": 9}}
    out = FullStackOPDv2(cfg2, device="cpu").run()
    rollout_rows = [m for m in out["metrics"]
                    if isinstance(m, dict) and m.get("phase") == "rollout"]
    assert rollout_rows, "L2+load_cache 刷新相位未执行（教师未加载 / 门控仍误跳）"
    assert all(r.get("rollout/n_appended", 0) > 0 for r in rollout_rows)


def test_same_card_normalization():
    from fullstack_opd_v2.pipeline import _same_card
    assert _same_card("cuda:0", "cuda:0") is True
    assert _same_card("cuda:0", "cuda:1") is False
    assert _same_card("cuda", "cuda:0") is True   # "cuda" 归一为当前卡（CPU 测试回落 "0"）
    assert _same_card("cuda", "cuda:1") is False
    assert _same_card("cpu", "cuda:1") is False


def test_l2_rollout_mem_enough_split_card():
    """异卡：rollout 卡只要求引擎份额（0.9×96=86.4），不叠加训练侧 25GB。"""
    from fullstack_opd_v2.pipeline import _l2_rollout_mem_enough
    # 96GB 卡：引擎 86.4 + 训练 25 = 111.4 > 卡容量，同卡永远失败（符合预期）
    assert _l2_rollout_mem_enough(94.4, 86.4, 25.0) is False
    # 异卡 min_free=2.0：86.4+2=88.4 <= 94.4 → 通过
    assert _l2_rollout_mem_enough(94.4, 86.4, 2.0) is True
    assert _l2_rollout_mem_enough(85.0, 86.4, 2.0) is False


# --------------------------- P3 teacher offload + max_model_len 守卫（2026-08-19） ---------------------------

def _fake_teacher_model(device="cuda:0"):
    """记录 .to() 调用序列的 fake 教师模型。"""
    class _Fake:
        def __init__(self, device):
            self.device = device
            self.calls = []
        def to(self, d):
            self.calls.append(str(d))
            self.device = d
            return self
    return _Fake(device)


class _SilentLogger:
    def __init__(self):
        self.msgs = []
    def info(self, m):
        self.msgs.append(m)


def test_teacher_offload_default_false():
    """默认 teacher_offload=False（零回归）。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config()
    assert bool(cfg["stage2"].get("teacher_offload", False)) is False


def test_teacher_offload_config_parsed():
    """--set stage2.teacher_offload=true 能正确解析到 Stage2Cfg。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config(overrides=["stage2.teacher_offload=true"])
    assert cfg["stage2"]["teacher_offload"] is True


def test_refresh_chunk_default_and_config():
    """refresh_chunk 默认 4，--set stage2.refresh_chunk=2 可覆盖。"""
    from fullstack_opd_v2.config import load_config
    assert load_config()["stage2"]["refresh_chunk"] == 4
    assert load_config(overrides=["stage2.refresh_chunk=2"])["stage2"]["refresh_chunk"] == 2


def test_p3_teacher_move_offloads_models():
    """teacher_offload=true + cuda → teacher_rl/ref 搬 cpu；student_ref 不被触碰。"""
    from fullstack_opd_v2.pipeline import _p3_teacher_move
    rl = _fake_teacher_model("cuda:0")
    ref = _fake_teacher_model("cuda:0")
    student_ref = _fake_teacher_model("cuda:0")
    _p3_teacher_move(rl, ref, "cpu", enabled=True, device="cuda:0",
                     logger=_SilentLogger(), message="offload")
    assert rl.device == "cpu" and ref.device == "cpu"
    assert rl.calls == ["cpu"] and ref.calls == ["cpu"]
    # student_ref 未被 helper 触碰（它只接收 teacher_rl/teacher_ref）
    assert student_ref.calls == [] and student_ref.device == "cuda:0"


def test_p3_teacher_move_reload_on_refresh(monkeypatch):
    """refresh 相位：reload 搬回 GPU → 完成后 offload 回 CPU，empty_cache 被调用。"""
    from fullstack_opd_v2.pipeline import _p3_teacher_move
    import fullstack_opd_v2.pipeline as pl
    rl = _fake_teacher_model("cpu")
    ref = _fake_teacher_model("cpu")
    lg = _SilentLogger()
    ec = {"n": 0}
    # empty_cache 仅在 torch.cuda.is_available() 时被调用（_p3_teacher_move 有守卫）。
    # CPU-only 环境 is_available()=False → 不调用；CUDA 环境 reload+offload 各调一次。
    expected_ec = 2 if pl.torch.cuda.is_available() else 0
    if hasattr(pl.torch.cuda, "empty_cache"):
        def _ec():
            ec["n"] += 1
        monkeypatch.setattr(pl.torch.cuda, "empty_cache", _ec)
    _p3_teacher_move(rl, ref, "cuda:0", enabled=True, device="cuda:0", logger=lg, message="reload")
    assert rl.device == "cuda:0" and ref.device == "cuda:0"
    _p3_teacher_move(rl, ref, "cpu", enabled=True, device="cuda:0", logger=lg, message="offload")
    assert rl.device == "cpu" and ref.device == "cpu"
    assert lg.msgs == ["reload", "offload"]
    if expected_ec is not None:
        assert ec["n"] == expected_ec



def test_p3_teacher_move_exception_still_offloads():
    """refresh 相位抛异常时，finally 仍把教师 offload 回 CPU（try/finally 语义）。"""
    from fullstack_opd_v2.pipeline import _p3_teacher_move
    rl = _fake_teacher_model("cuda:0")
    ref = _fake_teacher_model("cuda:0")
    lg = _SilentLogger()
    try:
        # reload（模拟 refresh 相位入口）
        _p3_teacher_move(rl, ref, "cuda:0", enabled=True, device="cuda:0", logger=lg, message="reload")
        raise RuntimeError("fake rollout failure")   # 模拟 run_refresh_phase 抛异常
    except RuntimeError:
        pass
    finally:
        # pipeline 的 finally 必然执行 offload 回 CPU
        _p3_teacher_move(rl, ref, "cpu", enabled=True, device="cuda:0", logger=lg, message="offload")
    assert rl.device == "cpu" and ref.device == "cpu"
    assert lg.msgs == ["reload", "offload"]


def test_p3_teacher_move_noop_when_disabled():
    """enabled=False → no-op（不搬、不日志）。"""
    from fullstack_opd_v2.pipeline import _p3_teacher_move
    rl = _fake_teacher_model()
    ref = _fake_teacher_model()
    class _Bad:
        def info(self, *a):
            raise AssertionError("disabled 时不应调用日志")
    _p3_teacher_move(rl, ref, "cpu", enabled=False, device="cuda:0", logger=_Bad(), message="m")
    assert rl.calls == [] and ref.calls == []


def test_p3_teacher_move_noop_on_cpu_device():
    """非 cuda 设备 → no-op（CPU demo 零回归）。"""
    from fullstack_opd_v2.pipeline import _p3_teacher_move
    rl = _fake_teacher_model()
    class _Bad:
        def info(self, *a):
            raise AssertionError("cpu 设备不应调用日志")
    _p3_teacher_move(rl, None, "cpu", enabled=True, device="cpu", logger=_Bad(), message="m")
    assert rl.calls == []


def test_max_model_len_guard_rejects_too_small():
    """rollout_max_model_len=1024 < 1024+512=1536 → RuntimeError。"""
    from fullstack_opd_v2.pipeline import _check_rollout_max_model_len
    cfg = {"dataset": {"max_prompt_len": 1024}}
    s2cfg = {"rollout_max_model_len": 1024}
    l2cfg = {"rollout": {"max_new_tokens": 512}}
    with pytest.raises(RuntimeError, match="rollout_max_model_len"):
        _check_rollout_max_model_len(cfg, s2cfg, l2cfg)


def test_max_model_len_guard_accepts_valid():
    """rollout_max_model_len=2048 >= 1536 → 不抛，返回下限。"""
    from fullstack_opd_v2.pipeline import _check_rollout_max_model_len
    cfg = {"dataset": {"max_prompt_len": 1024}}
    s2cfg = {"rollout_max_model_len": 2048}
    l2cfg = {"rollout": {"max_new_tokens": 512}}
    assert _check_rollout_max_model_len(cfg, s2cfg, l2cfg) == 1536
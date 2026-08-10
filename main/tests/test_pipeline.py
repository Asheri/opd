"""pipeline.py 端到端冒烟 + L1 暖缓存（fat_N 计数、向后兼容、非法 source 抛错）、奖励趋势。"""
from __future__ import annotations

import pytest

from fullstack_opd_v2.exceptions import DataError
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
    from fullstack_opd_v2.exceptions import DataError
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

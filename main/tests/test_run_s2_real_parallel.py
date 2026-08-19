"""scripts/run_s2_real.py --parallel 纯函数单测（无 GPU / 无训练副作用）。

覆盖规格要求可测试的内核（脚本 modules 级纯函数，不触发任何训练/模型加载）：
- build_parallel_argv：2 实验双卡交叉分卡、3 实验 2 卡循环分配、4 卡循环回绕、
  显存收敛配置透传 + 强制 --load-cache + --names/--device/--parallel 规范化重写；
- merge_summaries：有/缺/损坏 summary.json 目录的合并、跳过与 stdout 警告；
- argparse：--parallel 可见、--parallel-child 隐藏内部标记可解析。
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_s2_real.py"


@pytest.fixture(scope="module")
def rs2():
    """以文件路径导入脚本（scripts/ 非包），得到模块级纯函数可测内核。"""
    spec = importlib.util.spec_from_file_location("run_s2_real", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def base_argv():
    """父进程 sys.argv 样例：--device cuda:3 特意与分配结果不同，用于验证不泄漏。"""
    return ["run_s2_real.py", "--config", "configs/skywork_17b.yaml",
            "--run-dir", "run_out",
            "--device", "cuda:3",
            "--names", "S2_E0_static", "S2_E1_opd512", "S2_E2_opd1024", "S2_E3_opd2048",
            "--n-steps", "30", "--eos-id", "151645"]


# ---------------------------------------------------------------- argv 解析辅助 ---
def _option_value(argv, opt):
    """取 --opt <value> 的单值（兼容 --opt=value 等号写法）。"""
    for i, tok in enumerate(argv):
        if tok == opt:
            return argv[i + 1]
    for tok in argv:
        if tok.startswith(opt + "="):
            return tok.split("=", 1)[1]
    raise AssertionError(f"{opt} 不在 argv: {argv}")


def _names(argv):
    """取 --names 的 nargs='+' 值列表。"""
    i = argv.index("--names")
    out = []
    j = i + 1
    while j < len(argv) and not argv[j].startswith("-"):
        out.append(argv[j])
        j += 1
    return out


def _sets(argv):
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--set"]


def _rollout_device(argv):
    """取最后一个 stage2.rollout_device（build_parallel_argv 追加在末尾 → 覆盖生效）。"""
    rds = [s.split("=", 1)[1] for s in _sets(argv)
           if s.startswith("stage2.rollout_device=")]
    assert rds, f"argv 缺少 stage2.rollout_device: {argv}"
    return rds[-1]


# ---------------------------------------------------------------- build_parallel_argv ---
def test_two_experiments_cross_cards(rs2, base_argv):
    """2 实验双卡交叉分卡：E1→训练 cuda:0/rollout cuda:1，E2→训练 cuda:1/rollout cuda:0。"""
    argv1 = rs2.build_parallel_argv(list(base_argv), "S2_E1_opd512", 0, 2)
    assert _names(argv1) == ["S2_E1_opd512"]
    assert _option_value(argv1, "--device") == "cuda:0"
    assert _rollout_device(argv1) == "cuda:1"
    assert "--load-cache" in argv1

    argv2 = rs2.build_parallel_argv(list(base_argv), "S2_E2_opd1024", 1, 2)
    assert _names(argv2) == ["S2_E2_opd1024"]
    assert _option_value(argv2, "--device") == "cuda:1"
    assert _rollout_device(argv2) == "cuda:0"
    assert "--load-cache" in argv2

    # 父 argv 原有 --device cuda:3 / 其它实验的 --names 都不能泄漏到子进程
    for argv in (argv1, argv2):
        assert "cuda:3" not in argv
        assert "--config" in argv and "--run-dir" in argv
    assert "S2_E2_opd1024" not in _names(argv1)   # E1 子进程只跑 E1
    assert "S2_E1_opd512" not in _names(argv2)   # E2 子进程只跑 E2


def test_three_experiments_round_robin_two_cards(rs2, base_argv):
    """3 实验循环分配到 2 卡：i=0→卡0、i=1→卡1、i=2→卡0（rollout 各自反向）。"""
    expects = [("S2_E0_static", "cuda:0", "cuda:1"),
               ("S2_E1_opd512", "cuda:1", "cuda:0"),
               ("S2_E2_opd1024", "cuda:0", "cuda:1")]
    for i, (name, card, rcard) in enumerate(expects):
        argv = rs2.build_parallel_argv(list(base_argv), name, i, 2)
        assert _option_value(argv, "--device") == card, f"{name} 训练卡"
        assert _rollout_device(argv) == rcard, f"{name} rollout 反向卡"
        assert _names(argv) == [name]


def test_round_robin_wraps_four_cards(rs2, base_argv):
    """4 卡循环回绕：i=3 → 训练 cuda:3、rollout 回到 cuda:0。"""
    argv = rs2.build_parallel_argv(list(base_argv), "S2_E3_opd2048", 3, 4)
    assert _option_value(argv, "--device") == "cuda:3"
    assert _rollout_device(argv) == "cuda:0"


def test_argv_preserves_memory_config_and_forces_load_cache(rs2):
    """显存收敛配置（--batch-size / --set offload/queue/staleness/rollout 引擎显存）逐子进程
    透传，且每个子进程都被强制追加 --load-cache，反向 rollout_device 覆盖用户显式值。"""
    mem_base = ["run_s2_real.py", "--config", "configs/skywork_17b.yaml",
                "--run-dir", "run_out", "--n-steps", "30",
                "--batch-size", "2",
                "--set", "stage2.offload_to_cpu=true",
                "--set", "stage2.queue_size=2",
                "--set", "stage2.staleness_queue_min=2",
                "--set", "stage2.rollout_engine=vllm",
                "--set", "stage2.rollout_gpu_mem=0.45",
                "--set", "stage2.rollout_device=cuda:9"]   # 用户显式值应被反向覆盖
    for name, i in [("S2_E1_opd512", 0), ("S2_E2_opd1024", 1)]:
        argv = rs2.build_parallel_argv(list(mem_base), name, i, 2)
        assert _option_value(argv, "--batch-size") == "2"
        sets = _sets(argv)
        for want in ["stage2.offload_to_cpu=true", "stage2.queue_size=2",
                     "stage2.staleness_queue_min=2", "stage2.rollout_engine=vllm",
                     "stage2.rollout_gpu_mem=0.45"]:
            assert want in sets, f"{name} 丢失 {want}"
        # 反向 rollout_device 追加在尾部（load_config 后者覆盖前者）
        assert _rollout_device(argv) == f"cuda:{(i + 1) % 2}", name
        assert "--load-cache" in argv
        assert _names(argv) == [name]


def test_build_parallel_argv_normalizes_forms(rs2):
    """--device=/--parallel= 等号写法与 --names A B 空白写法照常剔除并重写。"""
    argv = rs2.build_parallel_argv(
        ["run.py", "--config", "c.yaml", "--run-dir", "r",
         "--device=cuda:7", "--parallel=4", "--names", "A", "B",
         "--eos-id", "151645"],
        "B", 3, 4)
    assert _names(argv) == ["B"]
    assert _option_value(argv, "--device") == "cuda:3"
    assert _rollout_device(argv) == "cuda:0"
    assert "--parallel" not in argv        # 子进程保持串行（parallel_child 由父进程另传）
    assert "--load-cache" in argv
    assert _option_value(argv, "--eos-id") == "151645"   # 其余参数原样保留


def test_build_parallel_argv_rejects_zero_cards(rs2, base_argv):
    with pytest.raises(ValueError):
        rs2.build_parallel_argv(list(base_argv), "S2_E0_static", 0, 0)


# ---------------------------------------------------------------- merge_summaries ---
def test_merge_summaries_with_and_without_files(tmp_path, rs2, capsys):
    """有/缺/损坏/无 summary 字段的目录：合法条目合并，其余跳过并 stdout 警告。"""
    run_dir = tmp_path / "run"
    (run_dir / "E1_ok").mkdir(parents=True)
    (run_dir / "E1_ok" / "summary.json").write_text(json.dumps({
        "name": "E1_ok",
        "summary": {"reward_mean": 0.5, "n_steps": 3, "total_s": 12.3},
        "run_dir": str(run_dir / "E1_ok")}), encoding="utf-8")

    (run_dir / "E2_missing").mkdir()          # 缺 summary.json → 跳过 + 警告
    (run_dir / "E3_corrupt").mkdir()
    (run_dir / "E3_corrupt" / "summary.json").write_text("{not json", encoding="utf-8")
    (run_dir / "E4_no_field").mkdir()
    (run_dir / "E4_no_field" / "summary.json").write_text(
        json.dumps({"name": "E4_no_field"}), encoding="utf-8")

    merged = rs2.merge_summaries(str(run_dir))
    assert [m["name"] for m in merged] == ["E1_ok"]
    assert merged[0]["summary"] == {"reward_mean": 0.5, "n_steps": 3, "total_s": 12.3}
    assert merged[0]["run_dir"] == str(run_dir / "E1_ok")

    out = capsys.readouterr().out
    assert "E2_missing" in out and "缺失" in out
    assert "E3_corrupt" in out and "损坏" in out
    assert "E4_no_field" in out


def test_merge_summaries_ignores_non_dir_and_missing_dir(tmp_path, rs2):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "l2_experiment_summary.json").write_text("{}", encoding="utf-8")  # 根文件
    (run_dir / "E_ok").mkdir()
    (run_dir / "E_ok" / "summary.json").write_text(json.dumps(
        {"experiment": "E_ok", "summary": {"error": "boom"}}), encoding="utf-8")
    merged = rs2.merge_summaries(str(run_dir))
    assert [m["name"] for m in merged] == ["E_ok"]          # 根文件被忽略
    assert merged[0]["name"] == "E_ok"
    # experiment 字段回落为 name
    assert merged[0]["summary"] == {"error": "boom"}
    # 不存在的目录 → 空列表
    assert rs2.merge_summaries(str(tmp_path / "nope")) == []


# ---------------------------------------------------------------- argparse ---
def test_parallel_flag_parses(rs2, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_s2_real.py", "--config", "c.yaml",
                                      "--run-dir", "r", "--parallel", "3",
                                      "--names", "S2_E0_static"])
    args = rs2.parse_args()
    assert args.parallel == 3
    assert args.parallel_child is False
    # --parallel 默认 1
    monkeypatch.setattr(sys, "argv", ["run_s2_real.py", "--config", "c.yaml",
                                      "--run-dir", "r"])
    assert rs2.parse_args().parallel == 1


def test_printer_prefix_and_plain(rs2, capsys):
    """_printer：None → 原生 print 等价；带前缀 → [S2:<name>] 前缀。"""
    log = rs2._printer(None)
    log("hello")
    assert capsys.readouterr().out == "hello\n"

    log = rs2._printer("S2_E1_opd512")
    log("start", flush=True)
    assert capsys.readouterr().out == "[S2:S2_E1_opd512] start\n"


def test_printer_prefix_and_plain(rs2, capsys):
    """_printer：None → 原生 print 等价；带前缀 → [S2:<name>] 前缀。"""
    log = rs2._printer(None)
    log("hello")
    assert capsys.readouterr().out == "hello\n"

    log = rs2._printer("S2_E1_opd512")
    log("start", flush=True)
    assert capsys.readouterr().out == "[S2:S2_E1_opd512] start\n"


def test_parallel_child_flag_parses(rs2, monkeypatch):
    """--parallel-child 为隐藏内部标记（父进程 --parallel 追加），可解析且默认关闭。"""
    monkeypatch.setattr(sys, "argv", ["run_s2_real.py", "--config", "c.yaml",
                                      "--run-dir", "r", "--names", "S2_E0_static",
                                      "--parallel-child"])
    args = rs2.parse_args()
    assert args.parallel_child is True


# ---------------------------------------------------------------- _run_serial 分支 ---
def _fake_run_experiment(args, name, load_cache, prefix=None):
    """stub：不触 config/模型，只按名字建目录并返回固定结果（供分支行为单测）。"""
    d = os.path.join(args.run_dir, name)
    os.makedirs(d, exist_ok=True)
    summary = {"experiment": name, "reward_mean": 0.0, "n_steps": 1,
               "total_s": 1.0, "error": None if load_cache else "build"}
    return {"name": name, "summary": summary, "run_dir": d}


def test_run_serial_parallel_child_writes_only_experiment_summary(rs2, monkeypatch, tmp_path):
    """并行子进程分支：只写 run-dir/<实验名>/summary.json，绝不写共享 l2_experiment_summary.json。"""
    monkeypatch.setattr(rs2, "_run_experiment", _fake_run_experiment)
    args = argparse.Namespace(run_dir=str(tmp_path), parallel_child=True, load_cache=True)

    rs2._run_serial(args, ["S2_E1_opd512"])

    exp_json = tmp_path / "S2_E1_opd512" / "summary.json"
    assert exp_json.is_file()
    data = json.loads(exp_json.read_text(encoding="utf-8"))
    assert data["name"] == "S2_E1_opd512"
    assert data["summary"]["n_steps"] == 1
    assert not (tmp_path / "l2_experiment_summary.json").exists()   # 共享文件由父进程写


def test_run_serial_normal_writes_shared_dict(rs2, monkeypatch, tmp_path):
    """N=1 串行分支：保持改造前语义——写共享 dict {实验名: summary}，不写每实验 summary.json。"""
    monkeypatch.setattr(rs2, "_run_experiment", _fake_run_experiment)
    args = argparse.Namespace(run_dir=str(tmp_path), parallel_child=False, load_cache=False)

    rs2._run_serial(args, ["S2_E0_static", "S2_E1_opd512"])

    shared = tmp_path / "l2_experiment_summary.json"
    assert shared.is_file()
    data = json.loads(shared.read_text(encoding="utf-8"))
    assert set(data) == {"S2_E0_static", "S2_E1_opd512"}
    assert data["S2_E0_static"]["n_steps"] == 1
    # 串行路径不写每实验 summary.json
    assert not (tmp_path / "S2_E0_static" / "summary.json").exists()
    assert not (tmp_path / "S2_E1_opd512" / "summary.json").exists()


def test_run_serial_parallel_child_guards_uncaught_exceptions(rs2, monkeypatch, tmp_path):
    """并行子进程兜底：单实验内部除处理器再抛时仍写出含 error 的 summary.json（0 退出）。"""
    def boom(args, name, load_cache, prefix=None):
        raise RuntimeError("故意未捕获异常")
    monkeypatch.setattr(rs2, "_run_experiment", boom)
    args = argparse.Namespace(run_dir=str(tmp_path), parallel_child=True, load_cache=True)

    rs2._run_serial(args, ["S2_E1_opd512"])   # 不应再向上抛

    data = json.loads((tmp_path / "S2_E1_opd512" / "summary.json").read_text(encoding="utf-8"))
    assert data["summary"]["error"] == "故意未捕获异常"
    assert not (tmp_path / "l2_experiment_summary.json").exists()


def test_run_serial_normal_rethrows_uncaught(rs2, monkeypatch, tmp_path):
    """N=1 串行：_run_experiment 意外抛出（load_config 配置错误）保持改造前语义向上传播。"""
    def boom(args, name, load_cache, prefix=None):
        raise RuntimeError("配置错误")
    monkeypatch.setattr(rs2, "_run_experiment", boom)
    args = argparse.Namespace(run_dir=str(tmp_path), parallel_child=False, load_cache=False)

    with pytest.raises(RuntimeError, match="配置错误"):
        rs2._run_serial(args, ["S2_E0_static"])
    assert not (tmp_path / "l2_experiment_summary.json").exists()   # 传播前不写汇总


# ---------------------------------------------------------------- --stagger（2026-08-19） ---
def test_stagger_default_zero(rs2, monkeypatch):
    """--stagger 不传时默认 0（parse_args）。"""
    monkeypatch.setattr(sys, "argv", ["run_s2_real.py", "--config", "c.yaml",
                                      "--run-dir", "r", "--names", "S2_E1_opd512"])
    args = rs2.parse_args()
    assert args.stagger == 0.0


def test_stagger_strips_from_child_argv(rs2, base_argv):
    """--stagger 45 不透传到子进程 argv（只有父进程用）。"""
    argv = list(base_argv) + ["--stagger", "45"]
    child = rs2.build_parallel_argv(argv, "S2_E1_opd512", 0, 2)
    assert "--stagger" not in child
    assert "45" not in child


def test_stagger_sleeps_between_children(rs2, monkeypatch):
    """--parallel 2 --stagger 30：第 1 个子进程立即启动，sleep(30) 后再启动第 2 个。"""
    import time
    import multiprocessing
    sleeps = []
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    # 用一个假的 ctx 记录 p.start() 调用顺序
    started = []
    class _FakeProc:
        def __init__(self, *a, **k):
            self.name = k.get("name", "?")
            self.exitcode = 0
        def start(self):
            started.append(self.name)
        def join(self, *a, **k):
            pass

    monkeypatch.setattr(multiprocessing, "get_context", lambda *a, **k: type("C", (), {"Process": _FakeProc})())
    # 直接调 _run_parallel 的循环逻辑（通过模块函数）
    from types import SimpleNamespace
    args = SimpleNamespace(stagger=30.0, parallel=2, config="c.yaml", run_dir="r",
                           load_cache=True, cache_path=None, eos_id=None,
                           materialized=0, m_refresh=8, refresh_min=10,
                           refresh_size=None, batch_size=None, extra_sets=[],
                           names=None, device="cuda:0", n_steps=20,
                           parallel_child=False, refresh_cold=0)
    # 需要 names 合法 + cache 存在（否则走 build 路径）——这里只验证 sleep 行为，
    # 通过 monkeypatch _resolve_cache_path/_cache_exists/_run_experiment 短路。
    rs2._resolve_cache_path = lambda a, n: "x.pt"
    rs2._cache_exists = lambda p: True
    rs2._spawn_entry = lambda argv: None
    rs2._finish_parallel = lambda a, n_failed=0: None
    rs2._cleanup_stray_engines = lambda: None
    names = ["S2_E1_opd512", "S2_E2_opd1024"]
    rs2._run_parallel(args, names)
    # 第 1 个立即启动，第 2 个在 sleep 后 → 只 sleep 一次（30s）
    assert started == ["S2-S2_E1_opd512", "S2-S2_E2_opd1024"]
    assert len(sleeps) == 1
    assert sleeps[0] == 30.0
"""run.py 单测：run 目录管理。"""
import os

from fullstack_opd_v2.run import RunManager


def _cfg():
    return {"vocab_size": 64, "n_prompts": 16, "run": {"seed": 42}}


def test_run_manager_creates_run_dir_and_snapshot(tmp_path):
    rm = RunManager(_cfg(), run_dir=str(tmp_path / "exp1"))
    paths = rm.create()
    assert os.path.isdir(paths["run_dir"])
    assert os.path.isdir(paths["logs"])
    assert os.path.isdir(paths["checkpoints"])
    assert os.path.isfile(paths["config"])


def test_run_manager_auto_timestamp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rm = RunManager(_cfg(), run_dir=None)
    paths = rm.create()
    assert os.path.basename(os.path.dirname(paths["run_dir"])) == "runs"


def test_run_manager_paths_consistent(tmp_path):
    rm = RunManager(_cfg(), run_dir=str(tmp_path / "exp2"))
    paths = rm.create()
    assert paths["log_file"] == os.path.join(paths["logs"], "train.log")
    assert paths["metrics_csv"] == os.path.join(paths["run_dir"], "metrics.csv")
    assert paths["checkpoint_dir"] == paths["checkpoints"]


def test_run_manager_create_idempotent(tmp_path):
    """C5：重复 create 不崩，config.yaml 每次重写为最新 cfg。"""
    rm = RunManager(_cfg(), run_dir=str(tmp_path / "r"))
    rm.create()
    cfg2 = {**_cfg(), "vocab_size": 128}      # 新 cfg
    p2 = RunManager(cfg2, run_dir=str(tmp_path / "r")).create()
    import yaml
    snap = yaml.safe_load(open(p2["config"], encoding="utf-8"))
    assert snap["vocab_size"] == 128          # config 被重写为最新
    assert os.path.isdir(p2["logs"]) and os.path.isdir(p2["checkpoints"])
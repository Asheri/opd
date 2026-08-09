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
"""checkpoint.py 单测：断点保存/加载/续跑。"""
import os

import pytest
import torch

from fullstack_opd_v2.checkpoint import CheckpointManager
from fullstack_opd_v2.exceptions import CheckpointError
from fullstack_opd_v2.model import CausalToyLM


def _ckpt_dir(tmp_path):
    return os.path.join(str(tmp_path), "checkpoints")


def test_save_load_roundtrip(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(10, m, version=20, cfg={"a": 1})
    ck = cm.load(cm.latest())
    assert ck["step"] == 10
    assert ck["version"] == 20
    assert ck["cfg"] == {"a": 1}
    # 恢复 state_dict
    m2 = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    m2.load_state_dict(ck["state"])
    assert torch.equal(m2.emb.weight, m.emb.weight)


def test_latest_picks_highest_step(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(10, m, version=10, cfg={})
    cm.save(30, m, version=30, cfg={})
    latest = cm.latest()
    assert os.path.basename(latest) == "step_30.pt"


def test_every_throttle(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=3)
    for s in (1, 2, 3, 4, 5, 6):
        cm.save(s, m, version=s, cfg={})
    # 只存 step%3==0 的
    assert cm.latest() is not None
    assert os.path.basename(cm.latest()) == "step_6.pt"


def test_resume(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    cm.save(15, m, version=15, cfg={"seed": 7})
    ck = CheckpointManager(str(tmp_path), every=1).resume()
    assert ck["step"] == 15
    assert ck["cfg"] == {"seed": 7}


def test_manager_constructed_with_dir_but_loads_file(tmp_path):
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    p = cm.save(3, m, version=3, cfg={}, force=True)
    assert p is not None
    # 镜像 cli eval 调用形态：run_dir 传 "."，checkpoint_dir 显式给文件父目录
    cm2 = CheckpointManager(".", checkpoint_dir=os.path.dirname(p))
    ck = cm2.load(p)                                 # 文件路径 load
    assert ck["step"] == 3


def test_save_load_ref_anchors(tmp_path):
    """A3/D4：save 带 ref（KL 锚点）随断点落盘，load 原样返回。"""
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=1)
    p = cm.save(3, m, version=3, cfg={},
                ref={"ref_dists": torch.ones(2, 3, 4)}, force=True)
    ck = cm.load(p)
    assert torch.equal(ck["ref"]["ref_dists"], torch.ones(2, 3, 4))


def test_force_save_ignores_throttle(tmp_path):
    """D3：force=True 时即使 step 不满足 every 节流也必须落盘。"""
    m = CausalToyLM(vocab=64, d_model=48, n_layers=2)
    cm = CheckpointManager(str(tmp_path), every=10)
    p = cm.save(3, m, version=3, cfg={}, force=True)
    assert p is not None


def test_resume_empty_returns_none(tmp_path):
    """D3：空 run 目录（无任何断点）resume 返回 None 而非抛错。"""
    assert CheckpointManager(str(tmp_path), every=1).resume() is None


def test_load_missing_raises(tmp_path):
    """D3：加载不存在的断点文件抛 CheckpointError（精确类型可捕获）。"""
    with pytest.raises(CheckpointError):
        CheckpointManager(str(tmp_path)).load(str(tmp_path / "nope.pt"))
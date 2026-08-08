"""config.py 单测：YAML 真加载、schema 校验（未知键/非法值报错）、点分覆盖、端到端可用。"""
from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.pipeline import DEFAULT_CONFIG_V2, FullStackOPDv2

YAML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "configs", "fullstack_opd.yaml")


def test_load_defaults_no_args():
    cfg = load_config()
    assert cfg["stage2"]["lr"] == DEFAULT_CONFIG_V2["stage2"]["lr"]
    assert cfg["stage1"]["warmup_source"] == "none"
    assert cfg["vocab_size"] == 64


def test_load_real_yaml():
    cfg = load_config(path=YAML)
    assert cfg["stage1"]["cache_path"] == "fullstack_opd_cache_v2.pt"
    assert cfg["stage2"]["scheduling_mode"] == "fully_async"
    assert cfg["stage2"]["batch_size"] == 8


def test_unknown_top_level_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("vocab_size: 64\nbogus_key: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path=str(bad))


def test_unknown_nested_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage2:\n  n_steps: 5\n  bogus: 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path=str(bad))


def test_bad_enum_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("dtype: fp16\n", encoding="utf-8")   # 非法枚举
    with pytest.raises(ValidationError):
        load_config(path=str(bad))


def test_dotted_overrides():
    cfg = load_config(overrides=["stage2.n_steps=50", "stage1.warmup_source=mix",
                                 "stage1.warmup_M=4", "top_k_student=256"])
    assert cfg["stage2"]["n_steps"] == 50
    assert cfg["stage1"]["warmup_source"] == "mix"
    assert cfg["stage1"]["warmup_M"] == 4
    assert cfg["top_k_student"] == 256


def test_override_unknown_key_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides=["stage2.nonexistent=1"])


def test_override_bad_value_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides=["stage2.rollout_engine=foo"])


def test_loaded_config_runs_end_to_end(tmp_path):
    """加载后的配置可直接喂给 FullStackOPDv2 跑通（集成校验）。"""
    cfg = load_config(overrides=[
        "n_prompts=8",
        "stage0.n_rl_steps=3",
        "stage2.n_steps=4",
        "stage2.batch_size=4",
        f"stage1.cache_path={tmp_path / 'c.pt'}",
    ])
    out = FullStackOPDv2(cfg, device="cpu").run()
    assert len(out["metrics"]) == 4

"""export_sanity_check.py（E-0b）与 onpolicy_share.py（E-0d）纯函数单测。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from export_sanity_check import config_diff  # noqa: E402
from onpolicy_share import _detect_phase_col, pool_share  # noqa: E402


def test_config_diff_no_diff():
    a = {"hidden_size": 2048, "num_layers": 28, "vocab_size": 151643}
    assert config_diff(a, dict(a)) == {}


def test_config_diff_ignores_harmless():
    a = {"_name_or_path": "/old/path", "model_type": "qwen3", "hidden_size": 2048}
    b = {"_name_or_path": "/new/path", "model_type": "qwen3", "hidden_size": 2048}
    assert config_diff(a, b) == {}


def test_config_diff_reports_real_diff():
    a = {"hidden_size": 2048, "vocab_size": 151643}
    b = {"hidden_size": 2048, "vocab_size": 151936}
    d = config_diff(a, b)
    assert d == {"vocab_size": (151643, 151936)}


def test_config_diff_missing_key():
    a = {"hidden_size": 2048}
    b = {"hidden_size": 2048, "num_attention_heads": 16}
    d = config_diff(a, b)
    assert d == {"num_attention_heads": (None, 16)}


def test_detect_phase_col():
    assert _detect_phase_col(["step", "pool", "loss"]) == "pool"
    assert _detect_phase_col(["step", "phase", "loss"]) == "phase"
    assert _detect_phase_col(["step", "loss"]) is None


def test_pool_share_base_refresh():
    rows = [{"step": str(i), "pool": "base" if i % 3 else "refresh"} for i in range(12)]
    res = pool_share(rows, "pool")
    assert res["base_steps"] == 8
    assert res["refresh_steps"] == 4
    assert res["base_share"] == round(8 / 12, 4)
    assert res["refresh_share"] == round(4 / 12, 4)


def test_pool_share_no_phase_col():
    rows = [{"step": "1", "loss": "0.1"}]
    res = pool_share(rows, None)
    assert res["phase_col"] is None
    assert res["base_share"] is None      # 不伪造
    assert "available_cols" in res


def test_pool_share_empty_rows():
    res = pool_share([], "pool")
    assert res["base_steps"] == 0
    assert res["refresh_steps"] == 0
    assert res["base_share"] == 0.0

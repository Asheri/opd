"""model_factory.py 单测：可插拔模型工厂。"""
import pytest

from fullstack_opd_v2.model import CausalToyLM
from fullstack_opd_v2.model_factory import build_model
from fullstack_opd_v2.exceptions import ModelError


def _cfg(**over):
    cfg = dict(vocab_size=64, d_model=48, n_layers=2, model_kind="toy")
    cfg.update(over)
    return cfg


def test_build_model_toy_returns_causal_toy_lm():
    m = build_model(_cfg(), "cpu")
    assert isinstance(m, CausalToyLM)
    assert m.vocab == 64


def test_build_model_unknown_kind_raises():
    with pytest.raises(ModelError):
        build_model(_cfg(model_kind="hf"), "cpu")
"""delta_correctness_corr.py（E-1b Δ↔correct 相关性）纯函数单测。

不依赖 vLLM/GPU/scipy（scipy 缺失时 correlate 降级 ρ=0，有测试覆盖）。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from delta_correctness_corr import (  # noqa: E402
    _auc_pos_neg, _extract_prompt_logprobs, compute_delta, correlate, judge_response,
)


class _FakeLogprob:
    """vLLM Logprob 的最小替身（.logprob 属性）。"""
    def __init__(self, logprob):
        self.logprob = logprob


def test_extract_prompt_logprobs_basic():
    """vLLM prompt_logprobs 结构 → 每 token 条件 logprob 列表。"""
    pl = [
        {151643: _FakeLogprob(-0.1)},
        {100: _FakeLogprob(-0.2)},
        None,                       # 位置无 logprob → None
        {200: _FakeLogprob(-0.4)},
    ]
    got = _extract_prompt_logprobs(pl)
    assert got == [-0.1, -0.2, None, -0.4]


def test_extract_prompt_logprobs_empty():
    assert _extract_prompt_logprobs([]) == []
    assert _extract_prompt_logprobs([{}, None, {7: _FakeLogprob(0.0)}]) == [None, None, 0.0]


def test_compute_delta_basic():
    """response token 上的序列级 Δ：Σ(rl−ref) 与 per-token 均值。"""
    rl = [-1.0, -1.0, -1.0, -1.0]        # prompt 3 token + response 1 token
    ref = [-1.0, -1.0, -1.0, -2.0]       # response 位 rl 比 ref 高 1.0
    d = compute_delta(rl, ref, start_idx=3)
    assert d["delta_sum"] == 1.0
    assert d["delta_mean"] == 1.0
    assert d["n_tokens"] == 1


def test_compute_delta_multi_token():
    rl = [-1.0] * 5
    ref = [-1.0, -1.0, -1.0, -1.5, -0.5]  # response 2 token：+0.5, −0.5 → 和 0.0、均值 0.0
    d = compute_delta(rl, ref, start_idx=3)
    assert d["delta_sum"] == 0.0
    assert d["delta_mean"] == 0.0
    assert d["n_tokens"] == 2


def test_compute_delta_none_nan_skip():
    """None/NaN 的 token 跳过，不计入 n_tokens。"""
    rl = [-1.0, -1.0, -1.0, float("nan"), -1.0, None]
    ref = [-1.0, -1.0, -1.0, -1.0, -2.0, -2.0]
    d = compute_delta(rl, ref, start_idx=3)
    # token3 nan 跳过、token4 有效(+1.0)、token5 rl=None 跳过
    assert d["n_tokens"] == 1
    assert d["delta_sum"] == 1.0
    assert d["delta_mean"] == 1.0


def test_compute_delta_length_mismatch():
    """rl/ref 长度不一：只统计两者都有的位置。"""
    rl = [-1.0, -1.0, -1.0, -1.0, -1.0]
    ref = [-1.0, -1.0, -1.0, -2.0]      # 短一截
    d = compute_delta(rl, ref, start_idx=3)
    assert d["n_tokens"] == 1
    assert d["delta_sum"] == 1.0


def test_compute_delta_no_response():
    d = compute_delta([-1.0] * 3, [-1.0] * 3, start_idx=3)
    assert d["n_tokens"] == 0
    assert d["delta_sum"] == 0.0
    assert d["delta_mean"] == 0.0


def test_auc_pos_neg_perfect():
    """正确样本 Δ 全高于错误样本 → AUC=1.0。"""
    deltas = [1.0, 2.0, 3.0]        # 正确（正 Δ）
    wrong = [-1.0, -2.0, -3.0]      # 错误（负 Δ）
    corrects = [True, True, True, False, False, False]
    assert _auc_pos_neg(deltas + wrong, corrects) == 1.0


def test_auc_pos_neg_reverse():
    deltas = [-1.0, -2.0, -3.0]
    wrong = [1.0, 2.0, 3.0]
    corrects = [True, True, True, False, False, False]
    assert _auc_pos_neg(deltas + wrong, corrects) == 0.0


def test_auc_pos_neg_single_class():
    assert _auc_pos_neg([1.0, 2.0], [True, True]) == 0.5   # 无负样本 → 0.5


def test_correlate_perfect_positive():
    """Δ 与 correct 强正相关 → ρ>0.8、AUC=1（二值 correct 有 tie，ρ 到不了 1.0）。"""
    deltas = [i / 10 for i in range(1, 11)]
    corrects = [i >= 6 for i in range(1, 11)]
    s = correlate(deltas, corrects)
    assert s["n"] == 10
    assert s["spearman_rho"] > 0.8
    assert s["auc"] == 1.0


def test_correlate_reverse():
    deltas = [i / 10 for i in range(1, 11)]
    corrects = [i <= 5 for i in range(1, 11)]
    s = correlate(deltas, corrects)
    assert s["spearman_rho"] < -0.8


def test_correlate_small_n():
    s = correlate([0.1, 0.2], [True, False])
    assert s["n"] == 2
    assert s["spearman_rho"] == 0.0
    assert s["auc"] == 0.5


def test_judge_response_boxed_correct():
    """boxed 正确答案 + sympy 等价 → True。"""
    assert judge_response("reasoning\\n\\\\boxed{42}", "42") is True
    assert judge_response("simplify\\n\\\\boxed{0.5}", "1/2") is True


def test_judge_response_no_answer():
    assert judge_response("no answer here", "42") is False
    assert judge_response("", "42") is False

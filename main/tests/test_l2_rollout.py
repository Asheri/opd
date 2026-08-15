"""Stage 2：短 Rollout OPD 训练协议单测。

- config：L2RolloutCfg 默认/覆盖/拒未知键 + workload 下渗。
- model：detect_loop / generate_with_status / build_length_mask（不动 generate_batch）。
- rollout_vllm：parse_vllm_outputs 纯函数（eos/budget_stop/loop）。
- adaptive_cache：run_refresh_phase 注入 generator + loop 跳过 + ring buffer status 往返。
- pipeline：消费 l2.rollout.max_new_tokens + fallback + status 指标。
- experiment：STAGE2_ROLLOUT_MATRIX + build_config/run_matrix 泛化。
- report_stage2：Q1-Q4 报告。
"""
import pytest

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError


# --------------------------- 任务1：config L2RolloutCfg ---------------------------
def test_l2_rollout_defaults():
    cfg = load_config(overrides=["l2.enabled=true"])
    rollout = cfg["l2"]["rollout"]
    assert rollout["max_new_tokens"] == 1024
    assert rollout["allow_budget_stop"] is True
    assert rollout["eos_token_id"] is None        # 默认不判 EOS
    assert rollout["loop_detection"] is True
    assert rollout["pad_id"] == 0


def test_l2_rollout_overrides():
    cfg = load_config(overrides=["l2.rollout.max_new_tokens=2048",
                                 "l2.rollout.eos_token_id=0"])
    assert cfg["l2"]["rollout"]["max_new_tokens"] == 2048
    assert cfg["l2"]["rollout"]["eos_token_id"] == 0


def test_l2_rollout_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.rollout.unknown=1"])


def test_l2_rollout_disabled_default():
    # l2 默认全关时 rollout 子段仍存在（默认值 1024）
    cfg = load_config()
    assert cfg["l2"]["enabled"] is False
    assert cfg["l2"]["rollout"]["max_new_tokens"] == 1024


# --------------------------- 任务2：model.py 短 rollout ---------------------------
import torch

from fullstack_opd_v2.model import (
    CausalToyLM, build_length_mask, detect_loop, generate_batch,
    generate_with_status,
)


def test_detect_loop_true():
    r = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])          # 尾部周期 3
    assert detect_loop(r, periods=(2, 3, 4), min_len=6)


def test_detect_loop_false():
    r = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    assert not detect_loop(r, periods=(2, 3, 4), min_len=6)


def test_detect_loop_short_sequence_no_false_positive():
    # 短序列（<min_len）不误报
    r = torch.tensor([1, 2, 1, 2])
    assert not detect_loop(r, periods=(2, 3, 4), min_len=8)


def test_generate_with_status_no_eos_all_budget():
    m = CausalToyLM(vocab=64, max_len=64)
    pr = torch.randint(0, 64, (2, 5))
    out = generate_with_status(m, pr, max_new=8, eos_token_id=None)
    assert out["statuses"] == ["budget_stop", "budget_stop"]
    assert out["lengths"] == [8, 8]
    assert out["eos_pos"] == [None, None]
    assert out["looped"] == [False, False]
    mask = build_length_mask(out["responses"], out["lengths"], out["eos_pos"])
    assert mask.size() == (2, 8)
    assert mask.sum(1).tolist() == [8, 8]


def test_generate_with_status_eos_stops(tmp_path, monkeypatch):
    # 注入固定采样：首 token=0（eos）→ 提前停，mask 在 eos 后全 0
    import torch as _t
    m = CausalToyLM(vocab=64, max_len=64)
    pr = torch.randint(0, 64, (1, 5))
    # monkeypatch torch.multinomial 返回首步 0、其余 1
    calls = {"n": 0}

    def fake_multinomial(probs, num_samples=1):
        calls["n"] += 1
        if calls["n"] == 1:
            return _t.tensor([[0]])
        return _t.tensor([[1]])

    monkeypatch.setattr(_t, "multinomial", fake_multinomial)
    out = generate_with_status(m, pr, max_new=8, eos_token_id=0)
    assert out["statuses"] == ["eos"]
    assert out["lengths"] == [1]                 # eos_pos(0)+1 含 eos
    assert out["eos_pos"] == [0]
    mask = build_length_mask(out["responses"], out["lengths"], out["eos_pos"])
    assert mask.sum(1).tolist() == [1]           # eos 后全 0


def test_generate_with_status_loop_detected(tmp_path, monkeypatch):
    # 构造周期 3 重复输出 → 判 loop
    import torch as _t
    m = CausalToyLM(vocab=64, max_len=64)
    pr = torch.randint(0, 64, (1, 5))
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]            # 周期 3 重复（9 tokens）
    it = iter(seq)
    monkeypatch.setattr(_t, "multinomial",
                        lambda probs, num_samples=1: _t.tensor([[next(it)]]))

    out = generate_with_status(m, pr, max_new=9, eos_token_id=None,
                               loop_detection=True, loop_periods=(3,))
    assert out["statuses"] == ["loop"]
    assert out["looped"] == [True]


def test_generate_batch_unchanged():
    # 回归：generate_batch 行为不变（Stage 0/1 依赖）
    m = CausalToyLM(vocab=64, max_len=64)
    pr = torch.randint(0, 64, (2, 5))
    out = generate_batch(m, pr, max_new=4)
    assert out.size() == (2, 4)
    assert out.dtype == torch.long


# --------------------------- 任务3：vLLM 状态解析（纯函数） ---------------------------
from fullstack_opd_v2.rollout_vllm import parse_vllm_outputs


class _FakeOut:
    """mock vLLM RequestOutput：outputs[0].token_ids 为生成部分。"""
    def __init__(self, toks, finish_reason="length"):
        class _Comp:
            def __init__(self, toks, fr):
                self.token_ids = toks
                self.finish_reason = fr
        self.outputs = [_Comp(toks, finish_reason)]
        self.prompt_len = 0


def test_parse_vllm_outputs_eos():
    outs = [_FakeOut([1, 2, 0, 3], "stop")]        # eos=0 在第 2 位
    r = parse_vllm_outputs(outs, max_new=8, eos_token_id=0)
    assert r["statuses"] == ["eos"]
    assert r["lengths"] == [3]                       # eos_pos+1 含 eos
    assert r["eos_pos"] == [2]
    assert r["looped"] == [False]


def test_parse_vllm_outputs_budget_stop():
    outs = [_FakeOut([1, 2, 3], "length")]           # 无 eos，撞 max_new
    r = parse_vllm_outputs(outs, max_new=8, eos_token_id=0)
    assert r["statuses"] == ["budget_stop"]
    assert r["lengths"] == [3]
    assert r["eos_pos"] == [None]


def test_parse_vllm_outputs_loop():
    outs = [_FakeOut([1, 2, 3, 1, 2, 3, 1, 2, 3], "length")]   # 周期 3 重复
    r = parse_vllm_outputs(outs, max_new=16, eos_token_id=None,
                           loop_periods=(3,))
    assert r["statuses"] == ["loop"]
    assert r["looped"] == [True]


def test_parse_vllm_outputs_loop_disabled():
    outs = [_FakeOut([1, 2, 3, 1, 2, 3, 1, 2, 3], "length")]
    r = parse_vllm_outputs(outs, max_new=16, eos_token_id=None,
                           loop_detection=False)
    assert r["statuses"] == ["budget_stop"]          # 关闭 loop 检测 → 不判 loop
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



def test_l2_rollout_loop_periods_default():
    """IMP-1b：L2RolloutCfg 默认 loop_periods=(2,3,4)（原 detect_loop 硬编码值）。"""
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["loop_periods"] == (2, 3, 4)
    cfg0 = load_config()          # l2 关闭时默认仍存在
    assert cfg0["l2"]["rollout"]["loop_periods"] == (2, 3, 4)


def test_l2_rollout_loop_periods_override():
    """IMP-1b：loop_periods 可经 --set 点分覆盖（逗号 / 括号 / 方括号语法）。"""
    cfg = load_config(overrides=["l2.rollout.loop_periods=2,4,6"])
    assert cfg["l2"]["rollout"]["loop_periods"] == (2, 4, 6)
    cfg2 = load_config(overrides=["l2.rollout.loop_periods=(3,5)"])
    assert cfg2["l2"]["rollout"]["loop_periods"] == (3, 5)
    cfg3 = load_config(overrides=["l2.rollout.loop_periods=[4,6,8]"])
    assert cfg3["l2"]["rollout"]["loop_periods"] == (4, 6, 8)


def test_l2_rollout_loop_periods_bad_value_rejected():
    """IMP-1b：loop_periods 非 int 序列 → pydantic 校验拒绝（不静默吞）。"""
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.rollout.loop_periods=2,x,4"])


def test_l2_rollout_temperature_default():
    """IMP-1a：L2RolloutCfg 默认 temperature=0.7；可覆盖到 1.0（复现旧行为）。"""
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["temperature"] == 0.7
    cfg2 = load_config(overrides=["l2.rollout.temperature=1.0"])
    assert cfg2["l2"]["rollout"]["temperature"] == 1.0


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


def test_detect_loop_default_periods_detect_2_3_4():
    """IMP-1b：detect_loop 默认 periods=(2,3,4) 下，周期 2/3/4 重复都判 loop。"""
    assert detect_loop(torch.tensor([1, 2, 1, 2, 1, 2, 1, 2]))                 # 周期 2
    assert detect_loop(torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3]))              # 周期 3
    assert detect_loop(torch.tensor([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]))     # 周期 4
    # 周期 5（不在默认集合）不判 loop
    assert not detect_loop(torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))


def test_detect_loop_custom_periods_override():
    """IMP-1b：自定义 periods（如校准后 (5,)）替换默认集合生效。"""
    r = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert detect_loop(r, periods=(5,))            # 自定义 5 判 loop
    assert not detect_loop(r, periods=(2, 3, 4))   # 默认集合不判周期 5


def test_generate_with_status_loop_periods_custom(tmp_path, monkeypatch):
    """IMP-1b：generate_with_status 透传 loop_periods——周期 5 输出在 periods=(5,) 判 loop，
    默认 (2,3,4) 下不判（配置驱动，不硬编码）。"""
    import torch as _t
    m = CausalToyLM(vocab=64, max_len=64)
    pr = torch.randint(0, 64, (1, 5))
    seq = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]   # 周期 5（15 tokens）
    state = {"it": iter(seq)}
    monkeypatch.setattr(_t, "multinomial",
                        lambda probs, num_samples=1: _t.tensor([[next(state["it"])]]))
    out_default = generate_with_status(m, pr, max_new=len(seq), eos_token_id=None)
    state["it"] = iter(seq)      # 重置迭代器，两次采样独立（各消费 15 tokens）
    out_custom = generate_with_status(m, pr, max_new=len(seq), eos_token_id=None,
                                      loop_periods=(5,))
    assert out_default["statuses"] == ["budget_stop"]    # 默认 (2,3,4) 不判周期 5
    assert out_custom["statuses"] == ["loop"]            # 自定义 (5,) 判 loop
    assert out_custom["looped"] == [True]


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


def test_parse_vllm_outputs_stop_token_eos():
    """vLLM>=0.8 stop_token_ids 路径：eos 被消费但不入输出，finish_reason='stop'。
    语义 = toy（length=eos_pos+1 含 eos），eos 位置=len(new)。"""
    outs = [_FakeOut([1, 2, 3], "stop")]               # 3 token 后撞 eos
    r = parse_vllm_outputs(outs, max_new=8, eos_token_id=0)
    assert r["statuses"] == ["eos"]
    assert r["lengths"] == [4]                          # len(3)+1 含 eos
    assert r["eos_pos"] == [3]
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


# --------------------------- 任务4：run_refresh_phase 短 rollout + ring buffer status ---------------------------
from fullstack_opd_v2.adaptive_cache import (
    RefreshRingBuffer, DisagreementComputer, run_refresh_phase)
from fullstack_opd_v2.model import CausalToyLM


def _make_toy(vocab=8, d_model=8, n_layers=1):
    return CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers)


def _fake_rollout(responses, statuses, lengths, eos_pos, looped):
    """注入式 rollout_generator：返回固定合成 dict（不用真实采样，确定性）。

    ⚠️ 契约（P2 修复后）：注入的 rollout_generator 是【绑定方法】，签名
    gen(prompts, max_new, ...)；run_refresh_phase 以 prompts 为第一实参调用，
    不再把 student 当 self 传入（模块级 generate_with_status 才收 (model, prompts)）。
    """
    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        return {"responses": responses, "statuses": statuses, "lengths": lengths,
                "eos_pos": eos_pos, "looped": looped}
    return gen


def test_run_refresh_phase_inject_generator_and_status_roundtrip():
    """注入固定 rolloll rollout_generator → summary 计数正确 + ring buffer 存 status。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    n = 4
    # 4 样本：0=eos, 1=budget_stop, 2=loop, 3=invalid
    resp = torch.randint(1, V, (n, 6))
    statuses = ["eos", "budget_stop", "loop", "empty"]   # 长度 0 = empty（IMP-1d）
    lengths = [3, 6, 6, 0]
    eos_pos = [2, None, None, None]
    looped = [False, False, True, False]
    gen = _fake_rollout(resp, statuses, lengths, eos_pos, looped)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    # summary：loop/invalid 跳过 append，只 2 个进池
    # 成本字段（P1.3）：valid=[eos(3),budget_stop(6)] → rollout=9；名义预算=4×6=24；
    # teacher 前向=2×(3+6)=18
    # IMP-1d：summary 含运行时 wall_time（非确定）→ 精确键子集断言 + wall_time 非负
    expected = {"n_total": 4, "n_appended": 2, "n_eos": 1,
                "n_budget": 1, "n_loop": 1, "n_invalid": 0,
                "n_empty": 1, "valid_rate": 0.5,     # 2/4（IMP-1d）
                "generated_tokens": 15, "valid_tokens": 9,   # 3+6+6+0 / 3+6
                "rollout_tokens": 9, "expected_rollout_tokens": 24,
                "budgets_used": 24, "teacher_forward_tokens": 18,
                "loop_periods": (2, 3, 4),
                "temperature": 0.7,
                "repetition_penalty": 1.0,
                "loop_min_len": 8,
                "source": "student"}
    for k, v in expected.items():
        assert summary[k] == v, k
    assert summary["wall_time"] >= 0       # 生成 wall time 非负
    assert rb.size == 2
    # ring buffer 存的 status 只含 valid 子集（eos/budget_stop）
    assert sorted(rb._status) == ["budget_stop", "eos"]


def test_run_refresh_phase_injected_gen_called_with_prompts_not_student():
    """P2 修复：注入式 rollout_generator（绑定方法）以 prompts 为第一实参调用，
    不得把 student 当 self 传入（否则绑定方法 self 错乱）。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.arange(V * 5).view(V, 5) % V    # 可辨识的 prompt 张量 (V,5)
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["first_arg_is_prompts"] = prompts is not None
        seen["first_arg_shape"] = tuple(prompts.shape)
        seen["max_new"] = max_new
        # 全 budget_stop，valid，全部 append
        n = prompts.size(0)
        return {"responses": torch.ones(n, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * n,
                "lengths": [max_new] * n,
                "eos_pos": [None] * n, "looped": [False] * n}

    m_selected = 3
    run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag, prompts,
                      step=1, version=1, m_selected=m_selected, max_resp_len=6,
                      top_k=3, device="cpu", rollout_generator=gen)
    # 第一实参必须是 prompts（shape==(m_selected, P)），不是 student
    assert seen["first_arg_shape"] == (m_selected, prompts.size(1))
    assert seen["max_new"] == 6


def test_refresh_ring_buffer_status_roundtrip():
    """append 带 status → state_dict/load_state_dict 往返保留 status。"""
    V = 8
    rb = RefreshRingBuffer(capacity=4, top_k=3, vocab=V)
    rb.append(torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 3),
              generation_step=1, response_length=3,
              token_mask=torch.ones(3, dtype=torch.long),
              disagreement_abs=0.5, prompt_idx=0,
              response=torch.zeros(3, dtype=torch.long),
              s_old_ids=torch.zeros(3, 3, dtype=torch.long),
              s_old_logp=torch.zeros(3, 3), status="eos")
    rb.append(torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 3),
              generation_step=2, response_length=3,
              token_mask=torch.ones(3, dtype=torch.long),
              disagreement_abs=0.6, prompt_idx=1,
              response=torch.ones(3, dtype=torch.long),
              s_old_ids=torch.zeros(3, 3, dtype=torch.long),
              s_old_logp=torch.zeros(3, 3), status="budget_stop")
    sd = rb.state_dict()
    rb2 = RefreshRingBuffer(capacity=4, top_k=3, vocab=V)
    rb2.load_state_dict(sd)
    assert rb2._status == ["eos", "budget_stop"]
    # get() 也带 status
    g = rb2.get(torch.tensor([0, 1]))
    assert g["status"] == ["eos", "budget_stop"]


def test_run_refresh_phase_all_loop_no_append():
    """全部 loop → 无样本进池，summary 计数正确（teacher 前向不触发）。"""
    torch.manual_seed(1)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    resp = torch.randint(1, V, (n, 6))
    gen = _fake_rollout(resp, ["loop", "loop"], [6, 6], [None, None], [True, True])
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    expected = {"n_total": 2, "n_appended": 0, "n_eos": 0,
                "n_budget": 0, "n_loop": 2, "n_invalid": 0,
                "n_empty": 0, "valid_rate": 0.0,    # 0/2（IMP-1d）
                "generated_tokens": 12, "valid_tokens": 0,   # 6+6（loop 仍生成 token）/ 0
                "rollout_tokens": 0, "expected_rollout_tokens": 12,
                "budgets_used": 12, "teacher_forward_tokens": 0,
                "loop_periods": (2, 3, 4),
                "temperature": 0.7,
                "repetition_penalty": 1.0,
                "loop_min_len": 8,
                "source": "student"}
    for k, v in expected.items():
        assert summary[k] == v, k
    assert summary["wall_time"] >= 0
    assert rb.size == 0


# --------------------------- 任务5：pipeline 接线（消费 l2.rollout + 记录 status） ---------------------------
from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.pipeline import FullStackOPDv2


import csv as _csv


def _read_csv_headers(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return next(_csv.reader(f))


def test_pipeline_l2_rollout_consumes_max_new_and_records_status(tmp_path):
    """显式设 l2.rollout.max_new_tokens → pipeline 消费并落盘 rollout/ 状态指标。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.rollout.max_new_tokens=8"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    headers = _read_csv_headers(out["metrics_csv"])
    assert any(h.startswith("rollout/") for h in headers)
    # IMP-1b：rollout/loop_periods 随 summary 一并落盘（tuple 值 CSV 字符串化，不崩）
    assert "rollout/loop_periods" in headers
    assert "rollout/temperature" in headers
    # IMP-1c：repetition_penalty / loop_min_len / source 随 summary 落盘
    assert "rollout/repetition_penalty" in headers
    assert "rollout/loop_min_len" in headers
    assert "rollout/source" in headers


def test_pipeline_l2_rollout_fallback_cache_max_resp(tmp_path):
    """未设 l2.rollout.max_new_tokens → 回落 cache.max_response_length（toy=4）仍跑通。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    headers = _read_csv_headers(out["metrics_csv"])
    assert any(h.startswith("rollout/") for h in headers)


def test_pipeline_l2_rollout_custom_loop_periods(tmp_path):
    """IMP-1b：config 自定义 loop_periods → pipeline 透传到 run_refresh_phase，
    rollout/loop_periods 落盘值为自定义 tuple（配置驱动，非硬编码）。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.rollout.max_new_tokens=8",
        "l2.rollout.loop_periods=2,4,6"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    with open(out["metrics_csv"], encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    vals = {r["rollout/loop_periods"] for r in rows
            if r.get("rollout/loop_periods")}
    assert vals == {"(2, 4, 6)"}          # CSV 字符串化 tuple，值来自 config 覆盖


# --------------------------- 任务6：Stage 2 实验矩阵（S2_E0-E3） ---------------------------
from fullstack_opd_v2.experiment import (
    STAGE2_ROLLOUT_MATRIX, build_config, run_matrix)


def test_build_config_stage2_matrix():
    """S2 矩阵建配置：rollout 预算语义保留；extra 可压到 toy 预算验证协议。"""
    c = build_config("S2_E2_opd1024", n_steps=4, matrix=STAGE2_ROLLOUT_MATRIX,
                     **{"l2.rollout.max_new_tokens": 8})
    assert c["l2"]["rollout"]["max_new_tokens"] == 8    # extra 覆盖（toy 压速）
    assert c["l2"]["refresh_ratio"]["mode"] == "fixed"
    assert c["l2"]["rollout"]["loop_detection"] is True
    # 不传 extra 时矩阵语义值保留（真实 512/1024/2048 声明）
    c1 = build_config("S2_E1_opd512", n_steps=2, matrix=STAGE2_ROLLOUT_MATRIX)
    assert c1["l2"]["rollout"]["max_new_tokens"] == 512
    # 未知名报错
    with pytest.raises(KeyError):
        build_config("S2_E9_unknown", matrix=STAGE2_ROLLOUT_MATRIX)


def test_run_matrix_stage2_runs(tmp_path):
    """S2 矩阵跑通：4 实验（E0-E3）n_steps 对齐，rollout 预算覆盖到 toy 速跑。"""
    res = run_matrix(str(tmp_path), n_steps=4, device="cpu",
                     matrix=STAGE2_ROLLOUT_MATRIX,
                     **{"l2.rollout.max_new_tokens": 8})
    assert len(res) == 4
    assert {r["name"] for r in res} == set(STAGE2_ROLLOUT_MATRIX)
    # S2_E1-E3（l2.enabled=true）现在【真正触发刷新】——n_steps 含 refresh 训练步
    # （门控修复：selective 关闭时 selector=None=均匀随机选，刷新照常执行）
    by_name = {r["name"]: r for r in res}
    assert by_name["S2_E0_static"]["summary"]["n_steps"] == 4
    for n in ("S2_E1_opd512", "S2_E2_opd1024", "S2_E3_opd2048"):
        assert by_name[n]["summary"]["n_steps"] >= 4


# --------------------------- 任务7：报告 Q1-Q4（report_stage2.py） ---------------------------
from fullstack_opd_v2.report_stage2 import write_stage2_report


def test_write_stage2_report_placeholder(tmp_path):
    """无数据占位报告：文件生成 + 4 段 Q 标题齐全 + 表格占位。"""
    md = write_stage2_report([], [], str(tmp_path / "s2.md"))
    assert (tmp_path / "s2.md").exists()
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert f"## {q}" in md
    assert "（无数据）" in md          # 空结果优雅降级
    assert "—" in md or "待服务器" in md


def test_write_stage2_report_with_data(tmp_path):
    """喂占位结果 dict：表格渲染指标、4 段 Q 解读含数值。"""
    train = [{"name": "S2_E0_static", "summary": {"experiment": "S2_E0_static",
             "reward_mean": 1.2, "pg_loss_mean": 0.5, "kl_loss_mean": 0.1,
             "n_steps": 4}}]
    evalres = [{"name": "S2_E2_opd1024",
                "metrics": {"budget": 4096, "accuracy": 0.42}}]
    md = write_stage2_report(train, evalres, str(tmp_path / "s2_data.md"))
    assert "S2_E0_static" in md
    assert "1.2000" in md or "1.2" in md
    assert "4096" in md and "0.42" in md

def test_disagreement_gate_hard_off():
    """P1.4：compute_disagreement=False 时跳过 D 计算（硬 gate），append 的 disagreement_abs=0。"""
    torch.manual_seed(0)
    V = 8
    stu = _make_toy(V); t_rl = _make_toy(V); t_ref = _make_toy(V); s_ref = _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (4, 5))
    n = 4
    resp = torch.randint(1, V, (n, 6))
    gen = _fake_rollout(resp, ["budget_stop"] * n, [6] * n, [None] * n, [False] * n)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen, compute_disagreement=False)
    assert summary["n_appended"] == n
    assert all(d == 0.0 for d in rb._disagreements), "硬 gate 关闭时 disagreement 应为 0"
    assert rb.size == n
# --------------------------- IMP-1a：rollout temperature 透传 ---------------------------
def test_run_refresh_phase_temperature_passed_to_generator():
    """IMP-1a：run_refresh_phase 把 temperature 完整透传给注入的 rollout_generator，
    且 summary 记录实际 temperature。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["temperature"] = temperature
        m = prompts.size(0)
        return {"responses": torch.ones(m, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * m,
                "lengths": [max_new] * m,
                "eos_pos": [None] * m, "looped": [False] * m}

    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen, temperature=0.7)
    assert seen["temperature"] == 0.7            # 生成器收到配置的温度
    assert summary["temperature"] == 0.7         # summary 记录实际温度


def test_run_refresh_phase_temperature_1_0_old_behavior():
    """IMP-1a：temperature=1.0 可配置复现旧行为（显式传参，生成器收到 1.0）。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["temperature"] = temperature
        m = prompts.size(0)
        return {"responses": torch.ones(m, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * m,
                "lengths": [max_new] * m,
                "eos_pos": [None] * m, "looped": [False] * m}

    run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                      prompts, step=1, version=1, m_selected=n,
                      max_resp_len=6, top_k=3, device="cpu",
                      rollout_generator=gen, temperature=1.0)
    assert seen["temperature"] == 1.0            # 1.0 旧行为可复现

# --------------------------- IMP-1b：loop_periods 透传 ---------------------------
def test_run_refresh_phase_loop_periods_passed_to_generator():
    """IMP-1b：run_refresh_phase 把 loop_periods 完整透传给注入的 rollout_generator，
    summary 记录实际使用的 loop_periods（tuple）。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["loop_periods"] = loop_periods
        m = prompts.size(0)
        return {"responses": torch.ones(m, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * m,
                "lengths": [max_new] * m,
                "eos_pos": [None] * m, "looped": [False] * m}

    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen, loop_periods=(2, 4, 6))
    assert seen["loop_periods"] == (2, 4, 6)     # 生成器收到配置的周期集合
    assert summary["loop_periods"] == (2, 4, 6)  # summary 记录实际周期集合


def test_run_refresh_phase_loop_periods_default_summary():
    """IMP-1b：不显式传 loop_periods 时默认 (2,3,4) 透传，summary 记录默认值（零回归）。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["loop_periods"] = loop_periods
        m = prompts.size(0)
        return {"responses": torch.ones(m, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * m,
                "lengths": [max_new] * m,
                "eos_pos": [None] * m, "looped": [False] * m}

    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    assert seen["loop_periods"] == (2, 3, 4)
    assert summary["loop_periods"] == (2, 3, 4)
# --------------------------- IMP-1c：repetition_penalty + loop_min_len ---------------------------
from fullstack_opd_v2.model import apply_repetition_penalty


def test_l2_rollout_repetition_penalty_config():
    """IMP-1c：L2RolloutCfg 默认 repetition_penalty=1.0 / loop_min_len=8；可覆盖。"""
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["repetition_penalty"] == 1.0
    assert cfg["l2"]["rollout"]["loop_min_len"] == 8
    cfg2 = load_config(overrides=["l2.rollout.repetition_penalty=1.3",
                                  "l2.rollout.loop_min_len=16"])
    assert cfg2["l2"]["rollout"]["repetition_penalty"] == 1.3
    assert cfg2["l2"]["rollout"]["loop_min_len"] == 16


def test_apply_repetition_penalty_math():
    """IMP-1c：repetition_penalty>1 时已生成 token 的 logits 除以 penalty；<=1 零回归。"""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [1.0, 2.0, 3.0, 4.0]])
    past = torch.tensor([[0, 2], [1, 3]])       # 行0 见 token 0,2；行1 见 token 1,3
    out = apply_repetition_penalty(logits.clone(), past, 2.0)
    # 已见 token 减半：row0 -> [0.5, 2.0, 1.5, 4.0]；row1 -> [1.0, 1.0, 3.0, 2.0]
    assert out[0].tolist() == [0.5, 2.0, 1.5, 4.0]
    assert out[1].tolist() == [1.0, 1.0, 3.0, 2.0]
    # penalty=1.0（默认禁用）与 None 均零回归（不改数值）
    assert torch.equal(apply_repetition_penalty(logits, past, 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, past, None), logits)


def test_detect_loop_loop_min_len_controls():
    """IMP-1c：loop_min_len 门槛控制短序列是否判 loop（调高=降误报）。"""
    r = torch.tensor([1, 2, 3, 1, 2, 3])        # 15 token 周期 3 的前 6
    assert detect_loop(r, periods=(3,), min_len=6)          # min_len=6 判 loop
    assert not detect_loop(r, periods=(3,), min_len=16)     # min_len=16 不判（过严门槛）


def test_run_refresh_phase_repetition_controls_passed():
    """IMP-1c：run_refresh_phase 把 repetition_penalty / loop_min_len 完整透传给生成器，
    summary 记录实际值（可配置，非硬编码）。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    n = 2
    seen = {}

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4),
            repetition_penalty=1.0, loop_min_len=8):
        seen["repetition_penalty"] = repetition_penalty
        seen["loop_min_len"] = loop_min_len
        m = prompts.size(0)
        return {"responses": torch.ones(m, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * m,
                "lengths": [max_new] * m,
                "eos_pos": [None] * m, "looped": [False] * m}

    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen,
                                repetition_penalty=1.3, loop_min_len=16)
    assert seen["repetition_penalty"] == 1.3
    assert seen["loop_min_len"] == 16
    assert summary["repetition_penalty"] == 1.3
    assert summary["loop_min_len"] == 16
# --------------------------- IMP-1c：teacher rollout capability（仅诊断） ---------------------------
def test_l2_rollout_source_config():
    """IMP-1c：rollout_source 默认 student（禁止默认启用 teacher）；可覆盖 teacher；非法值拒绝。"""
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["rollout_source"] == "student"
    cfg2 = load_config(overrides=["l2.rollout.rollout_source=teacher"])
    assert cfg2["l2"]["rollout"]["rollout_source"] == "teacher"
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.rollout.rollout_source=ref"])


def test_run_refresh_phase_teacher_source_uses_teacher_rl(monkeypatch):
    """IMP-1c：rollout_source=teacher 时默认生成调用 teacher_rl（y~pi_teacher_rl，诊断专用）。"""
    from fullstack_opd_v2 import model as _model_mod
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    calls = {}

    def fake_gen(m, prompts, max_new, **kw):
        calls["model"] = m
        n = prompts.size(0)
        return {"responses": torch.ones(n, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * n,
                "lengths": [max_new] * n,
                "eos_pos": [None] * n, "looped": [False] * n}

    monkeypatch.setattr(_model_mod, "generate_with_status", fake_gen)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=2,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_source="teacher")
    assert calls["model"] is t_rl          # teacher RL generate 被调用
    assert summary["source"] == "teacher"   # teacher source 明确记录
    assert rb._source == ["teacher", "teacher"]   # 逐样本 metadata 保存 source


def test_run_refresh_phase_student_source_uses_student(monkeypatch):
    """IMP-1c：rollout_source=student（默认）时默认生成调用 student（y~pi_student）。"""
    from fullstack_opd_v2 import model as _model_mod
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (2, 5))
    calls = {}

    def fake_gen(m, prompts, max_new, **kw):
        calls["model"] = m
        n = prompts.size(0)
        return {"responses": torch.ones(n, max_new, dtype=torch.long),
                "statuses": ["budget_stop"] * n,
                "lengths": [max_new] * n,
                "eos_pos": [None] * n, "looped": [False] * n}

    monkeypatch.setattr(_model_mod, "generate_with_status", fake_gen)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=2,
                                max_resp_len=6, top_k=3, device="cpu")
    assert calls["model"] is stu           # student generate 被调用
    assert summary["source"] == "student"
    assert rb._source == ["student", "student"]


def test_refresh_ring_buffer_source_metadata_roundtrip():
    """IMP-1c：ring buffer 逐样本保存 rollout_source，state_dict 往返保留 + get 暴露。"""
    V = 8
    rb = RefreshRingBuffer(capacity=4, top_k=3, vocab=V)
    rb.append(torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 3),
              generation_step=1, response_length=3,
              token_mask=torch.ones(3, dtype=torch.long),
              disagreement_abs=0.5, prompt_idx=0,
              response=torch.zeros(3, dtype=torch.long),
              s_old_ids=torch.zeros(3, 3, dtype=torch.long),
              s_old_logp=torch.zeros(3, 3), status="budget_stop",
              source="teacher")
    rb.append(torch.zeros(3, 3, dtype=torch.long), torch.zeros(3, 3),
              generation_step=2, response_length=3,
              token_mask=torch.ones(3, dtype=torch.long),
              disagreement_abs=0.6, prompt_idx=1,
              response=torch.ones(3, dtype=torch.long),
              s_old_ids=torch.zeros(3, 3, dtype=torch.long),
              s_old_logp=torch.zeros(3, 3), status="budget_stop")
    sd = rb.state_dict()
    rb2 = RefreshRingBuffer(capacity=4, top_k=3, vocab=V)
    rb2.load_state_dict(sd)
    assert rb2._source == ["teacher", "student"]
    assert rb2.get(torch.tensor([0, 1]))["source"] == ["teacher", "student"]
# --------------------------- IMP-1d：Refresh Pool 冷启动保护（pipeline） ---------------------------
def test_l2_cache_min_refresh_pool_config():
    """IMP-1d：l2.cache.min_refresh_pool 默认 8；可覆盖。"""
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["cache"]["min_refresh_pool"] == 8
    cfg2 = load_config(overrides=["l2.cache.min_refresh_pool=16"])
    assert cfg2["l2"]["cache"]["min_refresh_pool"] == 16


def test_pipeline_cold_start_skips_refresh_training(tmp_path):
    """IMP-1d：池 < min_refresh_pool 时跳过 refresh 训练（不调 _train_step_refresh），
    仍记录 rollout metrics + refresh_train/skipped + refresh_pool/size；样本不丢。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=9",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4",
        "l2.cache.refresh_min_interval=3",
        "l2.cache.min_refresh_pool=1000"])   # 门槛远大于容量 → 永不达标，全部跳过
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    rollout_rows = [m for m in out["metrics"]
                    if isinstance(m, dict) and m.get("phase") == "rollout"]
    assert rollout_rows, "refresh 相位未跑"
    for r in rollout_rows:
        assert r.get("refresh_train/skipped") is True
        assert r["refresh_train/skip_reason"] == "cold_start_pool_too_small"
        assert 0 <= r["refresh_pool/size"] <= 8     # 池有样本、受容量限制
        assert "rollout/n_appended" in r             # rollout metrics 仍记录
    # 池大小单调不减（样本未丢）
    sizes = [r["refresh_pool/size"] for r in rollout_rows]
    assert sizes == sorted(sizes), "池大小应单调不减（样本未丢）"
    # 跳过轮不产生 pool="refresh" 训练行 → _train_step_refresh 未被调用
    assert not any(isinstance(m, dict) and m.get("pool") == "refresh"
                   for m in out["metrics"]), "冷启动轮不应调 _train_step_refresh"


def test_pipeline_cold_start_trains_after_pool_ready(tmp_path):
    """IMP-1d：池 ≥ min_refresh_pool（size=8 边界）后正常训练。loop_detection 关 →
    每轮 rollout 全 valid（池每轮 +4）：4(skip) → 8(train) → 12(train) → 16(train)。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=12",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=64", "l2.cache.max_response_length=4",
        "l2.cache.refresh_min_interval=3",
        "l2.cache.min_refresh_pool=8",
        "l2.rollout.loop_detection=false"])     # 全 valid，池增长确定
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    rollout_rows = [m for m in out["metrics"]
                    if isinstance(m, dict) and m.get("phase") == "rollout"]
    assert len(rollout_rows) >= 2
    sizes = [r["refresh_pool/size"] for r in rollout_rows]
    # 首轮池 < 8 → 跳过；后续池 ≥ 8 → 训练
    assert rollout_rows[0]["refresh_train/skipped"] is True
    assert rollout_rows[0]["refresh_train/skip_reason"] == "cold_start_pool_too_small"
    assert sizes[0] < 8
    assert any(r["refresh_train/skipped"] is False and r["refresh_pool/size"] >= 8
               for r in rollout_rows)
    # 训练确实发生（存在 pool="refresh" 行）
    assert any(isinstance(m, dict) and m.get("pool") == "refresh" for m in out["metrics"])
# --------------------------- IMP-1d：有效样本率定义（valid_rate + 完整 outcome） ---------------------------
def test_run_refresh_phase_valid_rate_breakdown():
    """IMP-1d：valid = non_empty ∧ ¬loop ∧ token 序列有效 → 完整 outcome 统计 + valid_rate。
    5 样本：eos(valid) / budget_stop(valid) / loop / invalid(非空) / empty(长度0)。
    valid_rate = 2/5 = 0.4；loop/invalid/empty 都不进 refresh 池。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _make_toy(V), _make_toy(V), _make_toy(V), _make_toy(V)
    rb = RefreshRingBuffer(capacity=16, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.randint(0, V, (5, 5))
    n = 5
    resp = torch.randint(1, V, (n, 6))
    statuses = ["eos", "budget_stop", "loop", "invalid", "empty"]
    lengths = [3, 6, 6, 4, 0]
    eos_pos = [2, None, None, None, None]
    looped = [False, False, True, False, False]
    gen = _fake_rollout(resp, statuses, lengths, eos_pos, looped)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    assert summary["n_total"] == 5                       # generated
    assert summary["n_eos"] == 1 and summary["n_budget"] == 1
    assert summary["n_loop"] == 1
    assert summary["n_invalid"] == 1 and summary["n_empty"] == 1
    assert summary["n_appended"] == 2                    # valid = eos + budget_stop
    assert summary["valid_rate"] == pytest.approx(0.4)   # 2/5（>=0.50 目标）
    # IMP-1d：effective rollout throughput
    assert summary["generated_tokens"] == 19            # 3+6+6+4+0（含 loop/invalid/empty）
    assert summary["valid_tokens"] == 9                 # 3+6（仅 valid）
    assert summary["wall_time"] >= 0
    assert rb.size == 2                                  # loop/invalid/empty 不进池
    assert sorted(rb._status) == ["budget_stop", "eos"]

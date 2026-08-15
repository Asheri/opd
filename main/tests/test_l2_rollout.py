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
    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0):
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
    statuses = ["eos", "budget_stop", "loop", "invalid"]
    lengths = [3, 6, 6, 0]
    eos_pos = [2, None, None, None]
    looped = [False, False, True, False]
    gen = _fake_rollout(resp, statuses, lengths, eos_pos, looped)
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag,
                                prompts, step=1, version=1, m_selected=n,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    # summary：loop/invalid 跳过 append，只 2 个进池（budgets=None 单预算路径，
    # 新增 token 记账：actual=sum(lengths)=3+6+6+0=15，budgets_used=6*4=24）
    assert summary == {"n_total": 4, "n_appended": 2, "n_eos": 1,
                       "n_budget": 1, "n_loop": 1, "n_invalid": 1,
                       "rollout_tokens": 15, "expected_rollout_tokens": 15,
                       "budgets_used": 24}
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

    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0):
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
    assert summary == {"n_total": 2, "n_appended": 0, "n_eos": 0,
                       "n_budget": 0, "n_loop": 2, "n_invalid": 0,
                       "rollout_tokens": 12, "expected_rollout_tokens": 12,
                       "budgets_used": 12}
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


def test_pipeline_l2_rollout_fallback_cache_max_resp(tmp_path):
    """未设 l2.rollout.max_new_tokens → 回落 cache.max_response_length（toy=4）仍跑通。"""
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4"])
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    headers = _read_csv_headers(out["metrics_csv"])
    assert any(h.startswith("rollout/") for h in headers)


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
    """S2 矩阵跑通：4 实验（E0-E3），rollout 预算覆盖到 toy 速跑。

    ⚠️ 任务7 后：refresh 触发条件删掉 `and selector is not None`，selective 关的
    S2_E1/E2/E3（l2.enabled=true）现在【真正触发刷新】——n_steps 含 refresh 训练步
    （4 base + 2 refresh ≈ 6）；E0_static（l2 关）仍恰好 4 步不刷。断言对齐新语义。
    """
    res = run_matrix(str(tmp_path), n_steps=4, device="cpu",
                     matrix=STAGE2_ROLLOUT_MATRIX,
                     **{"l2.rollout.max_new_tokens": 8})
    assert len(res) == 4
    assert {r["name"] for r in res} == set(STAGE2_ROLLOUT_MATRIX)
    by_name = {r["name"]: r for r in res}
    # E0_static（l2 关）无刷新 → 恰好 4 训练步
    assert by_name["S2_E0_static"]["summary"]["n_steps"] == 4
    # E1/E2/E3（l2 开）现在触发刷新 → 训练步 ≥ 4（含 refresh 训练步）
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
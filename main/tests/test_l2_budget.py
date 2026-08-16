"""Stage 3：Budget-Aware Selective Rollout 配置单测。

- L2RolloutCfg.token_budget_per_refresh：每轮刷新全局 rollout token 预算（None=无上限）。
- L2SelectiveRolloutCfg：budget_mode/fixed_budget/budget_set/budget_quantiles +
  token_aware/token_weight + value_weights 补 reward 项。
- 默认 budget_mode="fixed" 保持现有行为零回归。
"""
import pytest
import torch

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.exceptions import ConfigError
from fullstack_opd_v2.adaptive_cache import (
    assign_budgets, enforce_budget, compute_rollout_metrics, group_by_budget,
    PromptStateStore, RefreshSelector, DynamicRatioController)


def test_l2_budget_defaults():
    cfg = load_config(overrides=["l2.enabled=true"])
    assert cfg["l2"]["rollout"]["token_budget_per_refresh"] is None
    sr = cfg["l2"]["selective_rollout"]
    assert sr["budget_mode"] == "fixed"
    assert sr["fixed_budget"] == 1024
    assert sr["budget_set"] == (256, 512, 1024, 2048)
    assert sr["budget_quantiles"] == (0.25, 0.5, 0.75)
    assert sr["token_aware"] is False
    assert "reward" in sr["value_weights"]


def test_l2_budget_overrides():
    cfg = load_config(overrides=[
        "l2.selective_rollout.budget_mode=adaptive",
        "l2.selective_rollout.fixed_budget=512",
        "l2.rollout.token_budget_per_refresh=4096",
        "l2.selective_rollout.token_aware=true"])
    sr = cfg["l2"]["selective_rollout"]
    assert sr["budget_mode"] == "adaptive"
    assert sr["fixed_budget"] == 512
    assert cfg["l2"]["rollout"]["token_budget_per_refresh"] == 4096
    assert sr["token_aware"] is True


def test_l2_budget_unknown_key_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.selective_rollout.unknown=1"])


# ---- Stage 3 纯函数：assign_budgets ----


def test_assign_budgets_quantiles():
    v = torch.tensor([0.1, 0.3, 0.6, 0.9])
    budgets = assign_budgets(v)
    assert budgets.dtype == torch.long
    b = budgets.tolist()
    assert b[0] < b[-1]          # 低价值→低档，高价值→高档
    assert b[0] <= b[1] <= b[2] <= b[3]   # 单调
    for x in b:
        assert x in (256, 512, 1024, 2048)


def test_assign_budgets_all_equal():
    v = torch.full((4,), 0.5)
    budgets = assign_budgets(v)
    assert budgets.tolist() == [1024, 1024, 1024, 1024]  # 中档 budget_set[2]


def test_assign_budgets_4_buckets():
    v = torch.linspace(0.1, 0.9, 8)
    budgets = assign_budgets(v)
    assert len(set(budgets.tolist())) == 4   # 4 档都被覆盖


# ---- Stage 3 纯函数：enforce_budget ----


def test_enforce_budget_within():
    indices = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([256, 512, 256, 256])
    v = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=2000)
    assert out_i.tolist() == indices.tolist()
    assert out_b.tolist() == budgets.tolist()   # 未超预算原样返回


def test_enforce_budget_downgrade():
    indices = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([2048, 2048, 2048, 2048])
    v = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=3000)
    assert out_b.sum().item() <= 3000
    for x in out_b.tolist():
        assert x in (256, 512, 1024, 2048)   # 预算来自 budget_set


def test_enforce_budget_none():
    indices = torch.tensor([0, 1])
    budgets = torch.tensor([2048, 2048])
    v = torch.tensor([0.1, 0.2])
    out_i, out_b = enforce_budget(indices, budgets, v, budget_t=None)
    assert out_i.tolist() == indices.tolist()
    assert out_b.tolist() == budgets.tolist()


# ---- Stage 3 纯函数：compute_rollout_metrics ----


def test_compute_rollout_metrics():
    summary = dict(n_total=100, n_appended=40, n_eos=30, n_budget=20,
                   n_loop=10, n_invalid=5, rollout_tokens=200)
    m = compute_rollout_metrics(summary, budget_t=200)
    assert m['rollout/rollout_tokens'] == 200
    assert m['rollout/budget_utilization'] == pytest.approx(1.0)
    assert m['rollout/truncation_rate'] == pytest.approx(0.2)
    assert m['rollout/loop_rate'] == pytest.approx(0.1)
    assert m['rollout/eos_rate'] == pytest.approx(0.3)
    assert m['rollout/accuracy_proxy'] == pytest.approx(0.4)
    assert m['rollout/useful_per_token'] == pytest.approx(0.2)  # 40/200


def test_compute_rollout_metrics_divzero():
    summary = dict(n_total=0, n_appended=0, n_eos=0, n_budget=0,
                   n_loop=0, n_invalid=0, rollout_tokens=0)
    m = compute_rollout_metrics(summary, budget_t=200)  # rollout_tokens=0 → utilization 0.0
    for val in m.values():
        assert val == 0.0


# ---- Stage 3 纯函数：group_by_budget ----


def test_group_by_budget():
    cand = torch.tensor([0, 1, 2, 3])
    budgets = torch.tensor([256, 1024, 256, 512])
    buckets = group_by_budget(cand, budgets)
    assert buckets == {256: [0, 2], 1024: [1], 512: [3]}


# ---- Stage 3：RefreshSelector.select_with_budget + _value 补 reward ----


def _make_selector(n_prompts: int, updates: dict) -> PromptStateStore:
    """构造带历史的 PromptStateStore + 默认 RefreshSelector。

    updates: {prompt_id: (reward, disagreement, resp_len)} 逐 prompt 喂历史。
    """
    ps = PromptStateStore(n_prompts)
    for pid, (reward, disagreement, resp_len) in updates.items():
        ps.update(pid, reward, disagreement, resp_len, step=1)
    return ps


def test_select_with_budget_fixed():
    ps = _make_selector(20, {0: (1.0, 0.5, 100), 1: (0.2, 0.1, 50)})
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(n_selected=8, n_prompts=20)
    assert indices.shape == (8,)
    assert budgets.shape == (8,)
    assert budgets.dtype == torch.long
    assert (budgets == 1024).all().item()      # fixed 默认单预算 1024


def test_select_with_budget_adaptive_4buckets():
    # 喂足历史，reward/disagreement 差异明显 → 选中集内 V 有分位数区分
    updates = {i: (float(i) / 19, float(i % 5) / 5, 100 + i * 10) for i in range(20)}
    ps = _make_selector(20, updates)
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(
        n_selected=8, n_prompts=20, budget_mode="adaptive")
    assert indices.shape == (8,)
    assert budgets.shape == (8,)
    assert budgets.dtype == torch.long
    for b in budgets.tolist():
        assert b in (256, 512, 1024, 2048)      # 都来自 budget_set


def test_select_with_budget_cold_start():
    # times_seen 全 0 → select() 走 uniform，_value() 全 0（无历史）
    # → assign_budgets 全等 → 中档 1024 fallback
    ps = PromptStateStore(n_prompts=20)
    sel = RefreshSelector(ps)
    indices, budgets = sel.select_with_budget(
        n_selected=6, n_prompts=20, budget_mode="adaptive")
    assert indices.shape == (6,)
    assert budgets.shape == (6,)
    assert (budgets == 1024).all().item()       # 全等 v → 中档 1024


def test_value_includes_reward():
    # 构造 reward_ema 差异：高 reward prompt vs 低 reward（其余信号相同）。
    # 直接设 store 字段，保证 reward_var/disagreement/times_seen 完全一致（否则无 reward
    # 权重时两 prompt 也会因 uncertainty 差异而不同，无法隔离 reward 项）。
    ps = PromptStateStore(n_prompts=2)
    ps.times_seen[:] = 1
    ps.reward_var[:] = 0.0
    ps.disagreement_ema[:] = 0.0
    ps.reward_ema[0] = 0.9      # 高 reward
    ps.reward_ema[1] = 0.1      # 低 reward
    # 不含 reward 权重（旧 config）→ 两 prompt 值相同
    sel_no_r = RefreshSelector(ps)   # 默认无 reward 键
    v_no_r = sel_no_r._value()
    assert v_no_r[0].item() == pytest.approx(v_no_r[1].item(), abs=1e-6)
    # 含 reward 权重 → 高 reward prompt 值更高
    sel_r = RefreshSelector(ps, value_weights={
        "uncertainty": 0.4, "disagreement": 0.4, "novelty": 0.2, "reward": 0.5})
    v_r = sel_r._value()
    assert v_r[0].item() > v_r[1].item()


# ---- Stage 3：run_refresh_phase per-sample budget 分桶 ----

from fullstack_opd_v2.adaptive_cache import (
    RefreshRingBuffer, DisagreementComputer, run_refresh_phase)
from fullstack_opd_v2.model import CausalToyLM


def _toy(vocab=8, d_model=8, n_layers=1, max_len=64):
    """占位小 transformer（与 test_l2_rollout 一致）。"""
    return CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers, max_len=max_len)


def _recording_gen(log):
    """注入式 rollout_generator：记录每次被调用的 max_new，返回全 budget_stop。

    契约：绑定方法签名 gen(prompts, max_new, ...)，run_refresh_phase 以 prompts 为第一实参。
    """
    def gen(prompts, max_new, eos_token_id=None, loop_detection=True, pad_id=0,
            temperature=1.0, loop_periods=(2, 3, 4)):
        log.append(int(max_new))
        n = prompts.size(0)
        return {"responses": torch.ones(n, int(max_new), dtype=torch.long),
                "statuses": ["budget_stop"] * n,
                "lengths": [int(max_new)] * n,
                "eos_pos": [None] * n, "looped": [False] * n}
    return gen


def test_run_refresh_phase_budget_buckets():
    """per-sample budget 分桶：不同 prompt 按各自 budget(max_new) 生成，分桶生效。"""
    torch.manual_seed(0)
    V = 8
    # max_len 需 ≥ prompt(5)+max budget(2048)，否则 teacher 前向越界
    stu, t_rl, t_ref, s_ref = (_toy(V, max_len=4096),) * 4
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    n_prompts, m = 8, 4
    prompts = torch.randint(0, V, (n_prompts, 5))
    # 喂历史使 select 走 threaded 路径 → 选中互不重复的 prompt（避免 cand 重复映射）
    ps = PromptStateStore(n_prompts)
    for i in range(n_prompts):
        ps.update(i, reward=float(i) / (n_prompts - 1), disagreement=0.1,
                  resp_len=100 + i * 50, step=1)
    sel = RefreshSelector(ps)
    log = []
    gen = _recording_gen(log)
    budgets = torch.tensor([256, 512, 1024, 2048])
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, sel, rb, disag, prompts,
                                step=1, version=1, m_selected=m,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen, budgets=budgets)
    # 4 档 budget 各 1 个 prompt → 4 次 gen 调用，max_new 精确覆盖 budget_set
    assert sorted(log) == [256, 512, 1024, 2048]
    assert summary["n_total"] == m
    assert summary["n_appended"] == m
    assert summary["rollout_tokens"] == 256 + 512 + 1024 + 2048
    assert summary["expected_rollout_tokens"] == 256 + 512 + 1024 + 2048
    assert summary["budgets_used"] == 256 + 512 + 1024 + 2048
    assert summary["loop_periods"] == (2, 3, 4)     # IMP-1b：summary 记录周期集合
    assert rb.size == m


def test_run_refresh_phase_no_budget_regression():
    """budgets=None（默认）→ 单预算路径：gen 只调 1 次，max_new=默认 max_resp_len。"""
    torch.manual_seed(0)
    V = 8
    stu, t_rl, t_ref, s_ref = _toy(V), _toy(V), _toy(V), _toy(V)
    rb = RefreshRingBuffer(capacity=8, top_k=3, vocab=V)
    disag = DisagreementComputer()
    prompts = torch.arange(V * 5).view(V, 5) % V
    log = []
    gen = _recording_gen(log)
    m_selected = 3
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, None, rb, disag, prompts,
                                step=1, version=1, m_selected=m_selected,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen)
    assert log == [6]                       # 单次调用 + 默认 max_new（clamp 后 6）
    assert summary["rollout_tokens"] == 6 * m_selected
    assert summary["expected_rollout_tokens"] == 6 * m_selected
    assert summary["budgets_used"] == 6 * m_selected
    assert summary["loop_periods"] == (2, 3, 4)     # IMP-1b：单预算路径也记录


# ---- Stage 3：任务 5 pipeline 接线 smoke（adaptive 全链路：select_with_budget
#      → cand/budgets/budget_t 传入 run_refresh_phase → compute_rollout_metrics）----


def test_pipeline_adaptive_budget_smoke():
    """adaptive budget_mode + token_budget_per_refresh 端到端：cand/budgets 配对、
    per-sample 分桶生成、token 指标产生并（7 键）可落盘。"""
    torch.manual_seed(0)
    V = 8
    # max_len 需 ≥ prompt(5)+max budget(2048)，否则 teacher 前向越界
    stu, t_rl, t_ref, s_ref = (_toy(V, max_len=4096),) * 4
    rb = RefreshRingBuffer(capacity=16, top_k=3, vocab=V)
    disag = DisagreementComputer()
    n_prompts, m = 12, 6
    prompts = torch.randint(0, V, (n_prompts, 5))
    # 喂差异历史使选中集内 V 有分位数区分 → 4 档 budget 都出现
    ps = PromptStateStore(n_prompts)
    for i in range(n_prompts):
        ps.update(i, reward=float(i) / (n_prompts - 1), disagreement=float(i % 4) / 4,
                  resp_len=100 + i * 50, step=1)
    sel = RefreshSelector(ps)
    # 模拟 pipeline 接线：adaptive 预算 + 全局 token 预算。
    # budget_t 取大值避免触发 enforce_budget 降档（降档逻辑已由 test_enforce_budget_*
    # 覆盖），此处专注验证 cand/budgets 配对的 adaptive 全链路接线。
    budget_set = (256, 512, 1024, 2048)
    budget_t = 1 << 30
    indices, budgets = sel.select_with_budget(
        m, n_prompts, budget_mode="adaptive", budget_set=budget_set,
        quantiles=(0.25, 0.5, 0.75))
    assert indices.shape == (m,)
    assert budgets.shape == (m,)
    log = []
    gen = _recording_gen(log)
    # cand=indices：与 budgets 配对（改动 A），不再内部二次 select
    summary = run_refresh_phase(stu, t_rl, t_ref, s_ref, sel, rb, disag, prompts,
                                step=1, version=1, m_selected=m,
                                max_resp_len=6, top_k=3, device="cpu",
                                rollout_generator=gen,
                                cand=indices, budgets=budgets, budget_t=budget_t)
    # summary 带 token 指标
    assert summary["rollout_tokens"] > 0
    assert summary["expected_rollout_tokens"] == summary["budgets_used"]
    assert summary["rollout_tokens"] == summary["budgets_used"]
    # per-sample budget 分桶生效：gen 调用次数 = 选中集内不同 budget 档数
    assert sorted(set(log)) == sorted(set(budgets.tolist()))
    # token 效率指标：7 键 + useful_per_token 合法 float
    rm = compute_rollout_metrics(summary, budgets, budget_t)
    assert len(rm) == 7
    assert all(k.startswith("rollout/") for k in rm)
    assert isinstance(rm["rollout/useful_per_token"], float)
    assert rm["rollout/rollout_tokens"] == summary["rollout_tokens"]
    assert rm["rollout/budget_utilization"] >= 0.0


# ---- Stage 3：任务 5 S3_E2 FullStackOPDv2 全链路 smoke（selector 构造 →
#      run_refresh_phase(cand/budgets/budget_t) → compute_rollout_metrics →
#      hm.record 并入 rollout/ 键 → drc.update 传 rollout_efficiency）----

from fullstack_opd_v2.config import load_config
from fullstack_opd_v2.pipeline import FullStackOPDv2
import csv as _csv


def _read_csv_headers(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return next(_csv.reader(f))


def test_pipeline_s3_e2_adaptive_budget_smoke(tmp_path):
    """S3_E2：FullStackOPDv2 adaptive 预算全链路不崩 + rollout/* token 指标落盘。

    toy 模型 max_len=64，budget 必须 ≤ 剩余上下文（否则 generate_with_status 的
    ctx=prompt+已生成 越界位置编码）。故用 toy 友好 budget_set=(4,8,12,16)，
    预算语义链路（select_with_budget 分位数 → 分桶 → 记账）不变，仅值域压到 toy 内存。
    """
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=3", "stage2.n_steps=6",
        "stage2.batch_size=4", "l2.m_refresh=4",
        "l2.cache.refresh_size=8", "l2.cache.max_response_length=4"])
    sr = cfg["l2"]["selective_rollout"]
    sr["budget_mode"] = "adaptive"
    sr["budget_set"] = (4, 8, 12, 16)          # toy max_len=64 内，防位置编码越界
    cfg["l2"]["rollout"]["token_budget_per_refresh"] = 100   # 有限全局预算（不触发降档）
    out = FullStackOPDv2(cfg, device="cpu").run(run_dir=str(tmp_path))
    # 全链路不崩 + token 指标落盘（rollout/rollout_tokens 等，compute_rollout_metrics 产出）
    headers = _read_csv_headers(out["metrics_csv"])
    assert "rollout/rollout_tokens" in headers
    assert "rollout/budget_utilization" in headers
    assert "rollout/useful_per_token" in headers


# ---- Stage 3：任务 6 DynamicRatioController token 感知（第 4 信号）----


def _drc_past_warmup(token_aware, token_weight=0.1):
    """构造 adaptive + token_aware 的 controller，并直接推进 _step 越过 warmup。"""
    c = DynamicRatioController(
        mode="adaptive", token_aware=token_aware, token_weight=token_weight,
        warmup_steps=5)
    c._step = c.warmup + 1   # 绕过 warmup（warmup 内提前 return alpha0，不参与第 4 信号）
    return c


def test_drc_token_aware():
    """token_aware=True 时 rollout_efficiency 生效：
    省 token（eff>1，即 expected>actual）→ α 相对 baseline 升高（放宽）；
    超用（eff<1）→ α 相对 baseline 降低（收紧）。"""
    base_args = dict(base_age=0.5, policy_drift=0.3, refresh_quality=0.2)
    # 三个独立 controller，仅 rollout_efficiency 不同（其余信号一致）
    c_base = _drc_past_warmup(True)
    c_save = _drc_past_warmup(True)
    c_over = _drc_past_warmup(True)
    a_base = c_base.update(rollout_efficiency=None, **base_args)
    a_save = c_save.update(rollout_efficiency=1.5, **base_args)   # expected>actual，省 token
    a_over = c_over.update(rollout_efficiency=0.5, **base_args)   # expected<actual，超用
    assert a_save > a_base     # 省 token → 放宽 α
    assert a_over < a_base     # 超用 → 收紧 α
    # 第 4 信号确实被 EMA 记入（避免"只接收不参与"的假回归）
    assert c_save._ema["efficiency"] > 0
    assert c_over._ema["efficiency"] < 0


def test_drc_token_aware_off():
    """token_aware=False 时传 rollout_efficiency 与不传的 α 完全一致（零回归断言）。"""
    base_args = dict(base_age=0.5, policy_drift=0.3, refresh_quality=0.2)
    c_off_none = _drc_past_warmup(False)
    c_off_eff = _drc_past_warmup(False)
    a_none = c_off_none.update(rollout_efficiency=None, **base_args)
    a_eff = c_off_eff.update(rollout_efficiency=1.5, **base_args)
    assert a_none == a_eff
    # 且效率信号未被消费（EMA 保持 0）
    assert c_off_eff._ema["efficiency"] == 0.0

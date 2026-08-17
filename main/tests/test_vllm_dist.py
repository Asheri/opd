"""IMP-2/P0：run_refresh_phase 分布引擎 vLLM 接入的纯函数单测。

覆盖 _gather_support（ref top-K 在 rl 支撑的 logp）、_dist_topk_cached（引擎/HF 回落）、
_dist_rl_ref_delta（双引擎 vs HF 语义对齐）。dist_engines=None 的零回归由
test_l2_rollout / test_adaptive_cache 现有用例覆盖（run_refresh_phase 默认 None）。
"""
import torch
import pytest

from fullstack_opd_v2.adaptive_cache import (
    _gather_support, _dist_topk_cached, _dist_rl_ref_delta, _response_dists_topk,
)


class _FakeEngine:
    """fake vLLM 引擎：response_dists_topk 返回预设 (ids, logps)，按 logprob 降序。"""
    def __init__(self, ids, logps):
        self._ids = ids
        self._logps = logps
        self.calls = 0

    def response_dists_topk(self, prompts, responses, K=None):
        self.calls += 1
        return self._ids, self._logps


class _FakeHF:
    """fake HF 模型：如果被调（说明引擎未生效），抛错。"""
    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        raise AssertionError("HF 前向不应在 vLLM 引擎路径被调用")


def _mk_prompt_resp(B=1, P=3, T=4):
    prompts = torch.tensor([[1, 2, 3]] * B)
    responses = torch.tensor([[4, 5, 6, 7]] * B)
    return prompts, responses


# --------------------------- _gather_support ---------------------------
def test_gather_support_hit_takes_ref_logp():
    # ref top-K ids（按 logprob 降序排列，非按 id 排序）：id 40(logp -0.1), id 10(logp -0.5)
    ids = torch.tensor([[[40, 10]]])     # (1,1,2)
    logp = torch.tensor([[[-0.1, -0.5]]])
    query = torch.tensor([[[40]]])       # rl 支撑含 id 40
    out = _gather_support(ids, logp, query)
    assert torch.allclose(out, torch.tensor([[[-0.1]]]))


def test_gather_support_miss_fills_tail():
    ids = torch.tensor([[[40, 10]]])
    logp = torch.tensor([[[-0.1, -0.5]]])
    query = torch.tensor([[[99]]])       # rl 支撑 token 不在 ref top-K
    out = _gather_support(ids, logp, query)
    assert torch.allclose(out, torch.tensor([[[-30.0]]]))


def test_gather_support_multi_t():
    # T=3：位置0命中、位置1未命中、位置2命中
    ids = torch.tensor([[[40, 10], [5, 6], [7, 8]]])
    logp = torch.tensor([[[-0.1, -0.5], [-1.0, -2.0], [-0.3, -0.9]]])
    query = torch.tensor([[[40], [99], [8]]])
    out = _gather_support(ids, logp, query)
    assert torch.allclose(out, torch.tensor([[[-0.1], [-30.0], [-0.9]]]))


# --------------------------- _dist_topk_cached ---------------------------
def test_dist_topk_cached_engine_wins():
    prompts, responses = _mk_prompt_resp()
    eng = _FakeEngine(torch.zeros(1, 4, 2, dtype=torch.long),
                      torch.zeros(1, 4, 2))
    hf = _FakeHF()
    ids, lps = _dist_topk_cached(eng, hf, prompts, responses, K=2, chunk=2)
    assert eng.calls == 1
    assert hf.called is False       # 引擎路径不触发 HF


def test_dist_topk_cached_none_engine_uses_hf():
    # engine=None → HF per-chunk（用真实 _response_dists_topk 但 fake model 只要求可调）
    prompts, responses = _mk_prompt_resp()
    called = []

    class _M:
        def eval(self):
            return self

    def _fake_topk(model, p, r, K, chunk):
        called.append(K)
        return torch.zeros(p.size(0), r.size(1), K, dtype=torch.long), \
            torch.zeros(p.size(0), r.size(1), K)

    import fullstack_opd_v2.adaptive_cache as ac
    orig = ac._response_dists_topk
    ac._response_dists_topk = _fake_topk
    try:
        _dist_topk_cached(None, _M(), prompts, responses, K=2, chunk=2)
    finally:
        ac._response_dists_topk = orig
    assert called == [2]


# --------------------------- _dist_rl_ref_delta ---------------------------
def test_dist_rl_ref_delta_dual_engine():
    prompts, responses = _mk_prompt_resp()
    # rl top-K: id 40(logp -0.5)；ref top-K: id 40(logp -0.2) → delta = -0.5 - (-0.2) = -0.3
    rl_ids = torch.tensor([[[40]]]); rl_lp = torch.tensor([[[-0.5]]])
    ref_ids = torch.tensor([[[40]]]); ref_lp = torch.tensor([[[-0.2]]])
    rl_eng = _FakeEngine(rl_ids, rl_lp)
    ref_eng = _FakeEngine(ref_ids, ref_lp)
    ids_k, rl_k, delta_k = _dist_rl_ref_delta(rl_eng, ref_eng, None, None,
                                              prompts, responses, top_k=1, chunk=2)
    assert torch.equal(ids_k, rl_ids)
    assert torch.allclose(delta_k, torch.tensor([[[-0.3]]]))


def test_dist_rl_ref_delta_engine_miss_falls_back_hf():
    # rl 引擎给、ref 引擎 None → 回落 HF _rl_ref_delta_k（用 fake hf 联合函数）
    prompts, responses = _mk_prompt_resp()
    rl_eng = _FakeEngine(torch.zeros(1, 4, 1, dtype=torch.long),
                         torch.zeros(1, 4, 1))
    called = []

    def _fake_hr(rl_hf, ref_hf, p, r, top_k, chunk):
        called.append(top_k)
        return (torch.zeros(p.size(0), r.size(1), top_k, dtype=torch.long),
                torch.zeros(p.size(0), r.size(1), top_k),
                torch.zeros(p.size(0), r.size(1), top_k))

    import fullstack_opd_v2.adaptive_cache as ac
    orig = ac._rl_ref_delta_k
    ac._rl_ref_delta_k = _fake_hr
    try:
        _dist_rl_ref_delta(rl_eng, None, None, None, prompts, responses,
                           top_k=3, chunk=2)
    finally:
        ac._rl_ref_delta_k = orig
    assert called == [3]

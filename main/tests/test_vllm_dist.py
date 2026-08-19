import torch


def test_build_nccl_update_info():
    """NCCL update_info 纯函数：names/dtype/shapes 与 state_dict 一致、is_checkpoint_format=True。"""
    from fullstack_opd_v2.rollout_vllm import _build_nccl_update_info
    sd = {
        "model.embed_tokens.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 8, dtype=torch.float32),
    }
    info = _build_nccl_update_info(sd)
    # backend 由引擎启动时的 WeightTransferConfig 决定，update_info 不带 backend 键
    assert "backend" not in info
    assert info["names"] == list(sd.keys())
    assert info["dtype_names"] == ["bfloat16", "float32"]
    assert info["shapes"] == [[16, 8], [4, 8]]
    assert info["is_checkpoint_format"] is True
    assert info["packed"] is False


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
import torch


def test_build_nccl_update_info():
    """NCCL update_info 纯函数：names/dtype/shapes 与 state_dict 一致、is_checkpoint_format=True。"""
    from fullstack_opd_v2.rollout_vllm import _build_nccl_update_info
    sd = {
        "model.embed_tokens.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 8, dtype=torch.float32),
    }
    info = _build_nccl_update_info(sd)
    # backend 由引擎启动时的 WeightTransferConfig 决定，update_info 不带 backend 键
    assert "backend" not in info
    assert info["names"] == list(sd.keys())
    assert info["dtype_names"] == ["bfloat16", "float32"]
    assert info["shapes"] == [[16, 8], [4, 8]]
    assert info["is_checkpoint_format"] is True
    assert info["packed"] is False


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


def test_resolve_visible_device():
    from fullstack_opd_v2.rollout_vllm import _resolve_visible_device
    # 无重排：cuda:1 -> "1"（引擎独立卡）
    assert _resolve_visible_device("cuda:1", None) == "1"
    assert _resolve_visible_device("cuda:0", None) == "0"
    assert _resolve_visible_device("cuda", None) == "0"
    # 交叉分卡重排（CUDA_VISIBLE_DEVICES=1,0）：cuda:1 -> 物理0
    assert _resolve_visible_device("cuda:1", "1,0") == "0"
    assert _resolve_visible_device("cuda:0", "1,0") == "1"
    # 非 cuda：不注入
    assert _resolve_visible_device("cpu", None) is None
    assert _resolve_visible_device("mps", "1,0") is None

# --------------------------- NCCL 权重广播稳定性修复（2026-08-19） ---------------------------
# 覆盖：_prepare_weight_transfer_payload（payload 预检）、_run_with_timeout（主进程侧超时
# fail-fast）、shutdown（NCCL 组清理 + 注册表移除，不再早退）。纯 CPU 可测。
import time


def test_prepare_weight_transfer_payload_ok():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    sd = {
        "w1": torch.ones(4, 8, dtype=torch.bfloat16),
        "w2": torch.zeros(2, 3, dtype=torch.float32),
    }
    out = _prepare_weight_transfer_payload(sd, torch.device("cpu"))
    assert set(out) == {"w1", "w2"}
    for k, v in out.items():
        assert isinstance(v, torch.Tensor)
        assert v.is_contiguous()
        assert v.device == torch.device("cpu")


def test_prepare_weight_transfer_payload_string_device():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    out = _prepare_weight_transfer_payload({"w": torch.zeros(2)}, "cpu")
    assert out["w"].device == torch.device("cpu")


def test_prepare_weight_transfer_payload_non_contiguous_made_contiguous():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    t = torch.zeros(4, 8).t()          # (8,4)，非连续
    assert not t.is_contiguous()
    out = _prepare_weight_transfer_payload({"w": t}, torch.device("cpu"))
    assert out["w"].is_contiguous()
    assert out["w"].shape == (8, 4)


def test_prepare_weight_transfer_payload_empty_dict():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(ValueError):
        _prepare_weight_transfer_payload({}, torch.device("cpu"))


def test_prepare_weight_transfer_payload_non_dict():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(ValueError):
        _prepare_weight_transfer_payload(None, torch.device("cpu"))


def test_prepare_weight_transfer_payload_non_tensor():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(TypeError):
        _prepare_weight_transfer_payload({"w": 42}, torch.device("cpu"))


def test_prepare_weight_transfer_payload_bad_dtype():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(TypeError):
        _prepare_weight_transfer_payload(
            {"w": torch.zeros(2, dtype=torch.complex64)}, torch.device("cpu"))
    with pytest.raises(TypeError):
        _prepare_weight_transfer_payload(
            {"w": torch.zeros(2, dtype=torch.bool)}, torch.device("cpu"))


def test_prepare_weight_transfer_payload_empty_shape():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(ValueError):
        _prepare_weight_transfer_payload({"w": torch.zeros(0, 4)}, torch.device("cpu"))
    with pytest.raises(ValueError):
        _prepare_weight_transfer_payload({"w": torch.zeros(2, 0)}, torch.device("cpu"))


def test_prepare_weight_transfer_payload_bad_device():
    from fullstack_opd_v2.rollout_vllm import _prepare_weight_transfer_payload
    with pytest.raises(RuntimeError):
        _prepare_weight_transfer_payload({"w": torch.zeros(2)}, None)
    with pytest.raises(RuntimeError):
        _prepare_weight_transfer_payload({"w": torch.zeros(2)}, "not-a-device")


def test_run_with_timeout_returns_result():
    from fullstack_opd_v2.rollout_vllm import _run_with_timeout
    assert _run_with_timeout(lambda: 42, 5.0, "t") == 42


def test_run_with_timeout_propagates_error():
    from fullstack_opd_v2.rollout_vllm import _run_with_timeout
    def _boom():
        raise ValueError("boom")
    with pytest.raises(ValueError):
        _run_with_timeout(_boom, 5.0, "t")


def test_run_with_timeout_preserves_systemexit_type():
    from fullstack_opd_v2.rollout_vllm import _run_with_timeout
    def _exit():
        raise SystemExit(7)
    with pytest.raises(SystemExit):
        _run_with_timeout(_exit, 5.0, "t")


def test_run_with_timeout_times_out():
    from fullstack_opd_v2.rollout_vllm import _run_with_timeout
    with pytest.raises(RuntimeError, match="主进程侧超时"):
        _run_with_timeout(lambda: time.sleep(5), 0.2, "slow")


def test_build_nccl_update_info_length_consistency():
    from fullstack_opd_v2.rollout_vllm import _build_nccl_update_info
    info = _build_nccl_update_info({"a": torch.zeros(2), "b": torch.ones(3)})
    assert len(info["names"]) == len(info["dtype_names"]) == len(info["shapes"]) == 2


# --------------------------- shutdown 生命周期清理 ---------------------------
def test_shutdown_cleans_nccl_group_and_registry():
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine, _ACTIVE_ENGINES
    eng = object.__new__(VLLMRolloutEngine)
    eng._shutdown_done = False

    class _LLM:
        def __init__(self):
            self.called = 0
        def shutdown(self):
            self.called += 1

    class _Group:
        destroyed = 0
        def destroy(self):
            _Group.destroyed += 1

    eng.llm = _LLM()
    eng.llm.llm_engine = None       # llm.shutdown 成功路径不应触碰 engine
    eng._wt_group = _Group()
    _ACTIVE_ENGINES.append(eng)
    eng.shutdown()
    assert eng.llm.called == 1
    assert _Group.destroyed == 1
    assert eng._wt_group is None
    assert eng not in _ACTIVE_ENGINES


def test_shutdown_falls_back_to_engine_and_still_cleans():
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine
    eng = object.__new__(VLLMRolloutEngine)
    eng._shutdown_done = False

    class _LLM:
        def shutdown(self):
            raise RuntimeError("llm.shutdown 失败")

    class _Eng:
        def __init__(self):
            self.called = 0
        def shutdown(self):
            self.called += 1

    eng.llm = _LLM()
    eng.llm.llm_engine = _Eng()
    eng._wt_group = object()        # 无 destroy/close 的裸对象
    eng.shutdown()
    assert eng.llm.llm_engine.called == 1
    assert eng._wt_group is None


def test_shutdown_idempotent():
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine, _ACTIVE_ENGINES
    eng = object.__new__(VLLMRolloutEngine)
    eng._shutdown_done = False

    class _LLM:
        def __init__(self):
            self.called = 0
        def shutdown(self):
            self.called += 1

    eng.llm = _LLM()
    eng.llm.llm_engine = None
    eng._wt_group = None
    _ACTIVE_ENGINES.append(eng)
    eng.shutdown()
    eng.shutdown()                  # 第二次调用应直接返回
    assert eng.llm.called == 1
    assert eng not in _ACTIVE_ENGINES
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

# --------------------------- WT poisoned 语义（2026-08-19） ---------------------------
# 覆盖：init/update 超时或失败后 engine 标记 poisoned、poisoned 禁止复用（fail-closed）、
# shutdown 不清除 poisoned、pipeline fail-closed helper。纯 CPU 可测。
import sys
import types


class _FakeNCCLWeightTransferEngine:
    """fake NCCLWeightTransferEngine：trainer_init / trainer_send_weights 按测试替换。"""
    trainer_init = None
    trainer_send_weights = None


class _FakeGroup:
    def __init__(self, device="cpu"):
        self.device = device


class _NoopCM:
    def __init__(self, *a, **k):
        pass
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


class _FakeWTLLM:
    """fake vLLM LLM：init_weight_transfer_engine 成功；collective_rpc 可选阻塞。"""
    def __init__(self, blocking=False):
        self.llm_engine = self
        self.rpc_calls = 0
        self.blocking = blocking
        self.shutdown_called = 0

    def init_weight_transfer_engine(self, info):
        return None

    def collective_rpc(self, *a, **k):
        self.rpc_calls += 1
        if self.blocking:
            time.sleep(3600)
        return None

    def shutdown(self):
        self.shutdown_called += 1


def _install_fake_nccl_module(monkeypatch):
    """把 vllm.distributed.weight_transfer.nccl_engine 注册为 fake 模块（方法级 import 用）。"""
    pkg = types.ModuleType("vllm"); pkg.__path__ = []
    dist = types.ModuleType("vllm.distributed"); dist.__path__ = []
    wt = types.ModuleType("vllm.distributed.weight_transfer"); wt.__path__ = []
    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl.NCCLWeightTransferEngine = _FakeNCCLWeightTransferEngine
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist)
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer", wt)
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.nccl_engine", nccl)


def _fast_run_with_timeout(monkeypatch):
    """把模块级 _run_with_timeout 包一层：统一用 0.1s 超时（真实机制 + 快速，避免 150s 等待）。"""
    import fullstack_opd_v2.rollout_vllm as rv
    _real = rv._run_with_timeout
    def _fast(fn, timeout, label):
        return _real(fn, 0.1, label)
    monkeypatch.setattr(rv, "_run_with_timeout", _fast)


def _patch_cuda_stream(monkeypatch):
    """CPU-only 环境：把 torch.cuda.device / current_stream / set_device 打成 no-op
    （send 线程 2026-08-22 起在函数开头 set_device，覆盖整个 broadcast）。"""
    monkeypatch.setattr(torch.cuda, "device", _NoopCM)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)


def _mk_wt_engine(**attrs):
    """绕过 __init__（需真实 vLLM）构造 engine，预置 WT 相关属性。"""
    from fullstack_opd_v2.rollout_vllm import VLLMRolloutEngine
    eng = object.__new__(VLLMRolloutEngine)
    eng._wt_poisoned = False
    eng._wt_failure_reason = None
    eng._wt_sync_mode = "auto"
    eng._wt_warned = False
    eng._wt_group = None
    eng._wt_init_info = None
    eng._wt_timeout = 120.0
    eng._wt_double_send = True
    eng.tp_size = 1
    eng.device = "cpu"
    eng._learner_device = None
    eng.llm = None
    for k, v in attrs.items():
        setattr(eng, k, v)
    return eng


def test_init_timeout_poisons_engine(monkeypatch):
    _install_fake_nccl_module(monkeypatch)
    _fast_run_with_timeout(monkeypatch)
    eng = _mk_wt_engine()
    eng.llm = _FakeWTLLM()
    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: time.sleep(3600)
    with pytest.raises(RuntimeError):
        eng._weight_transfer_init_16()
    assert eng.weight_sync_poisoned is True
    assert "init:" in (eng.weight_sync_poison_reason or "")
    # 再次调用 update_weights 立即失败（不等待 3600s）
    with pytest.raises(RuntimeError, match="poisoned"):
        eng.update_weights({"w": torch.zeros(2)})


def test_update_first_round_timeout_poisons_engine(monkeypatch):
    _install_fake_nccl_module(monkeypatch)
    _fast_run_with_timeout(monkeypatch)
    _patch_cuda_stream(monkeypatch)
    eng = _mk_wt_engine(_wt_double_send=True)
    llm = _FakeWTLLM(blocking=True)
    eng.llm = llm
    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: _FakeGroup("cpu")
    _FakeNCCLWeightTransferEngine.trainer_send_weights = lambda *a, **k: None
    with pytest.raises(RuntimeError):
        eng._weight_transfer_update_16({"w": torch.zeros(2, dtype=torch.float32)})
    assert eng.weight_sync_poisoned is True
    assert llm.rpc_calls == 1          # 第一发超时 → 第二发不执行
    with pytest.raises(RuntimeError, match="poisoned"):
        eng.update_weights({"w": torch.zeros(2)})


def test_sender_exception_poisons_engine(monkeypatch):
    _install_fake_nccl_module(monkeypatch)
    _fast_run_with_timeout(monkeypatch)
    _patch_cuda_stream(monkeypatch)
    eng = _mk_wt_engine()
    eng.llm = _FakeWTLLM()
    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: _FakeGroup("cpu")
    def _bad_send(*a, **k):
        raise RuntimeError("fake nccl send failure")
    _FakeNCCLWeightTransferEngine.trainer_send_weights = _bad_send
    with pytest.raises(RuntimeError, match="fake nccl send failure"):
        eng._weight_transfer_update_16({"w": torch.zeros(2, dtype=torch.float32)})
    assert eng.weight_sync_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        eng.update_weights({"w": torch.zeros(2)})


@pytest.mark.parametrize("bad_sd", [
    {},                                                      # 空 dict
    {"w": 42},                                               # 非 tensor
    {"w": torch.zeros(2, dtype=torch.complex64)},            # 非支持 dtype
    {"w": torch.zeros(0, 4)},                                # 空 shape
])
def test_payload_preflight_failure_poisons_engine(monkeypatch, bad_sd):
    _install_fake_nccl_module(monkeypatch)
    _fast_run_with_timeout(monkeypatch)
    eng = _mk_wt_engine()
    eng.llm = _FakeWTLLM()
    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: _FakeGroup("cpu")
    _FakeNCCLWeightTransferEngine.trainer_send_weights = lambda *a, **k: None
    with pytest.raises((ValueError, TypeError)):
        eng._weight_transfer_update_16(bad_sd)
    assert eng.weight_sync_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        eng.update_weights({"w": torch.zeros(2)})


def test_shutdown_does_not_clear_poisoned(monkeypatch):
    _install_fake_nccl_module(monkeypatch)
    eng = _mk_wt_engine()
    llm = _FakeWTLLM()
    eng.llm = llm
    eng._mark_weight_transfer_poisoned("test reason")
    assert eng.weight_sync_poisoned is True
    eng.shutdown()
    assert eng.weight_sync_poisoned is True        # shutdown 不清除 poisoned
    assert eng.weight_sync_poison_reason == "test reason"
    with pytest.raises(RuntimeError, match="poisoned"):
        eng.update_weights({"w": torch.zeros(2)})


# --------------------------- pipeline fail-closed（2026-08-19） ---------------------------
def test_pipeline_raise_if_weight_sync_poisoned():
    from fullstack_opd_v2.pipeline import _raise_if_weight_sync_poisoned
    class _Mr:
        def __init__(self):
            self.records = []
        def record(self, m):
            self.records.append(dict(m))
    class _Poisoned:
        weight_sync_poisoned = True
    mr = _Mr()
    with pytest.raises(RuntimeError, match="poisoned"):
        _raise_if_weight_sync_poisoned(_Poisoned(), mr=mr,
                                       metric_key="rollout/weight_sync_poisoned",
                                       label="vLLM")
    assert mr.records == [{"rollout/weight_sync_poisoned": 1}]
    # 未 poisoned → 不抛、不记录
    class _OK:
        weight_sync_poisoned = False
    mr2 = _Mr()
    _raise_if_weight_sync_poisoned(_OK(), mr=mr2,
                                   metric_key="rollout/weight_sync_poisoned",
                                   label="vLLM")
    assert mr2.records == []

# --------------------------- prompt 右填充去填充（2026-08-19） ---------------------------
def test_strip_prompt_padding_basic():
    # 输入是 (B, P) 定长矩阵（数据层右 pad 到 max_prompt_len），各 P 同长
    from fullstack_opd_v2.rollout_vllm import _strip_prompt_padding
    prompts = torch.tensor([
        [1, 2, 3, 9, 9, 9],      # 去尾部 pad
        [4, 5, 9, 9, 9, 9],      # 更多 pad
        [7, 8, 9, 9, 9, 9],      # 少 pad
        [9, 9, 9, 9, 9, 9],      # 全 pad
    ])
    out = _strip_prompt_padding(prompts, 9)
    assert out == [[1, 2, 3], [4, 5], [7, 8], [9]]


def test_strip_prompt_padding_preserves_inner_pad():
    # 中间的 pad token 不去（只去右侧连续尾部）
    from fullstack_opd_v2.rollout_vllm import _strip_prompt_padding
    prompts = torch.tensor([[1, 9, 2, 9, 9]])
    assert _strip_prompt_padding(prompts, 9) == [[1, 9, 2]]


def test_strip_prompt_padding_empty_row_fallback():
    # 空行兜底为 [pad_id]（避免 vLLM 收到空 prompt）
    from fullstack_opd_v2.rollout_vllm import _strip_prompt_padding
    prompts = torch.tensor([[9, 9], [1, 9]], dtype=torch.long)
    out = _strip_prompt_padding(prompts, 9)
    assert out == [[9], [1]]

# --------------------------- NCCL trainer_init 线程内 set_device（2026-08-22） ---------------------------
def test_nccl_trainer_init_sets_device_in_thread(monkeypatch):
    """_weight_transfer_init_16：trainer_init 在 _run_with_timeout 新线程执行，torch.cuda.
    set_device 必须在线程内生效（线程局部）。E2 交叉布局（train@GPU1+vLLM@GPU0）实测
    rank0 落在默认 cuda:0 与 worker 冲突 → Duplicate GPU；修复后线程内显式 set_device。
    """
    import torch as _t
    import fullstack_opd_v2.rollout_vllm as rv
    _install_fake_nccl_module(monkeypatch)
    # 记录 set_device 的调用参数（CPU-only 无法真设，no-op 记录即可）
    set_calls = []
    _orig_set = _t.cuda.set_device
    monkeypatch.setattr(_t.cuda, "set_device", lambda d: set_calls.append(str(d)))
    # trainer_init 记录被调用（fake 返回带 device 的 group）
    inited = {}

    class _FakeGroup:
        device = "cuda:1"

    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: inited.setdefault("n", 0) or _FakeGroup()
    _FakeNCCLWeightTransferEngine.trainer_send_weights = lambda *a, **k: None

    eng = _mk_wt_engine(_learner_device="cuda:1")
    class _LLM:
        def __init__(self):
            self.llm_engine = self
        def init_weight_transfer_engine(self, info):
            return None
    eng.llm = _LLM()
    eng._weight_transfer_init_16()
    # set_device 必须被调用（至少线程内一次，device 应为 cuda:1 —— 训练卡）
    assert "cuda:1" in set_calls, f"set_device 未用训练卡：{set_calls}"
    assert inited.get("n") is not None
    _t.cuda.set_device = _orig_set

# --------------------------- _send 线程设备作用域（2026-08-22） ---------------------------
def test_send_sets_device_before_broadcast(monkeypatch):
    """_weight_transfer_broadcast_round 的 _send：set_device(_wt_dev) 必须在广播前、
    覆盖整个函数（线程局部）——否则 E2 sender 线程默认 cuda:0 与 cuda:1 communicator 错位。"""
    import fullstack_opd_v2.rollout_vllm as rv
    _install_fake_nccl_module(monkeypatch)
    _patch_cuda_stream(monkeypatch)
    set_calls = []
    monkeypatch.setattr(torch.cuda, "set_device",
                        lambda d: set_calls.append(str(d)))
    eng = _mk_wt_engine(_wt_double_send=True)
    class _LLM:
        def __init__(self):
            self.llm_engine = self
        def collective_rpc(self, *a, **k):
            return None
    eng.llm = _LLM()
    _FakeNCCLWeightTransferEngine.trainer_init = lambda info: _FakeGroup("cuda:1")
    # 记录广播调用时已 set_device（模拟线程当前设备 = group 设备）
    sent = []
    def _fake_send(iterable, group, stream):
        sent.append(set_calls[-1] if set_calls else None)
    _FakeNCCLWeightTransferEngine.trainer_send_weights = _fake_send
    # 直接跑 broadcast_round（绕过 update_16 的 init——这里单独验证 send 线程设备作用域）
    eng._wt_group = _FakeGroup("cuda:1")
    from fullstack_opd_v2.rollout_vllm import _build_nccl_update_info
    sd = {"w": torch.zeros(2, dtype=torch.float32)}
    info = _build_nccl_update_info(sd)
    eng._weight_transfer_broadcast_round(sd, info, "test")
    # 广播执行前必须已 set_device 到 group 设备（cuda:1）
    assert "cuda:1" in set_calls
    assert sent and sent[0] == "cuda:1", f"广播时线程设备未设为 cuda:1: {sent}"

"""model_factory.py 单测：可插拔模型工厂（toy + HF 骨架）。"""
import unittest.mock as mock

import pytest
import torch

from fullstack_opd_v2.model import CausalToyLM, build_length_mask
from fullstack_opd_v2.model_factory import build_model, HFCausalLM
from fullstack_opd_v2.exceptions import ModelError


def _cfg(**over):
    cfg = dict(vocab_size=64, d_model=48, n_layers=2, model_kind="toy")
    cfg.update(over)
    return cfg


def _fake_hf_module(vocab=152, hidden=768, maxlen=1024, n_layers=28):
    """模拟 transformers HF 模块（config + __call__ 返回 .logits + 训练/权重委托）。

    HFCausalLM.__init__ 走 `from_pretrained(...).to(device).eval()` 链式调用，
    故 to/eval 必须返回 self（否则 self.model 变成链条上新的 Mock）。
    """
    mod = mock.Mock()
    mod.config.vocab_size = vocab
    mod.config.hidden_size = hidden
    mod.config.max_position_embeddings = maxlen
    mod.config.num_hidden_layers = n_layers
    mod.training = False
    mod.to = mock.Mock(return_value=mod)
    mod.eval = mock.Mock(return_value=mod)

    def _call(input_ids):
        out = mock.Mock()
        out.logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab)
        return out
    mod.side_effect = _call
    # HFCausalLM.parameters/named_parameters 委托后需可迭代（去重逻辑 for p in ...）
    mod.parameters = mock.Mock(return_value=[])
    mod.named_parameters = mock.Mock(return_value=iter([]))
    return mod


def _patch_hf_factory(monkeypatch):
    """monkeypatch model_factory._HF_AutoModelForCausalLM → 返回 fake HF 模块的工厂。"""
    import fullstack_opd_v2.model_factory as MF
    mod = _fake_hf_module()
    factory = mock.Mock()
    factory.from_pretrained = mock.Mock(return_value=mod)
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)
    return factory, mod


def test_build_model_toy_returns_causal_toy_lm():
    m = build_model(_cfg(), "cpu")
    assert isinstance(m, CausalToyLM)
    assert m.vocab == 64


def test_build_model_unknown_kind_raises():
    with pytest.raises(ModelError):
        build_model(_cfg(model_kind="megatron"), "cpu")
    with pytest.raises(ModelError):
        build_model(_cfg(model_kind="vllm"), "cpu")


# --------------------------- HF 骨架（需 GPU 验证的适配器，本地只测接口/错误路径）---------------------------
def test_build_model_hf_missing_path_raises():
    """model_kind='hf' 但缺学生/教师路径 → 显式 ModelError（不静默走 toy）。"""
    with pytest.raises(ModelError):
        build_model(_cfg(model_kind="hf", student_path=None), "cpu", role="student")
    with pytest.raises(ModelError):
        build_model(_cfg(model_kind="hf", student_path="S"), "cpu", role="teacher")


def test_build_model_hf_loads_student_path(monkeypatch):
    factory, _ = _patch_hf_factory(monkeypatch)
    m = build_model(_cfg(model_kind="hf", student_path="MyStudent"), "cpu", role="student")
    assert isinstance(m, HFCausalLM)
    assert m.vocab == 152
    factory.from_pretrained.assert_called_once_with("MyStudent", torch_dtype=None)


def test_hf_causal_lm_interface_delegates(monkeypatch):
    """HFCausalLM 暴露 CausalToyLM 兼容接口：forward→logits、response_dists→(B,T,V)、委托。"""
    _patch_hf_factory(monkeypatch)
    m = HFCausalLM("fake/path", "cpu", dtype="auto")
    assert m.vocab == 152 and m.d_model == 768 and m.max_len == 1024
    assert m.n_layers == 28      # P1-B：scheduler 用 student.n_layers 构造 worker
    # forward → logits (B,L,V)
    ids = torch.zeros(2, 5, dtype=torch.long)
    assert m(ids).shape == (2, 5, 152)
    # response_dists → (B,T,V)（prompts(2,3)+responses(2,4) → 切片 (2,4,152)）
    r = m.response_dists(torch.zeros(2, 3, dtype=torch.long),
                         torch.zeros(2, 4, dtype=torch.long))
    assert r.shape == (2, 4, 152)
    # 训练/权重委托
    m.state_dict()
    m.parameters()
    m.train()
    m.eval()
    assert m.training is False


def test_hf_parameters_dedupes_tied_embeddings(monkeypatch):
    """P2（二次审查）：HF tie_word_embeddings 下 parameters() 产出同一对象两次 →
    Adam/clip 双更新（≈2× 步长）。HFCausalLM.parameters 必须按对象 id 去重。"""
    import fullstack_opd_v2.model_factory as MF
    p = torch.nn.Parameter(torch.zeros(4))
    mod = mock.Mock()
    mod.config.vocab_size = 152
    mod.config.hidden_size = 8
    mod.config.max_position_embeddings = 64
    mod.config.num_hidden_layers = 2
    mod.training = False
    mod.to = mock.Mock(return_value=mod)
    mod.eval = mock.Mock(return_value=mod)
    mod.parameters.return_value = iter([p, p])       # tied：同一对象两次
    mod.side_effect = lambda ids: mock.Mock(logits=torch.zeros(ids.size(0), ids.size(1), 152))
    factory = mock.Mock()
    factory.from_pretrained.return_value = mod
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)
    m = HFCausalLM("fake", "cpu")
    params = list(m.parameters())
    assert len(params) == 1 and params[0] is p


def test_hf_dtype_bfloat16_not_silently_fp32(monkeypatch):
    """P3（二次审查）：config Literal 允许 'bfloat16'，适配器字典必须覆盖，否则静默 fp32。"""
    import fullstack_opd_v2.model_factory as MF
    factory, _ = _patch_hf_factory(monkeypatch)
    HFCausalLM("fake", "cpu", dtype="bfloat16")
    factory.from_pretrained.assert_called_once_with(
        "fake", torch_dtype=torch.bfloat16)


# --------------------------- L2 §2.3：HFCausalLM generate_batch + attention_mask（rollout 骨架）---------------------------
def test_hf_call_passes_attention_mask(monkeypatch):
    """HFCausalLM.__call__(input_ids, attention_mask=None) 传 attention_mask（§2.3 变长序列骨架）。"""
    import fullstack_opd_v2.model_factory as MF
    seen = {}
    mod = mock.Mock()
    mod.config.vocab_size = 152
    mod.config.hidden_size = 8
    mod.config.max_position_embeddings = 64
    mod.config.num_hidden_layers = 2
    mod.training = False
    mod.to = mock.Mock(return_value=mod)
    mod.eval = mock.Mock(return_value=mod)
    mod.parameters.return_value = []

    def _call(input_ids, attention_mask=None):
        seen["mask"] = attention_mask
        out = mock.Mock()
        out.logits = torch.zeros(input_ids.size(0), input_ids.size(1), 152)
        return out
    mod.side_effect = _call
    factory = mock.Mock()
    factory.from_pretrained.return_value = mod
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)

    m = HFCausalLM("fake", "cpu")
    ids = torch.zeros(2, 5, dtype=torch.long)
    mask = torch.ones(2, 5, dtype=torch.long)
    m(ids)                       # 未传 mask：attention_mask 应为 None（兼容 toy 等长路径）
    assert seen["mask"] is None
    m(ids, attention_mask=mask)  # 传 mask：透传到 HF 模型
    assert seen["mask"] is mask


def test_hf_generate_batch_delegates(monkeypatch):
    """HFCausalLM.generate_batch 委托 model.generate 并只返回新生成部分（§2.3 rollout 骨架）。"""
    import fullstack_opd_v2.model_factory as MF
    mod = _fake_hf_module(vocab=152)
    out = torch.zeros(2, 8, dtype=torch.long)          # P(5)+T(3)
    mod.generate = mock.Mock(return_value=out)
    factory = mock.Mock()
    factory.from_pretrained.return_value = mod
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)

    m = HFCausalLM("fake", "cpu")
    prompts = torch.zeros(2, 5, dtype=torch.long)
    generated = m.generate_batch(prompts, max_new=3, temperature=1.0)
    # 委托 model.generate，返回 (B, max_new)=去掉 prompt 部分
    assert generated.shape == (2, 3)
    args, kw = mod.generate.call_args
    assert args[0] is prompts
    assert kw["max_new_tokens"] == 3
    assert kw["do_sample"] is True


# --------------------------- Stage 2：HFCausalLM.generate_with_status（真实 HF rollout 解阻塞）------------------
def _hf_m(monkeypatch):
    """构造 HFCausalLM（fake HF 模块，active 前向，generate_with_status 可跑）。"""
    import fullstack_opd_v2.model_factory as MF
    mod = _fake_hf_module(vocab=152)
    factory = mock.Mock()
    factory.from_pretrained.return_value = mod
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)
    return HFCausalLM("fake", "cpu")


def test_hf_generate_with_status_no_eos_all_budget(monkeypatch):
    """eos_token_id=None → 永不判 EOS，全 budget_stop，mask 全有效。"""
    m = _hf_m(monkeypatch)
    pr = torch.randint(0, 152, (2, 5))
    out = m.generate_with_status(pr, max_new=8, eos_token_id=None)
    assert out["statuses"] == ["budget_stop", "budget_stop"]
    assert out["lengths"] == [8, 8]
    assert out["eos_pos"] == [None, None]
    assert out["looped"] == [False, False]
    mask = build_length_mask(out["responses"], out["lengths"], out["eos_pos"])
    assert mask.size() == (2, 8)
    assert mask.sum(1).tolist() == [8, 8]


def test_hf_generate_with_status_eos_stops(monkeypatch):
    """首步采到 eos=0 → 提前停，length=eos_pos+1，mask eos 后全 0。"""
    import torch as _t
    m = _hf_m(monkeypatch)
    pr = torch.randint(0, 152, (1, 5))
    calls = {"n": 0}
    def fake_multinomial(probs, num_samples=1):
        calls["n"] += 1
        return _t.tensor([[0]]) if calls["n"] == 1 else _t.tensor([[1]])
    monkeypatch.setattr(_t, "multinomial", fake_multinomial)
    out = m.generate_with_status(pr, max_new=8, eos_token_id=0)
    assert out["statuses"] == ["eos"]
    assert out["lengths"] == [1]
    assert out["eos_pos"] == [0]
    mask = build_length_mask(out["responses"], out["lengths"], out["eos_pos"])
    assert mask.sum(1).tolist() == [1]


def test_hf_generate_with_status_loop_detected(monkeypatch):
    """周期 3 重复尾部 → 判 loop，looped=True。"""
    import torch as _t
    m = _hf_m(monkeypatch)
    pr = torch.randint(0, 152, (1, 5))
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    it = iter(seq)
    monkeypatch.setattr(_t, "multinomial",
                        lambda probs, num_samples=1: _t.tensor([[next(it)]]))
    out = m.generate_with_status(pr, max_new=9, eos_token_id=None, loop_periods=(3,))
    assert out["statuses"] == ["loop"]
    assert out["looped"] == [True]


# --------------------------- Stage 2：KV-cached 快速 rollout（真实 HF 大规模，152k 词表）-------------------
def _hf_m_with_generate(monkeypatch, gen_out):
    """构造带 model.generate 的 HFCausalLM（generate_with_status_kv 走 HF generate 路径）。"""
    import fullstack_opd_v2.model_factory as MF
    mod = _fake_hf_module(vocab=152)
    mod.generate = mock.Mock(return_value=gen_out)
    factory = mock.Mock()
    factory.from_pretrained.return_value = mod
    monkeypatch.setattr(MF, "_HF_AutoModelForCausalLM", factory)
    return HFCausalLM("fake", "cpu")


def test_hf_generate_with_status_kv_budget_stop(monkeypatch):
    """KV 路径 eos=None → HF generate（eos=-1）+ 全 budget_stop，length==max_new。

    HF generate 返回【prompt+生成】拼接（out[:, prompt_len:] 剥 prompt），故合成
    gen_out 需含 prompt 前缀段。
    """
    pr = torch.randint(1, 152, (2, 5))
    gen = torch.ones(2, 8, dtype=torch.long)                 # 8 个非 pad 生成 token
    m = _hf_m_with_generate(monkeypatch, torch.cat([pr, gen], dim=1))
    out = m.generate_with_status_kv(pr, max_new=8, eos_token_id=None, pad_id=0,
                                    loop_detection=False)    # 全 1 会触发 loop，先关检测
    args, kw = m.model.generate.call_args
    assert kw["eos_token_id"] == -1                           # 永不 EOS
    assert kw["max_new_tokens"] == 8
    assert out["statuses"] == ["budget_stop", "budget_stop"]
    assert out["lengths"] == [8, 8]
    assert out["eos_pos"] == [None, None]


def test_hf_generate_with_status_kv_eos(monkeypatch):
    """KV 路径 eos=0 → 序列含 0 时判 eos，length=eos_pos+1。"""
    pr = torch.ones(1, 5, dtype=torch.long)
    gen = torch.tensor([[5, 0, 9, 9]])                        # 生成段 [5, eos=0, pad, pad]
    m = _hf_m_with_generate(monkeypatch, torch.cat([pr, gen], dim=1))
    out = m.generate_with_status_kv(pr, max_new=4, eos_token_id=0, pad_id=9)
    assert out["statuses"] == ["eos"]
    assert out["lengths"] == [2]
    assert out["eos_pos"] == [1]


def test_hf_generate_with_status_kv_pad_stripped(monkeypatch):
    """KV 路径 budget 撞满补 pad → 去尾部 pad 再判，不误判 invalid。"""
    pr = torch.ones(1, 5, dtype=torch.long)
    gen = torch.tensor([[1, 2, 9, 9]])                        # 尾部 2 个 pad=9
    m = _hf_m_with_generate(monkeypatch, torch.cat([pr, gen], dim=1))
    out = m.generate_with_status_kv(pr, max_new=4, eos_token_id=None, pad_id=9,
                                    loop_detection=False)
    assert out["statuses"] == ["budget_stop"]

"""可插拔模型工厂：把 pipeline 里硬编码的 `CausalToyLM(...)` 构造集中起来。

- `build_model(cfg, device, role)` 按 `cfg["model_kind"]` 构造学生/教师模型。
- 默认 `model_kind="toy"` → `CausalToyLM`（单元可测的内核）。
- `model_kind="hf"` → `HFCausalLM`（真实 HF 模型适配器，**骨架**：代码完整但需 GPU/真实
  模型验证；本地无 GPU/模型无法实测）。megatron/vllm 仍抛 `ModelError`（由 async-opd/vLLM 承担）。
- `role` 用于区分 student/teacher：hf 下 student→`cfg["student_path"]`、teacher→`cfg["teacher_rl_path"]`。
"""

from __future__ import annotations

import torch

from .exceptions import ModelError
from .model import CausalToyLM

# transformers 可选导入（HF 骨架；未装时仅 HF 分支报错，toy 不受影响）
try:
    from transformers import AutoModelForCausalLM as _HF_AutoModelForCausalLM
    _HF_AVAILABLE = True
except Exception:                                     # pragma: no cover
    _HF_AutoModelForCausalLM = None
    _HF_AVAILABLE = False


class HFCausalLM:
    """真实 HF 模型适配器（⚠️ 骨架：需 GPU/真实模型验证）。

    包 `transformers.AutoModelForCausalLM`，暴露与 `CausalToyLM` 兼容接口，使
    pipeline/scheduler/losses 内核零改动：
      `__call__(ids) -> logits (B,L,V)`（供 `response_dists` / `generate_batch` 用）、
      `response_dists(prompts, responses) -> (B,T,V)`、`vocab / d_model / max_len`、
      `train / eval / to / training / state_dict / load_state_dict / parameters / zero_grad`。

    ⚠️ 骨架边界（GPU 验证时需逐项确认）：
    - 输入是【已 tokenize 的 id 张量】（等长、无 padding mask）——toy 流水线假设等长；
      真实变长序列需在 data 层补 `attention_mask`（forward 目前只传 input_ids）。
    - 未实现 KV cache：`generate_batch` 对增长序列 O(T²)（真实规模应走 vLLM rollout）。
    - 权重同步/断点走 `state_dict`（含 tied embedding/lm_head，load_state_dict 正常）。
    """

    def __init__(self, path: str, device: str = "cpu", dtype: str = "auto",
                 vocab: int | None = None, d_model: int | None = None,
                 max_len: int | None = None,
                 attn_implementation: str | None = None):
        if not _HF_AVAILABLE:
            raise ModelError("model_kind='hf' 需要 transformers（统一 GPU 环境应含）")
        # P2 修复（二次审查）：config Literal 允许 "bfloat16"/"float32"，适配器字典要覆盖，
        # 否则合法 dtype 静默落到 None（fp32）。"fp32" 意图即 fp32（torch_dtype=None）。
        _DT = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
               "float16": torch.float16, "fp16": torch.float16,
               "fp32": None, "float32": None, "auto": None}
        td = _DT.get(str(dtype).lower())
        # P-回归/显存修复：训练长序列（P+T≈3072）× 大隐层下，默认 SDPA 会物化完整
        # 注意力矩阵 (B,heads,T,T) 并保留给 backward → 28 层累积 80+GB OOM（部署实测）。
        # eval 早已用 flash_attention_2；训练模型同样开启（CUDA 环境 flash_attn 已装）。
        # None → 不传（CPU 单测/无 flash 环境回退默认，from_pretrained 断言不受影响）。
        _PT_KW = {"torch_dtype": td}
        if attn_implementation:
            _PT_KW["attn_implementation"] = attn_implementation
        try:
            self.model = _HF_AutoModelForCausalLM.from_pretrained(
                path, **_PT_KW).to(device).eval()
        except Exception as e:
            raise ModelError(f"HF 模型 {path!r} 加载失败（路径/HF id 无效？）：{e}") from e
        self.path = path
        self.device = device
        # 尺寸以真实模型 config 为准（toy 的 cfg["vocab_size"]=64 会错）
        self.vocab = vocab or self.model.config.vocab_size
        self.d_model = d_model or getattr(self.model.config, "hidden_size", None)
        self.max_len = max_len or getattr(self.model.config,
                                          "max_position_embeddings", 2048)
        # P1-B（二次审查）：scheduler 用 student.n_layers 构造 worker（CausalToyLM 分支）；
        # HFCausalLM 之前没有该属性 → model_kind='hf' 构造调度器即 AttributeError。
        self.n_layers = getattr(self.model.config, "num_hidden_layers", None)
        # P1.5：真实 pad_token_id（供变长 rollout 右 pad 与尾部去 pad；Qwen3 的 token 0
        # 不是 pad，默认 0 会误判/错误填充）。
        self._pad_token_id = getattr(self.model.config, "pad_token_id", None)
        # IMP-1 根因修复：Qwen 系 config.pad_token_id=None（HF 模型配置常不带 pad），
        # 但 tokenizer 有真实 pad（Qwen3=151643）。若保持 None，pipeline 会把 rollout
        # pad_id 落到 rollcfg.pad_id=0 → HF generate 自动推断 attention_mask 时无法识别
        # 数据层 right-pad（151643）→ 800+ pad 被当作有效上下文 → 长序列尾部 token 重复
        # （训练路径 75% loop vs 校准路径 0% loop 的矛盾根因，2026-08-17 GPU 实测定位）。
        # 回落到 tokenizer.pad_token_id：HF mask 推断正确，尾部去 pad 判据也正确。
        if self._pad_token_id is None:
            try:
                from transformers import AutoTokenizer
                self._pad_token_id = AutoTokenizer.from_pretrained(path).pad_token_id
            except Exception:
                self._pad_token_id = None

    @property
    def pad_token_id(self):
        return self._pad_token_id

    # ---- 内核用接口：forward / response_dists / generate（委托模块级实现）----
    def __call__(self, input_ids: torch.Tensor,
                 attention_mask=None) -> torch.Tensor:
        """(B,L) -> logits (B,L,V)，与 CausalToyLM.forward 对齐（供 response_dists 用）。

        §2.3：真实变长序列须传 attention_mask（L2 rollout 相位变长支持）；toy 等长路径可省略。
        """
        kw = {"input_ids": input_ids}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        return self.model(**kw).logits

    @torch.no_grad()
    @property
    def config(self):
        """代理到内部 HF 模型（scheduler 的 gradient_checkpointing/use_cache 读取）。"""
        return self.model.config

    def gradient_checkpointing_enable(self):
        """代理到内部 HF 模型（2026-08-18：backward OOM 根治的激活重计算开关）。"""
        return self.model.gradient_checkpointing_enable()

    def generate_batch(self, prompts: torch.Tensor, max_new: int = 8192,
                       temperature: float = 1.0) -> torch.Tensor:
        """自回归生成（§2.3 骨架：委托 HF generate，真实规模应走 vLLM）。

        (B,P) -> (B,T) 仅返回【新生成】部分（去掉 prompt），与 CausalToyLM.generate_batch 对齐。
        """
        out = self.model.generate(
            prompts, max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=0.95)
        return out[:, prompts.size(1):]

    @torch.no_grad()
    def generate_with_status(self, prompts: torch.Tensor, max_new: int,
                             eos_token_id=None, temperature: float = 1.0,
                             pad_id: int = 0, loop_detection: bool = True,
                             loop_periods=(2, 3, 4),
                             repetition_penalty: float = 1.0,
                             loop_min_len: int = 8) -> dict:
        """Stage 2：HF 短预算 rollout，与 model.generate_with_status 返回同构 dict。

        ⚠️ 骨架边界：无 KV cache，逐 token 前向 O(T²) —— 校准/小批量验证用；真实大规模
        rollout 应走 vLLM（rollout_engine='vllm'，瓶颈在 vLLM 端）。语义与 toy/vLLM 端
        完全对齐：eos_token_id=None → 永不判 EOS（全 budget_stop，除非 loop）；loop/invalid
        判定优先级 loop > invalid > eos > budget_stop。pad_id 只填变长空白，不参与判定。
        """
        from .model import detect_loop, build_length_mask as _blm
        from .model import apply_repetition_penalty
        B, P = prompts.size(0), prompts.size(1)
        device = prompts.device
        responses = torch.full((B, max_new), pad_id, dtype=torch.long, device=device)
        eos_pos: list[int | None] = [None] * B
        alive = torch.ones(B, dtype=torch.bool, device=device)
        was_training = self.training
        self.eval()
        for t in range(max_new):
            if not alive.any():
                break
            idx = alive.nonzero(as_tuple=False).squeeze(-1)   # (n_a,) alive 样本同长
            ctx = torch.cat([prompts[idx], responses[idx, :t]], dim=1)
            logits = self(ctx)[:, -1]                          # (n_a, V)
            if repetition_penalty > 1.0:
                logits = apply_repetition_penalty(
                    logits, responses[idx, :t], repetition_penalty)
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)      # (n_a,)
            responses[idx, t] = tok
            if eos_token_id is not None:
                hit = (tok == eos_token_id)
                for j, i in enumerate(idx.tolist()):
                    if bool(hit[j]):
                        eos_pos[i] = t
                        alive[i] = False
        if was_training:
            self.train()
        lengths = [max_new] * B
        for i in range(B):
            if eos_pos[i] is not None:
                lengths[i] = eos_pos[i] + 1
        statuses: list[str] = []
        looped: list[bool] = []
        for i in range(B):
            eff = responses[i, :max(1, lengths[i])]
            loop = loop_detection and detect_loop(eff, loop_periods, min_len=loop_min_len)
            if loop:
                statuses.append("loop"); looped.append(True)
            elif lengths[i] == 0:
                statuses.append("empty"); looped.append(False)   # IMP-1d：长度 0=empty
            elif eos_pos[i] is not None:
                statuses.append("eos"); looped.append(False)
            else:
                statuses.append("budget_stop"); looped.append(False)
        return {"responses": responses, "statuses": statuses, "lengths": lengths,
                "eos_pos": eos_pos, "looped": looped}

    @torch.no_grad()
    def generate_with_status_kv(self, prompts: torch.Tensor, max_new: int,
                                eos_token_id=None, temperature: float = 1.0,
                                pad_id: int = 0, loop_detection: bool = True,
                                loop_periods=(2, 3, 4),
                                repetition_penalty: float = 1.0,
                                loop_min_len: int = 8) -> dict:
        """Stage 2 真实 HF 大规模 rollout：KV-cached 快速路径，与 generate_with_status 同构。

        ⚠️ 性能边界：`generate_with_status`（无 KV cache 的逐 token 前向）在真实 152k 词表 +
        长序列下慢 1-2 个数量级（每 token 重算全前缀）。本方法用 HF `generate` 的 KV cache
        加速（~35 tok/s），采样后按 `detect_loop`/EOS 后处理，返回与 toy 完全同构的 dict。

        eos_token_id=None → 传 -1 让 HF 永不因 EOS 停（撞 max_new 才停，全 budget_stop，
        除非 loop），faithful 到协议「预算截断是常态」。末尾 pad_id 去除后再判 loop/EOS。
        """
        from .model import detect_loop
        B = prompts.size(0)
        device = prompts.device
        eos = int(eos_token_id) if eos_token_id is not None else -1   # -1 永不匹配
        _gen_kw = dict(max_new_tokens=int(max_new), do_sample=True,
                       temperature=max(float(temperature), 1e-6), top_p=0.95,
                       pad_token_id=pad_id, eos_token_id=eos)
        if repetition_penalty is not None and repetition_penalty > 1.0:
            _gen_kw["repetition_penalty"] = float(repetition_penalty)
        out = self.model.generate(prompts, **_gen_kw)
        new = out[:, prompts.size(1):]                       # (B, T_实际) 右 pad
        responses = torch.full((B, int(max_new)), pad_id, dtype=torch.long, device=device)
        responses[:, :new.size(1)] = new
        statuses: list[str] = []
        lengths: list[int] = []
        eos_pos: list[int | None] = []
        looped: list[bool] = []
        for i in range(B):
            seq = responses[i].tolist()
            L = int(max_new)
            while L > 0 and seq[L - 1] == pad_id:            # 去尾部 pad
                L -= 1
            ep = None
            if eos_token_id is not None and eos_token_id in seq[:L]:
                ep = seq.index(eos_token_id)
                L = ep + 1
            loop = loop_detection and detect_loop(
                torch.tensor(seq[:max(1, L)]), loop_periods, min_len=loop_min_len)
            if loop:
                statuses.append("loop"); looped.append(True)
            elif L == 0:
                statuses.append("empty"); looped.append(False)   # IMP-1d：长度 0=empty
            elif ep is not None:
                statuses.append("eos"); looped.append(False)
            else:
                statuses.append("budget_stop"); looped.append(False)
            lengths.append(L); eos_pos.append(ep)
        return {"responses": responses, "statuses": statuses, "lengths": lengths,
                "eos_pos": eos_pos, "looped": looped}

    def response_dists(self, prompts: torch.Tensor, responses: torch.Tensor,
                       dtype: torch.dtype | None = None):
        from .model import response_dists
        return response_dists(self, prompts, responses, dtype=dtype)

    # ---- 训练/权重接口：委托给 HF 模块 ----
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    @property
    def training(self) -> bool:
        return self.model.training

    def state_dict(self, *a, **k):
        return self.model.state_dict(*a, **k)

    def load_state_dict(self, *a, **k):
        return self.model.load_state_dict(*a, **k)

    def parameters(self, *a, **k):
        # P2（二次审查）：HF tie_word_embeddings 时 lm_head.weight 与 embed_tokens.weight
        # 是同一 Parameter 对象，parameters() 产出两次 → Adam / clip_grad_norm 对绑定权重
        # 双更新（等效步长 ≈2×，确定性静默语义错误）。按对象 id 去重。
        # ⚠️ 返回【生成器】而非 list——cache.py build 用 next(teacher.parameters()) 取 device，
        #     list 不可迭代（P3 实测 TypeError）。生成器每次调用重新迭代，可被多次消费。
        seen = set()
        for p in self.model.parameters(*a, **k):
            if id(p) not in seen:
                seen.add(id(p))
                yield p

    def named_parameters(self, *a, **k):
        seen = set()
        for name, p in self.model.named_parameters(*a, **k):
            if id(p) not in seen:
                seen.add(id(p))
                yield name, p

    def zero_grad(self, *a, **k):
        self.model.zero_grad(*a, **k)


def _hf_model_path(cfg: dict, role: str) -> str:
    """按角色取 HF 模型路径：student→student_path；teacher→teacher_rl_path。"""
    key = {"student": "student_path", "teacher": "teacher_rl_path"}.get(role)
    if key is None:
        raise ModelError(f"model_kind='hf' 不支持 role={role!r}（student|teacher）")
    path = cfg.get(key)
    if not path:
        raise ModelError(f"model_kind='hf' 但未配置 {key}（role={role}）")
    return path


def build_model(cfg: dict, device: str = "cpu", role: str = "student"):
    """按 model_kind 构建模型。默认 toy；hf 走 HFCausalLM 骨架；未知/未实现 kind 抛 ModelError。"""
    kind = cfg.get("model_kind", "toy")
    if kind == "toy":
        return CausalToyLM(
            vocab=cfg["vocab_size"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"]).to(device)
    if kind == "hf":
        # ⚠️ 骨架：本地无法 GPU 实测；需真实模型 + GPU 验证。
        # 长序列 × 大隐层训练前向必须 flash 注意力（否则 SDPA 物化注意力矩阵 OOM）；
        # CPU 单测不传（保持 from_pretrained 断言与无 flash 环境兼容）。
        _attn = "flash_attention_2" if str(device).startswith("cuda") else None
        return HFCausalLM(_hf_model_path(cfg, role), device,
                          dtype=cfg.get("dtype", "auto"),
                          attn_implementation=_attn)
    if kind in ("megatron", "vllm"):
        raise ModelError(
            f"model_kind={kind!r} 的真实模型集成尚未实现（由 async-opd/vLLM 承担）。"
            f"当前用 model_kind='toy'/'hf'。role={role}")
    raise ModelError(f"未知 model_kind={kind!r}（支持 toy/hf；megatron/vllm 待实现）role={role}")


__all__ = ["build_model", "HFCausalLM"]

"""L3 · vLLM rollout 引擎（TP=2，PagedAttention + 连续批处理 + FP8）。

把 toy 的 response_dists(model, prompts, responses) -> (B,T,V) 接口用 vLLM 包一层，
作为 AsyncOPD rollout 阶段的 drop-in 替换：

  - 真实推理吞吐由 vLLM 承担（取代 CausalToyLM 的朴素前向）；
  - 高吞吐 PagedAttention + 连续批处理让 rollout 不再是瓶颈（方案 L3）；
  - tensor_parallel_size = tp_size（2×PRO6000 上 NVLink 桥接，TP=2 FP8 推理）。

接口对齐：VLLMRolloutEngine.response_dists(prompts, responses) 与 model.response_dists
签名/语义一致（返回 log-softmax 的 (B,T,V)），故调度器与损失内核（π_old 加权 PG +
PPO clip + k3 KL）**一行不动**。

──────────────────────────────────────────────────────────────────────────
与算法内核的兼容性（重要，务必读）
──────────────────────────────────────────────────────────────────────────
本代码库的损失是「分布级」的：pg_loss 在全词表上计算重要性比
ratio(v) = π_cur(v)/π_old(v)，需要完整的 π_old 分布。

  - 小词表 / demo（vocab ≤ full_logprobs_cap）：本引擎请求 prompt_logprobs = vocab，
    精确重建完整 (B,T,V) 分布 → 与 toy response_dists 数值一致，内核安全。
  - 真实词表（V=128k）：请求全部 logprob 不现实。生产走 verl/slime 的 token-level
    PPO，它们直接把 vLLM 的「逐 token logπ_old」喂给 token 级损失。本引擎同时提供
    response_logprobs() -> (B,T) 供该路径使用；分布级内核只建议在小词表下与 vLLM 精确
    对齐（这也是研究内核与工业框架的自然分界）。

权重同步（colocated L6 / AsyncOPD 陈旧度）：learner 每步更新后把权重推入 vLLM。
不同 vLLM 版本 API 不同（>=0.6 用 LLM.update_weights，旧版用 model_executor.model
.load_weights），update_weights / update_weights_from_flat 已做适配并尝试，失败抛清晰错误。
"""

from __future__ import annotations

import torch

try:                                                    # pragma: no cover
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except Exception:                                       # pragma: no cover
    LLM = SamplingParams = None
    _VLLM_AVAILABLE = False


# 支撑外 logp：用一个极大负值近似 log 0（避免 -inf 直接参与比率/梯度数值）。
# 注意：不能太负——pg_loss 里会做 (s_cur - s_old).exp()，-1e4 会算出 exp(≈1e4)=inf，
# 再乘稀疏模式下为 0 的 delta → inf×0=nan。-30 下 exp(30)≈1e13，bf16 安全，恢复「π_old=0 处贡献为 0」。
_LOG_ZERO = -30.0


def vllm_available() -> bool:
    return _VLLM_AVAILABLE


class VLLMRolloutEngine:
    """vLLM 包成的 rollout 引擎，接口对齐 model.response_dists。

    参数
    ----
    model          : vLLM 模型路径或 HF 模型名（如 "Qwen/Qwen2.5-7B"）。
    tp_size        : tensor parallel 度（2×PRO6000 用 2，走 NVLink 桥）。
    dtype          : "auto" | "bf16" | "fp8" 等（Blackwell 可 "fp8" 进一步提速）。
    vocab_size     : 词表大小；不传则尝试从 engine 推断。
    full_logprobs_cap : 触发「精确完整分布」重建的词表上限；超过则用 top-K 截断。
    device         : 张量落回的设备。
    """

    def __init__(self, model, *, tp_size: int = 1, dtype: str = "auto",
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 2048,
                 vocab_size: int | None = None, full_logprobs_cap: int = 4096,
                 device: str = "cuda:0", **engine_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM 未安装（统一 GPU 环境应含 vllm）。L3 rollout 需要 vLLM 引擎。")
        self.tp_size = int(tp_size)
        self.dtype = dtype
        self.device = device
        self.full_cap = int(full_logprobs_cap)
        # 精确分布路径会请求 prompt_logprobs = min(vocab, full_cap)（最多 full_cap）。
        # vLLM 引擎默认 max_logprobs=20，超过会被 SamplingParams 校验拒绝 → 必须抬高上限，
        # 否则小词表精确重建路径在运行期才报错。
        engine_kwargs.setdefault("max_logprobs", max(self.full_cap, 20))
        self.llm = LLM(
            model=model,
            tensor_parallel_size=self.tp_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **engine_kwargs,
        )
        if vocab_size is not None:
            self.vocab_size = int(vocab_size)
        else:
            try:
                self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()
            except Exception as e:   # pragma: no cover
                raise RuntimeError(
                    "无法从 vLLM engine 推断词表大小，请显式传入 vocab_size=。") from e

    # --------------------------- 权重同步 ---------------------------
    def update_weights(self, state_dict: dict) -> bool:
        """把 learner 的新权重推入 vLLM（取代线程版 load_state_dict）。

        版本差异：
          - vLLM >= 0.6 : LLM.update_weights(weights)
          - 旧版        : llm.llm_engine.model_executor.model.load_weights(weights)
        离线 demo 不调用；上线按实际版本调整。返回是否成功。
        """
        try:
            if hasattr(self.llm, "update_weights"):
                self.llm.update_weights(state_dict)
                return True
            me = self.llm.llm_engine.model_executor.model
            me.load_weights(state_dict.items())
            return True
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "vLLM 权重同步失败：请按你的 vLLM 版本调整 update_weights "
                f"（>=0.6 用 LLM.update_weights）。底层错误：{e}")

    def update_weights_from_flat(self, tensors: list) -> bool:
        """按引擎参数顺序用拉取的扁平张量重建 state_dict 并推入 vLLM。

        适用于 NCCL 权重广播后（L5）的 Ray vLLM worker：广播携带的是与 learner
        参数同序的扁平张量。⚠️ 假设 learner 与 vLLM 引擎**同名同构**（verl/slime 的
        actor-rollout 即是如此）；异构命名需在此做映射（生产另接）。
        """
        try:
            named = list(self._engine_named_parameters())
        except Exception:
            named = []
        if len(named) != len(tensors):
            raise RuntimeError(
                f"vLLM 引擎参数数({len(named)})与广播张量数({len(tensors)})不一致；"
                "异构命名需做映射（verl 风格同构则可）。")
        sd = {name: t.reshape(shp) for (name, shp), t in zip(
            [(n, tuple(p.shape)) for n, p in named], tensors)}
        return self.update_weights(sd)

    def _engine_named_parameters(self):
        return self.llm.llm_engine.model_executor.model.named_parameters()

    # --------------------------- 核心：response_dists 接口对齐 ---------------------------
    def _prompt_seq(self, prompts, responses):
        """(B,P),(B,T) -> list[list[int]]：一次 cat + cpu + tolist，避免逐样本同步。"""
        full = torch.cat([prompts, responses], dim=1).detach().cpu()
        return full.tolist()

    def response_dists_topk(self, prompts: torch.Tensor,
                            responses: torch.Tensor,
                            K: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,P),(B,T) -> (ids, logps)，形状各 (B,T,K)。GPU 路径主接口（稀疏）。

        把 vLLM 的 prompt_logprobs 稀疏 dict 直接拍平成 (B,T,K) 的 (ids, logps)，
        不重建 dense (B,T,V)。返回的 ids 恰好对接 cache 的 searchsorted 支撑匹配。
        K：可选，限制返回 top-K（每槽按 logprob 降序，第 0 位最高概率）；None → full_cap。
        IMP-2/P0：vLLM logprobs dict 迭代顺序不保证按 logprob 排序，这里显式降序排序，
        保证按 K 截断 = 精确 top-K（否则 searchsorted 匹配/支撑语义错乱）。
        """
        prompts = prompts.detach()
        responses = responses.detach()
        B, P = prompts.shape
        T = responses.size(1)
        V = self.vocab_size
        _cap = V if V <= self.full_cap else self.full_cap
        k = min(int(K), _cap) if K is not None else _cap
        k = max(1, k)
        sampling = SamplingParams(temperature=0.0, prompt_logprobs=k, logprobs=0)
        seqs = self._prompt_seq(prompts, responses)
        outs = self.llm.generate(prompt_token_ids=seqs, sampling_params=sampling)

        # M3：logp 填充用 _LOG_ZERO（≈log 0）而非 0.0——0.0 是合法高概率，稀疏支撑匹配
        # 会把未填充槽位当成「token id=0 处 logp=0.0」从而污染 searchsorted 匹配 / 伪高置信。
        # 正常操作（V≤cap 枚全或 V>cap 取满 top-cap）len(items)==k 无残留；残留只出现在
        # 空 dict / 部分返回的异常路径，此时 -30 使 padding 槽位数值上≈0 贡献（M1 同哲学）。
        ids = torch.zeros((B, T, k), dtype=torch.long)
        lps = torch.full((B, T, k), _LOG_ZERO, dtype=torch.float32)
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                d = plp[P + t]
                if not d:
                    continue
                items = list(d.items())
                # 按 logprob 降序（vLLM dict 迭代顺序不保证有序）
                items.sort(key=lambda kv: kv[1].logprob, reverse=True)
                ids[b, t, :len(items)] = torch.tensor([int(tid) for tid, _ in items],
                                                      dtype=torch.long)
                lps[b, t, :len(items)] = torch.tensor([v.logprob for _, v in items],
                                                      dtype=torch.float32)
        return ids.to(self.device), lps.to(self.device)

    def response_dists(self, prompts: torch.Tensor,
                       responses: torch.Tensor) -> torch.Tensor:
        """(B,P),(B,T) -> (B,T,V) log-softmax，对齐 model.response_dists。

        小词表 / 数值对照用：展平 + 一次 scatter 重建 dense (B,T,V)，去掉逐元素
        setitem（外推 14×）。真实词表请用 response_dists_topk（稀疏，不建 (B,T,V)）。
        """
        prompts = prompts.detach()
        responses = responses.detach()
        B, P = prompts.shape
        T = responses.size(1)
        V = self.vocab_size
        k = V if V <= self.full_cap else self.full_cap
        sampling = SamplingParams(temperature=0.0, prompt_logprobs=k, logprobs=0)
        seqs = self._prompt_seq(prompts, responses)
        outs = self.llm.generate(prompt_token_ids=seqs, sampling_params=sampling)

        out = torch.full((B * T * V,), _LOG_ZERO, dtype=torch.float32)
        pos_l: list[int] = []
        val_l: list[float] = []
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                d = plp[P + t]
                if not d:
                    continue
                row = (b * T + t) * V
                pos_l.extend([row + int(tid) for tid in d.keys()])
                val_l.extend([v.logprob for v in d.values()])
        if pos_l:
            idx = torch.tensor(pos_l, dtype=torch.long)
            out[idx] = torch.tensor(val_l, dtype=torch.float32)
        return out.view(B, T, V).to(self.device)

    # --------------------------- 生产路径：逐 token logπ_old（token-level PPO）---------------------------
    def response_logprobs(self, prompts: torch.Tensor,
                          responses: torch.Tensor) -> torch.Tensor:
        """(B,P),(B,T) -> (B,T)：response 各 token 的 logπ_old(a_t)。

        生产用：verl/slime 把此直接喂 token 级 PPO（ratio_t = π_cur(a_t)/π_old(a_t)），
        无需重建全词表分布。与 response_dists 共享一次 vLLM 前向（这里单独给一个轻量版）。
        """
        prompts = prompts.detach().cpu()
        responses = responses.detach().cpu()
        B = prompts.size(0)
        P = prompts.size(1)
        T = responses.size(1)
        sampling = SamplingParams(temperature=0.0, prompt_logprobs=1, logprobs=0)
        seqs = [torch.cat([prompts[b], responses[b]]).tolist() for b in range(B)]
        outs = self.llm.generate(prompt_token_ids=seqs, sampling_params=sampling)
        out = torch.zeros((B, T), dtype=torch.float32)
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                d = plp[P + t]
                if not d:
                    continue
                # dict 里只有 1 项（prompt_logprobs=1）：取该 token 自身 logprob
                tok_id, lp = next(iter(d.items()))
                out[b, t] = float(lp.logprob)
        return out.to(self.device)

    # --------------------------- 自回归采样（可选，替代 generate_batch 做真实 rollout）---------------------------
    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, max_new: int = 8,
                 temperature: float = 1.0) -> torch.Tensor:
        """(B,P) -> (B,max_new)：用 vLLM 采样响应（真实 rollout，非离线固定）。

        离线固定 rollout（Lightning 设定）下本方法不调用；在线 rollout 时用。
        """
        prompts = prompts.detach().cpu()
        B = prompts.size(0)
        sampling = SamplingParams(
            temperature=max(temperature, 1e-6), top_p=0.9, max_tokens=max_new)
        seqs = [prompts[b].tolist() for b in range(B)]
        outs = self.llm.generate(prompt_token_ids=seqs, sampling_params=sampling)
        res = torch.zeros((B, max_new), dtype=torch.long)
        for b, o in enumerate(outs):
            toks = o.outputs[0].token_ids[:max_new]
            res[b, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        return res.to(self.device)

    # --------------------------- Stage 2 短 rollout（带 status）---------------------------
    @torch.no_grad()
    def generate_with_status(self, prompts: torch.Tensor, max_new: int,
                             eos_token_id=None, temperature: float = 1.0,
                             pad_id: int = 0, loop_detection: bool = True,
                             loop_periods=(2, 3, 4),
                             repetition_penalty: float = 1.0,
                             loop_min_len: int = 8) -> dict:
        """Stage 2：短预算 rollout（vLLM）——SamplingParams 定 max_new/eos，经
        parse_vllm_outputs 得 status，responses 变长右 pad 到 max_new。

        返回与 toy generate_with_status 同构的 dict（responses/statuses/lengths/
        eos_pos/looped），使 run_refresh_phase 引擎无关。
        """
        prompts = prompts.detach().cpu()
        B = prompts.size(0)
        sampling = SamplingParams(
            temperature=max(temperature, 1e-6), top_p=0.9, max_tokens=max_new,
            eos_token_id=eos_token_id,
            repetition_penalty=max(float(repetition_penalty), 1.0))
        seqs = [prompts[b].tolist() for b in range(B)]
        outs = self.llm.generate(prompt_token_ids=seqs, sampling_params=sampling)
        parsed = parse_vllm_outputs(outs, max_new, eos_token_id,
                                    loop_detection, loop_periods, loop_min_len)
        # 组装 responses：(B,max_new) 按 lengths 写入、pad 填充
        res = torch.full((B, max_new), pad_id, dtype=torch.long)
        for b, o in enumerate(outs):
            toks = o.outputs[0].token_ids[:max_new]
            n = parsed["lengths"][b]
            res[b, :n] = torch.tensor(toks[:n], dtype=torch.long)
        parsed["responses"] = res.to(self.device)
        return parsed


def parse_vllm_outputs(outs, max_new: int, eos_token_id=None,
                       loop_detection: bool = True,
                       loop_periods=(2, 3, 4),
                       loop_min_len: int = 8) -> dict:
    """Stage 2：把 vLLM RequestOutput 列表解析为同构 status dict（纯函数，CPU 可单测）。

    复用 budget_eval.generate_budget 的逐位 EOS 判定手法（eos in new + new.index(eos)）。
    vLLM outputs[i].token_ids 是【生成部分】（不含 prompt）；finish_reason 仅作参考，
    状态以 token 内容为准（eos 优先，其次 loop，其次 budget_stop / invalid）。

    返回：{"responses": None, "statuses": [...], "lengths": [...], "eos_pos": [...],
           "looped": [...]}（responses 由 generate_with_status 组装）。
    """
    from .model import detect_loop
    statuses: list[str] = []
    lengths: list[int] = []
    eos_pos: list[int | None] = []
    looped: list[bool] = []
    for o in outs:
        new = o.outputs[0].token_ids          # 生成部分（不含 prompt）
        if eos_token_id is not None and eos_token_id in new:
            ep = new.index(eos_token_id)
            status, length = "eos", ep + 1          # 含 eos
        else:
            ep, status, length = None, "budget_stop", len(new)
        loop = loop_detection and detect_loop(
            torch.tensor(new[:max(1, length)]), loop_periods, min_len=loop_min_len)
        if loop:
            status = "loop"
        elif length == 0:
            status = "empty"
        statuses.append(status); lengths.append(length)
        eos_pos.append(ep); looped.append(loop)
    return {"responses": None, "statuses": statuses, "lengths": lengths,
            "eos_pos": eos_pos, "looped": looped}

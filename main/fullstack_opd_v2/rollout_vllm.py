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


# C3：进程级活跃引擎注册表 + atexit 兜底——实验失败/kill 主进程时强制关闭
# vLLM V1 的 EngineCore 子进程（实测残留 ~40GB/卡，会卡死下一次引擎构建）。
_ACTIVE_ENGINES: list["VLLMRolloutEngine"] = []
_atexit_registered = False


def _shutdown_all_engines() -> None:
    for eng in list(_ACTIVE_ENGINES):
        try:
            eng.shutdown()
        except Exception:          # noqa: BLE001 —— 清理路径绝不抛
            pass
    _ACTIVE_ENGINES.clear()


import atexit
atexit.register(_shutdown_all_engines)


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
                 device: str = "cuda:0", weight_sync_mode: str = "auto",
                 **engine_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM 未安装（统一 GPU 环境应含 vllm）。L3 rollout 需要 vLLM 引擎。")
        self.tp_size = int(tp_size)
        self._wt_sync_mode = weight_sync_mode  # auto=0.16 NCCL 同步；off=逃生舱
        self.dtype = dtype
        self.device = device
        self.full_cap = int(full_logprobs_cap)
        # 精确分布路径会请求 prompt_logprobs = min(vocab, full_cap)（最多 full_cap）。
        # vLLM 引擎默认 max_logprobs=20，超过会被 SamplingParams 校验拒绝 → 必须抬高上限，
        # 否则小词表精确重建路径在运行期才报错。
        engine_kwargs.setdefault("max_logprobs", max(self.full_cap, 20))
        # vLLM>=0.16 NCCL 权重同步：引擎【启动时】就必须带 weight_transfer_config，
        # 否则 worker 的 init_weight_transfer_engine 直接拒绝（"Weight transfer not
        # configured"）→ trainer 的 TCPStore 等不到 rank1 → 死锁（2026-08-17 实测）。
        if str(weight_sync_mode).lower() != "off":
            try:
                from vllm.config.weight_transfer import WeightTransferConfig
                engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="nccl")
            except Exception as e:   # pragma: no cover —— 旧版 vLLM 无此配置
                import logging
                logging.getLogger(__name__).warning(
                    f"weight_transfer_config 不可用（旧版 vLLM？）：{e}")
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
        # C5：vocab 交叉验证——错误 vocab 在构造期必须立即失败（否则训练线程深处
        # out[idx] 才 IndexError，且落在 worker 线程 → 0% 空等）。fake/无 engine 时跳过。
        try:
            _real_vocab = int(self.llm.llm_engine.model_config.get_vocab_size())
        except Exception:          # noqa: BLE001 —— 单测 fake LLM 无该属性
            _real_vocab = None
        if _real_vocab is not None and self.vocab_size != _real_vocab:
            raise ValueError(
                f"vocab_size 交叉校验失败：传入 {self.vocab_size} ≠ vLLM 引擎真实词表 "
                f"{_real_vocab}。HF 路径应传 student.vocab（model.config.vocab_size），"
                "toy 的 cfg['vocab_size']=64 会在此处被拦截。")
        # C3：注册进进程级清理表
        _ACTIVE_ENGINES.append(self)

    # --------------------------- 清理 ---------------------------
    def shutdown(self) -> None:
        """关闭 vLLM 引擎，避免 EngineCore 子进程残留（实测 ~40GB/卡）。
        同时销毁 trainer 侧 NCCL weight-transfer 组。

        vLLM V1 的 EngineCore 是独立进程；主进程 kill 不会带走它。
        多版本适配：先试 LLM.shutdown()（>=0.16），再尝试
        engine 层 API；都失败时通知注册表移除即可
        （atexit 盘点）。可重入，无双次关闭副作用。
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        try:
            if self.llm is not None and hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
                return
        except Exception as e:     # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(f"vLLM shutdown 第一路失败：{e}")
        try:
            me = self.llm.llm_engine
            if hasattr(me, "shutdown"):
                me.shutdown()
                return
        except Exception as e:     # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(f"vLLM shutdown 第二路失败：{e}")
        # 销毁 trainer 侧 NCCL weight-transfer 组
        g = getattr(self, "_wt_group", None)
        if g is not None:
            try:
                del g
            except Exception:          # noqa: BLE001
                pass
            self._wt_group = None

    # --------------------------- generate 版本兼容 ---------------------------
    def _generate(self, seqs, sampling):
        """vLLM 版本兼容的批量生成入口。

        - vLLM 0.6~0.9x : LLM.generate(prompt_token_ids=seqs, ...)
        - vLLM >= 0.1x  : 移除该 kwarg，改传 TokensPrompt dict
          (prompts=[{"prompt_token_ids": s}, ...])。
        首次调用探测一次后缓存风格，后续零开销。
        """
        style = getattr(self, "_gen_style", None)
        if style == "kwarg":
            return self.llm.generate(prompt_token_ids=seqs,
                                     sampling_params=sampling)
        if style == "tokens_prompt":
            return self.llm.generate(
                prompts=[{"prompt_token_ids": list(s_)} for s_ in seqs],
                sampling_params=sampling)
        try:
            outs = self.llm.generate(prompt_token_ids=seqs,
                                     sampling_params=sampling)
            self._gen_style = "kwarg"
            return outs
        except TypeError:
            outs = self.llm.generate(
                prompts=[{"prompt_token_ids": list(s_)} for s_ in seqs],
                sampling_params=sampling)
            self._gen_style = "tokens_prompt"
            return outs

    # --------------------------- 权重同步 ---------------------------
    def update_weights(self, state_dict: dict) -> bool:
        """把 learner 的新权重推入 vLLM（取代线程版 load_state_dict）。

        版本适配（按顺序探测）：
          - vLLM 0.6~0.1x : LLM.update_weights(weights) —— 直接推 state_dict；
          - vLLM >= 0.16  : LLM.update_weights(request: WeightTransferUpdateRequest)
            改为 NCCL WeightTransferEngine 协议（init_weight_transfer_engine +
            update），且需 HF→vLLM 合并层名称映射（qkv_proj/gate_up_proj）——
            未接入前本方法警告一次并返回 False（rollout 用引擎现有权重，
            即初始策略；短 pilot 影响可忽略，正式训练前必须接入）；
          - 旧版 v0      : llm.llm_engine.model_executor.model.load_weights。
        返回 True=同步成功；False=版本不支持（已警告）。
        """
        if hasattr(self.llm, "update_weights"):
            import inspect
            try:
                _params = list(inspect.signature(
                    self.llm.update_weights).parameters.values())
            except (TypeError, ValueError):  # pragma: no cover
                _params = []
            if _params and _params[0].name == "request":
                # vLLM >= 0.16：request-style WeightTransferEngine API。
                # 逃生舱：rollout_weight_sync=off 时回落旧行为（warn 一次 + 返回 False）。
                if str(getattr(self, "_wt_sync_mode", "auto")).lower() == "off":
                    if not getattr(self, "_wt_warned", False):
                        self._wt_warned = True
                        import logging
                        logging.getLogger(__name__).warning(
                            "vLLM>=0.16 权重同步被 rollout_weight_sync=off 关闭；"
                            "rollout 将用引擎现有权重（初始策略）。正式训练请开启 NCCL 同步。")
                    return False
                return self._weight_transfer_update_16(state_dict)
            try:
                self.llm.update_weights(state_dict)
                return True
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "vLLM 权重同步失败：请按你的 vLLM 版本调整 update_weights "
                    f"（>=0.6 用 LLM.update_weights）。底层错误：{e}")
        try:
            me = self.llm.llm_engine.model_executor.model
            me.load_weights(state_dict.items())
            return True
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "vLLM 权重同步失败：请按你的 vLLM 版本调整 update_weights "
                f"（>=0.6 用 LLM.update_weights）。底层错误：{e}")

    # --------------------------- vLLM>=0.16 NCCL 权重同步 ---------------------------
    def _weight_transfer_init_16(self) -> None:
        """懒初始化 NCCL weight-transfer 组（trainer=rank0，worker 从 rank_offset 起）。

        vLLM 0.16 的 WeightTransferEngine：trainer 建 NCCL 组（master_addr/port 共享），
        worker 侧 init_weight_transfer_engine 加入。world_size = 1(trainer) + tp_size。
        """
        if getattr(self, "_wt_group", None) is not None:
            return
        try:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"vLLM NCCL WeightTransferEngine 不可用：{e}")
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._wt_init_info = {
            "master_address": "127.0.0.1",
            "master_port": int(port),
            "rank_offset": 1,                     # trainer=0，worker 从 1 起
            "world_size": 1 + int(self.tp_size),
        }
        # NCCL 组建立是【阻塞集合操作】：trainer（rank0）与 worker（rank1）必须【并发】
        # 调 init，否则一方等另一方加入 → 死锁（2026-08-17 实测：base 训练 20 步完成后
        # 卡死在 update_weights 初始化）。故 worker 侧 init 放后台线程，主线程 trainer_init
        # 阻塞握手；两者在同一 master_port/store 上完成 NCCL 组建立。
        _init_err: list[Exception] = []

        def _worker_init():
            try:
                self.llm.init_weight_transfer_engine({"init_info": self._wt_init_info})
            except Exception as e:  # noqa: BLE001
                _init_err.append(e)

        import threading
        t = threading.Thread(target=_worker_init, daemon=True)
        t.start()
        # NCCL 通信组禁止两个 rank 同卡（"Duplicate GPU detected"，2026-08-17 实测）。
        # trainer（rank0）必须在【训练卡】建组，vLLM worker（rank1）在 rollout 卡——
        # 布局必须是交叉分卡（CUDA_VISIBLE_DEVICES 重排）。显式把当前设备切到训练卡。
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(self.device)
        # trainer 侧 NCCL 组（rank 0，用当前 CUDA 设备 = 训练卡）
        self._wt_group = NCCLWeightTransferEngine.trainer_init(self._wt_init_info)
        t.join(timeout=120)
        if _init_err:
            raise RuntimeError(f"vLLM NCCL 权重同步初始化失败（worker 侧）：{_init_err[0]}")

    def _weight_transfer_update_16(self, state_dict: dict) -> bool:
        """0.16 NCCL 广播：update_info + 后台线程发权重。

        worker 端 update_weights → receive_weights 阻塞等 NCCL 广播；trainer 侧
        必须【并发】广播（同步 update_weights 会等 worker 返回，而 worker 在等广播）。
        故：后台线程 trainer_send_weights（顺序=state_dict.items()，与 update_info.names
        同序），主线程同步调用 llm.update_weights(update_info)。is_checkpoint_format=True
        → worker 用 model.load_weights（自动处理 HF→vLLM 合并层 qkv/gate_up 映射）。
        """
        import threading
        self._weight_transfer_init_16()
        update_info = _build_nccl_update_info(state_dict)
        # 防御：广播要求 tensor 在 NCCL 组设备（训练卡）上——调用方传 CPU 态会
        # AssertionError 且被吞 → collective_rpc 永久挂（2026-08-17 实测）。
        sd_dev = {k: (v.to(self.device) if v.device != torch.device(self.device)
                      else v) for k, v in state_dict.items()}
        update_info = _build_nccl_update_info(sd_dev)
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine)
        err: list[Exception] = []

        def _send():
            try:
                NCCLWeightTransferEngine.trainer_send_weights(
                    iter(sd_dev.items()), self._wt_group,
                    stream=torch.cuda.current_stream())
            except Exception as e:  # noqa: BLE001 —— 广播失败记入并让主线程抛
                err.append(e)

        t = threading.Thread(target=_send, daemon=True)
        t.start()
        try:
            # 用 engine 层 collective_rpc + 显式超时：广播失败时不无限挂，超时即抛。
            self.llm.llm_engine.collective_rpc(
                "update_weights", timeout=float(getattr(self, "_wt_timeout", 120.0)),
                kwargs={"update_info": update_info})
        finally:
            t.join(timeout=float(getattr(self, "_wt_timeout", 120.0)) + 30)
        if err:
            raise RuntimeError(f"vLLM NCCL 权重广播失败：{err[0]}")
        return True

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
        outs = self._generate(seqs, sampling)

        # M3：logp 填充用 _LOG_ZERO（≈log 0）而非 0.0——0.0 是合法高概率，稀疏支撑匹配
        # 会把未填充槽位当成「token id=0 处 logp=0.0」从而污染 searchsorted 匹配 / 伪高置信。
        # 正常操作（V≤cap 枚全或 V>cap 取满 top-cap）len(items)==k 无残留；残留只出现在
        # 空 dict / 部分返回的异常路径，此时 -30 使 padding 槽位数值上≈0 贡献（M1 同哲学）。
        ids = torch.zeros((B, T, k), dtype=torch.long)
        lps = torch.full((B, T, k), _LOG_ZERO, dtype=torch.float32)
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                if P + t >= len(plp):
                    raise RuntimeError(
                        "vLLM prompt_logprobs 长度不足（chunked-prefill/首 token None 版本差异）："
                        f"P+t={P + t} >= len(plp)={len(plp)}；请升级/降级对齐 vLLM 版本或降 context。")
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
        # I2：真实词表禁用稠密重建路径（(B,T,V) 在 V=151936、T=4096 下 ~2.5GB/批，
        # vLLM 逐 token prompt_logprobs 重建慢 ~50s/样本；应走 response_dists_topk）。
        if V > self.full_cap:
            raise RuntimeError(
                f"response_dists 稠密重建路径仅支持小词表 "
                f"(V={V} > full_logprobs_cap={self.full_cap})；请改用 "
                "response_dists_topk（稀疏 (B,T,K)）。这是架构约束，"
                "不是可以忽略的性能警告。")
        k = V if V <= self.full_cap else self.full_cap
        sampling = SamplingParams(temperature=0.0, prompt_logprobs=k, logprobs=0)
        seqs = self._prompt_seq(prompts, responses)
        outs = self._generate(seqs, sampling)

        out = torch.full((B * T * V,), _LOG_ZERO, dtype=torch.float32)
        pos_l: list[int] = []
        val_l: list[float] = []
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                if P + t >= len(plp):
                    raise RuntimeError(
                        "vLLM prompt_logprobs 长度不足："
                        f"P+t={P + t} >= len(plp)={len(plp)}；请对齐 vLLM 版本。")
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
        outs = self._generate(seqs, sampling)
        out = torch.zeros((B, T), dtype=torch.float32)
        for b, o in enumerate(outs):
            plp = o.prompt_logprobs
            for t in range(T):
                if P + t >= len(plp):
                    raise RuntimeError(
                        "vLLM prompt_logprobs 长度不足："
                        f"P+t={P + t} >= len(plp)={len(plp)}；请对齐 vLLM 版本。")
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
        outs = self._generate(seqs, sampling)
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
                             loop_min_len: int = 8,
                             top_p: float = 1.0) -> dict:
        """Stage 2：短预算 rollout（vLLM）——SamplingParams 定 max_new/eos，经
        parse_vllm_outputs 得 status，responses 变长右 pad 到 max_new。

        返回与 toy generate_with_status 同构的 dict（responses/statuses/lengths/
        eos_pos/looped），使 run_refresh_phase 引擎无关。
        """
        prompts = prompts.detach().cpu()
        B = prompts.size(0)
        # vLLM>=0.8 移除 SamplingParams.eos_token_id 字段（0.16 msgspec 校验拒绝）。
        # 改用 stop_token_ids=[eos]：vLLM 在即将生成 eos 时提前停（stop token 不入输出），
        # parse_vllm_outputs 按 finish_reason="stop" 恢复 toy 语义（eos 位置=len、length=len+1
        # 并在 generate_with_status 补写 eos token）。Qwen3+短预算实际 eos≈0，多为 budget_stop。
        # top_p 默认 1.0（纯温度采样，与 HF generate_with_status 一致）——此前硬编码
        # 0.9 的 nucleus 截断更贪婪，Qwen3+短预算下 loop 率显著升高（2026-08-17 实测
        # vLLM 6/8 vs HF 0/8）。可配置覆盖。
        _sp_kw = dict(temperature=max(temperature, 1e-6),
                      top_p=float(top_p),
                      max_tokens=max_new,
                      repetition_penalty=max(float(repetition_penalty), 1.0))
        if eos_token_id is not None:
            _sp_kw["stop_token_ids"] = [int(eos_token_id)]
        sampling = SamplingParams(**_sp_kw)
        seqs = [prompts[b].tolist() for b in range(B)]
        outs = self._generate(seqs, sampling)
        parsed = parse_vllm_outputs(outs, max_new, eos_token_id,
                                    loop_detection, loop_periods, loop_min_len)
        # 组装 responses：(B,max_new) 按 lengths 写入、pad 填充
        res = torch.full((B, max_new), pad_id, dtype=torch.long)
        for b, o in enumerate(outs):
            toks = o.outputs[0].token_ids[:max_new]
            n = parsed["lengths"][b]
            res[b, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            # stop_token_ids 路径（eos 不入输出）：在 eos 位置补写 eos token，还原
            # toy 语义（length=eos_pos+1 含 eos）。
            if parsed["statuses"][b] == "eos" and parsed["eos_pos"][b] == len(toks):
                res[b, len(toks)] = int(eos_token_id if eos_token_id is not None else pad_id)
        parsed["responses"] = res.to(self.device)
        return parsed


def _build_nccl_update_info(state_dict: dict) -> dict:
    """从 HF state_dict 构建 vLLM>=0.16 NCCL update_info（纯函数，CPU 可单测）。

    names/dtype_names/shapes 与 trainer_send_weights 的迭代顺序一致（state_dict.items()）。
    is_checkpoint_format=True → worker 用 model.load_weights（HF→vLLM 合并层映射自动）。
    注意：不带 "backend" 键——worker 的 NCCLWeightTransferUpdateInfo 只接受
    names/dtype_names/shapes/packed/is_checkpoint_format；backend 由引擎启动时的
    WeightTransferConfig 决定（2026-08-17 实测：带 backend 会 "unexpected keyword"）。
    """
    return {
        "names": list(state_dict.keys()),
        "dtype_names": [str(t.dtype).split(".")[-1] for t in state_dict.values()],
        "shapes": [list(t.shape) for t in state_dict.values()],
        "packed": False,
        "is_checkpoint_format": True,
    }


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
        fr = o.outputs[0].finish_reason
        if eos_token_id is not None and eos_token_id in new:
            ep = new.index(eos_token_id)
            status, length = "eos", ep + 1          # 含 eos
        elif (eos_token_id is not None and fr == "stop"):
            # vLLM>=0.8 stop_token_ids 路径：eos 被消费但【不入输出】；finish_reason=stop
            # 且我们只传了 eos 一个 stop token → 判 eos 停，位置=len(new)、length=len+1。
            ep, status, length = len(new), "eos", len(new) + 1
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

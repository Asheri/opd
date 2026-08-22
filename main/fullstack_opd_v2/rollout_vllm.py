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

import logging
import os
import threading

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


def _resolve_visible_device(device: str, current_env: str | None) -> str | None:
    """把逻辑 cuda:i 映射为 vLLM 子进程可见的 CUDA_VISIBLE_DEVICES。

    vLLM 1.x 的 LLM() 没有 device 参数，EngineCore 是 spawn 子进程，只认
    CUDA_VISIBLE_DEVICES（默认落在 cuda:0）。2026-08-18 GPU 实测：引擎与训练
    共卡 → 训练 (8,3072,151936) fp32 logits 41GB + 引擎 11.6GB = 95GB OOM，
    第二张卡全程空置。此处把 device 映射进子进程环境，引擎独立卡。

    - 无 CUDA_VISIBLE_DEVICES：直接返回设备号（"cuda:1" → "1"）。
    - 有重排（如 "1,0"）：按列表索引取物理号（cuda:1 → "0"）。
    - 非 cuda 设备返回 None（不注入，vLLM 自行选择）。
    """
    if not str(device).startswith("cuda"):
        return None
    idx = str(device).split(":")[-1] if ":" in str(device) else "0"
    if current_env:
        vis = [x.strip() for x in current_env.split(",") if x.strip()]
        if vis:
            try:
                return vis[int(idx) % len(vis)]
            except ValueError:
                return None
    return idx


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
                 learner_device: str | None = None,
                 **engine_kwargs):
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM 未安装（统一 GPU 环境应含 vllm）。L3 rollout 需要 vLLM 引擎。")
        self.tp_size = int(tp_size)
        self._wt_sync_mode = weight_sync_mode  # auto=0.16 NCCL 同步；off=逃生舱
        self._learner_device = learner_device   # trainer 侧 NCCL 组所在卡（训练卡）
        # WT poisoned 状态（2026-08-19）：NCCL init/update 一旦超时或失败即标记，后续
        # 禁止复用该 engine（daemon 线程无法安全杀死、communicator 状态不可信 →
        # fail-closed；正确恢复方式是进程退出 / 重建 engine）。
        self._wt_poisoned = False
        self._wt_failure_reason = None
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
        # 引擎核心是 spawn 子进程：把 device 映射为 CUDA_VISIBLE_DEVICES 注入，
        # 否则 vLLM 恒用默认 cuda:0 与训练卡冲突（2026-08-18 GPU 实测双卡 OOM）。
        _vis = _resolve_visible_device(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
        if _vis is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _vis
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

    # --------------------------- WT poisoned 状态 ---------------------------
    @property
    def weight_sync_poisoned(self) -> bool:
        """engine 是否已被标记为 weight-transfer poisoned（不可恢复，禁止复用）。"""
        return bool(getattr(self, "_wt_poisoned", False))

    @property
    def weight_sync_poison_reason(self) -> str | None:
        """poisoned 原因（原始异常 repr）。"""
        return getattr(self, "_wt_failure_reason", None)

    def _mark_weight_transfer_poisoned(self, reason: str) -> None:
        """把 engine 标记为 poisoned（不可恢复）：NCCL init/update 超时或失败后调用。

        - daemon 线程无法被 Python 安全杀死；超时后 NCCL communicator 状态不可信；
        - 一旦标记，engine 禁止复用，正确恢复方式是进程退出 / 重建 engine；
        - shutdown() 不得清除本状态（后续 update_weights 仍必须拒绝复用）。
        """
        self._wt_poisoned = True
        self._wt_failure_reason = str(reason)
        logging.getLogger(__name__).error(
            "[WT] engine poisoned: %s；该进程应中止；如需继续实验，请重建 vLLM engine。",
            self._wt_failure_reason)


    # --------------------------- 清理 ---------------------------
    def shutdown(self) -> None:
        """关闭 vLLM 引擎，避免 EngineCore 子进程残留（实测 ~40GB/卡）。
        同时销毁 trainer 侧 NCCL weight-transfer 组并从活跃注册表移除。

        vLLM V1 的 EngineCore 是独立进程；主进程 kill 不会带走它。
        多版本适配：先试 LLM.shutdown()（>=0.16），再尝试 engine 层 API。
        修复（2026-08-19）：此前 LLM.shutdown() 成功即 return，跳过 NCCL 组清理与
        注册表移除——串行多实验时泄漏 EngineCore / NCCL communicator / 端口。
        现在无论引擎关闭成功与否，都走到尾部清理。可重入，无双次关闭副作用。
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        _engine_shut = False
        try:
            if self.llm is not None and hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
                _engine_shut = True
        except Exception as e:     # noqa: BLE001
            logging.getLogger(__name__).warning(f"vLLM shutdown 第一路失败：{e}")
        if not _engine_shut:
            try:
                me = self.llm.llm_engine
                if hasattr(me, "shutdown"):
                    me.shutdown()
            except Exception as e:     # noqa: BLE001
                logging.getLogger(__name__).warning(f"vLLM shutdown 第二路失败：{e}")
        # 销毁 trainer 侧 NCCL weight-transfer 组（引擎关闭后不得残留 NCCL 组）
        g = getattr(self, "_wt_group", None)
        if g is not None:
            try:
                _destroy = getattr(g, "destroy", None) or getattr(g, "close", None)
                if callable(_destroy):
                    _destroy()
                del g
            except Exception:          # noqa: BLE001 —— 清理路径绝不抛
                pass
            self._wt_group = None
        # 从进程级活跃注册表移除（避免 atexit 二次 shutdown / 列表泄漏）
        if self in _ACTIVE_ENGINES:
            try:
                _ACTIVE_ENGINES.remove(self)
            except Exception:          # noqa: BLE001
                pass

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
        # poisoned 统一在最外层拦截（不受 rollout_weight_sync=off 影响）：一旦 NCCL
        # init/update 超时或失败，engine 的 communicator 状态不可信，禁止复用。
        if self.weight_sync_poisoned:
            raise RuntimeError(
                "vLLM NCCL weight transfer engine poisoned; "
                f"reason={getattr(self, '_wt_failure_reason', None)}")
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

        修复（2026-08-19）：trainer_init 与 worker_init 都是阻塞集合操作，任何一侧
        失败/不加入都会让另一侧永久挂起（日志停在 [WT] trainer_init 开始、GPU 0%）。
        现在 trainer_init 走 _run_with_timeout 主进程侧 fail-fast；worker_init join 后
        检查 is_alive（超时仍存活即抛，不再静默继续）。

        poisoned 语义（2026-08-19）：任何 init 失败/超时（trainer_init 超时、worker init
        线程超时仍存活、worker init 抛异常、NCCL init 异常）都会把 engine 标记为
        poisoned——daemon 线程无法被 Python 安全杀死，超时后 NCCL communicator 状态
        不可信；poisoned engine 禁止复用，正确恢复方式是进程退出 / 重建 engine。
        """
        if self.weight_sync_poisoned:
            raise RuntimeError(
                "vLLM NCCL weight transfer engine poisoned; "
                f"reason={getattr(self, '_wt_failure_reason', None)}")
        if getattr(self, "_wt_group", None) is not None:
            return
        try:
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
            _init_err: list[BaseException] = []

            def _worker_init():
                try:
                    self.llm.init_weight_transfer_engine({"init_info": self._wt_init_info})
                except BaseException as e:      # noqa: BLE001 —— 记录后主线程抛
                    _init_err.append(e)

            _wlog = logging.getLogger(__name__)
            t = threading.Thread(target=_worker_init, daemon=True, name="wt-worker-init")
            t.start()
            _wlog.info("[WT] worker_init 线程已启动（端口 %s, world_size=%s）",
                       port, 1 + int(self.tp_size))
            # NCCL 通信组禁止两个 rank 同卡（"Duplicate GPU detected"，2026-08-17 实测）。
            # trainer（rank0）必须在【训练卡】建组，vLLM worker（rank1）在 rollout 卡——
            # 布局必须是交叉分卡（CUDA_VISIBLE_DEVICES 重排）。显式把当前设备切到训练卡。
            # 2026-08-19 修复：此前误用 self.device（rollout 卡）建 trainer 侧 NCCL 组
            # → rank0/rank1 同卡 → NCCL_INVALID_USAGE(5)（NCCL_DEBUG 实锤 init.cc:737
            # "Duplicate GPU detected"）。NCCL 组禁止两个 rank 同卡，trainer 必须在该
            # 实验的【训练卡】建组（worker 在 rollout 卡）；未显式给定 learner_device 时
            # 回落 rollout 卡（旧调用兼容：单卡部署时两者同卡合法）。
            _trainer_dev = self._learner_device or self.device
            if str(_trainer_dev).startswith("cuda"):
                torch.cuda.set_device(_trainer_dev)
            _timeout = float(getattr(self, "_wt_timeout", 120.0))
            _wlog.info("[WT] trainer_init 开始（trainer_dev=%s, 端口 %s）", _trainer_dev, port)
            # ⚠️ 2026-08-22 GPU 实测（E2 交叉布局 train@GPU1+vLLM@GPU0）：NCCL trainer_init
            # 在 _run_with_timeout 的【新线程】里执行，而 torch.cuda.set_device 是【线程局部】
            # ——主线程设的 cuda:1 在新线程不生效，新线程用默认 cuda:0，导致 rank0(trainer) 与
            # rank1(worker) 都在 GPU0 → "Duplicate GPU detected / NCCL_INVALID_USAGE"。
            # E1（train@GPU0）恰好默认卡=训练卡所以从未暴露。修复：把 set_device 放线程内。
            def _trainer_init_on_device():
                if str(_trainer_dev).startswith("cuda"):
                    torch.cuda.set_device(_trainer_dev)
                return NCCLWeightTransferEngine.trainer_init(self._wt_init_info)

            # trainer_init 可能因 worker 未加入而永久阻塞 → 主进程侧限时 fail-fast
            self._wt_group = _run_with_timeout(
                _trainer_init_on_device,
                _timeout + 30.0, "trainer_init")
            _wlog.info("[WT] trainer_init 完成")
            t.join(timeout=_timeout + 30.0)
            if t.is_alive():
                raise RuntimeError(
                    "[WT] worker_init 线程超时仍存活——worker 未完成 init_weight_transfer_engine，"
                    "NCCL 组不完整；已 fail-fast。请检查 vLLM 侧日志（weight_transfer_config 是否"
                    "已配置、worker 是否启动）。")
            _wlog.info("[WT] worker_init join 返回（init_err=%d）", len(_init_err))
            if _init_err:
                raise RuntimeError(
                    f"vLLM NCCL 权重同步初始化失败（worker 侧）：{_init_err[0]}")
        except BaseException as e:      # noqa: BLE001 —— 含 KeyboardInterrupt/SystemExit
            self._mark_weight_transfer_poisoned(f"init: {e!r}")
            raise

    def _weight_transfer_update_16(self, state_dict: dict) -> bool:
        """0.16 NCCL 广播：update_info + 后台线程发权重。

        worker 端 update_weights → receive_weights 阻塞等 NCCL 广播；trainer 侧
        必须【并发】广播（同步 update_weights 会等 worker 返回，而 worker 在等广播）。
        故：后台线程 trainer_send_weights（顺序=state_dict.items()，与 update_info.names
        同序），主线程同步调用 llm.update_weights(update_info)。is_checkpoint_format=True
        → worker 用 model.load_weights（自动处理 HF→vLLM 合并层 qkv/gate_up 映射）。

        P0 修复（2026-08-18 C1 实测）：vLLM>=0.16 的 layerwise reload（initialize→
        load_weights→finalize）把层参数先移到 meta、缓存权重、再【异步】拷回原存储。
        变更后的【第一次】同步只部分生效（~3% logp 残留上一步权重，实测 0.036 均值）；
        第二次相同同步后才精确收敛（0.000000）。训练每步只同步一次 → rollout 走的是
        部分旧权重（off-policy）。解决：同权重双发（第二发强制收敛），~200ms/步
        （1.7B bf16 全量 NCCL），可接受。重度优化：TP=1 时改 param.copy_ 直拷可省。

        修复（2026-08-19）：
        - payload preflight（_prepare_weight_transfer_payload）：把空 dict / 非 Tensor /
          非法 dtype / 空形状 / device 错位 / 非连续等在 collective_rpc 之前拦截；
        - collective_rpc 主进程侧限时（_run_with_timeout）+ sender 线程 is_alive 检查，
          杜绝 [WT-update] 开始后 0% 空等；
        - sender 线程异常即时 error 日志（此前只 append，主线程卡住时不可见）。

        poisoned 语义（2026-08-19）：任何 update 失败（含 payload preflight / 第一发 /
        第二发 / collective_rpc 超时 / sender 线程异常或超时）都会把 engine 标记为
        poisoned——daemon 线程无法被 Python 安全杀死，超时后 NCCL communicator 状态
        不可信；poisoned engine 禁止复用，正确恢复方式是进程退出 / 重建 engine。
        """
        if self.weight_sync_poisoned:
            raise RuntimeError(
                "vLLM NCCL weight transfer engine poisoned; "
                f"reason={getattr(self, '_wt_failure_reason', None)}")
        try:
            _ulog = logging.getLogger(__name__)
            self._weight_transfer_init_16()
            _ulog.info("[WT-update] 开始（keys=%d）", len(state_dict))
            # 广播要求 tensor 在【NCCL 组设备】（训练卡 = self._wt_group.device）上：
            # - 传 CPU 态 → PyNcclCommunicator 的 broadcast 无 device assert，NCCL 用
            #   comm device 操作错误指针 → illegal memory access（异步）→ 后续挂起
            #   （2026-08-17 实测）；
            # - 传 rollout 卡张量 → 同样不匹配（2026-08-19 verify_weight_sync 实测：
            #   [WT-update] 开始后永久卡）。旧实现 to(self.device)（rollout 卡）是错的，
            #   run_s2_real 恰好因 current_device=训练卡 + 流巧合未暴露。
            # preflight 同时完成：非空/Tensor/dtype/空形状/device/contiguous 校验。
            sd_dev = _prepare_weight_transfer_payload(state_dict, self._wt_group.device)
            update_info = _build_nccl_update_info(sd_dev)
            self._weight_transfer_broadcast_round(sd_dev, update_info, "第一发")
            if getattr(self, "_wt_double_send", True):
                # P0 收敛修复：worker 端 layerwise reload 的异步拷回使「变更后首次同步」只部分
                # 生效（实测 ~3% logp 残留）。同权重再发一次强制收敛（实测第二次精确 0.000000）。
                # 关闭逃生舱：_wt_double_send=False（不推荐，训练会 off-policy）。
                update_info2 = _build_nccl_update_info(sd_dev)
                self._weight_transfer_broadcast_round(sd_dev, update_info2, "第二发")
            _ulog.info("[WT-update] 完成")
            return True
        except BaseException as e:      # noqa: BLE001 —— 含 KeyboardInterrupt/SystemExit
            # init 已标记则保留更具体的 init 原因（不覆盖）；否则标记 update 原因。
            if not self.weight_sync_poisoned:
                self._mark_weight_transfer_poisoned(f"update: {e!r}")
            raise

    def _weight_transfer_broadcast_round(self, sd_dev: dict, update_info: dict,
                                         label: str) -> None:
        """单轮 NCCL 权重广播：sender 线程 trainer_send_weights + 主线程 collective_rpc 并发。

        修复（2026-08-19）：
        - collective_rpc 传参 timeout 只影响远端 worker，vLLM 0.16 的 call_utility 内部
          future.result() 不带超时——主线程可能永久卡在 [WT-update] 开始、GPU 0%。
          这里用 _run_with_timeout 给【主进程侧】加超时，超时即 fail-fast；
        - sender 线程 join 后检查 is_alive：仍存活说明 NCCL 广播未完成，继续执行会
          并发操作同一 communicator → 抛错；
        - sender 线程异常立即 error 日志（此前只 append，主线程卡住时不可见）。

        poisoned 语义（2026-08-19）：本函数是最外层广播入口——_run_with_timeout 超时 /
        collective_rpc 异常 / sender 线程异常 / sender join 超时 / sender 仍存活都会把
        engine 标记为 poisoned（daemon 线程无法被 Python 安全杀死，超时后 NCCL
        communicator 状态不可信）；poisoned engine 禁止复用，正确恢复是重建 engine。
        """
        try:
            _ulog = logging.getLogger(__name__)
            _timeout = float(getattr(self, "_wt_timeout", 120.0))
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine)
            _wt_dev = self._wt_group.device
            err: list[BaseException] = []

            def _send():
                try:
                    # stream 必须是对应 NCCL comm 设备（训练卡）的流；当前设备可能已被
                    # 其他 CUDA 操作改到 rollout 卡/参考模型卡 → 显式切回 comm 设备取流。
                    with torch.cuda.device(_wt_dev):
                        _stream = torch.cuda.current_stream()
                    NCCLWeightTransferEngine.trainer_send_weights(
                        iter(sd_dev.items()), self._wt_group, stream=_stream)
                except BaseException as e:      # noqa: BLE001 —— 立即日志 + 记录，主线程抛
                    _ulog.error("[WT] %s sender 线程异常：%r", label, e)
                    err.append(e)

            t = threading.Thread(target=_send, daemon=True, name=f"wt-send-{label}")
            t.start()
            _rpc_error: BaseException | None = None
            try:
                # 用 _run_with_timeout 包裹：collective_rpc 主进程侧超时即 fail-fast
                _run_with_timeout(
                    lambda: self.llm.llm_engine.collective_rpc(
                        "update_weights", timeout=_timeout,
                        kwargs={"update_info": update_info}),
                    _timeout + 30.0, f"{label}/collective_rpc")
            except BaseException as e:          # noqa: BLE001
                _rpc_error = e
                raise
            finally:
                t.join(timeout=_timeout + 30.0)
                if t.is_alive():
                    _ulog.error("[WT] %s sender 线程超时仍存活（NCCL 广播未完成）", label)
                    if _rpc_error is None:
                        raise RuntimeError(
                            f"[WT] {label} sender 线程超时仍存活（NCCL 广播未完成）——"
                            "已 fail-fast，避免与后续同步并发操作同一 communicator。")
            _ulog.info("[WT] %s 返回（send_err=%d）", label, len(err))
            if err:
                raise RuntimeError(f"vLLM NCCL 权重广播失败（{label}）：{err[0]}")
        except BaseException as e:      # noqa: BLE001 —— 含 KeyboardInterrupt/SystemExit
            self._mark_weight_transfer_poisoned(f"{label}: {e!r}")
            raise

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
                # clamp：vLLM prompt_logprobs=k 个别位置可能返回 k+1 项（含特殊 token）
                # ——超出的截断，不写越界（2026-08-17 实测 len=33 vs k=32）。
                _n = min(len(items), k)
                ids[b, t, :_n] = torch.tensor([int(tid) for tid, _ in items[:_n]],
                                              dtype=torch.long)
                lps[b, t, :_n] = torch.tensor([v.logprob for _, v in items[:_n]],
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
        # 2026-08-19 修复：数据层（JsonLinesDataLoader）把 prompt 右 pad 到
        # max_prompt_len（Qwen3 pad=151643）。HF generate 靠 attention_mask 自动排除
        # pad；vLLM 不会——尾部 pad token 作为真实上下文进入模型，导致生成退化/loop
        # （GPU 实测：HF 0 loop vs 旧 vLLM 路径 5/8 loop）。此处按 pad_id 去填充。
        seqs = _strip_prompt_padding(prompts, pad_id)
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



def _run_with_timeout(fn, timeout: float, label: str):
    """在后台线程执行阻塞调用并限时等待；超时（线程仍存活）立即抛错 fail-fast。

    vLLM 0.16 的 NCCL weight-transfer 阻塞集合调用（trainer_init / collective_rpc）内部
    用 future.result() 不带超时——主线程可能永久阻塞（日志停在 [WT-...] 开始、GPU
    利用率 0%、进程不死）。这里用线程 + join + is_alive 实现【主进程侧】超时：

    - 超时且线程仍存活 → RuntimeError（fail-fast，不再 0% 空等）；
    - 被调函数抛异常 → 原样向上传播（保留 SystemExit / KeyboardInterrupt 语义）。

    超时后 engine 会被调用方标记为 poisoned（2026-08-19）：
    - daemon 线程无法被 Python 安全杀死——超时后该线程可能仍卡在 NCCL 调用中；
    - 超时后 NCCL communicator 状态不可信，engine 禁止复用；
    - 正确恢复方式是进程退出 / 重建 engine；
    - 本函数【不会】在超时线程内强行 shutdown（vLLM shutdown 可能阻塞，且可能与残留
      NCCL 线程二次死锁）——本轮策略是 fail-closed，不是在线修复 communicator。
    """
    if not callable(fn):
        raise TypeError(f"_run_with_timeout({label})：fn 必须可调用")
    _res: dict = {}

    def _runner():
        try:
            _res["ret"] = fn()
        except BaseException as e:      # noqa: BLE001 —— 记录后由主线程原样抛
            _res["err"] = e

    t = threading.Thread(target=_runner, daemon=True, name=f"wt-timeout-{label}")
    t.start()
    t.join(timeout=float(timeout))
    if t.is_alive():
        raise RuntimeError(
            f"[WT] {label} 主进程侧超时（>{float(timeout):.0f}s）：疑似 NCCL 停滞或对端"
            "未就绪，已 fail-fast（不再 0% 空等）。请检查 worker 侧日志与 NCCL 组状态。")
    err = _res.get("err")
    if err is not None:
        raise err
    return _res.get("ret")


# NCCL 广播支持的 dtype（vLLM NCCLWeightTransferEngine / NCCL 原生数据类型子集）。
# 权重同步场景实际只出现浮点；显式排除 bool / complex / int16 等非 NCCL 标准类型。
_NCCL_SUPPORTED_DTYPES = frozenset({
    torch.int8, torch.uint8,
    torch.int32, torch.int64,
    torch.float16, torch.float32, torch.float64, torch.bfloat16,
})


def _prepare_weight_transfer_payload(state_dict: dict, wt_device) -> dict:
    """NCCL 权重广播 payload 预检与就位（纯函数，CPU 可单测）。

    在调用 collective_rpc() / 启动 sender 线程【之前】校验 state_dict，把常见错误
    （空 dict、非 Tensor、非法 dtype、空形状、device 错位、非连续）提前拦截，避免在
    训练线程深处或 NCCL 异步层爆错（配合 worker 线程即 0% 空等）。

    校验项：
    - state_dict 必须是非空 dict；
    - 每个值必须是 torch.Tensor；
    - dtype 必须是 NCCL 可支持类型；
    - 不允许空 shape / 零元素；
    - tensor 最终落在 wt_device（.to(wt_device)），非连续先 .contiguous()；
    - wt_device 不可用（None / 解析失败 / 无 type 属性）时抛清晰错误，而不是后续
      AttributeError。

    返回可直接广播的 {name: tensor}（全部在 wt_device 上、连续）。
    """
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise ValueError(
            "vLLM NCCL 权重广播 payload 为空：state_dict 必须是非空 dict。")
    if wt_device is None:
        raise RuntimeError(
            "vLLM NCCL 权重广播：_wt_group.device 为 None——NCCL weight-transfer 组"
            "未成功建立（trainer_init 未完成/失败）。")
    if isinstance(wt_device, str):
        try:
            wt_device = torch.device(wt_device)
        except Exception as e:      # noqa: BLE001
            raise RuntimeError(
                f"vLLM NCCL 权重广播：_wt_group.device 解析失败：{wt_device!r}") from e
    if not hasattr(wt_device, "type"):
        raise RuntimeError(
            f"vLLM NCCL 权重广播：_wt_group.device 不是有效 torch.device"
            f"（{type(wt_device).__name__}）。")
    out: dict = {}
    for name, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(
                f"vLLM NCCL 权重广播：参数 {name} 不是 torch.Tensor"
                f"（{type(v).__name__}）。")
        if v.dtype not in _NCCL_SUPPORTED_DTYPES:
            raise TypeError(
                f"vLLM NCCL 权重广播：参数 {name} dtype {v.dtype} 非 NCCL 支持类型"
                f"（支持：{sorted(str(d) for d in _NCCL_SUPPORTED_DTYPES)}）。")
        if v.numel() == 0 or any(s == 0 for s in v.shape):
            raise ValueError(
                f"vLLM NCCL 权重广播：参数 {name} 形状非法（shape={tuple(v.shape)}，"
                "不允许空维度/零元素）。")
        if v.device != wt_device:
            v = v.to(wt_device)
        if not v.is_contiguous():
            v = v.contiguous()
        out[name] = v
    return out



def _strip_prompt_padding(prompts: torch.Tensor, pad_id: int) -> list[list[int]]:
    """去掉每个 prompt 行右侧的 pad token（vLLM 生成前，纯函数，CPU 可单测）。

    数据层（JsonLinesDataLoader）把 prompt 右 pad 到 max_prompt_len（Qwen3 pad=151643）。
    HF generate 靠 attention_mask 自动排除 pad；vLLM 不会——尾部 pad 作为真实上下文
    进入模型，导致生成退化/loop。返回去 pad 后的变长 token 序列；空行兜底为 [pad_id]。
    """
    seqs: list[list[int]] = []
    for b in range(int(prompts.size(0))):
        row = [int(x) for x in prompts[b].tolist()]
        while row and row[-1] == int(pad_id):
            row.pop()
        if not row:
            row = [int(pad_id)]
        seqs.append(row)
    return seqs


def _build_nccl_update_info(state_dict: dict) -> dict:
    """从 HF state_dict 构建 vLLM>=0.16 NCCL update_info（纯函数，CPU 可单测）。

    names/dtype_names/shapes 与 trainer_send_weights 的迭代顺序一致（state_dict.items()）。
    is_checkpoint_format=True → worker 用 model.load_weights（HF→vLLM 合并层映射自动）。
    注意：不带 "backend" 键——worker 的 NCCLWeightTransferUpdateInfo 只接受
    names/dtype_names/shapes/packed/is_checkpoint_format；backend 由引擎启动时的
    WeightTransferConfig 决定（2026-08-17 实测：带 backend 会 "unexpected keyword"）。

    修复（2026-08-19）：加 names/dtype_names/shapes 长度一致性守卫（三者必须等长，
    否则 trainer_send_weights 迭代顺序与 worker 端 reload 错位）。
    """
    names = list(state_dict.keys())
    values = list(state_dict.values())
    dtype_names = [str(t.dtype).split(".")[-1] for t in values]
    shapes = [list(t.shape) for t in values]
    if not (len(names) == len(dtype_names) == len(shapes)):
        raise ValueError(
            "vLLM NCCL update_info 构建失败：names/dtype_names/shapes 长度不一致"
            f"（{len(names)}/{len(dtype_names)}/{len(shapes)}）。")
    return {
        "names": names,
        "dtype_names": dtype_names,
        "shapes": shapes,
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

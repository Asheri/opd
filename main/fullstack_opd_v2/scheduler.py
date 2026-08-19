"""v2 Stage 2 调度器：AsyncOPD 四阶段全异步流水线，队列里流动的是【批次】。

  PromptFeeder      : 流式喂出批次索引 (B,)
  RolloutCollector  : 用（可能陈旧的）student 快照对离线固定 rollout 批量重算 s_old，
                      权重只在版本推进时加载（v1 每样本加载一次）
  TeacherScorer     : 从 Lightning 张量缓存零拷贝取 Δ_T 贴到批次上（★无 live teacher）
  TrainDispatcher   : 按 mini-batch 训练——learner 时刻用当前 student 一次批量前向
                      重算 s_cur → π_old 加权 PG + PPO clip → k3 KL 正则 → 梯度裁剪
                      → 更新并 publish 新权重

算法内核与 v1 审阅修复后一致；重构的是执行底座（批量化 / 设备常驻 / 权重按需加载）。
"""

from __future__ import annotations

import os

import queue
import threading
import time

import torch

from .buffer import StalenessQueue, WeightStore, _MIN_QUEUE_SIZE
from .losses import pg_loss, low_var_kl, low_var_kl_support, expected_reward
from .model import CausalToyLM, response_dists

# pg_loss 失配屏蔽阈值（log 空间）：s_old < -20 视为支撑外 log0 近似（π_old≈0），贡献=0。
# 正常 s_old 最小值 ≈ -ln V（V=128k 时约 -11.5），10 以内的余量不误伤；_LOG_ZERO=-30 闭合。
LOG_RATIO_MAX = 20.0

# L2 refresh 刷新相位的 ratio 硬化阈值（log 空间）。refresh 样本的 s_old 是 rollout 时刻
# 学生快照、在 ring buffer 滞留后陈旧——学生漂移使 s_cur top-K 里某 token 高概率（logp≈0）
# 而 s_old 中等低（如 -15）时，ratio=exp(0-(-15))=3.3e6 爆炸；delta<0 时 pg_loss 悲观下界取
# 未 clip 的 ratio*delta → pg 无界（部署实测 E1=400/E2=134）。grad_clip 兜底训练仍稳定，但
# loss 数值失真。
# 修复分两层（lb 语义）：
#   1) log_ratio_clip=REFRESH_LOG_RATIO_MAX：对 logr 全局 clamp（IS 权重上界），从根上限定
#      ratio≤exp(3)≈20——这是根因硬化（s_old<-5 屏蔽管不住 s_old∈[-5,0] 的 ratio 放大）。
#   2) log_ratio_max=REFRESH_LOG_RATIO_MAX：纵深防御，屏蔽 s_old<-3 的支撑外 log0 近似位。
# 3.0 保证单 token |ratio*delta|≤exp(3)*delta_clip≈40，支撑重归一后 pg 回到个位数量级。
# base 路径（_train_step）s_old 来自 cache 快照、陈旧度受 staleness_threshold 双截断约束，
# ratio 温和，保持 LOG_RATIO_MAX=20 不误伤。
REFRESH_LOG_RATIO_MAX = 3.0

# base 路径（_train_step）的 IS 权重上界。base 的 s_old 来自 Async worker 异步快照，正常时
# 与 s_cur 同版本漂移小、logr<1，无需 clip。但【起步瞬态】会爆：初始大 lr 一步后学生剧烈
# 漂移，而 worker 异步滞后仍持初始权重 → s_old 与 s_cur 差极大（logr 可达 ~8），悲观下界在
# delta<0 时取未 clip 的 ratio*delta → pg 单步上千（部署实测步1/3 各 ~3800/~4300，E1/E2
# 确定性复现）。grad_clip 兜底训练仍稳定（步4+ 稳态 pg<27），但会污染 loss 监控。用 5.0
# 只挡天文 ratio（exp(5)≈148，正常 logr 远小于此），保留正常 IS 放大，不改变稳态语义。
BASE_LOG_RATIO_CLIP = 5.0

# 队列操作超时（秒）：put 满 / get 空时的轮询间隔，平衡吞吐与线程响应
_PUT_TIMEOUT = 0.5        # 入队侧（prompt→rollout、rollout→scorer、scorer→staleness_q）
_REFRESH_CHUNK = 4  # refresh 训练 chunk 大小（2026-08-18 双卡并行 OOM 实测：整批 (M,T,V) 前向+backward 峰值撞顶，chunk 化 + 梯度累积减半）
_GET_TIMEOUT = 1.0        # 消费侧空转轮询（rollout/scorer 无批次时）
_DISPATCH_GET_TIMEOUT = 10.0  # 训练调度器等待批次（远超 GET_TIMEOUT，容忍长队列空窗）

# ---- 分布式 / 高性能推理（GPU 部署骨架）：ray / megatron-core / vllm 可选 ----
# L5 用 Ray 把 rollout 拆到多卡进程；L2 用 megatron 把 learner 切 TP=2+SP；
# L3 用 vLLM TP=2 取代朴素前向做 rollout（response_dists 接口包一层）。
try:                                                # pragma: no cover
    import ray
except Exception:                                   # pragma: no cover
    ray = None

try:                                                # pragma: no cover
    from .rollout_vllm import VLLMRolloutEngine, vllm_available
except Exception:                                   # pragma: no cover
    VLLMRolloutEngine = None
    vllm_available = lambda: False

try:                                                # pragma: no cover
    from megatron.core import parallel_state as mpu  # noqa: F401
    _MEGATRON_AVAILABLE = True
except Exception:                                   # pragma: no cover
    _MEGATRON_AVAILABLE = False
    mpu = None

try:                                                # pragma: no cover
    from .model_megatron import MegatronCausalToyLM
except Exception:                                   # pragma: no cover
    MegatronCausalToyLM = None


_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16,
              "float32": torch.float32, "bfloat16": torch.bfloat16}


_OPD_MEM_TRACE = os.environ.get("OPD_MEM_TRACE") == "1"


def _mem(tag: str) -> None:
    """显存打点（调试用）：OPD_MEM_TRACE=1 时打印 memory_allocated/reserved。"""
    if not _OPD_MEM_TRACE or not torch.cuda.is_available():
        return
    print(f"[MEM:{tag}] allocated={torch.cuda.memory_allocated()/2**30:.1f}GiB "
          f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB", flush=True)


class AsyncBatchedScheduler:
    def __init__(self, student: CausalToyLM, cache, prompts: torch.Tensor,
                 responses: torch.Tensor, ref_dists: torch.Tensor | None,
                 ref_ids: torch.Tensor | None, ref_logp: torch.Tensor | None,
                 cfg: dict, device, rollout_engine=None,
                 initial_version: int = 0):
        self.student = student
        self.cache = cache
        self.prompts = prompts          # (N, P) device
        self.responses = responses      # (N, T) device
        # KL 锚点：dense 模式给 ref_dists (N,T,V)；稀疏模式给 ref_ids/ref_logp (N,T,Kr)
        self.ref_dists = ref_dists      # dense 锚点（可为 None）
        self.ref_ids = ref_ids          # 稀疏锚点 token id
        self.ref_logp = ref_logp        # 稀疏锚点 logp
        # P1-1：ref 锚点按 token id 预排序，训练期 searchsorted 二分匹配（省 O(K²)）。
        # ⚠️ 与 cache.ids_sorted **不同源**：ref_ids 是初始 student 分布的 top-K，
        # token 值与顺序都和 teacher 缓存不同，必须单独预排序，不能复用 cache 的。
        self.ref_ids_sorted: torch.Tensor | None = None
        self.ref_logp_sorted: torch.Tensor | None = None
        if self.ref_ids is not None:
            self.ref_ids_sorted, _ro = self.ref_ids.sort(dim=-1)
            self.ref_logp_sorted = self.ref_logp.gather(-1, _ro)
        self.kl_mode = "dense" if ref_dists is not None else "topk"
        self.cfg = cfg
        self.device = device
        self.batch = cfg.get("batch_size", 8)
        self.n_prompts = prompts.size(0)

        # 稀疏配置（真实词表 / GPU 部署）
        # 稀疏缓存模式下必须按 student 支撑展开 Δ_T；若未显式给 top_k_student，
        # 默认复用教师缓存的 K，避免误用 dense 路径拿到 None。
        self.top_k_student = cfg.get("top_k_student", 0) or 0
        if cache.mode == "topk" and self.top_k_student <= 0:
            self.top_k_student = cache.top_k
        self.ref_tail_logp = cfg.get("ref_tail_logp", -1e2)   # 支撑外 ref logp（≈log 0）
        # 稀疏支撑重归一化（对齐原始 Direct-OPD 的 softmax(student_topk_logp)）：
        # 开 → pg_loss 把 π_old 在 Δ≠0 支撑上重归一、low_var_kl_support 把 π_cur 在
        # top-K 上重归一（条件期望）；关 → 原「非归一截断」有界近似。默认关（保既有
        # 行为/测试）；GPU 稀疏预设（gpu_skeleton_2gpu.yaml / CLOUD_CONFIG）开。
        self.renormalize_topk = bool(cfg.get("renormalize_topk_support", False))
        # Δ_T 数值护栏（部署实测 P1）：真实教师对 log-ratio 差可达 ±10 → PG 无界爆炸、
        # 学生坍缩。非 None 时 pg_loss 先 clamp Δ_T 到 ±delta_clip。
        self.delta_clip = cfg.get("delta_clip")

        # bf16 自动混合精度（L1）；仅在 cuda + 配置时启用
        self.dtype = _DTYPE_MAP.get(str(cfg.get("dtype", "fp32")).lower(), torch.float32)
        self.amp = (str(device).startswith("cuda") and self.dtype == torch.bfloat16)
        # P3（2026-08-19）：refresh 训练 chunk 大小可配（原模块级 _REFRESH_CHUNK=4 硬编码，
        # v5 OOM 实测双卡并行 + vLLM 共卡时 chunk=4 的 (4,T,V) 前向仍撞顶，降到 2 可减半）。
        self.refresh_chunk = max(1, int(cfg.get("refresh_chunk", 4)))

        # colocated CPU offload 钩子（L6）：rollout 阶段把 learner 权重换出到 CPU
        self.offload_to_cpu = bool(cfg.get("offload_to_cpu", False))
        # 激活重计算（GPU 显存，默认关）：Qwen3-1.7B × (4,3072) 的 backward 需重放
        # 28 层前向激活 ≈ 25GB，叠加 s_cur/s_old/log_softmax 大张量 → 80GB 撞顶
        # （2026-08-18 loss.backward OOM 实测）。开 → HF gradient_checkpointing 每层
        # 前向时丢弃激活、backward 重算（显存省 ~90%，时间换空间，数值语义零变化）。
        # 需 use_cache=False 配套（rollout 走 vLLM 引擎，student 不做自回归推理）。
        if cfg.get("gradient_checkpointing", False):
            if not hasattr(self.student, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "gradient_checkpointing=true 仅支持 HF 模型（student 无 "
                    "gradient_checkpointing_enable），toy/其他骨架请关闭该开关")
            gc = getattr(self.student.config, "use_cache", None)
            if gc is True:
                self.student.config.use_cache = False
            self.student.gradient_checkpointing_enable()

        # IMP-2/P1 显存：staleness_q 深度可配（默认 16）。真实词表下在途 s_old 稠密
        # (B,T,V) 大张量，槽位越多峰值显存越高；显存受限时用
        # --set stage2.staleness_queue_min=2 收紧（会降低异步流水深度，不影响损失）。
        self.staleness_q = StalenessQueue(
            cfg.get("staleness_threshold", 4),
            min_queue_size=int(cfg.get("staleness_queue_min", _MIN_QUEUE_SIZE)))
        # G5（§2 Q4 契约）：base 池样本是否跳过消费侧陈旧度截断。base 的 s_old 由
        # RolloutCollector 每次用当前权重重算、天然带新版本（恒新），截断从不误触发；
        # L2 交替相位把此标志置 False 显式落实「base 跳过、仅 refresh 受截断」契约。
        self.staleness_drop_base = bool(cfg.get("staleness_drop_base", True))
        self.weight_store = WeightStore(offload_to_cpu=self.offload_to_cpu)
        self.stop = threading.Event()
        # IMP-2/P1 稳健性：worker 线程异常 → 记录并停线程，主循环快速失败
        # （此前 collector 线程死亡 → _train_dispatcher 空等 queue → GPU 0% 空挂）。
        self._thread_error: Exception | None = None
        self._err_lock = threading.Lock()
        self.metrics: list = []
        # P2-1：只观测计数器（不改控制流）
        self._n_rollout = 0          # rollout 实际前向次数
        self._n_dropped_consume = 0  # 消费侧因过旧丢弃数
        self._n_dropped_qfull = 0    # 队列满丢弃数（rollout→scorer 或 scorer→staleness_q）
        self._rollout_idle = 0.0     # RolloutCollector 累计空转秒
        self._scorer_idle = 0.0      # TeacherScorer 累计空转秒
        self._qfull_streak = 0       # I4：连续队满次数（自适应背压，不丢成品）

        # 优化器 + 超参提升为实例属性，供 _train_step 在两种调度器间复用
        self.kl_coef = cfg.get("kl_reg_coef", 0.05)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.use_topk = (self.top_k_student > 0) and (cache.mode == "topk")
        self.opt = self._build_optimizer(student, cfg)
        _mem("scheduler.__init__:opt_ready")

        # 初始化权重快照（首版 = student 当前权重），供 rollout worker 加载版本 0。
        # 注意：不调用 self._publish() —— 分布式版的 _publish 已被 NCCL 广播覆盖，
        # 父类 init 阶段 broadcaster 尚未就绪；直接填 snapshot 即可让线程版正常 acquire，
        # 分布式版走 NCCL 广播不读此 store。
        self.weight_store._snapshot = {
            k: v.detach().clone() for k, v in student.state_dict().items()}
        self.weight_store._version = 0

        # resume（T11）：断点续跑时恢复版本号，使 staleness age 计算与陈旧度一致。
        if initial_version > 0:
            self.staleness_q._cur_version = initial_version
            self.weight_store._version = initial_version

        # L3 · vLLM rollout 引擎（可选）：供 pipeline 的 L2 刷新相位生成与稀疏分布前向
        # （generate_with_status / response_dists_topk，见 pipeline 接线）。
        # ⚠️ IMP-2/P1 实测修复（2026-08-17）：base 池 s_old【恒用 HF/toy worker 前向】，
        # 不走 vLLM 的稠密 response_dists 重建——真实词表（V=151936）下 vLLM 逐 token
        # prompt_logprobs 重建 (B,T,V) 慢 ~50s/样本且占显存，且 vocab 错误时会 IndexError
        # 挂死 collector 线程→主循环空等（GPU 占用 0%）。vLLM 的职责是【生成】与【稀疏
        # top-K 分布】，base 池稠密 s_old 由 HF worker 一次前向完成（与 toy 路径一致）。
        self.rollout_engine = rollout_engine
        if self.rollout_engine is not None and not vllm_available():
            raise RuntimeError("cfg 指定 vLLM rollout 但 vllm 未安装（统一 GPU 环境应含）。")
        # 旧快照前向模型（base 池 s_old）：toy → CausalToyLM 副本；hf → 同 student
        # 路径再加载一份 HFCausalLM（权重随后经 weight_store 快照覆盖）。
        # P1-B：model_kind 由 pipeline 从顶层注入 s2cfg；student_path 同理（hf 时
        # build_model 按 role=student 读它）。n_layers 用 getattr 防御（非 toy student
        # 可能无该属性，避免 CausalToyLM 分支构造崩溃）。
        if cfg.get("model_kind") == "hf":
            from .model_factory import build_model as _build_model
            self.worker = _build_model(cfg, device, role="student")
        else:
            self.worker = CausalToyLM(vocab=student.vocab, d_model=student.d_model,
                                      n_layers=getattr(student, "n_layers",
                                                       cfg.get("n_layers", 2))).to(device)
        self._loaded_ver = -1

    # --------------------------- 优化器 ---------------------------
    def _build_optimizer(self, student, cfg):
        """按 cfg['optimizer'] 构造优化器：adam（默认 fp32）/ adamw_8bit（bnb，7B 单卡）。

        多学生并发（GPU_MEMORY_AND_PARALLEL_PLAN §7）：4B/7B 的 fp32-Adam 超单卡
        （4B 61.8GB / 7B 121.6GB > 96GB），8-bit Adam（bnb AdamW8bit）把优化器状态压到
        int8（7B 总 ~60GB GPU）→ 单卡可训。⚠️ bnb 缺省时显式报错（不静默回退 fp32 导致 OOM）。
        """
        opt_type = cfg.get("optimizer", "adam")
        lr = cfg.get("lr", 1e-3)
        if opt_type == "adamw_8bit":
            try:
                from bitsandbytes.optim import AdamW8bit
            except Exception as e:                     # pragma: no cover
                raise RuntimeError(
                    f"optimizer='adamw_8bit' 需要 bitsandbytes（pip install bitsandbytes）：{e}")
            return AdamW8bit(student.parameters(), lr=lr)
        if opt_type == "adam":
            return torch.optim.Adam(student.parameters(), lr=lr)
        raise RuntimeError(f"未知 optimizer={opt_type!r}（adam | adamw_8bit）")

    # --------------------------- 权重同步 ---------------------------
    def _publish(self) -> int:
        v = self.weight_store.publish(self.student.state_dict())
        self.staleness_q.advance_version()
        return v

    # --------------------------- 四个解耦阶段 ---------------------------
    def _prompt_feeder(self):
        while not self.stop.is_set():
            idxs = torch.randint(0, self.n_prompts, (self.batch,))
            try:
                self._pq.put(idxs, timeout=_PUT_TIMEOUT)
            except queue.Full:
                continue

    def _rollout_collector(self):
        while not self.stop.is_set():
            try:
                idxs = self._pq.get(timeout=_GET_TIMEOUT)
            except queue.Empty:
                self._rollout_idle += 1.0      # 空转 1s
                continue
            # ★ 只在权重版本推进时才加载（v1 每样本一次 load_state_dict）
            snap, ver = self.weight_store.acquire_if_newer(self._loaded_ver)
            if snap is not None:
                # colocated CPU offload（L6）：快照可能在 CPU，加载前搬回设备
                if self.offload_to_cpu:
                    snap = {k: v.to(self.device) for k, v in snap.items()}
                # IMP-2/P1：base 池 s_old 恒用 HF/toy worker（vLLM 只负责 L2 生成）。
                self.worker.load_state_dict(snap)
                self._loaded_ver = ver
            self._n_rollout += 1
            idxs_dev = idxs.to(self.device)
            self.worker.eval()
            with torch.no_grad():
                # P-显存修复：worker 前向产物 s_old 是 (B,T,V) 大张量，HF lm_head 输出
                # 在 autocast 下仍 fp32（2.5GB/份@batch2）。队列 queue_size=2 + 训练
                # backward 同时驻留 → 与 train_step 叠加 OOM（部署实测 84GB）。autocast
                # 省前向激活 + 显式 .to 让队列里的 s_old 全程 bf16（砍半）。
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype,
                                        enabled=self.amp):
                    s_old = response_dists(self.worker, self.prompts[idxs_dev],
                                           self.responses[idxs_dev], dtype=self.dtype)  # (B,T,V)
                if self.dtype is not None and s_old.dtype != self.dtype:
                    s_old = s_old.to(self.dtype)
            try:
                # L6 offload（与权重快照同开关 offload_to_cpu）：在途 s_old (B,T,V) 大
                # 张量入队前搬 CPU。GPU 实测：queue_size 默认 8 槽 × 7.46GB/份 ≈ 60GB
                # 驻留，叠加训练前向 OOM 95GB；offload 后队列全 CPU，_train_step 已有
                # s_old.to(device, dtype) 拷回——数值完全一致（同张量换设备），分布式
                # 路径 894 行 s_old.cpu() 同款先例。开关关（默认）保持原 GPU 行为。
                if self.offload_to_cpu:
                    s_old = s_old.cpu()
                self._rq.put((idxs, s_old, self._loaded_ver), timeout=_PUT_TIMEOUT)
                self._qfull_streak = 0
            except queue.Full:
                # I4：队满时自适应退避（连续队满 → 指数加长 sleep），不丢弃已算完的
                # 成品 s_old（此前直接丢，积压期训练饿死+算力浪费）。
                self._rollout_idle += 0.5
                self._qfull_streak += 1
                time.sleep(min(0.1 * self._qfull_streak, 2.0))
                continue

    def _ref_logp_at_student_topk(self, idxs: torch.Tensor,
                                  student_ids: torch.Tensor) -> torch.Tensor:
        """取初始 student 在 student top-K 支撑上的 logp（稀疏 KL 锚点）。

        self.ref_ids/ref_logp: (N,T,Kr)。student_ids: (B,T,Ks)。
        匹配上的取 teacher/ref logp；未匹配（student 高概率但初始分布几乎为零）填
        ref_tail_logp（≈ -1e2），使 k3 正确给出强惩罚（防策略漂移）。
        """
        # P1-1：searchsorted 二分匹配替代 O(Ks×Kr) 全对比较。ref 锚点已升序预排序，
        # found 是 bool 张量（`==` 结果），torch.where(bool, a, b) 语义与原来 has>0.0 等价。
        rids_sorted = self.ref_ids_sorted[idxs]              # (B, T, Kr) 已升序
        rlogp_sorted = self.ref_logp_sorted[idxs]            # (B, T, Kr)
        Kr = rids_sorted.size(-1)
        # L2 refresh 样本的响应长（rollout 预算 512/1024/2048）可能 ≠ 锚点 T（缓存响应长
        # 2048）：searchsorted 要求除排序维外各维匹配，按 min T 截断避免维度崩。截断是
        # 近似——刷新响应尾部超出锚点的位置失去 KL 锚点（真实场景刷新短于锚点，不丢）。
        Tb, Ts = rids_sorted.size(1), student_ids.size(1)
        if Ts != Tb:
            T = min(Ts, Tb)
            rids_sorted = rids_sorted[:, :T]
            rlogp_sorted = rlogp_sorted[:, :T]
            student_ids = student_ids[:, :T]
        pos = torch.searchsorted(rids_sorted, student_ids.contiguous()).clamp(max=Kr - 1)
        found = rids_sorted.gather(-1, pos) == student_ids
        gathered = rlogp_sorted.gather(-1, pos)              # (B, T, Ks)
        return gathered.where(found,
                              torch.full_like(gathered, self.ref_tail_logp))

    def _teacher_scorer(self):
        while not self.stop.is_set():
            try:
                idxs, s_old, ver = self._rq.get(timeout=_GET_TIMEOUT)
            except queue.Empty:
                self._scorer_idle += 1.0
                continue
            if self.cache.mode == "topk":
                # 稀疏模式：Δ_T 按 student top-K 支撑在 learner 现场展开（见 _train_dispatcher），
                # 此处只透传 idxs，避免搬运 (B,T,V)。
                delta_payload = None
            else:
                delta_payload = self.cache.get_delta(idxs.to(self.device))   # (B,T,V) 零拷贝
            try:
                self.staleness_q.put((idxs, s_old, delta_payload), version=ver, timeout=_PUT_TIMEOUT)
            except queue.Full:
                self._n_dropped_qfull += 1     # M5：scored 样本因队满被丢弃
                continue

    def _train_step(self, done, idxs, s_old, delta, ver):
        """单步训练（线程版与分布式版共用）。

        入参 idxs / s_old 为 CPU 张量（分布式场景下由 worker 进程回传）；
        delta 为 dense 模式的 (B,T,V) 迁移对象——稀疏模式下传 None，由 learner
        现场按 student 支撑展开。陈旧度超阈值返回 None（截断），否则返回 metric dict。
        """
        threshold = self.cfg.get("staleness_threshold", 4)
        # 陈旧度截断（消费侧，与 v1 审阅修复一致）。G5：base 池跳过（恒新），
        # 仅 refresh 池（存 rollout 时刻行为 s_old）受陈旧度约束。
        if self.staleness_drop_base and self.staleness_q.current_version - ver > threshold:
            return None
        self.student.train()
        idxs_dev = idxs.to(self.device)
        p_b = self.prompts[idxs_dev]
        r_b = self.responses[idxs_dev]
        _mem("train_step:start")

        # learner 时刻用【当前】student 重算分布（recompute 代理）。
        # bf16 自动混合精度（L1）：包住前向 + 损失 + 反向（bf16 有范围，无需 GradScaler）。
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype,
                                enabled=self.amp):
            # P5 dtype 透传（2026-08-18 GPU 实测）：HF lm_head 输出在 autocast 下仍为
            # fp32，(B,T,V) fp32 双份（logits+log_softmax）驻留推高训练峰值——改在函数内
            # 立即转 bf16，fp32 不离开 response_dists（峰值减半）。下游 373 行防御转换
            # 保留（幂等）。真实词表 V=151936 下每份 (4,3072,V) fp32≈7.5GB。
            s_cur = self.student.response_dists(p_b, r_b, dtype=self.dtype)  # (B,T,V) 带梯度
            # P-显存修复：HF lm_head 输出在 autocast 下仍为 fp32（transformers 行为），
            # 真实词表 V=152k 下 (B,T,V) fp32=2.5GB/张，而 pg_loss 内部 ~11 个中间张量
            # 全堆 fp32 → 25GB+，叠加 worker/队列/激活 OOM（部署实测 87GB）。转回 autocast
            # dtype（bf16）砍半——Δ_T 本就在 bf16 域，PG 中间量 bf16 精度足够。
            if self.dtype is not None and s_cur.dtype != self.dtype:
                s_cur = s_cur.to(self.dtype)
            # 与 s_cur 同设备同精度，保证 ratio 一致（M2：分布式路径 s_old 由 worker 进程
            # 回传在 CPU，此前只转 dtype 不转 device → s_cur-s_old 设备不匹配崩溃）。
            s_old = s_old.to(self.device, dtype=s_cur.dtype)
            # P0：稀疏 base（默认开）不物化稠密 p_old=s_old.exp()（T=4096 下 ~5GB 浪费），
            # 改在支撑上取 s_old_at.exp()。仅稠密兜底路径需要 p_old。
            _sparse_base = bool(self.use_topk and self.cfg.get("base_sparse_pg", True))
            if not _sparse_base:
                # P2-2：缓存 p_old 供 pg_loss 与 adv 监控复用；全 1 mask 走 None 快路径
                # ⚠️ mask 快路径前提：调度器无 padding（responses 等长），恒全 1。
                p_old = s_old.exp()

            # ★ Direct-OPD 迁移对象：按 student 自身 top-K 支撑取 Δ_T（L4 稀疏缓存）
            # P0（OOM 根治，base_sparse_pg 默认开）：base 池 PG 只在 student top-K 支撑
            # 上计算——与 refresh 路径（_train_step_refresh）同构，中间量从 (B,T,V=151936)
            # 缩到 (B,T,K=256) ≈ 缩小 594 倍，T=4096 的 base 步峰值 ~80GB → 几 GB。
            # 数值等价：稠密版 delta 仅支撑非零且在【同一支撑】上重归一，稀疏版直接
            # 把支撑值喂 pg_loss（delta_at / s_old_at / values），分母一致。
            if self.use_topk:
                s_topk = torch.topk(s_cur, self.top_k_student, dim=-1)   # (B,T,K)
                if _sparse_base:
                    delta_at = self.cache.delta_at_student_topk(
                        idxs_dev, s_topk.indices, self.device)           # (B,T,K)
                    if self.dtype is not None and delta_at.dtype != self.dtype:
                        delta_at = delta_at.to(self.dtype)
                    # s_old 稠密 → 在 s_cur top-K 支撑上 gather（精确，无需 tail 语义）
                    s_old_at = s_old.gather(-1, s_topk.indices)          # (B,T,K)
                    sup = torch.ones_like(s_topk.values, dtype=torch.bool)
                    loss_pg = pg_loss(s_topk.values, s_old_at, delta_at, None,
                                      self.clip_eps, p_old=s_old_at.exp(),
                                      log_ratio_max=LOG_RATIO_MAX,
                                      log_ratio_clip=BASE_LOG_RATIO_CLIP,
                                      renormalize_support=self.renormalize_topk,
                                      support=sup, delta_clip=self.delta_clip)
                else:
                    # 旧稠密路径（兜底，base_sparse_pg=false 回退）
                    delta_d = self.cache.delta_for_student_topk(
                        idxs_dev, s_topk.indices,
                        vocab_out=getattr(self.student, "vocab", None))  # (B,T,V) 支撑外=0
                    if self.dtype is not None and delta_d.dtype != self.dtype:
                        delta_d = delta_d.to(self.dtype)
                    pg_support = None
                    if self.renormalize_topk:
                        pg_support = torch.zeros_like(delta_d, dtype=torch.bool)
                        pg_support.scatter_(-1, s_topk.indices, True)
                    loss_pg = pg_loss(s_cur, s_old, delta_d, None, self.clip_eps,
                                      p_old=p_old, log_ratio_max=LOG_RATIO_MAX,
                                      log_ratio_clip=BASE_LOG_RATIO_CLIP,
                                      renormalize_support=self.renormalize_topk,
                                      support=pg_support, delta_clip=self.delta_clip)
                # 稀疏 KL 锚点
                if self.kl_mode == "topk":
                    ref_at = self._ref_logp_at_student_topk(
                        idxs_dev, s_topk.indices)               # (B,T,Ks)
                    loss_kl = low_var_kl_support(s_topk.values, ref_at, None,
                                                 renormalize_support=self.renormalize_topk)
                else:
                    loss_kl = low_var_kl(s_cur, self.ref_dists[idxs_dev], None)
            else:
                # dense 模式（demo 默认）：delta 应为完整 (B,T,V)。
                # M2 防御：DistAsyncScheduler.run 传 None（worker 只回传 idxs/s_old），
                # 此处现场从缓存零拷贝取 Δ_T，与线程版 _teacher_scorer 的 get_delta 同源——
                # 线程版已传真实 delta 则跳过，不重复搬运。
                if delta is None:
                    delta = self.cache.get_delta(idxs_dev)
                delta_d = delta
                # P-显存修复（同 topk 分支）：delta fp32 与 bf16 s_cur 运算拉回 fp32。
                if self.dtype is not None and delta_d.dtype != self.dtype:
                    delta_d = delta_d.to(self.dtype)
                loss_pg = pg_loss(s_cur, s_old, delta_d, None, self.clip_eps, p_old=p_old,
                                  log_ratio_max=LOG_RATIO_MAX,
                                  log_ratio_clip=BASE_LOG_RATIO_CLIP,
                                  delta_clip=self.delta_clip)
                loss_kl = low_var_kl(s_cur, self.ref_dists[idxs_dev], None)

            loss = loss_pg + self.kl_coef * loss_kl

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.opt.step()

        version = self._publish()
        with torch.no_grad():
            if _sparse_base:
                p_cur = s_topk.values.exp()                        # (B,T,K) 支撑上 E
                reward = (p_cur * delta_at).sum(-1).mean()
                adv = (s_old_at.exp() * delta_at).sum(-1).mean()
            else:
                reward = expected_reward(s_cur.detach(), delta_d, None,
                                         p_dists=s_cur.detach().exp()).mean()
                adv = expected_reward(s_old, delta_d, None, p_dists=p_old.detach()).mean()

        # C3：热路径 5 个标量收集成一个大张量一次 device→cpu，避免逐 .item() 各触发
        # 一次同步。autocast 下 loss 族可能为 bf16/fp16，adv/reward 在 no_grad 外为
        # fp32——先统一转 fp32 再 stack（stack 要求同 dtype）。
        scalars = [loss, loss_pg, loss_kl, adv, reward]
        scalars = [s.float() if s.dtype != torch.float32 else s for s in scalars]
        loss_v, pg_v, kl_v, adv_v, rew_v = torch.stack(scalars).detach().cpu().tolist()
        return {
            "step": done,
            "version": version,
            "age": version - ver,
            "batch": int(s_cur.size(0)),
            "loss": loss_v,
            "pg_loss": pg_v,
            "kl_loss": kl_v,
            "adv_mean": adv_v,
            "reward": rew_v,
        }

    # --------------------------- L2 双池 feeder：refresh 训练步（G1，闭环核心） ---------------------------
    def _train_step_refresh(self, done, rb_idxs, rb, on_step=None):
        """L2 refresh 池样本的 teacher-free 稀疏 top-K PG + KL（§2 双池，G1 闭环）。

        refresh 样本的 Δ_T（教师 top-K）与行为策略 s_old（生成时学生 top-K）都已在
        rollout 相位存进 ring buffer；这里按 s_cur 当前 top-K 支撑展开，做稀疏支持
        重归一 PG + KL（与 base 稀疏路径同内核对齐 Direct-OPD）。全程无 teacher 前向。

        rb_idxs: (B,) ring buffer 局部槽位。token_mask 处理变长（真实 EOS padding）。
        """
        if not self.use_topk:
            # dense/toy 模式（top_k_student=0）：ring buffer 只存稀疏 top-K（teacher Δ、
            # 行为 s_old、ref 锚点都是 top-K 支撑），无法做稠密 refresh 训练。产出
            # pool="refresh" no-op 步（维持交替相位闭环与测试语义，loss=0 无学习信号）。
            # 真实规模（top_k_student>0，由 cache.top_k 驱动，cache_mode=topk）走下方
            # 稀疏 top-K 路径。
            print("[scheduler] refresh 训练相位在 dense/toy（top_k_student=0）下为 no-op；"
                  "真实规模需稀疏 top-K 配置", flush=True)
            return {
                "step": done, "version": self.staleness_q.current_version,
                "age": 0, "pool": "refresh", "batch": self.batch,
                "loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0,
                "adv_mean": 0.0, "reward": 0.0,
            }
        # 显存（2026-08-18 双卡并行实测）：refresh 训练整批 (8,2048,V) 前向+backward，
        # 在 ~67GB 驻留（模型+锚点+对侧 vLLM 引擎 12GB 同卡）上再触发 9.77GB 分配 → OOM。
        # 拆 chunk（_REFRESH_CHUNK=4，batch=8 → 2 块）【独立小批更新】：每 chunk 完整
        # forward/backward/step（4 条/次），峰值增量减半（~5GB），图完全独立无共享。
        # refresh 训练的小批粒度是超参选择（更接近标准 SGD），不影响 rollout/验收口径。
        chunks = list(rb_idxs.split(max(1, min(self.refresh_chunk, rb_idxs.size(0)))))
        self.student.train()
        acc = {"loss": [], "pg": [], "kl": [], "rew": [], "adv": []}
        version = self.staleness_q.current_version
        for chunk in chunks:
            batch = rb.get(chunk)
            # 拷贝成独立张量（ring buffer 全局 storage 共享，防止任何视图别名问题）
            p_b = self.prompts[batch["prompt_idx"]].to(self.device)
            r_b = batch["responses"].to(self.device)
            mask = batch["token_masks"].to(self.device).clone()
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype,
                                    enabled=self.amp):
                s_cur = self.student.response_dists(p_b, r_b, dtype=self.dtype)  # (B,T,V) 带梯度（P5）
                # P-显存修复（同 _train_step）：HF lm_head 输出 autocast 下仍 fp32，真实
                # V=152k 大张量堆 fp32 OOM；转回 autocast dtype（bf16）。refresh 只用 topk
                # 小张量，bf16 精度足够。
                if self.dtype is not None and s_cur.dtype != self.dtype:
                    s_cur = s_cur.to(self.dtype)
                s_topk = torch.topk(s_cur, self.top_k_student, dim=-1)
                # Δ_T 展开到 s_cur top-K（教师 top-K 命中处取 delta_k，未命中=0）
                delta_at = rb.delta_at_student_topk(
                    chunk, s_topk.indices, self.device)                # (B,T,Ks)
                # 行为策略 s_old 展开到 s_cur top-K（未命中填 ref_tail_logp≈log 0）
                s_old_at = rb.s_old_at_student_topk(
                    chunk, s_topk.indices, self.device,
                    tail_logp=self.ref_tail_logp)                      # (B,T,Ks)
                # IMP-3（Refresh KL Anchor Correctness）：refresh KL 锚点优先用 ring buffer 存的
                # 【初始 student 在 rollout 响应上的 top-K】（rollout 相位 per-chunk 算好）。
                # 旧断点/旧调用无 ref_anchor_* 时回落静态 fat_responses 锚点（向后兼容；
                # 注意该回落路径存在锚点错位——token 重合 ~2%、支撑外 ~27% → kl_loss 爆炸，
                # 见 IMP-3 报告，仅对旧 checkpoint/旧数据生效）。
                if rb.ref_anchor_ids is not None:
                    ref_at = rb.ref_anchor_at_student_topk(
                        chunk, s_topk.indices, self.device, tail_logp=self.ref_tail_logp)
                elif self.kl_mode == "dense":
                    _T = min(self.ref_dists[batch["prompt_idx"]].size(1),
                             s_topk.indices.size(1))
                    ref_at = self.ref_dists[batch["prompt_idx"]][:, :_T].gather(
                        -1, s_topk.indices[:, :_T])                    # (B,T,Ks)
                else:
                    ref_at = self._ref_logp_at_student_topk(
                        batch["prompt_idx"], s_topk.indices)           # (B,T,Ks)
                # 支撑 = 完整 s_cur top-K（ones），与 KL 同源重归一（对齐原始 Direct-OPD）
                sup = torch.ones_like(s_topk.values, dtype=torch.bool)
                loss_pg = pg_loss(s_topk.values, s_old_at, delta_at, mask, self.clip_eps,
                                  p_old=s_old_at.exp(), log_ratio_max=REFRESH_LOG_RATIO_MAX,
                                  log_ratio_clip=REFRESH_LOG_RATIO_MAX,
                                  renormalize_support=True, support=sup,
                                  delta_clip=self.delta_clip)
                loss_kl = low_var_kl_support(s_topk.values, ref_at, mask,
                                             renormalize_support=True)
                loss = loss_pg + self.kl_coef * loss_kl
                with torch.no_grad():
                    p_cur = s_topk.values.exp()
                    reward = (p_cur * delta_at).sum(-1) * mask
                    adv = (s_old_at.exp() * delta_at).sum(-1) * mask
                    acc["loss"].append(loss.detach().float())
                    acc["pg"].append(loss_pg.detach().float())
                    acc["kl"].append(loss_kl.detach().float())
                    acc["rew"].append((reward.sum() / max(mask.sum(), 1)).float())
                    acc["adv"].append((adv.sum() / max(mask.sum(), 1)).float())
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
            self.opt.step()
            version = self._publish()
        # 展示用聚合：各 chunk 标量均值
        loss_v, pg_v, kl_v, adv_v, rew_v = [
            float(torch.stack(acc[k]).mean()) if acc[k] else 0.0 for k in
            ("loss", "pg", "kl", "adv", "rew")]
        return {
            "step": done,
            "version": version,
            "age": 0,
            "pool": "refresh",
            "batch": int(rb_idxs.size(0)),
            "loss": loss_v,
            "pg_loss": pg_v,
            "kl_loss": kl_v,
            "adv_mean": adv_v,
            "reward": rew_v,
        }

    def train_refresh_phase(self, rb, alpha: float, n_refresh_steps: int,
                            start_step: int, on_step=None) -> int:
        """L2 双池 feeder：从 ring buffer 采 n_refresh_steps 批做 refresh 训练。

        α 已由 DynamicRatioController 决定（G3 应用）；n_refresh_steps 由 pipeline 按
        α/(1-α)·n_base 折算。采样带确定性 generator（可复现）。返回实际完成步数。
        """
        if rb.size == 0 or n_refresh_steps <= 0:
            return 0
        gen = torch.Generator().manual_seed(42)
        done = start_step
        completed = 0
        for _ in range(n_refresh_steps):
            rb_idxs = rb.sample(self.batch, gen)
            if rb_idxs.numel() == 0:
                break
            m = self._train_step_refresh(done, rb_idxs, rb)
            self.metrics.append(m)
            if on_step is not None:
                try:
                    on_step(m)
                except Exception as e:         # I3
                    import logging
                    logging.getLogger(__name__).warning(f"refresh on_step 回调异常：{e}")
            done += 1
            completed += 1
        return completed

    def _train_dispatcher(self, n_steps: int, on_step=None, start_step: int = 0):
        done = start_step
        # C2：无进度 watchdog——线程活着但永不产出的“挂住”（如 vLLM IPC 死锁）也快速失败。
        stall_timeout = float(self.cfg.get("rollout_stall_timeout", 900))
        last_progress = time.monotonic()
        while done < start_step + n_steps:
            # 快速失败：任一 worker 线程死亡（异常）立即抛出，不空等（避免 GPU 0%）。
            with self._err_lock:
                if self._thread_error is not None:
                    raise RuntimeError(
                        "worker 线程异常（训练中止，避免空等）："
                        f"{type(self._thread_error).__name__}: {self._thread_error}") \
                        from self._thread_error
            if time.monotonic() - last_progress > stall_timeout:
                raise RuntimeError(
                    f"rollout 停滞：{stall_timeout:.0f}s 无成功训练步"
                    "（生产者在排队/IPC 上挂住；请查 collector/scorer 线程）")
            try:
                (idxs, s_old, delta), ver, _ = self.staleness_q.get(timeout=_DISPATCH_GET_TIMEOUT)
            except queue.Empty:
                continue
            m = self._train_step(done, idxs, s_old, delta, ver)
            if m is None:
                self._n_dropped_consume += 1   # 消费侧因过旧丢弃
                continue
            last_progress = time.monotonic()
            self.metrics.append(m)
            if on_step is not None:            # T8：每成功一步回调（checkpoint/metrics 用）
                try:
                    on_step(m)
                except Exception as e:         # I3：回调异常不得静默吞（至少告警）
                    import logging
                    logging.getLogger(__name__).warning(f"on_step 回调异常（训练继续）：{e}")
            done += 1
        self.stop.set()

    # --------------------------- 入口 ---------------------------
    def run(self, n_steps: int, on_step=None, start_step: int = 0):
        self._pq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))
        self._rq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))

        def _guard(fn):
            """把线程目标包一层：异常 → 记录 + 停其他线程（让主循环快速失败）。

            C1：捕获 BaseException 而非 Exception——SystemExit/KeyboardInterrupt 等
            异常也可能出现在 worker 线程（vLLM/底层库），漏捕即永久 0% 空等。
            """
            def _wrapped(*a, **k):
                try:
                    fn(*a, **k)
                except BaseException as e:      # noqa: BLE001 —— 必须捕获全部线程异常
                    with self._err_lock:
                        if self._thread_error is None:
                            self._thread_error = e
                    self.stop.set()
            return _wrapped

        threads = [
            threading.Thread(target=_guard(self._rollout_collector), name="RolloutCollector"),
            threading.Thread(target=_guard(self._teacher_scorer), name="TeacherScorer"),
            threading.Thread(target=_guard(self._prompt_feeder), name="PromptFeeder"),
        ]
        for t in threads:
            t.start()

        self._train_dispatcher(n_steps, on_step=on_step, start_step=start_step)

        for t in threads:
            t.join(timeout=5)
        rollouts = self._n_rollout
        trained = len(self.metrics)
        stale_drops = self.staleness_q.n_rejected + self._n_dropped_consume
        qfull_drops = self._n_dropped_qfull
        # M5：停机尾段 = 已算完但主循环已到 n_steps 未及消费的 rollout（残差口径）。
        #   rollouts = trained + stale_drops + qfull_drops + shutdown_tail 恒等成立。
        shutdown_tail = max(0, rollouts - trained - stale_drops - qfull_drops)
        from collections import Counter
        ages = Counter(m["age"] for m in self.metrics)
        self.summary = {
            "rollout_forwards": rollouts,
            "trained_steps": trained,
            "dropped_at_put": self.staleness_q.n_rejected,     # 入队侧陈旧拒收
            "dropped_at_consume": self._n_dropped_consume,     # 消费侧陈旧截断
            "dropped_queue_full": qfull_drops,                 # 队列满丢弃
            "shutdown_tail": shutdown_tail,                    # 停机尾段（残差）
            "stale_discard_ratio": stale_drops / max(rollouts, 1),  # 纯陈旧丢弃率
            "waste_ratio": (rollouts - trained) / max(rollouts, 1),  # 总浪费（陈旧+队满+停机尾）
            "rollout_idle_s": round(self._rollout_idle, 2),
            "scorer_idle_s": round(self._scorer_idle, 2),
            "age_histogram": dict(sorted(ages.items())),   # {age: 步数}，兑现设计 §7.1
        }
        return self.metrics


# ============================================================================
#  GPU 部署骨架（2×RTX PRO 6000）：L5 Ray 多 worker + NCCL 权重广播，L2 TP=2 钩子
# ============================================================================
#  这些类把「线程 + Queue」换成真实的多进程/多卡协调，但**算法内核一行不动**：
#  _train_step（π_old 加权 PG + PPO clip + k3 KL + staleness 双截断）被直接复用。
#  默认（CPU demo）路径不触碰这些类；只在 cfg["distributed"]=True 时走它们的入口。
# ============================================================================


class WeightBroadcaster:
    """L5 · learner↔rollout 的**异步**权重同步（NCCL P2P，经 NVLink 极速）。

    替代线程版 WeightStore（acquire_if_newer + load_state_dict）。AsyncOPD 的
    rollout 与 learner **解耦**，二者步数天然不对齐——因此这里用**非阻塞 P2P
    (isend/irecv)** 而非集体 broadcast：learner 每步把最新权重+版本推给所有 worker
    rank（fire-and-forget，发送前先等上一轮完成避免队列堆积）；worker 在 rollout
    开始时拉取。若需要严格一致的同步场景，可用同步版 sync()（集体 broadcast）。

    若需把「推」与「训练」重叠（方案 L6 双缓冲），把 push_async 返回的 work 句柄
    交给一个后台流等待即可。
    """

    def __init__(self, pg=None, device=None):
        if not torch.distributed.is_available():
            raise RuntimeError("WeightBroadcaster 需要 torch.distributed(NCCL)")
        self.pg = pg              # None → 默认进程组（rank0=learner, rank1..W=worker）
        self.device = device
        self._inflight = []       # 上一轮 P2P 句柄，发送前等待避免堆积

    @property
    def rank(self) -> int:
        return torch.distributed.get_rank(self.pg)

    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size(self.pg)

    @property
    def worker_ranks(self):
        return [r for r in range(self.world_size) if r != 0]

    def push_async(self, model, version: int):
        """learner（rank0）侧：非阻塞把权重+版本 isend 给所有 worker rank。"""
        for h in self._inflight:               # 等上一轮推完，防止 P2P 队列堆积
            h.wait()
        self._inflight = []
        if self.rank != 0:
            return self._inflight
        dev = next(model.parameters()).device
        ver_t = torch.tensor([version], dtype=torch.long, device=dev)
        for w in self.worker_ranks:
            self._inflight.append(
                torch.distributed.isend(ver_t, dst=w, group=self.pg, tag=0))
        for i, p in enumerate(model.parameters()):
            t = p.data.contiguous()
            for w in self.worker_ranks:
                self._inflight.append(
                    torch.distributed.isend(t, dst=w, group=self.pg, tag=1 + i))
        return self._inflight

    def pull_async(self, model):
        """worker 侧：非阻塞从 rank0 irecv 权重+版本。返回 (handles, ver_buf, recv_bufs)。"""
        if self.rank == 0:
            return [], None, []
        dev = next(model.parameters()).device
        ver_buf = torch.empty(1, dtype=torch.long, device=dev)
        handles = [torch.distributed.irecv(ver_buf, src=0, group=self.pg, tag=0)]
        recv_bufs = []
        for i, p in enumerate(model.parameters()):
            buf = torch.empty_like(p.data, device=dev)
            handles.append(torch.distributed.irecv(buf, src=0, group=self.pg, tag=1 + i))
            recv_bufs.append((p, buf))
        return handles, ver_buf, recv_bufs

    @staticmethod
    def finalize_pull(handles, recv_bufs):
        for h in handles:
            h.wait()
        for p, buf in recv_bufs:
            p.data.copy_(buf)

    def sync(self, model):
        """同步版（集体 broadcast）：所有 rank 调用，rank0 为源。用于需要时强一致。"""
        for p in model.parameters():
            buf = p.data if self.rank == 0 else torch.empty_like(p.data, device=p.device)
            torch.distributed.broadcast(buf, src=0, group=self.pg)
            if self.rank != 0:
                p.data.copy_(buf)


def parallelize_learner_tp2(model, tp_size: int = 2, sp: bool = True):
    """L2 · Megatron **TP=2 + Sequence Parallel** 接入点（2×PRO6000 有 NVLink）。

    2 卡经 NVLink 通信极快，learner 首选 TP=2+SP（每层 all-reduce / reduce-scatter 快，
    比 FSDP 简洁）。统一环境已含 megatron-core 0.16.1。

    ⚠️ **关键**：nn.Linear **无法被「事后切分」**——ToyModel 不能由本函数变 TP。
    本函数从 toy 模型读出维度（vocab/d_model/n_layers），构造**同构的
    MegatronCausalToyLM**（用 ColumnParallelLinear / RowParallelLinear /
    VocabParallelEmbedding / RMSNorm 重建）。构造前必须在 launcher 调用一次：

        mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                      sequence_parallel=sp)

    返回的 megatron 模型与 CausalToyLM 同接口（forward / response_dists），且
    response_dists 在内部 all-gather 词表分片还原完整 (B,T,V)，Stage2 的
    π_old 加权 PG + PPO clip + k3 KL 内核**一行不动**。
    """
    if not _MEGATRON_AVAILABLE or MegatronCausalToyLM is None:
        raise RuntimeError(
            "Megatron-core 未安装（统一环境已含 megatron-core 0.16.1）。"
            "L2 需要把 learner 用 megatron 的 parallel 层重建，"
            "ToyModel 的 nn.Linear 不能被本函数切分。")
    if mpu is None or not mpu.model_parallel_is_initialized():
        raise RuntimeError(
            "L2 集成点：构造 MegatronCausalToyLM 前必须先在 launcher 调用一次 "
            "mpu.initialize_model_parallel(tensor_model_parallel_size=%d, "
            "sequence_parallel=%s) 建立 TP 组。ToyModel 无法被 parallelize_learner_tp2 "
            "事后切分。" % (tp_size, sp))
    vocab = getattr(model, "vocab", None)
    d_model = getattr(model, "d_model", None)
    n_layers = getattr(model, "n_layers",
                       len(getattr(model, "enc", list(())).layers))
    if vocab is None or d_model is None:
        raise RuntimeError("parallelize_learner_tp2: 输入的模型缺少 vocab/d_model 属性。")
    return MegatronCausalToyLM(
        vocab=vocab, d_model=d_model, n_layers=n_layers,
        n_head=getattr(model, "n_head", 4),
        max_len=getattr(model, "max_len", 64), sp=sp)


class _RayRolloutWorkerImpl:
    """L5 · 单卡 rollout worker（Ray actor 内部实现；外覆 ray.remote 装饰）。

    在独立进程/GPU 上跑「rollout + teacher-scoring」两阶段（抽取自线程版）：
      1) 通过 WeightBroadcaster.pull_async 从 learner 拉最新权重（NCCL P2P）；
      2) 用该快照对离线固定 rollout 重算 s_old（Lightning 设定，对齐教师 Δ_T）；
      3) 稀疏模式下只透传 idxs——Δ_T 由 learner 现场按 student 支撑展开。
    真实 rollout 用 vLLM TP=2（方案 L3）替换 response_dists；teacher 一致性 /
    稀疏 Δ_T 逻辑沿用 cache（L4）。
    """

    def __init__(self, worker_rank, world_size, master_addr, master_port,
                 prompts, responses, vocab, d_model, n_layers, cfg, device,
                 top_k_student, cache_top_k):
        if torch.distributed.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=worker_rank, world_size=world_size)
        self.device = device
        self.prompts = prompts.to(device)
        self.responses = responses.to(device)
        self.broadcaster = WeightBroadcaster()
        self.cfg = cfg
        self.top_k_student = top_k_student or cache_top_k
        self.n_prompts = prompts.size(0)

        if cfg.get("rollout_engine") == "vllm":
            # L3 · 真实 rollout：vLLM TP=2 引擎（PagedAttention + FP8）。
            # 权重经 NCCL 拉到 self._param_buf（与 learner 同构的占位参数容器），
            # 再由 vLLM 引擎按自身参数顺序重建并载入（verl 风格同构假设）。
            if VLLMRolloutEngine is None:
                raise RuntimeError("L3 需要 vllm；统一 GPU 环境应含 vllm。")
            self._param_buf = CausalToyLM(vocab=vocab, d_model=d_model,
                                          n_layers=n_layers).to(device)
            self.worker = VLLMRolloutEngine(
                # C4：与主路径对齐——model 回落 student_path（HF 离线不联网下载）、
                # tp 默认 1（单卡放置；多卡 TP 显式配置）。
                model=(cfg.get("rollout_model") or cfg.get("student_path")
                       or "Qwen/Qwen2.5-7B"),
                tp_size=int(cfg.get("rollout_tp_size", 1)),
                dtype=cfg.get("rollout_dtype", "auto"),
                vocab_size=vocab,
                full_logprobs_cap=int(cfg.get("rollout_logprobs_cap", 4096)),
                device=device)
        else:
            # 占位：ToyModel 朴素前向（本地 demo / 未启用 L3 时）
            self._param_buf = None
            self.worker = CausalToyLM(vocab=vocab, d_model=d_model,
                                      n_layers=n_layers).to(device)

    def rollout_and_score(self, idxs):
        # 1) 拉最新权重（非阻塞 P2P，NVLink 加速）—— vLLM 路径拉到占位容器再转引擎
        target = self._param_buf if self._param_buf is not None else self.worker
        handles, ver_buf, recv_bufs = self.broadcaster.pull_async(target)
        if handles:
            WeightBroadcaster.finalize_pull(handles, recv_bufs)
            if self._param_buf is not None:
                # L3 · 把拉到的扁平张量推入 vLLM 引擎（同名同构假设，见 rollout_vllm.py）
                self.worker.update_weights_from_flat(
                    [p.detach() for p in self._param_buf.parameters()])
        version = int(ver_buf.item()) if ver_buf is not None else 0
        # 2) 用当前快照重算 s_old（离线固定 rollout，Lightning 设定）
        #    vLLM 路径走 VLLMRolloutEngine.response_dists（接口对齐 (B,T,V)）
        idxs_dev = idxs.to(self.device)
        with torch.no_grad():
            if isinstance(self.worker, VLLMRolloutEngine):
                s_old = self.worker.response_dists(self.prompts[idxs_dev],
                                                   self.responses[idxs_dev])  # (B,T,V)
            else:
                self.worker.eval()
                s_old = response_dists(self.worker, self.prompts[idxs_dev],
                                       self.responses[idxs_dev], dtype=self.dtype)              # (B,T,V)
        # 3) 稀疏：只透传 idxs；Δ_T 在 learner 现场按 student 支撑展开
        return idxs.cpu(), s_old.cpu(), version


# ray 不可用时 RayRolloutWorker=None；不影响 CPU demo 导入
RayRolloutWorker = (ray.remote(num_gpus=1)(_RayRolloutWorkerImpl)
                    if ray is not None else None)


class DistAsyncScheduler(AsyncBatchedScheduler):
    """L5 · 2 卡（2×RTX PRO 6000）分布式异步调度器骨架。

    结构：learner（驱动进程，rank0）跑 _train_step；N 个 Ray actor 各占 1 卡跑
    rollout + teacher-scoring；权重经 WeightBroadcaster（NCCL P2P，NVLink 加速）
    异步推送。四阶段异步 + staleness 双截断 + π_old 加权 PG + k3 KL（即 _train_step）
    **完全复用**线程版内核，只把「线程 + Queue」换成「Ray actor + NCCL 权重广播」。

    L2 钩子：若 cfg["tp_size"] > 1，learner 走 Megatron TP=2+SP（见 parallelize_learner_tp2）。

    ⚠️ 骨架：进程组需由启动器建立（rank0=驱动, rank1..W=worker）。真实 rollout
    用 vLLM TP=2（L3）替换 response_dists；colocated 的 learner↔rollout 权重换入
    换出走 CPU offload（L6，已在 WeightStore 就绪）。
    """

    def __init__(self, student, cache, prompts, responses, ref_dists, ref_ids,
                 ref_logp, cfg, device, dist_rank, dist_world_size,
                 master_addr, master_port):
        super().__init__(student, cache, prompts, responses, ref_dists,
                         ref_ids, ref_logp, cfg, device)
        self._ver = 0
        self.dist_rank = dist_rank
        self.dist_world_size = dist_world_size

        # L2 护栏：Megatron TP=2 learner 与本调度器的并发 rollout-worker 模型【架构互斥】。
        #   原因：tp_size=2 的 learner 其 forward 里的 TP 集合通信（all-gather/reduce-scatter）
        #   需要 rank0+rank1 协同执行；但本类把 rank1..W 派作 Ray rollout worker（rollout 循环里
        #   永不进 learner forward）→ TP 集合通信死锁。且 sharded learner 经 WeightBroadcaster
        #   广播 model.parameters() 只含 rank0 的那一半，rollout worker 拿到的是残缺权重。
        #   2×PRO6000 的 L2 正确形态是「colocated 交替相位」：learner TP=2 与 rollout vLLM TP=2
        #   同驻 2 卡、按相位交替 + CPU offload 换入换出（见 OPTIMIZATION_PLAN_2xRTXPRO6000），
        #   而非这里的并发 learner/rollout 分卡。该交替调度器尚未实现。
        if int(cfg.get("tp_size", 1)) > 1:
            raise RuntimeError(
                "L2 (learner Megatron TP=2 跨双卡) 与 DistAsyncScheduler 的并发 "
                "rank1-as-rollout-worker 模型互斥：learner 的 TP 集合通信需要 rank1 协同，"
                "而 rank1 被派作 rollout worker 会在 all-gather 上死锁；且 sharded learner 的"
                "权重广播只含一半参数。请改用 colocated 交替相位调度（learner TP=2 ↔ rollout "
                "vLLM TP=2 同驻双卡 + CPU offload），该调度器待实现。MegatronCausalToyLM / "
                "parallelize_learner_tp2 可在该编排下复用；tp_size=1 时本调度器不受影响。")

        # NCCL 权重广播器（默认组：rank0=learner, rank1..W=worker）
        self.broadcaster = WeightBroadcaster()

        # 派生 W 个 Ray rollout worker（每卡 1 个）
        W = dist_world_size - 1
        if RayRolloutWorker is None:
            raise RuntimeError("L5 需要 ray；统一环境未含，请 pip install ray")
        self.workers = [
            RayRolloutWorker.remote(
                worker_rank=i + 1, world_size=dist_world_size,
                master_addr=master_addr, master_port=master_port,
                prompts=prompts.cpu(), responses=responses.cpu(),
                vocab=student.vocab, d_model=student.d_model,
                n_layers=student.n_layers,
                cfg=cfg, device="cuda:0",
                top_k_student=self.top_k_student, cache_top_k=cache.top_k,
            )
            for i in range(W)
        ]
        self._rr = 0
        self.prefetch = cfg.get("prefetch", 4)

    def _publish(self):
        """覆盖父类：用 NCCL P2P 广播最新权重（替代 WeightStore 的 load_state_dict）。"""
        self._ver += 1
        self.broadcaster.push_async(self.student, self._ver)
        self.staleness_q.advance_version()
        return self._ver

    def _rand_idxs(self):
        return torch.randint(0, self.n_prompts, (self.batch,))

    def run(self, n_steps: int):
        """异步训练主循环：用 ray future 流水线 rollout worker，避免阻塞。"""
        pending = []
        done = 0
        while done < n_steps:
            # 喂 prompt（非阻塞，维持 prefetch 个在途 rollout）
            while len(pending) < self.prefetch:
                idxs = self._rand_idxs()
                w = self.workers[self._rr % len(self.workers)]
                self._rr += 1
                pending.append(w.rollout_and_score.remote(idxs))
            # 收一个就绪的 rollout → 训练（权重版本随样本回传，做 staleness 截断）
            ready, pending = ray.wait(pending, num_returns=1)
            idxs, s_old, ver = ray.get(ready[0])
            m = self._train_step(done, idxs, s_old, None, ver)
            if m is None:
                continue
            self.metrics.append(m)
            done += 1
        for w in self.workers:
            ray.kill(w)
        return self.metrics


def launch_distributed_scheduler(student, cache, prompts, responses, ref_dists,
                                 ref_ids, ref_logp, cfg, master_addr="127.0.0.1",
                                 master_port=29500, n_gpus: int = 2):
    """L5 启动器骨架：驱动(rank0) + W 个 Ray worker(rank1..W) 组成 torch.distributed 组。

    真实集群用 torchrun --nproc_per_node=... 或 ray 集群启动；此处给单节点 2 卡骨架。
    ⚠️ 需按集群调整 master_addr/master_port 与 CUDA_VISIBLE_DEVICES；
       真实 rollout 用 vLLM TP=2（L3）、learner 可切 Megatron TP=2（L2, cfg["tp_size"]）。
    """
    if not torch.distributed.is_available():
        raise RuntimeError("launch_distributed_scheduler 需要 torch.distributed(NCCL)")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", init_method=f"tcp://{master_addr}:{master_port}",
            rank=0, world_size=n_gpus)
    sched = DistAsyncScheduler(
        student, cache, prompts, responses, ref_dists, ref_ids, ref_logp,
        cfg, device="cuda:0", dist_rank=0, dist_world_size=n_gpus,
        master_addr=master_addr, master_port=master_port)
    return sched.run(cfg.get("n_steps", 30))

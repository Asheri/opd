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

import queue
import threading

import torch

from .buffer import StalenessQueue, WeightStore
from .losses import pg_loss, low_var_kl, low_var_kl_support, expected_reward
from .model import CausalToyLM, response_dists

# pg_loss 失配屏蔽阈值（log 空间）：s_old < -20 视为支撑外 log0 近似（π_old≈0），贡献=0。
# 正常 s_old 最小值 ≈ -ln V（V=128k 时约 -11.5），10 以内的余量不误伤；_LOG_ZERO=-30 闭合。
LOG_RATIO_MAX = 20.0

# 队列操作超时（秒）：put 满 / get 空时的轮询间隔，平衡吞吐与线程响应
_PUT_TIMEOUT = 0.5        # 入队侧（prompt→rollout、rollout→scorer、scorer→staleness_q）
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

        # colocated CPU offload 钩子（L6）：rollout 阶段把 learner 权重换出到 CPU
        self.offload_to_cpu = bool(cfg.get("offload_to_cpu", False))

        self.staleness_q = StalenessQueue(cfg.get("staleness_threshold", 4))
        self.weight_store = WeightStore(offload_to_cpu=self.offload_to_cpu)
        self.stop = threading.Event()
        self.metrics: list = []
        # P2-1：只观测计数器（不改控制流）
        self._n_rollout = 0          # rollout 实际前向次数
        self._n_dropped_consume = 0  # 消费侧因过旧丢弃数
        self._n_dropped_qfull = 0    # 队列满丢弃数（rollout→scorer 或 scorer→staleness_q）
        self._rollout_idle = 0.0     # RolloutCollector 累计空转秒
        self._scorer_idle = 0.0      # TeacherScorer 累计空转秒

        # 优化器 + 超参提升为实例属性，供 _train_step 在两种调度器间复用
        self.kl_coef = cfg.get("kl_reg_coef", 0.05)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.use_topk = (self.top_k_student > 0) and (cache.mode == "topk")
        self.opt = self._build_optimizer(student, cfg)

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

        # L3 · vLLM rollout 引擎（可选）：包成 response_dists 的 drop-in 替换。
        # 提供时，rollout 阶段用 vLLM TP=2 推理取代 self.worker 的朴素前向；
        # self.worker（ToyModel 副本）仅作权重对照，无需常驻显存。
        self.rollout_engine = rollout_engine
        if self.rollout_engine is not None and not vllm_available():
            raise RuntimeError("cfg 指定 vLLM rollout 但 vllm 未安装（统一 GPU 环境应含）。")
        if self.rollout_engine is None:
            # 旧快照前向模型：toy → CausalToyLM 副本；hf → 同 student 路径再加载一份
            # HFCausalLM（权重随后经 weight_store 快照覆盖）。⚠️ hf 骨架：需 GPU 验证。
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
        else:
            self.worker = None
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
                if self.rollout_engine is not None:
                    # L3 · vLLM 路径：把新权重推入 vLLM 引擎（取代 load_state_dict）
                    self.rollout_engine.update_weights(snap)
                else:
                    self.worker.load_state_dict(snap)
                self._loaded_ver = ver
            self._n_rollout += 1
            idxs_dev = idxs.to(self.device)
            if self.rollout_engine is not None:
                # L3 · vLLM TP=2 推理：response_dists 接口对齐，返回 (B,T,V) 落设备
                with torch.no_grad():
                    s_old = self.rollout_engine.response_dists(
                        self.prompts[idxs_dev], self.responses[idxs_dev])
            else:
                self.worker.eval()
                with torch.no_grad():
                    s_old = response_dists(self.worker, self.prompts[idxs_dev],
                                           self.responses[idxs_dev])     # (B,T,V) device
            try:
                self._rq.put((idxs, s_old, self._loaded_ver), timeout=_PUT_TIMEOUT)
            except queue.Full:
                self._rollout_idle += 0.5      # 因队列满等待 0.5s
                self._n_dropped_qfull += 1     # M5：已算完的 rollout 因队满被丢弃
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
        # 陈旧度截断（消费侧，与 v1 审阅修复一致）
        if self.staleness_q.current_version - ver > threshold:
            return None
        self.student.train()
        idxs_dev = idxs.to(self.device)
        p_b = self.prompts[idxs_dev]
        r_b = self.responses[idxs_dev]

        # learner 时刻用【当前】student 重算分布（recompute 代理）。
        # bf16 自动混合精度（L1）：包住前向 + 损失 + 反向（bf16 有范围，无需 GradScaler）。
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype,
                                enabled=self.amp):
            s_cur = self.student.response_dists(p_b, r_b)      # (B,T,V) 带梯度
            # 与 s_cur 同设备同精度，保证 ratio 一致（M2：分布式路径 s_old 由 worker 进程
            # 回传在 CPU，此前只转 dtype 不转 device → s_cur-s_old 设备不匹配崩溃）。
            s_old = s_old.to(self.device, dtype=s_cur.dtype)
            # P2-2：缓存 p_old 供 pg_loss 与 adv 监控复用，省掉 expected_reward 内部那次 s_old.exp()；全 1 mask 走 None 快路径
            # ⚠️ mask 快路径前提：调度器无 padding（responses 等长），恒全 1。
            #    若将来引入真实 padding mask 必须改回传 mask。
            p_old = s_old.exp()

            # ★ Direct-OPD 迁移对象：按 student 自身 top-K 支撑取 Δ_T（L4 稀疏缓存）
            if self.use_topk:
                s_topk = torch.topk(s_cur, self.top_k_student, dim=-1)
                # 方案 A（对齐论文）：展开维度按 student 词表（7B=152064 > teacher 151936），
                # student 超出 teacher 词表的 top-K id 在 searchsorted 未命中 → Δ=0。
                delta_d = self.cache.delta_for_student_topk(
                    idxs_dev, s_topk.indices,
                    vocab_out=getattr(self.student, "vocab", None))  # (B,T,V) 支撑外=0
                # P2-G（二次审查）：renormalize 时把【完整 student top-K】掩码显式传给
                # pg_loss，与 low_var_kl_support 的支撑（也是完整 student top-K）一致——
                # 否则 pg 用 delta!=0（student∩teacher 交集）归一、KL 用完整 top-K 归一，
                # 分母不同 → λ_kl 权衡漂移。未覆盖 token 的 Δ=0 贡献 0、但计入分母。
                pg_support = None
                if self.renormalize_topk:
                    pg_support = torch.zeros_like(delta_d, dtype=torch.bool)
                    pg_support.scatter_(-1, s_topk.indices, True)
                loss_pg = pg_loss(s_cur, s_old, delta_d, None, self.clip_eps, p_old=p_old,
                                  log_ratio_max=LOG_RATIO_MAX,
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
                loss_pg = pg_loss(s_cur, s_old, delta_d, None, self.clip_eps, p_old=p_old,
                                  log_ratio_max=LOG_RATIO_MAX, delta_clip=self.delta_clip)
                loss_kl = low_var_kl(s_cur, self.ref_dists[idxs_dev], None)

            loss = loss_pg + self.kl_coef * loss_kl

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.opt.step()

        version = self._publish()
        with torch.no_grad():
            reward = expected_reward(s_cur.detach(), delta_d, None, p_dists=s_cur.detach().exp()).mean()
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
        batch = rb.get(rb_idxs)
        p_b = self.prompts[batch["prompt_idx"]].to(self.device)
        r_b = batch["responses"].to(self.device)
        mask = batch["token_masks"].to(self.device)
        self.student.train()
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype,
                                enabled=self.amp):
            s_cur = self.student.response_dists(p_b, r_b)      # (B,T,V) 带梯度
            s_topk = torch.topk(s_cur, self.top_k_student, dim=-1)
            # Δ_T 展开到 s_cur top-K（教师 top-K 命中处取 delta_k，未命中=0）
            delta_at = rb.delta_at_student_topk(
                rb_idxs, s_topk.indices, self.device)          # (B,T,Ks)
            # 行为策略 s_old 展开到 s_cur top-K（未命中填 ref_tail_logp≈log 0）
            s_old_at = rb.s_old_at_student_topk(
                rb_idxs, s_topk.indices, self.device,
                tail_logp=self.ref_tail_logp)                  # (B,T,Ks)
            # KL 锚点：dense 模式从 ref_dists 取，稀疏模式从 ref_top-K 取（同 base 路径）
            if self.kl_mode == "dense":
                ref_at = self.ref_dists[batch["prompt_idx"]].gather(
                    -1, s_topk.indices)                        # (B,T,Ks)
            else:
                ref_at = self._ref_logp_at_student_topk(
                    batch["prompt_idx"], s_topk.indices)       # (B,T,Ks)
            # 支撑 = 完整 s_cur top-K（ones），与 KL 同源重归一（对齐原始 Direct-OPD）
            sup = torch.ones_like(s_topk.values, dtype=torch.bool)
            loss_pg = pg_loss(s_topk.values, s_old_at, delta_at, mask, self.clip_eps,
                              p_old=s_old_at.exp(), log_ratio_max=LOG_RATIO_MAX,
                              renormalize_support=True, support=sup,
                              delta_clip=self.delta_clip)
            loss_kl = low_var_kl_support(s_topk.values, ref_at, mask,
                                         renormalize_support=True)
            loss = loss_pg + self.kl_coef * loss_kl

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.opt.step()
        version = self._publish()
        with torch.no_grad():
            p_cur = s_topk.values.exp()
            reward = (p_cur * delta_at).sum(-1) * mask
            adv = (s_old_at.exp() * delta_at).sum(-1) * mask
            reward = reward.sum() / mask.sum()
            adv = adv.sum() / mask.sum()
        scalars = [loss, loss_pg, loss_kl, adv, reward]
        scalars = [s.float() if s.dtype != torch.float32 else s for s in scalars]
        loss_v, pg_v, kl_v, adv_v, rew_v = torch.stack(scalars).detach().cpu().tolist()
        # 标量从标量张量转 float（reward/adv 已是标量）
        return {
            "step": done,
            "version": version,
            "age": 0,
            "pool": "refresh",
            "batch": int(r_b.size(0)),
            "loss": loss_v,
            "pg_loss": pg_v,
            "kl_loss": kl_v,
            "adv_mean": float(adv_v),
            "reward": float(rew_v),
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
                except Exception:
                    pass
            done += 1
            completed += 1
        return completed

    def _train_dispatcher(self, n_steps: int, on_step=None, start_step: int = 0):
        done = start_step
        while done < start_step + n_steps:
            try:
                (idxs, s_old, delta), ver, _ = self.staleness_q.get(timeout=_DISPATCH_GET_TIMEOUT)
            except queue.Empty:
                continue
            m = self._train_step(done, idxs, s_old, delta, ver)
            if m is None:
                self._n_dropped_consume += 1   # 消费侧因过旧丢弃
                continue
            self.metrics.append(m)
            if on_step is not None:            # T8：每成功一步回调（checkpoint/metrics 用）
                try:
                    on_step(m)
                except Exception:
                    pass                       # 回调异常不影响训练
            done += 1
        self.stop.set()

    # --------------------------- 入口 ---------------------------
    def run(self, n_steps: int, on_step=None, start_step: int = 0):
        self._pq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))
        self._rq: "queue.Queue" = queue.Queue(maxsize=self.cfg.get("queue_size", 8))

        threads = [
            threading.Thread(target=self._rollout_collector, name="RolloutCollector"),
            threading.Thread(target=self._teacher_scorer, name="TeacherScorer"),
            threading.Thread(target=self._prompt_feeder, name="PromptFeeder"),
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
                model=cfg.get("rollout_model", "Qwen/Qwen2.5-7B"),
                tp_size=int(cfg.get("rollout_tp_size", 2)),
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
                                       self.responses[idxs_dev])              # (B,T,V)
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

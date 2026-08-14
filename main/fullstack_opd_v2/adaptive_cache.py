"""L2 Adaptive Staleness-Aware Teacher Cache（§3-§6 + §13 整合）。

全部新增逻辑集中于此，cache.py/scheduler.py 只做薄扩展（最小侵入）。
单向依赖（§13.1，禁止循环修改彼此内部状态）：
  RefreshSelector -> DisagreementComputer -> RefreshRingBuffer
  -> CacheHealthMonitor -> DynamicRatioController -> Feeder

设计原则：可解释、可监控、compute overhead 可控、可 ablation、不破坏训练。
所有类经 l2.enabled 总开关控制，关闭时 pipeline 退回 L0/L1 静态路径。

本文件当前实现（任务 2.1-2.3 + 3.1 + 4.1）：
- `RefreshRingBuffer`：L2 refresh pool 动态 ring buffer（§2 双池结构；run_refresh_phase 依赖 append）。
- `DisagreementComputer`：§3 Teacher-Student Disagreement（rollout 阶段计算，_train_step 保持 teacher-free）。
- `run_refresh_phase` / `_build_mask`：§3.3 + §6.5 rollout 相位编排（student 生成 -> 4 logp -> D_i^abs -> append）。
- `CacheHealthMonitor`：§4 七维监控 + rule-based health score + alert cooldown（Observe-only，不自动改训练）。
- `DynamicRatioController`：§5 三信号 controller α（EMA + max_step_change + cold start + fixed/linear/adaptive）。
"""
from __future__ import annotations

import torch


class RefreshRingBuffer:
    """Refresh Pool 动态 ring buffer（§2 双池结构）。

    base 池（TensorTeacherCache 原 ids/delta_k）不动；refresh 池独立张量，
    append 进新样本，满后 FIFO 淘汰最旧，高 disagreement 样本价值保护免淘汰一轮。
    持久化字段：ids/delta_k（训练查表）+ generation_step/response_length/token_mask/disagreement_abs。

    索引：refresh 样本用局部 idx [0, size)；双池 feeder 负责 base/refresh 混合。
    """

    def __init__(self, capacity: int, top_k: int, vocab: int,
                 value_protect_quantile: float = 0.9):
        self.capacity = capacity
        self.top_k = top_k
        self.vocab = vocab
        self.value_protect_quantile = value_protect_quantile
        # ring buffer 槽位（预分配 capacity，append 原地写）
        self.ids: torch.Tensor | None = None        # (cap, T, K)
        self.delta_k: torch.Tensor | None = None    # (cap, T, K)
        self._gen_steps: list[int] = []             # 每槽 generation_step
        self._resp_lens: list[int] = []
        self._token_masks: list[torch.Tensor] = []
        self._disagreements: list[float] = []       # 价值保护用
        self._protected: list[bool] = []            # 价值保护标记
        self._write_pos = 0     # 环形写指针
        self.size = 0           # 当前有效样本数

    def _ensure_alloc(self, T: int, device, dtype):
        if self.ids is None:
            self.ids = torch.zeros(self.capacity, T, self.top_k,
                                   dtype=torch.long, device=device)
            self.delta_k = torch.zeros(self.capacity, T, self.top_k,
                                       dtype=dtype, device=device)

    def append(self, ids: torch.Tensor, delta_k: torch.Tensor,
               generation_step: int, response_length: int,
               token_mask: torch.Tensor, disagreement_abs: float):
        """append 一条样本（ids/delta_k: (T,K)）。满则 FIFO 淘汰（价值保护除外）。"""
        T = ids.size(0)
        self._ensure_alloc(T, ids.device, delta_k.dtype)
        # 满且当前写指针指向的样本非受保护 -> 淘汰；受保护则跳过该槽（顺延）
        if self.size >= self.capacity:
            # 找下一个非受保护槽淘汰
            for _ in range(self.capacity):
                if not self._protected[self._write_pos]:
                    break
                self._protected[self._write_pos] = False   # 保护只用一次
                self._write_pos = (self._write_pos + 1) % self.capacity
        pos = self._write_pos
        self.ids[pos] = ids
        self.delta_k[pos] = delta_k
        # 列表按 pos 索引（ring buffer 槽位复用）
        if pos < len(self._gen_steps):
            self._gen_steps[pos] = generation_step
            self._resp_lens[pos] = response_length
            self._token_masks[pos] = token_mask
            self._disagreements[pos] = disagreement_abs
            self._protected[pos] = False
        else:
            self._gen_steps.append(generation_step)
            self._resp_lens.append(response_length)
            self._token_masks.append(token_mask)
            self._disagreements.append(disagreement_abs)
            self._protected.append(False)
        # 价值保护：高于分位的样本标记
        if disagreement_abs > self._value_threshold():
            self._protected[pos] = True
        self._write_pos = (pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _value_threshold(self) -> float:
        """当前 disagreement 的价值保护分位（无样本时 inf）。"""
        if not self._disagreements:
            return float("inf")
        return float(torch.tensor(self._disagreements).quantile(
            self.value_protect_quantile).item())

    def get(self, idxs: torch.Tensor) -> dict:
        """(B,) 局部 idx -> {ids, delta_k, gen_steps, resp_lens, token_masks, disagreements}。"""
        if self.size == 0:
            T = self.ids.size(1) if self.ids is not None else 0
            K = self.top_k
            return {"ids": torch.empty(0, T, K, dtype=torch.long),
                    "delta_k": torch.empty(0, T, K)}
        return {
            "ids": self.ids[idxs],
            "delta_k": self.delta_k[idxs],
            "gen_steps": [self._gen_steps[i] for i in idxs.tolist()],
            "resp_lens": [self._resp_lens[i] for i in idxs.tolist()],
            "token_masks": torch.stack([self._token_masks[i] for i in idxs.tolist()]),
            "disagreements": [self._disagreements[i] for i in idxs.tolist()],
        }

    def mean_disagreement(self) -> float:
        """池内平均 disagreement（§5 刷新质量信号 / §4 监控）。"""
        if not self._disagreements:
            return 0.0
        return float(sum(self._disagreements) / len(self._disagreements))


class DisagreementComputer:
    """§3 Teacher-Student Disagreement（rollout 阶段计算，_train_step 保持 teacher-free）。

    token-level: D_t = [logπ_T^RL(y_t) − logπ_T^Ref(y_t)] − [logπ_S(y_t) − logπ_Ref(y_t)]
    response-level: D_i^abs = Σ_t m_t|D_t| / Σ_t m_t （主指标）
                    D_i^mean = Σ_t m_t D_t / Σ_t m_t

    同源性：teacher_ref(R1-Distill-1.5B) ≠ student ref(初始 Qwen3-1.7B)，
    故须分别存 4 个 logp，不能用 Δ_T−Δ_S 简化式。
    所有 logp 均为 chosen-token logp（gather 自生成 response，非 top-k 概率支撑）。
    """

    def compute(self, teacher_rl_logp: torch.Tensor, teacher_ref_logp: torch.Tensor,
                student_logp: torch.Tensor, student_ref_logp: torch.Tensor,
                mask: torch.Tensor) -> dict:
        """四个 (B,T) chosen-token logp + (B,T) mask -> {abs: (B,), mean: (B,)}。

        mask: 有效 token（含 EOS，不含其后的 padding）。
        """
        delta_t = (teacher_rl_logp - teacher_ref_logp)   # Δ_T^{(t)}
        delta_s = (student_logp - student_ref_logp)      # Δ_S^{(t)}
        d_t = delta_t - delta_s                          # D_t（不同源完整式）
        m = mask.float()
        denom = m.sum(-1) + 1e-8
        return {
            "abs": (m * d_t.abs()).sum(-1) / denom,      # D_i^abs（主指标）
            "mean": (m * d_t).sum(-1) / denom,           # D_i^mean
        }

    def gather_chosen_logp(self, log_dists: torch.Tensor,
                           responses: torch.Tensor) -> torch.Tensor:
        """(B,T,V) log-softmax -> (B,T) chosen-token logp（gather）。

        rollout 阶段 teacher 前向产生 (B,T,V)，用此取 chosen token 的精确 logp。
        禁止把"未进 top-k"当概率 0（§3.4）。
        """
        return log_dists.gather(2, responses.unsqueeze(-1)).squeeze(-1)


def run_refresh_phase(student, teacher_rl, teacher_ref, student_ref,
                      selector, ring_buffer, disag, prompts, step, version,
                      m_selected, max_resp_len, top_k, device):
    """§3.3 + §6.5 rollout 相位：selective 选 prompt -> student 生成
    -> 4 个 chosen logp -> D_i^abs -> append_refresh。teacher 前向在此（_train_step 不动）。

    返回 refresh 样本数。per-token logp 算完即弃（§11），只存标量。
    """
    from .model import generate_batch, token_logprobs, response_dists
    cand = selector.select(m_selected, prompts.size(0)) if selector else \
        torch.randint(0, prompts.size(0), (m_selected,))
    p_b = prompts[cand].to(device)
    responses = generate_batch(student, p_b, max_new=max_resp_len)
    student_logp = token_logprobs(student, p_b, responses)        # (M,T)
    with torch.no_grad():
        student_ref.eval()
        ref_logp = token_logprobs(student_ref, p_b, responses)
        rl_dist = response_dists(teacher_rl, p_b, responses)      # (M,T,V)
        ref_dist = response_dists(teacher_ref, p_b, responses)
        rl_chosen = disag.gather_chosen_logp(rl_dist, responses)
        ref_chosen = disag.gather_chosen_logp(ref_dist, responses)
        tk = rl_dist.topk(top_k, dim=-1)
        ids_k, rl_k = tk.indices, tk.values
        delta_k = rl_k - ref_dist.gather(-1, tk.indices)
    mask = _build_mask(responses)    # EOS 之后 padding=0（§3.4）
    D = disag.compute(rl_chosen, ref_chosen, student_logp, ref_logp, mask)
    for i in range(responses.size(0)):
        ring_buffer.append(ids_k[i], delta_k[i], generation_step=step,
            response_length=int(mask[i].sum()), token_mask=mask[i],
            disagreement_abs=float(D["abs"][i].detach()))
    return responses.size(0)


def _build_mask(responses, pad_id=0):
    """§3.4：EOS 计入（mask=1），其后的 padding 不计入（mask=0）。
    toy 等长时全 1；真实变长按 pad_token_id 判定（首个 pad 之后置 0）。"""
    # 简化：找到每行首个 pad，其后置 0（真实场景用 tokenizer.pad_token_id）
    is_pad = (responses == pad_id)
    # 首个 pad 之后全 pad
    cum = is_pad.cumsum(dim=1)
    mask = (cum <= 1) | (~is_pad)   # 首个 pad 位置仍算有效（EOS 场景需按实际 EOS id）
    return mask.long()


class CacheHealthMonitor:
    """§4 Cache Health Monitor（只 Observe->Diagnose，不自动改训练）。

    七维监控（Base/Refresh 分开）+ rule-based health score + alert cooldown。
    性能约束：batch-level aggregation，不逐 token loop，不全量扫 base pool，
    用 counters/EMA/reservoir。经 MetricsRecorder.record() 加字段落盘。
    """

    def __init__(self, health: dict, alert_cooldown: int = 50):
        self.thresholds = health
        self.alert_cooldown = alert_cooldown
        self.last_status = "HEALTHY"
        self._last_alert_step = -1
        self._alert_count = 0
        # counters（EMA/reservoir，不全量扫描）
        self._lookup = {"total": 0, "hit": 0, "miss": 0, "invalid": 0, "duplicate": 0}
        self._reuse_counts: dict[int, int] = {}   # sample_id -> 次数

    def record_lookup(self, hit: bool, invalid: bool = False, duplicate: bool = False):
        self._lookup["total"] += 1
        if hit:
            self._lookup["hit"] += 1
        else:
            self._lookup["miss"] += 1
        if invalid:
            self._lookup["invalid"] += 1
        if duplicate:
            self._lookup["duplicate"] += 1

    def record_reuse(self, sample_id: int):
        self._reuse_counts[sample_id] = self._reuse_counts.get(sample_id, 0) + 1

    def classify(self, hit_rate: float = 1.0, refresh_age_p95: float = 0,
                 reuse_p95: float = 0, max_length_ratio: float = 0) -> str:
        """rule-based 三级（§4.3）。任一 critical -> CRITICAL；任一 warning -> WARNING。"""
        worst = "HEALTHY"
        for metric, val in [("hit_rate", hit_rate), ("refresh_age_p95", refresh_age_p95),
                            ("reuse_p95", reuse_p95), ("max_length_ratio", max_length_ratio)]:
            th = self.thresholds.get(metric, {})
            # 未配置阈值的指标不参与判定（视为 HEALTHY）
            if not th:
                continue
            # hit_rate 越低越坏；其余越高越坏
            bad = (val < th["critical"]) if metric == "hit_rate" else (val > th["critical"])
            warn = (val < th["warning"]) if metric == "hit_rate" else (val > th["warning"])
            if bad:
                return "CRITICAL"
            if warn:
                worst = "WARNING"
        return worst

    def record(self, step: int, **metrics) -> dict:
        """聚合 + 状态分类 + alert cooldown。返回待 record 的指标 dict。"""
        status = self.classify(
            hit_rate=metrics.get("hit_rate", 1.0),
            refresh_age_p95=metrics.get("refresh_age_p95", 0),
            reuse_p95=metrics.get("reuse_p95", 0),
            max_length_ratio=metrics.get("max_length_ratio", 0))
        reason = self._reason(status, metrics)
        # alert cooldown：同 status 在 cooldown 内不重复记
        if status != "HEALTHY" and (step - self._last_alert_step) < self.alert_cooldown \
                and status == self.last_status:
            pass
        else:
            self._alert_count += 1
            self._last_alert_step = step
        self.last_status = status
        return {**metrics, "cache_health/status": status,
                "cache_health/reason": reason,
                **{f"lookup/{k}": v for k, v in self._lookup.items()},
                "lookup/hit_rate": self._lookup["hit"] / max(1, self._lookup["total"])}

    def _reason(self, status, metrics) -> str:
        if status == "HEALTHY":
            return ""
        # 找首个触发的指标
        for m in ["hit_rate", "refresh_age_p95", "reuse_p95", "max_length_ratio"]:
            th = self.thresholds.get(m, {})
            if not th:
                continue
            val = metrics.get(m, 0)
            bad = (val < th["critical"]) if m == "hit_rate" else (val > th["critical"])
            if bad:
                return f"{m} critical ({val})"
        return "unknown"


class DynamicRatioController:
    """§5 Dynamic Refresh Ratio（三信号 controller）。

    α_t = clip(α_0 + λA·Ã_B − λD·D̃_drift + λQ·Q̃_t, α_min, α_max)
    所有信号 normalize（EMA + x/(1+|x|) 映射），EMA 平滑防震荡；max_step_change 限幅。
    模式：fixed(α=initial) / linear(0.1->0.5) / adaptive(完整 controller)。
    cold start：N_R 不足时 α_actual=min(α, N_R/N_batch) fallback base（§5.5）。
    α_max<1：保留 base 作 stationary anchor。

    纯无状态函数（update 内部推进 step/EMA），不读 CacheHealthMonitor 状态，
    由 pipeline 在交替相位显式把监控指标喂进来（consume metrics，非 Monitor 闭环）。
    """

    def __init__(self, initial=0.30, min=0.10, max=0.60, mode="adaptive",
                 age_weight=0.25, drift_weight=0.50, quality_weight=0.25,
                 ema_beta=0.9, warmup_steps=500, max_step_change=0.05):
        self.alpha0 = initial
        self.min, self.max = min, max
        self.mode = mode
        self.w = dict(age=age_weight, drift=drift_weight, quality=quality_weight)
        self.beta = ema_beta
        self.warmup = warmup_steps
        self.max_step = max_step_change
        self._ema = dict(age=0.0, drift=0.0, quality=0.0)
        self._step = 0
        self._last_alpha = initial
        # linear 模式起止（§5.6）
        self._lin_start, self._lin_end = 0.1, 0.5

    def _norm(self, key, x):
        """EMA + 简单 normalize（x/(1+|x|) 映射到 [-1,1] 附近，防极值爆炸）。"""
        self._ema[key] = self.beta * self._ema[key] + (1 - self.beta) * x
        return self._ema[key] / (1 + abs(self._ema[key]))

    def update(self, base_age, policy_drift, refresh_quality) -> float:
        """推进一步，返回本轮 α（三信号 or 按模式降级）。"""
        self._step += 1
        if self.mode == "fixed":
            return self.alpha0
        if self.mode == "linear":
            frac = min(1.0, self._step / 1000)
            return self._lin_start + frac * (self._lin_end - self._lin_start)
        # adaptive：warmup 内用 initial（§5.5 cold start）
        if self._step <= self.warmup:
            self._last_alpha = self.alpha0
            return self.alpha0
        a_b = self._norm("age", base_age)
        d_drift = self._norm("drift", policy_drift)
        q = self._norm("quality", refresh_quality)
        raw = self.alpha0 + self.w["age"] * a_b - self.w["drift"] * d_drift \
            + self.w["quality"] * q
        raw = max(self.min, min(self.max, raw))
        # max_step_change 限幅（§5.4）
        raw = self._last_alpha + max(-self.max_step, min(self.max_step, raw - self._last_alpha))
        raw = max(self.min, min(self.max, raw))
        self._last_alpha = raw
        return raw

    def cold_start_adjust(self, alpha, n_refresh, n_batch) -> float:
        """§5.5：refresh 不足时 α_actual=min(α, N_R/N_batch)。"""
        return min(alpha, n_refresh / max(1, n_batch))
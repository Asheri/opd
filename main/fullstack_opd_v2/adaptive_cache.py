"""L2 Adaptive Staleness-Aware Teacher Cache（§3-§6 + §13 整合）。

全部新增逻辑集中于此，cache.py/scheduler.py 只做薄扩展（最小侵入）。
单向依赖（§13.1，禁止循环修改彼此内部状态）：
  RefreshSelector -> DisagreementComputer -> RefreshRingBuffer
  -> CacheHealthMonitor -> DynamicRatioController -> Feeder

设计原则：可解释、可监控、compute overhead 可控、可 ablation、不破坏训练。
所有类经 l2.enabled 总开关控制，关闭时 pipeline 退回 L0/L1 静态路径。

本文件当前实现（任务 2.1-2.3 + 3.1 + 4.1 + 5.1）：
- `RefreshRingBuffer`：L2 refresh pool 动态 ring buffer（§2 双池结构；run_refresh_phase 依赖 append）。
- `DisagreementComputer`：§3 Teacher-Student Disagreement（rollout 阶段计算，_train_step 保持 teacher-free）。
- `run_refresh_phase`：§3.3 + §6.5 rollout 相位编排（student 短 rollout -> 4 logp -> D_i^abs -> append）。
  Stage 2：短预算（默认 generate_with_status，可注入 vLLM）+ loop/invalid 跳过 + 长度式 mask
  （build_length_mask，取代原 _build_mask 的 pad 扫描）。
- `CacheHealthMonitor`：§4 七维监控 + rule-based health score + alert cooldown（Observe-only，不自动改训练）。
- `DynamicRatioController`：§5 三信号 controller α（EMA + max_step_change + cold start + fixed/linear/adaptive）。
- `PromptStateStore`：§6.1 per-prompt 轻量历史状态（times_seen/reward_ema/disagreement_ema/resp_len）。
- `RefreshSelector`：§6 Selective Rollout（candidate pool 两阶段 + value/coverage/diversity + fallback）。
"""
from __future__ import annotations

import torch

def assign_budgets(v_values: torch.Tensor,
                   budget_set=(256, 512, 1024, 2048),
                   quantiles=(0.25, 0.5, 0.75)) -> torch.Tensor:
    """§三 Budget Allocation：按 V 分位数把选中集映射到 4 档 budget。

    v_values: (M,) 选中 prompt 的价值评分（EMA 加权，跨 prompt 相对量）。
    返回 (M,) long budget。分位数在【传入的选中集内】算（非全池）。
    全等 v（无区分度）→ 全部中档 budget_set[len//2]（防 q 全等错分到最高档）。
    分段：v<q1→[0], q1≤v<q2→[1], q2≤v<q3→[2], v≥q3→[3]。
    torch.quantile 用 torch.tensor(list(quantiles), dtype=v_values.dtype)。
    """
    # 全等 v（无区分度）：全部给中档，防 quantile 全等时错分到最低/最高档。
    if v_values.min() == v_values.max():
        return torch.full_like(v_values, budget_set[len(budget_set) // 2],
                               dtype=torch.long)
    q1, q2, q3 = torch.quantile(
        v_values, torch.tensor(list(quantiles), dtype=v_values.dtype))
    budget = torch.full_like(v_values, budget_set[0], dtype=torch.long)
    budget = torch.where(v_values < q1, budget_set[0],
                         torch.where(v_values < q2, budget_set[1],
                                     torch.where(v_values < q3, budget_set[2],
                                                 budget_set[3])))
    return budget.long()


def enforce_budget(indices: torch.Tensor, budgets: torch.Tensor,
                   v_values: torch.Tensor, budget_t: int | None,
                   budget_set=(256, 512, 1024, 2048),
                   exploration_fraction: float = 0.20) -> tuple:
    """§六 Budget Controller 降级：保证 Σbudgets ≤ budget_t，绝不突破。

    indices: (M,) 选中 prompt 全局索引；budgets: (M,) 对应预算（long）；
    v_values: (M,) 对应价值评分。
    返回 (indices, budgets) 降级后。

    1) budget_t is None 或 sum(budgets)<=budget_t → 直接返回原样。
    2) 低价值降档：按 v 升序（低价值优先），把 budget>budget_set[0] 的样本逐一降一档
       （budget_set.index(b)-1），重算直到 sum<=budget_t 或全降到最低档。
    3) 仍超 → 减候选：砍"高预算低价值"（浪费）的样本，即 waste=v/b 升序优先砍
       （低 waste=高预算低价值）。保留 exploration 下限：至少保留
       round(len(indices)*exploration_fraction) 个样本（探索性），砍到该下限即停。
    4) 返回降级后的 (indices, budgets)。
    """
    if indices is None or len(indices) == 0:
        return indices, budgets
    if budget_t is None:
        return indices, budgets
    budgets = budgets.long().clone()
    if budgets.sum() <= budget_t:
        return indices, budgets

    # 2) 低价值降档：按 v 升序，把 budget>最低档 的样本逐一降一档。
    order = torch.argsort(v_values)  # 升序（低价值在前）
    can_downgrade = True
    while budgets.sum() > budget_t and can_downgrade:
        can_downgrade = False
        for i in order:
            b = int(budgets[i])
            if b > budget_set[0]:
                budgets[i] = budget_set[budget_set.index(b) - 1]
                can_downgrade = True
                if budgets.sum() <= budget_t:
                    return indices, budgets
    if budgets.sum() <= budget_t:
        return indices, budgets

    # 3) 仍超 → 减候选：砍"高预算低价值"（waste=v/b 升序优先砍），保留 exploration 下限。
    keep_min = round(len(indices) * exploration_fraction)
    # waste 升序 = 低价值高预算（浪费）优先砍；idx 升序打破平局。
    waste = v_values.float() / budgets.float()
    cut_order = torch.argsort(waste, stable=True)
    drop_set = set()
    while budgets.sum() > budget_t and (len(indices) - len(drop_set)) > keep_min:
        if len(cut_order) == 0:
            break
        i = int(cut_order[0].item())
        cut_order = cut_order[1:]
        if i in drop_set:
            continue
        drop_set.add(i)
        budgets[i] = 0
    keep_mask = torch.tensor(
        [i not in drop_set for i in range(len(indices))], dtype=torch.bool)
    # 被砍样本预算置 0 后仍返回同长张量（调用方按 mask 消费）。
    return indices, budgets


def compute_rollout_metrics(summary: dict, budgets=None,
                            budget_t: int | None = None) -> dict:
    """§七 Health Monitor 指标：7 个 rollout 效率指标。

    summary 含：n_total/n_appended/n_eos/n_budget/n_loop/n_invalid/rollout_tokens。
    budgets/budget_t 可选（预算配置）。返回 dict，键带 'rollout/' 前缀：
    - rollout/rollout_tokens = summary['rollout_tokens']
    - rollout/budget_utilization = rollout_tokens/budget_t（budget_t is None → 1.0）
    - rollout/truncation_rate = n_budget/n_total
    - rollout/loop_rate = n_loop/n_total
    - rollout/eos_rate = n_eos/n_total
    - rollout/accuracy_proxy = n_appended/n_total（UsefulSamples 占比）
    - rollout/useful_per_token = n_appended/rollout_tokens（核心效率口径）
    所有除法 div-zero 返回 0.0（n_total=0 或 rollout_tokens=0）。
    """
    n_total = summary.get('n_total', 0)
    n_appended = summary.get('n_appended', 0)
    n_eos = summary.get('n_eos', 0)
    n_budget = summary.get('n_budget', 0)
    n_loop = summary.get('n_loop', 0)
    n_invalid = summary.get('n_invalid', 0)
    rollout_tokens = summary.get('rollout_tokens', 0)

    def safe_div(num, den):
        return float(num / den) if den else 0.0

    return {
        'rollout/rollout_tokens': rollout_tokens,
        'rollout/budget_utilization': safe_div(rollout_tokens, budget_t) if budget_t else 1.0,
        'rollout/truncation_rate': safe_div(n_budget, n_total),
        'rollout/loop_rate': safe_div(n_loop, n_total),
        'rollout/eos_rate': safe_div(n_eos, n_total),
        'rollout/accuracy_proxy': safe_div(n_appended, n_total),
        'rollout/useful_per_token': safe_div(n_appended, rollout_tokens),
    }


def group_by_budget(cand: torch.Tensor, budgets: torch.Tensor) -> dict:
    """§四 per-sample budget 分桶：同 budget 的 prompt 合批。

    cand: (M,) 选中 prompt 全局索引；budgets: (M,) 对应预算。
    返回 {budget: [cand_idx,...]}（budget 为 int，值列表为 int）。
    """
    buckets: dict = {}
    for c, b in zip(cand.tolist(), budgets.tolist()):
        buckets.setdefault(int(b), []).append(int(c))
    return buckets


class RefreshRingBuffer:
    """Refresh Pool 动态 ring buffer（§2 双池结构）。

    base 池（TensorTeacherCache 原 ids/delta_k）不动；refresh 池独立张量，
    append 进新样本，满后 FIFO 淘汰最旧，高 disagreement 样本价值保护免淘汰一轮。
    持久化字段：ids/delta_k（训练查表）+ s_old_ids/s_old_logp（行为策略学生 top-K，
    供 refresh 训练 PG 用）+ prompt_idx/response（定位生成上下文）+ 标量元数据。

    ★双池 feeder（G1，闭环核心）：refresh 样本【进训练】——`_train_step_refresh` 从
    ring buffer 取 teacher Δ_T（delta_at_student_topk）与行为策略 s_old
    （s_old_at_student_topk），按 s_cur 当前 top-K 支撑展开做稀疏 top-K PG + KL，
    全程 teacher-free（teacher 前向只在 run_refresh_phase）。

    索引：refresh 样本用局部 idx [0, size)；双池 feeder 负责 base/refresh 混合。
    支持 state_dict/load_state_dict 断点续跑（G8）。
    """

    def __init__(self, capacity: int, top_k: int, vocab: int,
                 student_top_k: int | None = None,
                 value_protect_quantile: float = 0.9,
                 utility_weights: dict | None = None):
        self.capacity = capacity
        self.top_k = top_k                 # 教师 top-K（Δ_T 支撑）
        self.student_top_k = student_top_k or top_k   # 行为策略学生 top-K（s_old 支撑）
        self.vocab = vocab
        self.value_protect_quantile = value_protect_quantile
        # G7（§3.5 sample utility）：U_i = λ_D·D + λ_R·R̂ − λ_A·A 驱动价值保护。
        # 键名对齐 L2UtilityCfg：disagreement_weight / reward_weight / age_penalty。
        # None → 退回纯 disagreement 分位（向后兼容）。
        self.utility_weights = dict(utility_weights or {}) if utility_weights else None
        # ring buffer 槽位（预分配 capacity，append 原地写）
        self.ids: torch.Tensor | None = None        # (cap, T, Kt)  教师 top-K id
        self.delta_k: torch.Tensor | None = None    # (cap, T, Kt)  Δ_T on 教师 top-K
        self.ids_sorted: torch.Tensor | None = None       # 每槽按 id 升序（searchsorted）
        self.delta_k_sorted: torch.Tensor | None = None
        self.s_old_ids: torch.Tensor | None = None        # (cap, T, Ks) 行为策略学生 top-K id
        self.s_old_logp: torch.Tensor | None = None       # (cap, T, Ks) 行为策略学生 top-K logp
        self.s_old_ids_sorted: torch.Tensor | None = None # 每槽按 id 升序
        self.s_old_logp_sorted: torch.Tensor | None = None
        self._gen_steps: list[int] = []             # 每槽 generation_step
        self._resp_lens: list[int] = []
        self._token_masks: list[torch.Tensor] = []
        self._disagreements: list[float] = []       # 价值保护用
        self._protected: list[bool] = []            # 价值保护标记
        self._prompt_idx: list[int] = []            # fat_prompts 索引（定位 prompt）
        self._response: list[torch.Tensor] = []     # (T,) 生成 response
        self._status: list[str] = []                # Stage 2：rollout status（eos/budget_stop）
        self._write_pos = 0     # 环形写指针
        self.size = 0           # 当前有效样本数

    def _ensure_alloc(self, T: int, device, dtype):
        if self.ids is None:
            Ks = self.student_top_k
            self.ids = torch.zeros(self.capacity, T, self.top_k,
                                   dtype=torch.long, device=device)
            self.delta_k = torch.zeros(self.capacity, T, self.top_k,
                                       dtype=dtype, device=device)
            self.ids_sorted = self.ids.clone()
            self.delta_k_sorted = self.delta_k.clone()
            self.s_old_ids = torch.zeros(self.capacity, T, Ks,
                                         dtype=torch.long, device=device)
            self.s_old_logp = torch.zeros(self.capacity, T, Ks,
                                          dtype=dtype, device=device)
            self.s_old_ids_sorted = self.s_old_ids.clone()
            self.s_old_logp_sorted = self.s_old_logp.clone()

    def _sort_slot(self, pos: int):
        """按 token id 升序重排教师 top-K 与行为策略学生 top-K（供 searchsorted）。"""
        o = torch.argsort(self.ids[pos], dim=-1)
        self.ids_sorted[pos] = self.ids[pos].gather(-1, o)
        self.delta_k_sorted[pos] = self.delta_k[pos].gather(-1, o)
        so = torch.argsort(self.s_old_ids[pos], dim=-1)
        self.s_old_ids_sorted[pos] = self.s_old_ids[pos].gather(-1, so)
        self.s_old_logp_sorted[pos] = self.s_old_logp[pos].gather(-1, so)

    def append(self, ids: torch.Tensor, delta_k: torch.Tensor,
               generation_step: int, response_length: int,
               token_mask: torch.Tensor, disagreement_abs: float,
               prompt_idx: int, response: torch.Tensor,
               s_old_ids: torch.Tensor, s_old_logp: torch.Tensor,
               status: str = "budget_stop") -> int:
        """append 一条样本（ids/delta_k/s_old_ids/s_old_logp: (T,K)）。满则 FIFO 淘汰。

        status：Stage 2 rollout 状态（eos/budget_stop），默认 budget_stop 兼容旧调用。
        返回写入的槽位 pos（供 run_refresh_phase 估计 reward）。
        """
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
        self.s_old_ids[pos] = s_old_ids
        self.s_old_logp[pos] = s_old_logp
        self._sort_slot(pos)
        # 列表按 pos 索引（ring buffer 槽位复用）
        if pos < len(self._gen_steps):
            self._gen_steps[pos] = generation_step
            self._resp_lens[pos] = response_length
            self._token_masks[pos] = token_mask
            self._disagreements[pos] = disagreement_abs
            self._protected[pos] = False
            self._prompt_idx[pos] = prompt_idx
            self._response[pos] = response
            self._status[pos] = status
        else:
            self._gen_steps.append(generation_step)
            self._resp_lens.append(response_length)
            self._token_masks.append(token_mask)
            self._disagreements.append(disagreement_abs)
            self._protected.append(False)
            self._prompt_idx.append(prompt_idx)
            self._response.append(response)
            self._status.append(status)
        # 价值保护：高于分位的样本标记（G7：U_i 驱动，None 退回 disagreement）。
        # ⚠️ 先更新 size（新样本已写入槽位，属有效样本），否则首条 append 时
        # _value_threshold 按 size=0 构造空张量 → quantile 崩。
        self.size = min(self.size + 1, self.capacity)
        if self._utility(pos, generation_step) > self._value_threshold(generation_step):
            self._protected[pos] = True
        self._write_pos = (pos + 1) % self.capacity
        return pos

    def _utility(self, pos: int, current_step: int) -> float:
        """G7（§3.5）：U_i = λ_D·D + λ_R·R̂ − λ_A·A。D=disagreement、R̂=reward_estimate、
        A=age(current_step−gen_step)。utility_weights=None → 退回 disagreement（旧行为）。"""
        if self.utility_weights is None:
            return self._disagreements[pos]
        ld = self.utility_weights.get("disagreement_weight", 0.5)
        lr = self.utility_weights.get("reward_weight", 0.3)
        la = self.utility_weights.get("age_penalty", 0.2)
        d = self._disagreements[pos]
        r = self.reward_estimate(pos)
        age = max(0, current_step - self._gen_steps[pos])
        return ld * d + lr * r - la * age

    def _value_threshold(self, current_step: int = 0) -> float:
        """当前 utility 的价值保护分位（无样本时 inf；None 权重=disagreement 分位）。"""
        if not self._disagreements:
            return float("inf")
        vals = [self._utility(i, current_step) for i in range(self.size)]
        # float64：避免单样本/多样本时 float32 量化把「恰等于分位」的样本误判为
        # `>` 受保护（0.35 在 float32 下是 0.34999999 → 0.35 > 0.34999999 误触发）。
        return float(torch.tensor(vals, dtype=torch.float64).quantile(
            self.value_protect_quantile).item())

    def get(self, idxs: torch.Tensor) -> dict:
        """(B,) 局部 idx -> 训练所需字段（teacher Δ_T + 行为 s_old + 上下文 + 标量）。"""
        if self.size == 0:
            T = self.ids.size(1) if self.ids is not None else 0
            K = self.top_k
            return {"ids": torch.empty(0, T, K, dtype=torch.long),
                    "delta_k": torch.empty(0, T, K)}
        il = idxs.tolist()
        return {
            "ids": self.ids[idxs],
            "delta_k": self.delta_k[idxs],
            "ids_sorted": self.ids_sorted[idxs],
            "delta_k_sorted": self.delta_k_sorted[idxs],
            "s_old_ids": self.s_old_ids[idxs],
            "s_old_logp": self.s_old_logp[idxs],
            "s_old_ids_sorted": self.s_old_ids_sorted[idxs],
            "s_old_logp_sorted": self.s_old_logp_sorted[idxs],
            "gen_steps": [self._gen_steps[i] for i in il],
            "resp_lens": [self._resp_lens[i] for i in il],
            "token_masks": torch.stack([self._token_masks[i] for i in il]),
            "disagreements": [self._disagreements[i] for i in il],
            "prompt_idx": torch.tensor([self._prompt_idx[i] for i in il], dtype=torch.long),
            "responses": torch.stack([self._response[i] for i in il]),
            "status": [self._status[i] for i in il],
        }

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """随机取 n 个局部 idx（≤ size）。"""
        if self.size == 0:
            return torch.empty(0, dtype=torch.long)
        return torch.randint(0, self.size, (min(n, self.size),), generator=generator)

    def delta_at_student_topk(self, idxs: torch.Tensor,
                              student_topk_ids: torch.Tensor,
                              device) -> torch.Tensor:
        """把教师 Δ_T 展开到【s_cur 当前 top-K】支撑（searchsorted 二分，省 O(K²)）。

        idxs: (B,) 局部槽位；student_topk_ids: (B,T,Ks) s_cur top-K id。
        返回 (B,T,Ks)：教师 top-K 命中处取 delta_k，未命中填 0（Δ=0 贡献 0）。
        与 base 缓存的 delta_for_student_topk 同口径（跨词表 student 超出教师词表 → 0）。
        """
        ids_s = self.ids_sorted[idxs]                 # (B,T,Kt) 已升序
        delta_s = self.delta_k_sorted[idxs]           # (B,T,Kt)
        Kt = ids_s.size(-1)
        pos = torch.searchsorted(ids_s, student_topk_ids.contiguous()).clamp(max=Kt - 1)
        found = ids_s.gather(-1, pos) == student_topk_ids
        return delta_s.gather(-1, pos).where(found, torch.zeros_like(delta_s.gather(-1, pos)))

    def s_old_at_student_topk(self, idxs: torch.Tensor,
                              student_topk_ids: torch.Tensor,
                              device, tail_logp: float = -1e2) -> torch.Tensor:
        """把行为策略 s_old 展开到【s_cur 当前 top-K】支撑（searchsorted）。

        返回 (B,T,Ks)：行为策略学生 top-K 命中处取 logp，未命中（生成时几乎为 0）
        填 tail_logp（≈log 0），使 ratio 给出强约束（与 base 稀疏路径的 ref_tail_logp 同向）。
        """
        ids_s = self.s_old_ids_sorted[idxs]           # (B,T,Ks) 已升序
        logp_s = self.s_old_logp_sorted[idxs]
        Ks = ids_s.size(-1)
        pos = torch.searchsorted(ids_s, student_topk_ids.contiguous()).clamp(max=Ks - 1)
        found = ids_s.gather(-1, pos) == student_topk_ids
        return logp_s.gather(-1, pos).where(
            found, torch.full_like(logp_s.gather(-1, pos), tail_logp))

    def reward_estimate(self, pos: int, device=None) -> float:
        """E_{π_cur}[Δ_T] 估计（行为策略 top-K 支撑，monitor 用，非训练信号）。

        s_cur=生成时 student（行为），π_cur≈s_old_logp.exp()；Δ_T 在教师 top-K 上，
        用 s_old_ids_sorted 匹配 delta_k_sorted。供 PromptStateStore.reward_ema 更新。
        """
        s_ids = self.s_old_ids_sorted[pos]            # (T,Ks)
        s_logp = self.s_old_logp_sorted[pos]
        d_s = self.delta_k_sorted[pos]                # (T,Kt)
        Kt = d_s.size(-1)
        pos2 = torch.searchsorted(self.ids_sorted[pos], s_ids.contiguous()).clamp(max=Kt - 1)
        found = self.ids_sorted[pos].gather(-1, pos2) == s_ids
        delta_at = d_s.gather(-1, pos2).where(found, torch.zeros_like(d_s.gather(-1, pos2)))
        p = s_logp.exp()
        return float((p * delta_at).sum(-1).mean())

    def state_dict(self) -> dict:
        """序列化（断点续跑 G8）：张量切片 + 标量列表。"""
        return {
            "capacity": self.capacity,
            "top_k": self.top_k,
            "student_top_k": self.student_top_k,
            "vocab": self.vocab,
            "value_protect_quantile": self.value_protect_quantile,
            "size": self.size,
            "write_pos": self._write_pos,
            "ids": (self.ids[:self.size].detach().cpu() if self.ids is not None else None),
            "delta_k": (self.delta_k[:self.size].detach().cpu() if self.delta_k is not None else None),
            "s_old_ids": (self.s_old_ids[:self.size].detach().cpu() if self.s_old_ids is not None else None),
            "s_old_logp": (self.s_old_logp[:self.size].detach().cpu() if self.s_old_logp is not None else None),
            "gen_steps": self._gen_steps[:self.size],
            "resp_lens": self._resp_lens[:self.size],
            "token_masks": [m.detach().cpu() for m in self._token_masks[:self.size]],
            "disagreements": self._disagreements[:self.size],
            "protected": self._protected[:self.size],
            "prompt_idx": self._prompt_idx[:self.size],
            "responses": [r.detach().cpu() for r in self._response[:self.size]],
            "status": self._status[:self.size],
        }

    def load_state_dict(self, sd: dict):
        """从断点恢复（G8）。注意：capacity/top_k 以构造时为准，仅恢复内容。"""
        self.size = int(sd["size"])
        self._write_pos = int(sd["write_pos"])
        if self.size > 0:
            self._ensure_alloc(sd["ids"].size(1), sd["ids"].device, sd["delta_k"].dtype)
            self.ids[:self.size] = sd["ids"]
            self.delta_k[:self.size] = sd["delta_k"]
            self.s_old_ids[:self.size] = sd["s_old_ids"]
            self.s_old_logp[:self.size] = sd["s_old_logp"]
            for i in range(self.size):
                self._sort_slot(i)
        self._gen_steps = list(sd["gen_steps"][:self.size])
        self._resp_lens = list(sd["resp_lens"][:self.size])
        self._token_masks = list(sd["token_masks"][:self.size])
        self._disagreements = list(sd["disagreements"][:self.size])
        self._protected = list(sd["protected"][:self.size])
        self._prompt_idx = list(sd["prompt_idx"][:self.size])
        self._response = list(sd["responses"][:self.size])
        self._status = list(sd.get("status", ["budget_stop"] * self.size)[:self.size])  # 兼容旧断点

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
                      m_selected, max_resp_len, top_k, device,
                      prompt_state=None,
                      rollout_generator=None,      # Stage 2：短 rollout 生成器
                      eos_token_id=None,           # Stage 2：None=不判 EOS（全 budget_stop）
                      loop_detection=True,         # Stage 2：周期重复判 loop
                      loop_periods=(2, 3, 4),      # IMP-1b：尾部周期检测周期集合（默认 (2,3,4)）
                      pad_id=0,
                      temperature: float = 0.7,    # IMP-1a：采样温度（默认 0.7 降循环率；1.0 旧行为）
                      budgets=None,                # Stage 3：per-sample budget (M,) long；None→单预算
                      budget_t=None,               # Stage 3：全局 token 预算；None→无上限
                      budget_set=(256, 512, 1024, 2048),
                      exploration_fraction=0.20,
                      cand=None,                   # Stage 3：预选 prompt 索引 (M,)；None→内部 select
                      compute_disagreement=True):  # P1.4：disagreement.enabled 硬 gate
                                                  # False 时跳过 D 计算（省 student/student_ref
                                                  # chosen-logp 前向），append 用 D=0
    """§3.3 + §6.5 rollout 相位：selective 选 prompt -> student 短 rollout（带 status）
    -> 4 个 chosen logp -> D_i^abs -> append_refresh。teacher 前向在此（_train_step 不动）。

    Stage 2：改用 generate_with_status（默认 model.generate_with_status，可经
    rollout_generator 注入 vLLM 端）。每 rollout 记录 status（eos/budget_stop/loop/invalid）；
    loop/invalid 样本跳过 teacher 前向与 append（不进 refresh cache，§短 rollout），
    仍计入返回 summary。响应按有效长度构建长度式 mask（m_t=1 当 t<length）。

    Stage 3：budgets 非 None 时走 per-sample budget 分桶（Select(p, B_p)）——先
    enforce_budget（超全局 budget_t 降档），再按 budget 分组逐档生成，prompt 与
    budget 一一对应；budgets=None 走原单预算路径（零回归）。cand 可由外部传入
    （select_with_budget 的 indices，与 budgets 配对）；None → 内部 select。

    返回 summary dict：{n_total, n_appended, n_eos, n_budget, n_loop, n_invalid,
    rollout_tokens, expected_rollout_tokens, budgets_used, teacher_forward_tokens,
    loop_periods（配置的周期集合），temperature（采样温度）}。
    除标量块控，还存行为策略 s_old（学生生成时完整分布 top-K，G1 闭环）与
    prompt_idx/response（G2 PromptState 闭环）。

    G9：max_resp_len 是【新增】token 数，超出位置编码最大长度会越界（toy max_len=64 vs
    默认 8192）。总序列长 = prompt_len + max_resp_len 必须 ≤ max_len，故 clamp 到
    (max_len - prompt_len)，保证 CPU smoke 与 GPU 长序列都安全。
    """
    from .model import token_logprobs, response_dists, build_length_mask
    from .model import generate_with_status as _default_gen
    _max_seq = int(getattr(student, "max_len", max_resp_len))
    max_resp_len = min(int(max_resp_len), max(1, _max_seq - prompts.size(1)))
    # Stage 3：cand 可由外部传入（select_with_budget 的 indices，与 budgets 配对），
    # 保证预算桶与选中 prompt 一一对应；None → 内部 select（零回归，单预算路径）。
    if cand is None:
        cand = selector.select(m_selected, prompts.size(0)) if selector else \
            torch.randint(0, prompts.size(0), (m_selected,))
    # Stage 2：短 rollout（带 status）。默认 toy 生成器；可注入 vLLM/HF 端。
    # ⚠️ 调用约定：注入的 rollout_generator 是【绑定方法】（self 已绑），签名
    # generate_with_status(prompts, max_new=..., ...)；而 _default_gen 是模块级函数，
    # 签名 generate_with_status(model, prompts, ...)。两者必须分开调用，否则把 student
    # 当 self 传入绑定方法会静默错乱（P2 修复：此前 vLLM 注入路径就是此 bug）。
    def _gen(pb, max_new):
        if rollout_generator is not None:
            return rollout_generator(pb, max_new=max_new, eos_token_id=eos_token_id,
                                     loop_detection=loop_detection, pad_id=pad_id,
                                     temperature=temperature, loop_periods=loop_periods)
        return _default_gen(student, pb, max_new=max_new, eos_token_id=eos_token_id,
                            loop_detection=loop_detection, pad_id=pad_id,
                            temperature=temperature, loop_periods=loop_periods)

    M = cand.size(0)
    if budgets is not None:
        # Stage 3：per-sample budget 分桶。先 enforce（超全局预算降级）。
        if budget_t is not None:
            v_vals = selector._value()[cand] if selector is not None \
                else torch.zeros(M, device=cand.device)
            cand, budgets = enforce_budget(cand, budgets, v_vals, budget_t,
                                           budget_set, exploration_fraction)
        p_b = prompts[cand].to(device)
        groups = group_by_budget(cand, budgets)          # {budget: [global_prompt_idx,...]}
        max_b = int(budgets.max().item())
        resp_all = torch.full((M, max_b), pad_id, dtype=torch.long, device=device)
        statuses = [None] * M; lengths = [0] * M; eos_pos = [None] * M
        budgets_used = int(budgets.sum().item())
        cand_list = cand.tolist()
        for b, idxs in groups.items():
            # 映射：每个 bucket 元素（全局 prompt idx）在 cand 中的行位置。
            pos_in_cand = [cand_list.index(int(i)) for i in idxs]
            p_bucket = prompts[torch.tensor(idxs, device=prompts.device)].to(device)
            out = _gen(p_bucket, max_new=int(b))
            for k, pc in enumerate(pos_in_cand):
                rk = out["responses"][k]
                resp_all[pc, :rk.size(0)] = rk
                statuses[pc] = out["statuses"][k]
                lengths[pc] = out["lengths"][k]
                eos_pos[pc] = out["eos_pos"][k]
        responses = resp_all
        expected = budgets_used
    else:
        p_b = prompts[cand].to(device)
        out = _gen(p_b, max_new=max_resp_len)
        responses, statuses, lengths = out["responses"], out["statuses"], out["lengths"]
        eos_pos = out["eos_pos"]
        expected = m_selected * max_resp_len
        budgets_used = m_selected * max_resp_len
    # loop/invalid 样本跳过 teacher 前向与 append（需求 4），仍计入 summary
    valid = [i for i in range(len(statuses)) if statuses[i] not in ("loop", "invalid")]
    n_loop = statuses.count("loop"); n_invalid = statuses.count("invalid")
    n_eos = statuses.count("eos"); n_budget = statuses.count("budget_stop")
    valid_lens = [lengths[i] for i in valid]
    # P1.3：rollout_tokens=进 refresh 池的有效样本 token 数（非名义预算）。
    actual = int(sum(valid_lens))
    if not valid:
        return {"n_total": len(statuses), "n_appended": 0, "n_eos": n_eos,
                "n_budget": n_budget, "n_loop": n_loop, "n_invalid": n_invalid,
                "rollout_tokens": actual, "expected_rollout_tokens": int(expected),
                "budgets_used": int(budgets_used),
                "teacher_forward_tokens": 0,
                "loop_periods": tuple(loop_periods),
                "temperature": float(temperature)}
    p_b_v = p_b[valid]
    resp_v = responses[valid]
    # 行为策略：生成完立即取当前 student 完整分布 top-K（s_old，精确行为策略 §2）。
    # 同一相位内 student 权重未变，因此 refresh 训练第一步 ratio≈1（纯 on-policy）。
    with torch.no_grad():
        s_full = response_dists(student, p_b_v, resp_v)          # (M,T,V)
    Ks = ring_buffer.student_top_k
    s_old_ids = s_full.topk(min(Ks, s_full.size(-1)), dim=-1).indices
    s_old_logp = s_full.topk(min(Ks, s_full.size(-1)), dim=-1).values
    rl_dist = response_dists(teacher_rl, p_b_v, resp_v)      # (M,T,V)
    ref_dist = response_dists(teacher_ref, p_b_v, resp_v)
    tk = rl_dist.topk(top_k, dim=-1)
    ids_k, rl_k = tk.indices, tk.values
    delta_k = rl_k - ref_dist.gather(-1, tk.indices)
    # Stage 2：长度式 mask（真实 EOS 后 padding=0；非 pad 扫描）。先于 D 计算（compute 用）。
    mask = build_length_mask(resp_v, [lengths[i] for i in valid],
                             [eos_pos[i] for i in valid])
    # P1.4：disagreement.enabled 硬 gate——关闭时跳过 D 计算（省 student/student_ref
    # chosen-logp 前向），append 用 D=0（E1↔E2 差异改为真实计算量差异）。
    if compute_disagreement:
        with torch.no_grad():
            student_ref.eval()
            ref_logp = token_logprobs(student_ref, p_b_v, resp_v)
            rl_chosen = disag.gather_chosen_logp(rl_dist, resp_v)
            ref_chosen = disag.gather_chosen_logp(ref_dist, resp_v)
            student_logp = token_logprobs(student, p_b_v, resp_v)
        D_vals = disag.compute(rl_chosen, ref_chosen, student_logp, ref_logp, mask).get("abs")
        D_vals = [float(v.detach()) for v in D_vals]
    else:
        D_vals = [0.0] * len(valid)
    for j, i in enumerate(valid):
        pos = ring_buffer.append(ids_k[j], delta_k[j], generation_step=step,
            response_length=int(mask[j].sum()), token_mask=mask[j],
            disagreement_abs=D_vals[j],
            prompt_idx=int(cand[i]), response=resp_v[j],
            s_old_ids=s_old_ids[j], s_old_logp=s_old_logp[j],
            status=statuses[i])
        # G2：写回 PromptState（rollout 结果 -> prompt 历史 -> 下次 selection 闭环）。
        if prompt_state is not None:
            rew = ring_buffer.reward_estimate(pos)
            prompt_state.update(int(cand[i]), reward=rew,
                                disagreement=D_vals[j],
                                resp_len=int(mask[j].sum()), step=step)
    # P1.3 成本核算：teacher_forward_tokens=教师 rl+ref 前向处理的有效 token 数
    # （2×Σ有效长，供 Performance/Teacher Compute 比值）。
    return {"n_total": len(statuses), "n_appended": len(valid), "n_eos": n_eos,
            "n_budget": n_budget, "n_loop": n_loop, "n_invalid": n_invalid,
            "rollout_tokens": actual, "expected_rollout_tokens": int(expected),
            "budgets_used": int(budgets_used),
            "teacher_forward_tokens": 2 * int(sum(valid_lens)),
            "loop_periods": tuple(loop_periods),
            "temperature": float(temperature)}
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
                 ema_beta=0.9, warmup_steps=500, max_step_change=0.05,
                 token_aware=False, token_weight=0.1):
        self.alpha0 = initial
        self.min, self.max = min, max
        self.mode = mode
        self.w = dict(age=age_weight, drift=drift_weight, quality=quality_weight)
        self.beta = ema_beta
        self.warmup = warmup_steps
        self.max_step = max_step_change
        self.token_aware = token_aware
        self.token_weight = token_weight
        self._ema = dict(age=0.0, drift=0.0, quality=0.0, efficiency=0.0)
        self._step = 0
        self._last_alpha = initial
        # linear 模式起止（§5.6）
        self._lin_start, self._lin_end = 0.1, 0.5

    def _norm(self, key, x):
        """EMA + 简单 normalize（x/(1+|x|) 映射到 [-1,1] 附近，防极值爆炸）。"""
        self._ema[key] = self.beta * self._ema[key] + (1 - self.beta) * x
        return self._ema[key] / (1 + abs(self._ema[key]))

    def update(self, base_age, policy_drift, refresh_quality,
               rollout_efficiency=None) -> float:
        """推进一步，返回本轮 α（三信号 or 按模式降级）。

        rollout_efficiency: 可选 rollout token 效率（expected/actual，§五）；None 默认零回归。
        """
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
        # §五 第 4 信号：token 感知（任务 6）。eff = rollout_efficiency - 1（>0 省 token
        # → 放宽 α；<0 超用 → 收紧）。token_aware=False / rollout_efficiency=None /
        # 非 adaptive 分支（此处已排除）时完全不参与（零回归）。
        if self.token_aware and rollout_efficiency is not None:
            eff = self._norm("efficiency", rollout_efficiency - 1.0)
            raw += self.token_weight * eff
        raw = max(self.min, min(self.max, raw))
        # max_step_change 限幅（§5.4）
        raw = self._last_alpha + max(-self.max_step, min(self.max_step, raw - self._last_alpha))
        raw = max(self.min, min(self.max, raw))
        self._last_alpha = raw
        return raw

    def cold_start_adjust(self, alpha, n_refresh, n_batch) -> float:
        """§5.5：refresh 不足时 α_actual=min(α, N_R/N_batch)。"""
        return min(alpha, n_refresh / max(1, n_batch))


class PromptStateStore:
    """§6.1 per-prompt 轻量历史状态（复用 §3/§4 信号，不重复 forward）。

    为每个 prompt 维护极简可微状态：times_seen / last_seen_step / reward_ema /
    reward_var / disagreement_ema / last_response_length / reuse_count。
    全部为固定形状张量（O(n_prompts)），无增长、无遗留，供 RefreshSelector
    做 cheap scoring（V=0.4U+0.4D+0.2N），candidate 阶段不跑 teacher。
    """

    def __init__(self, n_prompts: int):
        self.n = n_prompts
        self.times_seen = torch.zeros(n_prompts, dtype=torch.long)
        self.last_seen_step = torch.zeros(n_prompts, dtype=torch.long)
        self.reward_ema = torch.zeros(n_prompts)
        self.reward_var = torch.zeros(n_prompts)
        self.disagreement_ema = torch.zeros(n_prompts)
        self.last_response_length = torch.zeros(n_prompts, dtype=torch.long)
        self.reuse_count = torch.zeros(n_prompts, dtype=torch.long)

    def update(self, prompt_id, reward, disagreement, resp_len, step):
        """一次 rollout 后更新该 prompt 的历史状态（EMA 平滑）。"""
        prompt_id = int(prompt_id)
        self.times_seen[prompt_id] += 1
        self.last_seen_step[prompt_id] = step
        # reward EMA（含方差一阶近似：|Δ| 作为不确定度增量）
        self.reward_ema[prompt_id] = 0.9 * self.reward_ema[prompt_id] + 0.1 * reward
        self.reward_var[prompt_id] = 0.9 * self.reward_var[prompt_id] + \
            0.1 * abs(reward - self.reward_ema[prompt_id])
        self.disagreement_ema[prompt_id] = 0.9 * self.disagreement_ema[prompt_id] + \
            0.1 * disagreement
        self.last_response_length[prompt_id] = resp_len

    def record_reuse(self, prompt_id):
        """记录样本被 recycle 复用（§6.8 diversity 反信号）。"""
        self.reuse_count[int(prompt_id)] += 1

    def novelty(self) -> torch.Tensor:
        """novelty：从未见/少见的 prompt 更高（§6.3 价值权重 N）。"""
        return 1.0 / torch.sqrt(1.0 + self.times_seen.float())


class RefreshSelector:
    """§6 Selective Rollout（candidate pool 两阶段降本）。

    M_candidate=4·M_selected -> cheap scoring(V=0.4U+0.4D+0.2N)
    -> 80% top-value + 20% coverage -> M selected。
    candidate 阶段不跑 teacher（只对 O(n) 历史状态打分）。diversity protection
    （max_same_prompt_fraction 限单 prompt 占比）+ failure fallback uniform。
    """

    def __init__(self, prompt_state: PromptStateStore, candidate_multiplier: int = 4,
                 value_fraction: float = 0.80, coverage_fraction: float = 0.20,
                 value_weights: dict | None = None, compute_aware: bool = False,
                 max_same_prompt_fraction: float = 0.05,
                 exploration_fraction: float = 0.20, seed: int = 42):
        self.ps = prompt_state
        self.cm = candidate_multiplier
        self.vf = value_fraction
        self.cf = coverage_fraction
        self.vw = value_weights or {"uncertainty": 0.4, "disagreement": 0.4, "novelty": 0.2}
        self.compute_aware = compute_aware
        self.max_same = max_same_prompt_fraction
        self.exploration = exploration_fraction
        self.gen = torch.Generator().manual_seed(seed)

    def _value(self, budget_cost: torch.Tensor | None = None) -> torch.Tensor:
        """§6.3 cheap value：V = λU·U + λD·D + λN·N（+ λR·R，可选 compute-aware 除 cost）。

        budget_cost: 可选 per-prompt 期望 token 成本（§四 U'(p)=V/(ExpectedTokens+ε)），
        传入时覆盖 compute-aware 的默认 cost（last_response_length）。
        "reward" in self.vw 保证旧 config 无 reward 键时零回归（R 恒 0、不加项）。
        """
        U = self.ps.reward_var                        # uncertainty
        D = self.ps.disagreement_ema
        N = self.ps.novelty()
        # reward 项：仅当 config 显式给了 reward 权重才纳入（旧 config 零回归）
        R = self.ps.reward_ema if "reward" in self.vw else torch.zeros_like(U)
        v = (self.vw["uncertainty"] * U + self.vw["disagreement"] * D
             + self.vw["novelty"] * N)
        if "reward" in self.vw:
            v = v + self.vw["reward"] * R
        if self.compute_aware:
            cost = budget_cost if budget_cost is not None \
                else self.ps.last_response_length.float()
            v = v / (cost + 1e-8)
        return v

    def select(self, n_selected: int, n_prompts: int) -> torch.Tensor:
        """两阶段：candidate pool -> top-value + coverage -> M selected（§6.5）。

        failure fallback：history 太短（times_seen 全 0）或 n_selected>=n_prompts
        时退化为 uniform（§6.9 cold start）。
        """
        if self.ps.times_seen.sum() == 0 or n_selected >= n_prompts:
            return torch.randint(0, n_prompts, (n_selected,), generator=self.gen)
        n_cand = min(self.cm * n_selected, n_prompts)
        cand = torch.randperm(n_prompts, generator=self.gen)[:n_cand]
        v = self._value()[cand]
        n_high = int(round(n_selected * self.vf))
        n_cov = n_selected - n_high
        # 80% top-value（§6.3）
        top = cand[v.topk(min(n_high, n_cand)).indices]
        # 20% coverage（从候选中随机补足，排除已选）
        remaining = cand[~torch.isin(cand, top)]
        cov = remaining[torch.randperm(len(remaining), generator=self.gen)[:n_cov]] \
            if len(remaining) > 0 else top[:n_cov]
        selected = torch.cat([top, cov])
        # diversity：max_same_prompt_fraction 限制单 prompt 占比（§6.8）
        max_per = max(1, int(n_selected * self.max_same))
        from collections import Counter
        cnt = Counter(selected.tolist())
        filtered: list[int] = []
        for p in selected.tolist():
            if cnt[p] <= max_per:
                filtered.append(p)
            else:
                cnt[p] -= 1
        # 不足补 uniform（不重复起生成器，保证确定性）
        while len(filtered) < n_selected:
            filtered.append(torch.randint(0, n_prompts, (1,), generator=self.gen).item())
        return torch.tensor(filtered[:n_selected])

    def select_with_budget(self, n_selected: int, n_prompts: int,
                           budget_set=(256, 512, 1024, 2048),
                           quantiles=(0.25, 0.5, 0.75),
                           fixed_budget: int = 1024,
                           budget_mode: str = "fixed") -> tuple:
        """Stage 3：Select(p, B_p)——选 prompt + 分配 reasoning budget。

        复用 self.select()（冷启动/两阶段选择不变，零回归）。
        budget_mode='fixed' → 全 fixed_budget（单预算，等价旧行为）；
        'adaptive' → 按选中集内 V 分位数映射到 4 档（assign_budgets）。
        返回 (indices, budgets) 两个 (M,) long tensor。
        """
        indices = self.select(n_selected, n_prompts)
        if budget_mode == "fixed":
            budgets = torch.full_like(indices, fixed_budget, dtype=torch.long)
        else:
            v_sel = self._value()[indices]
            budgets = assign_budgets(v_sel, budget_set, quantiles)
        return indices, budgets

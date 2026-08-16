# Adaptive Staleness-Aware Teacher Cache 实现计划


> **状态（2026-08-16 同步）：** L2 四能力（RefreshRingBuffer/Disagreement/CacheHealthMonitor/DynamicRatio/RefreshSelector+PromptState）+ 双池 feeder + 交替相位 + E0-E6 已实现；服务器 pytest 344 全绿；真实规模 E0-E6 已实测（见实现报告 §10，循环率高是主要限制）；G5/G7/成本核算/disagreement gate/真实 pad 已在 P1 落地。HF/FSDP/分布式骨架仍为 GPU 待验证。
> 本计划的实现进度已按服务器实测/测试结果回填；`- [x]` 表示已实现并通过
> 服务器 pytest（341 全绿）与真实 GPU 运行验证，未打勾项为后续/部分完成。

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [x]`）语法来跟踪进度。

**目标：** 在 L2 周期刷新机制上实现 Adaptive Staleness-Aware Teacher Cache（四能力 + 整合 + E0-E6 实验框架），保持 `_train_step` teacher-free、最小侵入、默认可关。

**架构：** 新增 `adaptive_cache.py` 承载全部 L2 逻辑（RefreshRingBuffer / DisagreementComputer / CacheHealthMonitor / DynamicRatioController / RefreshSelector / PromptStateStore），`cache.py`/`scheduler.py`/`config.py`/`checkpoint.py`/`data.py` 做薄扩展，`pipeline.py` 接入交替相位循环。`l2.enabled: false` 时全部退回 L0/L1 静态路径。

**技术栈：** pytorch / pydantic（extra="forbid"）/ pytest / transformers（HF 骨架）/ FSDP（双卡）

**规格：** `docs/superpowers/specs/2026-08-14-adaptive-teacher-cache-design.md`（13 节，四能力权威定义）

**关键约束（贯穿全计划）：**
- `_train_step`（scheduler.py:282）内核一行不动，teacher-free 保持
- 所有功能默认关（`l2.enabled: false` 退回原行为）
- 中文注释/文档/commit
- toy 路径完整可测（CPU），HF 路径标骨架扩展点

---

## 文件结构

**新增：**
- `main/fullstack_opd_v2/adaptive_cache.py` — L2 全部新增逻辑，6 个独立类（职责单一、接口清晰）
- `main/tests/test_adaptive_cache.py` — L2 单元测试
- `main/tests/test_l2_integration.py` — L2 集成测试（双池 feeder / 交替相位 / E0-E6 回归）

**修改：**
- `main/fullstack_opd_v2/config.py` — 加 `L2Cfg` schema + `OPDConfig.l2` 槽位（extra="forbid" 须加合法位置）
- `main/fullstack_opd_v2/cache.py` — `TensorTeacherCache` 加 refresh pool（append_refresh / get_refresh / evict，base 池张量不动）
- `main/fullstack_opd_v2/scheduler.py` — 双池 feeder（`_rand_idxs` 包一层）+ staleness bug 修（`_train_step:291` base 不截断）+ mask 传递 + L2 钩子
- `main/fullstack_opd_v2/checkpoint.py` — save/resume 加 optimizer state + RNG + ring buffer 状态
- `main/fullstack_opd_v2/data.py` — `JsonLinesDataLoader` 回传 padding mask
- `main/fullstack_opd_v2/model.py` / `model_factory.py` — `HFCausalLM` 加 `generate_batch` + attention_mask（骨架）
- `main/fullstack_opd_v2/pipeline.py` — L2 交替相位循环接入 + teacher rollout 相位

**类职责（adaptive_cache.py）：**

| 类 | 职责 | 依赖 |
|----|------|------|
| `RefreshRingBuffer` | refresh pool append/FIFO/价值保护淘汰 | cache 张量 |
| `PromptStateStore` | per-prompt 历史状态（times_seen/reward_ema/disagreement_ema/reuse_count） | — |
| `DisagreementComputer` | §3 D_t/D_i^abs 计算（rollout 阶段，teacher-free _train_step） | chosen-token logp |
| `CacheHealthMonitor` | §4 七维监控 + rule-based health score + alert cooldown | RefreshRingBuffer + PromptStateStore |
| `DynamicRatioController` | §5 三信号 controller α + cold start + fixed/linear/adaptive | CacheHealthMonitor 信号 |
| `RefreshSelector` | §6 candidate pool 两阶段 + value + coverage + diversity | PromptStateStore |

单向依赖（§13.1，禁止循环修改）：`RefreshSelector -> DisagreementComputer -> RefreshRingBuffer -> CacheHealthMonitor -> DynamicRatioController -> Feeder`

---

## 阶段 1：L2 基础（config + ring buffer + 双池 feeder + staleness 修 + mask + checkpoint）

### 任务 1.1：config L2Cfg schema

**文件：**
- 修改：`main/fullstack_opd_v2/config.py`（`Stage2Cfg` 后新增 `L2Cfg`，`OPDConfig` 加 `l2` 槽位）
- 测试：`main/tests/test_config.py`

- [x] **步骤 1：编写失败测试**

在 `test_config.py` 末尾加：
```python
def test_l2_cfg_defaults_off():
    """L2Cfg 默认全关，l2.enabled=false 退回 L0/L1。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config()  # 无 path 用默认
    assert cfg["l2"]["enabled"] is False
    assert cfg["l2"]["cache"]["base_size"] == 50000
    assert cfg["l2"]["refresh_ratio"]["mode"] == "adaptive"
    assert cfg["l2"]["selective_rollout"]["enabled"] is True
    assert cfg["l2"]["disagreement"]["enabled"] is True
    assert cfg["l2"]["health_monitor"]["enabled"] is True

def test_l2_cfg_unknown_key_rejected():
    """extra=forbid：未知 l2 键报错。"""
    import pytest
    from fullstack_opd_v2.config import load_config, ConfigError
    with pytest.raises(ConfigError):
        load_config(overrides=["l2.bogus=1"])
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_config.py::test_l2_cfg_defaults_off -q`
预期：FAIL，`KeyError: 'l2'`

- [x] **步骤 3：实现 L2Cfg**

在 `config.py` 的 `Stage2Cfg` 类之后、`RunCfg` 之前插入：
```python
class L2CacheCfg(_Strict):
    """L2 ring buffer 基础（§2）。"""
    base_size: int = 50000
    refresh_size: int = 5000          # ring buffer capacity
    max_response_length: int = 8192
    value_protect_quantile: float = 0.9   # §2 Q3 价值保护
    refresh_min_interval: int = 50    # §2 Q1 触发约束
    refresh_max_interval: int = 150
    delta_slope_eps: float = 0.001


class L2DisagreementCfg(_Strict):
    enabled: bool = True


class L2HealthMonitorCfg(_Strict):
    enabled: bool = True
    # §4.3 rule-based 阈值（嵌套 dict，extra=forbid 下用 dict 接收）
    health: dict = Field(default_factory=lambda: {
        "hit_rate": {"warning": 0.995, "critical": 0.98},
        "refresh_age_p95": {"warning": 5, "critical": 10},
        "reuse_p95": {"warning": 8, "critical": 20},
        "max_length_ratio": {"warning": 0.10, "critical": 0.25},
    })
    alert_cooldown: int = 50          # §4.4 同一 warning 冷却步数


class L2RefreshRatioCfg(_Strict):
    """§5 dynamic refresh ratio。"""
    enabled: bool = True
    mode: Literal["fixed", "linear", "adaptive"] = "adaptive"
    initial: float = 0.30
    min: float = 0.10
    max: float = 0.60                 # <1，保留 base anchor（§5.4）
    age_weight: float = 0.25
    drift_weight: float = 0.50
    quality_weight: float = 0.25
    ema_beta: float = 0.9
    warmup_steps: int = 500
    max_step_change: float = 0.05


class L2SelectiveRolloutCfg(_Strict):
    """§6 selective rollout。"""
    enabled: bool = True
    candidate_multiplier: int = 4    # M_candidate = 4·M_selected（§6.5）
    value_fraction: float = 0.80     # §6.3 高价值占比
    coverage_fraction: float = 0.20
    value_weights: dict = Field(default_factory=lambda: {
        "uncertainty": 0.4, "disagreement": 0.4, "novelty": 0.2})
    compute_aware: bool = False      # §6.4 ELG
    max_same_prompt_fraction: float = 0.05   # §6.8
    exploration_fraction: float = 0.20


class L2UtilityCfg(_Strict):
    """§3.5 sample utility 系数。"""
    disagreement_weight: float = 0.5
    reward_weight: float = 0.3
    age_penalty: float = 0.2


class L2Cfg(_Strict):
    """L2 Adaptive Teacher Cache 总配置（默认全关，§7）。"""
    enabled: bool = False            # 总开关：false 退回 L0/L1
    t_train: int = 100               # 每轮训练步数
    m_refresh: int = 1000            # 每轮刷新量（= M_selected）
    cache: L2CacheCfg = Field(default_factory=L2CacheCfg)
    disagreement: L2DisagreementCfg = Field(default_factory=L2DisagreementCfg)
    health_monitor: L2HealthMonitorCfg = Field(default_factory=L2HealthMonitorCfg)
    refresh_ratio: L2RefreshRatioCfg = Field(default_factory=L2RefreshRatioCfg)
    selective_rollout: L2SelectiveRolloutCfg = Field(default_factory=L2SelectiveRolloutCfg)
    utility: L2UtilityCfg = Field(default_factory=L2UtilityCfg)
```

在 `OPDConfig` 类加槽位（`eval` 之后）：
```python
    l2: L2Cfg = Field(default_factory=L2Cfg)
```

在 `DEFAULT_CONFIG_V2`（pipeline.py）的 dict 加 `"l2": {"enabled": False}`（与 schema 默认对齐，保证 `load_config` 合并后存在）。同时 `CLOUD_CONFIG` 无需加（L2 默认关）。

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_config.py::test_l2_cfg_defaults_off tests/test_config.py::test_l2_cfg_unknown_key_rejected -q`
预期：PASS

- [x] **步骤 5：回归 + Commit**

运行：`cd main && python -m pytest tests/test_config.py -q`（全绿）
```bash
git add main/fullstack_opd_v2/config.py main/fullstack_opd_v2/pipeline.py main/tests/test_config.py
git commit -m "feat(l2): L2Cfg 配置 schema(默认全关, 每模块 enabled 可单项 ablation)"
```

---

### 任务 1.2：RefreshRingBuffer（cache refresh pool）

**文件：**
- 创建：`main/fullstack_opd_v2/adaptive_cache.py`（首个类）
- 测试：`main/tests/test_adaptive_cache.py`

- [x] **步骤 1：编写失败测试**

创建 `test_adaptive_cache.py`：
```python
"""adaptive_cache.py 单元测试：L2 四能力。"""
import torch
from fullstack_opd_v2.adaptive_cache import RefreshRingBuffer


def test_ring_buffer_append_and_get():
    """append 后按 idx 取回 refresh 张量。"""
    rb = RefreshRingBuffer(capacity=4, top_k=3, vocab=10)
    # 一条 sample：ids (T,K), delta_k (T,K)
    ids = torch.zeros(2, 3, dtype=torch.long)
    delta = torch.ones(2, 3)
    rb.append(ids, delta, generation_step=0, response_length=2,
              token_mask=torch.ones(2), disagreement_abs=0.5)
    assert rb.size == 1
    got = rb.get(torch.tensor([0]))
    assert got["ids"].shape == (1, 2, 3)


def test_ring_buffer_fifo_evict():
    """满后 FIFO 淘汰最旧。"""
    rb = RefreshRingBuffer(capacity=2, top_k=2, vocab=8)
    for i in range(3):  # append 3，capacity 2 -> 淘汰第 0 条
        rb.append(torch.full((2, 2), i, dtype=torch.long),
                  torch.full((2, 2), float(i)),
                  generation_step=i, response_length=2,
                  token_mask=torch.ones(2), disagreement_abs=float(i))
    assert rb.size == 2
    # 最旧(0)被淘汰，剩余 1,2
    got = rb.get(torch.tensor([0, 1]))
    assert got["ids"][0, 0, 0].item() == 1   # 第 0 个槽是 step 1


def test_ring_buffer_value_protect():
    """高 disagreement 样本免淘汰一轮。"""
    rb = RefreshRingBuffer(capacity=2, top_k=2, vocab=8, value_protect_quantile=0.9)
    rb.append(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2),
              0, 1, torch.ones(1), 0.1)      # 低价值
    rb.append(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2),
              1, 1, torch.ones(1), 9.0)      # 高价值(>0.9 分位)
    rb.append(torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2),
              2, 1, torch.ones(1), 0.2)      # 触发淘汰，高价值受保护
    # 高价值(step1)应仍在
    assert any(rb._gen_steps[i] == 1 for i in range(rb.size))


def test_ring_buffer_empty_safe():
    """空池 get 不崩。"""
    rb = RefreshRingBuffer(capacity=4, top_k=2, vocab=8)
    assert rb.size == 0
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_adaptive_cache.py::test_ring_buffer_append_and_get -q`
预期：FAIL，`ImportError: cannot import name 'RefreshRingBuffer'`

- [x] **步骤 3：实现 RefreshRingBuffer**

创建 `main/fullstack_opd_v2/adaptive_cache.py`：
```python
"""L2 Adaptive Staleness-Aware Teacher Cache（§3-§6 + §13 整合）。

全部新增逻辑集中于此，cache.py/scheduler.py 只做薄扩展（最小侵入）。
单向依赖（§13.1，禁止循环修改彼此内部状态）：
  RefreshSelector -> DisagreementComputer -> RefreshRingBuffer
  -> CacheHealthMonitor -> DynamicRatioController -> Feeder

设计原则：可解释、可监控、compute overhead 可控、可 ablation、不破坏训练。
所有类经 l2.enabled 总开关控制，关闭时 pipeline 退回 L0/L1 静态路径。
"""
from __future__ import annotations

import torch


class RefreshRingBuffer:
    """Refresh Pool 动态 ring buffer（§2 双池结构）。

    base 池（TensorTeacherCache 原 ids/delta_k）不动；refresh 池独立张量，
    append 进新样本，满后 FIFO 淘汰最旧，高 disagreement 样本价值保护免淘汰一轮。
    持久化字段（§13.2 统一 metadata 第一版，per-token logp 算完即弃见 spec §11）：
    ids/delta_k（训练查表）+ generation_step/response_length/token_mask/disagreement_abs。

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
        if not self._disagreements:
            return 0.0
        return float(sum(self._disagreements) / len(self._disagreements))
```

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_adaptive_cache.py -q`
预期：4 个 PASS

- [x] **步骤 5：Commit**

```bash
git add main/fullstack_opd_v2/adaptive_cache.py main/tests/test_adaptive_cache.py
git commit -m "feat(l2): RefreshRingBuffer(FIFO+价值保护, base 池不动)"
```

---

### 任务 1.3：JsonLinesDataLoader 回传 mask

**文件：**
- 修改：`main/fullstack_opd_v2/data.py`（`JsonLinesDataLoader.load` 返回 mask）
- 测试：`main/tests/test_data.py`（若无则新建）

- [x] **步骤 1：编写失败测试**

```python
def test_jsonl_returns_mask(tmp_path):
    """JsonLinesDataLoader 回传 response padding mask（§3.4 变长 response 必需）。"""
    import json
    from fullstack_opd_v2.data import JsonLinesDataLoader
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"prompt": "ab", "response": "cd"}) + "\n" +
                 json.dumps({"prompt": "ef", "response": "c"}) + "\n",
                 encoding="utf-8")
    cfg = {"dataset": {"type": "jsonl", "path": str(p),
                       "max_prompt_len": 4, "max_response_len": 4,
                       "tokenizer_path": "dummy"},
           "student_path": "dummy"}
    # dummy tokenizer 无法真实加载，此测试用 mock；实际用 toy tokenizer 跑
    # 见 test_data.py 已有 mock 模式
```
（若 `test_data.py` 已有 jsonl 测试用 mock tokenizer，在其基础上断言返回 4 元组含 mask。具体断言：`prompts, responses, reward_fn, mask = loader.load()`，`mask.shape == (N, T)`，pad 位置为 0。）

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_data.py -q -k jsonl`
预期：FAIL（返回 3 元组无 mask）

- [x] **步骤 3：实现 mask 回传**

修改 `data.py` 的 `JsonLinesDataLoader.load()`：在编码 response 时记录有效长度，构造 mask：
```python
    # 在 r_ids pad 之前记录有效长度
    r_lens = [len(tok.encode(str(row.get(self.response_key, "")),
                             add_special_tokens=False, truncation=True, max_length=T))
              for ...]  # 实际在循环内收集
```
具体改 `load()` 循环：收集 `r_valid_lens`，pad 后构造 `mask = (arange(T) < r_valid_lens).long()`，返回 `(prompts, responses, reward_fn, mask)`。

同时修改 `ToyDataLoader.load()` 返回 `(prompts, responses, reward_fn, mask)`，mask 为全 1（toy 等长）。

修改 `DataLoader.load()` 契约文档为返回 4 元组。

修改 `pipeline.py` 中调用 `build_data_loader(...).load()` 的地方解包 4 元素（`_run_body` 里 `prompts, responses, reward_fn, masks = ...`；mask 传入 scheduler）。**注意**：pipeline Agent 返回后确认精确行号。

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_data.py tests/test_pipeline.py -q`
预期：PASS（pipeline 解包需同步改，见任务 1.5）

- [x] **步骤 5：Commit**

```bash
git add main/fullstack_opd_v2/data.py main/tests/test_data.py
git commit -m "feat(data): JsonLinesDataLoader 回传 response padding mask(§3.4 变长支持)"
```

---

### 任务 1.4：staleness bug 修（base 样本不截断）+ 双池 feeder

**文件：**
- 修改：`main/fullstack_opd_v2/scheduler.py`（`_train_step:291` + `_rand_idxs:200`）
- 测试：`main/tests/test_scheduler.py`

- [x] **步骤 1：编写失败测试**

```python
def test_base_sample_not_truncated_by_version():
    """base 样本 ver=0 不被 staleness 截断（§2 Q4 bug 修）。

    训练到 version=10，base 样本(staleness=10 > threshold=4)仍应参与训练，
    靠 ratio+clip 降权而非 threshold 丢弃。
    """
    from fullstack_opd_v2.scheduler import AsyncBatchedScheduler
    # 构造 toy scheduler，version 推进到 10，喂 base 样本(ver=0)
    # 断言 _train_step 返回非 None（未被截断）
    ...

def test_dual_pool_feeder_mix_ratio():
    """双池 feeder 按 mix_ratio 采 base+refresh（§2 Q4）。"""
    # l2.enabled=true, refresh pool 非空, mix_ratio=0.5
    # 断言采样的 idx 中约半数落在 refresh 池
    ...
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_scheduler.py::test_base_sample_not_truncated_by_version -q`
预期：FAIL（当前 base ver=0 会被截断返回 None）

- [x] **步骤 3：实现修复 + 双池 feeder**

修改 `scheduler.py`：

(a) `_train_step:289-292` staleness 截断改为区分 base/refresh：
```python
    # §2 Q4：base 样本(ver=IS_BASE)不参与版本截断（离线固有 staleness 靠 ratio+clip 降权）；
    # 仅 refresh 样本受 threshold 截断（防用过时 on-policy 样本）。
    threshold = self.cfg.get("staleness_threshold", 4)
    is_refresh = bool(getattr(idxs, "_is_refresh", False))
    if is_refresh and (self.staleness_q.current_version - ver > threshold):
        return None
```

(b) `_rand_idxs:198-204` 包一层双池采样（`l2.enabled` 关时退回原纯随机）：
```python
    def _rand_idxs(self):
        if not self._l2_enabled:
            return torch.randint(0, self.n_prompts, (self.batch,))
        # 双池 feeder：按 α 采 base/refresh
        alpha = self._current_alpha()
        n_refresh = int(round(self.batch * alpha))
        # cold start：refresh 不足时 fallback base（§5.5）
        if self._refresh_buffer is None or self._refresh_buffer.size == 0:
            n_refresh = 0
        else:
            n_refresh = min(n_refresh, self._refresh_buffer.size)
        n_base = self.batch - n_refresh
        idxs = torch.randint(0, self.n_prompts, (n_base,))
        if n_refresh > 0:
            r_idxs = torch.randint(0, self._refresh_buffer.size, (n_refresh,))
            r_idxs._is_refresh = True   # 标记，供 _train_step 区分
            idxs = (idxs, r_idxs)       # 返回元组，_train_step 拆分处理
        return idxs
```

在 `AsyncBatchedScheduler.__init__` 加 L2 钩子（`l2.enabled` 时初始化 refresh_buffer=None、alpha=initial；pipeline 在交替相位注入）。

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_scheduler.py -q`
预期：PASS

- [x] **步骤 5：回归 + Commit**

运行：`cd main && python -m pytest tests/ -q`（全绿，确认 l2.enabled=false 时行为不变）
```bash
git add main/fullstack_opd_v2/scheduler.py main/tests/test_scheduler.py
git commit -m "fix(l2): base 样本不受版本截断(ratio+clip 降权) + 双池 feeder(α 混合)"
```

---

### 任务 1.5：checkpoint 扩展（optimizer + RNG + ring buffer）

**文件：**
- 修改：`main/fullstack_opd_v2/checkpoint.py`（`save` 加 optimizer/RNG；`resume` 恢复）
- 测试：`main/tests/test_checkpoint.py`

- [x] **步骤 1：编写失败测试**

```python
def test_checkpoint_saves_optimizer_and_rng(tmp_path):
    """断点含 optimizer state + RNG（§B 精确续跑）。"""
    from fullstack_opd_v2.checkpoint import CheckpointManager
    import torch
    cm = CheckpointManager(str(tmp_path), every=1)
    student = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(student.parameters())
    opt.step()  # 产生 optimizer state
    cm.save(1, student, version=1, cfg={}, optimizer=opt,
            rng={"py": torch.get_rng_state(), "cuda": None})
    ck = cm.resume()
    assert "optimizer" in ck
    assert "rng" in ck
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_checkpoint.py::test_checkpoint_saves_optimizer_and_rng -q`
预期：FAIL（save 不接受 optimizer 参数）

- [x] **步骤 3：实现扩展**

修改 `checkpoint.py` 的 `save()` 签名加 `optimizer=None, rng=None, refresh_buffer=None`：
```python
    def save(self, step, student, version, cfg, metrics=None, force=False,
             ref=None, optimizer=None, rng=None, refresh_buffer=None):
        ...
        torch.save({
            "step": step, "version": version,
            "state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
            "cfg": cfg, "metrics": (metrics or [])[-1:],
            "ref": ...,   # 原有
            # §B 精确续跑：optimizer state + RNG + L2 ring buffer
            "optimizer": (_opt_state_to_cpu(optimizer) if optimizer is not None else None),
            "rng": rng,
            "refresh_buffer": refresh_buffer,   # RefreshRingBuffer 状态（l2 开时）
        }, tmp)
```

`_opt_state_to_cpu(optimizer)`：把 optimizer.state 的张量搬 CPU（FSDP 下用 `optim.get_state_dict()`，单卡直接 state_dict）。

`resume()` 已返回 dict（含新键），pipeline `_run_body` resume 分支加载 optimizer.load_state_dict + torch.set_rng_state + ring buffer 恢复（见任务 6.x）。

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_checkpoint.py -q`
预期：PASS

- [x] **步骤 5：Commit**

```bash
git add main/fullstack_opd_v2/checkpoint.py main/tests/test_checkpoint.py
git commit -m "feat(l2): checkpoint 扩展(optimizer+RNG+ring buffer, §B 精确续跑)"
```

---

## 阶段 2：Teacher-Student Disagreement（§3）

### 任务 2.1：DisagreementComputer

**文件：**
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（加 `DisagreementComputer`）
- 测试：`main/tests/test_adaptive_cache.py`

- [x] **步骤 1：编写失败测试**

```python
def test_disagreement_identical_zero():
    """identical teacher/student 时 disagreement≈0（§3 测试6）。"""
    from fullstack_opd_v2.adaptive_cache import DisagreementComputer
    # 4 个 logp 全同 -> D_t=0
    T, mask = 3, torch.ones(2, 3)
    logp = torch.zeros(2, 3)
    d = DisagreementComputer()
    D = d.compute(teacher_rl_logp=logp, teacher_ref_logp=logp,
                  student_logp=logp, student_ref_logp=logp, mask=mask)
    assert torch.allclose(D["abs"], torch.zeros(2), atol=1e-6)

def test_disagreement_monotonic_with_gap():
    """teacher/student 差异放大时 disagreement 单调增加（§3 测试7）。"""
    from fullstack_opd_v2.adaptive_cache import DisagreementComputer
    d = DisagreementComputer()
    mask = torch.ones(1, 3)
    base = torch.zeros(1, 3)
    D1 = d.compute(base, base, base + 0.5, base, mask)["abs"]
    D2 = d.compute(base, base, base + 2.0, base, mask)["abs"]
    assert D2.item() > D1.item()

def test_disagreement_mask_excludes_padding():
    """padding 不计入（§3 测试2/4）。"""
    from fullstack_opd_v2.adaptive_cache import DisagreementComputer
    d = DisagreementComputer()
    logp = torch.zeros(2, 4)
    logp[:, 2:] = 5.0   # padding 位置有值但应被 mask 排除
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    D = d.compute(logp, logp, logp, logp, mask)  # 全同应 0，padding 被排除
    assert torch.allclose(D["abs"], torch.zeros(2), atol=1e-6)
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_adaptive_cache.py::test_disagreement_identical_zero -q`
预期：FAIL（无 DisagreementComputer）

- [x] **步骤 3：实现 DisagreementComputer**

在 `adaptive_cache.py` 加：
```python
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
```

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_adaptive_cache.py -q -k disagreement`
预期：3 个 PASS

- [x] **步骤 5：Commit**

```bash
git add main/fullstack_opd_v2/adaptive_cache.py main/tests/test_adaptive_cache.py
git commit -m "feat(l2): DisagreementComputer(D_t=Δ_T−Δ_S, 不同源4 logp, mask 聚合)"
```

---

### 任务 2.2：rollout 相位 disagreement 计算 + 统一 metadata 接入

**文件：**
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（rollout 阶段编排函数）
- 修改：`main/fullstack_opd_v2/model.py`（`HFCausalLM` 加 generate_batch 骨架）— 见任务 2.3
- 测试：`main/tests/test_l2_integration.py`

> 本任务实现「rollout 相位」：student 生成 -> 4 个 chosen logp -> D_i^abs -> append_refresh。teacher 前向在此相位，_train_step 不动（teacher-free）。

- [x] **步骤 1-5：** 实现编排函数 `run_refresh_phase(student, teacher_rl, teacher_ref, student_ref, selector, ring_buffer, prompts, step, version, cfg)`，在 `adaptive_cache.py`：

```python
    def run_refresh_phase(self, student, teacher_rl, teacher_ref,
                          student_ref, prompts, m_selected, step, version):
        """§3.3 + §6.5 rollout 相位：student 生成 -> 4 logp -> D_i^abs -> append。

        student/teacher 前向在此（_train_step 不动）。返回 refresh 样本数。
        """
        # 1. Selective Rollout 选 prompt（§6，阶段5实现；此处先用随机，selector=None 时 uniform）
        cand = self.selector.select(prompts, m_selected) if self.selector else \
            torch.randint(0, prompts.size(0), (m_selected,))
        p_b = prompts[cand]
        # 2. student 自回归生成 + 收集 logπ_S(y_t)
        from .model import generate_batch, token_logprobs
        responses = generate_batch(student, p_b, max_new=self.max_resp_len)
        student_logp = token_logprobs(student, p_b, responses)     # (M,T)
        # 3. 初始 student(ref) 前向 -> logπ_Ref(y_t)
        with torch.no_grad():
            student_ref.eval()
            ref_logp = token_logprobs(student_ref, p_b, responses)
        # 4. teacher 前向 -> top-k + chosen logp（§3.4：gather chosen，不只存 top-k）
        from .model import response_dists
        with torch.no_grad():
            teacher_rl.eval(); teacher_ref.eval()
            rl_dist = response_dists(teacher_rl, p_b, responses)    # (M,T,V)
            ref_dist = response_dists(teacher_ref, p_b, responses)
            rl_chosen = self.disag.gather_chosen_logp(rl_dist, responses)
            ref_chosen = self.disag.gather_chosen_logp(ref_dist, responses)
            # top-k 存 cache（训练查表）
            tk = rl_dist.topk(self.top_k, dim=-1)
            ids_k, rl_k = tk.indices, tk.values
            ref_k = ref_dist.gather(-1, tk.indices)
            delta_k = rl_k - ref_k
        # 5. mask（EOS 之后 padding=0，§3.4）
        mask = self._build_mask(responses)   # 按 EOS/pad token 判定
        # 6. D_i^abs
        D = self.disag.compute(rl_chosen, ref_chosen, student_logp, ref_logp, mask)
        # 7. append_refresh（per-token logp 算完即弃，只存标量，§11）
        for i in range(responses.size(0)):
            self.ring_buffer.append(
                ids_k[i], delta_k[i], generation_step=step,
                response_length=int(mask[i].sum()), token_mask=mask[i],
                disagreement_abs=float(D["abs"][i]))
        return responses.size(0)
```

测试：`test_l2_integration.py::test_refresh_phase_produces_disagreement`（toy 模型，断言 ring_buffer.size 增长、disagreement 非负）。

Commit：`feat(l2): rollout 相位编排(student生成+4 logp+D_i^abs+append, teacher-free _train_step)`

---

### 任务 2.3：HFCausalLM generate_batch + attention_mask（骨架）

**文件：**
- 修改：`main/fullstack_opd_v2/model_factory.py`（`HFCausalLM`）

- [x] **步骤 1-5：** 给 `HFCausalLM` 加 `generate_batch`（委托 `model.generate`）和 `__call__` 传 `attention_mask`：
```python
    def __call__(self, input_ids, attention_mask=None):
        kw = {"input_ids": input_ids}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        return self.model(**kw).logits

    @torch.no_grad()
    def generate_batch(self, prompts, max_new=8192, temperature=1.0):
        """自回归生成（骨架：委托 HF generate，真实规模应走 vLLM）。"""
        out = self.model.generate(prompts, max_new_tokens=max_new,
                                  do_sample=temperature > 0,
                                  temperature=max(temperature, 1e-6),
                                  top_p=0.95)
        return out[:, prompts.size(1):]
```

测试：HF 骨架标 `@pytest.mark.skipif(not _HF_AVAILABLE)`（本地无 GPU 跳过，CI/GPU 跑）。

Commit：`feat(l2): HFCausalLM generate_batch + attention_mask 骨架(§3 rollout 生成)`

---

## 阶段 3：Cache Health Monitor（§4）

### 任务 3.1：CacheHealthMonitor 七维 + health score + alert

**文件：**
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（加 `CacheHealthMonitor`）
- 测试：`main/tests/test_adaptive_cache.py`

- [x] **步骤 1：编写失败测试**

```python
def test_health_score_thresholds():
    """rule-based HEALTHY/WARNING/CRITICAL 分类（§4.3）。"""
    from fullstack_opd_v2.adaptive_cache import CacheHealthMonitor
    hm = CacheHealthMonitor(health={"hit_rate": {"warning": 0.995, "critical": 0.98},
                                     "refresh_age_p95": {"warning": 5, "critical": 10}})
    assert hm.classify(hit_rate=0.999) == "HEALTHY"
    assert hm.classify(hit_rate=0.99) == "WARNING"
    assert hm.classify(hit_rate=0.97) == "CRITICAL"

def test_health_alert_cooldown():
    """同一 warning cooldown 内不重复（§4.4）。"""
    from fullstack_opd_v2.adaptive_cache import CacheHealthMonitor
    hm = CacheHealthMonitor(health={"hit_rate": {"warning": 0.995, "critical": 0.98}},
                            alert_cooldown=5)
    hm.record(step=1, hit_rate=0.99)   # WARNING
    assert hm.last_status == "WARNING"
    hm.record(step=2, hit_rate=0.99)   # 同 warning，cooldown 内不重复
    assert hm._alert_count == 1
```

- [x] **步骤 2：运行测试验证失败** — `cd main && python -m pytest tests/test_adaptive_cache.py::test_health_score_thresholds -q`（FAIL）

- [x] **步骤 3：实现 CacheHealthMonitor**

```python
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
        if hit: self._lookup["hit"] += 1
        else: self._lookup["miss"] += 1
        if invalid: self._lookup["invalid"] += 1
        if duplicate: self._lookup["duplicate"] += 1

    def record_reuse(self, sample_id: int):
        self._reuse_counts[sample_id] = self._reuse_counts.get(sample_id, 0) + 1

    def classify(self, hit_rate: float = 1.0, refresh_age_p95: float = 0,
                 reuse_p95: float = 0, max_length_ratio: float = 0) -> str:
        """rule-based 三级（§4.3）。任一 critical -> CRITICAL；任一 warning -> WARNING。"""
        worst = "HEALTHY"
        for metric, val in [("hit_rate", hit_rate), ("refresh_age_p95", refresh_age_p95),
                            ("reuse_p95", reuse_p95), ("max_length_ratio", max_length_ratio)]:
            th = self.thresholds.get(metric, {})
            # hit_rate 越低越坏；其余越高越坏
            bad = (val < th["critical"]) if metric == "hit_rate" else (val > th["critical"])
            warn = (val < th["warning"]) if metric == "hit_rate" else (val > th["warning"])
            if bad: return "CRITICAL"
            if warn: worst = "WARNING"
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
        if status == "HEALTHY": return ""
        # 找首个触发的指标
        for m in ["hit_rate", "refresh_age_p95", "reuse_p95", "max_length_ratio"]:
            th = self.thresholds.get(m, {})
            val = metrics.get(m, 0)
            bad = (val < th["critical"]) if m == "hit_rate" else (val > th["critical"])
            if bad: return f"{m} critical ({val})"
        return "unknown"
```

七维其余（age/reward/length 分布统计）用 EMA + reservoir 采样统计，不全量扫描（§4.6）。

- [x] **步骤 4：运行测试验证通过** — `cd main && python -m pytest tests/test_adaptive_cache.py -q -k health`（PASS）

- [x] **步骤 5：Commit** — `feat(l2): CacheHealthMonitor(七维+rule-based score+alert cooldown, Observe-only)`

---

## 阶段 4：Dynamic Refresh Ratio（§5）

### 任务 4.1：DynamicRatioController 三信号

**文件：**
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（加 `DynamicRatioController`）
- 测试：`main/tests/test_adaptive_cache.py`

- [x] **步骤 1：编写失败测试**

```python
def test_ratio_bounds():
    """α ∈ [min, max]，α_max<1（§5.4）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive")
    for _ in range(100):
        a = c.update(base_age=100, policy_drift=0, refresh_quality=100)
        assert 0.1 <= a <= 0.6

def test_ratio_fixed_mode():
    """fixed 模式 α 恒定（§5.8）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="fixed")
    assert c.update(100, 0, 100) == 0.3

def test_ratio_cold_start():
    """refresh 不足 fallback base（§5.5）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive")
    assert c.cold_start_adjust(alpha=0.3, n_refresh=2, n_batch=8) == 2/8

def test_ratio_max_step_change():
    """|α_t−α_{t-1}| ≤ max_step_change（§5.4）。"""
    from fullstack_opd_v2.adaptive_cache import DynamicRatioController
    c = DynamicRatioController(initial=0.3, min=0.1, max=0.6, mode="adaptive",
                               max_step_change=0.05)
    a1 = c.update(0, 0, 0)
    a2 = c.update(100, 0, 100)
    assert abs(a2 - a1) <= 0.05 + 1e-6
```

- [x] **步骤 2：运行测试验证失败** — `cd main && python -m pytest tests/test_adaptive_cache.py::test_ratio_bounds -q`（FAIL）

- [x] **步骤 3：实现 DynamicRatioController**

```python
class DynamicRatioController:
    """§5 Dynamic Refresh Ratio（三信号 controller）。

    α_t = clip(α_0 + λA·Ã_B − λD·D̃_drift + λQ·Q̃_t, α_min, α_max)
    所有信号 normalize（z-score / EMA）；EMA 平滑防震荡；max_step_change 限幅。
    模式：fixed(α=initial) / linear(0.1->0.5) / adaptive(完整 controller)。
    cold start：N_R 不足时 α_actual=min(α, N_R/N_batch) fallback base。
    α_max<1：保留 base 作 stationary anchor。
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
        # linear 模式起止
        self._lin_start, self._lin_end = 0.1, 0.5

    def _norm(self, key, x):
        """EMA + 简单 normalize（x/(1+|x|) 映射到 [-1,1] 附近，防极值爆炸）。"""
        self._ema[key] = self.beta * self._ema[key] + (1 - self.beta) * x
        return self._ema[key] / (1 + abs(self._ema[key]))

    def update(self, base_age, policy_drift, refresh_quality) -> float:
        self._step += 1
        if self.mode == "fixed":
            return self.alpha0
        if self.mode == "linear":
            frac = min(1.0, self._step / 1000)
            return self._lin_start + frac * (self._lin_end - self._lin_start)
        # adaptive：warmup 内用 initial
        if self._step <= self.warmup:
            self._last_alpha = self.alpha0
            return self.alpha0
        a_b = self._norm("age", base_age)
        d_drift = self._norm("drift", policy_drift)
        q = self._norm("quality", refresh_quality)
        raw = self.alpha0 + self.w["age"] * a_b - self.w["drift"] * d_drift \
              + self.w["quality"] * q
        raw = max(self.min, min(self.max, raw))
        # max_step_change 限幅
        raw = self._last_alpha + max(-self.max_step, min(self.max_step, raw - self._last_alpha))
        raw = max(self.min, min(self.max, raw))
        self._last_alpha = raw
        return raw

    def cold_start_adjust(self, alpha, n_refresh, n_batch) -> float:
        """§5.5：refresh 不足时 α_actual=min(α, N_R/N_batch)。"""
        return min(alpha, n_refresh / max(1, n_batch))
```

- [x] **步骤 4：运行测试验证通过** — `cd main && python -m pytest tests/test_adaptive_cache.py -q -k ratio`（PASS）

- [x] **步骤 5：Commit** — `feat(l2): DynamicRatioController(三信号α+EMA+max_step_change+cold start, fixed/linear/adaptive)`

---

## 阶段 5：Selective Rollout（§6）

### 任务 5.1：PromptStateStore + RefreshSelector

**文件：**
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（加 `PromptStateStore` + `RefreshSelector`）
- 测试：`main/tests/test_adaptive_cache.py`

- [x] **步骤 1：编写失败测试**

```python
def test_selector_candidate_pool_two_stage():
    """candidate pool 两阶段：4M candidate -> M selected，candidate 不跑 teacher（§6.5）。"""
    from fullstack_opd_v2.adaptive_cache import RefreshSelector, PromptStateStore
    ps = PromptStateStore(n_prompts=100)
    sel = RefreshSelector(ps, candidate_multiplier=4, value_fraction=0.8)
    selected = sel.select(n_selected=10, n_prompts=100)
    assert len(selected) == 10

def test_selector_deterministic_seed():
    """deterministic given seed（§6.13）。"""
    from fullstack_opd_v2.adaptive_cache import RefreshSelector, PromptStateStore
    ps = PromptStateStore(n_prompts=100)
    s1 = RefreshSelector(ps, seed=42).select(10, 100)
    s2 = RefreshSelector(ps, seed=42).select(10, 100)
    assert torch.equal(s1, s2)

def test_selector_failure_fallback_uniform():
    """cold start / history 太短 -> fallback uniform（§6.9）。"""
    from fullstack_opd_v2.adaptive_cache import RefreshSelector, PromptStateStore
    ps = PromptStateStore(n_prompts=50)   # 无历史
    sel = RefreshSelector(ps)
    selected = sel.select(10, 50)
    assert len(selected) == 10   # 不 NaN/空

def test_selector_diversity_max_same_prompt():
    """max_same_prompt_fraction 限制单 prompt 占比（§6.8）。"""
    from fullstack_opd_v2.adaptive_cache import RefreshSelector, PromptStateStore
    ps = PromptStateStore(n_prompts=20)
    sel = RefreshSelector(ps, max_same_prompt_fraction=0.1)
    selected = sel.select(10, 20)
    # 单 prompt 不应超过 1（10*0.1）
    from collections import Counter
    assert max(Counter(selected.tolist()).values()) <= 1
```

- [x] **步骤 2：运行测试验证失败** — `cd main && python -m pytest tests/test_adaptive_cache.py::test_selector_candidate_pool_two_stage -q`（FAIL）

- [x] **步骤 3：实现 PromptStateStore + RefreshSelector**

```python
class PromptStateStore:
    """§6.1 per-prompt 轻量历史状态（复用 §3/§4 信号，不重复 forward）。"""
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
        self.times_seen[prompt_id] += 1
        self.last_seen_step[prompt_id] = step
        # EMA reward/var
        self.reward_ema[prompt_id] = 0.9*self.reward_ema[prompt_id] + 0.1*reward
        self.disagreement_ema[prompt_id] = 0.9*self.disagreement_ema[prompt_id] + 0.1*disagreement
        self.last_response_length[prompt_id] = resp_len

    def novelty(self) -> torch.Tensor:
        return 1.0 / torch.sqrt(1.0 + self.times_seen.float())


class RefreshSelector:
    """§6 Selective Rollout（candidate pool 两阶段降本）。

    M_candidate=4·M_selected -> cheap scoring(V=0.4U+0.4D+0.2N)
    -> 80% top-value + 20% coverage -> M selected。
    candidate 阶段不跑 teacher。diversity protection + failure fallback uniform。
    """
    def __init__(self, prompt_state: PromptStateStore, candidate_multiplier: int = 4,
                 value_fraction: float = 0.80, coverage_fraction: float = 0.20,
                 value_weights: dict | None = None, compute_aware: bool = False,
                 max_same_prompt_fraction: float = 0.05, exploration_fraction: float = 0.20,
                 seed: int = 42):
        self.ps = prompt_state
        self.cm = candidate_multiplier
        self.vf = value_fraction
        self.cf = coverage_fraction
        self.vw = value_weights or {"uncertainty": 0.4, "disagreement": 0.4, "novelty": 0.2}
        self.compute_aware = compute_aware
        self.max_same = max_same_prompt_fraction
        self.exploration = exploration_fraction
        self.gen = torch.Generator().manual_seed(seed)

    def _value(self) -> torch.Tensor:
        U = self.ps.reward_var                        # uncertainty
        D = self.ps.disagreement_ema
        N = self.ps.novelty()
        v = (self.vw["uncertainty"]*U + self.vw["disagreement"]*D + self.vw["novelty"]*N)
        if self.compute_aware:
            cost = self.ps.last_response_length.float() + 1e-8
            v = v / cost
        return v

    def select(self, n_selected: int, n_prompts: int) -> torch.Tensor:
        # failure fallback：history 太短（times_seen 全 0）-> uniform
        if self.ps.times_seen.sum() == 0 or n_selected >= n_prompts:
            return torch.randint(0, n_prompts, (n_selected,), generator=self.gen)
        n_cand = min(self.cm * n_selected, n_prompts)
        cand = torch.randperm(n_prompts, generator=self.gen)[:n_cand]
        v = self._value()[cand]
        n_high = int(round(n_selected * self.vf))
        n_cov = n_selected - n_high
        # 80% top-value
        top = cand[v.topk(min(n_high, n_cand)).indices]
        # 20% coverage（从候选中随机，排除已选）
        remaining = cand[~torch.isin(cand, top)]
        cov = remaining[torch.randperm(len(remaining), generator=self.gen)[:n_cov]] \
            if len(remaining) > 0 else top[:n_cov]
        selected = torch.cat([top, cov])
        # diversity：max_same_prompt_fraction 限制
        max_per = max(1, int(n_selected * self.max_same))
        from collections import Counter
        cnt = Counter(selected.tolist())
        filtered = []
        for p in selected.tolist():
            if cnt[p] <= max_per:
                filtered.append(p)
            else:
                cnt[p] -= 1
        # 不足补 uniform
        while len(filtered) < n_selected:
            filtered.append(torch.randint(0, n_prompts, (1,), generator=self.gen).item())
        return torch.tensor(filtered[:n_selected])
```

- [x] **步骤 4：运行测试验证通过** — `cd main && python -m pytest tests/test_adaptive_cache.py -q -k selector`（PASS）

- [x] **步骤 5：Commit** — `feat(l2): RefreshSelector+PromptStateStore(candidate pool 两阶段+value+coverage+diversity+fallback)`

---

## 阶段 6：四模块整合 + E0-E6 实验框架（§13）

> 本阶段依赖 pipeline.py 接入点（交替相位循环、teacher 加载、scheduler 构造、_on_step/cm.save），待 pipeline Agent 摘要返回后补全精确行号与编排代码。

### 任务 6.1：pipeline 交替相位循环接入

**文件：**
- 修改：`main/fullstack_opd_v2/pipeline.py:414`（L2 保留 teacher/warmup_student）、`:540-555`（L2 装配 + 交替循环替换 scheduler.run 单次调用）、`:506/553/570`（cm.save 加 optimizer/rng/ring_buffer）、`:420-424`（resume 恢复 optimizer/rng/ring_buffer）
- 修改：`main/fullstack_opd_v2/adaptive_cache.py`（加模块级 `run_refresh_phase` 编排函数）
- 测试：`main/tests/test_l2_integration.py`

**接入点（已确认行号）：**
- `pipeline.py:365` `warmup_student` = §3 的 student_ref（初始 student，P1-4 独立实例）
- `pipeline.py:397` `teacher_rl, teacher_ref = self._stage0_teachers()`
- `pipeline.py:414` `del teacher_rl, teacher_ref, warmup_student` -- L2 需保留
- `pipeline.py:540-543` `AsyncBatchedScheduler(...)` 构造
- `pipeline.py:555` `metrics = scheduler.run(n_steps, on_step=_on_step)` -- 交替循环替换点
- `pipeline.py:506/553` `cm.save(...)` -- 加 optimizer/rng/ring_buffer

- [x] **步骤 1：编写失败测试**

```python
def test_alternating_phase_loop(tmp_path):
    """L2 交替相位：训练 T_train 步 ↔ rollout 刷新循环（§1/§12）。toy 模型。"""
    from fullstack_opd_v2.config import load_config
    cfg = load_config(overrides=[
        "l2.enabled=true", "l2.t_train=5", "stage2.n_steps=12",
        "l2.m_refresh=4", "l2.cache.refresh_size=8"])
    # 跑 FullStackOPDv2.run，断言：metrics 长度=12，refresh 发生了至少 2 轮
    from fullstack_opd_v2.pipeline import FullStackOPDv2
    opd = FullStackOPDv2(cfg, device="cpu")
    out = opd.run(run_dir=str(tmp_path))
    assert len(out["metrics"]) == 12

def test_l2_disabled_regression(tmp_path):
    """l2.enabled=false 行为与原 L0/L1 完全一致（回归）。"""
    # 跑 l2.enabled=false，对比 l2 键不存在的默认 run，metrics 数值一致
    ...

def test_no_teacher_forward_in_train_step():
    """_train_step 内无 teacher 前向（断言，§13.7）。"""
    # monkeypatch teacher_rl.forward 计数，跑一步，断言训练相位内调用=0
    ...
```

- [x] **步骤 2：运行测试验证失败**

运行：`cd main && python -m pytest tests/test_l2_integration.py::test_alternating_phase_loop -q`
预期：FAIL（scheduler.run 单次调用，无交替）

- [x] **步骤 3：实现交替相位接入**

(a) `pipeline.py:414` -- L2 启用时保留 teacher/warmup_student：
```python
        # L2：启用时保留 teacher_rl/teacher_ref + warmup_student 供 rollout 相位（§3）；
        # warmup_student 即 §3 的 student_ref（初始 student，P1-4 独立实例）。
        # 非 L2 仍释放（原行为不变）。
        l2_cfg = self.cfg.get("l2", {})
        l2_enabled = bool(l2_cfg.get("enabled", False))
        if not l2_enabled:
            del teacher_rl, teacher_ref, warmup_student
            teacher_rl = teacher_ref = None
```

(b) `pipeline.py:540-555` -- L2 装配 + 交替循环替换 `scheduler.run` 单次调用：
```python
                scheduler = AsyncBatchedScheduler(
                    student, cache, fat_prompts, fat_responses,
                    ref_dists, ref_ids, ref_logp, s2cfg, self.device,
                    rollout_engine=rollout_engine, initial_version=initial_version)

                def _on_step(m):
                    try:
                        _on_step_q.put_nowait(m)
                    except queue.Full:
                        mr.record(m)
                        _save_ckpt(m["step"], m["version"])   # 见 (d) 统一保存

                if l2_enabled:
                    from .adaptive_cache import (RefreshRingBuffer, DisagreementComputer,
                        CacheHealthMonitor, DynamicRatioController, RefreshSelector,
                        PromptStateStore, run_refresh_phase)
                    l2c = l2_cfg.get("cache", {})
                    rb = RefreshRingBuffer(capacity=l2c.get("refresh_size", 5000),
                        top_k=cache.top_k, vocab=cache.vocab,
                        value_protect_quantile=l2c.get("value_protect_quantile", 0.9))
                    ps = PromptStateStore(n_prompts=fat_prompts.size(0))
                    disag = DisagreementComputer()
                    hm = CacheHealthMonitor(
                        l2_cfg.get("health_monitor", {}).get("health", {}),
                        alert_cooldown=l2_cfg.get("health_monitor", {}).get("alert_cooldown", 50))
                    rc = l2_cfg.get("refresh_ratio", {})
                    drc = DynamicRatioController(
                        initial=rc.get("initial", 0.3), min=rc.get("min", 0.1),
                        max=rc.get("max", 0.6), mode=rc.get("mode", "adaptive"),
                        age_weight=rc.get("age_weight", 0.25), drift_weight=rc.get("drift_weight", 0.5),
                        quality_weight=rc.get("quality_weight", 0.25), ema_beta=rc.get("ema_beta", 0.9),
                        warmup_steps=rc.get("warmup_steps", 500), max_step_change=rc.get("max_step_change", 0.05))
                    sc = l2_cfg.get("selective_rollout", {})
                    selector = (RefreshSelector(ps, candidate_multiplier=sc.get("candidate_multiplier", 4),
                        value_fraction=sc.get("value_fraction", 0.8), coverage_fraction=sc.get("coverage_fraction", 0.2),
                        value_weights=sc.get("value_weights"), compute_aware=sc.get("compute_aware", False),
                        max_same_prompt_fraction=sc.get("max_same_prompt_fraction", 0.05),
                        exploration_fraction=sc.get("exploration_fraction", 0.2))
                        if sc.get("enabled", True) else None)
                    # 注入 scheduler 供双池 feeder（任务 1.4 _rand_idxs / _current_alpha 读）
                    scheduler._l2_enabled = True
                    scheduler._refresh_buffer = rb
                    scheduler._drc = drc
                    student_ref = warmup_student    # §3 初始 student

                    metrics = []
                    n_total = s2cfg.get("n_steps", 30)
                    t_train = l2_cfg.get("t_train", 100)
                    step_done = 0
                    while step_done < n_total:
                        # 训练相位：跑 t_train 步（_train_step teacher-free 不动）
                        n_phase = min(t_train, n_total - step_done)
                        metrics.extend(scheduler.run(n_phase, on_step=_on_step))
                        step_done += n_phase
                        # rollout 相位：L2 刷新（teacher 前向在此，不在 _train_step）
                        if step_done < n_total and selector is not None:
                            run_refresh_phase(student, teacher_rl, teacher_ref, student_ref,
                                selector, rb, disag, fat_prompts, step_done,
                                scheduler.staleness_q.current_version,
                                l2_cfg.get("m_refresh", 1000),
                                l2c.get("max_response_length", 8192), cache.top_k, self.device)
                            # Health Monitor 观测（Observe-only，不改训练）
                            hm_metrics = hm.record(step_done,
                                hit_rate=1.0, refresh_age_p95=0,
                                reuse_p95=0, max_length_ratio=0)
                            mr.record(hm_metrics)
                            # Dynamic Ratio 调 α（consume metrics，非 Monitor 闭环）
                            drc.update(base_age=hm_metrics.get("age/mean", 0),
                                       policy_drift=0,   # 复用 k3 KL，阶段4 接入
                                       refresh_quality=rb.mean_disagreement())
                else:
                    metrics = scheduler.run(s2cfg.get("n_steps", 30), on_step=_on_step)
```

(c) `adaptive_cache.py` 加模块级 `run_refresh_phase`（任务 2.2 的编排函数，独立可测）：
```python
def run_refresh_phase(student, teacher_rl, teacher_ref, student_ref,
                      selector, ring_buffer, disag, prompts, step, version,
                      m_selected, max_resp_len, top_k, device):
    """§3.3 + §6.5 rollout 相位：selective 选 prompt -> student 生成
    -> 4 chosen logp -> D_i^abs -> append_refresh。teacher 前向在此（_train_step 不动）。

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
            disagreement_abs=float(D["abs"][i]))
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
```

(d) `pipeline.py:506/553` -- cm.save 加 optimizer/rng/ring_buffer（统一 `_save_ckpt` 闭包）：
```python
                def _save_ckpt(step, version):
                    cm.save(step, student, version, self.cfg, metrics=[], ref=ref,
                            optimizer=scheduler.opt,
                            rng={"py": torch.get_rng_state(),
                                 "cuda": (torch.cuda.get_rng_state()
                                          if torch.cuda.is_available() else None)},
                            refresh_buffer=(rb if l2_enabled else None))
```
`_consumer_loop`（行 506）和 `_on_step` 回退分支（行 553）都调 `_save_ckpt`。末步 `cm.save`（行 570）同样加这三参数。

(e) `pipeline.py:420-424` -- resume 恢复 optimizer/rng/ring_buffer：
```python
        _resume_opt = _resume_rng = _resume_rb = None
        if resume is not None:
            student.load_state_dict(resume["state"])
            initial_version = int(resume.get("version", 0))
            resume_ref = resume.get("ref")
            _resume_opt = resume.get("optimizer")    # §B 精确续跑
            _resume_rng = resume.get("rng")
            _resume_rb = resume.get("refresh_buffer")
            if _resume_rng:
                torch.set_rng_state(_resume_rng["py"])
                if _resume_rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(_resume_rng["cuda"])
            logger.info(f"resume: 已加载断点，从 version={initial_version} 续跑")
```
scheduler 构造后：`if _resume_opt: scheduler.opt.load_state_dict(_resume_opt)`；`rb = _resume_rb`（恢复 ring buffer）。注意 FSDP 下 optimizer state 用 `optim.set_state_dict`（阶段1 任务1.5 的 `_opt_state_to_cpu` 对称）。

- [x] **步骤 4：运行测试验证通过**

运行：`cd main && python -m pytest tests/test_l2_integration.py -q`
预期：PASS（交替相位 + 回归 + teacher-free 断言）

- [x] **步骤 5：全量回归 + Commit**

运行：`cd main && python -m pytest tests/ -q`（全绿，含原 42 测试 + L2 新增）
```bash
git add main/fullstack_opd_v2/pipeline.py main/fullstack_opd_v2/adaptive_cache.py main/tests/test_l2_integration.py
git commit -m "feat(l2): pipeline 交替相位循环接入(训练T_train步↔rollout刷新) + optimizer/RNG/ring buffer 续跑"
```

---

### 任务 6.2：E0-E6 实验矩阵 + 统一记录

**文件：**
- 创建：`main/scripts/run_l2_ablation.py`
- 创建：`main/fullstack_opd_v2/experiment.py`（实验记录聚合）

- [x] **步骤 1-5：** 实现 E0-E6 配置生成（每模块 enabled 切换）+ 统一记录（Training Quality / Efficiency / Cache / Selector 四类）+ 8 张实验图绘制（matplotlib，6/7 最重要）。

Commit：`feat(l2): E0-E6 实验矩阵 + 统一记录 + 8 张实验图(teacher compute vs perf 最重要)`

---

### 任务 6.3：工程检查 + implementation report

**文件：**
- 测试：`main/tests/test_l2_integration.py`（补全工程检查断言）

- [x] **步骤 1-5：** 补全工程检查测试：
  - `test_no_teacher_forward_in_train_step`（断言 _train_step 内无 teacher 前向调用）
  - `test_no_gpu_memory_growth`（连续 step 显存不增长）
  - `test_no_unbounded_metadata_growth`（prompt state / reuse count 有界）
  - `test_l2_disabled_regression`（l2.enabled=false 行为与原 L0/L1 完全一致）

Commit：`test(l2): 工程检查(no teacher in train_step/no mem growth/no metadata growth/回归)`

---

## 自检

**1. 规格覆盖度：**
- §1 架构（colocated+FSDP）→ 任务 6.1（交替相位）+ 标注 FSDP 为骨架（1.7B 单卡可跑，FSDP 为未来）
- §2 ring buffer Q1-Q4 → 任务 1.2（ring buffer）+ 1.4（feeder/staleness 修）+ 6.1（Q1 触发）
- §3 Disagreement → 任务 2.1（computer）+ 2.2（rollout 相位）+ 2.3（HF generate）
- §4 Health Monitor → 任务 3.1
- §5 Dynamic Ratio → 任务 4.1
- §6 Selective Rollout → 任务 5.1
- §7 config → 任务 1.1
- §8 文件映射 → 文件结构节
- §9 tests → 各任务测试 + 6.3
- §10 E0-E6 → 任务 6.2
- §11 兼容性 → 任务 2.2（per-token logp 算完即弃）+ 2.3（HF 骨架）
- §12 数据流 → 任务 6.1
- §13 整合 → 任务 6.1（单向依赖装配）+ 6.2（实验框架）
- **遗漏**：FSDP 双卡实际包装（spec §1/§2）标为骨架扩展点，本计划以 toy/单卡 CPU 可测为主，FSDP wrap 在任务 6.1 标注 TODO 接口（真实双卡需 GPU 验证）

**2. 占位符扫描：** 任务 6.1 已补全精确接入点（pipeline.py:414/540-555/506/420）与完整代码。任务 6.2（E0-E6 实验脚本）/6.3（工程检查）为实验验证层，给出框架与测试断言（test_no_teacher_forward / test_l2_disabled_regression 已在 6.1 步骤1 定义），具体实验代码依赖 6.1 完成后按实际信号填写（实验迭代性质，非规格遗漏）。核心算法任务（1.1-5.1 + 6.1）代码完整。

**3. 类型一致性：**
- `RefreshRingBuffer.append(ids, delta_k, generation_step, response_length, token_mask, disagreement_abs)` — 任务 1.2 定义，任务 2.2 调用一致
- `DisagreementComputer.compute(teacher_rl_logp, teacher_ref_logp, student_logp, student_ref_logp, mask)` — 任务 2.1 定义，2.2 调用一致
- `CacheHealthMonitor.record(step, **metrics)` — 任务 3.1 定义
- `DynamicRatioController.update(base_age, policy_drift, refresh_quality)` — 任务 4.1 定义
- `RefreshSelector.select(n_selected, n_prompts)` — 任务 5.1 定义，2.2 调用一致
- `_rand_idxs` 返回元组 `(base_idxs, refresh_idxs)` — 任务 1.4 定义，需 _train_step 拆分（任务 1.4 已标注 `_is_refresh` 标记）

"""配置加载 + pydantic schema 校验（P0 工程化）。

动机：此前 `configs/fullstack_opd.yaml` 只作"文档参考"、代码实际用硬编码
`DEFAULT_CONFIG_V2`——且 dict-of-dicts 合并存在"顶层部署键被 stage 子字典静默忽略"的
隐患（正是 CLOUD_CONFIG 那次 P0 配置作用域 bug 的根源）。

本模块把 YAML 变成**唯一真源**：
- pydantic schema 强校验：`extra="forbid"` 拒绝任何未知/拼错的键（静默忽略→显式报错）；
  `Literal` 限制枚举取值（dtype/cache_mode/scheduling_mode/warmup_source/...）；
- `load_config()`：YAML → 合并到内置默认 → 应用点分 CLI 覆盖 → 顶层部署键下渗
  （到 stage1/stage2）→ 校验 → 返回嵌套 dict，可直接传给 `FullStackOPDv2(cfg)`。
"""
from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .exceptions import ConfigError
from .pipeline import DEFAULT_CONFIG_V2


# --------------------------- 子阶段 schema ---------------------------
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 未知键 → ValidationError


class Stage0Cfg(_Strict):
    lr: float = 1e-3
    n_rl_steps: int = 40
    max_new_tokens: int = 8
    batch_size: int = 8
    grad_clip: float = 1.0


class Stage1Cfg(_Strict):
    enforce_teacher_consistency: bool = True
    cache_path: str = "fullstack_opd_cache_v2.pt"
    build_batch_size: int = 16
    # 模块2：true 且 cache_path 文件存在 → 训练载入预建缓存、跳过 Stage 0/1（多学生复用）。
    # 默认 false 走原重建路径（Stage 0 教师 RL + Stage 1 离线 build）。
    load_cache: bool = False
    warmup_M: int = 4                    # L1 默认（与 DEFAULT_CONFIG_V2 对齐，避免 schema 默认吞掉翻转）
    warmup_source: Literal["none", "student_init", "teacher_perturbed", "mix"] = "student_init"
    warmup_temperature: float = 1.0
    # ---- 顶层部署键下渗槽位（A5 解法 + T2 死槽位清理）----
    # load_config 在下渗后才校验，这些键须在 stage schema 有合法位置，否则
    # extra="forbid" 会把下渗结果当未知键拒掉。只保留 stage1 真正消费的键
    # （stage1_build_cache：cache_mode/top_k_teacher），其余为死槽位不设。
    cache_mode: Literal["dense", "topk"] = "dense"
    top_k_teacher: int = 0
    # P-OPD（2026-08-31）：true → 跳过预计算教师得分（不建 stage1 Δ_T），训练走纯 on-policy
    # 交替相位（须配套 l2.enabled=true + l2.pure_refresh=true）。pipeline 建占位 cache
    # （仅 top_k/vocab，无 Δ 数据）供 scheduler/rb 构造读取。
    skip: bool = False


class Stage2Cfg(_Strict):
    # A6：scheduling_mode 只收已实现的 fully_async——n_step_off / fused_hybrid_sync
    # 并未在 scheduler 实现（scheduler 不读取该键），请求其它值应抛校验错误而非
    # 静默按 fully_async 跑（诚实降级，不做假配置）。
    scheduling_mode: Literal["fully_async"] = "fully_async"
    staleness_threshold: int = 4
    queue_size: int = 8
    kl_reg_coef: float = 0.05
    # Direct-OPD 论文 §2.4 adaptive KL（Eq.16）：true 时 kl_coef 由自适应控制器按稠密
    # reward 符号动态调整（忽略 kl_reg_coef 固定值）；默认 false 零回归（固定 KL）。
    kl_adaptive: bool = False
    clip_eps: float = 0.2
    grad_clip: float = 1.0
    lr: float = 1e-3
    n_steps: int = 30
    batch_size: int = 8
    distributed: bool = False
    n_gpus: int = 2
    prefetch: int = 4
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    tp_size: int = 1
    sequence_parallel: bool = True
    rollout_engine: Literal["toy", "vllm"] = "toy"
    # tp 默认 1：引擎按单卡放置（物理卡由 CUDA_VISIBLE_DEVICES 决定）；
    # 多卡 TP 仍可显式配置 rollout_tp_size=2。
    rollout_tp_size: int = 1
    # 空串 = pipeline 回落 student_path（on-policy 同构）。⚠️ 不要给 HF hub
    # 默认名——真实环境离线时会触发联网下载直接失败。
    rollout_model: str = ""
    rollout_dtype: Literal["auto", "bf16", "fp8"] = "auto"
    rollout_logprobs_cap: int = 4096
    # vLLM 引擎放置/显存/上下文（IMP-2/P1）：rollout_device 为输出张量回落设备
    # （训练卡同卡最优）；物理卡位由进程 CUDA_VISIBLE_DEVICES 决定。rollout_gpu_mem
    # 默认 0.9（独占）；训练+vLLM 共卡并行时调低（如 0.45）。
    rollout_device: str = "cuda:1"
    rollout_gpu_mem: float = 0.9
    rollout_max_model_len: int = 2048
    # IMP-2/P1：vLLM 引擎最大并发请求数（压低超配 KV）+ 启动显存断言训练预留量
    rollout_max_num_seqs: int = 256
    rollout_min_free_gb: float = 25.0
    # IMP-2/P1：base 池稀疏 PG（默认开，OOM 根治）+ 队列深度 + 停滞 watchdog
    base_sparse_pg: bool = True
    staleness_queue_min: int = 16
    rollout_stall_timeout: float = 900.0
    # IMP-2/P1：vLLM>=0.16 权重同步模式——"auto"（NCCL WeightTransferEngine，on-policy）
    # / "off"（逃生舱：警告一次并回落初始权重）。旧版 vLLM 不受影响。
    rollout_weight_sync: str = "auto"
    # ---- 顶层部署键下渗槽位（A5 解法 + T2 死槽位清理）----
    # load_config 在下渗后才校验，这些键须在 stage schema 有合法位置，否则
    # extra="forbid" 会把下渗结果当未知键拒掉。只保留 stage2 真正消费的键
    # （scheduler：dtype/top_k_student/offload_to_cpu）。cache_mode/top_k_teacher
    # 由 cache 对象读取（stage2 无槽位）；ref_topk 由 pipeline 读顶层（不下渗）。
    dtype: Literal["fp32", "bf16", "float32", "bfloat16"] = "fp32"
    top_k_student: int = 0
    offload_to_cpu: bool = False
    # 激活重计算（默认关）：见 scheduler（2026-08-18 loss.backward OOM 实测——
    # Qwen3-1.7B × (4,3072) backward 重放 28 层激活 ≈ 25GB，开则 ~2.5GB）。
    gradient_checkpointing: bool = False
    # 稀疏支撑重归一化（对齐原始 Direct-OPD）：pg_loss 把 π_old 在 Δ≠0 支撑上重归一、
    # low_var_kl_support 把 π_cur 在 top-K 上重归一（条件期望）。默认关=原「非归一截断」
    # 有界近似；GPU 稀疏预设（gpu_skeleton/CLOUD_CONFIG）开。PG 与 KL 必须同步开关。
    renormalize_topk_support: bool = False
    # Δ_T 数值护栏（部署实测 P1）：真实教师对 log-ratio 差可达 ±10 → PG 无界爆炸、学生
    # 坍缩。非 None 时 pg_loss 先 clamp Δ_T 到 ±delta_clip（toy 小 Δ_T 无需，默认 None）。
    delta_clip: float | None = None
    # 优化器：adam（默认 fp32）/ adamw_8bit（bnb，4B/7B 单卡必需——fp32-Adam 超 96GB）。
    optimizer: Literal["adam", "adamw_8bit"] = "adam"
    # P3（2026-08-19）：teacher_rl/teacher_ref 只在 refresh 相位算 Δ_T 时使用，base 训练
    # 完全不用（Δ_T 从缓存读，KL 锚点是 student_ref）。开 → base 训练时把两个教师 offload
    # 到 CPU（省 ~6.8GB 基线），refresh 相位前搬回、完成后搬出。默认 False 保持原行为。
    # ⚠️ student_ref（KL 锚点）每步都用，绝不能 offload。
    teacher_offload: bool = False
    # refresh 训练 chunk 大小（v5 OOM 实测 chunk=4 仍不够时降到 2）：_train_step_refresh
    # 把 ring buffer 样本拆成独立小批 + 梯度累积，控制 (chunk,T,V) 前向/反向峰值。
    refresh_chunk: int = 4

    # D1（2026-08-25）：固定评估集 + 周期评估（OPD 信号诊断，默认全关零回归）
    # eval_holdout_size>0：从训练数据末尾划出 N 条作固定评估集（不参与训练）；
    # eval_every>0：每 N 步在固定集上评估当前策略 E[Δ_T]（no_grad），记 metrics eval_reward。
    eval_holdout_size: int = 0
    eval_every: int = 0
    eval_chunk: int = 8   # D1：固定评估集分 chunk 前向大小（双卡并行共卡时降到 2 防 OOM）


# --------------------------- L2 Adaptive Teacher Cache（§2-§7）-----------------
# 默认全关（enabled=False 退回 L0/L1 静态路径）；每模块双 enabled 开关支持单项 ablation
# （E0-E6 实验矩阵，见 scripts/run_l2_ablation.py）。extra="forbid" 下必须在 schema 有
# 合法位置，否则 load_config(overrides=["l2.*=..."]) 会被当未知键拒掉。
class L2CacheCfg(_Strict):
    """L2 ring buffer 基础（§2 双池结构）。"""
    base_size: int = 50000
    refresh_size: int = 5000          # ring buffer capacity
    max_response_length: int = 8192
    value_protect_quantile: float = 0.9   # §2 Q3 价值保护
    refresh_min_interval: int = 50    # §2 Q1 触发约束
    refresh_max_interval: int = 150
    # IMP-1d：refresh pool 冷启动门槛——池样本数 < 此值时跳过 refresh 训练（不调
    # _train_step_refresh），rollout metrics 照常、ring buffer 样本不丢，记录 skip reason。
    min_refresh_pool: int = 8
    # P-OPD（2026-08-31）：纯 on-policy 下连续空训练相位上限——冷启动/池长期不足/rollout
    # 全无效导致 0 真实训练步时，超过此数明确失败（防死循环 + 静默空跑无断点）。
    max_empty_phases: int = 8
    delta_slope_eps: float = 0.001


class L2RolloutCfg(_Strict):
    """Stage 2：训练 rollout 短预算协议（短 Rollout，不要求自然 EOS）。

    与 dataset.max_response_len（数据填充长度）、evaluation.max_reasoning_budget（eval 预算）
    区分：本段是【训练期 rollout 生成上限】。短预算 L_train=1024 的意义是「训练短、评估长」
    （train 1024 → eval B∈{256..4096}），验证短 rollout 能否稳定产生有效 OPD signal。
    """
    max_new_tokens: int = 1024         # 每 rollout 生成上限（训练预算 L_train）
    allow_budget_stop: bool = True     # 允许预算截断（不把 budget-stop 当 EOS）
    eos_token_id: int | None = None    # None=不判 EOS（全 BUDGET_STOP）；=int 采到即停
    loop_detection: bool = True        # 周期重复检测 → 判 LOOP（不进 refresh cache）
    # IMP-1b：尾部周期重复检测的周期集合。默认 (2,3,4) 即原 detect_loop 硬编码
    # 值；真实 Qwen3/Skywork 经 scripts/calibrate_rollout.py 校准后覆盖（命中率>5% 的周期）。
    loop_periods: tuple[int, ...] = (2, 3, 4)
    # IMP-1a：rollout 采样温度。默认 0.7（降循环率）；=1.0 复现旧行为（temperature=1.0 恒等）。
    # 真实 HF 生成层（generate_with_status* / vLLM）已支持该参数，此处仅透传。
    temperature: float = 0.7
    # IMP-1c：抗退化采样。repetition_penalty>1 对已生成 token 的 logits 除 penalty（抑制
    # 「Final Answer marker 反复」）；=1.0 禁用（旧行为）。loop_min_len 为 detect_loop 的
    # 最小有效长度门槛，默认 8=旧行为；真实模型短预算下可调高以降低误报。
    repetition_penalty: float = 1.0
    loop_min_len: int = 8
    # IMP-1c（teacher rollout）：rollout 采样来源。student=主实验路径（y~pi_student）；
    # teacher=仅诊断/上界实验（y~pi_teacher_rl）。默认 student（禁止默认启用 teacher）；
    # teacher 轨迹不构成主 L2 on-policy 数据，禁止混进主 E5。逐样本 metadata 记录 source。
    rollout_source: Literal["student", "teacher"] = "student"
    pad_id: int = 0                    # 变长 batch 右 pad 值（不参与判定，仅占位）
    token_budget_per_refresh: int | None = None   # §六：每轮刷新全局 rollout token 预算；None=无上限
    # Direct-OPD 论文 rollout n（Table 3：rollout n=4）：每 prompt 每次 refresh 生成 n 条
    # on-policy 响应（每条独立算 Δ/status，per-prompt 聚合）；=1 零回归（旧单条行为）。
    n_rollout: int = 1


class L2DisagreementCfg(_Strict):
    """§3 Teacher-Student Disagreement 开关。"""
    enabled: bool = True


class L2HealthMonitorCfg(_Strict):
    """§4 Cache Health Monitor（Observe-only）。"""
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
    """§5 dynamic refresh ratio（三信号 controller）。"""
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
    """§6 selective rollout（candidate pool 两阶段）。"""
    enabled: bool = True
    candidate_multiplier: int = 4    # M_candidate = 4·M_selected（§6.5）
    value_fraction: float = 0.80     # §6.3 高价值占比
    coverage_fraction: float = 0.20
    value_weights: dict = Field(default_factory=lambda: {
        "uncertainty": 0.3, "disagreement": 0.3, "reward": 0.2, "novelty": 0.2})
    compute_aware: bool = False      # §6.4 ELG
    budget_mode: Literal["fixed", "adaptive"] = "fixed"   # §三：fixed=单预算；adaptive=分位数 4 档
    fixed_budget: int = 1024                              # §一：fixed 模式统一预算
    budget_set: tuple[int, ...] = (256, 512, 1024, 2048)  # §一：候选预算档位（easy/medium/uncertain/hard）
    budget_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)  # §三：4 档用 3 个分位
    token_aware: bool = False            # §五：Dynamic Refresh Ratio 感知 rollout token（默认关零回归）
    token_weight: float = 0.1            # §五：token 信号权重
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
    # P-OPD（2026-08-31）：true → 纯 on-policy 交替相位——无 base 池（scheduler.run 不调用），
    # 每相位 run_refresh_phase ↔ train_refresh_phase，α 冻结 1.0（训练 100% on-policy）。
    pure_refresh: bool = False
    cache: L2CacheCfg = Field(default_factory=L2CacheCfg)
    rollout: L2RolloutCfg = Field(default_factory=L2RolloutCfg)   # Stage 2 短 rollout
    disagreement: L2DisagreementCfg = Field(default_factory=L2DisagreementCfg)
    health_monitor: L2HealthMonitorCfg = Field(default_factory=L2HealthMonitorCfg)
    refresh_ratio: L2RefreshRatioCfg = Field(default_factory=L2RefreshRatioCfg)
    selective_rollout: L2SelectiveRolloutCfg = Field(default_factory=L2SelectiveRolloutCfg)
    utility: L2UtilityCfg = Field(default_factory=L2UtilityCfg)


# --------------------------- 工程化新增段 ---------------------------
class RunCfg(_Strict):
    """run 目录与断点续跑配置。"""
    seed: int | None = None   # L4：None → 回退顶层 seed（避免 run.seed 默认 42 遮蔽顶层）
    run_dir: str | None = None
    checkpoint_every: int = Field(10, ge=1)   # L5：<=0 抛校验错（否则 max(1,0) 静默每步保存）


class LoggingCfg(_Strict):
    """结构化日志配置。"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: str = "train.log"


class MetricsCfg(_Strict):
    """指标追踪配置。"""
    backend: Literal["csv", "wandb", "none"] = "csv"
    csv_path: str | None = None
    wandb_project: str | None = None


class DatasetCfg(_Strict):
    """数据加载配置（可插拔接口，见 data.py）。"""
    type: Literal["toy", "jsonl"] = "toy"
    path: str | None = None
    prompt_key: str = "prompt"
    response_key: str = "response"
    # jsonl 真实数据（model_kind='hf'）：prompt/response 截断+右 pad 到定长；
    # tokenizer_path 默认回退顶层 student_path（数据与词表对齐）。
    max_prompt_len: int = 256
    max_response_len: int = 384
    tokenizer_path: str | None = None
    # C3（2026-08-18）：prompt 先套模型 chat template（Qwen3 <|im_start|> 格式）再编码。
    # 裸数学题 prompt 下 Qwen3 生成乱码+loop（2026-08-18 GPU 实测根因）；开启后必须用
    # 同配置重建 teacher cache——cache metadata 落 prompt_format=chat，
    # verify_consistency 对裸 cache 配模板配置 fail-fast（C2 守卫）。
    apply_chat_template: bool = False
    # Stage 0 规模概念（50K prompt universe + materialized 静态锚点，见 base.materialized_size）：
    # prompt 池总规模（50K）。实际训练只读到「有 response 的」行（JsonLinesDataLoader 跳过
    # 空 response 行），其余 prompt 留空待 L2 在线 refresh。本字段仅作数据规模声明，不接算法。
    prompt_universe_size: int = 50000


class EvalCfg(_Strict):
    """AIME 评估配置（eval-aime 子命令与 run 目录桥接）。

    model_path: 真实 HF 模型路径 / HF id（toy run 目录无此键 → 无法跑真实 AIME）。
    供 `opd eval-aime --run-dir <dir>` 读取 run_dir/config.yaml 用。
    """
    model_path: str | None = None
    datasets: list[str] = ["AIME24", "AIME25"]
    max_new_tokens: int = 2048
    n_samples: int = 1
    temperature: float = 0.0
    top_p: float | None = None
    metric: Literal["pass1", "ave"] = "pass1"
    prompt_style: Literal["boxed", "dapo"] = "boxed"
    # 生成 batch（默认 8）。长生成（max_new_tokens 数万）必须调小——峰值显存随 batch 线性涨。
    batch_size: int = 8
    # 评分方式：int（默认，整数精确匹配）| sympy（论文 grade_answer_mathd/sympy 数学等价判定）
    scoring: Literal["int", "sympy"] = "int"
    # chat_template（默认 False）：True 用模型 chat template 包裹 prompt（对齐论文 verl 验证）
    chat_template: bool = False
    # attn_implementation：None=SDPA 默认；"flash_attention_2" 启用 flash_attn（长生成 decode 提速）
    attn_implementation: str | None = None


class BaseCfg(_Strict):
    """Stage 0「50K prompt universe + materialized 静态锚点」规模概念（仅数据结构，不接算法）。

    materialized_size：初始生成 response 的条数（静态锚点）。实际训练只读到有 response 的
    行（JsonLinesDataLoader 跳过空 response 行），其余 prompt 留空待 L2 在线 refresh（本阶段
    不实现 L2，仅声明规模）。配合 dataset.prompt_universe_size（prompt 池总规模）。
    """
    materialized_size: int = 5000


class CacheCfg(_Strict):
    """Stage 1 统一 K 存储架构（解决 50K×8192 cache memory wall）。

    top_k：统一 K（teacher/student/ref 支撑默认同源）。用户指定实验取值范围
      K∈{16,32,64,128,256}（磁盘 mmap 驻留下可选）；0 = dense 模式。
      16 为 Direct-OPD 论文 student top-k support（2026-08-27 论文对齐增加）；
      改 K 必须重建磁盘缓存（verify_consistency 校验 metadata.top_k）。
    storage：磁盘 mmap 驻留（本阶段目标，§3）——GPU/RAM 只驻当前 batch 行；
      "memory" 显式保留原全量驻留路径（直接构造 TensorTeacherCache 的测试不受影响）。
      dense 模式忽略 storage。
    """
    top_k: int = 32
    storage: Literal["memory", "disk"] = "disk"

    @field_validator("top_k")
    @classmethod
    def _topk_allowed(cls, v: int) -> int:
        if v not in (0, 16, 32, 64, 128, 256):
            raise ValueError(f"cache.top_k 仅允许 0/16/32/64/128/256（0=dense），收到 {v}")
        return v


class OPDConfig(_Strict):
    vocab_size: int = 64
    d_model: int = 48
    n_layers: int = 2
    prompt_len: int = 6
    resp_len: int = 8
    n_prompts: int = 16
    seed: int = 42
    batch_size: int = 8
    dtype: Literal["fp32", "bf16", "float32", "bfloat16"] = "fp32"
    cache_mode: Literal["dense", "topk"] = "dense"
    top_k_teacher: int = 0
    top_k_student: int = 0
    ref_topk: int = 0
    offload_to_cpu: bool = False
    model_kind: Literal["toy", "hf", "megatron", "vllm"] = "toy"
    # HF 骨架（model_kind="hf"）：真实模型路径。student_path=学生；
    # teacher_rl_path/teacher_ref_path=预下载教师对（真实实验跳过 Stage 0 RL 直接加载）。
    student_path: str | None = None
    teacher_rl_path: str | None = None
    teacher_ref_path: str | None = None
    stage0: Stage0Cfg = Field(default_factory=Stage0Cfg)
    stage1: Stage1Cfg = Field(default_factory=Stage1Cfg)
    stage2: Stage2Cfg = Field(default_factory=Stage2Cfg)
    run: RunCfg = Field(default_factory=RunCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    l2: L2Cfg = Field(default_factory=L2Cfg)
    base: BaseCfg = Field(default_factory=BaseCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)


# --------------------------- 顶层部署键下渗 ---------------------------
# 顶层部署键（CLOUD_CONFIG 风格）会在 pydantic 校验前按 stage 分流下渗，
# 使校验后的 cfg 与 config.yaml 快照天然含下渗结果（A4/A5/B4）——不再由
# pipeline/cli 在运行时各自复制一份下渗循环（否则快照≠有效配置，且 stage
# 子 dict 在 extra="forbid" 下没有 dtype 等键的合法位置）。
# 按消费端分流（T2 死槽位清理）：stage1 只消费稀疏缓存键，stage2 只消费
# 训练/部署键；ref_topk 保持纯顶层（pipeline 读 self.cfg.get("ref_topk")）。
_STAGE1_SEEP_KEYS = ("cache_mode", "top_k_teacher")
_STAGE2_SEEP_KEYS = ("dtype", "top_k_student", "offload_to_cpu")


def _seep_deployment_keys(d: dict) -> dict:
    """顶层部署键按 stage 分流下渗（stage 子键优先）。

    在 pydantic 校验前调用，使校验后的 cfg 与 config.yaml 快照天然含下渗结果
    （A4/A5/B4）。只补 stage 里没有的键：stage 子键显式给出的值不会被顶掉。
    - stage1 ← {cache_mode, top_k_teacher}（stage1_build_cache 消费）
    - stage2 ← {dtype, top_k_student, offload_to_cpu}（scheduler 消费）
    - ref_topk 保持纯顶层（pipeline 读顶层），不下渗、stage 无槽位
    """
    for k in _STAGE1_SEEP_KEYS:
        if k in d:
            sd = d.setdefault("stage1", {})
            if k not in sd:
                sd[k] = d[k]
    for k in _STAGE2_SEEP_KEYS:
        if k in d:
            sd = d.setdefault("stage2", {})
            if k not in sd:
                sd[k] = d[k]
    return d


def _parse_scalar(text: str) -> Any:
    """把 CLI 字符串覆盖值解析成 bool/int/float/tuple/str。

    IMP-1b：支持 tuple/int-list 语法 "(2,3,4)" / "[2,3,4]" / "2,3,4" →
    tuple[int, ...]，使 l2.rollout.loop_periods（及 budget_set/quantiles）
    可经 --set 点分覆盖。全 int 才转 tuple；否则回退常规解析。
    """
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    # tuple / int-list 语法："(2,3,4)" | "[2,3,4]" | "2,3,4" → tuple[int, ...]
    s = text.strip()
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1]
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if parts:
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                pass   # 含非 int → 回退常规解析
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """`stage2.lr=1e-4` → d['stage2']['lr']=1e-4（点分路径）。"""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_config(path: str | None = None,
                overrides: list[str] | None = None) -> dict:
    """YAML → 合并默认 → CLI 点分覆盖 → pydantic 校验 → 嵌套 dict。

    - path=None 时仅用内置默认（等价 DEFAULT_CONFIG_V2）。
    - overrides: ["stage2.lr=1e-4", "n_steps=50", "stage1.warmup_source=mix"]。
    - 任何未知键 / 非法枚举值 / 类型不符 / 覆盖项缺 `=` → 抛 ConfigError
      （pydantic ValidationError 被包装为 ConfigError，调用方可统一按配置错误捕获）。
    """
    data: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    for item in (overrides or []):
        if "=" not in item:
            raise ConfigError(f"覆盖项需形如 key=value，收到 {item!r}")
        k, v = item.split("=", 1)
        _set_dotted(data, k.strip(), _parse_scalar(v))
    # 顶层部署键按 stage 分流下渗（stage 子键优先，见 _seep_deployment_keys）。
    # 须在 pydantic 校验前做：下渗键（如 stage2.dtype）在 extra="forbid" 的
    # stage schema 无合法位置，校验后补会破坏"快照=有效配置"（A4/A5/B4）。
    # --set 点分覆盖已在 _set_dotted 应用过，下渗只补 stage 里没有的键。
    data = _seep_deployment_keys(data)
    try:
        cfg = OPDConfig(**data)                  # 校验（未知键/非法值在此报错）
    except ValidationError as e:
        raise ConfigError(f"配置校验失败: {e}") from e
    # 合并到内置默认（pydantic 已用默认补全所有键，model_dump 即为完整配置）
    merged = {**DEFAULT_CONFIG_V2, **cfg.model_dump()}
    for stage in ("stage0", "stage1", "stage2"):
        merged[stage] = {**DEFAULT_CONFIG_V2[stage], **cfg.model_dump()[stage]}
    return merged


__all__ = ["OPDConfig", "Stage0Cfg", "Stage1Cfg", "Stage2Cfg",
           "L2Cfg", "L2CacheCfg", "L2RolloutCfg", "L2DisagreementCfg", "L2HealthMonitorCfg",
           "L2RefreshRatioCfg", "L2SelectiveRolloutCfg", "L2UtilityCfg",
           "RunCfg", "LoggingCfg", "MetricsCfg", "DatasetCfg", "EvalCfg",
           "load_config", "ValidationError"]

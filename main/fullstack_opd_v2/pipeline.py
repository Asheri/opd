"""v2 编排器：全栈 OPD 流水线的批量化重构版。

  小模型 RL ──► 离线缓存教师对 Δ_T ──► Direct-OPD 训练跑在 AsyncOPD 批量调度器上

相对 v1 的底层变化：
- toy 数据 = 堆叠张量 (N,P)/(N,T) 设备常驻（v1 是 list of CPU tensors，逐步搬运）；
- 规则奖励 = 查找表 (V,) 向量化索引（v1 逐 token python 循环）；
- stage0 REINFORCE 批量化（一次批量解码 + 一次批量前向）；
- stage1 缓存 = 设备常驻张量 + 预计算 delta；
- stage2 调度器队列传 batch，learner 按 mini-batch 更新；
- 每个 stage 计时，便于 benchmark 对比。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import torch


def _mem_point(tag: str) -> None:
    """显存打点（调试用，OPD_MEM_TRACE=1）。"""
    import os as _os
    if _os.environ.get("OPD_MEM_TRACE") == "1" and torch.cuda.is_available():
        print(f"[MEM:{tag}] allocated={torch.cuda.memory_allocated()/2**30:.1f}GiB "
              f"reserved={torch.cuda.memory_reserved()/2**30:.1f}GiB", flush=True)

from .cache import TensorTeacherCache
from .model import CausalToyLM, generate_batch, token_logprobs, response_dists
from .scheduler import AsyncBatchedScheduler, launch_distributed_scheduler
from .data import build_data_loader
from .model_factory import build_model
from .run import RunManager
from .checkpoint import CheckpointManager
from .metrics import MetricsRecorder
from .logging import setup_logging, get_logger, close_logging
from .exceptions import DataError, ModelError, TrainingError

DEFAULT_CONFIG_V2 = {
    "vocab_size": 64,
    "d_model": 48,
    "n_layers": 2,
    "prompt_len": 6,
    "resp_len": 8,
    "n_prompts": 16,
    "seed": 42,
    "batch_size": 8,
    # ---- GPU 部署相关（与硬件无关，本地 demo 默认 fp32 / dense 仍可跑）----
    "dtype": "fp32",          # "fp32" | "bf16"（L1；bf16 仅 cuda 生效）
    "cache_mode": "dense",    # "dense" | "topk"（L4；真实词表用 topk）
    "top_k_teacher": 0,       # >0 → 教师缓存每位置存 teacher top-K（L4 稀疏）
    "top_k_student": 0,       # >0 → 训练时在 student top-K 支撑上取 Δ_T（L4）
    "ref_topk": 0,            # >0 → KL 锚点存初始 student top-K（避免 (N,T,V) 撑爆）
    "offload_to_cpu": False,  # colocated 换入换出（L6）
    "model_kind": "toy",      # 可插拔模型工厂（model_factory.py）：toy 默认 / hf 骨架
    # HF 骨架路径（model_kind="hf" 时生效）：学生 + 预下载教师对（跳过 Stage 0 RL）
    "student_path": None,
    "teacher_rl_path": None,
    "teacher_ref_path": None,
    # ---- 工程化新增段（run 目录 / 日志 / 指标 / 数据）----
    "run": {"seed": None, "run_dir": None, "checkpoint_every": 10},
    "logging": {"level": "INFO", "file": "train.log"},
    "metrics": {"backend": "csv", "csv_path": None, "wandb_project": None},
    "dataset": {"type": "toy", "path": None, "prompt_key": "prompt",
                "response_key": "response",
                "max_prompt_len": 256, "max_response_len": 384,
                "tokenizer_path": None},
    "eval": {"model_path": None, "datasets": ["AIME24", "AIME25"],
             "max_new_tokens": 2048, "n_samples": 1, "temperature": 0.0},
    # ---- L2 Adaptive Staleness-Aware Teacher Cache（默认全关，enabled=false 退回 L0/L1）----
    # 完整 schema 见 config.py L2Cfg；此处仅给总开关，pipeline/测试经 load_config 读全量。
    "l2": {"enabled": False},
    "stage0": {              # 小模型 RL（产生 post-RL weak teacher）
        "lr": 1e-3, "n_rl_steps": 40, "max_new_tokens": 8,
        "batch_size": 8, "grad_clip": 1.0,
    },
    "stage1": {              # Lightning-OPD 离线缓存
        "enforce_teacher_consistency": True,
        "cache_path": "fullstack_opd_cache_v2.pt",
        "build_batch_size": 16,
        "load_cache": False,      # 模块2：true → 载入预建缓存跳过 Stage 0/1（多学生复用）
        # ---- L1 离线 rollout 暖缓存（缓解曝光偏差）----
        "warmup_M": 4,            # L1 默认：学生 ref 一次性 rollout 拼「胖 D」N×(1+M)（消曝光偏差）
        "warmup_source": "student_init",  # none | student_init | teacher_perturbed | mix
        "warmup_temperature": 1.0,
    },
    "stage2": {              # Direct-OPD + AsyncOPD
        "scheduling_mode": "fully_async",
        "staleness_threshold": 4,
        "queue_size": 8,
        "kl_reg_coef": 0.05,
        "clip_eps": 0.2,
        "grad_clip": 1.0,
        "lr": 1e-3,
        "n_steps": 30,
        "batch_size": 8,
        # ---- GPU 部署骨架开关（L5/L2），本地 CPU demo 默认全关 ----
        "distributed": False,      # True → 走 DistAsyncScheduler（Ray worker + NCCL 广播）
        "n_gpus": 2,              # 2×RTX PRO 6000
        "prefetch": 4,            # 在途 rollout 数（Ray future 流水线）
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "tp_size": 1,             # >1 → L2 Megatron TP=2+SP（需 model.py 换 megatron parallel 层）
        "sequence_parallel": True,
        # ---- L3 vLLM rollout 替换（GPU 部署）："toy" 走朴素前向，"vllm" 走 vLLM TP=2 ----
        "rollout_engine": "toy",  # "toy" | "vllm"
        "rollout_tp_size": 2,     # vLLM tensor parallel（NVLink 桥）
        "rollout_model": "Qwen/Qwen2.5-7B",  # vLLM 模型路径/名（仅 vllm 时生效）
        "rollout_dtype": "auto",  # "auto" | "bf16" | "fp8"（Blackwell 可 fp8）
        "rollout_logprobs_cap": 4096,  # 触发「精确完整分布」重建的词表上限（>则 top-K 截断）
        # 稀疏支撑重归一化（对齐原始 Direct-OPD）：默认关=非归一有界近似；
        # GPU 稀疏预设（CLOUD_CONFIG / gpu_skeleton）置 true。PG 与 KL 同步开关。
        "renormalize_topk_support": False,
        # 优化器（多学生并发：4B/7B 用 adamw_8bit 压优化器内存，单卡可训）
        "optimizer": "adam",
    },
}

# ---- 2×RTX PRO 6000 部署预设（见 OPTIMIZATION_PLAN_2xRTXPRO6000.md）----
# 本地不跑（玩具模型无法到 7B），仅作为上服务器时的配置起点。
# 尺寸依据 §0.1：learner（被训练 student）≤13B（8-bit Adam+梯度检查点）；
#               teacher/rollout（仅推理）可上 70B（TP=2 / fp8）。
CLOUD_CONFIG = {
    **DEFAULT_CONFIG_V2,
    "vocab_size": 128000,      # 真实词表（Qwen/Llama 系）；dense 缓存会因 (N,T,V) 撑爆 → 必须 topk
    "dtype": "bf16",           # L1：Blackwell 原生 bf16 tensor core
    "cache_mode": "topk",      # L4：稀疏缓存，体积 ↓~1000×
    "top_k_teacher": 256,      # 教师每位置 top-K（覆盖绝大部分概率质量）
    "top_k_student": 256,      # 训练时 student top-K 支撑
    "ref_topk": 256,           # KL 锚点稀疏化（避免 (N,T,V) 锚点 OOM）
    "offload_to_cpu": True,    # L6：colocated 换入换出，避免 2×96GB 同卡同时驻留
    # L3：工程化新段显式覆盖（骨架 demo 仍 toy 模型；断点/指标按云部署调优）
    "model_kind": "toy",       # 骨架 demo：toy 内核；真实 7B 由 async-opd 承担
    "run": {"seed": None, "run_dir": None, "checkpoint_every": 50},
    "logging": {"level": "INFO", "file": "train.log"},
    "metrics": {"backend": "csv", "csv_path": None, "wandb_project": "opd"},
    "dataset": {"type": "toy", "path": None, "prompt_key": "prompt",
                "response_key": "response",
                "max_prompt_len": 256, "max_response_len": 384,
                "tokenizer_path": None},
        "stage1": {           # L1：云部署默认开启暖缓存（与 L0 相比仅 Stage1 多一次离线采样）
            **DEFAULT_CONFIG_V2["stage1"],
            "warmup_M": 4,
            "warmup_source": "student_init",   # L1：与默认一致，学生 ref rollout
            "warmup_temperature": 1.0,
        },
        "stage2": {
            **DEFAULT_CONFIG_V2["stage2"],
            "batch_size": 32,       # 4090 无关：PRO 6000 有 NVLink，批更大更值
            "n_steps": 1000,
            # 稀疏支撑重归一化：GPU 稀疏预设开（对齐原始 Direct-OPD 条件期望）
            "renormalize_topk_support": True,
            # learner 用 Megatron TP=2+SP（NVLink）；rollout 用 vLLM TP=2+FP8（见方案 L2/L3）
            # 此处仅为 demo 内核的 GPU 配置骨架，真实并行由 verl/slime/vLLM 接管。
            "rollout_engine": "vllm",   # L3：vLLM TP=2 rollout 替换朴素前向
            "rollout_tp_size": 2,
            "rollout_model": "Qwen/Qwen2.5-7B",
            "rollout_dtype": "auto",
        },
}


# ----------------------------- Stage 0 -----------------------------
def stage0_small_rl(prompts, reward_fn, cfg: dict, device,
                    vocab: int, d_model: int, n_layers: int, teacher=None):
    """批量 REINFORCE：产生 post-RL weak teacher；pre-RL reference 为其训练前副本。

    reward_fn: `(B,T) token` -> `(B,T) 奖励` 的可调用对象（来自 DataLoader，toy 为查找表）。
    teacher: 预构建的初始模型（B2：由 build_model 注入，model_kind 配置生效）；
             None 时退回 CausalToyLM 默认构造（原行为不变）。
    """
    weak = teacher if teacher is not None else \
        CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers).to(device)
    ref = CausalToyLM(vocab=vocab, d_model=d_model, n_layers=n_layers).to(device)
    ref.load_state_dict(weak.state_dict())
    ref.eval()

    opt = torch.optim.Adam(weak.parameters(), lr=cfg.get("lr", 1e-3))
    B = cfg.get("batch_size", 8)
    N = prompts.size(0)
    for _ in range(cfg.get("n_rl_steps", 40)):
        idxs = torch.randint(0, N, (B,), device=device)
        p_b = prompts[idxs]                                   # (B, P)
        r = generate_batch(weak, p_b, cfg.get("max_new_tokens", 8))   # (B, T)
        weak.train()
        logp = token_logprobs(weak, p_b, r)                   # (B, T) 保留梯度
        reward = reward_fn(r)                                 # (B, T) 奖励（可插拔）
        loss = -(logp * (reward - reward.mean())).mean()      # REINFORCE + 均值基线
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weak.parameters(), cfg.get("grad_clip", 1.0))
        opt.step()
    weak.eval()
    return weak, ref


# ----------------------------- 编排器 -----------------------------
def _same_card(a: str, b: str) -> bool:
    """训练卡与 rollout 卡是否同卡（"cuda"/"cuda:0" 按卡号归一）。"""
    def _idx(x: str) -> str:
        x = str(x)
        if x.startswith("cuda") and ":" in x:
            return x.split(":")[-1]
        if x == "cuda":
            return str(torch.cuda.current_device()) if torch.cuda.is_available() else "0"
        return x
    return _idx(a) == _idx(b)


def _l2_rollout_mem_enough(free_gb: float, eng_gb: float, min_free: float) -> bool:
    """L2/P2 显存预算：同卡（训练+rollout 共卡）需留训练侧余量 min_free；
    异卡只需引擎份额（rollout 卡上没有训练驻留，叠加训练余量会假 OOM——
    96GB 卡 0.9 引擎 + 25GB 训练预留 = 111GB > 卡容量，恒失败）。"""
    return free_gb >= eng_gb + min_free

def _check_rollout_max_model_len(cfg: dict, s2cfg: dict, l2_cfg: dict) -> int:
    """vLLM max_model_len 安全守卫（纯函数，CPU 可单测）。

    rollout_max_model_len 必须 >= max_prompt_len + max_new_tokens，否则 prompt+response
    恰好等于上限时 vLLM 内部 special token / 余量会导致截断或 init 失败
    （v3 实测 max_model_len=6144 的 KV cache 太大也 init 失败）。返回所需下限。
    """
    _mml = int(s2cfg.get("rollout_max_model_len", 2048))
    _needed = int((cfg.get("dataset") or {}).get("max_prompt_len", 1024)) \
        + int((l2_cfg.get("rollout") or {}).get("max_new_tokens", 512))
    if _mml < _needed:
        raise RuntimeError(
            f"[L2] rollout_max_model_len={_mml} < max_prompt_len + max_new_tokens"
            f"={_needed}；vLLM 会截断或 init 失败。请调大 rollout_max_model_len 或减小"
            "max_prompt_len。")
    return _needed


def _p3_teacher_move(teacher_rl, teacher_ref, target, *, enabled: bool,
                    device: str, logger, message: str) -> None:
    """P3：把 teacher_rl/teacher_ref 搬到 target（"cpu" offload 或训练卡 reload）。

    - enabled=False 或非 cuda 设备时 no-op（默认零回归）；
    - 搬移后 empty_cache 释放 GPU 缓存（target 为 "cpu" 时把保留的教师显存归还）；
    - ⚠️ student_ref 不在此列：它是 KL 锚点（ref_dists[idxs]）每步都用，必须常驻。
    """
    if not enabled or not str(device).startswith("cuda"):
        return
    if teacher_rl is not None:
        teacher_rl.to(target)
    if teacher_ref is not None:
        teacher_ref.to(target)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(message)


def _raise_if_weight_sync_poisoned(engine, *, mr, metric_key: str, label: str) -> None:
    """poisoned engine 必须 fail-closed（不受 require_weight_sync=false 影响）。

    NCCL init/update 一旦超时或失败，engine 被标记 weight_sync_poisoned，其 NCCL
    communicator / EngineCore 状态不可信。这里立即记录指标并中止：禁止继续复用该
    engine 或进入 refresh phase（正确恢复：进程退出 / 重建 engine）。
    """
    if getattr(engine, "weight_sync_poisoned", False):
        mr.record({metric_key: 1})
        raise RuntimeError(
            f"[L2] {label} weight-transfer engine poisoned; "
            "拒绝继续复用可能状态不一致的 engine。")


class FullStackOPDv2:
    def __init__(self, cfg: dict | None = None, device: str = "cpu"):
        self.cfg = {**DEFAULT_CONFIG_V2, **(cfg or {})}
        self.device = device
        # P2（二次审查）：hf 真实模型 + toy 随机数据 = 在噪声上静默训练（toy 随机 id 0-63
        # 在真实词表 ~152k 里是合法 token，不报错但无意义）。显式拦，要求真实数据。
        if self.cfg.get("model_kind") == "hf":
            ds = self.cfg.get("dataset") or {}
            if ds.get("type", "toy") == "toy":
                raise DataError(
                    "model_kind='hf' 需要真实数据（dataset.type='jsonl' + dataset.path）；"
                    "toy 随机数据（vocab 0-63）与真实词表模型训练无意义")
        # 可插拔数据加载（data.py）：toy 默认，与旧 _make_toy_data 同源
        # C3：保留 loader 实例以取原始 prompt 文本（教师各自模板重编码用）
        self._data_loader = build_data_loader(self.cfg, self.device)
        self.prompts, self.responses, self.reward_fn = self._data_loader.load()

    @property
    def raw_prompt_texts(self):
        """C3：原始 prompt 文本（jsonl 场景与 self.prompts 行对齐）。"""
        return getattr(self._data_loader, "raw_prompt_texts", None)

    def _stage0_teachers(self):
        """返回 (teacher_rl, teacher_ref)。供 run()/CLI cache 复用。

        B2：教师初始权重由可插拔工厂 build_model 构建（model_kind 配置生效）并注入
        stage0_small_rl；后者内部仍用 CausalToyLM 构造 ref 副本（toy RL 阶段最小改动）。

        HF 骨架：真实实验的教师对是【预下载模型】（teacher_rl_path=JustRL-1.5B、
        teacher_ref_path=R1-Distill-Qwen-1.5B），**跳过 Stage 0 RL** 直接加载两档；
        teacher 一致性（同架构/词表/d_model）由 TensorTeacherCache.build 校验。
        ⚠️ 骨架：需 GPU/真实模型验证。
        """
        if self.cfg.get("model_kind") == "hf":
            teacher_rl = build_model(self.cfg, self.device, role="teacher")
            from .model_factory import HFCausalLM
            ref_path = self.cfg.get("teacher_ref_path")
            if not ref_path:
                raise ModelError(
                    "model_kind='hf' 但未配置 teacher_ref_path（预下载教师对）")
            teacher_ref = HFCausalLM(ref_path, self.device,
                                     dtype=self.cfg.get("dtype", "auto"))
            return teacher_rl, teacher_ref
        vocab = self.cfg["vocab_size"]
        d_model = self.cfg["d_model"]
        n_layers = self.cfg["n_layers"]
        teacher = build_model(self.cfg, self.device, role="teacher")
        return stage0_small_rl(self.prompts, self.reward_fn, self.cfg["stage0"],
                               self.device, vocab, d_model, n_layers, teacher=teacher)

    def run(self, run_dir: str | None = None, resume: dict | None = None) -> dict:
        """跑全栈流水线（Stage 0/1/2），落盘 run 目录（config/日志/checkpoint/metrics）。

        - run_dir: 显式指定则复用（如 --resume）；None 时自动时间戳。
        - resume: CheckpointManager.resume() 的结果（含 state/version），加载学生权重并
          恢复版本号续跑（Stage 0/1 确定性重放，Stage 2 从断点版本继续）。
        - 计时落 run_dir/timings.json（衡量异步+预加载的时间优化）。
        - 学生 checkpoint 每 checkpoint_every 步落盘（供 AIME 蒸馏后评估）。
        """
        torch.manual_seed(self.cfg.get("run", {}).get("seed") or self.cfg.get("seed", 42))

        # ---- 工程化基础设施：run 目录 + 日志 + 指标 + checkpoint ----
        rdir = run_dir or (self.cfg.get("run") or {}).get("run_dir")
        rm = RunManager(self.cfg, run_dir=rdir)
        paths = rm.create()
        lcfg = self.cfg.get("logging", {})
        setup_logging(level=lcfg.get("level", "INFO"), log_file=paths["log_file"])
        logger = get_logger("opd")
        mcfg = self.cfg.get("metrics", {})
        mr = MetricsRecorder(backend=mcfg.get("backend", "csv"),
                             append=(resume is not None),   # L1：resume 续写保留历史
                             run_dir=paths["run_dir"],
                             csv_path=mcfg.get("csv_path"),
                             wandb_project=mcfg.get("wandb_project"))
        cm = CheckpointManager(paths["run_dir"],
                               every=(self.cfg.get("run") or {}).get("checkpoint_every", 10))
        logger.info(f"run 目录: {paths['run_dir']}  (config/日志/checkpoint/metrics 已就绪)")

        # A7：主体抽到 _run_body()；无论成功还是异常，finally 都释放
        # MetricsRecorder 的 CSV 与 logging 的 FileHandler（Windows 下句柄不释放
        # 会导致临时目录无法清理）。基础设施初始化留在 try 之外——若 setup_logging
        # 失败则无资源可释放。
        try:
            return self._run_body(paths, cm, mr, logger, resume=resume)
        finally:
            mr.close()
            close_logging("opd")     # 释放 train.log 句柄（Windows 下临时目录清理必需）

    def _run_body(self, paths, cm, mr, logger, resume=None) -> dict:
        """Stage 0/1/2 + 统一尾部（计时落盘/末步保存），返回结果 dict。

        由 run() 用 try/finally 包住调用：成功/异常路径都会执行 mr.close() 与
        close_logging()（A7），此处不再负责资源释放。
        """
        vocab = self.cfg["vocab_size"]
        d_model = self.cfg["d_model"]
        n_layers = self.cfg["n_layers"]
        timings: dict = {}

        # P-OPD（2026-08-31）：纯 on-policy——删除 stage1 预计算教师得分与固定数据集 D。
        # 占位 cache 仅提供 top_k/vocab（scheduler 构造读取），不含 Δ 数据；base 池训练
        # 被纯 refresh 交替相位取代，不消费缓存。
        s1cfg = dict(self.cfg["stage1"])
        cache_block = self.cfg.get("cache") or {}
        student = build_model(self.cfg, self.device, role="student")

        teacher_rl = teacher_ref = None
        # P-OPD：加载教师对（跳过 Stage 0 RL；纯 on-policy 每相位实时打分）。
        logger.info("[Stage 0] 加载教师对（跳过 RL；纯 on-policy 实时打分）")
        t = time.perf_counter()
        teacher_rl, teacher_ref = self._stage0_teachers()
        timings["stage0_rl"] = time.perf_counter() - t

        # 占位 cache（无预计算 Δ）：仅提供 top_k/vocab 供 scheduler/rb 构造读取；
        # base 池训练被纯 refresh 交替相位取代，不消费缓存数据。
        logger.info("[Stage 1] 跳过：P-OPD 纯 on-policy（无预计算教师得分 / 固定 D）")
        t = time.perf_counter()
        K = int(cache_block.get("top_k") or s1cfg.get("top_k_teacher") or 16)
        cache = TensorTeacherCache(enforce_consistency=False, top_k=K)
        cache.mode = "topk"
        cache.vocab = int(getattr(student, "vocab", 0) or 0)
        if cache.vocab:
            cache.top_k = min(cache.top_k, cache.vocab)   # toy 小词表 clamp
        fat_prompts, fat_responses = self.prompts, self.responses
        timings["stage1_cache"] = 0.0

        # P2（二次审查）：教师对与 warmup_student 在 Stage 1 后不再需要。HF 路径下它们是
        # 独立加载的完整模型（2×teacher 1.5B + 1×student 副本，7B 档 ≈21GB），不释放会
        # 让 Stage 2 训练白白多驻留（GPU 打包预算里没有这笔）。返回 dict 的 teacher 字段
        # 无任何消费方（_cmd_train 只读 metrics/timings/run_dir），置 None 即可。
        # L2（任务 6.1）：启用时【保留】teacher_rl/teacher_ref + warmup_student 供 rollout
        # 刷新相位（§3 student_ref = 初始 student = warmup_student，P1-4 独立实例）。
        # 非 L2 仍释放（原行为不变，L0/L1 静态路径零开销）。
        l2_cfg = self.cfg.get("l2", {})
        l2_enabled = bool(l2_cfg.get("enabled", False))
        # P-OPD：warmup_student 已删（无预计算/预热）；非 L2 时教师对释放（P2 语义不变）。
        if not l2_enabled:
            del teacher_rl, teacher_ref
            teacher_rl = teacher_ref = None

        # resume（T11）：加载断点学生权重 + 恢复版本号，Stage 2 从该版本续跑
        initial_version = 0
        resume_ref = None
        _resume_start = 0
        _resume_opt = _resume_rng = _resume_rb = None
        if resume is not None:
            student.load_state_dict(resume["state"])
            initial_version = int(resume.get("version", 0))
            _resume_start = int(resume.get("step", resume.get("version", 0)))
            resume_ref = resume.get("ref")       # A3/D4：断点内 KL 锚点（旧断点可能为 None）
            _resume_opt = resume.get("optimizer")
            _resume_rng = resume.get("rng")
            _resume_rb = resume.get("refresh_buffer")
            if _resume_rng:
                torch.set_rng_state(_resume_rng["py"])
                if _resume_rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state(_resume_rng["cuda"])
            logger.info(f"resume: 已加载断点学生权重，从 version={initial_version} 续跑")

        logger.info("[Stage 2] Direct-OPD 训练跑在 AsyncOPD 批量调度器上")
        t = time.perf_counter()
        # KL 正则锚点：student 初始分布（在【胖 D】上计算，与训练上下文同源）。
        #  - ref_topk>0（GPU/真实词表）→ 只存初始分布 top-K，训练时按 student 支撑取回（L4/L6 防 OOM）
        #  - 否则 dense (N,T,V)（demo 默认，小词表无压力）
        # A3/D4：resume 时若断点带 ref 锚点则【直接恢复】（跳过重算，保持
        # 「KL 锚点 = 初始 student 分布」不变式）。注意此时 student 已 load 断点
        # 权重、非初始——因此旧断点（无 ref）必须新建一个初始 student 来算锚点。
        ref_dists = ref_ids = ref_logp = None
        resume_anchor = (resume_ref or {}).get("ref_dists")
        if resume_anchor is None:
            resume_anchor = (resume_ref or {}).get("ref_ids")
        if resume_ref and resume_anchor is not None:
            # 断点张量 load 后在 CPU；按当前 device 搬移（与 response_dists 产出同设备）
            ref_dists = resume_ref.get("ref_dists")
            if ref_dists is not None:
                ref_dists = ref_dists.to(self.device)
            ref_ids = resume_ref.get("ref_ids")
            if ref_ids is not None:
                ref_ids = ref_ids.to(self.device)
            ref_logp = resume_ref.get("ref_logp")
            if ref_logp is not None:
                ref_logp = ref_logp.to(self.device)
            logger.info("resume: KL 锚点直接恢复自断点 ref（跳过重算，不变式保持）")
        else:
            anchor_model = student
            if resume is not None:                 # 旧断点无 ref：重建初始 student 算锚点
                anchor_model = build_model(self.cfg, self.device, role="student")
                _mem_point("stage2:anchor_model_ready")
                logger.info("resume: 断点无 ref 锚点，重建初始 student 重算 KL 锚点（旧断点兼容）")
            anchor_model.eval()
            # Stage 1 统一 K：ref_topk 显式给出时优先；cache_mode=topk 且未显式给 ref_topk 时
            # 回落 cache.top_k（与 student 支撑同源）。dense 场景保持 ref_topk=0 → 精确 KL，
            # 绝不因 cache.top_k 默认值把默认 dense 路径改成稀疏 KL 锚点（回归防护）。
            ref_topk = self.cfg.get("ref_topk", 0)
            if not ref_topk and self.cfg.get("cache_mode", "dense") == "topk":
                ref_topk = int((self.cfg.get("cache") or {}).get("top_k") or 0)
            if ref_topk and ref_topk > 0:
                # 真实词表（V≈152k）：KL 锚点必须【逐 chunk topk】，绝不先 cat 出完整
                # (N_fat,T,V) dense——N=2000,T=384,V=151936 时 = 233GB 必 OOM（部署实测）。
                # 每 chunk 前向 → topk 截断 → 释放，峰值 = (B,T,V) 单 chunk。
                ids_l, logp_l = [], []
                bs = s1cfg.get("build_batch_size", 16)
                with torch.no_grad():
                    for i in range(0, fat_prompts.size(0), bs):
                        sl = slice(i, min(i + bs, fat_prompts.size(0)))
                        chunk = response_dists(anchor_model, fat_prompts[sl],
                                               fat_responses[sl])     # (c,T,V)
                        Kr = min(int(ref_topk), chunk.size(-1))
                        tk = chunk.topk(Kr, dim=-1)
                        ids_l.append(tk.indices)
                        logp_l.append(tk.values)
                        del chunk
                ref_ids = torch.cat(ids_l)                             # (N,T,Kr)
                ref_logp = torch.cat(logp_l)
            else:
                with torch.no_grad():
                    ref_dists = response_dists(anchor_model, fat_prompts,
                                               fat_responses)          # (N_fat,T,V) 小词表
        # 打包 KL 锚点，随每个断点落盘（供 resume 恢复不变式）
        ref = {"ref_dists": ref_dists, "ref_ids": ref_ids, "ref_logp": ref_logp}

        # 部署键下渗已在 load_config 完成（config.py 校验前），cfg["stage2"] 天然含
        # dtype/offload_to_cpu/top_k_student；这里直接取用。
        s2cfg = dict(self.cfg["stage2"])
        # P1-B（二次审查）：model_kind / student_path 是顶层键（不在 stage2、不下渗），
        # 但 scheduler 要用它决定 worker 的构造方式（hf → HFCausalLM 副本；toy →
        # CausalToyLM 副本）。若不注入，scheduler 从 s2cfg 读 model_kind 恒为 None →
        # 恒走 CausalToyLM 分支 → 对 HFCausalLM student 取 n_layers 即 AttributeError。
        s2cfg["model_kind"] = self.cfg.get("model_kind", "toy")
        s2cfg["student_path"] = self.cfg.get("student_path")

        # ---- C1：on_step checkpoint/metrics 异步落盘（不阻塞训练线程）----
        # 训练线程只把每步结果放入有界队列；后台 daemon 消费线程串行执行
        # mr.record + cm.save。队列满（消费落后）时 _on_step 回退同步执行，防无限堆积。
        # 注意：cm.save 在消费线程执行时用「落盘时刻的 student 权重」——checkpoint 语义
        # = 落盘时刻的 student 状态（断点本就节流保存，权重以落盘时刻为准），与异步自洽。
        _on_step_q = queue.Queue(maxsize=64)
        # L2（任务 6.1）：闭包引用的 scheduler / rb 在下方 else 分支才赋值，先初始化占位
        # 避免消费线程在它们未定义时 NameError。_save_ckpt 统一封装 cm.save 增强参数。
        _scheduler_ref = None
        _rb_ref = None

        def _save_ckpt(step, version):
            _opt = _scheduler_ref.opt if _scheduler_ref is not None else None
            cm.save(step, student, version, self.cfg, metrics=[], ref=ref,
                    optimizer=_opt,
                    rng={"py": torch.get_rng_state(),
                         "cuda": (torch.cuda.get_rng_state()
                                  if torch.cuda.is_available() else None)},
                    refresh_buffer=((_rb_ref.state_dict() if _rb_ref is not None else None)
                                    if l2_enabled else None))

        def _consumer_loop():
            while True:
                m = _on_step_q.get()
                if m is None:                       # 哨兵：训练结束，停止消费
                    break
                try:
                    mr.record(m)
                    _save_ckpt(m["step"], m["version"])
                except Exception:
                    logger.exception("后台 checkpoint/metrics 落盘失败，跳过该步（step=%s）",
                                     m.get("step"))

        _consumer = threading.Thread(target=_consumer_loop, daemon=True,
                                     name="opd-onstep-consumer")
        _consumer.start()

        try:
            if bool(s2cfg.get("distributed", False)):
                # L5/L2 GPU 部署骨架：Ray 多 worker + NCCL 权重广播（取代线程版）。
                # ⚠️ 仅云 GPU 运行；需要 torch.distributed 已建组 + ray 已装。本地 CPU demo 默认不走。
                # L3 vLLM rollout 由各 Ray worker 从 cfg 自行构建（单独进程），learner 侧不持引擎。
                # 分布式骨架无 per-step 钩子：metrics 直接来自 launch_distributed_scheduler。
                metrics = launch_distributed_scheduler(
                    student, cache, fat_prompts, fat_responses,
                    ref_dists, ref_ids, ref_logp, s2cfg,
                    master_addr=s2cfg.get("master_addr", "127.0.0.1"),
                    master_port=s2cfg.get("master_port", 29500),
                    n_gpus=s2cfg.get("n_gpus", 2))
            else:
                # L3 vLLM rollout：单进程下在此构建引擎并注入 scheduler；"toy" 则不注入。
                rollout_engine = None
                dist_engines: dict | None = None   # IMP-2/P0：rollout 相位分布引擎 {s_old,rl,ref,ref_anchor}
                if s2cfg.get("rollout_engine") == "vllm":
                    from .rollout_vllm import VLLMRolloutEngine
                    # IMP-2/P1 实测修复：vLLM 引擎词表必须用【学生真实词表】
                    # （HF 路径 student.vocab = model.config.vocab_size=151936）。
                    # cfg["vocab_size"] 是 toy 默认 64，泄漏到引擎会让
                    # response_dists 稠密重建 out=(B*T*V) 越界 IndexError
                    # （2026-08-17 双卡实测：index 1049554 > size 1048576）。
                    _engine_vocab = int(getattr(student, "vocab", None) or vocab)
                    # L3/IMP-2：rollout vLLM 引擎放独立卡（rollout_device，默认 cuda:1），
                    # 避免与训练卡（cuda:0）的 student/teacher 显存冲突（vLLM 默认
                    # gpu_memory_utilization=0.9 独占）。tp_size=1 单卡 rollout；权重由
                    # 每次 rollout 相位前 update_weights 同步（on-policy）。
                    rollout_device = s2cfg.get("rollout_device", "cuda:1")
                    # P2（OOM 修复）：vLLM 按【总显存】比例预留（gpu_mem×total），
                    # 无视训练进程已占——建引擎前查剩余显存，不够立即 fail-fast
                    # （把 OOM 从训练中提前到启动，并打印双方占用）。
                    if str(rollout_device).startswith("cuda") and torch.cuda.is_available():
                        _free_gb = torch.cuda.mem_get_info(rollout_device)[0] / 2**30
                        # 2026-09-02（重建后恢复 bd74496 修复）：按 vLLM 检测到的卡显存动态算，
                        # 非硬编码 96.0——2×48GB（RTX4090）下 0.55×96=52.8 > 47 误判 fail-fast。
                        _rollout_total = torch.cuda.get_device_properties(
                            rollout_device).total_memory / 2**30
                        _eng_gb = float(s2cfg.get("rollout_gpu_mem", 0.9)) * _rollout_total
                        # 异卡（train@cuda:0 / rollout@cuda:1）时 rollout 卡无训练驻留，
                        # 只要求引擎份额；同卡（训练+vLLM 共卡）才叠加训练侧预留。
                        _min_free = (float(s2cfg.get("rollout_min_free_gb", 25.0))
                                     if _same_card(self.device, rollout_device) else 2.0)
                        if not _l2_rollout_mem_enough(_free_gb, _eng_gb, _min_free):
                            raise RuntimeError(
                                f"[L2] 剩余显存 {_free_gb:.1f}GB < vLLM 引擎预留 "
                                f"{_eng_gb:.1f}GB + 预留 {_min_free}GB（同卡训练余量/异卡 "
                                "仅引擎）；请调低 stage2.rollout_gpu_mem 或减少同卡并发"
                                "（避免训练中 OOM）。")
                    # vLLM max_model_len 安全守卫（2026-08-19）：必须 >= max_prompt_len +
                    # max_new_tokens，否则 prompt(1024)+response(512)=1536 恰好等于上限时，
                    # vLLM 内部 special token / 余量会导致截断或 init 失败（v3 实测
                    # max_model_len=6144 的 KV cache 太大也 init 失败）。
                    _check_rollout_max_model_len(self.cfg, s2cfg, l2_cfg)
                    rollout_engine = VLLMRolloutEngine(
                        # rollout 模型默认回落 student_path（on-policy 同构）；
                        # rollout_model 显式覆盖仅作诊断用。
                        model=(s2cfg.get("rollout_model")
                               or self.cfg.get("student_path")
                               or "Qwen/Qwen2.5-7B"),
                        tp_size=int(s2cfg.get("rollout_tp_size", 1)),
                        dtype=s2cfg.get("rollout_dtype", "auto"),
                        # 显存占比/上下文上限可配（默认 0.9/2048 不变）；双卡并行实验
                        # 训练+vLLM 共卡时调低 rollout_gpu_mem（如 0.5）防 OOM。
                        gpu_memory_utilization=float(s2cfg.get("rollout_gpu_mem", 0.9)),
                        max_model_len=int(s2cfg.get("rollout_max_model_len", 2048)),
                        max_num_seqs=int(s2cfg.get("rollout_max_num_seqs", 256)),
                        vocab_size=_engine_vocab,
                        full_logprobs_cap=int(s2cfg.get("rollout_logprobs_cap", 4096)),
                        weight_sync_mode=s2cfg.get("rollout_weight_sync", "auto"),
                        device=rollout_device,
                        learner_device=self.device)
                    # IMP-2/P0：rollout 相位 4 个分布前向（s_old/rl/ref/ref_anchor）切 vLLM
                    # （workflow-runner 计划）。可配置 l2.rollout.dist_engines（默认 false）；
                    # 各引擎低 gpu_memory_utilization 共存于 rollout_device（4×~0.25≤1.0）。
                    # s_old 引擎每次 rollout 前同步 student 权重（on-policy）；ref_anchor 保持
                    # 初始 student（不同步）；rl/ref 用 teacher。None → run_refresh_phase 走
                    # HF per-chunk（零回归）。
                    _l2roll = (self.cfg.get("l2") or {}).get("rollout") or {}
                    if _l2roll.get("dist_engines", False):
                        dist_engines = {}
                        # P-OPD：rl/ref 教师前向一律 HF（only_stu 需任意 ids logp，vLLM 无法
                        # 胜任）→ 不建 rl/ref vLLM 引擎（死重 OOM）。只 s_old/ref_anchor。
                        _de_specs = [
                            ("s_old", s2cfg.get("rollout_model") or self.cfg.get("student_path")),
                            ("ref_anchor", self.cfg.get("student_path")),
                        ]
                        for _k, _mp in _de_specs:
                            if _mp:
                                dist_engines[_k] = VLLMRolloutEngine(
                                    model=_mp, tp_size=1,
                                    dtype=s2cfg.get("rollout_dtype", "auto"),
                                    gpu_memory_utilization=float(
                                        _l2roll.get("dist_engine_gpu_mem", 0.25)),
                                    vocab_size=_engine_vocab,
                                    full_logprobs_cap=int(
                                        s2cfg.get("rollout_logprobs_cap", 4096)),
                                    weight_sync_mode=s2cfg.get("rollout_weight_sync", "auto"),
                                    device=rollout_device,
                                    learner_device=self.device)
                # D1（2026-08-25）固定评估集：P-OPD 下禁用——占位 cache 无预计算 Δ，无法做
                # E[Δ_T] holdout 评估（cache.slice 对 None 崩）。on-policy 信号用 rollout 相位
                # 的 E[Δ_T]（reward 指标）替代。
                _eval_holdout = int(s2cfg.get("eval_holdout_size", 0))
                _eval_every = int(s2cfg.get("eval_every", 0))
                _eval_cache = None
                _eval_prompts = None
                _eval_responses = None
                if _eval_holdout > 0 and _eval_every > 0:
                    logger.warning(
                        "[D1] eval_holdout 在 P-OPD（无预计算 Δ）下不可用 → 已禁用；"
                        "on-policy 信号用 rollout 相位 E[Δ_T]（reward 指标）替代。")
                    _eval_holdout = 0
                scheduler = AsyncBatchedScheduler(
                    student, cache, fat_prompts, fat_responses,
                    ref_dists, ref_ids, ref_logp, s2cfg, self.device,
                    rollout_engine=rollout_engine, initial_version=initial_version,
                    eval_cache=_eval_cache, eval_prompts=_eval_prompts,
                    eval_responses=_eval_responses)
                _scheduler_ref = scheduler
                # G8：resume 精确续跑——恢复 optimizer 状态（scheduler 刚新建 self.opt，
                # 从断点 load_state_dict 还原 Adam 动量/方差，否则续跑动量丢失）。
                if _resume_opt is not None:
                    try:
                        scheduler.opt.load_state_dict(_resume_opt)
                        logger.info("resume: 已恢复 optimizer 状态（精确续跑）")
                    except Exception as e:   # pragma: no cover —— 旧断点/形状失配降级
                        logger.warning(f"resume: optimizer 恢复失败（降级新建）：{e}")

                # T8/T10：每成功一步 → 指标落盘 + 按 checkpoint_every 存学生断点（供 AIME 蒸馏后评估）
                # A3/D4：断点随附 KL 锚点 ref（闭包捕获 Stage 2 已算好的 ref_dists/ids/logp）。
                # C1：_on_step 只入队（不阻塞）；队列满 → 回退同步执行（消费落后兜底）。
                def _on_step(m):
                    try:
                        _on_step_q.put_nowait(m)
                    except queue.Full:
                        mr.record(m)
                        _save_ckpt(m["step"], m["version"])

                if l2_enabled:
                    # ---- L2 交替相位循环（任务 6.1，§13 整合）----
                    # 训练相位（fit T_train 步，_train_step 一行不动）↔ rollout 刷新相位
                    # （teacher 前向在此，不在 _train_step）。关闭时走原单次 scheduler.run。
                    from .adaptive_cache import (RefreshRingBuffer, DisagreementComputer,
                                                 CacheHealthMonitor, DynamicRatioController,
                                                 RefreshSelector, PromptStateStore,
                                                 run_refresh_phase, compute_rollout_metrics,
                                                 refresh_cold_start_decision)
                    l2c = l2_cfg.get("cache", {})
                    rb = RefreshRingBuffer(
                        capacity=int(l2c.get("refresh_size", 5000)),
                        top_k=cache.top_k, vocab=cache.vocab,
                        student_top_k=int(s2cfg.get("top_k_student", 0) or cache.top_k),
                        value_protect_quantile=l2c.get("value_protect_quantile", 0.9),
                        # G7（§3.5）：sample utility U_i 驱动价值保护（L2UtilityCfg 权重）
                        utility_weights=l2_cfg.get("utility") or None)
                    _rb_ref = rb
                    # G8：resume 恢复 L2 ring buffer 内容（含行为策略 s_old + 元数据），
                    # 使续跑后 refresh 池保留、可继续被双池 feeder 消费。
                    if _resume_rb:
                        try:
                            rb.load_state_dict(_resume_rb)
                            logger.info(f"resume: 已恢复 L2 refresh ring buffer（{rb.size} 样本）")
                        except Exception as e:   # pragma: no cover —— 旧断点/形状失配降级
                            logger.warning(f"resume: ring buffer 恢复失败（空池重来）：{e}")
                    ps = PromptStateStore(n_prompts=fat_prompts.size(0))
                    disag = DisagreementComputer()
                    hm = CacheHealthMonitor(
                        l2_cfg.get("health_monitor", {}).get("health", {}),
                        alert_cooldown=int(l2_cfg.get("health_monitor", {})
                                           .get("alert_cooldown", 50)))
                    rc = l2_cfg.get("refresh_ratio", {})
                    drc = DynamicRatioController(
                        initial=rc.get("initial", 0.3), min=rc.get("min", 0.1),
                        max=rc.get("max", 0.6), mode=rc.get("mode", "adaptive"),
                        age_weight=rc.get("age_weight", 0.25),
                        drift_weight=rc.get("drift_weight", 0.5),
                        quality_weight=rc.get("quality_weight", 0.25),
                        ema_beta=rc.get("ema_beta", 0.9),
                        warmup_steps=rc.get("warmup_steps", 500),
                        max_step_change=rc.get("max_step_change", 0.05))
                    sc = l2_cfg.get("selective_rollout", {})
                    selector = (RefreshSelector(
                        ps, candidate_multiplier=sc.get("candidate_multiplier", 4),
                        value_fraction=sc.get("value_fraction", 0.8),
                        coverage_fraction=sc.get("coverage_fraction", 0.2),
                        value_weights=sc.get("value_weights"),
                        compute_aware=sc.get("compute_aware", False),
                        max_same_prompt_fraction=sc.get("max_same_prompt_fraction", 0.05),
                        exploration_fraction=sc.get("exploration_fraction", 0.2))
                        if sc.get("enabled", True) else None)
                    # 注入 scheduler 供双池 feeder（任务 1.4 接入点；当前 CPU toy 下只装配不消费）
                    scheduler._l2_enabled = True
                    scheduler._refresh_buffer = rb
                    scheduler._drc = drc
                    # G5（§2 Q4）：base 池跳过陈旧度截断（base s_old 恒新），
                    # 仅 refresh 池受截断。
                    scheduler.staleness_drop_base = False
                    # §3 初始 student（student_ref）：初始 student 副本（KL 锚点，P1-4 独立实例）。
                    student_ref = build_model(self.cfg, self.device, role="student")
                    # P3（2026-08-19）：teacher_rl/teacher_ref 只在 refresh 相位算 Δ_T 时用，
                    # offload 到 CPU 省显存；刷新相位前 reload、完成后 finally 搬回。student_ref
                    # 必须常驻（每步 ref_anchor 计算）。
                    _teacher_offload = bool(s2cfg.get("teacher_offload", False))
                    _p3_teacher_move(
                        teacher_rl, teacher_ref, "cpu",
                        enabled=_teacher_offload, device=self.device, logger=logger,
                        message="[P3] teacher offload: 训练相位（省显存）")

                    metrics = []
                    n_total = int(s2cfg.get("n_steps", 30))
                    t_train = int(l2_cfg.get("t_train", 100))
                    # P-OPD（2026-08-31）：纯 on-policy 交替相位——无 base 池（scheduler.run 不
                    # 调用），每相位 rollout（run_refresh_phase）↔ 训练（train_refresh_phase）交替，
                    # α 冻结 1.0。while 用 step_done（真实训练步）驱动；连续空训练相位（冷启动/
                    # 池空/rollout 全无效）超限明确失败（防死循环 + 静默空跑无断点）。
                    _pr_empty = 0
                    step_done = _resume_start   # resume：从断点 step 续跑（不重训已完成的步）
                    base_done = _resume_start
                    while step_done < n_total:
                        # 本相位目标训练步数（rollout 相位后由 train_refresh_phase 完成）
                        n_phase = min(t_train, n_total - step_done)
                        # rollout 刷新相位：teacher 前向在此（不在 _train_step），无条件每相位
                        # （100% on-policy）。
                        if True:
                            # IMP-1 显存修复：训练相位（scheduler.run）后 PyTorch 仍保留
                            # 大量未用缓存块（实测 alloc 44GB / reserved 78GB，~34GB 可释放）。
                            # rollout 的生成 + response_dists (M,P+T,V) 前向需要额外显存，
                            # 先 empty_cache 归还未引用块，否则生成 8×512 即 OOM（2026-08-17）。
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Stage 2：消费 l2.rollout 短预算协议（max_new_tokens / eos / loop）。
                            # fallback：未设 rollout 段 → 回落 cache.max_response_length（toy=4）。
                            rollcfg = l2_cfg.get("rollout", {})
                            max_new = int(rollcfg.get("max_new_tokens")
                                          or l2c.get("max_response_length", 8192))
                            eos_id = rollcfg.get("eos_token_id")
                            # IMP-1a：rollout 采样温度（默认 0.7，pipeline 不写死；1.0 复现旧行为）。
                            temperature = float(rollcfg.get("temperature", 0.7))
                            # IMP-1c：抗退化采样（repetition_penalty>1 抑制重复；loop_min_len 放宽误报）。
                            repetition_penalty = float(rollcfg.get("repetition_penalty", 1.0))
                            loop_min_len = int(rollcfg.get("loop_min_len", 8))
                            # IMP-1c：rollout 来源（默认 student=主路径；teacher=仅诊断/上界）。
                            rollout_source = str(rollcfg.get("rollout_source", "student"))
                            # Direct-OPD 论文 rollout n：每 prompt 每次 refresh 独立采样 n 条
                            # （默认 1 零回归；n>1 全部进入 refresh 池，多采样降方差）。
                            n_rollout = int(rollcfg.get("n_rollout", 1))
                            if rollout_source == "teacher":
                                logger.warning(
                                    "[L2] rollout_source=teacher 是诊断/上界实验专用（y~pi_teacher_rl），"
                                    "禁止默认启用、禁止混进主 E5；teacher 轨迹不构成主 on-policy 数据。")
                            # P1.5：真实 pad 判定——HF 学生用 config 真实 pad_token_id
                            # 替代 pad_id=0 近似（toy 默认 0 无碍；Qwen3 真实 pad≈1516xx）。
                            _pad_id = int(rollcfg.get("pad_id", 0))
                            if self.cfg.get("model_kind") == "hf":
                                _pad_id = int(getattr(student, "pad_token_id", None) or _pad_id)
                            # 注入 rollout_generator（绑定方法，run_refresh_phase 按
                            # GenerateWithStatus(prompts, max_new=...) 约定调用）：
                            #   vLLM 引擎优先；真实 HF 学生用 KV-cached 快速路径
                            #   generate_with_status_kv（152k 词表/长序列下 ~35 tok/s，
                            #   朴素逐 token 前向慢 1-2 个数量级）；否则 None → run_refresh_phase
                            #   内部回落 toy 模块级 generate_with_status。
                            if rollout_engine is not None:
                                _rollout_gen = getattr(rollout_engine,
                                                       "generate_with_status", None)
                            elif rollout_source == "teacher":
                                # IMP-1c：teacher rollout 用 teacher_rl 的生成器（诊断专用）
                                _rollout_gen = (getattr(teacher_rl, "generate_with_status_kv", None)
                                                or getattr(teacher_rl, "generate_with_status", None))
                            elif hasattr(student, "generate_with_status_kv"):
                                _rollout_gen = student.generate_with_status_kv
                            else:
                                _rollout_gen = None
                            # Stage 3：Budget-Aware Selective Rollout 接线（任务 5）。
                            # 仅当 budget_mode≠fixed 或显式设了 token_budget_per_refresh 才走
                            # per-sample budget 分桶；默认（budget_mode=fixed 且无 token_budget）
                            # → budgets=None 走原单预算路径，保证 Stage 2 零回归（任务 8 断言）。
                            use_budget = (sc.get("budget_mode", "fixed") != "fixed"
                                          or rollcfg.get("token_budget_per_refresh") is not None)
                            indices = budgets = None
                            budget_t = rollcfg.get("token_budget_per_refresh")
                            if use_budget:
                                _m_sel = int(l2_cfg.get("m_refresh", 1000))
                                if sc.get("budget_mode", "fixed") == "adaptive":
                                    indices, budgets = selector.select_with_budget(
                                        _m_sel, fat_prompts.size(0),
                                        budget_mode="adaptive",
                                        budget_set=sc.get("budget_set"),
                                        quantiles=sc.get("budget_quantiles"))
                                else:   # fixed 但显式设了 token_budget → 单档 fixed_budget
                                    indices, budgets = selector.select_with_budget(
                                        _m_sel, fat_prompts.size(0),
                                        budget_mode="fixed",
                                        fixed_budget=sc.get("fixed_budget", 1024))
                            # L3/IMP-2：rollout vLLM 引擎独立于 learner——每次 rollout 前把
                            # 当前 student 权重推入 vLLM（update_weights），保证 on-policy；
                            # toy（rollout_engine=None）跳过，零回归。
                            if rollout_engine is not None and hasattr(rollout_engine, "update_weights"):
                                try:
                                    _ok = rollout_engine.update_weights(student.state_dict())
                                except Exception as e:
                                    logger.warning(f"[L2] vLLM 权重同步失败（继续用引擎现有权重）：{e}")
                                    _ok = False
                                # I1：同步未通（如 vLLM>=0.16 尚未接入
                                # WeightTransferEngine）不再静默——记录指标；
                                # 配置 l2.rollout.require_weight_sync=true 时直接中止
                                # （避免正式训练静默违约 on-policy）。
                                # poisoned 必须 fail-closed：不受 require_weight_sync=false
                                # 影响——NCCL 超时后 communicator 状态不可信，不允许继续复用。
                                _raise_if_weight_sync_poisoned(
                                    rollout_engine, mr=mr,
                                    metric_key="rollout/weight_sync_poisoned",
                                    label="vLLM")
                                if _ok is False:
                                    mr.record({"rollout/weight_sync_failed": 1})
                                    if (l2_cfg.get("rollout") or {}).get("require_weight_sync", False):
                                        raise RuntimeError(
                                            "[L2] update_weights 返回 False（权重同步未通）；"
                                            "l2.rollout.require_weight_sync=true 已中止训练。")
                            # IMP-2/P0：s_old 分布引擎用当前 student 权重（on-policy）
                            if dist_engines and dist_engines.get("s_old") is not None:
                                try:
                                    dist_engines["s_old"].update_weights(student.state_dict())
                                except Exception as e:
                                    logger.warning(f"[L2] dist s_old 权重同步失败：{e}")
                                # poisoned 必须 fail-closed：不允许继续 refresh phase。
                                _raise_if_weight_sync_poisoned(
                                    dist_engines["s_old"], mr=mr,
                                    metric_key="rollout/dist_s_old_weight_sync_poisoned",
                                    label="dist s_old")
                            # P3：refresh 相位需要教师前向算 Δ_T → 搬回 GPU（base 训练已 offload）。
                            # 用 try/finally 保证：无论 rollout/refresh 训练成功或异常，教师都在
                            # 下一个 base 步开始前搬回 CPU 并 empty_cache（省 ~6.8GB）。
                            _p3_teacher_move(
                                teacher_rl, teacher_ref, self.device,
                                enabled=_teacher_offload, device=self.device, logger=logger,
                                message="[P3] teacher reload: refresh 相位（搬回 GPU）")
                            try:
                                rollout_summary = run_refresh_phase(
                                    student, teacher_rl, teacher_ref, student_ref,
                                    selector, rb, disag, fat_prompts, step_done,
                                    scheduler.staleness_q.current_version,
                                    int(l2_cfg.get("m_refresh", 1000)),
                                    max_new,
                                    cache.top_k, self.device, prompt_state=ps,
                                    rollout_generator=_rollout_gen,
                                    eos_token_id=eos_id,
                                    loop_detection=rollcfg.get("loop_detection", True),
                                    loop_periods=rollcfg.get("loop_periods", (2, 3, 4)),
                                    pad_id=_pad_id,
                                    temperature=temperature,
                                    repetition_penalty=repetition_penalty,
                                    loop_min_len=loop_min_len,
                                    rollout_source=rollout_source,
                                    compute_disagreement=bool(
                                        (l2_cfg.get("disagreement") or {}).get("enabled", True)),
                                    cand=indices, budgets=budgets, budget_t=budget_t,
                                    # P-OPD 修复：预算模式下 T 恒定需 budget_set 传入
                                    # run_refresh_phase（否则内部默认 2048 与 select 档位不一致，
                                    # 且各相位 budgets.max() 变化致 ring buffer 槽位 shape 崩）。
                                    budget_set=(sc.get("budget_set")
                                                or (256, 512, 1024, 2048)),
                                    dists_chunk=int(rollcfg.get("response_dists_chunk", 2)),
                                    n_rollout=int(rollcfg.get("n_rollout", 1)),   # C2：论文 rollout n 透传
                                    dist_engines=dist_engines)
                                # Stage 2：status 指标落盘（rollout/n_total/n_appended/n_eos/...）
                                roll_metrics = None
                                if isinstance(rollout_summary, dict):
                                    mr.record({f"rollout/{k}": v
                                               for k, v in rollout_summary.items()})
                                    # Stage 3：token 效率指标（键已带 rollout/ 前缀）落盘 mr。
                                    # 无条件调用（budgets=None 单预算也产出），供 S3 同口径对比。
                                    roll_metrics = compute_rollout_metrics(
                                        rollout_summary, budgets, budget_t)
                                    mr.record(roll_metrics)
                                    # 并入返回 metrics 列表（供 run_experiment / aggregate_stage3
                                    # 读 rollout/ 指标做 S3 同口径对比）。phase=rollout 供 n_steps
                                    # 过滤；不写 version，避免末步断点误取 rollout 行（_last_train）。
                                    metrics.append({
                                        "step": step_done,
                                        "phase": "rollout",
                                        # 原始 status 计数（rollout/n_total/n_appended/...，供
                                        # run_s2_real / test_l2_integration 消费）
                                        **{f"rollout/{k}": v
                                           for k, v in rollout_summary.items()},
                                        # 派生效率指标（compute_rollout_metrics，供 S3 同口径对比）
                                        **roll_metrics})
                                logger.info(f"[L2] rollout temperature={temperature:.3f} "
                                            f"(m_refresh={int(l2_cfg.get('m_refresh', 1000))})")
                                last_refresh = base_done
                                # Health Monitor 观测（Observe-only，不改训练）。
                                # hm.record 只按 4 个已知键做分类，其余 kwargs 原样透传，
                                # 把 rollout/ 键并入 hm.record 顶层安全、不破坏既有分类逻辑。
                                hm_metrics = hm.record(
                                    step_done, hit_rate=1.0,
                                    refresh_age_p95=0, reuse_p95=0, max_length_ratio=0,
                                    **(roll_metrics or {}))
                                mr.record(hm_metrics)
                                # P-OPD（2026-08-31）：α 冻结 1.0——训练全 on-policy（无 base anchor）。
                                alpha = 1.0
                                alpha_act = 1.0
                                n_refresh = n_phase
                                if n_refresh > 0:
                                    # IMP-1d：refresh pool 冷启动保护——池 < min_refresh_pool 时跳过
                                    # refresh 训练（不调 _train_step_refresh）；rollout metrics 照常、
                                    # ring buffer 样本不丢，记录 skip reason 与 pool size。
                                    min_refresh_pool = int(l2c.get("min_refresh_pool", 8))
                                    skip_train, skip_reason = refresh_cold_start_decision(
                                        rb.size, min_refresh_pool)
                                    guard = {"refresh_train/skipped": skip_train,
                                             "refresh_train/skip_reason": skip_reason,
                                             "refresh_pool/size": rb.size}
                                    mr.record(guard)
                                    # 并入本轮 rollout 行（不新增 phase 行，避免 n_steps 统计与
                                    # rollout 行 n_appended 断言被稀释）
                                    for _m in reversed(metrics):
                                        if isinstance(_m, dict) and _m.get("phase") == "rollout":
                                            _m.update(guard)
                                            break
                                    if skip_train:
                                        logger.info(
                                            f"[L2] refresh 训练跳过（冷启动：池 {rb.size} < "
                                            f"min_refresh_pool={min_refresh_pool}）")
                                        # P-OPD 空相位：不推进 step_done（真实训练步），计数；
                                        # 超限明确失败（防死循环 + 静默空跑无断点）。下相位继续填池。
                                        _pr_empty += 1
                                        if _pr_empty > int(l2c.get("max_empty_phases", 8)):
                                            raise RuntimeError(
                                                f"[L2] 纯 on-policy 连续 {_pr_empty} 个相位无真实"
                                                "训练步（ring buffer 未达 min_refresh_pool 或 "
                                                "rollout 全无效）；请调低 min_refresh_pool/加大 "
                                                "m_refresh/检查 rollout 质量。")
                                    else:
                                        # 2026-08-19 OOM 修复：rollout 相位（生成 + 教师前向
                                        # response_dists (M,P+T,V)）后 PyTorch 缓存分配器仍保留大量
                                        # 未用块（实测 alloc 44GB / reserved 78GB，~34GB 可释放）。
                                        # refresh 训练（_train_step_refresh 的 s_cur 前向）需要额外
                                        # 显存——先 empty_cache 归还未引用块再训练，否则 chunk=4 的
                                        # (4,T,V) response_dists 即 OOM（2026-08-19 双卡并行实测）。
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                        scheduler.metrics = []
                                        done = scheduler.train_refresh_phase(
                                            rb, alpha_act, n_refresh, step_done, _on_step)
                                        metrics.extend(scheduler.metrics)
                                        step_done += done
                                        if done == 0:
                                            # 池空（n_appended=0）且 min_refresh_pool=0：无训练步
                                            # → 计空相位（防死循环）。
                                            _pr_empty += 1
                                            if _pr_empty > int(l2c.get("max_empty_phases", 8)):
                                                raise RuntimeError(
                                                    f"[L2] 纯 on-policy 连续 {_pr_empty} 个相位"
                                                    "无真实训练步（rollout 全无效，池为空）；"
                                                    "请检查 rollout 质量/预算。")
                                        else:
                                            _pr_empty = 0   # 真实训练步发生 → 清零
                                        logger.info(
                                            f"[L2] α={alpha:.3f}→实际{alpha_act:.3f}，"
                                            f"refresh 训练 {done} 步（池 {rb.size}）")
                            finally:
                                # P3：refresh 完成后教师回 CPU + 释放缓存（异常路径也执行）。
                                _p3_teacher_move(
                                    teacher_rl, teacher_ref, "cpu",
                                    enabled=_teacher_offload, device=self.device,
                                    logger=logger,
                                    message="[P3] teacher offload: refresh 完成（回 CPU + empty_cache）")
                else:
                    # P-OPD（2026-08-31）：base 池训练（scheduler.run）已删除——所有训练必须
                    # 走 l2.enabled=true 纯 on-policy 交替相位（无预计算教师得分/固定 D）。
                    raise RuntimeError(
                        "base 池训练已删除（纯 on-policy，无预计算教师得分/固定 D）。"
                        "请设 l2.enabled=true（用 configs/qwen3_r1_onpolicy.yaml）。")
        finally:
            # C1：drain 后台队列（成功/异常都执行），确保 metrics/checkpoint 全部落盘
            try:
                _on_step_q.put(None)
            except queue.Full:      # 防御性：理论不可达（消费者在持续取）
                pass
            _consumer.join()
        timings["stage2_train"] = time.perf_counter() - t
        timings["total"] = sum(timings.values())

        # 计时落盘（时间优化指标1的证据）
        with open(os.path.join(paths["run_dir"], "timings.json"), "w", encoding="utf-8") as f:
            json.dump({k: round(v, 4) for k, v in timings.items()}, f, indent=2)
        # 末步断点无条件存（保证最终状态可被 AIME 蒸馏后评估），随附 KL 锚点 ref。
        # ⚠️ metrics 尾部可能是 rollout 相位行（phase=rollout，无 version）——取最后一个
        # 训练步（含 version）作末步断点。
        _last_train = next((m for m in reversed(metrics)
                            if isinstance(m, dict) and "version" in m), None)
        last_ck = cm.save(_last_train["step"], student, _last_train["version"],
                          self.cfg, force=True, ref=ref,
                          optimizer=(_scheduler_ref.opt if _scheduler_ref is not None
                                     else None),
                          rng={"py": torch.get_rng_state(),
                               "cuda": (torch.cuda.get_rng_state()
                                        if torch.cuda.is_available() else None)},
                          refresh_buffer=((_rb_ref.state_dict() if _rb_ref is not None else None)
                                          if l2_enabled else None)) \
            if _last_train is not None else None
        logger.info(f"训练完成: {len(metrics)} 步, 总耗时 {timings['total']:.2f}s, 断点 {last_ck or '无'}")

        return {
            "teacher_rl": teacher_rl,
            "teacher_ref": teacher_ref,
            "cache": cache,
            "student": student,
            "metrics": metrics,
            "timings": timings,
            "run_dir": paths["run_dir"],
            "checkpoints": paths["checkpoint_dir"],
            "metrics_csv": paths["metrics_csv"],
        }

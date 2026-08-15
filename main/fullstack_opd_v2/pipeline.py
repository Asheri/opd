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

from .cache import TensorTeacherCache
from .cache_store import (DiskTeacherCache, hash_models_from_cfg,
                          load_cache_metadata, verify_consistency,
                          write_cache_disk)
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


# ----------------------------- Stage 1 -----------------------------
def stage1_build_cache(prompts, responses, teacher_rl, teacher_ref,
                       cfg: dict, warmup_student=None,
                       storage: str = "memory", hashes: dict | None = None,
                       pad_id: int = 0):
    """Lightning-OPD：批量预计算教师对并预计算 Δ_T，训练期不再启 teacher server。

    cache_mode="topk"（且 top_k_teacher>0）时走 L4 稀疏缓存：每位置只存 teacher top-K，
    体积从 (N,T,V) 降到 (N,T,K)；真实词表 V=128k 下才存得下。

    **L1（离线 rollout 暖缓存，缓解曝光偏差）**：若 `cfg["warmup_M"]>0` 且
    `warmup_source≠"none"`，在 build 前用 student / teacher 分布对每个 prompt 额外
    采样 M 条响应，拼成「胖 D」`(N·(1+K),T)`（K=额外条数），使缓存覆盖学生/教师分布
    支撑。返回的 `fat_prompts/fat_responses` 供 Stage 2 入口的 KL 锚点与调度器使用。
    warmup 关闭时退化为原行为：`fat_* = 原 (prompts, responses)`。

    返回 `(cache, fat_prompts, fat_responses)`。
    """
    # Stage 1 统一 K：新口径以 cache.top_k 为准；兼容旧 top_k_teacher（下渗后仍在顶层）。
    cache_block = cfg.get("cache") or {}
    top_k = (cache_block.get("top_k")
             if cfg.get("cache_mode", "dense") == "topk"
             else 0) or cfg.get("top_k_teacher", 0) or 0
    cache = TensorTeacherCache(cfg.get("enforce_teacher_consistency", True), top_k=top_k)

    warmup_M = int(cfg.get("warmup_M", 0) or 0)
    source = cfg.get("warmup_source", "none")
    temp = float(cfg.get("warmup_temperature", 1.0))
    T = responses.size(1)

    _VALID_SOURCES = ("student_init", "teacher_perturbed", "mix")
    fat_prompts, fat_responses = prompts, responses
    if warmup_M > 0 and source not in (None, "none"):
        if source not in _VALID_SOURCES:
            raise DataError(
                f"stage1 warmup_source 必须是 {_VALID_SOURCES} 之一，收到 {source!r}")
        extra_p, extra_r = [], []
        # 注：采样只消耗 RNG、不改权重；warmup_student 即 Stage 2 的同一初始 student，
        # 因此「warmup 上下文分布」与「KL 锚点分布」同源，曝光偏差缓解自洽。
        if source in ("student_init", "mix"):
            if warmup_student is None:
                raise DataError(
                    "stage1 warmup_source='student_init'/'mix' 需要 warmup_student "
                    "(请在上游把初始 student 传入 stage1_build_cache)。")
            for _ in range(warmup_M):
                rp = generate_batch(warmup_student, prompts, max_new=T,
                                    temperature=temp)        # (B,T) 学生初始分布采样
                extra_p.append(prompts)
                extra_r.append(rp)
        if source in ("teacher_perturbed", "mix"):
            for _ in range(warmup_M):
                rt = generate_batch(teacher_rl, prompts, max_new=T,
                                    temperature=temp)        # (B,T) 教师扰动分布采样
                extra_p.append(prompts)
                extra_r.append(rt)
        if extra_p:
            fat_prompts = torch.cat([prompts, *extra_p], dim=0)
            fat_responses = torch.cat([responses, *extra_r], dim=0)

    cache.build(fat_prompts, fat_responses, teacher_rl, teacher_ref,
                cfg.get("build_batch_size", 16))
    # 落盘（Stage 1）：storage="disk" → 磁盘 mmap（最小 sufficient statistics + metadata +
    # checksum + 一致性哈希，解决 50K×8192 显存墙）；否则原 torch.save 全量缓存 +
    # fat 上下文（模块2：`opd train --set stage1.load_cache=true` 载入后跳过 Stage 0/1）。
    cache_path = cfg.get("cache_path", "fullstack_opd_cache_v2.pt")
    if storage == "disk":
        write_cache_disk(cache, cache_path, responses=fat_responses, pad_id=pad_id,
                         hashes=hashes, max_response_len=fat_responses.size(1),
                         max_prompt_len=fat_prompts.size(1),
                         dtype=str(cfg.get("dtype", "bf16")),
                         dataset_size=fat_prompts.size(0))
    else:
        cache.save(cache_path, prompts=fat_prompts, responses=fat_responses)
    return cache, fat_prompts, fat_responses


# ----------------------------- 编排器 -----------------------------
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
        self.prompts, self.responses, self.reward_fn = build_data_loader(
            self.cfg, self.device).load()

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

        # 模块2：多学生复用的「载入预建缓存」分支——`opd cache` 预建后（含 fat 上下文），
        # 训练跳过 Stage 0/1（教师 RL + cache build），直接载入。默认 load_cache=false
        # 走原重建路径。共享的只有教师权重；默认 student_init 下每学生缓存各建（§7.0）。
        s1cfg = dict(self.cfg["stage1"])
        cache_path = s1cfg.get("cache_path")
        # Stage 1：storage=disk 的缓存是「前缀」多文件（<prefix>.metadata.json 等），
        # 存在性按 metadata.json 判断；memory 按单个 .pt 判断。
        cache_block = self.cfg.get("cache") or {}
        storage = cache_block.get("storage", "memory")
        # dense 模式忽略 storage（磁盘只存 top-K 支撑）；dense 一律走内存 .pt 路径，
        # 避免 cache.storage 默认 disk 把默认 dense 配置错误导向磁盘。
        if self.cfg.get("cache_mode", "dense") != "topk":
            storage = "memory"
        # P-回归修复：stage1_build_cache 收到的是 s1cfg（stage1 子字典），读不到顶层
        # cache 块的 top_k（S1-4 统一 K 后参数改为 cache:{top_k,storage} 块，old 下渗槽位
        # top_k_teacher 恒为 0），导致 top_k 恒解析为 0 → 恒走 dense build → 真实词表下
        # 累积完整 (N,T,V) 两个列表（rl_full/ref_full）→ 80GB OOM。
        # 把顶层 cache.top_k 注入 s1cfg["top_k_teacher"]（stage1_build_cache 的回落槽位）。
        if self.cfg.get("cache_mode", "dense") == "topk":
            s1cfg["top_k_teacher"] = int(
                cache_block.get("top_k") or s1cfg.get("top_k_teacher") or 0)
        if storage == "disk":
            prebuilt_exists = bool(cache_path and os.path.isfile(f"{cache_path}.metadata.json"))
        else:
            prebuilt_exists = bool(cache_path and os.path.isfile(cache_path))
        use_prebuilt = bool(s1cfg.get("load_cache", False)) and prebuilt_exists

        # L1：把 student 提前创建，使「离线 warmup 采样」与「Stage 2 KL 锚点」共享同一份
        # 初始分布（两者都用初始 student 的分布，曝光偏差缓解才自洽）。
        student = build_model(self.cfg, self.device, role="student")
        # P1-4：warmup 专用独立初始 student——resume 会把断点权重 load 进 student，
        # 若 warmup 复用 student 则采样分布变成「续跑后的学生」，与 KL 锚点（初始
        # 分布，resume 从断点 ref 恢复）不同源，曝光偏差不变式被打破。此实例永不
        # 被 load_state_dict 覆盖，始终代表初始分布。
        # ⚠️ 多学生显存（部署实测）：build 路径还建 worker（旧快照），HF 下 student+
        # warmup_student+worker = 3 份模型——7B 档 + 教师对 + 优化器 > 96GB 必 OOM。
        # 故 warmup_student 仅在 warmup_M>0（真需要采样胖 D）时才建；L0（warmup_M=0）
        # 或载入预建缓存时都不建。
        warmup_student = (build_model(self.cfg, self.device, role="student")
                          if (not use_prebuilt
                              and int(s1cfg.get("warmup_M", 0) or 0) > 0) else None)

        teacher_rl = teacher_ref = None
        if use_prebuilt:
            logger.info(f"[Stage 0/1] 跳过：载入预建缓存 {cache_path}")
            t = time.perf_counter()
            if storage == "disk":
                # Stage 1：磁盘 mmap 缓存。先校验 metadata 一致性（tokenizer/教师/ref/
                # top_k/长度/checksum，不匹配 fail fast），再 batch-local mmap 加载。
                meta = load_cache_metadata(cache_path)
                hashes_now = hash_models_from_cfg(self.cfg)
                verify_consistency(meta, self.cfg, hashes_now)
                cache = DiskTeacherCache(cache_path, device=self.device,
                                         top_k=int(meta["top_k"]),
                                         vocab=int(meta["vocab"]))
                # 磁盘缓存不持久化 prompts/responses（最小 sufficient statistics）——
                # fat 上下文即当前数据集行（磁盘场景 warmup_M=0，无胖 D 扩展）。校验缓存
                # 行数与数据集一致，防止缓存与数据不同源时 scheduler 越界/错位。
                fat_prompts, fat_responses = self.prompts, self.responses
                if int(meta["num_samples"]) != fat_prompts.size(0):
                    raise DataError(
                        f"磁盘缓存 num_samples={meta['num_samples']} 与当前数据集行数 "
                        f"{fat_prompts.size(0)} 不一致；磁盘缓存只持久化 Δ_T，必须与数据集同源"
                        "（磁盘场景不支持 warmup 胖 D 扩展）")
            else:
                cache = TensorTeacherCache.load(cache_path)
                # P1-A（二次审查）：load() 用 map_location="cpu" 把缓存钉在 CPU——GPU 训练
                # 路径若不管搬设备，KL 锚点（CUDA 模型吃 CPU 输入）与 scheduler 索引
                # （CUDA idxs 索引 CPU 张量）必崩。build 路径的 fat_* 本就 device 张量，
                # 只有 load 路径需要显式搬（对称 resume 分支对 ref 张量的 .to(self.device)）。
                cache.to(self.device)
                # P3（二次审查）：load_cache 跳过 build，teacher 一致性校验（TensorTeacherCache
                # .build 开头）不再触发——此处补词表一致性把关：缓存词表 ≠ 当前模型词表时，
                # dense 路径 shape 崩溃、稀疏路径 scatter 越界/静默错位。
                sv = getattr(student, "vocab", None)
                if cache.vocab and sv and cache.vocab != sv:
                    raise DataError(
                        f"预建缓存词表 {cache.vocab} 与当前模型词表 {sv} 不匹配；"
                        "缓存与模型必须同词表（teacher 一致性语义）")
                fat_prompts, fat_responses = cache.prompts, cache.responses
                if fat_prompts is None or fat_responses is None:
                    raise DataError(
                        f"预建缓存 {cache_path} 未含 fat prompts/responses（旧格式）；"
                        "请用新版 `opd cache` 子命令重建（stage1_build_cache 已把 fat 上下文落盘）")
            timings["stage0_rl"] = 0.0
            timings["stage1_cache"] = time.perf_counter() - t
        else:
            logger.info("[Stage 0] 小模型 RL（批量 REINFORCE）→ post-RL weak teacher")
            t = time.perf_counter()
            teacher_rl, teacher_ref = self._stage0_teachers()
            timings["stage0_rl"] = time.perf_counter() - t

            logger.info("[Stage 1] Lightning-OPD 离线缓存教师对 Δ_T（批量预计算，无 live teacher）")
            t = time.perf_counter()
            # 部署键下渗已在 load_config 完成（config.py 校验前），cfg["stage1"] 天然含
            # cache_mode/top_k_teacher（顶层 CLOUD_CONFIG 风格也生效）；这里直接取用。
            # L1：warmup_M>0 时额外 rollout 采样拼成「胖 D」，返回 (cache, fat_p, fat_r)。
            cache, fat_prompts, fat_responses = stage1_build_cache(
                self.prompts, self.responses, teacher_rl, teacher_ref, s1cfg,
                warmup_student=warmup_student,
                storage=storage, hashes=hash_models_from_cfg(self.cfg),
                pad_id=int((self.cfg.get("dataset") or {}).get("pad_id", 0)))
            timings["stage1_cache"] = time.perf_counter() - t

        # P2（二次审查）：教师对与 warmup_student 在 Stage 1 后不再需要。HF 路径下它们是
        # 独立加载的完整模型（2×teacher 1.5B + 1×student 副本，7B 档 ≈21GB），不释放会
        # 让 Stage 2 训练白白多驻留（GPU 打包预算里没有这笔）。返回 dict 的 teacher 字段
        # 无任何消费方（_cmd_train 只读 metrics/timings/run_dir），置 None 即可。
        # L2（任务 6.1）：启用时【保留】teacher_rl/teacher_ref + warmup_student 供 rollout
        # 刷新相位（§3 student_ref = 初始 student = warmup_student，P1-4 独立实例）。
        # 非 L2 仍释放（原行为不变，L0/L1 静态路径零开销）。
        l2_cfg = self.cfg.get("l2", {})
        l2_enabled = bool(l2_cfg.get("enabled", False))
        if not l2_enabled:
            del teacher_rl, teacher_ref, warmup_student
            teacher_rl = teacher_ref = None

        # resume（T11）：加载断点学生权重 + 恢复版本号，Stage 2 从该版本续跑
        initial_version = 0
        resume_ref = None
        # §B 精确续跑（L2，任务 6.1）：断点可含 optimizer state + RNG + L2 ring buffer
        _resume_opt = _resume_rng = _resume_rb = None
        if resume is not None:
            student.load_state_dict(resume["state"])
            initial_version = int(resume.get("version", 0))
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
                if s2cfg.get("rollout_engine") == "vllm":
                    from .rollout_vllm import VLLMRolloutEngine
                    rollout_engine = VLLMRolloutEngine(
                        model=s2cfg.get("rollout_model", "Qwen/Qwen2.5-7B"),
                        tp_size=int(s2cfg.get("rollout_tp_size", 1)),
                        dtype=s2cfg.get("rollout_dtype", "auto"),
                        vocab_size=vocab,
                        full_logprobs_cap=int(s2cfg.get("rollout_logprobs_cap", 4096)),
                        device=self.device)
                scheduler = AsyncBatchedScheduler(
                    student, cache, fat_prompts, fat_responses,
                    ref_dists, ref_ids, ref_logp, s2cfg, self.device,
                    rollout_engine=rollout_engine, initial_version=initial_version)
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
                                                 run_refresh_phase, compute_rollout_metrics)
                    l2c = l2_cfg.get("cache", {})
                    rb = RefreshRingBuffer(
                        capacity=int(l2c.get("refresh_size", 5000)),
                        top_k=cache.top_k, vocab=cache.vocab,
                        student_top_k=int(s2cfg.get("top_k_student", 0) or cache.top_k),
                        value_protect_quantile=l2c.get("value_protect_quantile", 0.9))
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
                    sc_tok = l2_cfg.get("selective_rollout", {})
                    drc = DynamicRatioController(
                        initial=rc.get("initial", 0.3), min=rc.get("min", 0.1),
                        max=rc.get("max", 0.6), mode=rc.get("mode", "adaptive"),
                        age_weight=rc.get("age_weight", 0.25),
                        drift_weight=rc.get("drift_weight", 0.5),
                        quality_weight=rc.get("quality_weight", 0.25),
                        ema_beta=rc.get("ema_beta", 0.9),
                        warmup_steps=rc.get("warmup_steps", 500),
                        max_step_change=rc.get("max_step_change", 0.05),
                        token_aware=sc_tok.get("token_aware", False),
                        token_weight=sc_tok.get("token_weight", 0.1))
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
                    # §3 初始 student（student_ref）：warmup_student 即 P1-4 独立实例；
                    # warmup_M=0 时未建，则新建一份初始 student 副本。
                    student_ref = warmup_student if warmup_student is not None else \
                        build_model(self.cfg, self.device, role="student")

                    metrics = []
                    n_total = int(s2cfg.get("n_steps", 30))
                    t_train = int(l2_cfg.get("t_train", 100))
                    # G4：refresh 触发时机由 min/max_interval 约束（§2 Q1），非无条件每相位。
                    # 初始 last_refresh 置负，保证首个相位必触发一次 refresh（冷启动）。
                    refresh_min = max(1, int(l2c.get("refresh_min_interval", 50)))
                    refresh_max = max(refresh_min, int(l2c.get("refresh_max_interval", 150)))
                    last_refresh = -refresh_min
                    # base_done 只计 base 训练步（=n_total 口径）；step_done 全局单调（含 refresh
                    # 补充步），供 step 编号与 checkpoint 文件名递增不冲突。
                    base_done = 0
                    step_done = 0
                    while base_done < n_total:
                        # 训练相位：跑 n_phase 步（_train_step teacher-free 不动）。
                        # ⚠️ scheduler.run 每次会 set self.stop 且 metrics 跨相位累计——
                        # 相位边界必须重置 stop 事件与 metrics，否则后续相位线程立即退出。
                        scheduler.stop.clear()
                        scheduler.metrics = []
                        n_phase = min(t_train, n_total - base_done)
                        metrics.extend(scheduler.run(n_phase, on_step=_on_step,
                                                     start_step=step_done))
                        step_done += n_phase
                        base_done += n_phase
                        # rollout 刷新相位：teacher 前向在此（不在 _train_step）。
                        # G4：距上次刷新 >= min_interval 才触发（max_interval 强制），
                        # 否则本相位纯训练、跳过 refresh 与 α 更新。
                        elapsed = base_done - last_refresh
                        if (elapsed >= refresh_min or elapsed >= refresh_max) \
                                and selector is not None:
                            # Stage 2：消费 l2.rollout 短预算协议（max_new_tokens / eos / loop）。
                            # fallback：未设 rollout 段 → 回落 cache.max_response_length（toy=4）。
                            rollcfg = l2_cfg.get("rollout", {})
                            max_new = int(rollcfg.get("max_new_tokens")
                                          or l2c.get("max_response_length", 8192))
                            eos_id = rollcfg.get("eos_token_id")
                            # 注入 rollout_generator（绑定方法，run_refresh_phase 按
                            # GenerateWithStatus(prompts, max_new=...) 约定调用）：
                            #   vLLM 引擎优先；真实 HF 学生用 KV-cached 快速路径
                            #   generate_with_status_kv（152k 词表/长序列下 ~35 tok/s，
                            #   朴素逐 token 前向慢 1-2 个数量级）；否则 None → run_refresh_phase
                            #   内部回落 toy 模块级 generate_with_status。
                            if rollout_engine is not None:
                                _rollout_gen = getattr(rollout_engine,
                                                       "generate_with_status", None)
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
                                cand=indices, budgets=budgets, budget_t=budget_t)
                            # Stage 2：status 指标落盘（rollout/n_total/n_appended/n_eos/...）
                            roll_metrics = None
                            if isinstance(rollout_summary, dict):
                                mr.record({f"rollout/{k}": v
                                           for k, v in rollout_summary.items()})
                                # Stage 3：token 效率指标（键已带 rollout/ 前缀）落盘 mr。
                                roll_metrics = compute_rollout_metrics(
                                    rollout_summary, budgets, budget_t)
                                mr.record(roll_metrics)
                            last_refresh = base_done
                            # Health Monitor 观测（Observe-only，不改训练）。
                            # hm.record 只按 4 个已知键（hit_rate/refresh_age_p95/reuse_p95/
                            # max_length_ratio）做分类，其余 kwargs 原样透传，故把 rollout/ 键
                            # 并入 hm.record 顶层安全、不破坏既有分类逻辑（任务 5 谨慎要求）。
                            hm_metrics = hm.record(
                                step_done, hit_rate=1.0,
                                refresh_age_p95=0, reuse_p95=0, max_length_ratio=0,
                                **(roll_metrics or {}))
                            mr.record(hm_metrics)
                            # Dynamic Ratio 调 α（consume metrics，非 Monitor 闭环）
                            # 任务 6：传 rollout_efficiency（expected/actual tokens，§五）。
                            # >1 省 token → 放宽 α；<1 超用 → 收紧。方向修正（此前写反）。
                            _eff = (rollout_summary.get("expected_rollout_tokens") / max(
                                1, rollout_summary.get("rollout_tokens", 1))
                                if isinstance(rollout_summary, dict) else None)
                            alpha = drc.update(
                                base_age=hm_metrics.get("age/mean", 0),
                                policy_drift=0,
                                refresh_quality=rb.mean_disagreement(),
                                rollout_efficiency=_eff)
                            # G3：α 真实应用——refresh 训练步数 = α/(1-α)·n_base（双池 feeder）。
                            # cold start：refresh 池不足时 α_actual 收缩（§5.5）。
                            alpha_act = drc.cold_start_adjust(
                                alpha, rb.size, max(1, n_phase * scheduler.batch))
                            n_refresh = int(round(alpha_act / max(1e-6, 1 - alpha_act) * n_phase))
                            n_refresh = max(0, min(n_refresh, n_phase))  # 不超 base 步数
                            if n_refresh > 0:
                                scheduler.metrics = []
                                done = scheduler.train_refresh_phase(
                                    rb, alpha_act, n_refresh, step_done, _on_step)
                                metrics.extend(scheduler.metrics)
                                step_done += done
                                logger.info(
                                    f"[L2] α={alpha:.3f}→实际{alpha_act:.3f}，"
                                    f"refresh 训练 {done} 步（池 {rb.size}）")
                else:
                    metrics = scheduler.run(s2cfg.get("n_steps", 30), on_step=_on_step)
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
        # 末步断点无条件存（保证最终状态可被 AIME 蒸馏后评估），随附 KL 锚点 ref
        last_ck = cm.save(metrics[-1]["step"], student, metrics[-1]["version"],
                          self.cfg, force=True, ref=ref,
                          optimizer=(_scheduler_ref.opt if _scheduler_ref is not None
                                     else None),
                          rng={"py": torch.get_rng_state(),
                               "cuda": (torch.cuda.get_rng_state()
                                        if torch.cuda.is_available() else None)},
                          refresh_buffer=((_rb_ref.state_dict() if _rb_ref is not None else None)
                                          if l2_enabled else None)) \
            if metrics else None
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

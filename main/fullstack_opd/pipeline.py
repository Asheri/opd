"""FullStackOPD 编排器：把三篇论文串成一条流水线。

小模型 RL ──► 离线缓存教师对 Δ_T ──► Direct-OPD 训练跑在 AsyncOPD 调度器上
"""

from __future__ import annotations

import time

import torch

from .models import ToyModel
from .stages import stage0_small_rl, stage1_build_cache, stage2_train

DEFAULT_CONFIG = {
    "vocab_size": 64,        # demo 极小词表；真实为 32k~150k
    "d_model": 48,
    "n_layers": 2,
    "prompt_len": 6,
    "resp_len": 8,
    "n_prompts": 16,
    "seed": 42,              # ★ 修复：全局播种，保证 stage0/模型初始化可复现
    "stage0": {              # 小模型 RL（产生 post-RL weak teacher）
        "d_model": 48, "n_layers": 2, "lr": 1e-3,
        "n_rl_steps": 40, "max_new_tokens": 8,
    },
    "stage1": {              # Lightning-OPD 离线缓存
        "enforce_teacher_consistency": True,
        "cache_path": "fullstack_opd_cache.pt",
    },
    "stage2": {              # Direct-OPD + AsyncOPD
        "scheduling_mode": "fully_async",
        "staleness_threshold": 4,
        "queue_size": 8,
        "kl_reg_coef": 0.05,
        "clip_eps": 0.2,
        "grad_clip": 1.0,    # ★ 修复：梯度裁剪（真实 RL 标配）
        "lr": 1e-3,
        "n_steps": 30,
    },
}


class FullStackOPD:
    def __init__(self, cfg: dict | None = None, device: str = "cpu"):
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.device = device
        self._make_toy_data()

    def _make_toy_data(self):
        vocab = self.cfg["vocab_size"]
        rng = torch.Generator().manual_seed(0)
        self.prompts = [
            torch.randint(0, vocab, (self.cfg["prompt_len"],), generator=rng)
            for _ in range(self.cfg["n_prompts"])
        ]
        self.responses = [
            torch.randint(0, vocab, (self.cfg["resp_len"],), generator=rng)
            for _ in range(self.cfg["n_prompts"])
        ]
        # 「偶数 token」= demo 里 RL 追求的「好 action」集合
        self.good_set = set(range(0, vocab, 2))

    def run(self) -> dict:
        # ★ 修复：全局播种（ToyModel 初始化、stage0 采样此前都走未播种的全局 RNG）
        torch.manual_seed(self.cfg.get("seed", 42))
        vocab = self.cfg["vocab_size"]
        d_model = self.cfg["d_model"]
        n_layers = self.cfg["n_layers"]
        timings: dict = {}

        # ---- Stage 0 ----
        print("[Stage 0] 小模型 RL → post-RL weak teacher (+ pre-RL reference)")
        t = time.perf_counter()
        teacher_rl, teacher_ref = stage0_small_rl(
            self.prompts, self.cfg["stage0"], self.device, vocab, self.good_set
        )
        timings["stage0_rl"] = time.perf_counter() - t

        # ---- Stage 1 ----
        print("[Stage 1] Lightning-OPD 离线缓存教师对 Δ_T（无 live teacher）")
        t = time.perf_counter()
        cache = stage1_build_cache(
            self.prompts, self.responses, teacher_rl, teacher_ref,
            self.cfg["stage1"], self.device
        )
        timings["stage1_cache"] = time.perf_counter() - t

        # ---- Stage 2 ----
        print("[Stage 2] Direct-OPD 训练跑在 AsyncOPD 调度器上")
        t = time.perf_counter()
        student_init = ToyModel(vocab=vocab, d_model=d_model,
                                n_layers=n_layers).state_dict()
        student, metrics = stage2_train(
            cache, student_init, self.prompts, self.responses,
            self.cfg["stage2"], self.device, vocab, d_model, n_layers
        )
        timings["stage2_train"] = time.perf_counter() - t
        timings["total"] = sum(timings.values())

        return {
            "teacher_rl": teacher_rl,
            "teacher_ref": teacher_ref,
            "cache": cache,
            "student": student,
            "metrics": metrics,
            "timings": timings,
        }

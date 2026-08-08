"""三阶段 stage 函数：

  stage0_small_rl     : 小模型 RL → post-RL weak teacher (+ pre-RL reference 副本)
  stage1_build_cache  : Lightning-OPD 离线缓存「教师对」Δ_T（无 live teacher）
  stage2_train        : Direct-OPD 训练跑在 AsyncOPD 调度器上

真实工程里这些 stage 分别由三个 clone 下来的 repo 承担：
  stage0  → Direct-OPD/verl 的 RLVR（GRPO/PPO）产出 post-RL weak teacher
  stage1  → Lightning-OPD/data_curation/prepare_lightning_opd.py
  stage2  → async-opd/opd 的调度器 + Direct-OPD/verl 的 dp_actor
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .lightning_cache import OfflineTeacherPairCache
from .models import ToyModel
from .async_scheduler import AsyncOPDScheduler


# ----------------------------- Stage 0 -----------------------------
def logp_of_response(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                     device) -> torch.Tensor:
    """返回 (T,)：response 每个 token 在 model 当前参数下的对数概率（保留梯度）。"""
    full = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0).to(device)
    logits = model(full)
    logp = F.log_softmax(logits, dim=-1)
    P = prompt_ids.size(0)
    T = response_ids.size(0)
    assert P >= 1, "demo 要求非空 prompt"
    pred_positions = torch.arange(P, P + T, device=device) - 1
    lp = logp[0, pred_positions].gather(1, response_ids.unsqueeze(-1).to(device)).squeeze(-1)
    return lp


def _toy_reward(response_ids: torch.Tensor, good_set: set) -> torch.Tensor:
    """demo 用的规则奖励：token 在 good_set（偶数 token）则 +1，否则 -0.2。
    小模型 RL 会因此把概率质量推向「偶数 token」——这就是我们要复用的 RL 策略偏移。
    """
    return torch.tensor([1.0 if int(t) in good_set else -0.2 for t in response_ids],
                        dtype=torch.float32)


def stage0_small_rl(prompts, cfg: dict, device, vocab: int, good_set: set):
    """小模型 RL：产生 post-RL weak teacher；pre-RL reference 为其训练前副本。

    返回 (teacher_rl, teacher_ref)，都是 ToyModel。
    """
    d_model = cfg.get("d_model", 48)
    n_layers = cfg.get("n_layers", 2)
    weak = ToyModel(vocab=vocab, d_model=d_model, n_layers=n_layers).to(device)
    ref = ToyModel(vocab=vocab, d_model=d_model, n_layers=n_layers).to(device)
    ref.load_state_dict(weak.state_dict())  # pre-RL reference = 训练前副本

    opt = torch.optim.Adam(weak.parameters(), lr=cfg.get("lr", 1e-3))
    n_steps = cfg.get("n_rl_steps", 40)
    max_new = cfg.get("max_new_tokens", 8)

    for step in range(n_steps):
        weak.train()
        idx = step % len(prompts)
        p = prompts[idx].to(device)
        with torch.no_grad():
            r = weak.generate(p, max_new=max_new, device=device)
        logp = logp_of_response(weak, p, r, device)         # (T,) 保留梯度
        reward = _toy_reward(r, good_set).to(device)
        # REINFORCE + baseline（均值基线减方差）：最大化 Σ_t logπ(a_t|s_t) · (reward_t − b)
        baseline = reward.mean()
        loss = -(logp * (reward - baseline)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return weak, ref


# ----------------------------- Stage 1 -----------------------------
def stage1_build_cache(prompts, responses, teacher_rl, teacher_ref, cfg: dict,
                       device) -> OfflineTeacherPairCache:
    """Lightning-OPD：一次性预计算「教师对」Δ_T 并落盘，训练期不再启 teacher server。"""
    cache = OfflineTeacherPairCache(
        enforce_consistency=cfg.get("enforce_teacher_consistency", True)
    )
    cache.build(prompts, responses, teacher_rl, teacher_ref, device)
    cache.save(cfg.get("cache_path", "lightning_cache.pt"))
    return cache


# ----------------------------- Stage 2 -----------------------------
def stage2_train(cache, student_init_state, prompts, responses, cfg: dict,
                 device, vocab: int, d_model: int, n_layers: int):
    """Direct-OPD 训练跑在 AsyncOPD 调度器上。返回 (student, metrics)。"""
    student = ToyModel(vocab=vocab, d_model=d_model, n_layers=n_layers).to(device)
    student.load_state_dict(student_init_state)
    scheduler = AsyncOPDScheduler(student, cache, prompts, responses, cfg, device)
    metrics = scheduler.run(cfg.get("n_steps", 30))
    return student, metrics

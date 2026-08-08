"""★ AsyncOPD：KL 损失 + learner 时刻「用当前 student 重算」代理。

对应真实代码：async-opd/opd/loss/kl.py
  - sparse_forward_kl   KL(teacher||student)：对陈旧 rollout 鲁棒
  - sparse_reverse_kl   KL(student||teacher)：脆弱
  - dense_aligned_kl(mode="reverse_kl") / chunked_dense_kl_from_hidden：
        learner 时刻用当前 student 重算 reverse-KL 信号（稳健代理）
  - async-opd/opd/trainer/ac_opd.py 的 prox_logprobs

关键发现：Direct-OPD 把 Δ_T 当 reward（PG 形式）本身即「用当前 student 重算」的
稳健估计量，因此全栈叠加天然用上 AsyncOPD 推荐的稳健估计量。这里用 PPO clip 把
陈旧 (old) 样本的 ratio 截断，等价于 learner 时刻重算。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_forward_kl(student_logits, teacher_topk_logps, teacher_topk_indices,
                      mask=None) -> torch.Tensor:
    """KL(teacher || student) —— 对陈旧 rollout 鲁棒（AsyncOPD 发现）。"""
    student_topk = torch.gather(F.log_softmax(student_logits, dim=-1),
                                -1, teacher_topk_indices)
    kl = (teacher_topk_logps.exp() * (teacher_topk_logps - student_topk)).sum(-1)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def sparse_reverse_kl(student_logits, teacher_topk_logps, teacher_topk_indices,
                      mask=None) -> torch.Tensor:
    """KL(student || teacher) —— 脆弱，需 learner 时刻用当前 student 重算。"""
    student_topk = torch.gather(F.log_softmax(student_logits, dim=-1),
                                -1, teacher_topk_indices)
    kl = (student_topk.exp() * (student_topk - teacher_topk_logps)).sum(-1)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def low_var_kl(student_logp, ref_logp, mask=None) -> torch.Tensor:
    """低方差 KL 正则（Lightning 隐式正则 / Direct-OPD KL-to-ref，防止策略漂移）。

    student_logp, ref_logp: (T, V) log-softmax 分布。

    k3 逐点估计量：k3(v) = exp(r) − r − 1，其中 r = logπ_ref(v) − logπ(v)。
    ★ 修复：k3 必须在【采样分布 π_student 下取期望】才等于 KL(π||π_ref)：
        Σ_v π(v)·k3(v) = Σ_v [π_ref(v) − π(v)(logπ_ref(v) − logπ(v)) − π(v)]
                       = Σ_v π(v)(logπ(v) − logπ_ref(v)) = KL(π||π_ref)
    原实现对 (T,V) 等权 mean —— 那不是 KL，只是碰巧非负的量（量级与梯度都错）。
    """
    log_r = ref_logp - student_logp                       # (T, V)
    k3 = log_r.exp() - log_r - 1.0                        # (T, V) 逐点估计量
    kl = (student_logp.exp() * k3).sum(-1)                # E_{a~π}[k3] = KL(π||ref), (T,)
    if mask is not None:
        return (kl * mask).sum() / (mask.sum() + 1e-8)
    return kl.mean()


def policy_gradient_kl(student_logp_cur, student_logp_old, advantage,
                       mask=None, clip_eps: float = 0.2) -> torch.Tensor:
    """Direct-OPD 的 PG 损失 + AsyncOPD「learner 时刻用当前 student 重算」代理。

    正确形式 = 按行为策略 π_old 加权的逐 vocab 重要性采样：

      ratio(v) = exp(logp_cur(v) − logp_old(v))          # (T, V) learner 重算
      loss     = − mean_t Σ_v π_old(v) · min(ratio(v)·Δ(v), clip(ratio(v))·Δ(v))

    为什么必须按 π_old 加权（两个失败形式的对照）：
    ① 若对 (T,V) 等权 mean（demo 最初实现）：目标变成 −(1/V)Σ_v Δ_T(v)，
       只是「均匀平均的 Δ_T」，不是 Direct-OPD 的目标分布，方向碰巧对但权重错。
    ② 若用 token 级标量 adv_t=E_{π_old}[Δ_T] × 动作级 ratio(a_t)（标准 PPO 外形）：
       E_{a~π_old}[ratio(a)]·adv_t = adv_t·Σ_a π_cur(a) = adv_t 对 θ 是【常数】，
       一阶梯度恒为 0（仅剩 clip 边界噪声）——审阅时实测 E[Δ_T] 不升反降。

    本形式在 ratio=1 处精确等于 Direct-OPD 的真实目标：
      −Σ_v π_old(v)·ratio(v)·Δ(v) = −Σ_v π_cur(v)·Δ(v) = −E_{π_cur}[Δ_T]
    梯度 Σ_v Δ(v)·∇π_cur(v) ≠ 0 ✓；clip 则提供 AsyncOPD 的陈旧样本鲁棒截断。

    student_logp_cur / student_logp_old: (T, V) log-softmax 分布（cur 带梯度）
    advantage:                           (T, V) 即 Δ_T = logπ_rl − logπ_ref
    """
    ratio = (student_logp_cur - student_logp_old).exp()          # (T, V)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    pg_pointwise = torch.min(unclipped, clipped)                 # (T, V) 悲观下界
    pg = -(student_logp_old.exp() * pg_pointwise).sum(-1)        # E_{π_old}[·], (T,)
    if mask is not None:
        return (pg * mask).sum() / (mask.sum() + 1e-8)
    return pg.mean()

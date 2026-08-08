"""★ Stage 1 —— Lightning-OPD：离线「教师对」log-prob 缓存（消除常驻教师）。

真实代码：
- Lightning-OPD/data_curation/prepare_lightning_opd.py::phase2_logprobs
      （teacher sglang 服务把每 token 的 response logprob 写回 parquet 的
        metadata["teacher_log_probs"]）
- Lightning-OPD/slime/rollout/on_policy_distillation.py::post_process_rewards
      （训练期 is_lightning_opd/is_offline_opd 为真时直接复用缓存的
        teacher_log_probs，不再调用 teacher）

本 demo 把该思想扩展到 Direct-OPD 所需的「教师对」：
  缓存 post-RL teacher 与 pre-RL reference 在每条 (prompt, response) 上的
  next-token 分布，并预计算 Δ_T = logπ_rl − logπ_ref。

Teacher Consistency（教师一致性，Lightning 的关键前提）：
  SFT 与 OPD 必须用同一 teacher（同 tokenizer / 词表 / 架构），否则引入不可约
  梯度偏差。这里在 build 时 assert 校验。
"""

from __future__ import annotations

import torch


class TeacherConsistencyError(Exception):
    """SFT 与 OPD 的 teacher 不一致时抛出（Lightning 论文证明这会导致梯度偏差）。"""


class OfflineTeacherPairCache:
    def __init__(self, enforce_consistency: bool = True):
        self.enforce = enforce_consistency
        # prompt_id -> {"rl": (T,V), "ref": (T,V), "resp": (T,)}
        self.entries: dict = {}

    @torch.no_grad()
    def build(self, prompts, responses, teacher_rl, teacher_ref, device) -> "OfflineTeacherPairCache":
        if self.enforce:
            # ★ 修复：原先只查 type+vocab 太弱（同架构不同配置查不出）。
            #    真实场景应比对 config.json / tokenizer 哈希；这里补齐全部结构字段。
            same = (
                type(teacher_rl) is type(teacher_ref)
                and teacher_rl.vocab == teacher_ref.vocab
                and getattr(teacher_rl, "d_model", None) == getattr(teacher_ref, "d_model", None)
                and getattr(teacher_rl, "max_len", None) == getattr(teacher_ref, "max_len", None)
            )
            if not same:
                raise TeacherConsistencyError(
                    "teacher_rl 与 teacher_ref 必须共享架构/词表/隐藏维度/上下文长度 "
                    "(teacher consistency: SFT 与 OPD 同一 teacher)"
                )
        for i, (p, r) in enumerate(zip(prompts, responses)):
            rl_D = teacher_rl.response_distributions(p, r, device)   # (T, V)
            ref_D = teacher_ref.response_distributions(p, r, device)  # (T, V)
            self.entries[i] = {
                "rl": rl_D.cpu(),
                "ref": ref_D.cpu(),
                "resp": r.cpu(),
            }
        return self

    # ---- 训练期查询（★无 live teacher，只查离线缓存）----
    def get_dists(self, prompt_id: int):
        e = self.entries[prompt_id]
        return e["rl"], e["ref"]

    def delta(self, prompt_id: int) -> torch.Tensor:
        """Δ_T = logπ_rl − logπ_ref，形状 (T, V)。即 Direct-OPD 的迁移对象。"""
        e = self.entries[prompt_id]
        return e["rl"] - e["ref"]

    def save(self, path: str) -> None:
        torch.save({"entries": self.entries, "enforce": self.enforce}, path)

    @classmethod
    def load(cls, path: str) -> "OfflineTeacherPairCache":
        # weights_only=True：缓存只含 tensor/基本类型，拒绝任意 pickle 反序列化
        ck = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(enforce=ck["enforce"])
        obj.entries = ck["entries"]
        return obj

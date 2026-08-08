"""v2 Lightning 离线教师对缓存。

两种模式（由 `top_k` 决定）：
- **dense 模式**（`top_k <= 0`，demo 默认）：存完整 (N,T,V) 张量，零拷贝索引。
- **top-K 稀疏模式**（`top_k > 0`，GPU 部署默认，见 OPTIMIZATION_PLAN_2xRTXPRO6000.md L4）：
  每个 (n,t) 位置只存 teacher 自己的 top-K 个 (token_id, logp_rl, logp_ref)，
  体积从 (N,T,V) 降到 (N,T,K)，K≪V（真实词表 V=32k~150k 下 ↓约 1000×），
  可直接落盘/ mmap 跨进程共享。训练时按 **student 的 top-K 支撑**把 delta 展开回填到
  dense (B,T,V)（支撑外置 0），因此 `losses.py` 的分布级 PG 内核**零改动**。

Teacher Consistency（Lightning-OPD 关键前提）：SFT 与 OPD 必须同一 teacher，
build 时校验架构/词表/隐藏维度/上下文长度（真实场景应比对 config.json + tokenizer 哈希）。
"""

from __future__ import annotations

import torch

from .model import response_dists


class TeacherConsistencyError(Exception):
    """SFT 与 OPD 的 teacher 不一致时抛出（会导致不可约梯度偏差）。"""


class TensorTeacherCache:
    def __init__(self, enforce_consistency: bool = True, top_k: int = 0):
        """
        top_k:
          <= 0  → dense 模式，存完整 (N,T,V)；
          >  0  → 稀疏 top-K 模式，每位置存 teacher 的 top-K（token_id, logp_rl, logp_ref）。
        """
        self.enforce = enforce_consistency
        self.top_k = top_k
        self.mode = "dense" if top_k <= 0 else "topk"
        self.vocab = 0
        # dense 模式
        self.rl: torch.Tensor | None = None      # (N, T, V)
        self.ref: torch.Tensor | None = None     # (N, T, V)
        self.delta: torch.Tensor | None = None   # (N, T, V) = rl − ref，预计算
        # 稀疏模式
        self.ids: torch.Tensor | None = None      # (N, T, K)  teacher top-K token id
        self.rl_k: torch.Tensor | None = None     # (N, T, K)  teacher_rl logp @ top-K
        self.ref_k: torch.Tensor | None = None    # (N, T, K)  teacher_ref logp @ top-K
        self.delta_k: torch.Tensor | None = None  # (N, T, K)  = rl_k − ref_k，预计算

    # --------------------------- 构建 ---------------------------
    @torch.no_grad()
    def build(self, prompts, responses, teacher_rl, teacher_ref,
              batch_size: int = 16, device=None) -> "TensorTeacherCache":
        if self.enforce:
            ok = (
                type(teacher_rl) is type(teacher_ref)
                and teacher_rl.vocab == teacher_ref.vocab
                and getattr(teacher_rl, "d_model", None) == getattr(teacher_ref, "d_model", None)
                and getattr(teacher_rl, "max_len", None) == getattr(teacher_ref, "max_len", None)
            )
            if not ok:
                raise TeacherConsistencyError(
                    "teacher_rl 与 teacher_ref 必须共享架构/词表/隐藏维度/上下文长度"
                )
        dev = device if device is not None else (
            next(teacher_rl.parameters()).device if list(teacher_rl.parameters())
            else torch.device("cpu"))
        teacher_rl.eval()
        teacher_ref.eval()
        N = prompts.size(0)

        if self.mode == "dense":
            rl_full, ref_full = [], []
            for i in range(0, N, batch_size):
                sl = slice(i, min(i + batch_size, N))
                rl_full.append(response_dists(teacher_rl, prompts[sl], responses[sl]))
                ref_full.append(response_dists(teacher_ref, prompts[sl], responses[sl]))
            rl_full = torch.cat(rl_full)
            ref_full = torch.cat(ref_full)
            self.vocab = rl_full.size(-1)
            self.rl = rl_full
            self.ref = ref_full
            self.delta = rl_full - ref_full
        else:
            # 稀疏 top-K：⚠️ 必须【逐 chunk】取 teacher top-K，绝不把完整 (N,T,V) 稠密
            # 张量 cat 出来再 topk —— 否则 build 阶段就把 L4 要省的内存又花掉了
            # （真实词表 V=128k × 规模 N 下 build 即 OOM）。
            ids_l, rlk_l, refk_l = [], [], []
            self.vocab = 0
            for i in range(0, N, batch_size):
                sl = slice(i, min(i + batch_size, N))
                rl_c = response_dists(teacher_rl, prompts[sl], responses[sl])   # (c,T,V)
                ref_c = response_dists(teacher_ref, prompts[sl], responses[sl])
                self.vocab = rl_c.size(-1)
                Kt = min(self.top_k, self.vocab)
                # teacher 自己的 top-K（Direct-OPD 迁移对象定义在 teacher 高概率支撑上）
                tk = rl_c.topk(Kt, dim=-1)
                ids_l.append(tk.indices)                               # (c,T,Kt)
                rlk_l.append(tk.values)                                # (c,T,Kt)
                refk_l.append(ref_c.gather(-1, tk.indices))            # (c,T,Kt)
            self.ids = torch.cat(ids_l)                                # (N,T,Kt)
            self.rl_k = torch.cat(rlk_l)
            self.ref_k = torch.cat(refk_l)
            self.delta_k = self.rl_k - self.ref_k                      # 预计算一次
        return self

    # --------------------------- 读取 ---------------------------
    def get_delta(self, idxs: torch.Tensor) -> torch.Tensor:
        """dense 模式：(B,) -> (B, T, V)。"""
        if self.mode != "dense":
            raise RuntimeError("sparse 缓存请用 delta_for_student_topk()")
        return self.delta[idxs]

    def topk(self, idxs: torch.Tensor):
        """稀疏模式：(B,) -> (ids (B,T,Kt), delta_k (B,T,Kt))。"""
        return self.ids[idxs], self.delta_k[idxs]

    def delta_for_student_topk(self, idxs: torch.Tensor,
                               student_topk_ids: torch.Tensor | None,
                               fill: float = 0.0) -> torch.Tensor:
        """把离线缓存的 Δ_T 展开成 dense (B,T,V)，仅在 student 的 top-K 支撑上有值，其余 = fill。

        - dense 模式：直接返回完整 (B,T,V) delta（忽略 student_topk_ids）。
        - 稀疏模式：student_topk_ids: (B,T,Ks)。对每个 student top-K token，若在 teacher
          top-K 内有匹配则取 teacher delta，否则 fill（默认 0）。再 scatter 回 (B,T,V)。
          支撑外的 Δ 置 0 = Direct-OPD 的「student 高概率支撑外 teacher 偏移可忽略」近似，
          且保证 `losses.pg_loss` 的 E_{π_old}[·] 加权只作用在 on-policy 支撑上。
        """
        if self.mode == "dense":
            return self.delta[idxs]
        # 稀疏展开。⚠️ idxs 是一维批次索引 (B,)，不是 (B,T)——B/T 要从支撑张量取。
        B = idxs.size(0)
        T = student_topk_ids.size(1)
        Kt = self.ids.size(-1)
        Ks = student_topk_ids.size(-1)
        teacher_ids = self.ids[idxs]                       # (B, T, Kt)
        teacher_delta = self.delta_k[idxs]                 # (B, T, Kt)
        # 匹配：(B,T,Ks,Kt)，student id 是否落在 teacher top-K
        match = (student_topk_ids.unsqueeze(-1) == teacher_ids.unsqueeze(-2)).to(
            teacher_delta.dtype)                           # (B, T, Ks, Kt)
        matched = (match * teacher_delta.unsqueeze(-2)).sum(-1)   # (B, T, Ks)
        out = torch.full((B, T, self.vocab), fill,
                         dtype=matched.dtype, device=matched.device)
        out.scatter_(-1, student_topk_ids, matched)
        return out

    # --------------------------- 持久化 ---------------------------
    def save(self, path: str) -> None:
        if self.mode == "dense":
            torch.save({"mode": "dense", "vocab": self.vocab,
                        "rl": self.rl, "ref": self.ref, "delta": self.delta,
                        "enforce": self.enforce}, path)
        else:
            # 稀疏张量极小，可直接 torch.save；生产环境改用 mmap 跨进程共享（见方案 L4/L6）
            torch.save({"mode": "topk", "vocab": self.vocab, "top_k": self.top_k,
                        "ids": self.ids, "rl_k": self.rl_k,
                        "ref_k": self.ref_k, "delta_k": self.delta_k,
                        "enforce": self.enforce}, path)

    @classmethod
    def load(cls, path: str) -> "TensorTeacherCache":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("mode") == "topk":
            obj = cls(enforce_consistency=ck["enforce"], top_k=ck["top_k"])
            obj.vocab = ck["vocab"]
            obj.ids, obj.rl_k, obj.ref_k, obj.delta_k = (
                ck["ids"], ck["rl_k"], ck["ref_k"], ck["delta_k"])
        else:
            obj = cls(enforce_consistency=ck["enforce"], top_k=0)
            obj.vocab = ck["vocab"]
            obj.rl, obj.ref, obj.delta = ck["rl"], ck["ref"], ck["delta"]
        return obj

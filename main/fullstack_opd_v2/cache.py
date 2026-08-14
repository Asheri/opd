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


@torch.no_grad()
def expand_student_topk_delta(ids_sorted: torch.Tensor,
                              delta_k_sorted: torch.Tensor,
                              student_topk_ids: torch.Tensor,
                              vocab: int,
                              vocab_out: int | None = None,
                              fill: float = 0.0,
                              mask: torch.Tensor | None = None) -> torch.Tensor:
    """把 teacher top-K 支撑展开成 dense (B,T,V)，仅在 student 的 top-K 支撑上有值。

    纯张量逻辑（无状态、无 self），供 in-memory `TensorTeacherCache` 与磁盘
    `DiskTeacherCache` 共用（Stage 1 磁盘 mmap 存储，S1-3）。语义与
    `TensorTeacherCache.delta_for_student_topk` 完全一致：

    - `ids_sorted`/`delta_k_sorted`：(B,T,Kt)，teacher top-K 已按 token id 升序预排序。
    - `student_topk_ids`：(B,T,Ks)，student 的 top-K 支撑。
    - 每个 student top-K token 用 searchsorted 二分定位到 ≤ 它的 teacher 位置，命中
      （教师支撑含该 id）取 teacher delta，未命中置 0；再 scatter 回 (B,T,V)。
    - `vocab_out`：展开维度。默认 max(vocab, student_topk_ids.max()+1)；跨词表
      （student vocab > teacher vocab，如 7B=152064 vs 151936）时扩展对齐 ratio。
    - `mask`（可选，S1-5 变长）：(B,T) 有效 token 掩码，padding 位置 Δ 置 0 → 不参与
      PG/KL 统计。None = 全 valid（兼容旧行为）。
    """
    B = student_topk_ids.size(0)
    T = student_topk_ids.size(1)
    Kt = ids_sorted.size(-1)
    pos = torch.searchsorted(ids_sorted, student_topk_ids.contiguous()).clamp(max=Kt - 1)
    found = ids_sorted.gather(-1, pos) == student_topk_ids
    matched = delta_k_sorted.gather(-1, pos) * found           # 未匹配置 0
    if vocab_out is None:
        vocab_out = max(vocab, int(student_topk_ids.max()) + 1)
    if vocab_out < vocab:
        raise ValueError(
            f"vocab_out={vocab_out} < teacher vocab={vocab}：展开维度不能小于缓存词表")
    out = torch.full((B, T, vocab_out), fill, dtype=matched.dtype, device=matched.device)
    out.scatter_(-1, student_topk_ids, matched)
    if mask is not None:
        out = out * mask.unsqueeze(-1)                         # padding 位置 Δ=0
    return out


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
        # P1-1：按 token id 升序预排序的 teacher top-K 支撑，训练期 torch.searchsorted 二分
        # 匹配（省 O(K²) 全对比较，峰值显存从 (B,T,Ks,Kt) 降到 (B,T,Ks)）。
        self.ids_sorted: torch.Tensor | None = None              # (N,T,Kt) teacher top-K token id 升序
        self.delta_k_sorted: torch.Tensor | None = None          # (N,T,Kt) 按 ids_sorted 对齐的 delta_k

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
            self.delta = rl_full - ref_full
            del rl_full, ref_full          # P1-2：只留 delta，释放 (N,T,V) 两份（GPU 省 2/3 显存）
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
            # P1-1：按 token id 升序预排序，训练期 searchsorted 二分匹配（省 O(K²) 全对比较）
            self.ids_sorted, _order = self.ids.sort(dim=-1)
            self.delta_k_sorted = self.delta_k.gather(-1, _order)
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
                               fill: float = 0.0,
                               vocab_out: int | None = None) -> torch.Tensor:
        """把离线缓存的 Δ_T 展开成 dense (B,T,V)，仅在 student 的 top-K 支撑上有值，其余 = fill。

        - dense 模式：直接返回完整 (B,T,V) delta（忽略 student_topk_ids）。
        - 稀疏模式：student_topk_ids: (B,T,Ks)。对每个 student top-K token，若在 teacher
          top-K 内有匹配则取 teacher delta，否则 fill（默认 0）。再 scatter 回 (B,T,V)。
          支撑外的 Δ 置 0 = Direct-OPD 的「student 高概率支撑外 teacher 偏移可忽略」近似，
          且保证 `losses.pg_loss` 的 E_{π_old}[·] 加权只作用在 on-policy 支撑上。

        vocab_out（方案 A · 对齐论文跨词表）：展开张量的 vocab 维度。默认 None →
        max(student_topk_ids)+1（student 词表）。这允许 **student vocab > teacher vocab**
        （如 7B：student 152064 vs teacher 151936）：student 超出 teacher 词表的 top-K id
        在 searchsorted 未命中 → matched=0，scatter 进扩展维度，与 `ratio`(152064) 对齐。
        """
        if self.mode == "dense":
            return self.delta[idxs]
        # 稀疏展开。⚠️ idxs 是一维批次索引 (B,)，不是 (B,T)——B/T 要从支撑张量取。
        # 核心逻辑抽为模块级纯函数 expand_student_topk_delta（S1-3，in-memory/磁盘共用）：
        # searchsorted 二分匹配替代 O(Ks×Kt) 全对比较，数值与全对比较完全一致。
        teacher_ids_sorted = self.ids_sorted[idxs]             # (B, T, Kt) 已升序
        teacher_delta_sorted = self.delta_k_sorted[idxs]       # (B, T, Kt)
        return expand_student_topk_delta(teacher_ids_sorted, teacher_delta_sorted,
                                         student_topk_ids, self.vocab, vocab_out, fill)

    # --------------------------- 设备迁移 ---------------------------
    def to(self, device) -> "TensorTeacherCache":
        """把缓存全部张量（含可选 fat prompts/responses）搬到 device。

        P1-A（二次审查）：load() 用 map_location="cpu" 把缓存钉在 CPU，load_cache 训练
        分支若不搬设备，GPU 路径在 KL 锚点 / scheduler 索引处必崩设备不匹配。
        """
        for attr in ("delta", "ids", "rl_k", "ref_k", "delta_k",
                     "ids_sorted", "delta_k_sorted", "prompts", "responses"):
            t = getattr(self, attr, None)
            if t is not None:
                setattr(self, attr, t.to(device))
        return self

    # --------------------------- 持久化 ---------------------------
    def save(self, path: str, prompts: torch.Tensor | None = None,
             responses: torch.Tensor | None = None) -> None:
        """落盘缓存（可选携带 fat prompts/responses，供训练 load 后直接索引）。

        多学生并发（GPU_MEMORY_AND_PARALLEL_PLAN §7）：`opd cache` 预建后，
        `opd train --set stage1.load_cache=true` 载入并【跳过 Stage 0/1】。
        训练需要 (fat) prompts/responses 索引上下文与算 KL 锚点，故 save 时把它们
        一并落盘；旧缓存（无 prompts/responses）load 后为 None，训练载入会显式报错。
        """
        payload = {"mode": self.mode, "vocab": self.vocab, "enforce": self.enforce}
        if self.mode == "dense":
            payload["delta"] = self.delta
        else:
            payload.update({"top_k": self.top_k, "ids": self.ids, "rl_k": self.rl_k,
                            "ref_k": self.ref_k, "delta_k": self.delta_k,
                            "ids_sorted": self.ids_sorted,
                            "delta_k_sorted": self.delta_k_sorted})
        if prompts is not None or responses is not None:
            payload["prompts"] = prompts
            payload["responses"] = responses
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "TensorTeacherCache":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("mode") == "topk":
            obj = cls(enforce_consistency=ck["enforce"], top_k=ck["top_k"])
            obj.vocab = ck["vocab"]
            obj.ids, obj.rl_k, obj.ref_k, obj.delta_k = (
                ck["ids"], ck["rl_k"], ck["ref_k"], ck["delta_k"])
            if ck.get("ids_sorted") is None:
                obj.ids_sorted, _o = obj.ids.sort(dim=-1)          # 旧缓存兼容：现场预排序
                obj.delta_k_sorted = obj.delta_k.gather(-1, _o)
            else:
                obj.ids_sorted = ck["ids_sorted"]
                obj.delta_k_sorted = ck["delta_k_sorted"]
        else:
            obj = cls(enforce_consistency=ck["enforce"], top_k=0)
            obj.vocab = ck["vocab"]
            obj.delta = ck["delta"]          # rl/ref 不再落盘（只留 delta）
        # 可选 fat 上下文（多学生 load_cache 路径用；旧缓存可能为 None）
        obj.prompts = ck.get("prompts")
        obj.responses = ck.get("responses")
        return obj

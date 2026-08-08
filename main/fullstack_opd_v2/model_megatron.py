"""L2 · Megatron-core TP=2 + Sequence Parallel 版 learner（CausalToyLM 并行骨架）。

把 toy 的 nn.Linear 序列换成 megatron 的并行层，使 learner 在 2 卡（NVLink）上按
张量并行切分，每层用 all-reduce（TP）/ reduce-scatter + all-gather（SP）通信。

本文件是**骨架**而非生产训练循环：
- 真实启动需在 launcher 中调用一次
      mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size,
                                    sequence_parallel=sp)
  建立 TP 组，之后才能构造本模型（megatron 的并行层依赖 parallel_state）。
- 本模型可被 parallelize_learner_tp2() 构造，替换 toy learner。

与 CausalToyLM 同接口：
- forward(ids: (B,T)) -> logits (B,T,V)，其中 V 维度在每个 TP rank 上按词表分片；
- response_dists(prompts, responses) -> (B,T,V) **完整** log-softmax：rank 间
  all-gather 还原词表并行，于是 Stage2 的分布级 PG + k3 KL 内核**一行不动**。

megatron-core 未安装或 parallel_state 未初始化时，本文件可安全导入（类可定义），
但实例化会抛清晰 RuntimeError（与 L3/L5 的降级策略一致）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- megatron-core 可选导入：缺失不影响本文件被 import ----
try:                                                # pragma: no cover
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
    from megatron.core.tensor_parallel.rmsnorm import RMSNorm
    from megatron.core.tensor_parallel.mappings import (
        gather_from_sequence_parallel_region,
        scatter_to_sequence_parallel_region,
        gather_from_tensor_model_parallel_region,
    )
    _MEGATRON_AVAILABLE = True
except Exception:                                   # pragma: no cover
    mpu = None
    ColumnParallelLinear = RowParallelLinear = VocabParallelEmbedding = None
    RMSNorm = None
    gather_from_sequence_parallel_region = None
    scatter_to_sequence_parallel_region = None
    gather_from_tensor_model_parallel_region = None
    _MEGATRON_AVAILABLE = False


def megatron_model_available() -> bool:
    return _MEGATRON_AVAILABLE


class _MegatronAttention(nn.Module):
    """TP 注意力的局部头实现：每 rank 负责 n_head/tp 个头。

    qkv 用 ColumnParallelLinear(gather_output=False) 把隐藏维分片，注意力在
    复制后的完整序列上计算（ColumnParallel 内部已 all-gather 序列），输出再经
    RowParallelLinear(input_is_parallel=True, sp) 做 reduce-scatter 回序列分片。
    """

    def __init__(self, d_model, n_head, sp, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.sp = sp
        # QKV：列并行，输出按隐藏维（头维）分片
        self.qkv = ColumnParallelLinear(
            d_model, 3 * d_model, bias=False,
            gather_output=False, sequence_parallel=sp)
        # 输出投影：行并行，输入已是隐藏分片，SP 下 reduce-scatter 出序列分片
        self.o_proj = RowParallelLinear(
            d_model, d_model, bias=False,
            input_is_parallel=True, sequence_parallel=sp)
        self.dropout = dropout
        self._mask_cache = {}

    def _causal_mask(self, T, device):
        m = self._mask_cache.get(T)
        if m is None:
            m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                           diagonal=1)
            self._mask_cache[T] = m
        return m

    def forward(self, x):
        # x: (B, T, d/tp) 隐藏维分片；序列在 SP 下分片、非 SP 下复制
        #    （ColumnParallel 内部已把序列 all-gather 成完整 T）
        B, T, _ = x.shape
        qkv = self.qkv(x)                          # (B, T, 3*d/tp)
        q, k, v = qkv.chunk(3, dim=-1)
        # 本 rank 负责的局部头数（从特征维反推，避免依赖 tp_size）
        n_local = q.size(-1) // self.head_dim
        q = q.view(B, T, n_local, self.head_dim).transpose(1, 2)   # (B,h,T,hd)
        k = k.view(B, T, n_local, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_local, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = self._causal_mask(T, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, self.dropout)
        ctx = torch.matmul(attn, v)                # (B, h, T, hd)
        ctx = ctx.transpose(1, 2).reshape(B, T, n_local * self.head_dim)
        out = self.o_proj(ctx)                     # (B, T/tp, d) 序列分片（SP）
        return out


class _MegatronBlock(nn.Module):
    """单 Transformer 块：RMSNorm + 注意力 + RMSNorm + MLP，全程 TP/SP 友好。

    不手动做序列 gather/scatter —— 序列分片由 RMSNorm(sp=True) 与
    Column/RowParallelLinear(sp=True) 原生管理：输入是序列分片，RMSNorm 在分片
    上做统计量 all-reduce，ColumnParallel 内部 all-gather 序列做矩阵乘，
    RowParallel 内部 reduce-scatter 回序列分片。sp=False 时退化为纯 TP（序列复制）。
    """

    def __init__(self, d_model, n_head, d_ff, sp, dropout=0.0):
        super().__init__()
        self.sp = sp
        self.norm1 = RMSNorm(d_model, eps=1e-5, sequence_parallel=sp)
        self.attn = _MegatronAttention(d_model, n_head, sp, dropout)
        self.norm2 = RMSNorm(d_model, eps=1e-5, sequence_parallel=sp)
        self.fc1 = ColumnParallelLinear(
            d_model, d_ff, bias=False,
            gather_output=False, sequence_parallel=sp)
        self.fc2 = RowParallelLinear(
            d_ff, d_model, bias=False,
            input_is_parallel=True, sequence_parallel=sp)

    def forward(self, x):
        # x: (B, T/tp, d) 序列分片（SP）或 (B, T, d) 复制（非 SP）
        residual = x
        x = self.norm1(x)
        a = self.attn(x)                           # (B, T/tp, d) 序列分片
        x = residual + a
        residual = x
        x = self.norm2(x)
        h = self.fc1(x)                            # (B, T, d_ff/tp)
        h = F.gelu(h)
        h = self.fc2(h)                            # (B, T/tp, d) 序列分片
        return residual + h


class MegatronCausalToyLM(nn.Module):
    """Megatron-core TP=2 + SP 版 CausalToyLM（L2 learner 骨架）。

    与 CausalToyLM 同接口：forward(ids)->logits，response_dists()->(B,T,V) log-softmax。
    词表在 TP rank 间分片（VocabParallelEmbedding + 输出头 ColumnParallel
    gather_output=False），response_dists 用 gather_from_tensor_model_parallel_region
    还原完整词表，使下游分布级损失内核无需改动。
    """

    def __init__(self, vocab, d_model, n_layers, n_head=4, max_len=64,
                 sp=True, dropout=0.0):
        super().__init__()
        if not _MEGATRON_AVAILABLE:
            raise RuntimeError(
                "megatron-core 未安装（统一 GPU 环境应含 megatron-core 0.16.1）。"
                "L2 需要把 learner 用 megatron 的 parallel 层重建；"
                "ToyModel 的 nn.Linear 不能被 parallelize_learner_tp2 切分。")
        if mpu is None or not mpu.model_parallel_is_initialized():
            raise RuntimeError(
                "L2 集成点：构造 MegatronCausalToyLM 前必须在 launcher 调用一次 "
                "mpu.initialize_model_parallel(tensor_model_parallel_size=tp_size, "
                "sequence_parallel=%s) 建立 TP 组，再实例化本模型。" % sp)

        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        self.n_head = n_head
        self.sp = sp
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()

        # 词表并行嵌入：每 rank 持 V/tp 行，forward 内部 reduce 出复制隐藏态
        self.emb = VocabParallelEmbedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))   # 复制
        self.blocks = nn.ModuleList([
            _MegatronBlock(d_model, n_head, 4 * d_model, sp, dropout)
            for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model, eps=1e-5, sequence_parallel=sp)
        # 输出头：词表并行，gather_output=False → logits 按 V 分片
        self.head = ColumnParallelLinear(
            d_model, vocab, bias=False,
            gather_output=False, sequence_parallel=sp)
        self._mask_cache = {}

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L) -> logits (B, L, V/tp)（词表分片）
        L = ids.size(1)
        x = self.emb(ids) + self.pos[:, :L]        # (B, L, d) 复制（emb 内部 reduce）
        if self.sp:
            # 进入 SP：把复制序列切到本 rank 的序列分片
            x = scatter_to_sequence_parallel_region(x)   # (B, L/tp, d)
        for blk in self.blocks:
            x = blk(x)                                   # (B, L/tp, d) 序列分片
        # ⚠️ 不要在此显式 gather 序列：SP 下 final_norm 必须在【分片】序列上做统计
        #    （RMSNorm sp=True 会跨 rank all-reduce 统计量）；head 是 ColumnParallel(sp=True)，
        #    其内部会对序列分片做 all-gather 再做矩阵乘。若此处先 gather 成复制序列，
        #    final_norm 会把统计量 ×tp 双计、head 又会把复制序列再 all-gather 成 L*tp 爆炸。
        x = self.final_norm(x)                     # sp: (B,L/tp,d)；非 sp: (B,L,d)
        logits = self.head(x)                      # (B, L, V/tp) 词表分片（sp 内部 all-gather 序列）
        return logits

    def response_dists(self, prompts: torch.Tensor,
                       responses: torch.Tensor) -> torch.Tensor:
        """与 CausalToyLM.response_dists 同语义：(B,P),(B,T) -> (B,T,V) log-softmax。

        forward 产出词表分片的 logits，这里 all-gather 回完整词表再 log_softmax，
        使 Stage2 的分布级 PG + k3 KL 内核无需任何改动。
        """
        P = prompts.size(1)
        T = responses.size(1)
        full = torch.cat([prompts, responses], dim=1)          # (B, P+T)
        logits = self.forward(full)                            # (B, P+T, V/tp)
        logits_full = gather_from_tensor_model_parallel_region(logits)  # (B, P+T, V)
        logp = F.log_softmax(logits_full, dim=-1)
        return logp[:, P - 1:P - 1 + T]                        # (B, T, V)

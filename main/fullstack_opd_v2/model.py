"""v2 模型层：批次优先的因果 LM（设备常驻）。

相对 v1 的底层重构：
- 所有接口原生 (B, ...) 批次形状：一次前向覆盖整个 batch
  （v1 每个样本都 unsqueeze(0) 单独前向，是最大瓶颈）。
- 因果 mask 按 (长度, 设备) 缓存复用，不再每步重建。
- generate_batch：整批同步自回归解码（v1 逐样本逐 token 串行）。
- 方法内不切换 eval()/train()、不加 no_grad —— 由调用方管理，避免模式抖动。

保留 v1 审阅修复的正确性内核：因果注意力（不偷看未来 token）、dropout=0。
真实规模下此类由 vLLM / HF ActorRollout 替代（见 requirements-unified.txt）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalToyLM(nn.Module):
    def __init__(self, vocab: int = 64, d_model: int = 48, n_layers: int = 2,
                 max_len: int = 64, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len
        self.n_head = n_head
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, batch_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)
        self._mask_cache: dict = {}

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        key = (L, str(device))
        m = self._mask_cache.get(key)
        if m is None:
            m = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._mask_cache[key] = m
        return m

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L) -> logits (B, L, V)
        L = ids.size(1)
        x = self.emb(ids) + self.pos[:, :L]
        h = self.enc(x, mask=self._causal_mask(L, ids.device))
        return self.head(h)

    def response_dists(self, prompts: torch.Tensor,
                       responses: torch.Tensor) -> torch.Tensor:
        """(B,P),(B,T) -> (B,T,V) log-softmax（同模块级 response_dists 语义）。

        抽出为方法，使 scheduler._train_step 可统一用 self.student.response_dists(...)
        调用 toy 与 Megatron 两种 learner（Megatron 版在内部 all-gather 词表分片）。
        """
        return response_dists(self, prompts, responses)


def response_dists(model: CausalToyLM, prompts: torch.Tensor,
                   responses: torch.Tensor) -> torch.Tensor:
    """(B,P),(B,T) -> (B,T,V) log-softmax。

    第 t 行 = 预测 response 第 t 个 token 的 next-token 分布（上下文 = prompt + a_{<t}），
    即 OPD 里的 π(·|s_t)。一次批量前向完成。
    """
    P = prompts.size(1)
    T = responses.size(1)
    full = torch.cat([prompts, responses], dim=1)          # (B, P+T)
    logp = F.log_softmax(model(full), dim=-1)              # (B, P+T, V)
    return logp[:, P - 1:P - 1 + T]                        # (B, T, V)


def token_logprobs(model: CausalToyLM, prompts: torch.Tensor,
                   responses: torch.Tensor) -> torch.Tensor:
    """(B,P),(B,T) -> (B,T)：实际 response token 的 logπ(a_t|s_t)（保留梯度）。"""
    d = response_dists(model, prompts, responses)
    return d.gather(2, responses.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_batch(model: CausalToyLM, prompts: torch.Tensor, max_new: int = 8,
                   temperature: float = 1.0) -> torch.Tensor:
    """整批自回归解码：(B,P) -> (B,max_new)。每步仅一次批量前向。

    注：未实现 KV cache（nn.TransformerEncoder 不便做增量解码）；
    demo 规模下批量摊销已足够，真实规模由 vLLM 接管推理。
    """
    was_training = model.training
    model.eval()
    ids = prompts
    for _ in range(max_new):
        logits = model(ids)[:, -1]                         # (B, V)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
    if was_training:
        model.train()
    return ids[:, prompts.size(1):]


def detect_loop(response: torch.Tensor, periods=(2, 3, 4),
                min_len: int = 8) -> bool:
    """Stage 2：尾部周期重复检测（token 级）。

    对尾部（含完整生成的）序列做周期自相关：对每个 p∈periods，若有效长度 ≥2p 且
    ≥min_len，且末尾两段完全重复（response[-p:]==response[-2p:-p]）→ 判 loop。
    min_len 防短序列误报。response: (T,) long（应传有效长度，不含 pad）。
    """
    L = response.size(0)
    for p in periods:
        if L >= 2 * p and L >= min_len:
            if bool(torch.equal(response[-p:], response[-2 * p:-p])):
                return True
    return False


def apply_repetition_penalty(logits: torch.Tensor, past: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """对已生成 token 施加 repetition penalty（logits 除以 penalty，>1 抑制重复）。

    logits: (B, V)；past: (B, T) 已生成 token（调用方保证不含 pad）。对每行中出现在
    past 的 token：logits[b, tok] /= penalty（标准 HF 语义）。penalty<=1.0 直接返回原
    张量（默认禁用，零回归）。返回新张量，不改入参。
    """
    if penalty is None or penalty <= 1.0:
        return logits
    out = logits.clone()
    for b in range(past.size(0)):
        seen = torch.unique(past[b])
        if seen.numel():
            out[b].index_put_((seen,), out[b][seen] / penalty)
    return out


@torch.no_grad()
def generate_with_status(model: CausalToyLM, prompts: torch.Tensor, max_new: int,
                         eos_token_id=None, temperature: float = 1.0, pad_id: int = 0,
                         loop_detection: bool = True,
                         loop_periods=(2, 3, 4),
                         repetition_penalty: float = 1.0,
                         loop_min_len: int = 8) -> dict:
    """Stage 2：短预算 rollout——逐 token 采样 + EOS 提前停 + 预算截断。

    EOS 语义：把 eos_token_id 当普通可采样 token，采到即停（不要求自然 EOS）；
    默认 eos_token_id=None → 永不判 EOS，全部 BUDGET_STOP（除非 loop）。已停样本
    本步起不再前向（alive 掩码），其已生成 token 保留，其余位置右 pad 到 max_new。

    返回同构 dict（与 vLLM 端对齐）：
      responses: (B, max_new) long（pad_id 填充）
      statuses:  list[str] ∈ {eos, budget_stop, loop, invalid}
      lengths:   list[int]（有效 token 数，eos 样本含结尾 eos）
      eos_pos:   list[int|None]（None=非 eos）
      looped:    list[bool]
    """
    B, P = prompts.size(0), prompts.size(1)
    device = prompts.device
    responses = torch.full((B, max_new), pad_id, dtype=torch.long, device=device)
    eos_pos: list[int | None] = [None] * B
    alive = torch.ones(B, dtype=torch.bool, device=device)
    was_training = model.training
    model.eval()
    for t in range(max_new):
        if not alive.any():
            break
        idx = alive.nonzero(as_tuple=False).squeeze(-1)   # (n_a,) alive 样本同长
        # 上下文 = prompt + 已生成前 t 个 token（alive 样本恒同长 P+t）
        ctx_a = torch.cat([prompts[idx], responses[idx, :t]], dim=1)
        logits = model(ctx_a)[:, -1]                      # (n_a, V)
        if repetition_penalty > 1.0:
            logits = apply_repetition_penalty(
                logits, responses[idx, :t], repetition_penalty)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        tok = torch.multinomial(probs, 1).squeeze(-1)     # (n_a,)
        responses[idx, t] = tok                          # 写回当前步 token
        if eos_token_id is not None:
            hit = (tok == eos_token_id)
            for j, i in enumerate(idx.tolist()):
                if bool(hit[j]):
                    eos_pos[i] = t
                    alive[i] = False
    if was_training:
        model.train()
    # 有效长度：eos 样本=eos_pos+1（含 eos）；其余（budget_stop/loop）跑满预算
    lengths = [max_new] * B
    for i in range(B):
        if eos_pos[i] is not None:
            lengths[i] = eos_pos[i] + 1
    # 状态判定（优先级：loop > invalid > eos > budget_stop）
    statuses: list[str] = []
    looped: list[bool] = []
    for i in range(B):
        eff = responses[i, :max(1, lengths[i])]          # 有效长度（不含 pad）
        loop = loop_detection and detect_loop(eff, loop_periods, min_len=loop_min_len)
        if loop:
            statuses.append("loop"); looped.append(True)
        elif lengths[i] == 0:
            statuses.append("invalid"); looped.append(False)
        elif eos_pos[i] is not None:
            statuses.append("eos"); looped.append(False)
        else:
            statuses.append("budget_stop"); looped.append(False)
    return {"responses": responses, "statuses": statuses, "lengths": lengths,
            "eos_pos": eos_pos, "looped": looped}


def build_length_mask(responses: torch.Tensor, lengths, eos_pos,
                      device=None) -> torch.Tensor:
    """Stage 2：长度式掩码（替换 _build_mask 的 pad 扫描）。

    (B,T) responses + (B,) lengths + (B,) eos_pos → (B,T) long mask。
    mask[t]=1 当 t<length（eos 样本含 eos；budget_stop 样本全有效）；padding 及
    budget-stop 之后的 token 全 0。用 arange 构造，不扫描 pad 值，杜绝 token-0 误伤。
    """
    T = responses.size(1)
    lens = torch.tensor(lengths, dtype=torch.long,
                        device=device or responses.device)
    return (torch.arange(T, device=lens.device)[None, :] < lens[:, None]).long()

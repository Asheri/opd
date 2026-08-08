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

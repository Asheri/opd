"""可运行的极小 transformer「替身」模型。

真实场景里这里应替换为三个 clone 下来的 repo 的 LogProbModel / ActorRollout：
- async-opd/opd/rollout/base.py::BaseRolloutWorker
- Direct-OPD/verl/verl/workers/actor/dp_actor.py::DataParallelPPOActor
- Lightning-OPD/slime/backends/.../model.py

本文件只为了让 demo 在仅依赖 torch 的 CPU 环境下也能跑通，证明全栈叠加逻辑正确。
对外暴露两类能力，分别对应三篇论文的需求：
  * response_distributions(prompt, response) -> (T, V) log-softmax 分布
        = 对每个响应 token，给出「预测它的 next-token 分布」。
        用作 teacher 的离线缓存、student 的 on-policy logp、KL 正则。
  * generate(prompt) -> response
        rollout 生成（小模型 RL、以及 AsyncOPD 的 RolloutCollector）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyModel(nn.Module):
    def __init__(self, vocab: int = 64, d_model: int = 48, n_layers: int = 2,
                 max_len: int = 64, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        # dropout 默认 0.0：demo 需要确定性（TransformerEncoderLayer 默认 0.1 会引入噪声）
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, 4 * d_model, batch_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    @staticmethod
    def _causal_mask(L: int, device) -> torch.Tensor:
        """因果掩码：位置 i 只能 attend 到 <= i 的位置（True = 禁止 attend）。

        ★ 修复：没有它，TransformerEncoder 是双向注意力，「预测第 t 个 token 的分布」
        会偷看到包括答案在内的未来 token，导致 π(a_t|s_t) 语义错误，且与
        generate() 的自回归模式不一致。真实 LM（Qwen/Llama 等）都是因果 LM。
        """
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: (B, L) -> logits (B, L, V)
        L = ids.size(1)
        x = self.emb(ids) + self.pos[:, :L]
        h = self.enc(x, mask=self._causal_mask(L, ids.device))
        return self.head(h)

    def response_distributions(self, prompt_ids: torch.Tensor,
                               response_ids: torch.Tensor,
                               device) -> torch.Tensor:
        """返回 (T, V) 的 log-softmax 分布。

        第 t 行 = 预测「响应第 t 个 token」的 next-token 分布
        （上下文 = prompt + 之前的响应 token）。这正是 OPD 里策略 π(a|s) 的形式。
        """
        self.eval()
        P = prompt_ids.size(0)
        T = response_ids.size(0)
        assert P >= 1, "demo 要求非空 prompt（P==0 时第一个 response token 无前文可预测）"
        full = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0).to(device)
        logits = self(full)                          # (1, L, V)
        logp = F.log_softmax(logits, dim=-1)         # (1, L, V)
        # 响应第 t 个 token 由位置 (P-1 + t) 的 logit 预测
        pred_positions = torch.arange(P, P + T, device=device) - 1
        D = logp[0, pred_positions]                  # (T, V)
        return D

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 8,
                 temperature: float = 1.0, device="cpu") -> torch.Tensor:
        """贪心/采样 rollout，返回 response token 序列。"""
        self.eval()
        ids = prompt_ids.clone().to(device)
        for _ in range(max_new):
            logits = self(ids.unsqueeze(0).to(device))[:, -1]
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(0)
            ids = torch.cat([ids, nxt])
        return ids[prompt_ids.size(0):]


def response_distributions(model: nn.Module, prompt_ids: torch.Tensor,
                           response_ids: torch.Tensor, device) -> torch.Tensor:
    return model.response_distributions(prompt_ids, response_ids, device)


def topk_of(dists: torch.Tensor, K: int):
    """dists: (T, V) log-softmax -> (values (T,K), indices (T,K))"""
    return torch.topk(dists, K, dim=-1)

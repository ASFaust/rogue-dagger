"""Set-transformer scorer: one scalar score per run / suggestion token.

No positional encoding: the order of runs and suggestions carries no meaning.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import feature_dim


class Scorer(nn.Module):
    def __init__(self, d_max: int = 10, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_max = d_max
        self.embed = nn.Sequential(nn.Linear(feature_dim(d_max), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tok: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """tok [E, S, F], pad [E, S] -> scores [E, S - 1] (global token dropped), fp32."""
        h = self.trunk(self.embed(tok), src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 1:])).squeeze(-1).float()


def pairwise_loss(scores: torch.Tensor, values: torch.Tensor, valid: torch.Tensor, tol: float = 1e-6):
    """Pairwise BCE on p(i beats j) = sigmoid(s_i - s_j), target = [v_i < v_j].

    scores, values, valid: [E, M]. Pairs with |v_i - v_j| <= tol carry no label.
    Returns (loss averaged per env then over envs, pair accuracy, mean #pairs).
    """
    dv = values[:, :, None] - values[:, None, :]
    m = valid[:, :, None] & valid[:, None, :] & (dv.abs() > tol)
    logit = scores[:, :, None] - scores[:, None, :]
    target = (dv < 0).float()
    bce = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    npair = m.sum((1, 2))
    has = npair > 0
    per_env = (bce * m).sum((1, 2)) / npair.clamp(min=1)
    loss = per_env[has].mean() if has.any() else scores.sum() * 0
    acc = (((logit > 0).float() == target).float() * m).sum() / m.sum().clamp(min=1)
    return loss, acc, npair.float().mean()


def sample_action(scores: torch.Tensor, valid: torch.Tensor, temperature: float = 1.0, greedy: bool = False,
                  generator: torch.Generator | None = None):
    """Multinomial over softmax(scores / T) restricted to valid -> (action [E], entropy [E])."""
    logits = (scores / max(temperature, 1e-8)).masked_fill(~valid, float("-inf"))
    p = torch.softmax(logits, -1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
    if greedy:
        return logits.argmax(-1), ent
    return torch.multinomial(p, 1, generator=generator)[:, 0], ent

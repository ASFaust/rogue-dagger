"""Set-transformer scorer: one scalar score per run / suggestion token.

No positional encoding: the order of runs and suggestions carries no meaning.

Two readouts:
  token  the global token (budget, horizon, incumbent, ...) goes through the trunk with the
         run / suggestion tokens; a linear-ish head maps each final embedding to a score.
  mlp    no global token in the trunk: the trunk sees only runs and suggestions, and the
         score is MLP([final embedding, global features]). The trunk output does not depend
         on the horizon, so one trunk pass can be read out at several horizons.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import N_GLOBAL, OFF_G, feature_dim


def _fp32(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a small output head in fp32 even under bf16 autocast: bf16 scores are quantised to
    ~1/128 relative (0.06 at |s| = 8), which a low policy temperature turns into log-prob noise."""
    with torch.autocast(x.device.type, enabled=False):
        return head(x.float())


class Scorer(nn.Module):
    def __init__(self, d_max: int = 10, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.0, readout: str = "token"):
        super().__init__()
        assert readout in ("token", "mlp")
        self.d_max, self.readout = d_max, readout
        self.embed = nn.Sequential(nn.Linear(feature_dim(d_max), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        d_in = d_model + (N_GLOBAL if readout == "mlp" else 0)
        self.head = nn.Sequential(nn.Linear(d_in, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tok: torch.Tensor, pad: torch.Tensor, glob: torch.Tensor | None = None) -> torch.Tensor:
        """tok [E, S, F], pad [E, S] -> scores [E, S - 1] (global token dropped), fp32.

        readout "mlp" only: glob [E, K, N_GLOBAL] reads the same trunk output out at K global
        states (e.g. K horizons) -> scores [E, K, S - 1]. Default: the global token's features.
        """
        if self.readout == "token":
            h = self.trunk(self.embed(tok), src_key_padding_mask=pad)
            return _fp32(self.head, self.norm(h[:, 1:])).squeeze(-1)
        h = self.norm(self.trunk(self.embed(tok[:, 1:]), src_key_padding_mask=pad[:, 1:]))   # [E, M, d]
        single = glob is None
        if single:
            glob = tok[:, :1, OFF_G:OFF_G + N_GLOBAL]
        E, K, M = h.shape[0], glob.shape[1], h.shape[1]
        x = torch.cat([h[:, None].expand(E, K, M, h.shape[-1]),
                       glob[:, :, None].to(h.dtype).expand(E, K, M, glob.shape[-1])], -1)
        s = _fp32(self.head, x).squeeze(-1)
        return s[:, 0] if single else s


def pairwise_loss(scores: torch.Tensor, values: torch.Tensor, valid: torch.Tensor, tol: float = 1e-6,
                  weight: torch.Tensor | None = None):
    """Pairwise BCE on p(i beats j) = sigmoid(s_i - s_j), target = [v_i < v_j].

    scores, values, valid: [E, M]. Pairs with |v_i - v_j| <= tol carry no label.
    weight: optional [E, M, M] pair weights (per-env loss is the weighted mean).
    Returns (loss averaged per env then over envs, pair accuracy, mean #pairs).
    """
    dv = values[:, :, None] - values[:, None, :]
    m = valid[:, :, None] & valid[:, None, :] & (dv.abs() > tol)
    logit = scores[:, :, None] - scores[:, None, :]
    target = (dv < 0).float()
    bce = nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    npair = m.sum((1, 2))
    has = npair > 0
    w = m.float() if weight is None else m.float() * weight
    per_env = (bce * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1e-8)
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


class Critic(nn.Module):
    """State value V(s) for RL: the scorer's trunk (global token in the trunk, as in readout
    "token") with a value head on the global token's final embedding.

    Same module names as Scorer, so a pairwise-trained "token" checkpoint initialises embed,
    trunk and norm (see load_trunk); only the value head starts from scratch.
    """

    def __init__(self, d_max: int = 10, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.0, **_):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(feature_dim(d_max), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, tok: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """tok [E, S, F], pad [E, S] -> V [E], fp32."""
        h = self.trunk(self.embed(tok), src_key_padding_mask=pad)
        return _fp32(self.value_head, self.norm(h[:, 0])).squeeze(-1)

    def load_trunk(self, scorer_state: dict) -> int:
        """Copy embed / trunk / norm from a Scorer state dict. -> number of tensors copied."""
        own = self.state_dict()
        sd = {k: v for k, v in scorer_state.items() if k.split(".")[0] in ("embed", "trunk", "norm")
              and k in own and own[k].shape == v.shape}
        self.load_state_dict(sd, strict=False)
        return len(sd)


class QHead(nn.Module):
    """Per-token Q(s, a) MLP over [frozen token embedding | global features] (automl-exam's QHead)."""

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """e [B, M, d], g [B, N_GLOBAL] -> Q [B, M] (fp32; no masking here)."""
        x = torch.cat([e.float(), g.float()[:, None].expand(-1, e.shape[1], -1)], -1)
        return _fp32(self.mlp, x).squeeze(-1)


class SoftQ(nn.Module):
    """Implicit-actor soft Q policy on a frozen scorer trunk: pi = softmax(beta * min(Q1, Q2)).

    embed / trunk / norm have the Scorer's names (load_trunk copies them from a pairwise checkpoint);
    features() is the frozen part, h1 / h2 the twin Q heads, log_beta the learned inverse temperature.
    forward() returns beta * min(Q1, Q2), so it plugs into policy_step / evaluate.py at temperature 1.
    """

    def __init__(self, d_max: int = 10, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.0, head_hidden: int = 64, **_):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(feature_dim(d_max), d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.h1 = QHead(d_model + N_GLOBAL, head_hidden)
        self.h2 = QHead(d_model + N_GLOBAL, head_hidden)
        self.log_beta = nn.Parameter(torch.zeros(()), requires_grad=False)   # set by the SAC trainer

    load_trunk = Critic.load_trunk

    def features(self, tok: torch.Tensor, pad: torch.Tensor):
        """-> per-token embeddings [E, S - 1, d] (global token dropped), global features [E, N_GLOBAL]."""
        h = self.trunk(self.embed(tok), src_key_padding_mask=pad)
        return self.norm(h[:, 1:]), tok[:, 0, OFF_G:OFF_G + N_GLOBAL]

    def forward(self, tok: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        e, g = self.features(tok, pad)
        return torch.minimum(self.h1(e, g), self.h2(e, g)) * self.log_beta.exp()


def load_policy(ckpt: dict, device) -> nn.Module:
    """Policy network from a train.py / rl.py (Scorer) or sac.py (SoftQ) checkpoint, in eval mode."""
    cls = SoftQ if ckpt.get("kind") == "softq" else Scorer
    kw = dict(ckpt["cfg"]["model"])
    if cls is SoftQ:
        kw["head_hidden"] = ckpt["cfg"]["sac"]["head_hidden"]
    m = cls(d_max=10, **kw).to(device).eval()
    m.load_state_dict(ckpt["model"])
    return m

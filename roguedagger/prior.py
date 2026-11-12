"""The ifBO / FT-PFN learning-curve prior, batched over tasks on the GPU.

A port of ``ifbo/priors/ftpfn_prior.py`` (DatasetPrior + MyRNG + comb) that
keeps one independent task per batch row. A task is a freshly initialised
Bayesian neural net (random architecture, random weights) that maps a config in
[0, 1]^d to 23 latent outputs; each output is pushed through the empirical CDF
in ``output_sorted.npy`` to a uniform and then through an inverse CDF to one
curve parameter. The curve is a weighted sum of four basis curves (pow4, exp,
log, hill) with a possible breaking point, plus Gaussian noise.

Differences from the reference implementation, all deliberate:
  * every task row can have its own hp dimension d (inputs are zero-padded
    after the standard normalisation, so padded dims never reach the BNN),
    its own depth and width (padded units have zero outgoing weights);
  * the basis curves are evaluated in float64 on the device, so the
    finite-difference tail gradients (EPS = 1e-9) match numpy;
  * every gamma draw in the reference uses shape a = 1, so gamma.ppf is the
    exponential inverse CDF -log(1 - u).

Curves are returned as *losses*: 1 - performance, in [0, 1].
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

N_OUT = 23
MAX_HIDDEN_LAYERS = 13  # num_layers ~ randint(8, 16) -> 6..13 hidden->hidden layers
MAX_WIDTH = 149          # num_hidden ~ randint(36, 150)
_SORTED_PATH = Path(__file__).with_name("output_sorted.npy")


class PriorTasks:
    """``n_tasks`` independent ifBO tasks with hp dims up to ``d_max``."""

    def __init__(self, n_tasks: int, d_max: int, device: torch.device | str, generator: torch.Generator | None = None):
        import numpy as np

        self.n = n_tasks
        self.d_max = d_max
        self.device = torch.device(device)
        self.gen = generator
        self.sorted = torch.from_numpy(np.load(_SORTED_PATH)).float().to(self.device)
        E, D, H, K = n_tasks, d_max, MAX_WIDTH, MAX_HIDDEN_LAYERS
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.W0, self.b0 = z(E, D, H), z(E, H)
        self.Wh, self.bh = z(E, K, H, H), z(E, K, H)
        self.Wo, self.bo = z(E, H, N_OUT), z(E, N_OUT)
        self.n_hidden = torch.zeros(E, dtype=torch.long, device=self.device)
        self.pre_std, self.out_std = z(E), z(E)
        self.y0, self.ymax = z(E), z(E)
        self.d = torch.ones(E, dtype=torch.long, device=self.device)

    # ---- random helpers that respect the optional generator -------------------
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self.gen)

    def _randint(self, lo, hi, n):  # [lo, hi)
        return torch.randint(lo, hi, (n,), device=self.device, generator=self.gen)

    # ---- task sampling ---------------------------------------------------------
    def resample(self, idx: torch.Tensor, d: torch.Tensor) -> None:
        """Draw fresh tasks for rows ``idx`` with hp dims ``d`` (same length)."""
        n = idx.numel()
        if n == 0:
            return
        H, K, D = MAX_WIDTH, MAX_HIDDEN_LAYERS, self.d_max
        num_layers = self._randint(8, 16, n)
        width = self._randint(36, 150, n)
        init_std = 0.089 + (0.193 - 0.089) * self._rand(n)
        sparse = 0.145
        unit = (torch.arange(H, device=self.device)[None] < width[:, None]).float()  # [n, H]
        inp = (torch.arange(D, device=self.device)[None] < d[:, None]).float()        # [n, D]
        s = init_std[:, None]

        self.W0[idx] = self._randn(n, D, H) * s[:, :, None] * inp[:, :, None] * unit[:, None, :]
        self.b0[idx] = self._randn(n, H) * s * unit
        keep = (self._rand(n, K, H, H) < (1 - sparse)).float()
        self.Wh[idx] = (self._randn(n, K, H, H) * s[:, :, None, None] / math.sqrt(1 - sparse) * keep
                        * unit[:, None, :, None] * unit[:, None, None, :])
        self.bh[idx] = self._randn(n, K, H) * s[:, :, None] * unit[:, None, :]
        self.Wo[idx] = self._randn(n, H, N_OUT) * s[:, :, None] * unit[:, :, None]
        self.bo[idx] = self._randn(n, N_OUT) * s
        self.n_hidden[idx] = num_layers - 2
        self.pre_std[idx] = 0.0003 + (0.0014 - 0.0003) * self._rand(n)
        self.out_std[idx] = 0.0004 + (0.0013 - 0.0004) * self._rand(n)
        u1, u2 = self._rand(n), self._rand(n)
        self.y0[idx] = torch.minimum(u1, u2)
        self.ymax[idx] = torch.where(self._rand(n) < 0.25, torch.maximum(u1, u2), torch.ones_like(u1))
        self.d[idx] = d

    # ---- BNN forward -------------------------------------------------------------
    @torch.no_grad()
    def bnn(self, x: torch.Tensor) -> torch.Tensor:
        """x: [E, M, d_max] configs in [0, 1] -> [E, M, 23] BNN outputs."""
        inp = (torch.arange(self.d_max, device=self.device)[None] < self.d[:, None]).float()
        h = (x - 0.5) / math.sqrt(1 / 12) * inp[:, None, :]
        pre = self.pre_std[:, None, None]
        h = torch.tanh(torch.bmm(h, self.W0) + self.b0[:, None] + self._randn(*h.shape[:2], MAX_WIDTH) * pre)
        for k in range(MAX_HIDDEN_LAYERS):
            hn = torch.tanh(torch.bmm(h, self.Wh[:, k]) + self.bh[:, k, None] + self._randn(*h.shape) * pre)
            h = torch.where((k < self.n_hidden)[:, None, None], hn, h)
        out = torch.bmm(h, self.Wo) + self.bo[:, None]
        return out + self._randn(*out.shape) * self.out_std[:, None, None]

    @torch.no_grad()
    def curve_params(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map configs to curve parameters, exactly as MyRNG consumes the 23 outputs."""
        N = self.sorted.numel()
        idx = torch.searchsorted(self.sorted, self.bnn(x).contiguous()).double()  # [E, M, 23]
        eps = 0.5 / N

        def uni(k, a=0.0, b=1.0):
            return (b - a) * idx[..., k] / N + a

        def u_open(k):
            return uni(k, eps, 1 - eps)

        def normal(k, loc, scale):
            return loc + scale * math.sqrt(2) * torch.erfinv(2 * u_open(k) - 1)

        def expo(k):  # gamma(a=1) and exponential(scale=1)
            return -torch.log1p(-u_open(k))

        y0 = self.y0.double()[:, None]
        ymax = self.ymax.double()[:, None]
        p = {}
        p["Yinf"] = (ymax - y0) * idx[..., 0] / N + y0
        w = torch.stack([expo(k) for k in (1, 2, 3, 4)], -1)
        p["w"] = w / w.sum(-1, keepdim=True)
        p["alpha"] = torch.stack([
            torch.exp(normal(5, 1.0, 1.0)),
            torch.exp(normal(6, 0.0, 1.0)),
            1.0 + torch.exp(normal(7, -4.0, 1.0)),
            torch.exp(normal(8, 0.5, 0.5)),
        ], -1)
        xsat_max = 10 ** normal(9, 0.0, 1.0)
        rel = torch.stack([expo(k) for k in (10, 11, 12, 13)], -1)
        p["Xsat"] = xsat_max[..., None] * rel / rel.max(-1, keepdim=True).values
        p["PREC"] = torch.stack([1.0 / 10 ** uni(k, -3.0, 0.0) for k in (14, 15, 16, 17)], -1)
        p["Rpsat"] = torch.stack([1.0 - expo(k) for k in (18, 19, 20, 21)], -1)
        p["sigma"] = torch.exp(normal(22, -5.0, 1.0))
        p["Y0"] = y0.expand_as(p["Yinf"])
        return p

    @torch.no_grad()
    def curves(self, x: torch.Tensor, steps_per_run: torch.Tensor, t_max: int) -> torch.Tensor:
        """Loss curves [E, M, t_max] for configs x [E, M, d_max].

        Point k (0-based) is at prior time (k + 1) / steps_per_run[e]; points
        beyond steps_per_run are still computed (t > 1) but callers mask them.
        """
        p = self.curve_params(x)
        k = torch.arange(1, t_max + 1, device=self.device, dtype=torch.float64)
        t = (k[None] / steps_per_run.double()[:, None])[:, None, :]  # [E, 1, T]
        y = comb(t.expand(*x.shape[:2], t_max), p)
        y = y + self._randn(*y.shape).double() * p["sigma"][..., None]
        return (1.0 - y.clamp(0.0, 1.0)).float()


def comb(x: torch.Tensor, p: dict[str, torch.Tensor]) -> torch.Tensor:
    """Weighted basis-curve combination; x [E, M, T], params [E, M] or [E, M, 4]."""
    EPS = 1e-9
    Y0 = p["Y0"][..., None]
    Yinf = p["Yinf"][..., None]
    xe = torch.tensor([EPS, 2 * EPS], dtype=torch.float64, device=x.device)

    def par(name, j):
        return p[name][..., j][..., None]

    def brk(j):
        xs, rp = par("Xsat", j), par("Rpsat", j)
        return torch.where(x < xs, x, rp * (x - xs) + xs)

    def with_tail(f, xb):
        # exponential tail for x <= 0, matched to the slope just right of 0
        fe = f(xe.view(1, 1, 2))
        grad = (fe[..., 1:2] - fe[..., 0:1]) / EPS
        tail = Y0 * torch.exp(xb * (grad + EPS) / Y0)
        return torch.where(xb > 0, f(xb.clamp_min(EPS)), tail)

    a, xs, pr = (lambda j: par("alpha", j)), (lambda j: par("Xsat", j)), (lambda j: par("PREC", j))

    def f_pow(z):
        return Yinf - (Yinf - Y0) * ((pr(0) ** (1 / a(0)) - 1) / xs(0) * z + 1) ** -a(0)

    def f_exp(z):
        return Yinf - (Yinf - Y0) * pr(1) ** (-((z / xs(1)) ** a(1)))

    def f_log(z):
        return Yinf - (Yinf - Y0) * torch.log(a(2)) / torch.log((a(2) ** pr(2) - a(2)) * z / xs(2) + a(2))

    def f_hill(z):
        return Yinf - (Yinf - Y0) / ((z / xs(3)) ** a(3) * (pr(3) - 1) + 1)

    w = p["w"]
    out = 0.0
    for j, f in enumerate((f_pow, f_exp, f_log, f_hill)):
        out = out + w[..., j][..., None] * with_tail(f, brk(j))
    return out

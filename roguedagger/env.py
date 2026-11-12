"""Batched freeze-thaw "shop" environment on the GPU, driven by the ifBO prior.

E independent environments, one per batch row. Each has its own spec:
    d   hp dimension (1..d_max)
    T   steps per nominal full training (prior time t = steps / T)
    B   total budget in steps (the seeded first step included)
    n   shop size
    cap whether runs stop at T (tabular-benchmark style) or may go past t = 1

State: started runs (deck) + n shop suggestions that are redrawn at EVERY step.
Action: index into [runs (R_max slots) | shop (N_max slots)]. Picking a shop
item starts it as a new run. Either way the chosen run reveals one more point.

Every curve (runs and shop items) is materialised in full when the config is
first drawn, so the pairwise labels are exact: an item's value for horizon H is
the minimum of its curve over points [0, len + H), past points included, cut at
T when the env is capped. Values come from a running-min (cummin) table, so a
label lookup is a gather.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .prior import PriorTasks


@dataclass
class EnvConfig:
    d_range: tuple = (1, 10)
    T_range: tuple = (10, 100)            # log-uniform int
    budget_full_range: tuple = (2.0, 30.0)  # budget in full trainings, log-uniform
    B_range: tuple = (20, 1200)           # clamp for B in steps
    n_range: tuple = (4, 64)              # shop size, log-uniform int
    p_cap: float = 0.0                    # prob. an episode caps runs at T
    R_max: int = 384                      # max runs per env; shop is masked when full
    shop_mode: str = "uniform"            # uniform | mixed
    p_mutate: float = 0.5                 # mixed: fraction of shop items that mutate a top run
    mutate_std: float = 0.1
    mutate_topk: int = 5


def _log_uniform_int(lo, hi, n, device, gen):
    u = torch.rand(n, device=device, generator=gen)
    return torch.exp(math.log(lo) + u * (math.log(hi + 1) - math.log(lo))).floor().long().clamp(lo, hi)


class ShopEnv:
    def __init__(self, n_envs: int, cfg: EnvConfig, device, d_max: int = 10, seed: int | None = None,
                 fixed_spec: dict | None = None):
        self.E, self.cfg, self.dev, self.d_max = n_envs, cfg, torch.device(device), d_max
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator(device=self.dev)
            self.gen.manual_seed(seed)
        self.fixed_spec = fixed_spec
        self.N = cfg.n_range[1] if fixed_spec is None else fixed_spec["n"]
        self.R = cfg.R_max
        self.L = (cfg.B_range[1] if fixed_spec is None else fixed_spec["B"]) + 1  # curve points kept
        self.prior = PriorTasks(n_envs, d_max, self.dev, generator=self.gen)
        E, R, N, L, D, dev = self.E, self.R, self.N, self.L, d_max, self.dev
        self.run_x = torch.zeros(E, R, D, device=dev)
        self.run_curve = torch.zeros(E, R, L, device=dev)
        self.run_cmin = torch.zeros(E, R, L, device=dev)
        self.run_len = torch.zeros(E, R, dtype=torch.long, device=dev)
        self.run_last = torch.zeros(E, R, dtype=torch.long, device=dev)
        self.n_runs = torch.zeros(E, dtype=torch.long, device=dev)
        self.shop_x = torch.zeros(E, N, D, device=dev)
        self.shop_curve = torch.zeros(E, N, L, device=dev)
        self.shop_cmin = torch.zeros(E, N, L, device=dev)
        self.d = torch.ones(E, dtype=torch.long, device=dev)
        self.T = torch.ones(E, dtype=torch.long, device=dev)
        self.B = torch.ones(E, dtype=torch.long, device=dev)
        self.n = torch.ones(E, dtype=torch.long, device=dev)
        self.cap = torch.zeros(E, dtype=torch.bool, device=dev)
        self.consumed = torch.zeros(E, dtype=torch.long, device=dev)
        self.best = torch.ones(E, device=dev)
        self.oracle = torch.ones(E, device=dev)   # best value reachable by any item ever offered
        self.n_new = torch.zeros(E, dtype=torch.long, device=dev)
        self.reset(torch.arange(E, device=dev))
        self._draw_shop()

    # ------------------------------------------------------------------ helpers
    def _rand(self, *s):
        return torch.rand(*s, device=self.dev, generator=self.gen)

    @property
    def ar(self):
        return torch.arange(self.E, device=self.dev)

    def _sample_spec(self, idx):
        n, c, dev, g = idx.numel(), self.cfg, self.dev, self.gen
        if self.fixed_spec is not None:
            fs = self.fixed_spec
            self.d[idx] = torch.as_tensor(fs["d"], device=dev).expand(n).long() if "d" in fs else \
                torch.randint(c.d_range[0], c.d_range[1] + 1, (n,), device=dev, generator=g)
            self.T[idx], self.B[idx], self.n[idx] = fs["T"], fs["B"], fs["n"]
            self.cap[idx] = bool(fs.get("cap", False))
            return
        self.d[idx] = torch.randint(c.d_range[0], c.d_range[1] + 1, (n,), device=dev, generator=g)
        T = _log_uniform_int(c.T_range[0], c.T_range[1], n, dev, g)
        u = torch.rand(n, device=dev, generator=g)
        lo, hi = map(math.log, c.budget_full_range)
        full = torch.exp(lo + u * (hi - lo))
        self.T[idx] = T
        self.B[idx] = (full * T).round().long().clamp(c.B_range[0], c.B_range[1])
        self.n[idx] = _log_uniform_int(c.n_range[0], c.n_range[1], n, dev, g)
        self.cap[idx] = torch.rand(n, device=dev, generator=g) < c.p_cap

    def _materialise(self, x, idx=None):
        """Curves + running min for configs x [E', M, D] of env rows idx (all if None)."""
        g = (lambda t: t) if idx is None else (lambda t: t[idx])
        T, B, cap = g(self.T), g(self.B), g(self.cap)
        c = self.prior.curves(x, T, self.L, idx)
        # a capped env can't go beyond T, nobody beyond B. Pad with the last legal
        # value so the running min stays meaningful (padded points are never revealed).
        lim = torch.where(cap, T, B + 1).clamp(max=self.L)
        pos = torch.arange(self.L, device=self.dev)
        last = c.gather(2, (lim - 1)[:, None, None].expand(-1, c.shape[1], 1))
        c = torch.where(pos[None, None] < lim[:, None, None], c, last)
        return c, torch.cummin(c, dim=2).values

    def _draw_shop(self):
        E, N, D = self.E, self.N, self.d_max
        x = self._rand(E, N, D)
        if self.cfg.shop_mode == "mixed":
            # mutate a random one of the top-k runs (by best value so far) for a fraction of slots
            k = min(self.cfg.mutate_topk, self.R)
            seen = self.run_cmin.gather(2, (self.run_len - 1).clamp(min=0)[..., None])[..., 0]
            seen = torch.where(self.run_len > 0, seen, torch.full_like(seen, float("inf")))
            top = seen.topk(k, dim=1, largest=False).indices                      # [E, k]
            n_top = (self.n_runs.clamp(max=k)).clamp(min=1)
            pick = (self._rand(E, N) * n_top[:, None]).floor().long()
            parent = self.run_x.gather(1, top.gather(1, pick)[..., None].expand(-1, -1, D))
            child = (parent + torch.randn(E, N, D, device=self.dev, generator=self.gen) * self.cfg.mutate_std).clamp(0, 1)
            use = (self._rand(E, N) < self.cfg.p_mutate) & (self.n_runs > 0)[:, None]
            x = torch.where(use[..., None], child, x)
        dmask = (torch.arange(D, device=self.dev)[None] < self.d[:, None]).float()
        x = x * dmask[:, None]
        self.shop_x = x
        self.shop_curve, self.shop_cmin = self._materialise(x)
        valid = self.shop_valid()
        best_offer = torch.where(valid, self.shop_cmin[..., -1], torch.ones_like(self.shop_cmin[..., -1])).min(1).values
        self.oracle = torch.minimum(self.oracle, best_offer)

    # ------------------------------------------------------------------ masks
    def shop_valid(self):
        return torch.arange(self.N, device=self.dev)[None] < self.n[:, None]

    def run_used(self):
        return torch.arange(self.R, device=self.dev)[None] < self.n_runs[:, None]

    def choosable(self):
        """[E, R + N] legal actions."""
        runs = self.run_used() & ~(self.cap[:, None] & (self.run_len >= self.T[:, None]))
        shop = self.shop_valid() & (self.n_runs < self.R)[:, None]
        return torch.cat([runs, shop], 1)

    # ------------------------------------------------------------------ dynamics
    def reset(self, idx):
        """New spec + task for env rows idx, seeded with one random config trained one step
        (as in ifBO's Algorithm 1). The shop is NOT redrawn here; step() does that for all."""
        if idx.numel() == 0:
            return
        n, D = idx.numel(), self.d_max
        self._sample_spec(idx)
        self.prior.resample(idx, self.d[idx])
        dmask = (torch.arange(D, device=self.dev)[None] < self.d[idx][:, None]).float()
        x = self._rand(n, 1, D) * dmask[:, None]
        c, cm = self._materialise(x, idx)
        self.run_len[idx] = 0
        self.run_last[idx] = 0
        self.run_x[idx, 0] = x[:, 0]
        self.run_curve[idx, 0] = c[:, 0]
        self.run_cmin[idx, 0] = cm[:, 0]
        self.run_len[idx, 0] = 1
        self.run_last[idx, 0] = 1
        self.n_runs[idx] = 1
        self.consumed[idx] = 1
        self.best[idx] = c[:, 0, 0]
        self.oracle[idx] = cm[:, 0, -1]
        self.n_new[idx] = 0

    def _apply(self, action):
        """Apply actions for envs with action >= 0. Returns observed loss [E]."""
        act_env = action >= 0
        is_shop = act_env & (action >= self.R)
        e = self.ar
        # promote shop picks into a fresh run slot
        if is_shop.any():
            es = e[is_shop]
            slot = self.n_runs[es]
            j = action[es] - self.R
            self.run_x[es, slot] = self.shop_x[es, j]
            self.run_curve[es, slot] = self.shop_curve[es, j]
            self.run_cmin[es, slot] = self.shop_cmin[es, j]
            self.run_len[es, slot] = 0
            self.n_runs[es] += 1
            self.n_new[es] += 1
        run = torch.where(is_shop, self.n_runs - 1, action.clamp(min=0))
        ea, ra = e[act_env], run[act_env]
        pos = self.run_len[ea, ra]
        y = self.run_curve[ea, ra, pos.clamp(max=self.L - 1)]
        self.run_len[ea, ra] += 1
        self.consumed[ea] += 1
        self.run_last[ea, ra] = self.consumed[ea]
        self.best[ea] = torch.minimum(self.best[ea], y)
        return y

    def step(self, action):
        """action [E] -> dict with done mask and episode stats of finished envs."""
        self._apply(action)
        done = self.consumed >= self.B
        out = {"done": done.clone()}
        if done.any():
            de = done.nonzero()[:, 0]
            out.update(final_best=self.best[de].clone(), oracle=self.oracle[de].clone(),
                       n_runs=self.n_runs[de].clone(), n_new=self.n_new[de].clone(),
                       B=self.B[de].clone(), T=self.T[de].clone(), n=self.n[de].clone())
        self.reset(done.nonzero()[:, 0])
        self._draw_shop()
        return out

    # ------------------------------------------------------------------ labels
    def horizon(self, f_start=1.0, f_end=1.0):
        rem = (self.B - self.consumed).clamp(min=1)
        prog = self.consumed.float() / self.B.float()
        f = f_start + (f_end - f_start) * prog
        return (f * rem.float()).round().long().clamp(min=1)

    def values(self, H, include_past: bool = True):
        """Label values [E, R + N]: min loss over points [0, len + H), cut at the cap.

        include_past=False (ablation) uses only the future window [len, len + H) for runs,
        i.e. what the next H steps on that run can still reach."""
        if not include_past:
            return self._future_values(H)
        lim_run = self.run_len + H[:, None]
        lim_shop = H[:, None].expand(-1, self.N)
        cap_lim = torch.where(self.cap, self.T, torch.full_like(self.T, self.L))
        lim_run = torch.minimum(lim_run, cap_lim[:, None]).clamp(1, self.L)
        lim_shop = torch.minimum(lim_shop, cap_lim[:, None]).clamp(1, self.L)
        vr = self.run_cmin.gather(2, (lim_run - 1)[..., None])[..., 0]
        vs = self.shop_cmin.gather(2, (lim_shop - 1)[..., None])[..., 0]
        return torch.cat([vr, vs], 1)

    def _future_values(self, H):
        cap_lim = torch.where(self.cap, self.T, torch.full_like(self.T, self.L))
        r = int(self.n_runs.max())
        lo = self.run_len[:, :r]
        hi = torch.minimum(lo + H[:, None], cap_lim[:, None]).clamp(max=self.L)
        pos = torch.arange(self.L, device=self.dev)
        win = (pos[None, None] >= lo[..., None]) & (pos[None, None] < hi[..., None])
        vr = torch.full((self.E, self.R), 1.0, device=self.dev)
        vr[:, :r] = torch.where(win, self.run_curve[:, :r], torch.full_like(self.run_curve[:, :r], 2.0)).min(2).values
        vr = vr.clamp(max=1.0)   # empty window (finished run): never choosable anyway
        lim_shop = torch.minimum(H, cap_lim).clamp(1, self.L)[:, None].expand(-1, self.N)
        vs = self.shop_cmin.gather(2, (lim_shop - 1)[..., None])[..., 0]
        return torch.cat([vr, vs], 1)

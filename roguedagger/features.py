"""Token encoding: one global token, one per run, one per shop suggestion.

Every token has the same width F and the same layout. A slot is zero where it
doesn't apply to the token type.

    [0:3]    type flags            global / run / suggestion
    [3:14]   global (global token) budget, progress, horizon, time scale, counts, incumbent
    [14:19]  run meta              fidelity so far, freshness, cap
    [19:59]  curve (runs)          last 10 points + mask, 16-pt resample, best/last/first/slope
    [59:79]  hyperparameters       d_max values + d_max presence mask

Curve values are z-scored with the mean / std of *all revealed points of all
runs in that env*, so the loss scale never reaches the network, only relative
performance does.

The encoder needs only the observable state. ``Observation`` is filled from
ShopEnv here and from the tabular benchmark harness in evaluate.py, so both use
the same code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

N_LAST, N_RESAMPLE = 10, 16
OFF_G, OFF_R, OFF_C, OFF_H = 3, 14, 19, 59


def feature_dim(d_max: int) -> int:
    return OFF_H + 2 * d_max


@dataclass
class Observation:
    """Observable state of E envs, all on one device.

    run_curve  [E, R, L]  revealed losses (anything past run_len is ignored)
    run_len    [E, R]     points revealed per run (0 = empty slot)
    run_last   [E, R]     value of `consumed` when the run was last advanced
    run_x      [E, R, D]  run hyperparameters in [0, 1]
    shop_x     [E, N, D]
    shop_valid [E, N]
    d, T, B, consumed, H, cap  [E]
    """
    run_curve: torch.Tensor
    run_len: torch.Tensor
    run_last: torch.Tensor
    run_x: torch.Tensor
    shop_x: torch.Tensor
    shop_valid: torch.Tensor
    d: torch.Tensor
    T: torch.Tensor
    B: torch.Tensor
    consumed: torch.Tensor
    H: torch.Tensor
    cap: torch.Tensor


def from_env(env, H: torch.Tensor, r_cur: int | None = None) -> Observation:
    r = int(env.n_runs.max()) if r_cur is None else r_cur
    return Observation(env.run_curve[:, :r], env.run_len[:, :r], env.run_last[:, :r], env.run_x[:, :r],
                       env.shop_x, env.shop_valid(), env.d, env.T, env.B, env.consumed, H, env.cap)


def encode(o: Observation, d_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """-> tokens [E, 1 + R + N, F], key padding mask [E, 1 + R + N] (True = pad)."""
    E, R, L = o.run_curve.shape
    N = o.shop_x.shape[1]
    dev = o.run_curve.device
    F = feature_dim(d_max)
    tok = torch.zeros(E, 1 + R + N, F, device=dev)
    used = o.run_len > 0
    T, B = o.T.float(), o.B.float()
    rem = (o.B - o.consumed).float()

    # ---- normalisation stats over every revealed point of every run
    pos = torch.arange(L, device=dev)
    rev = pos[None, None] < o.run_len[..., None]                          # [E, R, L]
    cnt = rev.sum((1, 2)).clamp(min=1).float()
    vals = torch.where(rev, o.run_curve, torch.zeros_like(o.run_curve))
    mu = vals.sum((1, 2)) / cnt
    var = (torch.where(rev, (o.run_curve - mu[:, None, None]) ** 2, torch.zeros_like(vals))).sum((1, 2)) / cnt
    sd = var.sqrt().clamp(min=1e-6)
    z = lambda v: ((v - mu.view(-1, *([1] * (v.dim() - 1)))) / sd.view(-1, *([1] * (v.dim() - 1)))).clamp(-10, 10)

    big = torch.full_like(o.run_curve, float("inf"))
    best_run = torch.where(rev, o.run_curve, big).min(2).values            # [E, R]
    inc = torch.where(used, best_run, torch.full_like(best_run, float("inf"))).min(1).values

    # ---- global token
    g = tok[:, 0]
    g[:, 0] = 1
    g[:, OFF_G + 0] = rem / B
    g[:, OFF_G + 1] = o.consumed.float() / B
    g[:, OFF_G + 2] = o.H.float() / B
    g[:, OFF_G + 3] = torch.log1p(rem / T)
    g[:, OFF_G + 4] = torch.log1p(o.H.float() / T)
    g[:, OFF_G + 5] = torch.log1p(B / T)
    g[:, OFF_G + 6] = torch.log(T) / math.log(100)
    g[:, OFF_G + 7] = torch.log1p(used.sum(1).float())
    g[:, OFF_G + 8] = torch.log1p(o.shop_valid.sum(1).float())
    g[:, OFF_G + 9] = z(inc)
    g[:, OFF_G + 10] = o.cap.float()

    # ---- run tokens
    if R > 0:
        rt = tok[:, 1:1 + R]
        lenf = o.run_len.float()
        rt[..., 1] = 1
        rt[..., OFF_R + 0] = lenf / T[:, None]
        rt[..., OFF_R + 1] = torch.log1p(lenf) / math.log(1000)
        rt[..., OFF_R + 2] = torch.log1p((o.consumed[:, None] - o.run_last).clamp(min=0).float() / T[:, None])
        rt[..., OFF_R + 3] = (o.run_last == o.consumed[:, None]).float()
        rt[..., OFF_R + 4] = (o.cap[:, None] & (o.run_len >= o.T[:, None])).float()
        # last N_LAST points, most recent first
        k = torch.arange(N_LAST, device=dev)
        li = (o.run_len[..., None] - 1 - k).clamp(min=0)
        lv = o.run_curve.gather(2, li.clamp(max=L - 1))
        lm = (k[None, None] < o.run_len[..., None]).float()
        rt[..., OFF_C:OFF_C + N_LAST] = z(lv) * lm
        rt[..., OFF_C + N_LAST:OFF_C + 2 * N_LAST] = lm
        # resample of the whole revealed prefix
        q = torch.linspace(0, 1, N_RESAMPLE, device=dev)
        ri = (q[None, None] * (lenf[..., None] - 1).clamp(min=0)).round().long().clamp(max=L - 1)
        rt[..., OFF_C + 2 * N_LAST:OFF_C + 2 * N_LAST + N_RESAMPLE] = z(o.run_curve.gather(2, ri))
        s0 = OFF_C + 2 * N_LAST + N_RESAMPLE
        last = lv[..., 0]
        first = o.run_curve[..., 0]
        back = lv[..., min(3, N_LAST - 1)]
        rt[..., s0 + 0] = z(torch.where(used, best_run, mu[:, None].expand_as(best_run)))
        rt[..., s0 + 1] = z(last)
        rt[..., s0 + 2] = z(first)
        rt[..., s0 + 3] = ((last - back) / sd[:, None]).clamp(-10, 10) * (o.run_len > 3).float()
        rt[..., OFF_H:OFF_H + d_max] = o.run_x
        rt.mul_(used[..., None].float())

    # ---- shop tokens
    st = tok[:, 1 + R:]
    st[..., 2] = 1
    st[..., OFF_H:OFF_H + d_max] = o.shop_x
    st.mul_(o.shop_valid[..., None].float())

    # hp presence mask on every non-global token
    dm = (torch.arange(d_max, device=dev)[None] < o.d[:, None]).float()
    tok[:, 1:, OFF_H + d_max:] = dm[:, None] * torch.cat([used, o.shop_valid], 1)[..., None].float()

    pad = torch.cat([torch.zeros(E, 1, dtype=torch.bool, device=dev), ~used, ~o.shop_valid], 1)
    return tok, pad

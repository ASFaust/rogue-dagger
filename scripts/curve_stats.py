"""Summary statistics of real vs prior learning curves (docs/future-ideas/prior-sim2real.md, measurement 3).

For each real task (n configs, d hps, T points), `--twins` prior tasks with the same d, T and number of configs
(uniform random configs in [0, 1]^d). Per task, decision-relevant statistics on the normalised curves; then per
family, the median over real tasks vs over twins and the KS distance between the two distributions.

    PYTHONPATH=. python scripts/curve_stats.py taskset_dev pd1 lcbench [--twins 4]

Families: taskset (test), taskset_dev (data/taskset_dev), pd1, lcbench. Only taskset_dev may be used to *fit* the prior.
"""
import argparse

import numpy as np
import torch

from roguedagger.bench import DATA, load_lcbench, load_pd1, load_taskset
from roguedagger.prior import PriorTasks

LOADERS = {"taskset": load_taskset, "taskset_dev": lambda: load_taskset(DATA / "taskset_dev"), "pd1": load_pd1,
           "lcbench": load_lcbench}


def rank(a, axis=0):
    return np.argsort(np.argsort(a, axis=axis), axis=axis).astype(np.float64)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def stats(Y, X, rng, n_pairs=4000):
    """Y [n, T] loss (lower better), X [n, d]. Returns dict of scalar statistics."""
    n, T = Y.shape
    cm = np.minimum.accumulate(Y, axis=1)
    lo, hi = np.quantile(Y, 0.01), np.quantile(Y, 0.99)
    rngv = max(hi - lo, 1e-9)
    final = cm[:, -1]
    s = {}
    # 1. early-late rank correlation of the running best (what freeze-thaw relies on)
    for f in (0.1, 0.25, 0.5):
        k = max(0, int(round(f * T)) - 1)
        s[f"rank_corr@{f}"] = spearman(cm[:, k], final)
    # 2. crossings: fraction of config pairs whose running-best order at 10% of T differs from the final order
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    k = max(0, int(round(0.1 * T)) - 1)
    d0, d1 = cm[i, k] - cm[j, k], final[i] - final[j]
    ok = (np.abs(d0) > 1e-9) & (np.abs(d1) > 1e-9)
    s["flip@0.1"] = float((np.sign(d0[ok]) != np.sign(d1[ok])).mean()) if ok.any() else np.nan
    # 3. how much hyperparameters matter
    s["final_spread"] = float((np.quantile(final, 0.9) - np.quantile(final, 0.1)) / rngv)
    s["frac_near_best"] = float((final <= final.min() + 0.05 * rngv).mean())
    # 4.-5. curve shapes, each curve measured against its own range; flat curves (own range < 2% of the task's)
    # are counted, then excluded from the shape statistics
    own = Y.max(1) - Y.min(1)
    flat = own < 0.02 * rngv
    s["frac_flat"] = float(flat.mean())
    s["frac_diverge"] = float((Y[:, -1] > Y[:, 0] + 0.05 * rngv).mean())
    s["improve_late"] = float(np.median((cm[:, T // 2] - final) / rngv))   # how much the second half still gains
    Z, own = Y[~flat], own[~flat]
    if len(Z) >= 5:
        Zn = (Z - Z.min(1, keepdims=True)) / own[:, None]
        am = Z.argmin(1)
        s["frac_overfit"] = float(((am < 0.8 * (T - 1)) & (Zn[:, -1] > 0.1)).mean())
        if T >= 7:
            ker = np.ones(5) / 5
            sm = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="valid"), 1, Zn)
            s["noise"] = float(np.median((Zn[:, 2:-2] - sm).std(1)))
            dY = np.diff(Zn, axis=1)
            s["frac_up_steps"] = float((dY > 1e-6).mean())
            a, b = dY[:, :-1], dY[:, 1:]
            a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
            den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
            s["incr_autocorr"] = float(np.nanmedian(np.where(den > 0, (a * b).sum(1) / np.maximum(den, 1e-12),
                                                             np.nan)))
    # 6. config smoothness: rank corr of hp distance vs final-loss difference
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    s["cfg_smooth"] = spearman(dx, np.abs(final[i] - final[j]))
    return s


@torch.no_grad()
def twin_tasks(tasks, k, dev, seed=0):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    D = 10
    for t in tasks:
        n, d = t.X.shape
        d = min(d, D)
        T = t.Y.shape[1]
        pr = PriorTasks(k, D, dev, generator=g)
        pr.resample(torch.arange(k, device=dev), torch.full((k,), d, device=dev))
        x = torch.rand(k, n, D, device=dev, generator=g)
        x[:, :, d:] = 0
        out = []
        for c in range(0, n, 250):   # chunk configs to bound memory on long curves
            out.append(pr.curves(x[:, c:c + 250], torch.full((k,), T, device=dev), T).cpu().numpy())
        Y = np.concatenate(out, 1)
        for r in range(k):
            yield t.name, x[r, :, :d].cpu().numpy(), Y[r]


def ks(a, b):
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    z = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, z, "right") / len(a) - np.searchsorted(b, z, "right") / len(b)).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="+")
    ap.add_argument("--twins", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for fam in a.families:
        tasks = LOADERS[fam]()
        R = [stats(t.Y.astype(np.float64), t.X, rng) for t in tasks]
        S = [stats(Y.astype(np.float64), X, rng) for _, X, Y in twin_tasks(tasks, a.twins, a.device)]
        keys = list(dict.fromkeys(k for x in R + S for k in x))
        print(f"\n{fam}: {len(R)} real tasks, {len(S)} twins")
        print("statistic".ljust(16) + "real med".rjust(10) + "twin med".rjust(10) + "real IQR".rjust(16)
              + "twin IQR".rjust(16) + "KS".rjust(7))
        rows = []
        for key in keys:
            r = np.array([x.get(key, np.nan) for x in R])
            s = np.array([x.get(key, np.nan) for x in S])
            rows.append((ks(r, s), key, r, s))
        for D_, key, r, s in sorted(rows, key=lambda z: -np.nan_to_num(z[0])):
            q = lambda v: np.nanquantile(v, [0.25, 0.5, 0.75])
            qr, qs = q(r), q(s)
            print(key.ljust(16) + f"{qr[1]:10.3f}{qs[1]:10.3f}" + f"  [{qr[0]:5.2f},{qr[2]:5.2f}]".rjust(16)
                  + f"  [{qs[0]:5.2f},{qs[2]:5.2f}]".rjust(16) + f"{D_:7.2f}")


if __name__ == "__main__":
    main()

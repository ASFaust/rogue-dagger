"""Seed-aware comparison of configs, where each config is several independently trained models (training seeds).

    python scripts/compare_seeds.py pd1 wide=wide100k-n0-t0.25,wides1-n0-t0.25 \
        focus03=wfocus03-n0-t0.25,wf03s1-n0-t0.25 --ref wide

A single tag also works (e.g. ifbo=ifbo, which has no training seeds).
Per config, the per-task mean final regret is averaged over its models. The CI is a two-level bootstrap: resample
tasks, and within each config resample its models with replacement. So it includes training-seed variance, which
scripts/compare.py's task-only bootstrap doesn't. With 2-3 seeds the seed level is crude; read the CI as a lower
bound on the uncertainty.
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(family, tag):
    z = np.load(ROOT / "runs" / "eval" / family / f"{tag}.npz")
    return dict(zip(map(str, z["tasks"]), z["regret"][:, :, -1].mean(1)))   # task -> mean final regret over eval seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("groups", nargs="+", help="name=tag1,tag2,...")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()
    G = {}
    for g in a.groups:
        name, tags = g.split("=", 1)
        G[name] = [load(a.family, t) for t in tags.split(",")]
    tasks = sorted(set.intersection(*[set(m) for ms in G.values() for m in ms]))
    M = {n: np.array([[m[t] for t in tasks] for m in ms]) for n, ms in G.items()}   # [n_models, n_tasks]
    print(f"{a.family}: {len(tasks)} tasks")
    print("config".ljust(12) + "models".rjust(7) + "mean".rjust(9) + "  per model")
    for n, x in M.items():
        print(n.ljust(12) + f"{len(x):7d}{x.mean():9.4f}  " + " ".join(f"{v:.4f}" for v in x.mean(1)))
    rng = np.random.default_rng(0)
    T = len(tasks)
    ref = M[a.ref]
    for n, x in M.items():
        if n == a.ref:
            continue
        point = (ref.mean(0) - x.mean(0)).mean()
        boots = np.empty(a.boot)
        for b in range(a.boot):
            ti = rng.integers(0, T, T)
            r = ref[rng.integers(0, len(ref), len(ref))][:, ti].mean()
            m = x[rng.integers(0, len(x), len(x))][:, ti].mean()
            boots[b] = r - m
        lo, hi = np.percentile(boots, [2.5, 97.5])
        wins = int(((ref.mean(0) - x.mean(0)) > 0).sum())
        print(f"  {n} vs {a.ref}: better on {wins}/{T} tasks, gain {point:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] "
              f"(seeds resampled)")


if __name__ == "__main__":
    main()

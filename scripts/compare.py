"""Compare eval runs: mean normalised regret (final + at budget checkpoints), average rank,
per-task paired wins vs a reference method, and a regret-vs-budget plot.

    python scripts/compare.py taskset ifbo random_search rd-base-5k [--ref ifbo]
Reads runs/eval/<family>/<tag>.npz, writes runs/eval/<family>/compare_<tags>.png and prints a table.
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--ref", default="ifbo")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    d = ROOT / "runs" / "eval" / a.family
    runs = {}
    for t in a.tags:
        f = d / f"{t}.npz"
        if not f.exists():
            print(f"missing {f}")
            continue
        z = np.load(f)
        runs[t] = (z["regret"], list(z["tasks"]))
    # align on common tasks
    common = sorted(set.intersection(*[set(v[1]) for v in runs.values()]))
    R = {t: r[[names.index(c) for c in common]] for t, (r, names) in runs.items()}  # [task, seed, B]
    B = next(iter(R.values())).shape[-1]
    checkpoints = [b for b in (100, 250, 500, 1000) if b <= B]
    # ranks per task/seed/step on regret (lower is better), averaged
    stack = np.stack([R[t] for t in R])                      # [M, task, seed, B]
    n_seeds = min(s.shape[1] for s in R.values())
    stack = np.stack([R[t][:, :n_seeds] for t in R])
    ranks = stack.argsort(0).argsort(0).astype(float) + 1   # ties broken arbitrarily
    print(f"{a.family}: {len(common)} tasks, {n_seeds} seeds, budget {B}")
    hdr = "method".ljust(22) + "".join(f"regret@{b}".rjust(12) for b in checkpoints) + "rank@end".rjust(10) + "rank_avg".rjust(10)
    print(hdr)
    for i, t in enumerate(R):
        row = t.ljust(22) + "".join(f"{stack[i, :, :, b - 1].mean():12.4f}" for b in checkpoints)
        row += f"{ranks[i, :, :, -1].mean():10.2f}{ranks[i].mean():10.2f}"
        print(row)
    if a.ref in R:
        ref = stack[list(R).index(a.ref)][..., -1].mean(1)     # per task mean final
        for i, t in enumerate(R):
            if t == a.ref:
                continue
            mine = stack[i][..., -1].mean(1)
            diff = ref - mine
            rng = np.random.default_rng(0)
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {t} vs {a.ref}: better on {int((diff > 0).sum())}/{len(diff)} tasks, "
                  f"mean final regret gain {diff.mean():+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(1, B + 1)
        for i, t in enumerate(R):
            ax[0].plot(x, stack[i].mean((0, 1)), label=t)
            ax[1].plot(x, ranks[i].mean((0, 1)), label=t)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("steps spent")
        ax[0].set_ylabel("mean normalised regret")
        ax[1].set_xlabel("steps spent")
        ax[1].set_ylabel("average rank")
        ax[0].legend()
        ax[0].set_title(a.family)
        fig.tight_layout()
        out = d / f"compare_{'_'.join(R)}.png"
        fig.savefig(out, dpi=120)
        print(f"plot: {out}")


if __name__ == "__main__":
    main()

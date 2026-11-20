"""How rare are the top configs? Regret of the best of k random configs, real tables vs matched prior twins.

Per real task (d, T, n_cfg) one or more prior twins with the same d and T, n_cfg random configs, capped at T.
Regret is normalised by the min / max over all points (the benchmark convention). Also: fraction of configs
within 1 % / 5 % of the range from the best.

    PYTHONPATH=. python scripts/best_of_k.py lcbench [--reps 4]
"""
import argparse

import numpy as np
import torch

from roguedagger.bench import load_family
from roguedagger.prior import PriorTasks

KS = (1, 5, 10, 20, 50, 100, 200, 500)


def stats(best, lo, hi, rng, n_draw=400):
    nb = (best - lo) / (hi - lo)
    out = []
    for k in KS:
        idx = np.array([rng.choice(len(nb), size=min(k, len(nb)), replace=False) for _ in range(n_draw)])
        out.append(nb[idx].min(1).mean())
    return out + [(nb - nb.min() < 0.01).mean(), (nb - nb.min() < 0.05).mean()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tasks = load_family(a.family)
    real = np.array([stats(np.asarray(t.Y).min(1), np.asarray(t.Y).min(), np.asarray(t.Y).max(), rng) for t in tasks])
    twin = []
    g = torch.Generator(device=a.device).manual_seed(0)
    for t in tasks:
        n, T = t.Y.shape
        d = t.X.shape[1]
        P = PriorTasks(a.reps, 10, a.device, generator=g)
        P.resample(torch.arange(a.reps, device=a.device), torch.full((a.reps,), d, device=a.device))
        x = torch.rand(a.reps, n, 10, device=a.device, generator=g) * (torch.arange(10, device=a.device) < d)
        Y = P.curves(x, torch.full((a.reps,), T, device=a.device), T).cpu().numpy()
        for r in range(a.reps):
            twin.append(stats(Y[r].min(1), Y[r].min(), Y[r].max(), rng))
    twin = np.array(twin)
    hdr = "".join(f"k={k:<6d}" for k in KS) + "near1%   near5%"
    print(f"{a.family}: {len(tasks)} tasks, {a.reps} twins each; mean regret of the best of k random configs")
    print("        " + hdr)
    for nm, v in (("real", real), ("prior", twin)):
        print(f"{nm:8s}" + "".join(f"{x:<8.4f}" for x in v.mean(0)))


if __name__ == "__main__":
    main()

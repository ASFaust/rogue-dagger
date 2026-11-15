"""Matched synthetic twins (docs/future-ideas/prior-sim2real.md, measurement 1).

For every real benchmark task, prior tasks with the same d and T, budget 1000, shop 32 and capped runs, so only
the curve distribution differs from the benchmark eval. Each model runs on the same seeded twins with the
checkpoint and temperature of one of its benchmark evals; the result is a normalised regret per twin task.

    PYTHONPATH=. python scripts/twins.py rlwide-n32-t0.25 wfocus03-n32-t1.0 ... [--reps 4]
    PYTHONPATH=. python scripts/twins.py --rho        # Spearman rho per family: twin vs benchmark regret

Regret is normalised by a pool of 1000 random configs per twin task (min / max loss over all their points), the
prior analogue of the table's best / worst value that the benchmark regret uses.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from roguedagger.bench import load_family
from roguedagger.env import ShopEnv
from roguedagger.model import load_policy
from roguedagger.train import env_config, policy_step

ROOT = Path(__file__).resolve().parent.parent
FAMS = ("taskset", "pd1", "lcbench")
OUT = ROOT / "runs" / "twins"


def specs(reps):
    fam, name, d, T = [], [], [], []
    for f in FAMS:
        for t in load_family(f):
            for _ in range(reps):
                fam.append(f), name.append(t.name), d.append(t.X.shape[1]), T.append(t.Y.shape[1])
    return fam, name, np.array(d), np.array(T)


@torch.no_grad()
def run(tag, reps, seed, device):
    args = ast.literal_eval(str(np.load(ROOT / "runs/eval/taskset" / f"{tag}.npz")["args"]))
    ck = torch.load(ROOT / args["ckpt"], map_location=device)
    model = load_policy(ck, device)
    tc = ck["cfg"]["train"]
    ecfg = env_config(ck["cfg"])
    ecfg.R_max, ecfg.ref_pool, ecfg.shop_mode = 1000, 1000, "uniform"
    fam, name, d, T = specs(reps)
    E = len(d)
    spec = {"d": torch.tensor(d, device=device), "T": torch.tensor(T, device=device), "B": 1000, "n": 32,
            "cap": True}
    env = ShopEnv(E, ecfg, device, seed=seed, fixed_spec=spec)
    reg = torch.full((E,), float("nan"), device=device)
    nruns = torch.zeros(E, device=device)
    finished = torch.zeros(E, dtype=torch.bool, device=device)
    while not finished.all():
        a = policy_step(model, env, args["temperature"], args["greedy"], h=(tc["h_start"], tc["h_end"]))
        out = env.step(a)
        new = out["done"] & ~finished
        if new.any():
            reg[new] = out["regret"][new]
            de = out["done"].nonzero()[:, 0]
            nruns[de[new[de]]] = out["n_runs"][new[de]].float()
            finished |= new
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{tag}.npz", family=np.array(fam), task=np.array(name), regret=reg.cpu().numpy(),
             n_runs=nruns.cpu().numpy(), args=json.dumps(dict(reps=reps, seed=seed, ckpt=args["ckpt"],
                                                              temperature=args["temperature"])))
    r = reg.cpu().numpy()
    print(tag, " ".join(f"{f} {r[np.array(fam) == f].mean():.4f}" for f in FAMS),
          f"runs {nruns.mean().item():.0f}", flush=True)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def rho(tags=None):
    rows = []
    for p in sorted(OUT.glob("*.npz")):
        if tags and p.stem not in tags:
            continue
        z = np.load(p)
        tw = {f: z["regret"][z["family"] == f].mean() for f in FAMS}
        bench = {}
        for f in FAMS:
            q = ROOT / "runs/eval" / f / f"{p.stem}.npz"
            bench[f] = np.load(q)["regret"][:, :, -1].mean() if q.exists() else np.nan
        rows.append((p.stem, tw, bench))
    groups = {"all": rows, "pairwise": [r for r in rows if not r[0].startswith(("rl", "sac"))],
              "RL": [r for r in rows if r[0].startswith(("rl", "sac"))]}
    print("model".ljust(26) + "".join(f"{f[:7]:>9}tw{f[:7]:>9}bn" for f in FAMS))
    for nm, tw, bn in rows:
        print(nm.ljust(26) + "".join(f"{tw[f]:11.4f}{bn[f]:11.4f}" for f in FAMS))
    for g, rs in groups.items():
        if len(rs) < 3:
            continue
        cells = []
        for f in FAMS:
            ok = [r for r in rs if np.isfinite(r[2][f])]
            cells.append(f"{f} {spearman([r[1][f] for r in ok], [r[2][f] for r in ok]):+.2f}")
        print(f"rho {g} ({len(rs)}): " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rho", action="store_true")
    a = ap.parse_args()
    if a.rho:
        return rho(a.tags or None)
    for t in a.tags:
        if (OUT / f"{t}.npz").exists():
            print(t, "exists, skipped")
            continue
        run(t, a.reps, a.seed, a.device)


if __name__ == "__main__":
    main()

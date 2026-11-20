"""When does the policy start new runs on matched prior twins? (BACKLOG step 2, LCBench analysis.)

Prior tasks with each LCBench task's d and T, budget 1000, a large uniform shop (stand-in for the full table), the
deploy temperature; capped or uncapped. Prints new-run starts per tenth of the budget and the final regret.

    PYTHONPATH=. python scripts/twin_starts.py runs/wide-focus03-v1/last.pt --family lcbench --shop 512 --tau 0.25
"""
import argparse

import numpy as np
import torch

from roguedagger.bench import load_family
from roguedagger.env import ShopEnv
from roguedagger.model import load_policy
from roguedagger.train import env_config, policy_step


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--family", default="lcbench")
    ap.add_argument("--shop", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shop-bias", type=float, nargs="+", default=[0.0])
    ap.add_argument("--caps", default="1,0", help="1 = capped, 0 = uncapped")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=a.device)
    model = load_policy(ck, a.device)
    tc = ck["cfg"]["train"]
    tasks = load_family(a.family)
    d = np.repeat([t.X.shape[1] for t in tasks], a.reps)
    T = np.repeat([t.Y.shape[1] for t in tasks], a.reps)
    for cap, bias in [(c == "1", b) for c in a.caps.split(",") for b in a.shop_bias]:
        ecfg = env_config(ck["cfg"])
        ecfg.R_max, ecfg.ref_pool, ecfg.shop_mode = a.budget, 1000, "uniform"
        E = len(d)
        spec = {"d": torch.tensor(d, device=a.device), "T": torch.tensor(T, device=a.device), "B": a.budget,
                "n": a.shop, "cap": cap}
        env = ShopEnv(E, ecfg, a.device, seed=a.seed, fixed_spec=spec)
        starts = np.zeros(10)
        reg = torch.full((E,), float("nan"), device=a.device)
        finished = torch.zeros(E, dtype=torch.bool, device=a.device)
        while not finished.all():
            before = env.n_runs.clone()
            dec = (env.consumed * 10 // a.budget).clamp(max=9)
            act = policy_step(model, env, a.tau, False, h=(tc["h_start"], tc["h_end"]), shop_bias=bias)
            out = env.step(act)
            # a reset sets n_runs to 1, so count only envs that were not done
            newrun = (out["n_runs_now"] if "n_runs_now" in out else env.n_runs) > before
            m = (~finished) & newrun & ~out["done"]
            np.add.at(starts, dec[m].cpu().numpy(), 1)
            new = out["done"] & ~finished
            if new.any():
                reg[new] = out["regret"][new]
                finished |= new
        print(f"cap={cap!s:5s} bias={bias:+.2f} starts per decile {(starts / E).round(1)}  total {starts.sum() / E:.0f}  "
              f"regret {reg.mean().item():.4f}", flush=True)


if __name__ == "__main__":
    main()

"""Pick the deploy temperature on the synthetic prior instead of the benchmarks.

Held-out synthetic tasks (seed differs from the training-time eval), benchmark-like
shape (T steps per full training, budget B, capped), for several shop sizes and τ.

    CUDA_VISIBLE_DEVICES=4 python scripts/synth_tau.py runs/wide-v1/last.pt --n 32 512 --tau 1 0.5 0.25 0.15 0.1
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roguedagger.model import Scorer  # noqa: E402
from roguedagger.train import env_config, policy_step, synthetic_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--n", type=int, nargs="+", default=[32, 512])
ap.add_argument("--tau", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.15, 0.1])
ap.add_argument("--T", type=int, nargs="+", default=[20, 50, 200])
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--envs", type=int, default=128)
ap.add_argument("--seed", type=int, default=777)
a = ap.parse_args()

dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev)
cfg = ck["cfg"]
model = Scorer(d_max=10, **cfg["model"]).to(dev).eval()
model.load_state_dict(ck["model"])
tc = cfg["train"]
h = (tc["h_start"], tc["h_end"])
ecfg = env_config(cfg)
print(f"{'n':>5} {'T':>4} " + " ".join(f"τ={t:<6}" for t in a.tau))
for n in a.n:
    for T in a.T:
        cfg["eval"] = {"n_envs": a.envs, "seed": a.seed, "T": T, "B": a.B, "n": n, "cap": True}
        row = []
        for t in a.tau:
            with torch.no_grad():
                r = synthetic_eval(lambda e: policy_step(model, e, t, h=h), cfg, dev, ecfg)
            row.append(r["regret"])
        print(f"{n:>5} {T:>4} " + " ".join(f"{x:<8.4f}" for x in row), flush=True)

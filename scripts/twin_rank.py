"""Is our ranking of unseen configs kNN-level because of the encoding/model, or because the prior doesn't match the
benchmark? (Twin diagnostic, journal 2026-11-20.)

For each benchmark task, a twin table: the task's own configs X (same count, same normalised values) with curves
from one ifBO prior task of the same d and T. Same protocol on both: the model runs the episode (full table, deploy
temperature); at steps K the unseen configs are ranked by our scorer (rd), FT-PFN (MFPI at T) and 5-NN, against
each config's true best value in the table. If rd beats kNN on the twins but not on the real tables, the gap is
sim-to-real (the prior); if rd is kNN-level on the twins too, it is the model / hp encoding.

    PYTHONPATH=. python scripts/twin_rank.py --ckpt runs/wide-focus03-v1/last.pt --source real twin
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from roguedagger.bench import Task, TabEnv, load_family          # noqa: E402
from roguedagger.model import load_policy                         # noqa: E402
from roguedagger.prior import PriorTasks                          # noqa: E402
from roguedagger.train import policy_step                         # noqa: E402
from roguedagger.features import encode, from_env                 # noqa: E402
from shop_rank import ftpfn_scores, knn_scores                    # noqa: E402


def rd_all(model, env, h):
    """Our scorer's score of every table config, per episode row [E, n_cfg] (higher = better)."""
    obs = from_env(env, env.horizon(*h))
    tok, pad = encode(obs, env.d_max)
    return model(tok, pad).float()[:, obs.run_len.shape[1]:].cpu().numpy()


def twin_task(task, seed, device, d_max=10, clean=False):
    n, d = task.X.shape
    T = task.Y.shape[1]
    g = torch.Generator(device=device).manual_seed(seed)
    pr = PriorTasks(1, d_max, device, generator=g)
    pr.resample(torch.tensor([0], device=device), torch.tensor([d], device=device))
    x = torch.zeros(1, n, d_max, device=device)
    x[0, :, :d] = torch.as_tensor(task.X, dtype=torch.float32, device=device)
    Y = pr.curves(x, torch.tensor([T], device=device), T)[0].cpu().numpy()
    if not clean:
        return Task(task.name + "-twin", task.family, task.X, Y), Y.min(1)
    pr.noise_scale = 0.0          # same task, observation noise off: the ranking target without noise
    return Task(task.name + "-twin", task.family, task.X, Y), pr.curves(x, torch.tensor([T], device=device), T)[0].cpu().numpy().min(1)


class One:
    """View of row e of a multi-episode TabEnv as an E=1 env (what shop_rank's scorers expect)."""

    def __init__(self, env, e):
        self.__dict__["_env"], self.__dict__["_e"] = env, e

    def __getattr__(self, k):
        v = getattr(self._env, k)
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == self._env.E:
            return v[self._e:self._e + 1]
        return v

    def horizon(self, *h):
        return self._env.horizon(*h)[self._e:self._e + 1]

    def choosable(self):
        return self._env.choosable()[self._e:self._e + 1]


def ard_scores(env, unseen):
    """GP with ARD Matern-5/2 on (run hps, best value of the run so far): in-context per-dim relevance."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    nr, d = int(env.n_runs[0]), int(env.d[0])
    X = env.run_x[0, :nr, :d].cpu().numpy()
    y = np.array([float(env.run_curve[0, i, :int(env.run_len[0, i])].min()) for i in range(nr)])
    y = (y - y.mean()) / (y.std() + 1e-9)
    k = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 1e2), nu=2.5) \
        + WhiteKernel(0.1, (1e-4, 1e1))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=1, random_state=0).fit(X, y)
    return -gp.predict(env.Xall[torch.as_tensor(unseen), :d].cpu().numpy())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--family", default="lcbench")
    ap.add_argument("--source", nargs="+", default=["real", "twin"])
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 300, 600])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--random-slots", action="store_true")
    ap.add_argument("--no-ftpfn", action="store_true")
    ap.add_argument("--ard", action="store_true", help="also rank by an ARD-GP fitted to the runs (sklearn)")
    ap.add_argument("--clean-target", action="store_true",
                    help="twins: rank against each config's noise-free best value (episodes still see noisy curves)")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=a.device)
    model = load_policy(ck, a.device)
    h = (ck["cfg"]["train"]["h_start"], ck["cfg"]["train"]["h_end"])
    meths = ["rd", "knn"] + ([] if a.no_ftpfn else ["ftpfn"]) + (["ard"] if a.ard else [])
    tasks = [t for t in load_family(a.family) if not a.tasks or t.name in a.tasks]
    for src in a.source:
        res = {(K, m): [] for K in a.steps for m in meths}
        for i, task0 in enumerate(tasks):
            if src == "real":
                task, best = task0, task0.Y.min(1)
            else:
                task, best = twin_task(task0, 1000 + i, a.device, clean=a.clean_target)
            lo, hi = task.Y.min(), task.Y.max()
            env = TabEnv(task, a.episodes, max(a.steps) + 1, n_shop=None, device=a.device, seed=i,
                         random_slots=a.random_slots)
            for K in sorted(a.steps):
                while int(env.consumed.min()) < K:
                    env.step(policy_step(model, env, a.tau, False, h=h, amp=True))
                rd = rd_all(model, env, h)
                for e in range(a.episodes):
                    v = One(env, e)
                    unseen = env.unseen[e].nonzero()[:, 0].cpu().numpy()
                    sco = {"rd": rd[e, unseen],
                           "knn": knn_scores(v, unseen)}
                    if not a.no_ftpfn:
                        sco["ftpfn"] = ftpfn_scores(v, unseen)
                    if a.ard:
                        sco["ard"] = ard_scores(v, unseen)
                    bu = best[unseen]
                    for m, s in sco.items():
                        o = np.argsort(-s)
                        res[(K, m)].append((spearmanr(s, -bu).correlation, (bu[o[0]] - lo) / (hi - lo),
                                            (bu[o[:10]].min() - lo) / (hi - lo)))
            print(f"{src} done {task0.name}", file=sys.stderr, flush=True)
        print(f"\n{a.family} {src}  {Path(a.ckpt).parent.name}  {len(tasks)} tasks x {a.episodes} episodes")
        print("step  method   rho    top1-regret  top10-regret")
        for K in a.steps:
            for m in meths:
                r = np.array(res[(K, m)])
                print(f"{K:4d}  {m:6s} {np.nanmean(r[:, 0]):+.3f}   {r[:, 1].mean():.4f}       {r[:, 2].mean():.4f}",
                      flush=True)


if __name__ == "__main__":
    main()

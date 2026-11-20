"""How well does a method rank the *unseen* configs? (BACKLOG step 2, LCBench selection error.)

Replays recorded episodes (step_cfg from evaluate.py) to steps K, and at each such state ranks every unseen
config by
  rd      our scorer's shop score (optionally averaged over hp orders)
  ftpfn   FT-PFN's P(y(T) > f_best) (MFPI at h = T, threshold = incumbent; FT-PFN works in 1 - loss)
  knn     mean best-so-far of the 5 nearest started runs in hp space (rank-free sanity baseline)
against the config's true best value in the table. Reports Spearman rho over unseen configs, and the regret
(normalised) of each method's top-1 and the best of its top-10 unseen configs.

    python scripts/shop_rank.py lcbench trace-wfocus03-n0-t0.25 --ckpt runs/wide-focus03-v1/last.pt --steps 100 300 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from roguedagger.bench import TabEnv, load_family          # noqa: E402
from roguedagger.features import OFF_H, encode, from_env   # noqa: E402
from roguedagger.model import load_policy                  # noqa: E402


def replay(task, sc, K, device):
    env = TabEnv(task, 1, len(sc), n_shop=None, device=device, seed=0)
    # the constructor already ran step 0 with its own random config; swap in the recorded one
    c0, c = int(env.run_cfg[0, 0]), int(sc[0])
    env.unseen[0, c0] = True
    env.run_cfg[0, 0], env.run_curve[0, 0], env.run_x[0, 0] = c, env.Yall[c], env.Xall[c]
    env.unseen[0, c] = False
    env.best[0] = env.trace[0, 0] = env.Yall[c, 0]
    env.step_cfg[0, 0] = c
    env._draw_shop()
    slot = {c: 0}
    for t in range(1, K):
        c = int(sc[t])
        if c in slot:
            a = slot[c]
        else:
            slot[c] = int(env.n_runs[0])
            a = env.R + c
        env.step(torch.tensor([a], device=env.dev))
    return env


@torch.no_grad()
def rd_scores(models, env, perms, h):
    obs = from_env(env, env.horizon(*h))
    tok, pad = encode(obs, env.d_max)
    r = obs.run_len.shape[1]
    d = int(env.d[0])
    acc = 0
    for m in models:
        for p in perms:
            t = tok.clone()
            t[..., OFF_H:OFF_H + d] = tok[..., OFF_H + p.to(tok.device)]
            acc = acc + m(t, pad).float()[0, r:]
    return (acc / (len(models) * len(perms))).cpu().numpy()   # [n_cfg], higher = better


@torch.no_grad()
def ftpfn_scores(env, unseen):
    from roguedagger.evaluate import _ftpfn
    m = _ftpfn()
    d, T = int(env.d[0]), int(env.T[0])
    nr = int(env.n_runs[0])
    xs, ys = [], []
    for k in range(nr):
        L = int(env.run_len[0, k])
        t = torch.arange(1, L + 1, dtype=torch.float32) / T
        xs.append(torch.cat([torch.full((L, 1), float(env.run_cfg[0, k]) + 1), t[:, None],
                             env.run_x[0, k, :d].cpu().expand(L, d)], 1))
        ys.append(1.0 - env.run_curve[0, k, :L].cpu())
    x_train, y_train = torch.cat(xs), torch.cat(ys).clamp(0, 1)
    ids = torch.as_tensor(unseen)
    x_test = torch.cat([(ids + 1).float()[:, None], torch.ones(len(ids), 1), env.Xall[ids, :d].cpu()], 1)
    logits = m(x_train=x_train, y_train=y_train, x_test=x_test)
    return m.model.criterion.pi(logits, best_f=torch.tensor(float(y_train.max()))).reshape(-1).numpy()


def knn_scores(env, unseen, k=5):
    nr = int(env.n_runs[0])
    d = int(env.d[0])
    Xr = env.run_x[0, :nr, :d].cpu().numpy()
    yb = np.array([float(env.run_curve[0, i, :int(env.run_len[0, i])].min()) for i in range(nr)])
    Xu = env.Xall[torch.as_tensor(unseen), :d].cpu().numpy()
    D = ((Xu[:, None] - Xr[None]) ** 2).sum(-1)
    nn = np.argsort(D, 1)[:, :min(k, nr)]
    return -yb[nn].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("trace")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--hp-perms", type=int, default=1)
    ap.add_argument("--steps", type=int, nargs="+", default=[100, 300, 600])
    ap.add_argument("--seeds", type=int, default=3, help="eval seeds (episodes) per task")
    ap.add_argument("--no-ftpfn", action="store_true")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from roguedagger.evaluate import hp_perms
    models = [load_policy(torch.load(c, map_location=a.device), a.device) for c in a.ckpt]
    tc = torch.load(a.ckpt[0], map_location="cpu")["cfg"]["train"]
    h = (tc["h_start"], tc["h_end"])
    z = np.load(ROOT / "runs" / "eval" / a.family / f"{a.trace}.npz")
    tasks = {t.name: t for t in load_family(a.family)}
    names = [n for n in map(str, z["tasks"]) if not a.tasks or n in a.tasks]
    meths = ["rd", "knn"] + ([] if a.no_ftpfn else ["ftpfn"])
    res = {(K, mth): [] for K in a.steps for mth in meths}
    for name in names:
        i = list(map(str, z["tasks"])).index(name)
        task = tasks[name]
        best = np.asarray(task.Y, float).min(1)
        lo, hi = np.asarray(task.Y).min(), np.asarray(task.Y).max()
        perms = hp_perms(task.X.shape[1], a.hp_perms, seed=i)
        for s in range(a.seeds):
            sc = z["step_cfg"][i, s]
            for K in a.steps:
                env = replay(task, sc, K, a.device)
                unseen = env.unseen[0].nonzero()[:, 0].cpu().numpy()
                sco = {"rd": rd_scores(models, env, perms, h)[unseen], "knn": knn_scores(env, unseen)}
                if not a.no_ftpfn:
                    sco["ftpfn"] = ftpfn_scores(env, unseen)
                inc = (float(env.best[0]) - lo) / (hi - lo)
                for mth, v in sco.items():
                    o = np.argsort(-v)
                    bu = best[unseen]
                    res[(K, mth)].append((spearmanr(v, -bu).correlation, (bu[o[0]] - lo) / (hi - lo),
                                          (bu[o[:10]].min() - lo) / (hi - lo), (bu.min() - lo) / (hi - lo), inc))
        print(f"done {name}", flush=True)
    print(f"\n{a.family} {a.trace}: {len(names)} tasks x {a.seeds} episodes; regrets normalised, of the true best value")
    print("step  method   rho   top1-regret  top10-regret  best-unseen  incumbent")
    for K in a.steps:
        for mth in meths:
            r = np.array(res[(K, mth)])
            print(f"{K:4d}  {mth:6s} {np.nanmean(r[:, 0]):+.3f}   {r[:, 1].mean():.4f}       {r[:, 2].mean():.4f}"
                  f"        {r[:, 3].mean():.4f}      {r[:, 4].mean():.4f}")


if __name__ == "__main__":
    main()

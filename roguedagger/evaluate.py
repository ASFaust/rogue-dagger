"""Run freeze-thaw HPO on tabular benchmarks and save normalised-regret traces.

Methods
  rd             Rogue&Dagger scorer (checkpoint), shop of --n-shop unseen configs per step
                 (--n-shop 0 = every unseen config, ifBO's tabular setting); sampled or --greedy
  ifbo           FT-PFN + MFPI-random over all partial runs + every unseen config (ifBO, ICML'24)
  random_search  full-fidelity random search

Protocol as in the ifBO paper: 1000 steps per task, one random config trained one step
first, 10 seeds. Regret is normalised by the table's best / worst value.

    python -m roguedagger.evaluate --method rd --ckpt runs/base-v1/last.pt --family taskset --device cuda
    python -m roguedagger.evaluate --method ifbo --family taskset --workers 40
Output: runs/eval/<family>/<tag>.npz with regret [n_tasks, n_seeds, budget], tasks, wall [n_tasks].
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .bench import TabEnv, load_family
from .train import policy_step, random_search_step

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------- ifBO
_FTPFN = None


def _ftpfn():
    global _FTPFN
    if _FTPFN is None:
        sys.path.insert(0, str(ROOT / "third_party"))
        from ifbo.surrogate import FTPFN
        _FTPFN = FTPFN(version="0.0.1", target_path=ROOT / "data" / "ftpfn")
        _FTPFN.eval()
    return _FTPFN


@torch.no_grad()
def ifbo_step(env: TabEnv, rng: np.random.Generator):
    """MFPI-random (Algorithm 2): random horizon h ~ U{1..bmax}, threshold
    f_best + 10^U(-4,-1) (1 - f_best); argmax P(y(b + h) > thr) over partial runs + unseen configs."""
    m = _ftpfn()
    d = int(env.d[0])
    T = int(env.T[0])
    actions = torch.zeros(env.E, dtype=torch.long)
    ch = env.choosable()
    for e in range(env.E):
        if env.done[e]:
            continue
        nr = int(env.n_runs[e])
        lens = env.run_len[e, :nr]
        xs, ys = [], []
        for k in range(nr):
            L = int(lens[k])
            t = torch.arange(1, L + 1, dtype=torch.float32) / T
            hp = env.run_x[e, k, :d].expand(L, d)
            # curve id = table index + 1, as in ifBO's experiment code (unique per config, <= 1000)
            xs.append(torch.cat([torch.full((L, 1), float(env.run_cfg[e, k]) + 1), t[:, None], hp], 1))
            ys.append(1.0 - env.run_curve[e, k, :L])
        x_train, y_train = torch.cat(xs), torch.cat(ys).clamp(0, 1)
        h = int(rng.integers(1, T + 1))
        f_best = float(y_train.max())
        thr = f_best + 10 ** rng.uniform(-4, -1) * (1 - f_best)
        run_ok = ch[e, :nr].nonzero()[:, 0]
        shop_ok = ch[e, env.R:].nonzero()[:, 0]
        q_run = torch.cat([(env.run_cfg[e, run_ok] + 1).float()[:, None],
                           ((lens[run_ok] + h).clamp(max=T).float() / T)[:, None], env.run_x[e, run_ok, :d]], 1)
        q_shop = torch.cat([(env.shop_ids[e, shop_ok] + 1).float()[:, None],
                            torch.full((len(shop_ok), 1), min(h, T) / T), env.shop_x[e, shop_ok, :d]], 1)
        x_test = torch.cat([q_run, q_shop])
        logits = m(x_train=x_train, y_train=y_train, x_test=x_test)
        pi = m.model.criterion.pi(logits, best_f=torch.tensor(thr)).reshape(-1)
        j = int(pi.argmax())
        actions[e] = int(run_ok[j]) if j < len(run_ok) else env.R + int(shop_ok[j - len(run_ok)])
    return actions


def run_ifbo_job(args):
    task, seed, budget = args
    torch.set_num_threads(1)
    t0 = time.time()
    env = TabEnv(task, 1, budget, n_shop=None, device="cpu", seed=seed)
    rng = np.random.default_rng(seed)
    while not env.done.all():
        env.step(ifbo_step(env, rng))
    return task.name, seed, env.normalised_regret()[0].numpy(), time.time() - t0, int(env.n_runs[0])


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["rd", "ifbo", "random_search"])
    ap.add_argument("--family", required=True, choices=["taskset", "pd1", "lcbench"])
    ap.add_argument("--ckpt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--n-shop", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--shop-mode", default="uniform", choices=["uniform", "mixed"])
    ap.add_argument("--h-start", type=float, default=None)
    ap.add_argument("--h-end", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tasks = load_family(args.family)
    if args.tasks:
        tasks = [t for t in tasks if t.name in args.tasks]
    assert tasks, f"no tasks for {args.family}"
    out_dir = ROOT / "runs" / "eval" / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.method
    regret = np.zeros((len(tasks), args.seeds, args.budget), dtype=np.float32)
    wall = np.zeros(len(tasks))
    n_runs = np.zeros((len(tasks), args.seeds))

    if args.method == "ifbo":
        import multiprocessing as mp
        jobs = [(t, s, args.budget) for t in tasks for s in range(args.seeds)]
        name_idx = {t.name: i for i, t in enumerate(tasks)}
        with mp.get_context("fork").Pool(args.workers) as pool:
            for k, (nm, s, r, dt, nr) in enumerate(pool.imap_unordered(run_ifbo_job, jobs)):
                regret[name_idx[nm], s] = r
                n_runs[name_idx[nm], s] = nr
                wall[name_idx[nm]] += dt / args.seeds
                print(f"[{k + 1}/{len(jobs)}] {nm} seed {s}: final {r[-1]:.4f} ({dt:.0f}s)", flush=True)
    else:
        model, h = None, (1.0, 1.0)
        if args.method == "rd":
            from .model import load_policy
            ck = torch.load(args.ckpt, map_location=args.device)
            model = load_policy(ck, args.device)
            tc = ck["cfg"]["train"]
            h = (args.h_start if args.h_start is not None else tc["h_start"],
                 args.h_end if args.h_end is not None else tc["h_end"])
        for i, task in enumerate(tasks):
            t0 = time.time()
            env = TabEnv(task, args.seeds, args.budget, n_shop=(args.n_shop or None), device=args.device, seed=0,
                         shop_mode=args.shop_mode)
            with torch.no_grad():
                while not env.done.all():
                    if args.method == "rd":
                        a = policy_step(model, env, args.temperature, args.greedy, h=h,
                                        amp=args.device.startswith("cuda"))
                    else:
                        a = random_search_step(env)
                    env.step(a)
            regret[i] = env.normalised_regret().cpu().numpy()
            n_runs[i] = env.n_runs.cpu().numpy()
            wall[i] = (time.time() - t0) / args.seeds
            print(f"{task.name}: final mean regret {regret[i, :, -1].mean():.4f} runs {n_runs[i].mean():.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    np.savez(out_dir / f"{tag}.npz", regret=regret, tasks=np.array([t.name for t in tasks]), wall=wall, n_runs=n_runs,
             args=str(vars(args)))
    print(f"saved {out_dir / (tag + '.npz')}; mean final regret {regret[:, :, -1].mean():.4f}")


if __name__ == "__main__":
    main()

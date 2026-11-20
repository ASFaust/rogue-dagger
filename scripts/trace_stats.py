"""Behaviour statistics of HPO episodes from evaluate.py's per-step config ids (step_cfg).

    python scripts/trace_stats.py lcbench trace-ifbo trace-wfocus03-n0-t0.25 [--per-task]

Per method, averaged over tasks and eval seeds:
  runs        configs started
  full        runs trained to max fidelity
  steps1      runs that got only 1 step
  top1/top10  P(table-best config started) / top-10 configs (by best value) started, of 10
  fid_top10   mean fidelity fraction reached by the started top-10 configs
  best_fid    fidelity fraction reached by the best config that was started (by table best value)
  t_start*    step at which the eventual best-started config was started
  gap_found   normalised regret of the best config started, if trained fully (selection error)
  gap_train   final regret minus gap_found (training error: not trained far enough)
  late_new    share of new-run starts in the last half of the budget
"""
import argparse
from pathlib import Path

import numpy as np

from roguedagger.bench import load_family

ROOT = Path(__file__).resolve().parent.parent


def episode_stats(sc, Y, lo, hi):
    T = Y.shape[1]
    best = Y.min(1)
    order = np.argsort(best)
    top10 = set(order[:10].tolist())
    cfgs, first = np.unique(sc[sc >= 0], return_index=True)
    steps = np.array([(sc == c).sum() for c in cfgs])
    fid = np.minimum(steps, T) / T
    started_best = cfgs[np.argmin(best[cfgs])]
    nb = lambda v: (v - lo) / (hi - lo)
    reached = np.array([Y[c, :min((sc == c).sum(), T)].min() for c in cfgs])
    final = nb(reached.min())
    found = nb(best[started_best])
    in_top = np.array([c in top10 for c in cfgs])
    half = len(sc) // 2
    return dict(runs=len(cfgs), full=(steps >= T).sum(), steps1=(steps == 1).sum(),
                top1=float(order[0] in set(cfgs.tolist())), top10=in_top.sum(),
                fid_top10=fid[in_top].mean() if in_top.any() else np.nan,
                best_fid=fid[cfgs == started_best][0], t_start=first[cfgs == started_best][0],
                gap_found=found, gap_train=final - found,
                late_new=(first >= half).sum() / len(cfgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--per-task", action="store_true")
    a = ap.parse_args()
    tasks = {t.name: t for t in load_family(a.family)}
    out = {}
    for tag in a.tags:
        z = np.load(ROOT / "runs" / "eval" / a.family / f"{tag}.npz")
        per = {}
        for i, name in enumerate(map(str, z["tasks"])):
            Y = np.asarray(tasks[name].Y, float)
            lo, hi = Y.min(), Y.max()
            per[name] = [episode_stats(z["step_cfg"][i, s], Y, lo, hi) for s in range(z["step_cfg"].shape[1])]
        out[tag] = per
    keys = list(next(iter(next(iter(out.values())).values()))[0].keys())
    common = sorted(set.intersection(*[set(p) for p in out.values()]))
    print(f"{a.family}: {len(common)} tasks")
    print("tag".ljust(30) + "".join(k.rjust(10) for k in keys))
    for tag, per in out.items():
        m = {k: np.nanmean([np.mean([e[k] for e in per[t]]) for t in common]) for k in keys}
        print(tag[:30].ljust(30) + "".join(f"{m[k]:10.4g}" for k in keys))
    if a.per_task:
        for t in common:
            print(t)
            for tag, per in out.items():
                print("  " + tag[:28].ljust(28) + "".join(f"{np.nanmean([e[k] for e in per[t]]):10.4g}" for k in keys))


if __name__ == "__main__":
    main()

"""Scorer pair accuracy on real states vs matched synthetic twins (docs/future-ideas/prior-sim2real.md, measurement 2).

Roll out a frozen scorer with its own policy (shop 32, budget 1000) on the real benchmark tasks and on matched prior
twins (same d, T, budget, shop, capped runs). Every `--every` steps, score all choosable items and compare every pair
with different labels (label = best loss up to len + H, past included, exactly as in training). Accumulate pairwise
accuracy and BCE per family, pair type (run-run, run-shop, shop-shop) and budget progress (thirds).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/pair_gap.py runs/wide-focus03-v1/last.pt --tau 1.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from roguedagger.bench import TabEnv, load_family
from roguedagger.env import ShopEnv
from roguedagger.features import encode, from_env
from roguedagger.model import load_policy, sample_action
from roguedagger.train import env_config, to_env_action

ROOT = Path(__file__).resolve().parent.parent
FAMS = ("taskset", "pd1", "lcbench")
TYPES = ("run-run", "run-shop", "shop-shop")


def real_values(env, H, Ycm):
    """Labels for a TabEnv: min of the table curve over [0, len + H) for runs, [0, H) for shop items (capped at T)."""
    T = env.T[:, None]
    lim_r = torch.minimum(env.run_len + H[:, None], T).clamp(min=1)
    vr = Ycm[env.run_cfg, (lim_r - 1).clamp(max=Ycm.shape[1] - 1)]
    lim_s = torch.minimum(H[:, None].expand(-1, env.N), T).clamp(min=1)
    vs = Ycm[env.shop_ids, lim_s - 1]
    return torch.cat([vr, vs], 1)


class Acc:
    def __init__(self, dev):
        self.n = torch.zeros(len(FAMS), 3, 3, device=dev, dtype=torch.float64)   # family, type, progress third
        self.c = torch.zeros_like(self.n)
        self.bce = torch.zeros_like(self.n)

    def add(self, fam_idx, s, v, valid, is_shop, prog):
        """fam_idx [E]; s, v, valid, is_shop [E, K]; prog [E] in [0, 1]."""
        ds = s[:, :, None] - s[:, None, :]                 # s_i - s_j
        dv = v[:, :, None] - v[:, None, :]                 # v_i - v_j (lower is better)
        ok = valid[:, :, None] & valid[:, None, :] & (dv.abs() > 1e-9)
        ok = ok & torch.triu(torch.ones_like(ok[0]), 1)[None]
        y = (dv < 0).float()                               # 1 if i better
        correct = ((ds > 0).float() == y).double()
        bce = F.binary_cross_entropy_with_logits(ds, y, reduction="none").double()
        typ = is_shop[:, :, None].long() + is_shop[:, None, :].long()   # 0 rr, 1 rs, 2 ss
        third = (prog * 3).long().clamp(max=2)
        for t in range(3):
            m = (ok & (typ == t)).double()
            idx = fam_idx * 9 + t * 3 + third
            self.n.view(-1).index_add_(0, idx, m.sum((1, 2)))
            self.c.view(-1).index_add_(0, idx, (m * correct).sum((1, 2)))
            self.bce.view(-1).index_add_(0, idx, (m * bce).sum((1, 2)))


def score(model, env, h, amp=True):
    H = env.horizon(*h)
    obs = from_env(env, H)
    r = obs.run_len.shape[1]
    tok, pad = encode(obs, env.d_max)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        s = model(tok, pad).float()
    valid = torch.cat([env.choosable()[:, :r], env.choosable()[:, env.R:]], 1)
    return H, r, s, valid


def rollout(model, env, h, tau, every, label_fn, fam_idx, acc):
    step = 0
    while True:
        alive = env.consumed < env.B
        if not alive.any():
            break
        H, r, s, valid = score(model, env, h)
        if step % every == every - 1:
            v = label_fn(env, H, r)
            is_shop = (torch.arange(valid.shape[1], device=s.device) >= r)[None].expand_as(valid)
            prog = env.consumed.float() / env.B.float()
            acc.add(fam_idx, s, v, valid & alive[:, None], is_shop, prog)
        a, _ = sample_action(s, valid, tau)
        out = env.step(to_env_action(a, r, env.R))
        step += 1
        if isinstance(env, ShopEnv) and out["done"].any():   # ShopEnv resets finished rows: all rows end together
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--envs", type=int, default=8, help="episodes per real task (and twins per real task)")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location=dev)
    model = load_policy(ck, dev)
    tc = ck["cfg"]["train"]
    h = (tc["h_start"], tc["h_end"])
    torch.manual_seed(0)
    real, twin = Acc(dev), Acc(dev)

    # real tasks
    d_all, T_all, f_all = [], [], []
    for fi, fam in enumerate(FAMS):
        for task in load_family(fam):
            env = TabEnv(task, a.envs, 1000, n_shop=32, device=dev, seed=0)
            Ycm = torch.cummin(env.Yall, 1).values

            def lab(env, H, r, Ycm=Ycm):
                v = real_values(env, H, Ycm)
                return torch.cat([v[:, :r], v[:, env.R:]], 1)
            rollout(model, env, h, a.tau, a.every, lab, torch.full((a.envs,), fi, device=dev), real)
            d_all += [task.X.shape[1]] * a.envs
            T_all += [task.Y.shape[1]] * a.envs
            f_all += [fi] * a.envs
        print(fam, "real done", flush=True)

    # twins, one batch
    ecfg = env_config(ck["cfg"])
    ecfg.R_max, ecfg.shop_mode = 1000, "uniform"
    spec = {"d": torch.tensor(d_all, device=dev), "T": torch.tensor(T_all, device=dev), "B": 1000, "n": 32,
            "cap": True}
    env = ShopEnv(len(d_all), ecfg, dev, seed=0, fixed_spec=spec)

    def lab_tw(env, H, r):
        v = env.values(H)
        return torch.cat([v[:, :r], v[:, env.R:]], 1)
    rollout(model, env, h, a.tau, a.every, lab_tw, torch.tensor(f_all, device=dev), twin)

    res = {}
    print(f"{a.ckpt}, tau {a.tau}: pair accuracy (BCE), real vs twin")
    print("family   type        progress   real acc  twin acc    real bce  twin bce   real n     twin n")
    for fi, fam in enumerate(FAMS):
        for t, tn in enumerate(TYPES):
            for p, pn in enumerate(("early", "mid", "late", "all")):
                sl = (fi, t, slice(None)) if pn == "all" else (fi, t, p)
                nr, nt = real.n[sl].sum().item(), twin.n[sl].sum().item()
                if nr == 0 or nt == 0:
                    continue
                ar, at = real.c[sl].sum().item() / nr, twin.c[sl].sum().item() / nt
                br, bt = real.bce[sl].sum().item() / nr, twin.bce[sl].sum().item() / nt
                res[f"{fam}/{tn}/{pn}"] = dict(real_acc=ar, twin_acc=at, real_bce=br, twin_bce=bt, real_n=nr,
                                               twin_n=nt)
                print(f"{fam:8} {tn:10}  {pn:8} {ar:9.3f} {at:9.3f}   {br:9.3f} {bt:9.3f} {nr:9.2e} {nt:9.2e}")
        nr, nt = real.n[fi].sum().item(), twin.n[fi].sum().item()
        ar, at = real.c[fi].sum().item() / nr, twin.c[fi].sum().item() / nt
        print(f"{fam:8} {'all':10}  {'all':8} {ar:9.3f} {at:9.3f}")
        res[f"{fam}/all/all"] = dict(real_acc=ar, twin_acc=at)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

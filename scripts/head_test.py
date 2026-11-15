"""Test 3 of docs/future-ideas/implicit-distribution.md: how much does the Bradley-Terry readout cost?

Freeze a trained scorer, roll it out on the synthetic prior (its own policy, as in training), and keep snapshots of
the final per-item embeddings (the input of the score head) with exact labels. Then fit fresh heads on the frozen
embeddings and compare pairwise BCE on held-out env rows:
  own        the checkpoint's own head (no refit)
  bt         fresh MLP -> s;          P(i beats j) = sigmoid(s_i - s_j)
  thurstone  fresh MLP -> (mu, sigma); P(i beats j) = Phi((mu_i - mu_j) / sqrt(sigma_i^2 + sigma_j^2))
If thurstone doesn't beat bt on held-out pairs, the extra freedom isn't needed (gate in thurstone-thompson.md).

    CUDA_VISIBLE_DEVICES=3 python scripts/head_test.py runs/wide-focus03-v1/last.pt
"""
import argparse
import copy
import math

import torch
from torch import nn

from roguedagger.env import ShopEnv
from roguedagger.features import encode, from_env
from roguedagger.model import load_policy, sample_action
from roguedagger.train import env_config, load_config, to_env_action


def collect(model, cfg, args, dev):
    ecfg = env_config(cfg)
    env = ShopEnv(args.envs, ecfg, dev, seed=args.seed)
    K, d = args.items, model.norm.normalized_shape[0]
    S = args.snaps * args.envs
    emb = torch.zeros(S, K, d, dtype=torch.float16, device=dev)
    val = torch.zeros(S, K, device=dev)
    ok = torch.zeros(S, K, dtype=torch.bool, device=dev)
    cont = torch.zeros(S, K, dtype=torch.bool, device=dev)
    shop = torch.zeros(S, K, dtype=torch.bool, device=dev)
    prog = torch.zeros(S, K, device=dev)           # run_len / T for runs, 0 for shop items
    row = torch.arange(args.envs, device=dev).repeat(args.snaps)
    feats = {}
    model.norm.register_forward_hook(lambda m, i, o: feats.__setitem__("h", o))
    s_i = 0
    for it in range(args.warmup + args.snaps * args.every):
        H = env.horizon()
        obs = from_env(env, H)
        r = obs.run_len.shape[1]
        tok, pad = encode(obs, env.d_max)
        ch = env.choosable()
        valid = torch.cat([ch[:, :r], ch[:, env.R:]], 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            s = model(tok, pad).float()
        if it >= args.warmup and (it - args.warmup) % args.every == 0:
            h = feats["h"]
            v = env.values(H)
            v = torch.cat([v[:, :r], v[:, env.R:]], 1)
            c = v < env.best[:, None] - 1e-6
            is_shop = torch.arange(valid.shape[1], device=dev)[None] >= r
            pr = torch.cat([env.run_len[:, :r].float() / env.T[:, None].float(),
                            torch.zeros(args.envs, env.N, device=dev)], 1)
            # keep up to K valid items per env, chosen at random
            key = torch.where(valid, torch.rand(valid.shape, device=dev), torch.full(valid.shape, 2.0, device=dev))
            idx = key.argsort(1)[:, :K]
            g = lambda x: x.expand(args.envs, -1).gather(1, idx)
            sl = slice(s_i, s_i + args.envs)
            emb[sl] = h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1])).half()
            val[sl], ok[sl], cont[sl], shop[sl], prog[sl] = g(v), g(valid), g(c), g(is_shop), g(pr)
            s_i += args.envs
        a, _ = sample_action(s, valid, args.tau)
        env.step(to_env_action(a, r, env.R))
    return dict(emb=emb, val=val, ok=ok, cont=cont, shop=shop, prog=prog, row=row)


class Head(nn.Module):
    def __init__(self, d, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))

    def forward(self, x):
        return self.net(x.float())


def pair_logp(kind, out):
    """-> log P(i beats j), log P(j beats i), each [S, K, K]."""
    if kind in ("bt", "own"):
        s = out[..., 0]
        z = s[:, :, None] - s[:, None, :]
        return nn.functional.logsigmoid(z), nn.functional.logsigmoid(-z)
    mu, sig = out[..., 0], nn.functional.softplus(out[..., 1]) + 1e-2
    z = (mu[:, :, None] - mu[:, None, :]) / (sig[:, :, None] ** 2 + sig[:, None, :] ** 2).sqrt()
    return torch.special.log_ndtr(z), torch.special.log_ndtr(-z)


def pair_stats(kind, out, D, pair_low):
    """Weighted and unweighted mean BCE and accuracy over labelled pairs."""
    dv = D["val"][:, :, None] - D["val"][:, None, :]
    m = D["ok"][:, :, None] & D["ok"][:, None, :] & (dv.abs() > 1e-6)
    lp, lq = pair_logp(kind, out)
    bce = -torch.where(dv < 0, lp, lq)
    w = torch.where(D["cont"][:, :, None] | D["cont"][:, None, :], 1.0, pair_low) * m
    acc = ((lp > lq) == (dv < 0)).float()
    mf = m.float()
    return {"bce_w": (bce * w).sum() / w.sum(), "bce": (bce * mf).sum() / mf.sum(),
            "acc": (acc * mf).sum() / mf.sum(), "n": mf.sum()}


def split(D, mask):
    return {k: v[mask] for k, v in D.items()}


def fit(kind, Dtr, Dva, d, args, dev, init=None):
    """Train on Dtr; keep the weights with the best validation bce_w (early stopping)."""
    head = Head(d, 1 if kind == "bt" else 2).to(dev)
    if init is not None and kind == "bt":
        head.net.load_state_dict(init.state_dict())
    elif init is not None:
        # mu = the trained BT score; constant sigma with Phi(z / (sigma sqrt 2)) ~ sigmoid(z): sigma sqrt 2 = 1.702, sigma ~ 1.20
        sd = init.state_dict()
        with torch.no_grad():
            head.net[0].load_state_dict(init[0].state_dict())
            head.net[2].weight.zero_()
            head.net[2].weight[0] = sd["2.weight"][0]
            head.net[2].bias[0] = sd["2.bias"][0]
            head.net[2].bias[1] = 0.827      # softplus(0.827) + 0.01 = 1.20
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    S = Dtr["emb"].shape[0]
    g = torch.Generator(device=dev).manual_seed(0)
    best = (evaluate(kind, head, Dva, args)["bce_w"], 0, copy.deepcopy(head.state_dict()))
    for step in range(args.steps):
        b = torch.randint(0, S, (args.batch,), device=dev, generator=g)
        loss = pair_stats(kind, head(Dtr["emb"][b]), {k: v[b] for k, v in Dtr.items()}, args.pair_low)["bce_w"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            head.eval()
            v = evaluate(kind, head, Dva, args)["bce_w"]
            head.train()
            if v < best[0]:
                best = (v, step + 1, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[2])
    print(f"  {kind}{' (init own)' if init is not None else ''}: best val bce_w {best[0]:.4f} at step {best[1]}")
    return head


@torch.no_grad()
def evaluate(kind, head, D, args):
    acc = {}
    for b in torch.arange(D["emb"].shape[0], device=D["emb"].device).split(512):
        st = pair_stats(kind, head(D["emb"][b]), {k: v[b] for k, v in D.items()}, args.pair_low)
        for k in ("bce_w", "bce", "acc"):
            acc.setdefault(k, []).append((st[k] * st["n"]).item())
        acc.setdefault("n", []).append(st["n"].item())
    n = sum(acc["n"])
    return {k: sum(acc[k]) / n for k in ("bce_w", "bce", "acc")}   # bce_w: pair-count weighted mean of batch means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--snaps", type=int, default=240)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--items", type=int, default=48)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pair_low", type=float, default=0.3)
    ap.add_argument("--probe", action="store_true", help="also run the linear probes (test 1)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = load_policy(ck, dev)
    cfg = load_config(None, [])
    cfg["env"] = ck["cfg"]["env"]
    torch.manual_seed(args.seed)
    D = collect(model, cfg, args, dev)
    te = D["row"] % 4 == 3                          # held-out env rows (independent task streams)
    va = D["row"] % 4 == 2                          # validation rows for early stopping
    Dtr, Dva, Dte = split(D, ~te & ~va), split(D, va), split(D, te)
    print(f"{args.ckpt}: {Dtr['emb'].shape[0]} train / {Dte['emb'].shape[0]} test states, "
          f"{int(D['ok'].sum(1).float().mean())} items per state")
    d = D["emb"].shape[-1]
    if model.readout != "token":
        print("note: mlp readout; own head needs global features, skipped")
    res = {}
    if model.readout == "token":
        own = Head(d, 1).to(dev)
        own.net.load_state_dict(model.head.state_dict())
        res["own"] = (own, "own")
    res["bt"] = (fit("bt", Dtr, Dva, d, args, dev), "bt")
    res["bt_ft"] = (fit("bt", Dtr, Dva, d, args, dev, init=model.head) if model.readout == "token" else None, "bt")
    res["thurstone"] = (fit("thurstone", Dtr, Dva, d, args, dev), "thurstone")
    if model.readout == "token":
        res["thurst_ft"] = (fit("thurstone", Dtr, Dva, d, args, dev, init=model.head), "thurstone")
    print(f"{'head':12s}{'train bce_w':>12s}{'test bce_w':>12s}{'test bce':>10s}{'test acc':>10s}")
    for name, (head, kind) in res.items():
        if head is None:
            continue
        a, b = evaluate(kind, head, Dtr, args), evaluate(kind, head, Dte, args)
        print(f"{name:12s}{a['bce_w']:12.4f}{b['bce_w']:12.4f}{b['bce']:10.4f}{b['acc']:10.4f}")
    # sigma sanity: larger for shop items and short runs, smaller for long runs
    head = res["thurst_ft" if "thurst_ft" in res else "thurstone"][0]
    with torch.no_grad():
        out = torch.cat([head(Dte["emb"][b]) for b in torch.arange(Dte["emb"].shape[0], device=dev).split(512)])
    sig = nn.functional.softplus(out[..., 1]) + 1e-2
    sig = sig / sig[Dte["ok"]].mean()                # scale gauge: relative to the mean
    ok = Dte["ok"]
    print("relative sigma: shop %.3f" % sig[ok & Dte["shop"]].mean().item(), end="")
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1e9)):
        mk = ok & ~Dte["shop"] & (Dte["prog"] >= lo) & (Dte["prog"] < hi)
        if mk.any():
            print(f" | runs len/T [{lo},{hi}) {sig[mk].mean().item():.3f} (n={int(mk.sum())})", end="")
    print()
    if args.probe:
        probe(Dtr, Dte)


def ridge(X, y, lam=1e-2):
    X = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1).double()
    A = X.T @ X + lam * len(X) * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    w = torch.linalg.solve(A, X.T @ y.double())
    return lambda Z: (torch.cat([Z, torch.ones(len(Z), 1, device=Z.device)], 1).double() @ w).float()


def probe(Dtr, Dte):
    """Test 1: linear probes from frozen embeddings to (i) the label (future best loss, z-scored within each state,
    so task scale drops out) and (ii) the squared error of probe (i): a heteroscedastic spread probe. Baseline: the
    same probes on item-type features only (shop flag, run progress bins)."""
    def flat(D, sub=400000):
        ok = D["ok"]
        v = torch.where(ok, D["val"], torch.nan)
        mu, sd = v.nanmean(1, keepdim=True), (v - v.nanmean(1, keepdim=True)).pow(2).nanmean(1, keepdim=True).sqrt()
        z = ((D["val"] - mu) / sd.clamp(min=1e-6)).clamp(-5, 5)
        keep = ok & (sd > 1e-6)
        X, y = D["emb"][keep].float(), z[keep]
        p = D["prog"][keep]
        B = torch.stack([D["shop"][keep].float()] + [((p >= lo) & (p < hi) & ~D["shop"][keep]).float()
                        for lo, hi in ((0, .1), (.1, .3), (.3, .7), (.7, 1e9))], 1)
        g = torch.Generator(device=X.device).manual_seed(0)
        i = torch.randperm(len(X), device=X.device, generator=g)[:sub]
        return X[i], B[i], y[i]
    Xa, Ba, ya = flat(Dtr)
    Xb, Bb, yb = flat(Dte)
    r2 = lambda y, p: 1 - ((y - p) ** 2).mean().item() / y.var().item()
    for name, A, Bt in (("embedding", Xa, Xb), ("item type", Ba, Bb)):
        f = ridge(A, ya)
        ea, eb = (ya - f(A)) ** 2, (yb - f(Bt)) ** 2
        g = ridge(A, ea)
        print(f"probe [{name:9s}] label R2 test {r2(yb, f(Bt)):.3f} | spread (sq. error) R2 test {r2(eb, g(Bt)):.3f}"
              f" | corr(pred spread, sq. error) {torch.corrcoef(torch.stack([g(Bt), eb]))[0, 1].item():.3f}")


if __name__ == "__main__":
    main()

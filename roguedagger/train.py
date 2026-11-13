"""DAgger-like training: the scorer's own multinomial policy drives E parallel envs.

Every iteration:
  1. encode the state of all E envs, score every run and suggestion (with grad)
  2. pairwise BCE against exact labels from the full curves (horizon H)
  3. one optimiser step
  4. sample actions from softmax(scores / tau) (detached), step the envs; finished
     envs reset immediately with a fresh task and spec

There is no replay buffer: one env per batch row, used once per step.

    python -m roguedagger.train --config configs/base.yaml --set train.iters=1000 wandb.mode=disabled
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml

from .env import EnvConfig, ShopEnv
from .features import encode, from_env, global_at_horizons
from .model import Scorer, pairwise_loss, sample_action

DEFAULTS = {
    "seed": 0,
    "name": None,
    "env": {},                       # EnvConfig overrides
    "model": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "readout": "token"},
    "train": {
        "n_envs": 256, "iters": 100_000, "lr": 3e-4, "wd": 0.01, "warmup": 1000,
        "grad_clip": 1.0, "temperature": 1.0, "h_start": 1.0, "h_end": 1.0,
        "amp": True, "label_past": True, "pair_low": 1.0, "n_horizons": 1, "log_every": 50, "ckpt_every": 5000, "eval_every": 5000,
    },
    "eval": {"n_envs": 128, "seed": 1234, "T": 50, "B": 1000, "n": 32, "cap": True,
             "temperature": 1.0},
    "wandb": {"project": "rogue-dagger", "mode": "online"},
}


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path, overrides):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        deep_update(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def env_config(cfg):
    ec = EnvConfig()
    for k, v in cfg.get("env", {}).items():
        setattr(ec, k, tuple(v) if isinstance(v, list) else v)
    return ec


def to_env_action(a, r_cur, R_max):
    """Map an index into [runs[:r_cur] | shop] onto the env's [R_max | shop] layout."""
    return torch.where(a < r_cur, a, a - r_cur + R_max)


# ---------------------------------------------------------------------------- policies
def policy_step(model, env, temperature=1.0, greedy=False, h=(1.0, 1.0), amp=True):
    H = env.horizon(*h)
    obs = from_env(env, H)
    r_cur = obs.run_len.shape[1]
    tok, pad = encode(obs, env.d_max)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        s = model(tok, pad)
    valid = torch.cat([env.choosable()[:, :r_cur], env.choosable()[:, env.R:]], 1)
    a, _ = sample_action(s.float(), valid, temperature, greedy)
    return to_env_action(a, r_cur, env.R)


def uniform_step(env):
    valid = env.choosable()
    a = torch.multinomial(valid.float(), 1)[:, 0]
    return a


def random_search_step(env):
    """Train the newest run to full fidelity T, then start a random config from the shop."""
    newest = (env.n_runs - 1).clamp(min=0)
    ln = env.run_len.gather(1, newest[:, None])[:, 0]
    cont = ln < env.T
    shop_ok = env.choosable()[:, env.R:].float()
    pick = torch.multinomial(shop_ok + (shop_ok.sum(1, keepdim=True) == 0).float(), 1)[:, 0]
    return torch.where(cont, newest, env.R + pick)


@torch.no_grad()
def synthetic_eval(policy_fn, cfg, device, ecfg):
    """Fixed tasks (seeded), one episode per env. -> dict of mean metrics."""
    ev = cfg["eval"]
    spec = {"T": ev["T"], "B": ev["B"], "n": ev["n"], "cap": ev["cap"]}
    env = ShopEnv(ev["n_envs"], ecfg, device, seed=ev["seed"], fixed_spec=spec)
    finished = torch.zeros(env.E, dtype=torch.bool, device=device)
    final = torch.zeros(env.E, device=device)
    oracle = torch.zeros(env.E, device=device)
    nruns = torch.zeros(env.E, device=device)
    while not finished.all():
        out = env.step(policy_fn(env))
        d = out["done"] & ~finished
        if d.any():
            idx = out["done"].nonzero()[:, 0]
            keep = ~finished[idx]
            sel = idx[keep]
            final[sel] = out["final_best"][keep]
            oracle[sel] = out["oracle"][keep]
            nruns[sel] = out["n_runs"][keep].float()
            finished |= d
    return {"final_best": final.mean().item(), "regret": (final - oracle).mean().item(),
            "n_runs": nruns.mean().item(), "final_per_env": final.cpu()}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    ecfg = env_config(cfg)
    name = cfg["name"] or time.strftime("rd-%Y%m%d-%H%M%S")
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    import wandb
    run = wandb.init(project=cfg["wandb"]["project"], name=name, config=cfg, mode=cfg["wandb"]["mode"],
                     dir=str(out_dir))

    model = Scorer(d_max=10, **cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"scorer params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["wd"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(i, tc["iters"]) / tc["iters"])))

    # baselines on the fixed eval tasks, once
    base = {}
    if tc["eval_every"]:
        for nm, fn in (("uniform", uniform_step), ("random_search", random_search_step)):
            r = synthetic_eval(fn, cfg, device, ecfg)
            base[nm] = r
            print(f"eval baseline {nm}: final {r['final_best']:.4f} regret {r['regret']:.4f}", flush=True)
            wandb.log({f"eval/{nm}_final": r["final_best"], f"eval/{nm}_regret": r["regret"]}, step=0)

    env = ShopEnv(tc["n_envs"], ecfg, device, seed=cfg["seed"])
    ep_stats, t0, hist = [], time.time(), []
    h = (tc["h_start"], tc["h_end"])
    for it in range(1, tc["iters"] + 1):
        model.train()
        H = env.horizon(*h)
        obs = from_env(env, H)
        r_cur = obs.run_len.shape[1]
        tok, pad = encode(obs, env.d_max)
        ch = env.choosable()
        valid = torch.cat([ch[:, :r_cur], ch[:, env.R:]], 1)
        def labels(Hk):
            v = env.values(Hk, include_past=tc["label_past"])
            return torch.cat([v[:, :r_cur], v[:, env.R:]], 1)

        def weights(v):
            if tc["pair_low"] >= 1.0:
                return None
            # contenders: items whose label value beats the current incumbent. Pairs with at
            # least one contender keep weight 1, pairs of two non-contenders get pair_low.
            cont = v < env.best[:, None] - 1e-6
            return torch.where(cont[:, :, None] | cont[:, None, :], 1.0, tc["pair_low"])

        K = tc["n_horizons"]
        if K > 1:
            # horizon 0 = the reported H (the policy acts on it); the others are log-uniform in
            # [1, remaining budget]. Labels are exact at any horizon; the trunk runs once.
            rem = (env.B - env.consumed).clamp(min=1).float()
            u = torch.rand(env.E, K - 1, device=device)
            Hs = torch.cat([H[:, None], rem[:, None].pow(u).round().long().clamp(min=1)], 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["amp"]):
                sk = model(tok, pad, glob=global_at_horizons(tok, Hs, env.B, env.T)).float()
            losses, accs, nps = zip(*[pairwise_loss(sk[:, k], v, valid, weight=weights(v))
                                      for k in range(K) for v in [labels(Hs[:, k])]])
            loss = torch.stack(losses).mean()
            acc, npair = accs[0], nps[0]          # logged at the reported horizon
            s = sk[:, 0]
        else:
            vals = labels(H)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc["amp"]):
                s = model(tok, pad)
            loss, acc, npair = pairwise_loss(s.float(), vals, valid, weight=weights(vals))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()

        with torch.no_grad():
            a, ent = sample_action(s.detach().float(), valid, tc["temperature"])
            is_new = a >= r_cur
            out = env.step(to_env_action(a, r_cur, env.R))
        if out["done"].any():
            ep_stats.append({k: out[k].float().cpu() for k in ("final_best", "oracle", "n_runs", "n_new", "B", "T", "n")})
        hist.append(torch.stack([loss.detach(), acc, npair, ent.mean(), is_new.float().mean(), gn,
                                 torch.tensor(float(r_cur), device=device)]))

        if it % tc["log_every"] == 0:
            hm = torch.stack(hist).mean(0).tolist()
            hist = []
            dt = time.time() - t0
            t0 = time.time()
            log = {"train/loss": hm[0], "train/pair_acc": hm[1], "train/pairs": hm[2], "train/entropy": hm[3],
                   "train/frac_new": hm[4], "train/grad_norm": hm[5], "train/r_cur": hm[6],
                   "train/lr": sched.get_last_lr()[0], "perf/it_per_s": tc["log_every"] / dt}
            if ep_stats:
                cat = {k: torch.cat([e[k] for e in ep_stats]) for k in ep_stats[0]}
                log.update({"episode/final_best": cat["final_best"].mean().item(),
                            "episode/regret": (cat["final_best"] - cat["oracle"]).mean().item(),
                            "episode/runs_per_full_budget": (cat["n_runs"] / (cat["B"] / cat["T"])).mean().item(),
                            "episode/count": len(cat["final_best"])})
                ep_stats = []
            wandb.log(log, step=it)
            print(f"it {it} " + " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log.items()), flush=True)

        if tc["eval_every"] and it % tc["eval_every"] == 0:
            model.eval()
            r = synthetic_eval(lambda e: policy_step(model, e, cfg["eval"]["temperature"], h=h, amp=tc["amp"]),
                               cfg, device, ecfg)
            rg = synthetic_eval(lambda e: policy_step(model, e, greedy=True, h=h, amp=tc["amp"]), cfg, device, ecfg)
            log = {"eval/policy_final": r["final_best"], "eval/policy_regret": r["regret"],
                   "eval/policy_n_runs": r["n_runs"], "eval/greedy_final": rg["final_best"],
                   "eval/greedy_regret": rg["regret"]}
            for nm, b in base.items():
                log[f"eval/win_vs_{nm}"] = (r["final_per_env"] < b["final_per_env"]).float().mean().item()
                log[f"eval/delta_vs_{nm}"] = (b["final_per_env"] - r["final_per_env"]).mean().item()
            wandb.log(log, step=it)
            print(f"EVAL it {it} " + json.dumps({k: round(v, 4) for k, v in log.items()}), flush=True)

        if it % tc["ckpt_every"] == 0 or it == tc["iters"]:
            torch.save({"model": model.state_dict(), "cfg": cfg, "it": it}, out_dir / "last.pt")
            if it % (tc["ckpt_every"] * 4) == 0:
                torch.save({"model": model.state_dict(), "cfg": cfg, "it": it}, out_dir / f"it{it}.pt")
    run.finish()


if __name__ == "__main__":
    main()

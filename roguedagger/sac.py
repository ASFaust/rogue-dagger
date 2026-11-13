"""Implicit-actor discrete SAC on a frozen pretrained trunk: the recipe of the automl-exam's
deployed policy (`sac_frozen3`, automl-exam/src/ftrl/frozen.py), ported to the shop env.

The pairwise-trained scorer (train.py) provides the trunk. It is frozen, its per-token
embeddings are computed once per env step, and the replay buffer stores those embeddings.
Only two small Q heads (+ the temperature) learn. Kept from the exam recipe, quirks included:

  * implicit actor: pi(a | s) = softmax(beta * min(Q1, Q2)) over the valid actions;
    there is no actor network and no policy loss
  * inverse temperature: beta = exp(log_beta) is the trained parameter (alpha = 1 / beta)
  * entropy target in nats, per state min(log P*, log |A_s|), with the target perplexity
    P* annealed linearly in nats from perplexity_start to perplexity_end over the first
    perplexity_anneal_frac of training, then held. States with |A_s| = 1 are skipped
  * mean-reward Bellman target y = (1 - gamma) r + gamma (1 - done) V(s'),
    V(s') = alpha * logsumexp(min(Qt1, Qt2)(s') / alpha), clamped to the range of every
    reward ever pushed (Q is a convex combination of rewards, so that range bounds it)
  * twin independent Q heads (MLP 64, SiLU) on [token embedding | global features], Huber
    loss, Adam, polyak 0.005 on the heads, critic LR warmup + cosine to 0.1x, grad clip 1
  * reward = log normalised-regret decrease x reward_scale (100), gamma 0.99
  * batch 512, 8 updates per env step of 128 envs (replay ratio 32), buffer 100k

Reward and env are the ones rl.py uses. Checkpoints have kind "softq"; evaluate.py loads
them through model.load_policy and deploys at temperature 1 (beta is inside the model).

    python -m roguedagger.sac --config configs/sac_frozen.yaml --set sac.steps=200 wandb.mode=disabled
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .env import ShopEnv
from .features import N_GLOBAL
from .model import SoftQ
from .rl import observe, reward
from .train import (DEFAULTS, deep_update, env_config, load_config, policy_step, random_search_step,
                    synthetic_eval, uniform_step)

DEFAULTS_SAC = deep_update(copy.deepcopy(DEFAULTS), {
    "env": {"ref_pool": 512},
    "train": {"n_envs": 128},
    "sac": {
        "init": "runs/wide-v1/last.pt",   # pairwise checkpoint whose trunk is frozen
        "head_hidden": 64,
        "gamma": 0.99, "polyak": 0.005,
        "perplexity_start": 10.0, "perplexity_end": 1.2, "perplexity_anneal_frac": 0.5,
        "critic_lr": 1e-4, "warmup_updates": 1000, "lr_min_factor": 0.1,
        "beta_lr": 0.01, "init_beta": 1.0, "grad_clip": 1.0,
        "reward": "log", "log_eps": 1e-3, "reward_scale": 100.0,
        "buffer_capacity": 100_000,       # transitions
        "pool_tokens_per_state": 160,     # embedding pool size = capacity x this (evicts earlier if short)
        "batch_size": 512, "learning_starts": 512, "updates_per_step": 8,
        "steps": 50_000,                  # env steps (each steps all n_envs envs)
        "log_every": 50, "eval_every": 2500, "ckpt_every": 2500,
    },
})


# ---------------------------------------------------------------------------- replay
class EmbeddingReplay:
    """Replay of frozen-trunk embeddings, packed: each state keeps only its real tokens
    (its n_runs run tokens, then its shop tokens) in one flat bf16 pool, written step by
    step as a ring. s' of a transition is the same env row's state one step later, so every
    state is stored once. Minibatches are gathered with index arithmetic into the layout
    [runs (max n_runs in batch) | shop (max shop size in batch)].
    """

    def __init__(self, n_envs: int, d: int, capacity: int, pool_tokens: int, device):
        self.E, self.dev = n_envs, device
        self.S = max(2, math.ceil(capacity / n_envs))              # step slots
        self.P = pool_tokens
        self.emb = torch.zeros(self.P, d, dtype=torch.bfloat16, device=device)
        self.ok = torch.zeros(self.P, dtype=torch.bool, device=device)   # choosable action
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)
        self.start = z(self.S, n_envs, dtype=torch.long)
        self.nr = z(self.S, n_envs, dtype=torch.long)
        self.ns = z(self.S, n_envs, dtype=torch.long)
        self.g = z(self.S, n_envs, N_GLOBAL)
        self.a = z(self.S, n_envs, dtype=torch.long)
        self.rew = z(self.S, n_envs)
        self.done = z(self.S, n_envs, dtype=torch.bool)
        self.alive = deque()          # (slot, pool start, pool length), oldest first
        self.w = 0                    # pool write pointer
        self.next_slot = 0
        self.r_min, self.r_max = float("inf"), -float("inf")

    def __len__(self):                # sampleable transitions (the newest step has no s' yet)
        return max(0, len(self.alive) - 1) * self.E

    def push(self, e, g, valid, n_runs, n_shop, r):
        """e [E, r + N, d] (runs block of width r, then shop), valid [E, r + N], g [E, N_GLOBAL];
        n_runs / n_shop [E] real tokens per row. -> slot (fill in a / rew / done with set_outcome)."""
        E = self.E
        L = n_runs + n_shop
        T = int(L.sum())
        assert T <= self.P, "embedding pool smaller than one step"
        w_old, wrapped = self.w, False
        if self.w + T > self.P:
            self.w, wrapped = 0, True
        lo, hi = self.w, self.w + T
        if len(self.alive) == self.S:
            self.alive.popleft()
        while self.alive:
            _, s0, n0 = self.alive[0]
            if (wrapped and s0 >= w_old) or (s0 < hi and s0 + n0 > lo):
                self.alive.popleft()
            else:
                break
        # pack: row-major, each row = its runs then its shop items
        Lm = int(L.max())
        j = torch.arange(Lm, device=self.dev)[None]
        src = torch.where(j < n_runs[:, None], j, j - n_runs[:, None] + r)        # index into [r | N]
        keep = j < L[:, None]
        src = src.clamp(max=e.shape[1] - 1)
        rows = e.gather(1, src[..., None].expand(-1, -1, e.shape[2]))[keep]       # [T, d]
        oks = valid.gather(1, src)[keep]
        self.emb[lo:hi] = rows.to(torch.bfloat16)
        self.ok[lo:hi] = oks
        slot = self.next_slot
        self.next_slot = (slot + 1) % self.S
        self.start[slot] = lo + torch.cumsum(L, 0) - L
        self.nr[slot], self.ns[slot], self.g[slot] = n_runs, n_shop, g
        self.alive.append((slot, lo, T))
        self.w = hi
        return slot

    def set_outcome(self, slot, a_packed, rew, done):
        self.a[slot], self.rew[slot], self.done[slot] = a_packed, rew, done
        self.r_min = min(self.r_min, float(rew.min()))
        self.r_max = max(self.r_max, float(rew.max()))

    def _gather(self, slot, row):
        """-> e [B, Rb + Nb, d], g [B, N_GLOBAL], valid [B, Rb + Nb] for (slot, row) pairs."""
        st, nr, ns = self.start[slot, row], self.nr[slot, row], self.ns[slot, row]
        Rb, Nb = int(nr.max()), int(ns.max())
        jr = torch.arange(Rb, device=self.dev)[None]
        js = torch.arange(Nb, device=self.dev)[None]
        idx = torch.cat([st[:, None] + jr, st[:, None] + nr[:, None] + js], 1)
        real = torch.cat([jr < nr[:, None], js < ns[:, None]], 1)
        idx = torch.where(real, idx, torch.zeros_like(idx))
        e = self.emb[idx]
        valid = self.ok[idx] & real
        return e, self.g[slot, row], valid, Rb

    def sample(self, B):
        k = torch.randint(0, len(self.alive) - 1, (B,))
        slots = [s for s, _, _ in self.alive]
        cur = torch.tensor([slots[i] for i in k.tolist()], device=self.dev)
        nxt = torch.tensor([slots[i + 1] for i in k.tolist()], device=self.dev)
        row = torch.randint(0, self.E, (B,), device=self.dev)
        e, g, valid, Rb = self._gather(cur, row)
        e2, g2, valid2, _ = self._gather(nxt, row)
        a = self.a[cur, row]
        nr = self.nr[cur, row]
        a = torch.where(a < nr, a, a - nr + Rb)                    # packed index -> batch layout
        return (e, g, valid), a, self.rew[cur, row], (e2, g2, valid2), self.done[cur, row]


# ---------------------------------------------------------------------------- learner
def masked(q, valid):
    return q.masked_fill(~valid, float("-inf"))


def soft_update(online, target, tau):
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.mul_(1 - tau).add_(tau * p)


def sac_update(model, ht1, ht2, opt_q, opt_b, batch, sc, target_entropy, r_min, r_max):
    (e, g, m), a, rew, (en, gn, mn), done = batch
    beta = model.log_beta.exp().detach()
    alpha = 1.0 / beta
    q1, q2 = masked(model.h1(e, g), m), masked(model.h2(e, g), m)
    qa1, qa2 = q1.gather(1, a[:, None])[:, 0], q2.gather(1, a[:, None])[:, 0]
    with torch.no_grad():
        qn = masked(torch.minimum(ht1(en, gn), ht2(en, gn)), mn)
        v_next = alpha * torch.logsumexp(qn / alpha, 1)
        y = rew * (1 - sc["gamma"]) + sc["gamma"] * (~done).float() * v_next
        clamp_frac = ((y < r_min) | (y > r_max)).float().mean()
        y = y.clamp(r_min, r_max)
    q_loss = F.smooth_l1_loss(qa1, y) + F.smooth_l1_loss(qa2, y)
    opt_q.zero_grad(set_to_none=True)
    q_loss.backward()
    gn_ = torch.nn.utils.clip_grad_norm_(list(model.h1.parameters()) + list(model.h2.parameters()), sc["grad_clip"])
    opt_q.step()

    # temperature: entropy of each state's policy towards min(log P*, log |A_s|) nats
    qmin = torch.minimum(q1, q2).detach()
    logp = torch.log_softmax(qmin * beta, 1)
    ent = -(logp.exp() * logp.masked_fill(~m, 0.0)).sum(1)
    n_act = m.sum(1)
    ok = n_act > 1
    gap = torch.log(n_act[ok].float()).clamp(max=target_entropy) - ent[ok]
    b_loss = (model.log_beta * gap).mean() if ok.any() else model.log_beta * 0
    opt_b.zero_grad(set_to_none=True)
    b_loss.backward()
    opt_b.step()
    soft_update(model.h1, ht1, sc["polyak"])
    soft_update(model.h2, ht2, sc["polyak"])
    with torch.no_grad():
        q_range = (qmin.max(1).values - qmin.masked_fill(~m, float("inf")).min(1).values)[ok].mean()
    return {"q_loss": q_loss.detach(), "td_error": (qa1.detach() - y).abs().mean(), "q_mean": qa1.detach().mean(),
            "target_mean": y.mean(), "v_next_mean": v_next.mean(), "clamp_frac": clamp_frac,
            "entropy": ent[ok].mean() if ok.any() else ent.sum() * 0, "entropy_gap": gap.mean() if ok.any() else ent.sum() * 0,
            "q_range": q_range, "grad_norm": gn_}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, defaults=DEFAULTS_SAC)
    sc, tc = cfg["sac"], cfg["train"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device("cuda")
    amp = tc["amp"]
    h = (tc["h_start"], tc["h_end"])

    ck = torch.load(sc["init"], map_location=device)
    cfg["model"] = ck["cfg"]["model"]
    assert cfg["model"].get("readout", "token") == "token", "SAC needs a token-readout trunk"
    tc["temperature"] = 1.0                       # beta lives inside the model
    model = SoftQ(d_max=10, head_hidden=sc["head_hidden"], **cfg["model"]).to(device)
    n = model.load_trunk(ck["model"])
    print(f"frozen trunk from {sc['init']} ({n} tensors)", flush=True)
    for mod in (model.embed, model.trunk, model.norm):
        mod.requires_grad_(False)
        mod.eval()
    model.log_beta.data.fill_(math.log(sc["init_beta"]))
    model.log_beta.requires_grad_(True)
    ht1, ht2 = copy.deepcopy(model.h1).requires_grad_(False), copy.deepcopy(model.h2).requires_grad_(False)
    opt_q = torch.optim.Adam(list(model.h1.parameters()) + list(model.h2.parameters()), lr=sc["critic_lr"])
    opt_b = torch.optim.Adam([model.log_beta], lr=sc["beta_lr"])

    total_updates = sc["steps"] * sc["updates_per_step"]
    def lr_at(u):
        if u < sc["warmup_updates"]:
            return sc["critic_lr"] * (u + 1) / sc["warmup_updates"]
        t = min((u - sc["warmup_updates"]) / max(1, total_updates - sc["warmup_updates"]), 1.0)
        return sc["critic_lr"] * (sc["lr_min_factor"] + (1 - sc["lr_min_factor"]) * 0.5 * (1 + math.cos(math.pi * t)))
    h0, h1_ = math.log(sc["perplexity_start"]), math.log(sc["perplexity_end"])
    anneal = max(1, int(sc["perplexity_anneal_frac"] * sc["steps"]))
    target_entropy_at = lambda s: h0 + (h1_ - h0) * min(s / anneal, 1.0)

    name = cfg["name"] or time.strftime("sac-%Y%m%d-%H%M%S")
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    import wandb
    run = wandb.init(project=cfg["wandb"]["project"], name=name, config=cfg, mode=cfg["wandb"]["mode"],
                     dir=str(out_dir), tags=["sac"])

    ecfg = env_config(cfg)
    ecfg_eval = copy.copy(ecfg)
    ecfg_eval.ref_pool = 0
    eval_policy = lambda e: policy_step(model, e, 1.0, h=h, amp=amp)
    base = {}
    if sc["eval_every"]:
        for nm, fn in (("uniform", uniform_step), ("random_search", random_search_step)):
            base[nm] = synthetic_eval(fn, cfg, device, ecfg_eval)
            wandb.log({f"eval/{nm}_regret": base[nm]["regret"]}, step=0)

    E = tc["n_envs"]
    env = ShopEnv(E, ecfg, device, seed=cfg["seed"])
    buf = EmbeddingReplay(E, cfg["model"]["d_model"], sc["buffer_capacity"],
                          sc["buffer_capacity"] * sc["pool_tokens_per_state"], device)
    upd, hist, act_hist, eps_hist, t0 = 0, [], [], [], time.time()
    for step in range(1, sc["steps"] + 1):
        with torch.no_grad():
            H, r, tok, pad, valid = observe(env, h)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                e, g = model.features(tok, pad)
            n_runs, n_shop = env.n_runs.clone(), env.n.clone()
            slot = buf.push(e, g, valid, n_runs, n_shop, r)
            beta = model.log_beta.exp()
            qmin = masked(torch.minimum(model.h1(e, g), model.h2(e, g)), valid)
            p = torch.softmax(qmin * beta, 1)
            a = torch.multinomial(p, 1)[:, 0]
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
            out = env.step(torch.where(a < r, a, a - r + env.R))
            rew = reward(out, sc["reward"], sc["log_eps"]) * sc["reward_scale"]
            buf.set_outcome(slot, torch.where(a < r, a, a - r + n_runs), rew, out["done"])
            act_hist.append(torch.stack([rew.mean(), (rew > 0).float().mean(), ent.mean(), (a >= r).float().mean()]))
            if out["done"].any():
                d = out["done"]
                eps_hist.append({"regret": out["regret"][d].cpu(),
                                 "runs_per_full": (out["n_runs"] / (out["B"] / out["T"])).float().cpu()})

        target_entropy = target_entropy_at(step)
        if len(buf) >= max(sc["learning_starts"], sc["batch_size"]):
            for _ in range(sc["updates_per_step"]):
                for gr in opt_q.param_groups:
                    gr["lr"] = lr_at(upd)
                hist.append(sac_update(model, ht1, ht2, opt_q, opt_b, buf.sample(sc["batch_size"]), sc,
                                       target_entropy, buf.r_min, buf.r_max))
                upd += 1

        if step % sc["log_every"] == 0:
            am = torch.stack(act_hist).mean(0).tolist()
            act_hist = []
            log = {"act/reward": am[0], "act/reward_nonzero": am[1], "act/entropy": am[2], "act/frac_new": am[3],
                   "sac/beta": model.log_beta.exp().item(), "sac/target_perplexity": math.exp(target_entropy),
                   "sac/lr": lr_at(upd), "buffer/transitions": len(buf), "perf/updates": upd,
                   "perf/env_steps_per_s": sc["log_every"] / (time.time() - t0)}
            t0 = time.time()
            if hist:
                log.update({f"sac/{k}": torch.stack([s[k] for s in hist]).float().mean().item() for k in hist[0]})
                log["sac/perplexity"] = math.exp(log["sac/entropy"])
                hist = []
            if eps_hist:
                reg = torch.cat([x["regret"] for x in eps_hist])
                log.update({"episode/norm_regret": reg.mean().item(),
                            "episode/beat_pool": (reg == 0).float().mean().item(),
                            "episode/runs_per_full_budget": torch.cat([x["runs_per_full"] for x in eps_hist]).mean().item(),
                            "episode/count": len(reg)})
                eps_hist = []
            wandb.log(log, step=step)
            print(f"s {step} " + " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log.items()), flush=True)

        if sc["eval_every"] and step % sc["eval_every"] == 0:
            r_ = synthetic_eval(eval_policy, cfg, device, ecfg_eval)
            rg = synthetic_eval(lambda e_: policy_step(model, e_, greedy=True, h=h, amp=amp), cfg, device, ecfg_eval)
            log = {"eval/policy_final": r_["final_best"], "eval/policy_regret": r_["regret"],
                   "eval/policy_n_runs": r_["n_runs"], "eval/greedy_regret": rg["regret"]}
            for nm, b in base.items():
                log[f"eval/delta_vs_{nm}"] = (b["final_per_env"] - r_["final_per_env"]).mean().item()
            wandb.log(log, step=step)
            print(f"EVAL s {step} " + json.dumps({k: round(v, 4) for k, v in log.items()}), flush=True)

        if step % sc["ckpt_every"] == 0 or step == sc["steps"]:
            torch.save({"kind": "softq", "model": model.state_dict(), "cfg": cfg, "it": step,
                        "targets": [ht1.state_dict(), ht2.state_dict()]}, out_dir / "last.pt")
    run.finish()


if __name__ == "__main__":
    main()

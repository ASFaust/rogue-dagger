"""RL fine-tuning of the scorer: on-policy PPO with GAE on the same batched shop env.

The pairwise (DAgger-like) training in train.py is the pretraining stage. This trainer
starts from such a checkpoint (`rl.init`, or from scratch with `rl.init: null`) and
optimises the policy pi(a | s) = softmax(scores / rl.temperature) for a regret reward:

    reward_t = log(regret_{t-1} + eps) - log(regret_t + eps)     (rl.reward: log)
    reward_t = regret_{t-1} - regret_t                           (rl.reward: linear)

regret = (incumbent - pool best) / (pool worst - pool best), the pool being `env.ref_pool`
random configs of the task (the prior's analogue of the benchmarks' table best / worst).
The incumbent never gets worse, so every reward is >= 0 and the return telescopes to the
episode's total regret reduction (log mode: log of the regret ratio).

Each update:
  1. roll the policy for rl.rollout steps in E parallel envs (one env per batch row, no
     replay; finished envs reset immediately), storing tokens, actions, log-probs, values
  2. GAE(gamma, lambda) advantages, bootstrapped with the critic at the last state
  3. rl.epochs passes of minibatch PPO: clipped surrogate + rl.aux_coef * pairwise BCE
     (the pretraining loss, exact labels as in train.py) on the actor, MSE on the critic

The critic is a separate network (model.Critic) whose trunk is initialised from the
checkpoint too. For the first rl.actor_warmup updates only the critic learns, so the
pretrained policy is not pushed around by an uncalibrated value baseline.

Checkpoints keep train.py's format ({"model", "cfg"}), so evaluate.py runs them unchanged
(deploy with --temperature = the checkpoint's train.temperature: rl.temperature, or the learned value).

    python -m roguedagger.rl --config configs/rl_wide.yaml --set rl.updates=20 wandb.mode=disabled
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .env import ShopEnv
from .features import encode, from_env
from .model import Critic, Scorer, pairwise_loss
from .train import (DEFAULTS, deep_update, env_config, load_config, policy_step, random_search_step,
                    synthetic_eval, uniform_step)

DEFAULTS_RL = deep_update(copy.deepcopy(DEFAULTS), {
    "env": {"ref_pool": 512},
    "rl": {
        "init": None,            # pairwise-trained checkpoint (pretraining) or None = from scratch
        "temperature": 0.25,     # policy = softmax(scores / temperature); also the deploy temperature
        "learn_temperature": False,  # train log(temperature) with the PPO loss (start: rl.temperature)
        "temperature_lr": 1e-3,
        "reward": "log",         # log | linear (normalised regret)
        "log_eps": 1e-3,
        "gamma": 1.0, "lam": 0.95,
        "rollout": 16,           # env steps per update
        "epochs": 2, "minibatch": 256,
        "clip": 0.2, "ent_coef": 0.0, "aux_coef": 1.0,
        "actor_lr": 3e-5, "critic_lr": 1e-4, "wd": 0.01, "grad_clip": 1.0,
        "warmup": 50,            # updates of linear LR warmup (each network, from when it starts learning)
        "actor_warmup": 100,     # updates in which only the critic learns
        "updates": 2500,         # x rollout 16 = 40k env steps per env row
        "log_every": 5, "eval_every": 250, "ckpt_every": 250,
    },
})


# ---------------------------------------------------------------------------- helpers
def reward(out: dict, mode: str, eps: float) -> torch.Tensor:
    """Per-step reward [E] from env.step's regret_prev / regret; >= 0 by construction."""
    if mode == "log":
        return torch.log(out["regret_prev"] + eps) - torch.log(out["regret"] + eps)
    if mode == "linear":
        return out["regret_prev"] - out["regret"]
    raise ValueError(f"unknown reward mode {mode!r}")


def gae(rew, val, done, v_boot, gamma, lam):
    """rew, val, done [K, E]; v_boot [E] = V(state after the last step). -> advantages, returns."""
    K = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(v_boot)
    for t in reversed(range(K)):
        nv = v_boot if t == K - 1 else val[t + 1]
        nonterm = 1.0 - done[t].float()
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def pad_runs(x: torch.Tensor, off: int, r: int, r_max: int, fill) -> torch.Tensor:
    """x [E, off + r + N, ...] -> [E, off + r_max + N, ...]: pad the run block with `fill`."""
    if r == r_max:
        return x
    shape = list(x.shape)
    shape[1] = r_max - r
    return torch.cat([x[:, :off + r], torch.full(shape, fill, dtype=x.dtype, device=x.device), x[:, off + r:]], 1)


class Temperature(torch.nn.Module):
    """Policy temperature: fixed, or a learned log tau that only the PPO loss trains (the pairwise
    aux loss does not depend on it, so it keeps the scores calibrated while RL sets the sharpness)."""

    def __init__(self, tau: float, learn: bool):
        super().__init__()
        self.log_tau = torch.nn.Parameter(torch.tensor(math.log(tau)), requires_grad=learn)

    def forward(self):
        return self.log_tau.exp()


def masked_logp(scores, valid, tau):
    """-> log pi over valid actions (−inf elsewhere), entropy [B]."""
    lp = torch.log_softmax((scores / tau).masked_fill(~valid, float("-inf")), -1)
    ent = -(lp.exp() * lp.masked_fill(~valid, 0.0)).sum(-1)
    return lp, ent


def observe(env, h):
    H = env.horizon(*h)
    obs = from_env(env, H)
    r = obs.run_len.shape[1]
    tok, pad = encode(obs, env.d_max)
    ch = env.choosable()
    valid = torch.cat([ch[:, :r], ch[:, env.R:]], 1)
    return H, r, tok, pad, valid


@torch.no_grad()
def collect(actor, critic, env, rc, h, amp, tau):
    """Roll the policy rc["rollout"] steps. -> dict of stacked [K, E, ...] tensors + episode stats."""
    steps, episodes = [], []
    for _ in range(rc["rollout"]):
        H, r, tok, pad, valid = observe(env, h)
        v = env.values(H)
        lab = torch.cat([v[:, :r], v[:, env.R:]], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            s = actor(tok, pad).float()
            val = critic(tok, pad).float()
        lp, ent = masked_logp(s, valid, tau)
        a = torch.multinomial(lp.exp(), 1)[:, 0]
        out = env.step(torch.where(a < r, a, a - r + env.R))
        steps.append({"tok": tok.to(torch.bfloat16), "pad": pad, "valid": valid, "lab": lab, "a": a, "r": r,
                      "logp": lp.gather(1, a[:, None])[:, 0], "val": val, "ent": ent,
                      "rew": reward(out, rc["reward"], rc["log_eps"]), "done": out["done"], "new": a >= r})
        if out["done"].any():
            d = out["done"]
            episodes.append({"regret": out["regret"][d].cpu(),
                             "runs_per_full": (out["n_runs"] / (out["B"] / out["T"])).float().cpu()})
    _, r, tok, pad, _ = observe(env, h)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        v_boot = critic(tok, pad).float()

    r_max = max(st["r"] for st in steps)
    for st in steps:   # common layout [global | r_max runs | shop]; shop actions shift with the padding
        r = st["r"]
        st["tok"] = pad_runs(st["tok"], 1, r, r_max, 0.0)
        st["pad"] = pad_runs(st["pad"], 1, r, r_max, True)
        st["valid"] = pad_runs(st["valid"], 0, r, r_max, False)
        st["lab"] = pad_runs(st["lab"], 0, r, r_max, 1.0)
        st["a"] = torch.where(st["a"] < r, st["a"], st["a"] + (r_max - r))
    keys = ("tok", "pad", "valid", "lab", "a", "logp", "val", "ent", "rew", "done", "new")
    batch = {k: torch.stack([st[k] for st in steps]) for k in keys}
    batch["v_boot"], batch["r_max"] = v_boot, r_max
    return batch, episodes


def set_lr(opt, step, warmup, total):
    """Linear warmup then cosine decay, in updates since this network started learning.
    Every param group keeps its own base LR ("base_lr"). -> LR of the first group."""
    f = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f
    return opt.param_groups[0]["lr"]


def ppo_update(actor, critic, temp, opt_a, opt_c, batch, rc, amp, train_actor):
    tau = temp().detach()
    assert (rc["rollout"] * batch["rew"].shape[1]) % rc["minibatch"] == 0, "rollout * n_envs must be a multiple of minibatch"
    adv, ret = gae(batch["rew"], batch["val"], batch["done"], batch["v_boot"], rc["gamma"], rc["lam"])
    K, E = adv.shape
    flat = lambda x: x.reshape(K * E, *x.shape[2:])
    tok, pad, valid, lab = map(flat, (batch["tok"], batch["pad"], batch["valid"], batch["lab"]))
    a, logp_old, ret_f = flat(batch["a"]), flat(batch["logp"]), flat(ret)
    adv_f = flat(adv)
    adv_n = (adv_f - adv_f.mean()) / adv_f.std().clamp(min=1e-8)
    mb = rc["minibatch"]
    behave_gap = None
    if train_actor:
        # Old log-probs from the update-time forward (same minibatch size and padded length):
        # bf16 trunk outputs depend on batch shape, and at a low temperature that noise alone
        # would push ~10-20% of the ratios out of the clip range before any weight changes.
        rollout_logp, logp_old = logp_old, torch.empty_like(logp_old)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for i in range(0, K * E, mb):
                s = actor(tok[i:i + mb].float(), pad[i:i + mb]).float()
                lp, _ = masked_logp(s, valid[i:i + mb], tau)
                logp_old[i:i + mb] = lp.gather(1, a[i:i + mb][:, None])[:, 0]
        behave_gap = (rollout_logp - logp_old).abs().mean().item()
    stats = []
    for _ in range(rc["epochs"]):
        perm = torch.randperm(K * E, device=adv.device)
        for i in range(0, K * E, mb):
            j = perm[i:i + mb]
            t_j, p_j = tok[j].float(), pad[j]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                v_new = critic(t_j, p_j).float()
            c_loss = F.mse_loss(v_new, ret_f[j])
            opt_c.zero_grad(set_to_none=True)
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), rc["grad_clip"])
            opt_c.step()
            st = {"critic_loss": c_loss.detach()}
            if train_actor:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    s = actor(t_j, p_j).float()
                lp, ent = masked_logp(s, valid[j], temp())
                lp_a = lp.gather(1, a[j][:, None])[:, 0]
                ratio = torch.exp(lp_a - logp_old[j])
                A = adv_n[j]
                pg = -torch.minimum(ratio * A, ratio.clamp(1 - rc["clip"], 1 + rc["clip"]) * A).mean()
                aux, acc, _ = pairwise_loss(s, lab[j], valid[j]) if rc["aux_coef"] > 0 else (pg * 0, pg * 0, 0)
                loss = pg - rc["ent_coef"] * ent.mean() + rc["aux_coef"] * aux
                opt_a.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + [temp.log_tau], rc["grad_clip"])
                opt_a.step()
                with torch.no_grad():
                    st.update(pg_loss=pg.detach(), aux_loss=aux.detach(), pair_acc=acc,
                              approx_kl=(logp_old[j] - lp_a).mean(), grad_norm=gn,
                              clip_frac=((ratio - 1).abs() > rc["clip"]).float().mean())
            stats.append(st)
    out = {k: torch.stack([s[k] for s in stats]).mean().item() for k in stats[0]}
    var = ret_f.var().clamp(min=1e-12)
    out["explained_var"] = (1 - (ret_f - flat(batch["val"])).var() / var).item()
    out["value_mean"], out["return_mean"] = flat(batch["val"]).mean().item(), ret_f.mean().item()
    if behave_gap is not None:
        out["behaviour_logp_gap"] = behave_gap
    return out


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, defaults=DEFAULTS_RL)
    rc, tc = cfg["rl"], cfg["train"]
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    amp = tc["amp"]
    h = (tc["h_start"], tc["h_end"])

    actor_sd = None
    if rc["init"]:
        ck = torch.load(rc["init"], map_location=device)
        cfg["model"] = ck["cfg"]["model"]          # architecture comes from the pretrained checkpoint
        actor_sd = ck["model"]
    tc["temperature"] = rc["temperature"]          # recorded for evaluate / the reader
    actor = Scorer(d_max=10, **cfg["model"]).to(device)
    critic = Critic(d_max=10, **cfg["model"]).to(device)
    if actor_sd is not None:
        actor.load_state_dict(actor_sd)
        n = critic.load_trunk(actor_sd)
        print(f"init from {rc['init']}; critic trunk tensors copied: {n}", flush=True)

    name = cfg["name"] or time.strftime("rl-%Y%m%d-%H%M%S")
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    import wandb
    run = wandb.init(project=cfg["wandb"]["project"], name=name, config=cfg, mode=cfg["wandb"]["mode"],
                     dir=str(out_dir), tags=["rl"])

    temp = Temperature(rc["temperature"], rc["learn_temperature"]).to(device)
    groups = [{"params": list(actor.parameters()), "base_lr": rc["actor_lr"]}]
    if rc["learn_temperature"]:
        groups.append({"params": [temp.log_tau], "base_lr": rc["temperature_lr"], "weight_decay": 0.0})
    opt_a = torch.optim.AdamW(groups, lr=rc["actor_lr"], weight_decay=rc["wd"], betas=(0.9, 0.98))
    opt_c = torch.optim.AdamW([{"params": list(critic.parameters()), "base_lr": rc["critic_lr"]}],
                              lr=rc["critic_lr"], weight_decay=rc["wd"], betas=(0.9, 0.98))

    ecfg = env_config(cfg)
    ecfg_eval = copy.copy(ecfg)
    ecfg_eval.ref_pool = 0                         # same eval tasks (same RNG stream) as train.py's runs
    eval_policy = lambda e: policy_step(actor, e, temp().item(), h=h, amp=amp)
    base = {}
    if rc["eval_every"]:
        for nm, fn in (("uniform", uniform_step), ("random_search", random_search_step)):
            base[nm] = synthetic_eval(fn, cfg, device, ecfg_eval)
            wandb.log({f"eval/{nm}_regret": base[nm]["regret"]}, step=0)
        actor.eval()
        r0 = synthetic_eval(eval_policy, cfg, device, ecfg_eval)
        print(f"eval init policy: regret {r0['regret']:.4f}", flush=True)
        wandb.log({"eval/policy_regret": r0["regret"], "eval/policy_final": r0["final_best"]}, step=0)

    env = ShopEnv(tc["n_envs"], ecfg, device, seed=cfg["seed"])
    steps_per_update = rc["rollout"] * tc["n_envs"]
    hist, eps_hist, t0 = [], [], time.time()
    for u in range(1, rc["updates"] + 1):
        actor.eval()
        critic.eval()
        batch, episodes = collect(actor, critic, env, rc, h, amp, temp().detach())
        eps_hist += episodes
        train_actor = u > rc["actor_warmup"]
        lr_c = set_lr(opt_c, u - 1, rc["warmup"], rc["updates"])
        lr_a = set_lr(opt_a, u - 1 - rc["actor_warmup"], rc["warmup"],
                      rc["updates"] - rc["actor_warmup"]) if train_actor else 0.0
        actor.train()
        critic.train()
        st = ppo_update(actor, critic, temp, opt_a, opt_c, batch, rc, amp, train_actor)
        st.update({"reward_mean": batch["rew"].mean().item(), "entropy": batch["ent"].mean().item(),
                   "frac_new": batch["new"].float().mean().item(), "r_max": batch["r_max"],
                   "lr_actor": lr_a, "lr_critic": lr_c, "temperature": temp().item()})
        hist.append(st)

        if u % rc["log_every"] == 0:
            log = {f"train/{k}": sum(d.get(k, 0.0) for d in hist) / len(hist) for k in hist[-1]}
            dt = time.time() - t0
            t0 = time.time()
            log["perf/env_steps_per_s"] = rc["log_every"] * rc["rollout"] / dt
            log["perf/env_steps"] = u * rc["rollout"]
            hist = []
            if eps_hist:
                reg = torch.cat([e["regret"] for e in eps_hist])
                log.update({"episode/norm_regret": reg.mean().item(),
                            "episode/log_regret": torch.log(reg + rc["log_eps"]).mean().item(),
                            "episode/beat_pool": (reg == 0).float().mean().item(),
                            "episode/runs_per_full_budget": torch.cat([e["runs_per_full"] for e in eps_hist]).mean().item(),
                            "episode/count": len(reg)})
                eps_hist = []
            wandb.log(log, step=u)
            print(f"u {u} " + " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in log.items()), flush=True)

        if rc["eval_every"] and u % rc["eval_every"] == 0:
            actor.eval()
            r = synthetic_eval(eval_policy, cfg, device, ecfg_eval)
            rg = synthetic_eval(lambda e: policy_step(actor, e, greedy=True, h=h, amp=amp), cfg, device, ecfg_eval)
            log = {"eval/policy_final": r["final_best"], "eval/policy_regret": r["regret"],
                   "eval/policy_n_runs": r["n_runs"], "eval/greedy_regret": rg["regret"]}
            for nm, b in base.items():
                log[f"eval/delta_vs_{nm}"] = (b["final_per_env"] - r["final_per_env"]).mean().item()
            wandb.log(log, step=u)
            print(f"EVAL u {u} " + json.dumps({k: round(v, 4) for k, v in log.items()}), flush=True)

        if u % rc["ckpt_every"] == 0 or u == rc["updates"]:
            tc["temperature"] = temp().item()      # the deploy temperature (learned or fixed)
            ck = {"model": actor.state_dict(), "critic": critic.state_dict(), "cfg": cfg, "it": u,
                  "env_steps": u * steps_per_update}
            torch.save(ck, out_dir / "last.pt")
    run.finish()


if __name__ == "__main__":
    main()

# RL extension (`roguedagger/rl.py`)

An RL stage on top of the pairwise (DAgger-like) training. It uses the same env, tokens,
scorer and evaluation. The policy is still softmax(scores / τ) over [runs | shop], so an RL
checkpoint deploys exactly like a pairwise one. The prior RL attempt (automl-exam: implicit
discrete SAC with replay) is a reference only.

## Design

### 1. Pretraining = the existing pairwise stage

The pairwise training in `train.py` already produces a policy (softmax of the scores),
trained on exact labels in the state distribution of that same policy. That makes it
the pretraining stage. RL starts from such a checkpoint (`rl.init`).

- **No permanent freezing.** RL is supposed to change the scores. As an anchor, the pairwise BCE stays
  on as an auxiliary loss (`rl.aux_coef`, default 1). Its labels are exact and free in the simulator, and
  it keeps the trunk's representation from drifting while the policy-gradient signal is still noisy.
- **Critic warmup instead of freezing.** The critic is new. For the first `rl.actor_warmup` updates
  (default 100) only the critic learns while the frozen pretrained actor drives the rollouts. Without this,
  an uncalibrated value baseline produces advantages that are pure noise and degrade the pretrained policy.
- The critic is a separate network (`model.Critic`) with the scorer's trunk and a value head on the global
  token. Its trunk is initialised from the same checkpoint, so it starts with the pretrained features.
  It is separate rather than shared so that value-regression gradients never reach the actor.
- Ablations: `rl.init: null` (no pretraining, `configs/rl_scratch.yaml`) and `rl.aux_coef: 0` (pure RL
  fine-tuning).

### 2. Reward: normalised regret, improvements only

    regret_t = (incumbent_t − pool_min) / (pool_max − pool_min)          clamped at 0
    log:     r_t = log(regret_{t−1} + ε) − log(regret_t + ε)       (default, ε = 1e-3)
    linear:  r_t = regret_{t−1} − regret_t

- `pool_min` and `pool_max` are the min and max loss over all legal points of `env.ref_pool` (512) random
  configs of the task, drawn once at reset. This is the prior's version of the benchmarks' "table best /
  worst" normalisation. It is fixed for the whole episode. That matters because `env.oracle` (best item
  offered so far) grows as shops are drawn and would make the reward non-stationary.
- The incumbent never gets worse, so **every reward is ≥ 0**: only improvements pay, and nothing is
  ever punished.
- The return (γ = 1) telescopes to the episode's total regret reduction, so maximising it means minimising
  final regret, the metric the benchmarks report.
- **Why log is the default:** the benchmark gaps sit at regret ≈ 0.01 (LCBench 0.016 vs 0.008). In linear
  reward, closing that gap is worth 0.008 of a return that is mostly earned in the first few steps
  (1 → 0.1). In log reward, 0.016 → 0.008 is worth as much as 0.5 → 0.25. ε caps the reward once the agent
  beats the pool (`episode/beat_pool` logs how often).
- Not implemented (would also be ≥ 0): an anytime variant that weights each improvement by the remaining
  budget fraction, i.e. area under the regret curve.

### 3. Algorithm: PPO + GAE, on-policy, no replay

The recommendation is on-policy PPO rather than discrete SAC, for this setup specifically:

- **The simulator is free and batched.** SAC's advantage is sample efficiency, which mattered in
  automl-exam. Here env steps cost almost nothing next to the network, so stability counts for more than
  sample efficiency.
- **Replay doesn't fit the state.** A state is a set of up to 1 + 384 + 64 tokens of curve features
  (≈ 70–140 KB each). An off-policy buffer of useful size is many GB. It would also break the
  "one env per batch row, no buffer" design. PPO keeps only the current rollout (16 steps × 256 envs).
- **Long horizons.** Episodes last up to 1200 steps. Soft Q-learning bootstraps through every step, with
  γ ≈ 1 and an entropy term in every target. GAE with a value baseline handles the credit assignment more
  robustly, and γ = 1 makes the objective exactly the final regret.
- **The pretrained model is a policy, not a Q-function.** Scores are only defined up to a per-state
  constant and a scale (pairwise BCE), so they don't transfer as Q-values. In actor-critic they are the
  actor directly, unchanged.

Per update: 16 steps × 256 envs of rollouts, then GAE (γ = 1, λ = 0.95). Then 2 epochs of minibatch-256 PPO
(clip 0.2): clipped surrogate + `aux_coef` × pairwise BCE on the actor, MSE on the critic. The actor LR is
3e-5 (fine-tuning) and the critic LR 1e-4, each with warmup + cosine. Advantages are normalised per update.

Two numerical details, both needed at τ = 0.25:
- The score / value heads run in fp32 even under bf16 autocast (`model._fp32`). bf16 scores are quantised
  to ~0.06 at |s| ≈ 8, which τ = 0.25 turns into ~0.25 logit noise.
- The bf16 trunk's output depends on batch shape. Old log-probs are therefore recomputed with the
  update-time forward before the PPO epochs. Without this, ~15% of ratios start outside the clip range
  at LR 0. `train/behaviour_logp_gap` logs the rollout-vs-recompute difference (~0.07 nats).

Not adopted, and why:
- Group-relative baselines (GRPO-style, several rollouts per task): episode-level advantages over 1000
  actions are too noisy.
- KL to the pretrained policy: the auxiliary BCE already plays the anchoring role.

### Temperature

τ is fixed during RL (`rl.temperature`, default 0.25, the deploy τ chosen on the synthetic prior for
`wide-v1`). RL optimises the policy at that τ, so RL checkpoints deploy at the same τ:
`T32=0.25 TFULL=0.25 sh scripts/eval_ckpts.sh FAMILY TAG:runs/<name>/last.pt`.
The from-scratch config uses τ = 1, since a random scorer's scores have no scale yet.

`rl.learn_temperature: true` trains log τ (start `rl.temperature`, LR `rl.temperature_lr`) with the PPO loss only.
With the aux BCE on, the scores stay calibrated as pairwise probabilities, and a fixed τ makes RL fight the BCE over the
score scale. A learned τ lets RL choose the policy's sharpness separately. The checkpoint stores the learned value
as `train.temperature`, which is the τ to deploy at.

## Running

    python -m roguedagger.rl --config configs/rl_wide.yaml --set name=rl-wide-v1
    python -m roguedagger.rl --config configs/rl_wide.yaml --set name=rl-wide-noaux-v1 rl.aux_coef=0
    python -m roguedagger.rl --config configs/rl_scratch.yaml --set name=rl-scratch-v1

Cost: ~3–4 env steps/s with 256 envs, about 3× slower per env step than pairwise training.
The default of 2500 updates (40k env steps) takes ~3–4 h.

Logged (wandb, tag `rl`):
- `train/*`: PPO stats, explained variance, entropy, share of new runs
- `episode/norm_regret`, `episode/beat_pool`: on-policy episodes
- `eval/*`: the same fixed synthetic tasks as `train.py` (comparable with the pairwise runs' curves)

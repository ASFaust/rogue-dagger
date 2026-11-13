# Backlog

## Now (v1 pipeline)
- [x] Repo, README, backlog
- [x] Vendor ifbo + FT-PFN weights, runs on py3.9
- [x] TaskSet eval data (24 tasks)
- [x] GPU-batched ifBO prior (`prior.py`); curve stats match the numpy reference (200 tasks × 16 configs)
- [x] Batched shop env (`env.py`): runs + fresh shop each step, async resets. Label value = min over curve[0 : len+H] (past included); prior runs uncapped (curves extend to the full budget); per-env `cap` flag for tabular-style max fidelity
- [x] Tokenizer (`features.py`): global token + run/shop tokens, normalisation across runs
- [x] Scorer (`model.py`): small set-transformer, scalar score per token
- [x] Training loop (`train.py`): pairwise BCE, multinomial rollouts, wandb, checkpoints, periodic synthetic eval vs random
- [x] Tabular benchmark loader + HPO loop (`bench.py`, `evaluate.py`)
- [x] FT-PFN + MFPI-random baseline + random search on TaskSet, PD1, LCBench
- [x] PD1 tabular processing (`scripts/process_pd1.py`, 28 ifBO task variants)
- [x] LCBench (35 tasks, first 1000 configs)
- [x] First comparison (`wide-v1` @100k, full-table shop): PD1 better than FT-PFN, TaskSet on par, LCBench behind (README, Results)
- [x] Ablations at 100k: horizon anneal (no effect), future-only label (worse on LCBench, not better elsewhere)
- [x] Pair weighting (`wide_focus`): LCBench much better (on par with FT-PFN), PD1 worse, TaskSet same
- [x] Pick the deploy τ without using the benchmarks (`scripts/synth_tau.py`: τ=0.25)
- [x] Training τ = deploy τ (`wide_t025`): worse on all three families (starts half as many runs in training)
  - at τ=1 the scores flatten during training (entropy 1.3 -> 3.2),
  the policy starts ever more runs and the on-policy loss rises, while greedy keeps improving
- [x] Scale: 300k iterations (`wide_long`) worse everywhere; train shops up to 256 (`wide_bigshop`) better on TaskSet + LCBench, worse on PD1
- [x] MLP readout (`wide_readout`): best TaskSet (0.047), PD1 ≈, LCBench worse (0.024)
- [x] Pair weighting 0.3 (`wide_focus03`): PD1 0.0059 (better than wide-v1 and FT-PFN), LCBench 0.014, TaskSet ≈
- [~] Pair weighting 0.3 + train shops to 256 (`wide_focus03_big`): training
- [ ] Close the LCBench gap (0.016 vs 0.008): early regret (@100) is about 2x FT-PFN's

## RL extension (docs/rl.md)
- [x] PPO + GAE trainer (`roguedagger/rl.py`) starting from a pairwise checkpoint; separate critic initialised from its trunk; critic-only warmup; pairwise BCE as aux loss
- [x] Reward: per-step decrease of log normalised regret (pool of 512 random configs per task), always >= 0
- [x] fp32 heads + recomputed old log-probs (bf16 noise otherwise clips ~15 % of PPO ratios at τ=0.25)
- [x] First runs: `rl-wide-v1` (PD1 +0.004 on full table, rest ≈), `rl-wide-noaux-v1` (shop 32 TaskSet +0.057, full table much worse), `rl-scratch-v1` (≈ slightly below wide-v1)
- [~] RL on shops up to 256 (`rl_big.yaml`): `rl-big-noaux-v1` (from wide-v1), `rl-big-focus03-v1` (from wide-focus03-v1, aux)
- [ ] Pure RL drifts to starting runs until the 384-run cap: check whether that is the reward (log, ε) or the shop-size specialisation
- [~] Learnable τ (`rl.learn_temperature`, log τ trained by the PPO loss only): implemented, not run yet
- [ ] Reward ablation: linear vs log; init from `wide-bigshop-v1`; shared vs separate critic if speed matters

## Open design questions (from PLAN.md)
- Shop density and size `n`: start with uniform; then try mutation of top runs + uniform; TPE; Langevin on the scorer's own gradient (v2)
- Horizon reported in the global token: full remaining vs under-reported vs annealed to full near the end
- Policy temperature τ (train vs deploy)
- Time unit: fixed normalisation + last-10-points features for now; wall-time steps only matter in deployment
- Pair weighting: plain BCE over all pairs vs weighting by |Δloss| or emphasising pairs near the top

- Train-time fidelity cap: should some prior episodes use a cap (as the benchmarks do), so the policy learns about finished runs? Default is no cap.

## Later
- Categorical hyperparameters (one-hot) — the eval benchmarks are all numeric, so prior + tokens are numeric-only for now
- Real-world harness: only the active run on the GPU; freeze = move state to RAM/disk, thaw = load back
- Wall-clock step unit for deployment

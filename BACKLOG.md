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
- [~] FT-PFN + MFPI-random baseline: TaskSet + PD1 done, LCBench running; random search done for all three
- [x] PD1 tabular processing (`scripts/process_pd1.py`, 28 ifBO task variants)
- [x] LCBench (35 tasks, first 1000 configs)
- [~] First training runs + comparison: PD1 on par with FT-PFN, TaskSet behind (README, Results)
- [ ] Find the TaskSet gap: label_past ablation, horizon anneal on wide, curve normalisation on log-loss curves

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

# Handoff (2026-11-13): state for the next session

Read this first, then `README.md` (method + results), `BACKLOG.md` (checklist), and `PLAN.md`
(the user's spec). Code, configs and docs are all committed; `data/` and `runs/` are git-ignored but live
on the shared disk (`/shared/work/private/rogue-dagger/{data,runs}`).

## Constraints from the user

- Shared server. Use GPUs **3–6** (RTX 6000 Ada). GPU 6 was allowed on 2026-11-13 "until further notice";
  PLAN.md's rule otherwise is max 3 GPUs, at least 2 free. GPUs 0–2 (A6000) stay free. Always
  `export CUDA_DEVICE_ORDER=PCI_BUS_ID`. Don't smoke-test on an idle GPU outside that budget.
- Python: `source /shared/work/venv/bin/activate`. Training logs to wandb, project `rogue-dagger`.
- Label definition is the user's and is settled: p(A beats B) where each item's value is its best loss over
  its whole curve up to `len + H` (past included). `label_past: false` exists only as a finished ablation.
- Prior runs are uncapped (t > 1 allowed); only the real benchmarks cap at max fidelity.
- Don't import tricks from the DAgger paper beyond "the same policy steers the rollouts".
- LCBench data was placed by the user (figshare blocks automated downloads; don't work around it).
- Commits: author Andreas Faust, push to `origin main` (git@github.com:ASFaust/rogue-dagger.git).
- Kill processes by PID, not `pkill -f <pattern>` (the pattern can match your own shell).

## Why the session ended

At ~06:10 UTC on 2026-11-13 the container lost GPU access for new processes
(`nvidia-smi`: "Failed to initialize NVML: Unknown Error"); processes already on the GPUs finished.
The user restarted the container. Every job started after 06:10 crashed immediately.

## Step 1 after the restart

```bash
cd /shared/work/private/rogue-dagger
sh scripts/resume_2026-11-13.sh
```

It checks CUDA and then starts, on GPUs 3–6:

| GPU | Job | Output |
|---|---|---|
| 3 | benchmark evals + synthetic τ sweep of `wide-long-v1` (300k) | `runs/logs/eval-wlong300k.log` |
| 4 | same for `wide-bigshop-v1` | `runs/logs/eval-wbig100k.log` |
| 5 | same for `wide-t025-v1` | `runs/logs/eval-wt025.log` |
| 6 | same for `wide-readout-v1`, plus training of `wide-focus03-v1` | `runs/logs/eval-wread100k.log`, `runs/logs/wide-focus03-v1.log` |

Evals take ~1 h per checkpoint (6 benchmark runs) plus the τ sweep; focus03 training ~3 h.
Then compare, e.g.:

```bash
for f in taskset pd1 lcbench; do
  python scripts/compare.py $f ifbo wide100k-n0-t0.25 wlong300k-n0-t0.25 wbig100k-n0-t0.25 wt025-n0-t0.25 wread100k-n0-t0.25 --ref ifbo --plot
  python scripts/compare.py $f wide100k-n0-t0.25 wread100k-n0-t0.25 --ref wide100k-n0-t0.25   # paired vs reference
done
```

Record the results in README "Results so far" and BACKLOG, commit, push. Then evaluate `wide-focus03-v1`
(`scripts/eval_ckpts.sh FAMILY wfocus03:runs/wide-focus03-v1/last.pt`) when it finishes.

## Checkpoints (all complete, `runs/<name>/last.pt`)

| Run | Config | What it tests | Benchmark eval |
|---|---|---|---|
| `wide-v1` | `wide.yaml` | reference | done (`wide100k-*`) |
| `wide-hanneal-v1` | `wide_hanneal.yaml` | horizon anneal | done: no effect |
| `wide-future-v1` | `wide_future.yaml` | future-only label | done: worse on LCBench |
| `wide-focus-v1` | `wide_focus.yaml` | pair weighting 0.1 | done: LCBench ≈ FT-PFN, PD1 worse |
| `wide-long-v1` | `wide_long.yaml` | 300k iterations | **pending** |
| `wide-bigshop-v1` | `wide_bigshop.yaml` | train shops up to 256 | **pending** |
| `wide-t025-v1` | `wide_t025.yaml` | training rollouts at τ=0.25 | **pending** |
| `wide-readout-v1` | `wide_readout.yaml` | no global token, MLP([emb, global]) readout, 4 horizons/step | **pending** |
| `wide-focus03-v1` | `wide_focus03.yaml` | pair weighting 0.3 | **not trained** (crashed at start) |

FT-PFN and random search baselines are done for all three families (`runs/eval/<family>/{ifbo,random_search}.npz`).

## Where things stand (details in README)

Headline (full-table shop, τ=0.25, τ chosen on the synthetic prior): `wide-v1` beats FT-PFN on PD1
(0.0098 vs 0.0151), is on par on TaskSet (0.062 vs 0.053, CI includes 0), and is behind on LCBench (0.016 vs 0.0076).
`wide-focus-v1` closes LCBench (0.0092) but loses PD1.

Preliminary synthetic-eval signal for the pending runs (shop 32, mean of last 4 in-training evals, regret):
wide-v1 0.0374 / greedy 0.0430; long 0.0365 / 0.0437; readout **0.0357** / 0.0481; bigshop 0.0420 / 0.0467;
t025 0.0420 (sampled at τ=0.25) / 0.0546. The synthetic shop-32 eval has not predicted benchmark results well.

## Open threads

- Training at τ=1 makes scores flatten over training (entropy 1.3 → 3.2), so on-policy loss/regret curves
  rise after ~10k while greedy and benchmark performance improve: the curves measure a moving state
  distribution. `wide-t025` was meant to fix that; it ended with high loss (0.60), pair acc 0.63 and ~335 runs in
  context. Check whether τ=0.25 rollouts over-train few runs or start too many.
- 55k → 100k helped a lot; 300k gives nothing extra on synthetic, so probably the LR decay mattered.
- Shop tokens carry only hyperparameters, so shop items are scored by in-context hp regression; shop-vs-shop
  pair accuracy (~0.59) is the measured weak spot and the likely reason for the LCBench gap.
- The user remarked that the "was just advanced" and "frozen for" run features shouldn't matter; keep them.
- Prior beyond t = 1: per-point i.i.d. noise slightly lowers the running min of flat curves
  (~0.008 extra over t = 1 → 20); minor, but a bias towards over-training flat runs.

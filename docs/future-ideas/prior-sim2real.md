# Measuring the prior's sim-to-real gap, then improving the prior

> **2026-11-20:** this line of work continues in the sister repo `../curve-faker`; this note is the record of
> what was done here.

Status: measured, first fix tested negative (2026-11-15; journal entries of that date). Curve statistics: prior noise
3-4× too large and white, too much overfitting/divergence, no dead configs, no near-best cluster. Matched-twin ρ:
the crude PD1 inversion was spec mismatch; pairwise ρ positive everywhere (TaskSet +0.56 significant); RL
uninformative (8 models). Scorer real vs twin: judgment transfers, except TaskSet comparisons of unseen configs.
Noise fix (×0.25, smoothed 1 point, calibrated on dev; `wide-focus03-noise`, 3 seeds): worse on the benchmarks
(journal 2026-11-16), so matching curve statistics is not a reliable target by itself.

## Why measure first

"Improve the prior" needs a target. Without a gap measure, any prior change can only be judged by retraining a
scorer and running the benchmarks, which is slow, noisy (seed noise ≈ 0.017 / 0.013 / 0.005) and uses the test set.
A gap measure that is cheap, decision-relevant and computed on *development* data (not the 3 benchmarks) would
let us iterate on the prior directly.

What the prior is (`roguedagger/prior.py`, a port of ifBO's): per task, a random BNN maps a config in [0, 1]^d to
23 latents; each becomes a curve parameter via an empirical CDF. A curve is a weighted sum of 4 basis curves (pow4,
exp, log, hill) with an optional breaking point, plus i.i.d. Gaussian noise per point. Gaps this suggests before
any measurement:
- noise is independent across points and across configs: no autocorrelated noise, no evaluation noise shared by
  all configs of a task (same validation set);
- smooth parametric shapes: no learning-rate-schedule drops, warm-up bumps, divergence (loss blowing up or NaN),
  and only limited overfitting shapes;
- the smoothness of the config-to-curve map is whatever a random BNN gives, not measured against real tasks.

## Measurements, from the decision down to the data

### 1. Decision level: does the prior rank policies like the real tasks do?

The gap that matters is whether being better on the prior means being better on real tasks. We have a free sample:
every model with both a synthetic eval (last `EVAL` line of the training log: 128 prior tasks, T 50, budget 1000,
shop 32, τ 1) and benchmark results at shop 32. Spearman ρ between synthetic and benchmark regret (2026-11-15):

| Models | TaskSet | PD1 | LCBench |
|---|---|---|---|
| all (28) | +0.24 | +0.10 | +0.11 |
| pairwise (21) | **+0.75** | **−0.32** | +0.51 |
| RL (7) | −0.18 | **−0.82** | −0.04 |

Reading: among pairwise models the prior ranks fairly well for TaskSet, moderately for LCBench, and *inversely*
for PD1. RL models are ranked inversely, PD1 most of all: the "RL exploits the prior" finding as a number.
Caveats: one noisy eval per model; the synthetic spec (T 50) doesn't match the benchmarks (PD1 has up to 1414
steps per training), so part of the PD1 inversion may be spec mismatch, not curve shape.

Better version, **matched synthetic twins:** for each real task, prior tasks with the same d, T, budget and shop
size, so only the curve distribution differs. Then ρ per family (and per task group, e.g. long vs short PD1 curves)
measures the curve gap alone. Cheap: the evaluation code exists; it needs a spec per real task.

This is also the missing model-selection signal for RL: if the twins' ρ becomes high after a prior fix, synthetic
eval becomes usable for selection again.

### 2. Model level: where does a model trained on the prior go wrong on real tasks?

- **Our scorer's pair accuracy, real vs twin.** Benchmark tables hold full curves, so labels are exact on real
  states too. Roll out a frozen scorer on real tasks and on their twins, compare pairwise BCE and accuracy, and
  break the gap down by pair type (shop vs run, run-vs-run, pairs whose labels cross, short vs long horizon,
  early vs late in the budget). Shows *which comparisons* the prior teaches wrongly. Uses `scripts/head_test.py`'s
  collection code with a `TabEnv`.
- **FT-PFN calibration.** A PFN approximates the posterior predictive under its prior, so on prior data it is
  calibrated by construction. On real data, any miscalibration is prior misspecification, and its form shows the
  direction: PIT histograms of predicted vs actual continuation values, split by horizon and run progress (e.g.
  "too confident that late improvement is small"). Cheap: FT-PFN already runs in our evals.

### 3. Data level: how do real and prior curves differ?

- **Summary statistics.** Per task, compute statistics on real and matched-prior curves, compare their
  distributions (1-D Wasserstein or KS per statistic), and rank the statistics by discrepancy. Decision-relevant
  ones first:
  1. early-late rank correlation: Spearman over configs of loss at fraction f of T vs final (the premise of
     freeze-thaw), as a function of f;
  2. crossing rate of curve pairs and where crossings happen (also relevant to the crossing idea);
  3. spread of final losses across configs, and the fraction of configs near the best (how much HPs matter);
  4. noise: residual size and autocorrelation after smoothing; non-monotone steps;
  5. divergence, flat curves, overfitting (best epoch before the last);
  6. config smoothness: correlation of final loss against config distance.
- **Classifier two-sample test.** Train a classifier to tell real from prior, at curve level and at task level (the
  set of curves of one task; our set transformer fits). Accuracy near 0.5 means indistinguishable; feature
  attributions say what gives the prior away. The task level matters because the cross-config structure (the BNN)
  can be wrong even if single curves look right. Risk: the classifier finds trivial tells (value range, grid of
  steps); normalise as our encoder does before classifying.

## Data, and keeping the test set clean

Fitting the prior to TaskSet / PD1 / LCBench would leak the test set. Development data instead:
- the rest of TaskSet (about 1150 tasks beyond the 24; `scripts/download_taskset.py` needs extending);
- other epoch-curve sets: Quick-Tune meta-dataset, NAS-Bench-201, HPOBench FCNet; surrogates (YAHPO, JAHS) with
  care, since they carry their own model bias;
- LCDB 1.1 only for statistics that don't need epoch curves (it is sample-wise).

Measure on dev, change the prior on dev, and check the 3 benchmarks only at the end. PD1 has no obvious dev
counterpart (long-curve deep-learning tasks); the PD1 inversion above makes that a priority.

## Improving the prior, once the gap is measured

In order of effort:
1. **Tune the existing prior's hyperparameters** (basis weight distributions, noise scale, break-point rate, BNN
   width/depth) to minimise the summary-statistic discrepancy on dev data. The prior is GPU-batched, so each
   candidate costs seconds; BO or CMA-ES over a dozen hyperparameters is feasible.
2. **Add the missing phenomena** the statistics and the classifier point to: correlated noise, a shared per-task
   evaluation-noise term, schedule drops, divergence, overfitting shapes.
3. **Mix real dev curves into training** (way (a)), with augmentations that keep the decision structure: monotone
   transforms of the loss per task, time warping, config subsets.
4. **A learned generative model** of whole tasks trained on dev data; most flexible, hardest to validate.

Validation for each change: the gap measures shrink on held-out dev tasks → train `wide-focus03` on the new prior
(3 seeds) → benchmarks. Success would show as higher twin ρ (especially PD1), and as RL improvements on the prior
starting to transfer.

## First steps (cheap to expensive)

1. ~~Matched twins for the 3 benchmarks + ρ over our ~30 existing models (no training).~~ Done (`scripts/twins.py`). Separates spec mismatch
   from curve mismatch, notably for the PD1 inversion.
2. ~~Scorer pair accuracy, real vs twin, by pair type (one frozen model, no training).~~ Done (`scripts/pair_gap.py`).
3. ~~Download more TaskSet tasks as dev data; summary statistics, real vs prior.~~ Done: `data/taskset_dev`
   (`scripts/taskset_dev_tasks.txt`), `scripts/curve_stats.py`.
4. FT-PFN PIT calibration on dev data.
5. Classifier two-sample test.

# Rogue&Dagger HPO

> Picking this up in a new session? Start with [HANDOFF.md](HANDOFF.md).

Freeze-thaw hyperparameter optimisation played like a roguelike deckbuilder.
Your **deck** is the set of training runs you have started. Each step a new
**shop** of `n` freshly sampled hyperparameter configs appears. With every unit
of compute you either keep training one of your runs (thaw) or buy a config
from the shop (start a new run). The full idea is in [PLAN.md](PLAN.md).

The aim is to replace FT-PFN's random acquisition (MFPI-random, from the ifBO
paper) with a learned policy, and to beat FT-PFN on the same benchmarks.

## Method

**MDP.** The state is the started runs plus `n` shop suggestions, with a new
random shop at every step. An action is a one-hot choice of which run or
suggestion gets the next step of compute. A run that has reached its maximum
fidelity can't be picked again.

**Scores.** A small set-transformer gives every run and suggestion a scalar
score `s`. It is trained with pairwise BCE:

    p(a beats b within horizon H) = sigmoid(s_a - s_b)

Here "a beats b" means that `a`'s best loss *over its whole curve up to `H`
more steps*, past points included, is lower than `b`'s. Past losses count:
a run whose loss diverges keeps its best in the past, and that best is what it
contributes to the incumbent. A suggestion has no past, so only its first `H`
points count. On the synthetic prior every curve exists in full, so these
labels are exact, and every training step gets O(N²) pairs.

**Fidelity cap.** The ifBO prior can simulate training beyond `t = 1` (one
nominal full training), so on the prior runs are never capped: a run can take
the whole budget. Only the real tabular benchmarks stop at their maximum
fidelity. There, a finished run can't be picked, and its value is its best
point up to that cap.

**Policy.** `softmax(s / τ)` is sampled as a multinomial. It is not greedy.

**DAgger-like training.** The same stochastic policy drives the environments
it is trained on. There is one environment per batch row and an environment
resets as soon as its episode ends. There is no replay buffer, and nothing else
is taken from the DAgger paper.

**Horizon knob.** The global token reports a horizon `H = f(progress) ·
remaining`, which can under-report the true remaining budget. These labels are
pseudo-labels: they assume all of `H` goes to one item, which almost never
happens. The temperature `τ` and `f` are the knobs to tune.

**Readout.** By default the global token goes through the trunk and a small head maps each
final embedding to a score. With `model.readout: mlp` the trunk sees only runs and
suggestions, and the score is `MLP([embedding, global features])`. The trunk output then
doesn't depend on the horizon, so `train.n_horizons: K` reads each state out at K horizons (the
reported one plus K−1 log-uniform in [1, remaining]) and trains on all K exact label sets.

**Tokens.** One global token carries the budget, elapsed, remaining and
horizon, normalised by the steps per full training and by the budget. There is
one token per run and one per suggestion. Each has type flags, the
hyperparameters in [0, 1] with a dimension mask, and its fidelity so far. Runs
also get curve features (last 10 points, a 16-point resample, best/last/slope),
normalised across all visible runs of that environment, so the loss scale
doesn't matter.

## Data

| Use | Source | State |
|---|---|---|
| Training | ifBO curve prior, ported to batched GPU torch (`roguedagger/prior.py`) | ✅ |
| Eval | TaskSet, 12 NLP tasks × {4p, 8p} = 24 tasks, same as ifBO | ✅ `scripts/download_taskset.py` |
| Eval | PD1 tabular, the 28 task variants ifBO used | ✅ `scripts/process_pd1.py` |
| Eval | LCBench, 35 tasks × first 1000 configs | ✅ `data_2k.zip` placed by hand (figshare refuses automated downloads), `scripts/process_lcbench.py` |

## Baseline

The baseline is FT-PFN + MFPI-random, following the ifBO paper (ICML'24,
branch `icml-2024`). It uses a budget of 1000 steps per task, 10 seeds, a
random first config, and regret normalised to [0, 1]. On tabular benchmarks
ifBO scores *every* unseen config in the table. Our shop is a random subset of
those configs, so we also run a "full-table shop" variant as a like-for-like
comparison.

The FT-PFN curve ids are table index + 1, as in ifBO's experiment code: unique per
config and within the model's 1000-id embedding. The `ifbo` package is vendored in `third_party/ifbo` (MIT). It has a
`from __future__ import annotations` patch so it runs on the shared Python 3.9
venv, and `submitit` is made optional. The weights go in `data/ftpfn/`.

## Layout

```
roguedagger/    package: prior.py, env.py, features.py, model.py, train.py (pairwise), rl.py (PPO), bench.py, evaluate.py
docs/           rl.md: design of the RL extension
configs/        YAML configs for training / eval
scripts/        data download + processing
third_party/    vendored ifbo (FT-PFN)
data/           downloaded data + weights (git-ignored)
runs/           checkpoints + eval outputs (git-ignored)
```

## Running

```bash
source /shared/work/venv/bin/activate
cd /shared/work/private/rogue-dagger
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m roguedagger.train --config configs/base.yaml
```

Training logs to wandb (project `rogue-dagger`). The server is shared: use at
most 3 GPUs at once, keep at least 2 free, and prefer GPUs 3–6.

Configs: `base.yaml` (uniform shop, full horizon), `h_anneal.yaml` (the
horizon starts at 30% of the remaining budget and anneals to 100%),
`wide.yaml` (4–300 steps per full training, budgets of 0.5–200 full
trainings, half the episodes capped at maximum fidelity, which together cover
PD1's range).

Evaluation (writes `runs/eval/<family>/<tag>.npz`):

```bash
python scripts/download_taskset.py data/taskset          # 24 TaskSet tasks
python scripts/process_pd1.py data/pd1/pd1 data/pd1      # after untarring pd1.tar.gz
python -m roguedagger.evaluate --method rd --ckpt runs/base-v1/last.pt --family taskset --device cuda --tag rd-base
python -m roguedagger.evaluate --method ifbo --family taskset --workers 64    # FT-PFN, CPU, ~200 GB RAM at 64 workers
python -m roguedagger.evaluate --method random_search --family taskset
python scripts/compare.py taskset ifbo random_search rd-base --ref ifbo --plot
```

`--n-shop 0` gives our policy the whole unseen table as its shop, the setting
ifBO uses. `--shop-mode mixed` turns half the shop slots into mutations of the
top runs, snapped to the nearest unseen config.

See [BACKLOG.md](BACKLOG.md) for what is done and what is open.

## Results so far (2026-11-12, work in progress)

Protocol from the ifBO paper: 1000 steps per task, 10 seeds, the first config
random. Regret is normalised by the table's best and worst value (stricter than
the paper's normalisation across algorithms). Numbers are mean final regret,
lower is better. Brackets: tasks won against FT-PFN, and the paired gain against
FT-PFN with a 95% bootstrap CI over tasks (positive = better than FT-PFN).

| Method | TaskSet (24) | PD1 (28) | LCBench (35) |
|---|---|---|---|
| FT-PFN + MFPI-random (ifBO) | **0.053** | 0.0151 | **0.0076** |
| Random search | 0.340 | 0.0496 | 0.0625 |
| R&D `wide-v1` @100k, shop 32, τ=1 | 0.095 (4/24; −0.042 [−0.063, −0.022]) | 0.0430 | 0.0227 |
| R&D `wide-v1` @100k, full table, τ=0.25 | 0.062 (10/24; −0.008 [−0.026, +0.009]) | **0.0098** (14/28; +0.005 [+0.002, +0.010]) | 0.0162 (5/35; −0.009 [−0.013, −0.005]) |
| R&D `wide-v1` @100k, full table, τ=0.15 | **0.051** (11/24; +0.003 [−0.012, +0.018]) | **0.0069** (16/28; +0.008 [+0.004, +0.013]) | 0.0181 (7/35; −0.011 [−0.016, −0.006]) |
| R&D `wide-v1` @100k, full table, τ=0.1 | 0.051 (13/24; +0.003 [−0.010, +0.016]) | 0.0058 (17/28; +0.009 [+0.004, +0.015]) | 0.0218 |
| R&D `wide-focus-v1` @100k (pair weighting), full table, τ=0.25 | 0.060 (11/24; −0.006 [−0.027, +0.015]) | 0.0169 (8/28; −0.002 [−0.008, +0.005]) | 0.0092 (10/35; −0.002 [−0.003, +0.000]) |
| R&D `wide-long-v1` @300k, full table, τ=0.25 | 0.070 (8/24; −0.016 [−0.029, −0.005]) | 0.0142 (12/28; +0.001 [−0.003, +0.005]) | 0.0311 (3/35; −0.024 [−0.036, −0.013]) |
| R&D `wide-bigshop-v1` @100k (train shops to 256), full table, τ=0.25 | **0.049** (13/24; +0.004 [−0.015, +0.023]) | 0.0191 (9/28; −0.004 [−0.008, −0.001]) | 0.0119 (5/35; −0.004 [−0.006, −0.003]) |
| R&D `wide-t025-v1` @100k (train τ=0.25), full table, τ=0.25 | 0.090 (5/24; −0.037 [−0.064, −0.015]) | 0.0141 (11/28; +0.001 [−0.002, +0.004]) | 0.0198 (3/35; −0.012 [−0.018, −0.007]) |
| R&D `wide-readout-v1` @100k (MLP readout), full table, τ=0.25 | pending | pending | 0.0238 (9/35; −0.016 [−0.033, −0.003]) |
| R&D `wide-hanneal-v1` @100k, full table, τ=0.25 | 0.059 | 0.0079 | 0.0165 |
| R&D `wide-v1` @55k, full table, τ=0.25 | 0.170 | 0.0113 | – |

* With the full table as its shop (ifBO's setting): significantly better than FT-PFN on PD1,
  on par on TaskSet, and still behind on LCBench (0.016–0.022 vs 0.008).
* **Choosing τ fairly:** the τ=0.15/0.1 rows were swept on the benchmarks themselves and are optimistic
  (lower τ helps TaskSet and PD1, hurts LCBench). On held-out synthetic tasks (`scripts/synth_tau.py`, seed 777,
  T ∈ {20, 50, 200}, budget 1000) the best τ for a 512-item shop is 0.15–0.25 and for a 32-item shop 0.25–0.5.
  τ=0.25 is therefore the fair headline setting: **PD1 better than FT-PFN, TaskSet on par, LCBench behind.**
* Pair weighting (`wide-focus-v1`, `pair_low: 0.1`: pairs where neither item beats the incumbent get 1/10
  weight), paired against `wide-v1`: LCBench +0.007 [+0.004, +0.011] (26/35 tasks, now on par with FT-PFN),
  TaskSet +0.002 [−0.008, +0.013], PD1 −0.007 [−0.012, −0.003] (worse). A trade-off, not a free gain.
* Scale / training-distribution runs (2026-11-13), paired against `wide-v1` (full table, τ=0.25):
  - **300k iterations** (`wide-long-v1`) is worse everywhere: TaskSet −0.008 [−0.020, +0.005], PD1 −0.004
    [−0.010, −0.000], LCBench −0.015 [−0.025, −0.007]. Longer training on the prior overfits it. 55k → 100k
    helped because of the extra steps or the LR decay, not because more is better.
  - **Train shops up to 256** (`wide-bigshop-v1`): TaskSet +0.013 [+0.001, +0.024] (19/24) and LCBench +0.004
    [+0.001, +0.008] are better, PD1 −0.009 [−0.016, −0.004] is worse. Another trade-off, like pair weighting.
  - **Training rollouts at τ=0.25** (`wide-t025-v1`) is worse everywhere: TaskSet −0.029, PD1 −0.004, LCBench
    −0.004 (all significant). At τ=0.25 the rollouts start about half as many runs (5–7 % of steps vs 10–14 %),
    so the scorer sees fewer, harder pairs (≈4.5k vs 12k per batch).
* Horizon anneal (`wide-hanneal-v1`) is within noise of the full horizon on all three families.
* Going from 55k to 100k training steps cut TaskSet regret from 0.170 to 0.062, so the policy was undertrained.
* Deployment knobs (TaskSet, early `wide-v1` snapshot): sampling beats greedy by about 3× in regret
  (0.13 vs 0.43–0.47). The best temperature depends on shop size: τ=1 for 32 items, τ≈0.25 for the full table.
* Label ablation `wide-future-v1` (`label_past: false`, only the next `H` points count), paired against
  `wide-v1` at full table, τ=0.25: TaskSet +0.009 [−0.003, +0.023] (not significant), PD1 −0.002
  [−0.004, +0.000], LCBench −0.004 [−0.007, −0.001] (significantly worse). Keeping past losses in the label stays the default.
* Run counts aren't the gap: FT-PFN starts 41–293 runs on PD1, we start a similar number.

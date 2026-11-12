# Rogue&Dagger HPO

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
| Eval | PD1 tabular | raw tarball downloaded, processing TODO |
| Eval | LCBench (35 tasks) | ❌ figshare refuses automated downloads; place `data_2k.zip` in `data/lcbench/` by hand |

## Baseline

The baseline is FT-PFN + MFPI-random, following the ifBO paper (ICML'24,
branch `icml-2024`). It uses a budget of 1000 steps per task, 10 seeds, a
random first config, and regret normalised to [0, 1]. On tabular benchmarks
ifBO scores *every* unseen config in the table. Our shop is a random subset of
those configs, so we also run a "full-table shop" variant as a like-for-like
comparison.

The `ifbo` package is vendored in `third_party/ifbo` (MIT). It has a
`from __future__ import annotations` patch so it runs on the shared Python 3.9
venv, and `submitit` is made optional. The weights go in `data/ftpfn/`.

## Layout

```
roguedagger/    package: prior.py, env.py, features.py, model.py, train.py, bench.py, evaluate.py
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

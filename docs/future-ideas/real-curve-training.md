# Training on real learning curves (or a better prior)

Status: idea, not queued. Origin: key finding 2 in [../journal.md](../journal.md) (user, 2026-11-15). RL is shelved;
the user may unshelve it at any time.

## Why

RL exploits structure of the ifBO prior that real curves don't have: the more a run improves on the prior, the
worse it transfers. The pairwise trade-offs between benchmark families (key finding 3) may be a milder form of the
same limit. Two ways forward:
- (a) train on real learning-curve data, not only the prior;
- (b) improve the prior so that its exploitable structure matches real curves.

## Data for (a)

- **Full TaskSet:** about 1000 tasks (1162 per ifBO), epoch curves for many optimizer configs. We evaluate on the
  24-task ifBO subset, so the rest can be training data with the 24 held out. Our loader already exists, and
  `scripts/download_taskset.py` would need to fetch more tasks. Risk: training on the same family as one benchmark;
  TaskSet results would then need a caveat or a different held-out split.
- **Smaller epoch-curve sets:** Quick-Tune meta-dataset (fine-tuning), NAS-Bench-201, HPOBench FCNet, and the
  surrogates JAHS-Bench-201 and YAHPO (which carry their own model bias).

Use: RL fine-tuning on real tasks, or pairwise training on a mix of prior and real curves. The tabular env
(`bench.TabEnv`) already plays the real tables.

## Data for (b)

- **LCDB 1.1** (265 OpenML datasets, 32 learners): sample-wise curves (performance vs training-set size), so it
  can't be freeze-thaw training data. But "about 14 % of curves are ill-behaved" is the kind of statistic a better
  prior would need to match.

## Open question

How to select models without a held-out set of real tasks. A split of the real training tasks (e.g. part of the
full TaskSet) as validation would also fix the missing model-selection signal that made RL unusable.

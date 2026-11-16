# Research journal

This is the narrative record: what we tried, why we tried it, what we learned, and which decisions followed.
The numbers live in README ("Results so far"), the checklist in BACKLOG.md, the RL details in docs/rl.md, and the
original spec in PLAN.md. Add a dated entry whenever an experiment finishes or a decision is made, including dead ends.

Conventions: regret = mean final normalised regret after 1000 steps (lower is better); "paired" = per-task
difference with a 95 % bootstrap CI over tasks; full table = the whole unseen table is the shop (ifBO's setting).

## Key findings so far

> **Caveat (2026-11-15):** training-seed noise is about as large as most of the differences below. Every CI here is a
> bootstrap over tasks for one trained model; none includes training-seed variance. Findings 1 and 3 are being re-checked
> with seed replicates (see the 2026-11-15 night results entry). Findings 2 and 4 rest on effects far larger than seed noise.

1. **The pairwise (DAgger-like) scorer is competitive with FT-PFN.** `wide-v1` at τ=0.25 on the full table is
   better than FT-PFN on PD1 (0.0098 vs 0.0151), on par on TaskSet (0.062 vs 0.053, n.s.), and behind on LCBench
   (0.016 vs 0.008). Pair weighting 0.3 (`wide-focus03-v1`) brings PD1 to 0.0059.
2. **The ifBO prior has structure that RL exploits and that doesn't carry over to real learning curves.**
   Every RL variant that improves regret on the synthetic prior transfers worse to the benchmarks, and the more
   it improves on the prior, the worse it transfers:
   - discrete SAC: best synthetic regret of any run (0.029 vs 0.065 for `wide-v1`), worst benchmarks (TaskSet 0.27 vs 0.062);
   - pure PPO: probes 700+ configs for one step each;
   - SAC: starts ≈100 runs and trains them long.
   
   Both behaviours pay on prior curves and fail on real tables. The pairwise loss's proxy label (which item has the
   best loss by len+H) seems to regularise against this, since it doesn't optimise the episode outcome directly.
   **Two ways forward for a working RL version:**
   - (a) train RL on real learning-curve data, not only on the prior;
   - (b) improve the ifBO prior so that its exploitable structure matches real curves.
   
   Either way, the synthetic eval can't be used to select RL models as things stand. If an RL version of the method is
   ever published, this finding belongs in it.
3. **No single pairwise variant wins everywhere.** Each training-distribution change trades one family against
   another. Pair weighting and big shops help LCBench and hurt PD1. Big shops and the MLP readout help TaskSet.
   **Interpretation (user, 2026-11-15):** this looks like sliding along a barrier. Changing the method's biases
   moves real-world performance sideways, not forward, which suggests the pairwise approach is also near the limit
   of what the ifBO prior can teach. Same root cause as finding 2, in a milder form.
   Open caveats that tonight's runs test:
   - Seed noise. If seed replicates move as much as the variants do, the "trades" are partly noise.
   - Coverage gaps inside the prior's own knobs. The PD1 loss tracks curve length, and training T stops at 300,
     below PD1's 1414. Fixing coverage could still move the frontier without new data (`wide-focus03-longT-v1`).
   
   If neither moves it, better data is the way forward for the pairwise method too: real curves, or a better prior.
4. **More training on the prior is not better.** 300k iterations is worse than 100k on all three families, which is
   another sign of overfitting to the prior.

## 2026-11-12: Building the pipeline, first comparison

- Implemented the method from PLAN.md:
  - GPU-batched ifBO prior, checked against the numpy reference;
  - shop env with a fresh random shop every step;
  - set-transformer scorer;
  - pairwise BCE on all pairs;
  - on-policy multinomial rollouts, one env per batch row.
- Label (user's decision, settled): an item's value is its best loss over the curve up to len+H, past included.
  The future-only ablation (`wide-future-v1`) is worse on LCBench and not better elsewhere.
- Benchmarks: TaskSet (24 tasks), PD1 (28 ifBO variants), LCBench (35 tasks, first 1000 configs). Baselines are
  FT-PFN with MFPI-random, and random search.
- 55k → 100k iterations cut TaskSet regret from 0.170 to 0.062, so the first model was undertrained.
- Deployment:
  - Sampling beats greedy by about 3×.
  - The best τ depends on shop size, so τ was chosen fairly on held-out synthetic tasks (`scripts/synth_tau.py`):
    τ=0.25 for the full table, τ≈1 for shop 32.
  - The τ=0.1/0.15 rows in README were swept on the benchmarks themselves and are optimistic.
- Horizon anneal has no effect.

## 2026-11-13: Training-distribution and architecture variants

- **Pair weighting** (`pair_low`): down-weights pairs where neither item beats the incumbent. Weight 0.1 makes
  LCBench as good as FT-PFN but hurts PD1. Weight 0.3 keeps the PD1 gain (best PD1 so far, 0.0059).
- **300k iterations** is worse everywhere: the model overfits the prior.
- **Train shops up to 256**: better on TaskSet and LCBench, worse on PD1.
- **Training rollouts at τ=0.25**: worse everywhere. The rollouts start half as many runs, so the scorer sees
  fewer, harder pairs.
- **MLP readout**: best TaskSet (0.047), worse LCBench.
- **The synthetic eval (shop 32) has not predicted the benchmark ranking well**, even for pairwise models.
- Started the **RL extension** at the user's request (docs/rl.md). Design: PPO fine-tuning from a pairwise
  checkpoint, normalised-regret reward (log, ≥ 0), separate critic, and the pairwise BCE as an auxiliary anchor.
  Ablations: no aux loss, and training from scratch.

## 2026-11-13/14: RL results, then shelving RL

- PPO with the aux loss (`rl-wide-v1`): PD1 gain over `wide-v1` (+0.004), level elsewhere. That is about what
  pair weighting 0.3 gets without RL.
- Pure PPO (`rl-wide-noaux-v1`):
  - At shop 32 it is the best TaskSet model (0.053, 21/24 vs `wide-v1`).
  - On the full table it first collapsed, then recovered to about `wide-v1` level.
  - It drifts toward probing many configs for a single step (TaskSet 749 runs per episode vs 343).
- The user pointed out that discrete SAC beat PPO in the automl-exam project, through better final policies,
  not sample efficiency. Ported the exam's `sac_frozen3` recipe: implicit actor softmax(β·minQ), a perplexity-
  annealed entropy target, mean-reward targets, and a frozen trunk.
- Big shops (4..256) for PPO and SAC. Synthetic regret dropped to 0.027–0.046, the best numbers of the project.
  Benchmarks:
  - pure PPO collapses on TaskSet (0.210);
  - PPO from focus03 loses that model's PD1 gain;
  - SAC is significantly worse on all three families (TaskSet 0.27).
- The pattern behind key finding 2. Why SAC beat PPO in the exam even without a sample limit (hypotheses only,
  untested):
  - per-action Q signal vs one sampled action;
  - an explicit entropy schedule vs our ent_coef 0;
  - γ=0.99 bias vs γ=1 variance;
  - a frozen trunk.
  
  Here the two end up about equally poor on transfer.
- **Decision (user, 2026-11-14): shelve RL**, stay with the pairwise approach, mix training parameters, and
  investigate the model. Reasons are in docs/rl.md, "Status". The code and tests stay in the repo.

## 2026-11-14: Pairwise focus03 on big shops, PD1 diagnosis

- `wide-focus03-big-v1` (pair weighting 0.3 + shops 4..256) vs `wide-v1`: LCBench better (0.0116, +0.005),
  PD1 much worse (0.0205, −0.011; −0.015 vs `wide-focus03-v1`). Same trade as `wide-bigshop`.
- Shop size is already drawn per env (log-uniform over `n_range`), so 4..256 training already mixes small and big
  shops. Only the weighting changes: a third of the episodes have shops over 64.
- **Where PD1 is lost:**
  - Per task, the big-shop deficit correlates with log(budget / curve length) at r = 0.82. It is concentrated on
    long-curve, low-budget tasks: translate T=1414/706, imagenet T=250/501.
  - PD1's T goes up to 1414, but training `T_range` stops at 300, so those tasks are out of distribution for every model.
- **Deploy-time shop size doesn't explain it.**
  - `wide-focus03-big-v1` scores 0.0205 at shop 64, at shop 128 and on the full table.
  - At shop 64 it starts as many runs as focus03 does on the full table (111 vs 110) and is still 3× worse.
  - So it's learned behaviour (it picks worse), not a deploy mismatch or a run-count difference. It's worse on
    short-curve PD1 tasks too (0.0083 vs 0.0015).
- **Nothing measured so far has been checked against seed noise.** That matters, because several conclusions
  rest on differences of about 0.004.
- Night runs, each evaluated automatically when it finishes (`runs/logs/night_2026-11-14.sh`):

  | Run | What it tests |
  |---|---|
  | `wide-focus03-s1`, `wide-s1` | Seed-1 replicates: training noise |
  | `wide-focus03-longT-v1` | T_range 4..1500: curve length alone |
  | `wide-focus03-mid-v1` | Shops 4..128 |

## 2026-11-15: Other learning-curve data

- **Full TaskSet** has about 1000 tasks (1162 per ifBO); we evaluate on the 24-task ifBO subset. The rest is the
  natural source of real training curves, for RL (way (a) of finding 2) or for mixing real and prior curves in
  pairwise training, with the 24 benchmark tasks held out. Risk: training on the same family as one benchmark.
- **LCDB 1.1** (265 OpenML datasets, 32 learners): the largest learning-curve database, but sample-wise
  (performance vs training-set size), not epoch curves. Not usable as freeze-thaw data. Possibly useful for
  way (b): its "about 14 % of curves are ill-behaved" is the kind of statistic a better prior would have to match.
- **Smaller epoch-curve sets**: Quick-Tune meta-dataset (fine-tuning), NAS-Bench-201, HPOBench FCNet, and the
  surrogates JAHS-Bench-201 and YAHPO.
- RL stays shelved for now; the user may unshelve it at any time.
- **Idea (user): a random rollout temperature per env.** Why it fits this method:
  - The pairwise label (best loss by len+H) doesn't depend on the policy. Mixing τ therefore only changes which
    states the scorer sees, never what the right answer is.
  - Training at τ=1 and deploying at 0.25 is itself a train/deploy state mismatch, the thing DAgger-style
    training is meant to avoid.
  - `wide-t025-v1` (τ=0.25 everywhere) was worse because the rollouts started half as many runs and gave half
    as many pairs. A mixture keeps most of the pair supply.
  
  Implemented as `train.temperature_range`: log-uniform per env, redrawn at reset. Two ranges, both queued:
  - `wide-focus03-tmix-v1` (τ 0.25..2): GPU 3, after the seed replicate.
  - `wide-focus03-tmixcold-v1` (τ 0.1..1): GPU 4, after longT. The user's preference: 0.25 is the deploy optimum
    on the real tables, and τ=2 is untested, so 0.25..2 is quite hot.

## 2026-11-15: Ideas discussed (now in docs/future-ideas/)

Design discussion with the user, no runs. Each idea has its own file in [future-ideas/](future-ideas/README.md):
- **Crossing-aware horizons** ([crossing-horizons.md](future-ideas/crossing-horizons.md)). A pair's label carries
  shape information only where the curves cross. Simplest form: an equality term (scores tied) at the crossing horizon.
- **The curve distribution is implicit** ([implicit-distribution.md](future-ideas/implicit-distribution.md)). Unlike
  FT-PFN's histogram head, we train binary tests and leave the representation of uncertainty to the network.
  - The finding: the Bradley–Terry readout can't represent every set of pairwise probabilities. A small margin is
    decisive against a safe item and irrelevant against a risky one, and one scalar per item can't encode both.
    Attention can absorb dependence shared across a state, but not this.
  - Diagnostic tests on a frozen model are written up there.
- **Thurstone readout + Thompson sampling** ([thurstone-thompson.md](future-ideas/thurstone-thompson.md)). A (μ, σ)
  per token fixes the BT example, and sampling gives P(best) without a temperature. Main risk (user): extra degrees of
  freedom. σ is weakly identified by binary pairs and would drive exploration, the same pattern as RL exploiting the
  prior. Gated on a held-out comparison of BT vs Thurstone heads.
- **Real-curve training** ([real-curve-training.md](future-ideas/real-curve-training.md)): the two ways forward
  from key finding 2, and the data for each.

## 2026-11-15: Night results. Seed noise is as large as the effects we've been reading

Seed-1 replicates, same config, only `seed` changed. Paired vs seed 0, full table, τ=0.25 (positive = seed 1 better):

| Family | `wide` s1 vs s0 | `wide-focus03` s1 vs s0 |
|---|---|---|
| TaskSet | 0.055 vs 0.062, +0.007 [−0.003, +0.018] | 0.047 vs 0.064, **+0.017 [+0.004, +0.033]** |
| PD1 | 0.0229 vs 0.0098, **−0.013 [−0.023, −0.005]** | 0.0117 vs 0.0059, **−0.006 [−0.009, −0.002]** |
| LCBench | 0.0209 vs 0.0162, −0.005 [−0.011, +0.000] | 0.0130 vs 0.0140, +0.001 [−0.002, +0.004] |

- **Identical configs give "significant" differences.** The task-bootstrap CI doesn't cover training noise. Seed-to-seed
  differences reach about 0.017 on TaskSet, 0.013 on PD1 and 0.005 on LCBench. Most variant effects we recorded
  (0.002–0.013) are inside that range.
- **The PD1 headline isn't robust for `wide`:** seed 1 is worse than FT-PFN (0.0229 vs 0.0151). For `wide-focus03`
  both seeds beat FT-PFN on PD1 (0.0059, 0.0117) and both are behind on LCBench (0.014, 0.013 vs 0.0076). TaskSet
  straddles FT-PFN (0.064, 0.047 vs 0.053).
- **The 2026-11-14 PD1 diagnosis was mostly seed noise.** The seed-1 replicate of `wide` also loses mainly on the
  long-curve, low-budget PD1 tasks (corr with log(budget/T) = 0.57; −0.046 on the 6 tasks with < 5 full trainings
  vs −0.004 on the rest). Those 6 tasks are where training seeds disagree most, so "big shops hurt long-curve PD1
  tasks" can't be separated from "those tasks are high-variance". The deploy-shop-size result still holds (it's
  within one model).
- **Variants** (vs `wide-focus03-v1` seed 0, full table):
  - `wide-focus03-longT-v1` (T up to 1500): PD1 −0.017 [−0.028, −0.009], also on the long tasks (−0.049);
    LCBench −0.004; TaskSet ≈. Extending T didn't help the long-curve tasks; if anything it hurt. Borderline against
    seed noise.
  - `wide-focus03-mid-v1` (shops 4..128): PD1 −0.003 (n.s.), LCBench −0.007 [−0.011, −0.003], TaskSet ≈.
    Inside or near seed noise.
- **Consequences:**
  - Comparisons need several training seeds per config, and a CI that resamples seeds as well as tasks.
  - The "sliding along a barrier" reading (key finding 3) may be partly noise: variants may move results less than
    it seemed. That needs re-checking with seeds before acting on it.
  - Seed-2 replicates of `wide` and `wide-focus03` are now running (GPUs 5, 6) to get 3 seeds of each reference.
- **Seed-aware comparison** (`scripts/compare_seeds.py`: per-config mean over training seeds; bootstrap over tasks
  and seeds). With 2 seeds each, full table, τ=0.25:

  | Family | FT-PFN | `wide` (2 seeds) | `wide-focus03` (2 seeds) |
  |---|---|---|---|
  | TaskSet | 0.053 | 0.058 (−0.005 [−0.022, +0.012]) | 0.055 (−0.002 [−0.024, +0.019]) |
  | PD1 | 0.0151 | 0.0164 (−0.001 [−0.013, +0.008]) | **0.0088 (+0.006 [+0.001, +0.012])** |
  | LCBench | 0.0076 | 0.0185 (−0.011 [−0.018, −0.005]) | 0.0135 (−0.006 [−0.010, −0.002]) |

  Brackets: gain vs FT-PFN (positive = better), seeds resampled. `wide-focus03` vs `wide`: PD1 +0.008
  [−0.002, +0.021], LCBench **+0.005 [+0.001, +0.011]**, TaskSet +0.003 (n.s.).

  Revised headline: `wide-focus03` beats FT-PFN on PD1, is on par on TaskSet, and is behind on LCBench. Pair
  weighting 0.3 is better than plain `wide` on LCBench (and probably PD1). With only 2 seeds the seed level of the
  bootstrap is crude.

## 2026-11-15: Cold rollout-temperature mix hurts (pair starvation)

`wide-focus03-tmixcold-v1`: `wide-focus03` with a per-env rollout temperature drawn log-uniform in [0.1, 1] at each
env reset (the reference trains at τ=1). One seed, vs the 2 `wide-focus03` seeds (`scripts/compare_seeds.py`):

| Family | `wide-focus03` (2 seeds) | tmixcold | gain, seeds resampled |
|---|---|---|---|
| TaskSet, full τ=0.25 | 0.055 | 0.092 | **−0.036 [−0.058, −0.014]**, better on 2/24 |
| PD1, full τ=0.25 | 0.0088 | 0.0112 | −0.002 (n.s.) |
| LCBench, full τ=0.25 | 0.0135 | 0.0268 | **−0.013 [−0.020, −0.007]**, better on 5/35 |
| TaskSet, shop 32 τ=1 | 0.085 | 0.105 | −0.020 [−0.038, −0.002] |
| PD1 / LCBench, shop 32 τ=1 | 0.030 / 0.024 | 0.029 / 0.026 | n.s. |

The TaskSet and LCBench losses are 2-3 times the largest seed-to-seed difference we've seen (0.017 / 0.005), so
this is a real effect despite one seed. Likely cause, from the training logs (mean of the last 2000 iters):
colder rollouts start fewer runs (`frac_new` 0.087 vs 0.13; 7.4 vs 12 full trainings per budget) and the batch has
about half the labelled pairs (7200 vs 13000 per env). Colder rollouts concentrate on fewer, longer runs, so the
scorer sees fewer and less varied comparisons: the pair starvation the config was meant to avoid, from the other
side. Labels don't depend on the policy, but the state distribution and the pair count do. The full-table τ=0.25
deploy setting didn't benefit from seeing cold states in training.

Conclusion: keep training rollouts at τ=1 (or hotter). The hot mix (0.25..2) is being evaluated.

## 2026-11-15: Hot rollout-temperature mix also hurts TaskSet

`wide-focus03-tmix-v1`: per-env rollout τ log-uniform in [0.25, 2] (redrawn at each env reset). One seed, vs the
2 `wide-focus03` seeds:

| Family | `wide-focus03` (2 seeds) | tmix | gain, seeds resampled |
|---|---|---|---|
| TaskSet, full τ=0.25 | 0.055 | 0.104 | **−0.049 [−0.093, −0.014]**, better on 6/24 |
| PD1, full τ=0.25 | 0.0088 | 0.0127 | −0.004 (n.s.) |
| LCBench, full τ=0.25 | 0.0135 | 0.0107 | +0.003 [+0.000, +0.006], better on 25/35 |
| TaskSet, shop 32 τ=1 | 0.085 | 0.095 | −0.010 (n.s.) |
| PD1, shop 32 τ=1 | 0.030 | 0.056 | **−0.026 [−0.047, −0.009]** |
| LCBench, shop 32 τ=1 | 0.024 | 0.020 | +0.004 [+0.000, +0.009] |

The TaskSet and PD1-shop-32 losses are far outside seed noise; the LCBench gain is within the seed-to-seed range
(0.005), so not a result. Training logs: 10700 labelled pairs per env vs 13000 for the reference, `frac_new` 0.116 vs
0.13. The hot mix also loses pairs (cold envs in the lower half of the range), but less than the cold mix, and still
loses more on TaskSet, so pair count alone doesn't explain it.

Conclusion: both τ mixes are worse. Training at a fixed τ=1 stays. The per-env temperature code
(`train.temperature_range`) stays in as an option, off by default.

## 2026-11-15: BT vs Thurstone on frozen embeddings: the gate fails

Test 3 of [future-ideas/implicit-distribution.md](future-ideas/implicit-distribution.md), `scripts/head_test.py`
(log: `runs/logs/head_test2.log`). Freeze a model, roll out its own policy on the prior (τ=1, 256 envs, 240
snapshots, up to 48 items per state), fit fresh heads on the final per-item embeddings. Env rows split 2:1:1 into
train / validation (early stopping) / test. Test = pairwise BCE with pair weights 0.3 as in training (`bce_w`).

| Model | own head | BT refit | BT from own | Thurstone | Thurstone from own |
|---|---|---|---|---|---|
| `wide-focus03-v1` | 0.5684 | **0.5665** | 0.5678 | 0.5739 | 0.5711 |
| `wide-v1` | 0.5956 | **0.5826** | (= own) | 0.5993 | 0.6021 |
| `wide-readout-v1`* | n/a | **0.5630** | n/a | 0.5640 | n/a |

\* MLP readout: the stored embeddings don't include the global features, so both heads lack horizon/budget input.
"From own": initialised from the trained head (Thurstone: μ = trained score, constant σ ≈ 1.2 so it starts equal
to the BT head). Test accuracy is 0.72-0.74 for every head.

- Thurstone is never better on held-out pairs (0.001-0.008 worse), even when it starts from the BT solution. On the
  training rows it fits slightly better, so the extra freedom only overfits: the user's identifiability concern.
- The learned σ is almost flat (relative σ 0.97-1.03 by item type for `wide-focus03`, exactly 1.000 for `wide-v1`).
  Only `wide-readout-v1` shows the expected trend, weakly (shop items 0.97, short runs 1.03, long runs 0.91).
- Refit BT heads gain little over the trained head (≤ 0.013), so the embeddings, not the head, set the limit.

Caveat: the embeddings were trained under a BT readout and may not carry spread information that a Thurstone
readout would need; an end-to-end Thurstone model could differ. Test 1 (probing the embeddings for the spread of
the future best loss) would tell whether that information is there.

Conclusion: the gate for [future-ideas/thurstone-thompson.md](future-ideas/thurstone-thompson.md) fails on frozen
embeddings. No end-to-end Thurstone ablation unless test 1 shows the spread is represented.

## 2026-11-15: Embedding probes (test 1): the spread is represented, weakly

`scripts/head_test.py --probe` (log: `runs/logs/probe_test.log`), same data as the BT/Thurstone test. Linear ridge
probes from frozen embeddings to (i) the label (future best loss up to len+H, z-scored within each state so task
scale drops out) and (ii) the squared error of probe (i), a per-item spread proxy. The prior gives one future per
item, so a true conditional spread isn't available; squared error of one draw is a very noisy target (even a perfect
σ² predictor correlates well below 1 with it). Baseline: the same probes on item type only (shop flag, run-progress
bins). Test rows held out.

| Model | label R² (emb / type) | spread R² (emb / type) | corr(pred spread, sq. error) (emb / type) |
|---|---|---|---|
| `wide-focus03-v1` | 0.337 / 0.053 | 0.041 / 0.005 | 0.202 / 0.069 |
| `wide-v1` | 0.334 / 0.051 | 0.040 / 0.004 | 0.205 / 0.069 |
| `wide-readout-v1` | 0.395 / 0.058 | 0.047 / 0.005 | 0.217 / 0.074 |

- The within-state outcome is linearly decodable (R² 0.33-0.40, linear only).
- Spread information is there, weakly: the embedding predicts where the probe errs about 3 times better than item
  type does (corr 0.20 vs 0.07). The ceiling for this noisy target is unknown, so "weak" is relative.
- Together with the head test: the embeddings do carry some per-item uncertainty, and an MLP Thurstone head on them
  could have used it, but it didn't improve held-out pair predictions. So per-item spread adds little to pairwise
  decisions on the prior beyond what the BT score already captures.

Conclusion: no end-to-end Thurstone ablation for now; [future-ideas/thurstone-thompson.md](future-ideas/thurstone-thompson.md)
stays parked. Test 2 (calibration vs FT-PFN) remains open.

## 2026-11-15: Crossing tie term implemented, ablation launched

Variant 1 of [future-ideas/crossing-horizons.md](future-ideas/crossing-horizons.md), as `train.cross_weight`
(MLP readout only; `roguedagger/train.py: crossing_pairs`). Per step and env, 64 random item pairs; where a pair's
label (best loss up to len+h, cap respected) strictly swaps sign between h and h+1 within the remaining budget, the
readout at h* = h + 0.5 is trained towards a tie with BCE at label 0.5 (shifted so a tie costs 0), weight 1. With
several crossings one is drawn at random. Test: crossings agree with `env.values` on both sides. On the smoke run
about 36 % of random pairs have a crossing; speed is unchanged (≈ 11-14 it/s).

`wide-readout-cross-v1` (seed 0) vs `wide-readout-v1` (seed 0) changes only this term. A single seed against a single
seed can only show effects well beyond the seed noise (≈ 0.017 TaskSet / 0.013 PD1 / 0.005 LCBench).

## 2026-11-15: Three seeds per reference model: the headline

Seed-2 replicates `wide-s2`, `wide-focus03-s2` done. Per-model full-table τ=0.25 regret (seeds 0 / 1 / 2):

| Family | `wide` | `wide-focus03` |
|---|---|---|
| TaskSet | 0.0617 / 0.0547 / 0.0506 | 0.0640 / 0.0468 / 0.0561 |
| PD1 | 0.0098 / 0.0229 / 0.0132 | 0.0059 / 0.0117 / 0.0108 |
| LCBench | 0.0162 / 0.0209 / 0.0113 | 0.0140 / 0.0130 / 0.0102 |

Seed 2 lands inside the range of seeds 0 and 1, except `wide` LCBench (0.0113, best of the three). Seed-aware
comparison (bootstrap over tasks and seeds), gain vs FT-PFN, positive = better:

| Family | FT-PFN | `wide` (3) | `wide-focus03` (3) | focus03 vs wide |
|---|---|---|---|---|
| TaskSet | 0.053 | 0.056 (−0.002 [−0.019, +0.015]) | 0.056 (−0.002 [−0.020, +0.015]) | +0.000 |
| PD1 | 0.0151 | 0.0153 (−0.000 [−0.008, +0.007]) | **0.0095 (+0.006 [+0.001, +0.011])**, 15/28 | +0.006 [−0.001, +0.016], 21/28 |
| LCBench | 0.0076 | 0.0161 (−0.009 [−0.016, −0.004]) | 0.0124 (−0.005 [−0.009, −0.001]) | +0.004 [−0.001, +0.010], 22/35 |

Shop 32, τ=1, focus03 vs wide: TaskSet +0.000, PD1 +0.011 [−0.001, +0.025] (18/28), LCBench −0.001.

- Headline holds with 3 seeds: `wide-focus03` beats FT-PFN on PD1 (CI excludes 0), ties on TaskSet, is behind on
  LCBench (gap 0.005, smaller than `wide`'s 0.009).
- `wide-focus03` vs `wide`: consistently ahead on PD1 and LCBench, but with 3 seeds both CIs now just include 0
  (the 2-seed LCBench CI excluded it). Call it "probably better", not established.
- `wide-focus03` has the smaller seed spread on PD1 (0.006 vs 0.013) and LCBench.

## 2026-11-15: User asks for a Thurstone run from scratch, crossing variant 2 and a table of all models

- **Thurstone from scratch** (the frozen-embedding test can't rule out a model trained for it end to end):
  `model.pair_model: thurstone`. The head outputs (μ, σ = softplus + 0.01), loss is probit pairwise BCE with the same
  labels and pair weights, policy is Thompson sampling (one draw μ + τ·σ·ε per item, argmax; τ = σ scale, 1 = exact
  Thompson, 0 = greedy on μ). Config `wide_focus03_thurstone.yaml`: `wide-focus03` + this, because focus03 is the
  3-seed headline model (the idea doc proposed the MLP readout; the token readout works the same way). Readout and
  policy change together, because Thompson sampling needs σ. Evals: shop 32 at τ 1, full table at τ 1 and 0.5.
  Two seeds (`wide-focus03-thurstone-v1`, `-s1`).
- **Crossing variant 2** (user: "both points around a crossing, after all it's discrete"): `train.cross_mode: sides`.
  At each sampled crossing, the readout at h and h+1 is trained on the ordinary labels there, which are opposite.
  The two horizons differ by about 1/T in the horizon features, so the best fit at the crossing is close to 0.5 on
  both sides, as with the tie; what the sides version adds is the direction (margin up before, down after), i.e.
  the sign of the slope in h. Two seeds (`wide-readout-cross-sides-v1`, `-s1`).
- **Table of all models:** [results.md](results.md), generated by `scripts/results_table.py` (registry of every
  model with the eval tags of its seeds; gains vs FT-PFN with the seed-aware bootstrap).
- Queue (`runs/logs/queue_thurst_sides.sh`): each starts when the GPU's current chain has written its evals.
  GPU 3: sides seed 0; GPU 4: sides seed 1; GPU 5: Thurstone seed 0; GPU 6: Thurstone seed 1.

## 2026-11-15: Crossing tie term (variant 1): no gain on the full table, worse on shop 32

2 seeds each: `wide-readout-v1`, `wide-readout-s1` vs `wide-readout-cross-v1`, `wide-readout-cross-s1`
(`compare_seeds.py`, gain of cross over readout, positive = better):

| Setting | TaskSet | PD1 | LCBench |
|---|---|---|---|
| full, τ 0.25 | +0.002 [−0.018, +0.023] | +0.002 [−0.006, +0.013] | −0.001 [−0.014, +0.014] |
| shop 32, τ 1 | **−0.021 [−0.043, −0.002]**, 4/24 | **−0.012 [−0.020, −0.005]**, 3/28 | −0.002 (n.s.) |

Per seed (readout / cross), full table: TaskSet 0.047, 0.037 / 0.048, 0.032; PD1 0.012, 0.020 / 0.010, 0.017;
LCBench 0.024, 0.012 / 0.025, 0.014. Shop 32: TaskSet 0.083, 0.060 / 0.094, 0.089; PD1 0.049, 0.049 / 0.061, 0.060.

- Full table: nothing. The two readout seeds alone differ by 0.012 on LCBench and 0.008 on PD1.
- Shop 32: both cross seeds are worse than both readout seeds on TaskSet and PD1.
- Not from run counts: shop-32 runs started per task are about the same (117-163 vs 130-155). Training entropy at τ 1
  is slightly higher with the tie term (3.26 vs 3.16), so the scores are a bit flatter, which a τ=1 deploy would
  feel more than τ=0.25. Plausible but not shown.
- Side note: the readout model's own seed spread (LCBench 0.024 vs 0.012) is the largest seen so far, so the
  single-seed `wide-readout-v1` results in the README (TaskSet best, LCBench worst) were partly seed luck.

Conclusion: the tie term doesn't help. Variant 2 (`sides`) is training; it differs in that it also trains the
direction of the margin's change across the crossing.

## 2026-11-15: Sim-to-real gap of the prior: first measurement, new future idea

User: before improving the ifBO prior we need to measure the sim-to-real gap. Plan written up in
[future-ideas/prior-sim2real.md](future-ideas/prior-sim2real.md) (decision-, model- and data-level measures, dev data
to keep the benchmarks clean, then ways to improve the prior).

First, free measurement: Spearman ρ across models between synthetic eval regret (last training-log eval: prior
tasks, T 50, shop 32, τ 1) and benchmark regret at shop 32:

| Models | TaskSet | PD1 | LCBench |
|---|---|---|---|
| all (28) | +0.24 | +0.10 | +0.11 |
| pairwise (21) | +0.75 | **−0.32** | +0.51 |
| RL (7) | −0.18 | **−0.82** | −0.04 |

The prior ranks pairwise models fairly well for TaskSet, moderately for LCBench, inversely for PD1; RL models
inversely (PD1 strongly). Caveat: one noisy eval per model and a single synthetic spec (T 50) that doesn't match
PD1's long curves. Next: matched synthetic twins (same d, T, budget, shop per real task) to separate spec mismatch
from curve mismatch.

## 2026-11-15: Big-shop training, second seed: the PD1 loss is real

`wide-focus03-big-s1` (seed 1 of `wide-focus03-big-v1`, train shops 4..256) finished. Two big-shop seeds vs the three
focus03 seeds (`scripts/compare_seeds.py`, gain = focus03 regret − big regret):

| | TaskSet | PD1 | LCBench |
|---|---|---|---|
| full table, τ 0.25 | +0.002 [−0.030, +0.031] | **−0.016 [−0.030, −0.007]** | +0.001 [−0.001, +0.004] |
| shop 32, τ 1 | **−0.022 [−0.041, −0.004]** | −0.017 [−0.045, +0.005] | −0.001 [−0.004, +0.003] |

- PD1: both big seeds (0.0205, 0.0301) are worse than all three focus03 seeds (0.0059–0.0117), worse on 23/28 tasks.
  The earlier single-seed finding (big shops hurt PD1, mostly on long-curve, low-budget tasks) stands.
- LCBench: the earlier +0.005 gain of big over `wide-v1` shrinks to +0.001 n.s. against focus03's three seeds.
- TaskSet full table: the two big seeds are 0.0735 and 0.0334, the largest seed gap seen on TaskSet; no conclusion.

Conclusion: training on shops up to 256 doesn't pay off; keep shops 4..64. Closes the backlog item.

## 2026-11-15: Thurstone readout from scratch: negative, closed

User's request: train the (μ, σ) Thurstone readout with the Thompson policy from scratch before discarding it
(`wide-focus03-thurstone-v1`/`-s1`, base `wide-focus03`, train σ scale 1). Two seeds vs the three focus03 seeds
(full table, focus03 at τ 0.25; gain = focus03 − Thurstone, seed-aware CIs):

| σ scale | TaskSet | PD1 | LCBench |
|---|---|---|---|
| 1.0 | 0.173 (−0.117 ▼) | 0.061 (−0.051 ▼) | 0.026 (−0.014 ▼) |
| 0.5 | 0.077 (−0.021) | 0.036 (−0.026 ▼) | 0.0137 (−0.001) |
| 0.25 | 0.062 (−0.007) | 0.022 (−0.012 ▼) | 0.0158 (−0.003) |
| 0.1 | 0.128 (−0.072 ▼) | 0.017 (−0.007 ▼) | 0.024 (−0.012) |
| focus03 | 0.056 | 0.0095 | 0.0124 |

Shop 32 at σ scale 1: TaskSet 0.073 vs 0.085 (+0.013 n.s.), PD1 0.056 vs 0.032 (−0.024 ▼), LCBench 0.018 vs 0.023
(+0.005 n.s.).

- Even with σ picked on the benchmarks (optimistic), no setting beats focus03 anywhere; the best (0.25) ties on
  TaskSet and LCBench and loses PD1. TaskSet and PD1 want different σ scales (0.25 vs 0.1).
- What σ learned (256 prior episodes, policy at scale 1): not flat this time (coefficient of variation over the
  items of a state 0.30-0.32, unlike the frozen-embedding heads), larger for shop items than for runs (0.24-0.27 vs
  0.34-0.41, sensible), but strongly tied to μ (|corr(μ, σ)| 0.75), so it is largely a function of the score. Mean
  σ ≈ the spread of μ within a state, so Thompson draws at scale 1 reorder items a lot: training rollouts start
  about 17 runs per budget vs about 10 for focus03.
- The constant-σ ablation planned for the case of a gain isn't needed.

Conclusion: Thurstone + Thompson doesn't beat Bradley–Terry + softmax; closed (future-ideas/thurstone-thompson.md).

## 2026-11-15: Crossing variant 2 (`sides`): negative; crossing idea closed

User's variant 2: "not an equal loss, but rather both points around a crossing. after all it's discrete." At each
mined crossing of a pair's label margin, ordinary BCE labels at the horizons on both sides (h* ± 0.5)
(`train.cross_mode: sides`, `wide-readout-cross-sides-v1`/`-s1`). Two seeds each vs `wide-readout` (2 seeds), gain =
readout − variant:

| | TaskSet | PD1 | LCBench |
|---|---|---|---|
| sides, full table | −0.009 [−0.032, +0.007] | **−0.007 [−0.017, −0.000]** | +0.001 [−0.008, +0.016] |
| sides, shop 32 | −0.004 [−0.026, +0.018] | −0.005 [−0.021, +0.009] | −0.004 [−0.010, +0.001] |
| tie (variant 1), full table | +0.002 | +0.002 | −0.001 |
| tie (variant 1), shop 32 | **−0.021** | **−0.012** | −0.002 |

Variant 2 avoids variant 1's shop-32 losses but gains nothing, and is slightly worse on the PD1 full table. Training
pair accuracy is unchanged (last 20 logs: 0.723 / 0.744 vs readout 0.719 / 0.748), so the extra labels are neither
hard nor helpful.

Conclusion: training at label crossings doesn't help in either form; idea closed
(future-ideas/crossing-horizons.md).

## 2026-11-15: Prior sim-to-real, measurement 3: curve statistics, real vs prior twins

New dev data: 120 TaskSet tasks × {adam4p, adam8p} = 240 tasks (`scripts/taskset_dev_tasks.txt`, 12 per sampled
family + 12 Fixed*, none from the test tasks' family; `data/taskset_dev`, 199 MB). `scripts/curve_stats.py`: per real
task, 4 prior twins with the same d, T and number of configs; per task, statistics of the normalised curves; KS
distance between the real and twin distributions (`runs/logs/curve_stats.log`). Shape statistics use each curve's
own range (a first version used the task's range, which real tasks' many bad configs inflate; that zeroed PD1's
overfitting artificially).

Medians, real vs twin (KS in brackets); dev = 240 TaskSet dev tasks:

| statistic | TaskSet dev | TaskSet test | LCBench | PD1 |
|---|---|---|---|---|
| flat curves (own range < 2% of task's) | 0.44 vs 0.00 (0.94) | 0.51 vs 0.00 (0.99) | 0.06 vs 0.00 (0.59) | 0.08 vs 0.00 (0.55) |
| final spread (q90 − q10, / task range) | 0.98 vs 0.45 (0.83) | 0.71 vs 0.36 (0.46) | 0.43 vs 0.41 (0.26) | 0.96 vs 0.45 (0.91) |
| configs within 5% of the best | 0.11 vs 0.01 (0.45) | 0.002 vs 0.011 (0.49) | 0.04 vs 0.01 (0.31) | 0.23 vs 0.03 (0.87) |
| noise (residual sd / own range) | 0.017 vs 0.072 (0.66) | 0.034 vs 0.082 (0.58) | 0.029 vs 0.077 (0.54) | 0.015 vs 0.068 (0.67) |
| autocorr. of increments | +0.57 vs −0.34 (0.61) | −0.33 vs −0.37 (0.34) | +0.31 vs −0.35 (0.64) | +0.72 vs −0.38 (0.88) |
| up-steps | 0.21 vs 0.46 (0.79) | 0.34 vs 0.47 (0.82) | 0.27 vs 0.47 (0.86) | 0.27 vs 0.47 (0.79) |
| overfit (min before 0.8 T, end >10% above) | 0.15 vs 0.59 (0.69) | 0.18 vs 0.63 (0.75) | 0.32 vs 0.58 (0.49) | 0.00 vs 0.52 (0.81) |
| diverge (end above start) | 0.03 vs 0.19 (0.61) | 0.01 vs 0.21 (0.72) | 0.03 vs 0.21 (0.55) | 0.00 vs 0.15 (0.73) |
| rank corr. running best at 0.1 T vs final | 0.93 vs 0.88 (0.38) | 0.93 vs 0.86 (0.40) | 0.89 vs 0.86 (0.38) | 0.91 vs 0.88 (0.41) |
| pair order flips after 0.1 T | 0.08 vs 0.14 (0.50) | 0.09 vs 0.15 (0.58) | 0.12 vs 0.16 (0.41) | 0.12 vs 0.15 (0.35) |

Reading (dev and test agree in direction on almost everything, so dev data is usable for fitting):
1. **Noise is too large and of the wrong kind.** Prior noise is ~3-4× the real residual, and white (increments
   anticorrelated, −0.35, as i.i.d. noise gives), while real fluctuations are smooth (increments positively
   correlated: PD1 +0.72, dev +0.57, LCBench +0.31; TaskSet test's 50 widely spaced points are the exception).
   The prior's up-steps (0.47 ≈ coin flip) are this noise. (Not its overfitting count: see the
   correction in the noise-calibration entry below.)
2. **Too much overfitting and divergence in the prior**: 0.5-0.6 of curves vs 0-0.3 (PD1: none, consistent with
   decaying LR schedules ending at their best); diverging 0.15-0.21 vs 0-0.03.
3. **Real tasks have dead configs and a cluster near the best; the prior has neither.** Flat curves: TaskSet about
   half (losses clipped at the initial loss: NaN or no learning), PD1/LCBench 6-8%; prior none. Final losses span
   the task's range far more in TaskSet and PD1 (0.96-0.98 vs 0.45), and PD1 has many near-best configs (23% within
   5% of the best vs 3%).
4. **Real curves are slightly more predictable early** (rank corr. at 0.1 T 0.89-0.93 vs 0.86-0.88; fewer order flips
   0.08-0.12 vs 0.14-0.16). Smaller gap than 1-3.

Which of these matter for decisions is open. A guess is 3: with many near-best configs (PD1), finding a good
config is easy and the budget is better spent training it, so a policy's balance of starting vs continuing matters
differently than on the prior. But the big-shop PD1 loss isn't clearly about starting more: big-shop models start
151-152 runs per episode on the PD1 full table vs 110-142 for focus03, and about the same at shop 32 (150-157 vs
143-150). To be checked against measurements 1-2 (twin ρ, scorer accuracy real vs twin), running.

## 2026-11-15: Prior sim-to-real, measurements 1-2: matched-twin ρ and the scorer on real vs twin states

**Matched twins** (`scripts/twins.py`, `runs/twins/`): for each of the 87 benchmark tasks, 16 prior tasks with its d
and T, budget 1000, shop 32, capped runs; every model with a shop-32 benchmark eval (41) runs on the same seeded
twins with that eval's checkpoint and temperature; regret normalised by a pool of 1000 random configs per twin.
Spearman ρ between twin regret and benchmark shop-32 regret, bootstrap over models:

| models | TaskSet | PD1 | LCBench |
|---|---|---|---|
| all | +0.29 [−0.07, +0.60] (41) | **+0.64 [+0.35, +0.79]** (35) | −0.10 [−0.47, +0.34] (34) |
| pairwise | **+0.56 [+0.20, +0.80]** (33) | +0.31 [−0.07, +0.60] (27) | +0.38 [−0.04, +0.75] (26) |
| RL | +0.19 [−0.74, +1.00] (8) | +0.21 [−0.86, +0.90] (8) | +0.07 [−0.69, +0.71] (8) |

- The PD1 inversion of the crude measurement (pairwise −0.32, RL −0.82, one synthetic spec with T 50) is gone with
  matched specs: it was spec mismatch (curve length, budget), not curve shape.
- RL: 8 models give no information (intervals span almost [−1, 1]). The crude "RL ρ −0.82" from 7 models carries
  the same caveat; it shouldn't be quoted as evidence that RL exploits the prior. (The other evidence for that
  stands: RL's best synthetic results with the worst benchmark results, SAC's ~100 runs per episode.)
- Pairwise: positive everywhere, significant on TaskSet only. Twins are a usable but weak model-selection signal.

**Scorer on real vs twin states** (`scripts/pair_gap.py`, `wide-focus03-v1`, own policy at τ 1, shop 32, 8
episodes per task and 8 twins per task, every 25 steps all choosable pairs with exact labels; `runs/logs/pair_gap*.log`).
Pair accuracy, real vs twin:

| | run-run | run-shop | shop-shop | top-1 hit | pairs with the best item | top item's gap to best / median's gap |
|---|---|---|---|---|---|---|
| TaskSet | 0.740 / 0.742 | **0.679 / 0.731** | **0.614 / 0.695** | 0.25 / 0.24 | 0.89 / 0.82 | 0.22 / 0.23 |
| PD1 | 0.764 / 0.726 | 0.738 / 0.723 | 0.697 / 0.694 | 0.10 / 0.18 | 0.77 / 0.78 | 0.23 / 0.29 |
| LCBench | 0.793 / 0.756 | 0.757 / 0.731 | 0.727 / 0.681 | 0.22 / 0.22 | 0.74 / 0.82 | 0.23 / 0.22 |

- The scorer's per-state judgment transfers: on PD1 and LCBench it is as accurate on real states as on twins or
  more. The exception is TaskSet comparisons involving unseen configs (−0.05 / −0.08), which fits the curve
  statistics: TaskSet hyperparameters barely predict the final loss (config smoothness 0.01 real vs 0.15 prior).
- PD1's low top-1 hit rate (0.10 vs 0.18) is near-ties (23% of PD1 configs are within 5% of the best): the
  tie-tolerant measure (label gap of the top-scored item to the best, relative to the median's gap) is better on
  real PD1 than on twins (0.23 vs 0.29).
- Caveat: no intervals (pairs within a task are strongly dependent); differences under ~0.03 should be read as
  equal.

Taken together: measured by what the scorer and the policy ranking see, the gap is smaller than the curve
statistics suggest. The prior's curves differ a lot from real ones (noise, dead configs, overfitting), but the
pairwise scorer judges real states about as well as prior states, apart from hyperparameter-only comparisons on
TaskSet.

## 2026-11-15: Noise fix of the prior: calibration on dev data, 3 training seeds launched

Noise is the clearest isolated discrepancy (3-4× too large and white on all 4 families), so per the plan agreed
with the user (measure; retrain only for one clear isolated discrepancy) one change is tested. Prior noise is now
configurable (`env.prior_noise_scale`, `env.prior_noise_smooth`: Gaussian-smoothed noise rescaled to unit variance;
defaults reproduce ifBO's prior). Calibrated on the TaskSet dev set only (`runs/logs/noise_calib.log`, 2 twins per
task):

| scale, smoothing | noise (real 0.017) | incr. autocorr (real +0.57) | up-steps (real 0.21) | KS sum |
|---|---|---|---|---|
| 1, 0 (ifBO) | 0.069 | −0.30 | 0.45 | 2.01 |
| 0.25, 0 | 0.031 | +0.11 | 0.39 | 1.12 |
| **0.25, 1** | **0.023** | **+0.65** | **0.35** | **0.99** |
| 0.25, 1.5 | 0.018 | +0.82 | 0.32 | 0.97 |
| 0.25, 2 | 0.016 | +0.88 | 0.31 | 1.04 |

Chose 0.25 / 1 (closest medians; KS sum ties with 0.25 / 1.5).

Correction to the curve-statistics entry: the prior's overfitting and divergence are *not* noise artefacts. They
barely move when the noise is scaled down (overfit 0.54 → 0.41, diverge 0.18 → 0.15 at scale 0.25); they come from
the basis-curve shapes.

Launched `wide-focus03-noise-v1/-s1/-s2` (configs/wide_focus03_noise.yaml, otherwise identical to `wide-focus03`;
GPUs 3-5; tags wf03noise, wf03noises1, wf03noises2; runs/logs/noise_runs.sh). Comparison: vs the 3 focus03 seeds
with compare_seeds.py. Note the in-training synthetic eval uses the changed prior, so its numbers aren't comparable
with other runs.

## 2026-11-16: Noise-fixed prior: worse, not better

`wide-focus03-noise-v1/-s1/-s2` (prior noise × 0.25, smoothed over 1 point, calibrated on TaskSet dev data; else
identical to `wide-focus03`) vs the 3 focus03 seeds (gain = focus03 − noise; seed-aware CIs):

| | TaskSet | PD1 | LCBench |
|---|---|---|---|
| full table, τ 0.25 | 0.068 vs 0.056: −0.012 [−0.041, +0.009] | 0.016 vs 0.0095: −0.007 [−0.018, +0.000] | 0.022 vs 0.012: −0.009 [−0.020, +0.000] |
| shop 32, τ 1 | 0.096 vs 0.085: −0.011 [−0.029, +0.006] | 0.052 vs 0.032: **−0.020 [−0.042, −0.003]** | 0.023 vs 0.023: −0.000 [−0.005, +0.006] |

All six point estimates are ≤ 0; one interval excludes 0. Making one aspect of the prior more realistic made the
policy worse. Possible reasons, untested:
- the large white noise acts as a regulariser (labels from noisy curves teach the scorer not to trust small
  differences, which may suit real tasks whose other mismatches are larger);
- the fix was fitted on dev curves whose residual noise (0.017) is half that of the TaskSet test tasks (0.034;
  LCBench 0.029), i.e. the calibration target may simply be too low for the benchmarks;
- other gaps (dead configs, near-best cluster, too much overfitting) interact with noise.
Training pair accuracy is higher (0.72-0.76 vs 0.71 at the end), as expected with less label noise.

Conclusion: the noise fix doesn't help; the option stays in the code (defaults = ifBO's prior). It also weakens the
case that matching curve statistics is the way to improve the prior: the measured statistics are not a reliable
target on their own. Further prior changes wait for the user.

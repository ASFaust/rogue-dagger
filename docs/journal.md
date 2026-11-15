# Research journal

This is the narrative record: what we tried, why we tried it, what we learned, and which decisions followed.
The numbers live in README ("Results so far"), the checklist in BACKLOG.md, the RL details in docs/rl.md, and the
original spec in PLAN.md. Add a dated entry whenever an experiment finishes or a decision is made, including dead ends.

Conventions: regret = mean final normalised regret after 1000 steps (lower is better); "paired" = per-task
difference with a 95 % bootstrap CI over tasks; full table = the whole unseen table is the shop (ifBO's setting).

## Key findings so far

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

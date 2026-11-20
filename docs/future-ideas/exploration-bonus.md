# Exploration level: bonus on unseen configs, and getting it right from training

Status: weak evidence (one diagnostic sweep, 2026-11-20), not queued. Results in the journal entry
"2026-11-20 night", step 2, item 5.

## Motivation

On LCBench we lose to FT-PFN mainly through **selection error from front-loaded exploration**: we stop starting new
configs early (focus03 seed 0: 134 runs, 11 % of starts in the 2nd half; FT-PFN: 206 runs, 39 %). The best unseen
config is still in the shop at step 100; we just don't start it. Ranking of unseen configs is kNN-level for every
method, FT-PFN included, and on the prior nobody beats kNN (twin diagnostic, 2026-11-20), so how much we explore
is the bigger lever, not ranking.

## Evidence (weak)

`evaluate.py --shop-bias b` adds b to every unseen config's score (BT models, policy_step only):

| b | 0 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|
| prior twins (capped, 4 per task) | 0.0254 | 0.0230 | 0.0220 | 0.0270 | 0.0280 |
| LCBench (focus03 seed 0) | 0.0140 | 0.0124 | 0.0105 | 0.0108 | **0.0085** (FT-PFN 0.0076) |

- LCBench regret tracks runs started across 6 focus03-type models (309 runs → 0.010, 105 → 0.028), and runs started
  vary 3× between training seeds: the exploration level is poorly pinned down by training.
- The policy under-explores on the prior itself (optimum b ≈ 0.1–0.2), and real LCBench wants much more (still
  improving at 0.8).

Why this is weak: b swept on LCBench itself (test-set tuning, optimistic); one seed; LCBench only; slot-biased model
at the lucky column order. And it can go the other way: the random-slots models *over*-explore on PD1's long-curve
tasks (imagenet_resnet, translate_xformer: ~190 runs, best config trained to 3 % fidelity) and lose PD1 by it.
So a single global bonus is unlikely to be right; the level should depend on budget vs curve length.

## Options

1. **Fixed bonus, chosen fairly**: pick b on non-test data (allowed TaskSet dev tasks or prior twins), then run it
   unchanged on all three families with seeds and a slot-fair model. Cheapest; tells whether a global b transfers.
2. **Bonus as a function of state** (e.g. of remaining budget / T, or of the share of budget already spent on new
   starts), chosen the same way.
3. **Fix it in training**: the prior rewards less exploration than real data. Either a prior where exploration pays
   more (curve-faker's job) or RL fine-tuning on real curves (BACKLOG step 3), where the return directly prices it.

## Ablation

Option 1 first: one slot-fair model, b ∈ {0, 0.1, 0.2, 0.4, 0.8} on the TaskSet dev tasks; pick b; eval at that b on
TaskSet / PD1 / LCBench, 3 seeds, seed-aware CIs vs b = 0 and vs FT-PFN. Watch PD1 (training error).

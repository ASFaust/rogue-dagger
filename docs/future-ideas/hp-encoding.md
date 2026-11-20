# Hyperparameter encoding without slot bias

Status: design, 2026-11-20 (BACKLOG item 4). Nothing implemented. Background: journal 2026-11-20 entries "All models
trained before 2026-11-20 have a slot bias", "Slots models: results", "Twin diagnostic".

## The symmetry we want

In the ifBO prior the hp dimensions are exchangeable: values iid U(0,1) per dim, first-layer BNN weights iid per dim.
Within one task, though, dimension j is the *same* hp in every config. So the scorer must be invariant to permuting
the active dims **jointly across all tokens of an env** (one permutation per env, applied to every run and shop
token), and must ignore inactive dims. It must *not* be invariant per config: pooling each config's dims on its own
would map (0.1, 0.9) and (0.9, 0.1) to the same thing.

Today's token puts the hps in 10 fixed slots (`features.py` [59:79], values + presence mask) and one MLP embeds the
whole token. Random slots (`env.random_slots`) gets invariance by augmentation: it costs capacity (slots models lose
PD1 by over-exploring), needs TTA over placements to be invariant at deploy (8× inference), and keeps d_max = 10.

What a new encoding can and can't buy (twin diagnostic): on the prior nothing ranks unseen configs better than 5-NN,
FT-PFN included. So the target is exact invariance at no capacity cost and a good inductive bias for similarity
between configs, not ranking beyond kNN.

## Idea 1: axial attention over (token, dim) cells (TabPFN-v2 style), recommended

Treat the hps of an env as a table: cell (i, j) = token i, dim j, embedded by a shared φ(x_ij) (Fourier features
of the value + a run/shop flag). Inactive dims are masked, not zero-filled. A few small layers alternate
- **attention across tokens within a dim** (per column): where along this dim the configs sit, and, once each cell
  also carries its token's performance summary (best/last loss of runs, from the existing curve features), how
  performance varies along this dim, i.e. in-context per-dim relevance;
- **attention across dims within a token** (per row): interactions between hps of one config.

No positional encoding over dims, so every layer is equivariant to joint dim permutations; then pool over dims
(mean + attention pool) to one hp embedding per token, added to the existing token embedding. The trunk, head and
pairwise loss stay as they are.

+ Exact invariance by construction: no slot bookkeeping, no TTA, any d (no d_max cap).
+ Per-dim relevance is computed from the data in context instead of being tied to a slot.
+ The column attention gives each config direct access to the others along each dim (the comparisons kNN and GP
  kernels make), where today the trunk has to extract them from raw slot values.
− Biggest change. Cost: per-dim attention over all tokens; a full table (~1000 shop + runs) × d ≤ 10 × 2–3 layers of
  width 64 is fine with SDPA/flash, but needs a check at the full-table eval.
− Performance info has to get into the cells (inject the token's curve summary), or the column attention only sees
  where configs are, not how good they are.

## Idea 2: kernel-biased attention (in-context ARD)

Keep the token trunk; drop the hp slots from the token (or keep random slots) and give hps to the trunk as an
attention bias between tokens i and k, per head h:

    bias_h(i, k) = − Σ_j  w_hj · g_h(|x_ij − x_kj|)

with g_h a small shared learned 1-D function and w_hj ≥ 0 per-dim relevance weights computed per env by a module
that is equivariant over dims (DeepSets over per-dim statistics of the runs, e.g. the rank correlation between x_j and
run performance). Summing over dims makes it invariant by construction.

+ Builds in the GP/kNN structure directly: attention mass flows to similar configs, with learned in-context lengthscales.
+ Smaller change than idea 1: one bias term per layer, the token layout loses its hp block.
− Bias is S × S × d per env; at full table S² ≈ 10⁶. Feasible with chunking or restricting it to shop → run attention
  (S × R), but not free, and flash attention can't take an arbitrary bias.
− The config itself is then only known through relations to others; a single config's absolute position (e.g. "at
  the edge of the range") needs a separate per-dim invariant feature.
Evidence: an ARD-GP fitted in context is kNN-level on ρ but reaches FT-PFN's top-10 on real LCBench by step 600
(0.028 vs 0.025), so learned relevance helps where it matters for picking.

## Idea 3: invariant hand-crafted relative features (cheapest)

Compute per (token, dim) features relative to the env's runs, then pool over dims with relevance weights, and add a
few joint features that are invariant because they sum over dims:
- per dim: the value's rank among the runs, distance to the incumbent along this dim, a 1-D neighbour estimate
  (mean performance of the nearest runs along this dim only), the dim's in-context relevance (rank correlation of x_j
  with run performance);
- joint: kNN estimate in the full space (plain and relevance-weighted distances), distance to the incumbent.

These replace the slot block; everything else unchanged.

+ Small, fast, exactly invariant; no new attention; easy to ablate feature by feature.
+ Hands the model the kNN signal it currently matches at best, so it can spend capacity on curves and exploration.
− Hand-designed: an additive approximation over dims, interactions only through the joint features. Caps what the
  model can learn about hps at what we thought of.
− The relevance statistics are noisy early in an episode (few runs).

Not picked: random per-episode dim codes (each dim gets a fresh random vector, the config embedding sums
MLP(code_j, x_ij)): no d_max cap, but still invariance in distribution only, needing TTA like random slots.

## Recommendation and test plan

Idea 1, since it is the only one that is invariant by construction and still learns hp interactions and relevance.
Idea 3 is the fallback if 1 is too slow at full table, and a cheap baseline in any case. Idea 2 can later be added
on top of 1 as a bias if column attention alone is not enough.

One change at a time, against the random-slots models (no TTA and 8-placement TTA) and focus03 at random order:
1. Unit test: scores unchanged (up to fp tolerance) under a random joint dim permutation, and for any d.
2. Train one seed with the focus03 config (100k), synthetic eval at equal iterations vs the slots models.
3. Benchmarks, full table τ 0.25 and shop 32 τ 1; `twin_rank.py` on LCBench (ρ, top-10 vs kNN). Expect: PD1
   recovers to the focus03-random level or better (no over-exploration), TaskSet/LCBench ≥ slots + TTA without TTA.
4. If it holds: 2 more seeds, seed-aware CIs vs slots and FT-PFN.

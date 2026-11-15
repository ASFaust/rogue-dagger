# Crossing-aware horizons for the pairwise loss

Status: tested, negative, closed (2026-11-15, see the journal). Both variants, 2 seeds each vs `wide-readout`
(2 seeds): variant 1 (tie target) n.s. on the full table, worse at shop 32 (TaskSet −0.021, PD1 −0.012); variant 2
(ordinary labels on both sides) n.s. except PD1 full table −0.007 [−0.017, −0.000]. Origin: user, 2026-11-15.

## Idea

A pair's label carries information about curve *shape* only where it changes with the horizon, i.e. where the two
curves swap order. Away from a crossing, every horizon repeats the same label. So the training signal should
concentrate at crossings.

## Why the architecture allows it

With the MLP readout (`model.readout: mlp`, `wide-readout-v1`) the trunk doesn't depend on the horizon; the horizon
enters only at the readout. One trunk pass can therefore be read out at many horizons cheaply.
`train.n_horizons` already does this, with K horizons per state drawn log-uniform in [1, remaining]. With the default
token readout, the horizon goes through the trunk via the global token, so every extra horizon costs a full trunk pass.

A comparison σ(s_a − s_b) needs both scores at the same horizon, so the horizon is per *pair*, not per token.

## Where a pair crosses

Labels use the best-so-far loss up to len+H. A pair (a, b) swaps at the first horizon h* where the order of
cummin_a(len_a + h) and cummin_b(len_b + h) changes. This can be computed exactly from the curves the env already
materialises.

## Two variants

1. **Equality at the crossing (simpler, try first; user's refinement).** No need to sample horizons around a
   crossing. At h* the two items are tied, so the extra loss is "scores equal at h*". Options:
   - pairwise BCE with soft label 0.5, whose gradient is σ(s_a − s_b) − 0.5;
   - (s_a − s_b)².

   BCE keeps it on the main loss's scale. Together with the normal labels at other horizons, it pins down where the
   score difference changes sign. Cost: one extra readout per crossing pair.
2. **Crossing-mined horizons.** Choose the K readout horizons per state at or near crossings instead of log-uniform,
   and assign each pair to the horizon nearest its crossing.

## Details and risks

- h* is discrete (the order flips between h and h+1), but the horizon features are continuous (H/B, log1p(H/T)).
  Read the tie out at a fractional h* between the two steps.
- Pairs tied over a whole stretch (both best-so-far curves flat and equal) aren't crossings; skip them.
- Noise wiggles near the start can create spurious crossings; cummin damps these.
- Over-weighting hard pairs: mix with the existing log-uniform horizons.
- It builds on the MLP readout, which was best on TaskSet but worse on LCBench.
- It needs the score to be a smooth function of the horizon, which the MLP readout gives cheaply.

## Ablation

Against `wide-readout-v1` (same readout, same K = 4), change exactly one thing: add the equality term (1), or choose
the horizons at crossings (2).

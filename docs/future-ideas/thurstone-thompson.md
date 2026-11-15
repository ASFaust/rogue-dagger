# Thurstone readout with Thompson-sampling policy

Status: gate failed on frozen embeddings (2026-11-15, see the journal): Thurstone heads don't beat BT on held-out
pairs and learn a nearly flat σ. Not queued. Origin: discussion with the user, 2026-11-15. Motivation: the Bradley–Terry limit in
[implicit-distribution.md](implicit-distribution.md). Gate: test 3 there.

## Idea

Each token outputs a mean and a spread, (μ_a, σ_a), instead of one score. Comparisons become

    P(a beats b) = Φ( (μ_b − μ_a) / sqrt(σ_a² + σ_b²) )        (lower = better)

Training is still pairwise BCE on the same labels, with an O(n) readout.

This fixes the BT example. X and S have small σ, so a small margin is decisive. R has a large σ, so the same margin
is noise. All three pairs fit.

## Policy: Thompson sampling instead of softmax

Draw one sample per item from N(μ, σ) and take the best. The selection probability is exactly P(this item is best)
under the model, which is what softmax(scores / τ) approximates under PLAN.md's "weak assumptions". The learned σ's
largely take over the role of the temperature τ, which so far has to be tuned per shop size.

## Risks

1. **Gauge freedom.** Per state, a shared shift of all μ, or a common scale of all (μ, σ), leaves every probability
   unchanged. It needs pinning, e.g. σ_a² = 1 + softplus(·)², or normalising the mean σ per state.
2. **Weak identifiability** (the user's concern: extra degrees of freedom in a transformer where every item attends
   to the others). Binary pairwise labels constrain σ_a only through comparisons against opponents of different
   spread. Many (μ, σ) settings fit the training pairs about equally well, so σ is a lightly constrained knob the
   network can spend on prior quirks.
3. **That knob would drive exploration.** Under Thompson sampling, σ decides how often uncertain items are tried.
   A weakly identified quantity steering the policy is the same failure pattern as RL exploiting the prior.
4. **Independence again.** Per-item (μ, σ) can't represent pair-specific dependence (near-duplicate configs). The
   transformer absorbs shared effects but not pair-specific ones. The red-bus problem also stays: k near-duplicates
   get about k times the selection mass.

## Extensions (only if needed)

- **Skew-symmetric term:** logit P(a>b) = (Thurstone or BT part) + u_aᵀ J u_b with J = −Jᵀ (blade–chest models,
  Chen & Joachims 2016; Schur/Hodge decomposition, Balduzzi et al. 2018). It covers intransitive and pair-specific
  structure. Note: sigmoid(aᵀb) doesn't work, because a symmetric form makes every P equal to 0.5.
- **Fully general pairwise head:** g(e_a, e_b) − g(e_b, e_a). There is no per-item P(best) for the policy then,
  which needs an aggregation step (Copeland/Borda-style).

## Plan

1. Gate: test 3 in [implicit-distribution.md](implicit-distribution.md). The Thurstone head must beat BT on
   held-out pairs on frozen embeddings.
2. Sanity check σ before trusting it for the policy: it should be larger for unstarted items and short runs than
   for long runs, and should shrink as a run progresses.
3. Ablation: an MLP-readout model with a Thurstone head + Thompson sampling vs `wide-readout-v1`. Deploy without τ,
   and also with a σ-scale knob as a fallback.

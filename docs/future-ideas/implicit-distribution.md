# The curve distribution is implicit: what it buys, where the limit is, how to test it

Status: framing plus two diagnostic tests on a frozen model, not queued. Origin: user, 2026-11-15.
Follow-up idea: [thurstone-thompson.md](thurstone-thompson.md).

## Framing

FT-PFN models the learning-curve distribution explicitly: a histogram head gives p(y_t | x, data) per config and time
point. We never decode a curve. We train binary tests (does a beat b by len+H?) and leave the stochastic
representation to the model. With the MLP readout, a token embedding is a learned representation of that curve's
predictive distribution, uncertainty included, in whatever form suits the network.

What this buys:
- **Joint, not marginal.** A pair label is a property of the joint distribution of both curves. FT-PFN's marginals
  have to treat two configs as independent when they are compared.
- **The decision quantity, not a proxy.** The label is the best loss over the window up to len+H; MFPI reads the
  prediction at one horizon. The model only has to represent what the decision needs.

## Dependence between curves

Two training runs are independent processes. What's correlated is our knowledge of their futures, given the partial
curves seen so far:
1. **Shared task-level unknowns:** how low the loss can go, whether curves keep improving late, when overfitting
   starts. The ifBO prior is built this way: task latents are shared, and curves are independent given them.
2. **Similar hyperparameters:** nearby configs have similar curves, because the map from config to curve is smooth.
   Near-duplicates rise and fall together, so their difference is much less uncertain than either curve alone.
3. **Shared evaluation noise** on real benchmarks (same validation set). Absent in the prior.

Because every token attends to the whole state, the transformer can absorb 1 and 3: a shift shared by all items
cancels in a score difference. It can't fully absorb 2 with per-item outputs. One item needs a small effective spread
against a nearby opponent and a large one against a distant opponent. In practice this costs little, because
choosing between near-duplicates matters little.

A related policy effect: a softmax over per-item scores (and Thompson sampling with independent draws) gives a
cluster of near-duplicates about k times the selection mass of one item (the "red bus / blue bus" problem). This
could matter for big shops on low-dimensional tasks, where shops are dense. It can be checked by splitting the
big-shop PD1 loss by task dimension.

## Where the limit is: the readout is Bradley–Terry

P(a beats b) = σ(s_a − s_b) forces the log-odds to add up along chains:
logit P(a>c) = logit P(a>b) + logit P(b>c).
Each pair alone can always be fitted. The problem is fitting all pairs with one scalar per item. Example (lower loss
is better, independent items): S always 0.50; R is 0.20 or 0.80 with probability ½ each; X has P(X beats S) = 0.9.

| Pair | True P | BT forces |
|---|---|---|
| R vs S | 0.5 | s_R − s_S = 0 |
| X vs S | 0.9 | s_X − s_S ≈ 2.2 |
| X vs R | 0.5 | s_X − s_R = 2.2, so BT predicts **0.9** |

Here a small margin is decisive against a safe item and irrelevant against a risky one, and a scalar can't encode
"how much a margin matters". This matters because the decision is about P(best), not expected loss: when only the
best outcome counts, variance has value of its own. Attention doesn't fix this, because all three items are in the
same state. Two things soften it:
- s_a(H) is a function of the horizon, so spread can show up in its shape;
- the readout sees context (incumbent, budget).

## Tests (frozen model, heads only, no rollouts)

1. **Probe the embeddings.** On prior tasks the true future is known. Fit linear probes from frozen token embeddings
   to (i) the future best loss up to len+H and (ii) its spread across prior draws for the same config and history.
   This shows whether the representation holds uncertainty or only a point estimate.
2. **Same question to both models.** On held-out prior pairs, compare our σ(s_a − s_b) with P(a beats b) computed from
   FT-PFN's histograms under independence. Check calibration (reliability, log loss) against the true outcomes.
3. **How much BT costs.** On the same frozen embeddings, fit and compare held-out pairwise BCE for (a) the BT head,
   (b) a Thurstone (μ, σ) head, and (c) optionally Thurstone plus a skew-symmetric term (see
   [thurstone-thompson.md](thurstone-thompson.md)). If (b) doesn't beat (a) on *held-out* pairs, the extra freedom
   isn't needed or only overfits.

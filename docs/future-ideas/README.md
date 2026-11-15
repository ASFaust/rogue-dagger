# Future ideas

One file per idea: motivation, design, risks, and the ablation that would test it. Ideas aren't queued until they're
moved to BACKLOG.md. Results go into [../journal.md](../journal.md). One change per ablation.

| Idea | Status | Gate / next step |
|---|---|---|
| [crossing-horizons.md](crossing-horizons.md): train the pairwise loss at curve crossings (equality at the tie) | idea | ablation vs `wide-readout-v1` |
| [implicit-distribution.md](implicit-distribution.md): the curve distribution is implicit; BT limit; diagnostic tests | tests not run | probes + calibration on a frozen model |
| [thurstone-thompson.md](thurstone-thompson.md): (μ, σ) readout, Thompson sampling instead of softmax/τ | idea | gated on BT-vs-Thurstone test on held-out pairs |
| [real-curve-training.md](real-curve-training.md): train on full TaskSet / real curves, or improve the prior | idea | RL shelved; needs a real-task validation split |

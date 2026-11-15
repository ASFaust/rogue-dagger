# Future ideas

One file per idea: motivation, design, risks, and the ablation that would test it. Ideas aren't queued until they're
moved to BACKLOG.md. Results go into [../journal.md](../journal.md). One change per ablation.

| Idea | Status | Gate / next step |
|---|---|---|
| [crossing-horizons.md](crossing-horizons.md): train the pairwise loss at curve crossings (equality at the tie) | variant 1 done: no gain (shop 32 worse); variant 2 running | 2 seeds vs `wide-readout` |
| [implicit-distribution.md](implicit-distribution.md): the curve distribution is implicit; BT limit; diagnostic tests | tests 1, 3 done: spread weakly present, BT wins | calibration (test 2) |
| [thurstone-thompson.md](thurstone-thompson.md): (μ, σ) readout, Thompson sampling instead of softmax/τ | tested 2026-11-15: negative | closed: from scratch, 2 seeds, no σ scale beats focus03 |
| [real-curve-training.md](real-curve-training.md): train on full TaskSet / real curves, or improve the prior | idea | RL shelved; needs a real-task validation split |
| [prior-sim2real.md](prior-sim2real.md): measure the prior's sim-to-real gap (decision, model, data level), then improve the prior | first measurement done (policy-ranking ρ) | matched synthetic twins, then scorer real-vs-twin pair accuracy |

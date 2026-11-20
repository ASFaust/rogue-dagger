# Future ideas

One file per idea: motivation, design, risks, and the ablation that would test it. Ideas aren't queued until they're
moved to BACKLOG.md. Results go into [../journal.md](../journal.md). One change per ablation.

| Idea | Status | Gate / next step |
|---|---|---|
| [crossing-horizons.md](crossing-horizons.md): train the pairwise loss at curve crossings (equality at the tie) | tested 2026-11-15: negative | closed: neither variant helps (2 seeds each) |
| [implicit-distribution.md](implicit-distribution.md): the curve distribution is implicit; BT limit; diagnostic tests | tests 1, 3 done: spread weakly present, BT wins | calibration (test 2) |
| [thurstone-thompson.md](thurstone-thompson.md): (μ, σ) readout, Thompson sampling instead of softmax/τ | tested 2026-11-15: negative | closed: from scratch, 2 seeds, no σ scale beats focus03 |
| [real-curve-training.md](real-curve-training.md): train on full TaskSet / real curves, or improve the prior | queued (BACKLOG step 3, 2026-11-20) | RL on the allowed TaskSet tasks, after TTA and the LCBench analysis |
| [prior-sim2real.md](prior-sim2real.md): measure the prior's sim-to-real gap (decision, model, data level), then improve the prior | measured 2026-11-15; noise fix negative 2026-11-16 | moved to the sister repo `../curve-faker` (2026-11-20) |
| [exploration-bonus.md](exploration-bonus.md): bonus on unseen configs; exploration level from training | weak evidence 2026-11-20 (LCBench 0.0140 → 0.0085 at b 0.8, tuned on test) | pick b on non-test data, then all families (BACKLOG step 5) |
| [hp-encoding.md](hp-encoding.md): hp encoding without slot bias; 3 designs (axial cells, kernel-biased attention, relative features) | decided 2026-11-20: build ideas 1 and 3 | BACKLOG item 4, handoff in the file |

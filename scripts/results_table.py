"""Build docs/results.md: every model we've trained, with its benchmark results.

    python scripts/results_table.py            # writes docs/results.md

Edit REGISTRY when a run is added. A model is a list of eval tags, one per training seed. Missing evals show as
"running" / "–", so queued runs can be registered early. Mean final normalised regret (lower is better), ifBO
protocol (1000 steps, 10 eval seeds). Gains vs FT-PFN use the two-level bootstrap of scripts/compare_seeds.py
(resample tasks, and the model's training seeds).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FAMS = ("taskset", "pd1", "lcbench")
FAM_NAMES = {"taskset": "TaskSet", "pd1": "PD1", "lcbench": "LCBench"}

# (section, name, eval tags (one per training seed), what it changes, full-table tau, shop-32 tau)
# tau for Thurstone models is the Thompson sigma scale.
REGISTRY = [
    ("Baselines", "FT-PFN + MFPI-random (ifBO)", ["ifbo"], "the method we compare against", None, None),
    ("Baselines", "Random search", ["random_search"], "", None, None),

    ("Pairwise, references", "`wide`", ["wide100k", "wides1", "wides2"],
     "reference: shops 4..64, T 4..300, 0.5..200 full trainings per budget, train τ 1, BT token readout", 0.25, 1.0),
    ("Pairwise, references", "`wide-focus03`", ["wfocus03", "wf03s1", "wf03s2"],
     "`wide` + pairs of two non-contenders weighted 0.3", 0.25, 1.0),

    ("Pairwise, variants of `wide`", "`wide-focus-v1`", ["wfocus100k"], "pair weight 0.1", 0.25, 1.0),
    ("Pairwise, variants of `wide`", "`wide-long-v1`", ["wlong300k"], "300k iterations", 0.25, 1.0),
    ("Pairwise, variants of `wide`", "`wide-bigshop-v1`", ["wbig100k"], "shops 4..256", 0.25, 1.0),
    ("Pairwise, variants of `wide`", "`wide-t025-v1`", ["wt025"], "training rollouts at τ 0.25", 0.25, 1.0),
    ("Pairwise, variants of `wide`", "`wide-hanneal-v1`", ["whan100k"], "horizon annealed during the episode", 0.25,
     1.0),
    ("Pairwise, variants of `wide`", "`wide-future-v1`", ["wfut100k"], "label: future window only (no past)", 0.25,
     1.0),
    ("Pairwise, variants of `wide`", "`wide-readout-v1`", ["wread100k", "wreads1"],
     "MLP readout (horizon outside the trunk), 4 label horizons per step", 0.25, 1.0),
    ("Pairwise, variants of `wide`", "`wide-v1` @55k", ["wide55k"], "early checkpoint", 0.25, 1.0),

    ("Pairwise, variants of `wide-readout`", "`wide-readout-cross`", ["wrcross", "wrcrosss1"],
     "+ tie target at label crossings (crossing idea, variant 1)", 0.25, 1.0),
    ("Pairwise, variants of `wide-readout`", "`wide-readout-cross-sides`", ["wrsides", "wrsidess1"],
     "+ ordinary labels at both horizons around each crossing (variant 2)", 0.25, 1.0),

    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-big`", ["wf03big", "wf03bigs1"], "shops 4..256", 0.25,
     1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-mid-v1`", ["wf03mid"], "shops 4..128", 0.25, 1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-longT-v1`", ["wf03longT"], "T 4..1500", 0.25, 1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-tmix-v1`", ["wf03tmix"],
     "rollout τ per env, log-uniform 0.25..2", 0.25, 1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-tmixcold-v1`", ["wf03tmixcold"],
     "rollout τ per env, log-uniform 0.1..1", 0.25, 1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-thurstone`", ["wf03thu", "wf03thus1"],
     "Thurstone (μ, σ) readout from scratch, Thompson-sampling policy (τ = σ scale)", 1.0, 1.0),
    ("Pairwise, variants of `wide-focus03`", "`wide-focus03-thurstone`, σ×0.5", ["wf03thu", "wf03thus1"],
     "same models, Thompson draws at half σ", 0.5, None),

    ("Deploy temperature (swept on the benchmarks: optimistic)", "`wide-v1`, τ 0.15", ["wide100k"], "", 0.15, None),
    ("Deploy temperature (swept on the benchmarks: optimistic)", "`wide-v1`, τ 0.1", ["wide100k"], "", 0.1, None),

    ("Early configs", "`base-v1` @65k", ["base65k"], "first training distribution (narrower)", 0.25, 1.0),
    ("Early configs", "`hanneal-v1` @65k", ["hanneal65k"], "`base-v1` + horizon anneal", 0.25, 1.0),

    ("RL (shelved 2026-11-14)", "`rl-wide-v1`", ["rlwide"], "PPO from `wide-v1`, aux BCE", 0.25, 0.25),
    ("RL (shelved 2026-11-14)", "`rl-wide-noaux-v1`", ["rlnoaux"], "PPO from `wide-v1`, no aux loss", 0.25, 0.25),
    ("RL (shelved 2026-11-14)", "`rl-wide-noaux-v1` @750 updates", ["rlnoaux750"], "earlier snapshot", 0.25, 0.25),
    ("RL (shelved 2026-11-14)", "`rl-scratch-v1`", ["rlscratch"], "PPO from scratch, aux BCE", 1.0, 1.0),
    ("RL (shelved 2026-11-14)", "`rl-big-noaux-v1`", ["rlbignoaux"], "PPO, shops 4..256, no aux", 0.25, 0.25),
    ("RL (shelved 2026-11-14)", "`rl-big-focus03-v1`", ["rlbigf03"], "PPO from focus03, shops 4..256", 0.25, 0.25),
    ("RL (shelved 2026-11-14)", "`sac-frozen-wide-v1`", ["sacwide"], "discrete SAC, frozen trunk", 1.0, 1.0),
    ("RL (shelved 2026-11-14)", "`sac-frozen-big-v1`", ["sacbig"], "discrete SAC, frozen trunk, shops 4..256", 1.0,
     1.0),
]

# eval tags of runs that are training / queued: their missing cells show "running" (all others show "–")
PENDING = {"wreads1", "wrcross", "wrcrosss1", "wrsides", "wrsidess1", "wf03thu", "wf03thus1", "wf03bigs1"}


def path(fam, tag):
    return ROOT / "runs" / "eval" / fam / f"{tag}.npz"


def eval_tag(tag, n, tau):
    return tag if tag in ("ifbo", "random_search") else f"{tag}-n{n}-t{tau}"


def load(fam, tag):
    p = path(fam, tag)
    if not p.exists():
        return None
    z = np.load(p)
    return dict(zip(map(str, z["tasks"]), z["regret"][:, :, -1].mean(1)))


def models(fam, tags, n, tau):
    """Per-seed task->regret dicts that exist."""
    if tau is None:
        return []
    return [m for m in (load(fam, eval_tag(t, n, tau)) for t in tags) if m is not None]


def gain_ci(ref, ms, boot=4000):
    """Gain of models ms (list of task dicts) over a single reference dict, bootstrap over tasks and seeds."""
    tasks = sorted(set(ref).intersection(*map(set, ms)))
    r = np.array([ref[t] for t in tasks])
    x = np.array([[m[t] for t in tasks] for m in ms])
    point = (r - x.mean(0)).mean()
    rng = np.random.default_rng(0)
    T, S = len(tasks), len(ms)
    b = np.empty(boot)
    for i in range(boot):
        ti = rng.integers(0, T, T)
        b[i] = r[ti].mean() - x[rng.integers(0, S, S)][:, ti].mean()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, lo, hi, int(((r - x.mean(0)) > 0).sum()), T


def fmt(v, fam):
    return "–" if v is None else (f"{v:.3f}" if fam == "taskset" else f"{v:.4f}")


def main():
    ifbo = {f: load(f, "ifbo") for f in FAMS}
    out = ["# Results: all models", "",
           "Generated by `scripts/results_table.py` (edit its REGISTRY, rerun). Mean final normalised regret on the",
           "ifBO benchmark tables, lower is better (1000 steps per task, 10 eval seeds, regret normalised by the",
           "table's best and worst value). Models with several training seeds show the mean over seeds.", "",
           "**Read with seed noise in mind** (journal, 2026-11-15): retraining one config with another seed moves",
           "results by up to about 0.017 (TaskSet), 0.013 (PD1) and 0.005 (LCBench). Single-seed differences of that",
           "size are not evidence; use `scripts/compare_seeds.py` for claims.", "",
           "Deploy settings: **full** = the whole table as the shop (ifBO's setting) at the full-table τ;",
           "**s32** = random shops of 32 configs at τ 1 (RL rows: their own τ). Thurstone rows: τ is the Thompson",
           "σ scale. `running` = run or evaluation in progress; `–` = not evaluated.", ""]

    out += ["## Mean final regret", "",
            "| Model | seeds | change | τ full | TaskSet full | PD1 full | LCBench full | TaskSet s32 | PD1 s32 | "
            "LCBench s32 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    gains, seeds_rows, sec = [], [], None
    for section, name, tags, change, tf, t32 in REGISTRY:
        if section != sec:
            out.append(f"| **{section}** | | | | | | | | | |")
            sec = section
        cells, any_ = [], False
        n_seen = 0
        for n, tau in ((0, tf), (32, t32)):
            for f in FAMS:
                if tags[0] in ("ifbo", "random_search"):
                    ms = [m for m in [load(f, tags[0])] if m is not None] if n == 0 else []
                else:
                    ms = models(f, tags, n, tau)
                n_seen = max(n_seen, len(ms))
                if ms:
                    any_ = True
                    cells.append(fmt(float(np.mean([np.mean(list(m.values())) for m in ms])), f))
                else:
                    cells.append("running" if tau is not None and PENDING & set(tags) else "–")
        seeds = f"{n_seen}/{len(tags)}" if len(tags) > 1 else "1"
        tfs = "" if tags[0] in ("ifbo", "random_search") else str(tf)
        out.append(f"| {name} | {seeds} | {change} | {tfs} | " + " | ".join(cells) + " |")
        if not any_ or tags[0] in ("ifbo", "random_search"):
            continue
        # gain vs FT-PFN, full table
        g = []
        for f in FAMS:
            ms = models(f, tags, 0, tf)
            if ms and ifbo[f] is not None:
                p, lo, hi, w, T = gain_ci(ifbo[f], ms)
                star = " ▲" if lo > 0 else (" ▼" if hi < 0 else "")
                g.append(f"{p:+.4f} [{lo:+.4f}, {hi:+.4f}]{star} ({w}/{T})")
            else:
                g.append("–")
        gains.append(f"| {name} | {seeds} | " + " | ".join(g) + " |")
        if len(tags) > 1:
            per = []
            for f in FAMS:
                vals = []
                for t in tags:
                    m = load(f, eval_tag(t, 0, tf))
                    vals.append("…" if m is None else fmt(float(np.mean(list(m.values()))), f))
                per.append(" / ".join(vals))
            seeds_rows.append(f"| {name} | " + " | ".join(per) + " |")

    out += ["", "## Gain vs FT-PFN (full table)", "",
            "Paired gain over FT-PFN (positive = better), 95% CI resampling tasks and the model's training seeds,",
            "and tasks won. ▲ / ▼: CI excludes 0. For single-seed models the CI excludes training-seed variance,",
            "so it is too narrow.", "",
            "| Model | seeds | TaskSet | PD1 | LCBench |", "|---|---|---|---|---|"] + gains
    out += ["", "## Per seed (full table)", "", "Seed 0 / 1 / 2.", "",
            "| Model | TaskSet | PD1 | LCBench |", "|---|---|---|---|"] + seeds_rows
    (ROOT / "docs" / "results.md").write_text("\n".join(out) + "\n")
    print("wrote docs/results.md")


if __name__ == "__main__":
    main()

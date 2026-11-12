"""PD1 raw jsonl -> one tabular npz per task, following mf-prior-bench's pd1-tabular processing.

* matched phase 1 data, except imagenet-resnet-1024, which only exists as unmatched phase 0
* 4 hyperparameters: lr initial value, lr decay steps factor, lr power, momentum
* metric: valid/error_rate
* configs whose curve length differs from the modal length are dropped (mostly diverged runs)
* curves longer than 100 points are additionally written subsampled with jumps 2, 5, 10
  (mfpbench's "-<jump>" variants; jump 1 = the full curve)

Output: data/pd1/<dataset>_<model>_<bs>[-<jump>].npz with hps, hp_names, valid [n_cfg, n_steps].
"""
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HPS = ["hps.lr_hparams.initial_value", "hps.lr_hparams.decay_steps_factor",
       "hps.lr_hparams.power", "hps.opt_hparams.momentum"]
SKIP_MODELS = {"max_pooling_cnn"}
SHORT = {"fashion_mnist": "fashion", "svhn_no_extra": "svhn", "translate_wmt": "translate",
         "wide_resnet": "wideresnet", "simple_cnn": "simplecnn", "xformer_translate": "xformertranslate"}


def rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    groups = defaultdict(list)
    for fname, keep in (("pd1_matched_phase1_results.jsonl.gz", lambda k: k[2] != 1024 or k[:2] != ("imagenet", "resnet")),
                        ("pd1_unmatched_phase0_results.jsonl.gz", lambda k: k == ("imagenet", "resnet", 1024))):
        for r in rows(raw_dir / fname):
            key = (r["dataset"], r["model"], int(r["hps.batch_size"]))
            if r["model"] in SKIP_MODELS or not keep(key):
                continue
            ve = r.get("valid/error_rate")
            if not isinstance(ve, list) or any(r.get(h) is None for h in HPS):
                continue
            ve = [1.0 if (v is None or not np.isfinite(v)) else min(float(v), 1.0) for v in ve]
            groups[key].append(([float(r[h]) for h in HPS], ve))
    for (ds, model, bs), items in sorted(groups.items()):
        mode = Counter(len(v) for _, v in items).most_common(1)[0][0]
        items = [it for it in items if len(it[1]) == mode]
        # deduplicate identical hp settings (keep first)
        seen, uniq = set(), []
        for h, v in items:
            if tuple(h) not in seen:
                seen.add(tuple(h))
                uniq.append((h, v))
        hps = np.array([h for h, _ in uniq])
        valid = np.array([v for _, v in uniq])
        name = f"{SHORT.get(ds, ds)}_{SHORT.get(model, model)}_{bs}"
        jumps = [1, 2, 5, 10] if mode > 100 else [None]
        for j in jumps:
            if j is None:
                cols = np.arange(mode)
                fn = name
            else:
                cols = np.linspace(1, mode, (mode - 1) // j, dtype=int) - 1 if j > 1 else np.arange(mode)
                if len(cols) < 10:
                    continue
                fn = f"{name}-{j}"
            np.savez(out_dir / f"{fn}.npz", hps=hps, hp_names=np.array(HPS), valid=valid[:, cols])
            print(f"{fn}: {len(hps)} configs x {len(cols)} steps", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/pd1/pd1", sys.argv[2] if len(sys.argv) > 2 else "data/pd1")

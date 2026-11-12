"""Download the 12 TaskSet NLP tasks x {adam4p, adam8p} used by DyHPO / ifBO.

Curves come from the public bucket gs://gresearch/task_set_data (fetched over
https), hyperparameters from the google-research repo. Output, per task and
optimizer: data/taskset/<task>__<opt>.npz with
    hps    [n_configs, n_hp]  raw hyperparameter values (column order = hp_names)
    hp_names
    steps  [n_steps]          training step at each curve point
    valid  [n_configs, n_steps]  valid1 loss, averaged over the 5 replicas
    test   [n_configs, n_steps]  valid2 loss (TaskSet's held-out split)
"""
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TASKS = [
    "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
    "FixedTextRNNClassification_imdb_patch32_GRU128_bs128",
    "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
]
OPTS = {"adam4p": "adam4p_wide_grid_1k", "adam8p": "adam8p_wide_grid_1k"}
CURVE_URL = "https://storage.googleapis.com/gresearch/task_set_data/{task}/{opt}_10000_replica5.npz"
HP_URL = ("https://raw.githubusercontent.com/google-research/google-research/master/"
          "task_set/optimizers/configs/{opt}.json")


def fetch(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def hp_table(opt):
    configs = json.loads(fetch(HP_URL.format(opt=opt.replace("_1k", ""))))
    # keys look like "<opt>_seed<i>", value[0] is the hp dict
    rows = {int(k.split("_seed")[1]): v[0] for k, v in configs.items()}
    names = sorted(rows[0].keys())
    hps = np.array([[float(rows[i][n]) for n in names] for i in sorted(rows)])
    return names, hps


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {short: hp_table(opt) for short, opt in OPTS.items()}

    def one(job):
        task, short = job
        path = out_dir / f"{task}__{short}.npz"
        if path.exists():
            return f"cached {path.name}"
        cc = np.load(io.BytesIO(fetch(CURVE_URL.format(task=task, opt=OPTS[short]))))
        ys = cc["ys"]  # [n_configs, n_replicas, n_steps, 4 splits]
        names, hps = tables[short]
        n = min(len(hps), ys.shape[0])
        np.savez(path, hps=hps[:n], hp_names=np.array(names), steps=cc["xs"],
                 valid=ys[:n, :, :, 1].mean(1), test=ys[:n, :, :, 2].mean(1))
        return f"saved {path.name} {ys.shape}"

    jobs = [(t, o) for t in TASKS for o in OPTS]
    with ThreadPoolExecutor(8) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/taskset")

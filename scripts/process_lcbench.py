"""LCBench data_2k.zip (35 tasks x 2000 configs x 52 epochs) -> data/lcbench/<task>.npz.

Keeps the 7 searched hyperparameters (ifBO paper, Table 3) and the validation
accuracy curve as an error rate, with epoch 0 dropped as recommended by the
LCBench authors. Fields: hps [n_cfg, 7], hp_names, valid [n_cfg, 51] (1 - val_acc/100).
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

HPS = ["batch_size", "learning_rate", "max_dropout", "max_units", "momentum", "num_layers", "weight_decay"]


def main(zip_path, out_dir):
    out_dir = Path(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        data = json.loads(zf.read("data_2k.json"))
    for ds, cfgs in sorted(data.items()):
        keys = sorted(cfgs, key=int)
        hps = np.array([[float(cfgs[k]["config"][h]) for h in HPS] for k in keys])
        acc = np.array([cfgs[k]["log"]["Train/val_accuracy"] for k in keys], dtype=np.float64)
        valid = (1 - acc / 100.0)[:, 1:]
        np.savez(out_dir / f"{ds}.npz", hps=hps, hp_names=np.array(HPS), valid=valid.astype(np.float32))
        print(f"{ds}: {hps.shape[0]} configs x {valid.shape[1]} epochs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/lcbench/data_2k.zip",
         sys.argv[2] if len(sys.argv) > 2 else "data/lcbench")

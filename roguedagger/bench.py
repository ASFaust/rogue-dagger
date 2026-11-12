"""Tabular benchmarks (TaskSet, PD1, LCBench if present) and a batched freeze-thaw env over them.

A Task is a finite table: configs X [n_cfg, d] normalised to [0, 1] and loss
curves Y [n_cfg, T] (lower is better, scaled into [0, 1] so FT-PFN can use
them). TabEnv runs E independent HPO episodes (seeds) on one task at once, with
the same attributes ShopEnv exposes, so features.from_env / train.policy_step
work unchanged:
  * runs are capped at the table length T (cap = True)
  * the shop is n configs drawn without replacement from the configs not yet
    started (n = None -> every unseen config, ifBO's tabular setting)
"""
from __future__ import annotations

import glob
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"

IFBO_PD1 = [
    "cifar100_wideresnet_2048", "cifar100_wideresnet_256", "cifar10_wideresnet_2048", "cifar10_wideresnet_256",
    "fashion_simplecnn_2048", "fashion_simplecnn_256",
    "imagenet_resnet_1024-1", "imagenet_resnet_1024-10", "imagenet_resnet_1024-2", "imagenet_resnet_1024-5",
    "imagenet_resnet_256-1", "imagenet_resnet_256-10", "imagenet_resnet_256-2", "imagenet_resnet_256-5",
    "imagenet_resnet_512-1", "imagenet_resnet_512-10", "imagenet_resnet_512-2", "imagenet_resnet_512-5",
    "lm1b_transformer_2048", "mnist_simplecnn_2048", "mnist_simplecnn_256", "svhn_wideresnet_1024",
    "svhn_wideresnet_256", "translate_xformertranslate_64-1", "translate_xformertranslate_64-10",
    "translate_xformertranslate_64-2", "translate_xformertranslate_64-5", "uniref50_transformer_128",
]


@dataclass
class Task:
    name: str
    family: str
    X: np.ndarray   # [n_cfg, d] in [0, 1]
    Y: np.ndarray   # [n_cfg, T] loss in [0, 1], lower is better


def normalise_hps(h: np.ndarray) -> np.ndarray:
    """Per column: log scale if strictly positive and spanning > 2 decades, then min-max to [0, 1]."""
    out = np.empty_like(h, dtype=np.float64)
    for j in range(h.shape[1]):
        c = h[:, j].astype(np.float64)
        if (c > 0).all() and c.max() / c.min() > 100:
            c = np.log(c)
        lo, hi = c.min(), c.max()
        out[:, j] = (c - lo) / (hi - lo) if hi > lo else 0.5
    return out


def _drop_constant(X):
    keep = X.std(0) > 0
    return X[:, keep] if keep.any() else X[:, :1]


def load_taskset(root=DATA / "taskset"):
    tasks = []
    for f in sorted(glob.glob(str(Path(root) / "*.npz"))):
        z = np.load(f)
        v = z["valid"][:, 1:]             # drop step 0, as in ifBO ("drop_epoch_0")
        ub = np.nanmedian(z["valid"][:, 0])  # median loss at step 0 = practitioner's upper bound
        v = np.where(np.isfinite(v), v, ub).clip(0, ub) / ub
        name = Path(f).stem.replace("FixedTextRNNClassification_", "")
        tasks.append(Task(name, "taskset", _drop_constant(normalise_hps(z["hps"])), v.astype(np.float32)))
    return tasks


def load_pd1(root=DATA / "pd1", names=IFBO_PD1):
    tasks = []
    for nm in names:
        f = Path(root) / f"{nm}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        tasks.append(Task(nm, "pd1", _drop_constant(normalise_hps(z["hps"])), z["valid"].clip(0, 1).astype(np.float32)))
    return tasks


def load_lcbench(root=DATA / "lcbench"):
    """LCBench data_2k: 35 tasks x 2000 configs x 52 epochs; we use val accuracy -> error, epoch 0 dropped."""
    zp = Path(root) / "data_2k.zip"
    if not zp.exists() or zp.stat().st_size < 1e6:
        return []
    with zipfile.ZipFile(zp) as zf:
        name = [n for n in zf.namelist() if n.endswith(".json")][0]
        data = json.loads(zf.read(name))
    tasks = []
    for ds, cfgs in sorted(data.items()):
        keys = sorted(cfgs, key=int)
        hp_names = [k for k in cfgs[keys[0]]["config"] if isinstance(cfgs[keys[0]]["config"][k], (int, float))]
        X = np.array([[float(cfgs[k]["config"][h]) for h in hp_names] for k in keys])
        Y = 1 - np.array([cfgs[k]["log"]["Train/val_accuracy"] for k in keys])[:, 1:] / 100.0
        tasks.append(Task(ds, "lcbench", _drop_constant(normalise_hps(X)), Y.clip(0, 1).astype(np.float32)))
    return tasks


def load_family(family):
    return {"taskset": load_taskset, "pd1": load_pd1, "lcbench": load_lcbench}[family]()


class TabEnv:
    """E parallel HPO episodes on one tabular task; ShopEnv-compatible attributes."""

    def __init__(self, task: Task, n_envs: int, budget: int = 1000, n_shop: int | None = 32,
                 device="cpu", seed: int = 0, d_max: int = 10, shop_mode: str = "uniform",
                 p_mutate: float = 0.5, mutate_std: float = 0.1, mutate_topk: int = 5):
        self.task, self.E, self.dev, self.d_max = task, n_envs, torch.device(device), d_max
        self.shop_mode, self.p_mutate, self.mutate_std, self.mutate_topk = shop_mode, p_mutate, mutate_std, mutate_topk
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        n_cfg, d = task.X.shape
        Tn = task.Y.shape[1]
        self.Xall = torch.zeros(n_cfg, d_max, device=self.dev)
        self.Xall[:, :d] = torch.as_tensor(task.X[:, :d_max], dtype=torch.float32, device=self.dev)
        self.Yall = torch.as_tensor(task.Y, device=self.dev)
        self.n_cfg = n_cfg
        self.L = Tn
        self.N = n_cfg if n_shop is None else min(n_shop, n_cfg)
        self.full_table = n_shop is None
        self.R = min(budget, n_cfg)
        E, R, dev = n_envs, self.R, self.dev
        self.d = torch.full((E,), min(d, d_max), dtype=torch.long, device=dev)
        self.T = torch.full((E,), Tn, dtype=torch.long, device=dev)
        self.B = torch.full((E,), budget, dtype=torch.long, device=dev)
        self.cap = torch.ones(E, dtype=torch.bool, device=dev)
        self.run_cfg = torch.zeros(E, R, dtype=torch.long, device=dev)
        self.run_curve = torch.zeros(E, R, Tn, device=dev)
        self.run_x = torch.zeros(E, R, d_max, device=dev)
        self.run_len = torch.zeros(E, R, dtype=torch.long, device=dev)
        self.run_last = torch.zeros(E, R, dtype=torch.long, device=dev)
        self.n_runs = torch.zeros(E, dtype=torch.long, device=dev)
        self.unseen = torch.ones(E, n_cfg, dtype=torch.bool, device=dev)
        self.consumed = torch.zeros(E, dtype=torch.long, device=dev)
        self.best = torch.full((E,), float("inf"), device=dev)
        self.trace = torch.zeros(E, budget, device=dev)   # incumbent after each step
        # seed: one random config, one step (Algorithm 1 of the ifBO paper)
        first = torch.randint(0, n_cfg, (E,), device=dev, generator=self.gen)
        self._start(torch.arange(E, device=dev), first)
        self._advance(torch.arange(E, device=dev), torch.zeros(E, dtype=torch.long, device=dev))
        self._draw_shop()

    # --- ShopEnv-compatible API ---------------------------------------------------
    def shop_valid(self):
        return self._shop_valid

    def choosable(self):
        runs = (torch.arange(self.R, device=self.dev)[None] < self.n_runs[:, None]) & (self.run_len < self.T[:, None])
        shop = self._shop_valid & (self.n_runs < self.R)[:, None]
        return torch.cat([runs, shop], 1)

    def horizon(self, f_start=1.0, f_end=1.0):
        rem = (self.B - self.consumed).clamp(min=1)
        prog = self.consumed.float() / self.B.float()
        f = f_start + (f_end - f_start) * prog
        return (f * rem.float()).round().long().clamp(min=1)

    @property
    def done(self):
        return self.consumed >= self.B

    def step(self, action):
        act = (~self.done).nonzero()[:, 0]
        a = action[act]
        is_shop = a >= self.R
        if is_shop.any():
            es = act[is_shop]
            cfg = self.shop_ids[es, a[is_shop] - self.R]
            self._start(es, cfg)
        run = torch.where(is_shop, self.n_runs[act] - 1, a)
        self._advance(act, run)
        self._draw_shop()
        return {"done": self.done}

    # --- internals -------------------------------------------------------------
    def _start(self, e, cfg):
        slot = self.n_runs[e]
        self.run_cfg[e, slot] = cfg
        self.run_curve[e, slot] = self.Yall[cfg]
        self.run_x[e, slot] = self.Xall[cfg]
        self.run_len[e, slot] = 0
        self.unseen[e, cfg] = False
        self.n_runs[e] += 1

    def _advance(self, e, run):
        pos = self.run_len[e, run]
        y = self.run_curve[e, run, pos.clamp(max=self.L - 1)]
        self.run_len[e, run] += 1
        self.consumed[e] += 1
        self.run_last[e, run] = self.consumed[e]
        self.best[e] = torch.minimum(self.best[e], y)
        self.trace[e, self.consumed[e] - 1] = self.best[e]

    def _draw_shop(self):
        """n unseen configs per env, without replacement (all unseen if full_table)."""
        if self.full_table:
            ids = torch.arange(self.n_cfg, device=self.dev)[None].expand(self.E, -1)
            self.shop_ids = ids
            self._shop_valid = self.unseen.clone()
        else:
            noise = torch.rand(self.E, self.n_cfg, device=self.dev, generator=self.gen)
            noise = noise.masked_fill(~self.unseen, 2.0)
            ids = noise.topk(self.N, dim=1, largest=False).indices
            dup = torch.zeros_like(ids, dtype=torch.bool)
            if self.shop_mode == "mixed":
                ids, dup = self._mutate_into(ids)
            self.shop_ids = ids
            self._shop_valid = self.unseen.gather(1, self.shop_ids) & ~dup
        self.shop_x = self.Xall[self.shop_ids]

    def _mutate_into(self, ids):
        """Replace a fraction of shop slots with the unseen config nearest to a Gaussian
        perturbation of one of the top-k runs (by best loss so far). Duplicates are dropped
        as invalid slots. Returns (ids, duplicate mask)."""
        E, N, D = self.E, self.N, self.d_max
        k = min(self.mutate_topk, self.R)
        pos = torch.arange(self.L, device=self.dev)
        rev = pos[None, None] < self.run_len[..., None]
        best = torch.where(rev, self.run_curve, torch.full_like(self.run_curve, float("inf"))).min(2).values
        top = best.topk(k, dim=1, largest=False).indices
        n_top = self.n_runs.clamp(max=k, min=1)
        pick = (torch.rand(E, N, device=self.dev, generator=self.gen) * n_top[:, None]).floor().long()
        parent = self.run_x.gather(1, top.gather(1, pick)[..., None].expand(-1, -1, D))
        dmask = (torch.arange(D, device=self.dev)[None] < self.d[:, None]).float()[:, None]
        target = (parent + torch.randn(E, N, D, device=self.dev, generator=self.gen) * self.mutate_std) * dmask
        dist = torch.cdist(target, self.Xall[None].expand(E, -1, -1))           # [E, N, n_cfg]
        dist = dist.masked_fill(~self.unseen[:, None], float("inf"))
        near = dist.argmin(-1)
        use = torch.rand(E, N, device=self.dev, generator=self.gen) < self.p_mutate
        out = torch.where(use, near, ids)
        # duplicate picks within a shop: keep one, mark the rest invalid
        srt, order = out.sort(1)
        dup_sorted = torch.zeros_like(srt, dtype=torch.bool)
        dup_sorted[:, 1:] = srt[:, 1:] == srt[:, :-1]
        dup = torch.zeros_like(dup_sorted).scatter(1, order, dup_sorted)
        return out, dup

    # --- metrics ----------------------------------------------------------------
    def normalised_regret(self):
        """[E, B] incumbent regret, normalised by the table's best and worst value."""
        lo, hi = self.Yall.min(), self.Yall.max()
        return ((self.trace - lo) / (hi - lo).clamp(min=1e-12)).clamp(0, 1)

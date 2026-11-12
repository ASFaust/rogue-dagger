"""Core invariants: prior sanity, label semantics (past included, cap), tabular shop, loss."""
import numpy as np
import pytest
import torch

from roguedagger.env import EnvConfig, ShopEnv
from roguedagger.features import encode, feature_dim, from_env
from roguedagger.model import Scorer, pairwise_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_prior_curves_finite_and_bounded():
    from roguedagger.prior import PriorTasks
    pt = PriorTasks(8, 10, DEV)
    pt.resample(torch.arange(8, device=DEV), torch.randint(1, 11, (8,), device=DEV))
    c = pt.curves(torch.rand(8, 5, 10, device=DEV), torch.full((8,), 20, device=DEV), 60)
    assert torch.isfinite(c).all() and c.min() >= 0 and c.max() <= 1


def _naive_value(curve, length, H, lim):
    return curve[: max(1, min(length + H, lim))].min()


@pytest.mark.parametrize("p_cap", [0.0, 1.0])
def test_label_values_include_past_and_respect_cap(p_cap):
    env = ShopEnv(6, EnvConfig(B_range=(20, 80), T_range=(5, 10), p_cap=p_cap), DEV, seed=3)
    for _ in range(15):
        a = torch.multinomial(env.choosable().float(), 1)[:, 0]
        env.step(a)
    H = env.horizon()
    v = env.values(H)
    for e in range(env.E):
        lim = int(env.T[e]) if bool(env.cap[e]) else env.L
        for r in range(int(env.n_runs[e])):
            exp = _naive_value(env.run_curve[e, r], int(env.run_len[e, r]), int(H[e]), lim)
            assert torch.isclose(v[e, r], exp)
            # past included: value never exceeds the best revealed point
            seen = env.run_curve[e, r, : int(env.run_len[e, r])].min()
            assert v[e, r] <= seen + 1e-6
        for j in range(int(env.n[e])):
            exp = _naive_value(env.shop_curve[e, j], 0, int(H[e]), lim)
            assert torch.isclose(v[e, env.R + j], exp)


def test_capped_runs_not_choosable():
    env = ShopEnv(4, EnvConfig(B_range=(60, 60), T_range=(5, 5), budget_full_range=(12, 12), p_cap=1.0), DEV, seed=0)
    for _ in range(4):  # seed step + 4 = T = 5 points on run 0
        env.step(torch.zeros(4, dtype=torch.long, device=DEV))
    assert (env.run_len[:, 0] == 5).all()
    assert not env.choosable()[:, 0].any()


def test_encode_shapes_and_pairwise_loss():
    env = ShopEnv(4, EnvConfig(), DEV, seed=1)
    H = env.horizon()
    tok, pad = encode(from_env(env, H), 10)
    assert tok.shape[-1] == feature_dim(10) and torch.isfinite(tok).all()
    m = Scorer(d_model=32, n_layers=1, n_heads=2, d_ff=64).to(DEV)
    s = m(tok, pad)
    vals = torch.tensor([[0.1, 0.2, 0.3]], device=DEV)
    good = torch.tensor([[3.0, 2.0, 1.0]], device=DEV)
    bad = -good
    ok = torch.ones(1, 3, dtype=torch.bool, device=DEV)
    assert pairwise_loss(good, vals, ok)[0] < pairwise_loss(bad, vals, ok)[0]
    assert pairwise_loss(good, vals, ok)[1] == 1.0
    assert s.shape[1] == tok.shape[1] - 1


def test_tabular_shop_only_unseen():
    from roguedagger.bench import Task, TabEnv
    rng = np.random.default_rng(0)
    task = Task("toy", "toy", rng.random((30, 3)), rng.random((30, 4)).astype(np.float32))
    env = TabEnv(task, 3, budget=40, n_shop=5, seed=0)
    while not env.done.all():
        valid = env.choosable()
        env.step(torch.multinomial(valid.float(), 1)[:, 0])
        for e in range(3):
            started = set(env.run_cfg[e, : int(env.n_runs[e])].tolist())
            shop = env.shop_ids[e][env.shop_valid()[e]].tolist()
            assert not started & set(shop)
    assert (env.run_len <= 4).all()


def test_future_only_values():
    env = ShopEnv(4, EnvConfig(B_range=(20, 80), T_range=(5, 10), p_cap=0.5), DEV, seed=5)
    for _ in range(12):
        env.step(torch.multinomial(env.choosable().float(), 1)[:, 0])
    H = env.horizon()
    v = env.values(H, include_past=False)
    for e in range(env.E):
        lim = int(env.T[e]) if bool(env.cap[e]) else env.L
        for r in range(int(env.n_runs[e])):
            lo = int(env.run_len[e, r])
            hi = min(lo + int(H[e]), lim)
            if hi > lo:
                assert torch.isclose(v[e, r], env.run_curve[e, r, lo:hi].min())

"""RL extension: reward sign + telescoping, GAE, run-block padding, critic init from a scorer."""
import torch

from roguedagger.env import EnvConfig, ShopEnv
from roguedagger.model import Critic, Scorer
from roguedagger.rl import gae, pad_runs, reward

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_reward_nonnegative_and_telescopes():
    env = ShopEnv(8, EnvConfig(B_range=(20, 60), T_range=(5, 10), ref_pool=32), DEV, seed=0)
    for mode in ("log", "linear"):
        tot = torch.zeros(env.E, device=DEV)
        start = env.norm_regret().clone()
        ended = torch.zeros(env.E, dtype=torch.bool, device=DEV)
        final = torch.zeros(env.E, device=DEV)
        while not ended.all():
            out = env.step(torch.multinomial(env.choosable().float(), 1)[:, 0])
            r = reward(out, mode, 1e-3)
            assert (r >= -1e-6).all()
            tot += r * (~ended)
            newly = out["done"] & ~ended
            final[newly] = out["regret"][newly]
            ended |= out["done"]
        exp = (torch.log(start + 1e-3) - torch.log(final + 1e-3)) if mode == "log" else start - final
        assert torch.allclose(tot, exp, atol=1e-4)
    assert (env.ref_hi > env.ref_lo).all()


def test_gae_matches_naive_returns_with_lambda_one():
    K, E = 6, 3
    rew, val = torch.rand(K, E), torch.rand(K, E)
    done = torch.zeros(K, E, dtype=torch.bool)
    done[2, 1] = True
    vb = torch.rand(E)
    _, ret = gae(rew, val, done, vb, 1.0, 1.0)
    for e in range(E):
        g = vb[e]
        for t in reversed(range(K)):
            g = rew[t, e] + (0.0 if done[t, e] else g)
            assert torch.isclose(ret[t, e], g)


def test_pad_runs_keeps_shop_block():
    x = torch.arange(1 + 2 + 3).float()[None]          # [global | 2 runs | 3 shop]
    y = pad_runs(x, 1, 2, 4, -1.0)
    assert y.tolist() == [[0, 1, 2, -1, -1, 3, 4, 5]]


def test_critic_loads_scorer_trunk():
    s = Scorer(d_max=10, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    c = Critic(d_max=10, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    n = c.load_trunk(s.state_dict())
    assert n > 0
    for k, v in c.state_dict().items():
        if not k.startswith("value_head"):
            assert torch.equal(v, s.state_dict()[k])


def test_embedding_replay_roundtrip():
    from roguedagger.sac import EmbeddingReplay
    E, r, N, d = 3, 4, 5, 8
    buf = EmbeddingReplay(E, d, capacity=E * 10, pool_tokens=E * 10 * (r + N), device=DEV)
    states = []
    for step in range(4):
        e = torch.randn(E, r + N, d, device=DEV).bfloat16().float()
        nr = torch.tensor([1, 4, 2], device=DEV)
        ns = torch.tensor([5, 3, 1], device=DEV)
        valid = torch.rand(E, r + N, device=DEV) > 0.3
        g = torch.randn(E, 11, device=DEV)
        slot = buf.push(e, g, valid, nr, ns, r)
        a = torch.tensor([0, 4 + 2, 1], device=DEV)         # run 0 / shop 2 / run 1 in [r | N] layout
        buf.set_outcome(slot, torch.where(a < r, a, a - r + nr), torch.rand(E, device=DEV), torch.zeros(E, dtype=torch.bool, device=DEV))
        states.append((e, valid, nr, ns, slot))
    e0, v0, nr, ns, slot0 = states[1]
    rows = torch.arange(E, device=DEV)
    eb, gb, vb, Rb = buf._gather(torch.full((E,), slot0, device=DEV), rows)
    for i in range(E):
        k, m = int(nr[i]), int(ns[i])
        assert torch.equal(eb[i, :k].float(), e0[i, :k])
        assert torch.equal(eb[i, Rb:Rb + m].float(), e0[i, r:r + m])
        assert torch.equal(vb[i, :k], v0[i, :k]) and torch.equal(vb[i, Rb:Rb + m], v0[i, r:r + m])
        assert not vb[i, k:Rb].any() and not vb[i, Rb + m:].any()
    batch = buf.sample(64)
    (e, g, m), a, rew, (e2, g2, m2), done = batch
    assert m.gather(1, a[:, None]).shape == (64, 1) and len(buf) == 3 * E


def test_embedding_replay_evicts_when_pool_wraps():
    from roguedagger.sac import EmbeddingReplay
    E, r, N, d = 2, 3, 2, 4
    buf = EmbeddingReplay(E, d, capacity=E * 100, pool_tokens=25, device=DEV)   # ~2.5 steps of 10 tokens
    for _ in range(20):
        nr = torch.tensor([3, 3], device=DEV); ns = torch.tensor([2, 2], device=DEV)
        slot = buf.push(torch.randn(E, r + N, d, device=DEV), torch.zeros(E, 11, device=DEV),
                        torch.ones(E, r + N, dtype=torch.bool, device=DEV), nr, ns, r)
        buf.set_outcome(slot, torch.zeros(E, dtype=torch.long, device=DEV), torch.zeros(E, device=DEV),
                        torch.zeros(E, dtype=torch.bool, device=DEV))
        spans = sorted((s0, s0 + n0) for _, s0, n0 in buf.alive)
        assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:])), "live steps overlap in the pool"
    assert 1 <= len(buf.alive) <= 2

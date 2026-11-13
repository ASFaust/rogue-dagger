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

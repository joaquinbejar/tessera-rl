# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference learner's conditional action distribution for the RTS
(tools/rts_policy.py, stage 5b Q4): legal under the full masks, a proper
joint distribution, and no illegal order in a played game. No training."""

import itertools
import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from rts_policy import Agent, Conditional, flat_actions  # noqa: E402

from tessera_rl._env import make  # noqa: E402
from tessera_rl.masks import head_views  # noqa: E402


def random_masks(gen, b, u, t, w, p, density=0.4):
    tm = gen.random((b, u, t)) < density
    gm = gen.random((b, u, t, w)) < density
    pm = gen.random((b, u, t, p)) < density
    return torch.as_tensor(tm), torch.as_tensor(gm), torch.as_tensor(pm)


def random_dist(gen, b, u, t, w, p):
    torch.manual_seed(int(gen.integers(1 << 30)))
    masks = random_masks(gen, b, u, t, w, p)
    return Conditional(torch.randn(b, u, t), torch.randn(b, u, t, w), torch.randn(b, u, t, p), *masks), masks


def legal(actions, masks):
    tm, gm, pm = masks
    b, u = actions.shape[:2]
    for i, j in itertools.product(range(b), range(u)):
        k, g, q = (int(x) for x in actions[i, j])
        for row, choice in ((tm[i, j], k), (gm[i, j, k], g), (pm[i, j, k], q)):
            if row.any():
                assert row[choice], (i, j, k, g, q)
            else:
                assert choice == 0, (i, j, k, g, q)


@pytest.mark.parametrize("greedy", [False, True])
def test_rts_policy_samples_are_legal_under_the_conditional_masks(greedy):
    gen = np.random.default_rng(1)
    for _ in range(20):
        dist, masks = random_dist(gen, 4, 5, 6, 9, 4)
        legal(dist.sample(greedy=greedy), masks)


def test_rts_policy_joint_probabilities_sum_to_one():
    # One slot, small heads: enumerate every joint action.
    gen = np.random.default_rng(2)
    for _ in range(10):
        dist, _ = random_dist(gen, 1, 1, 3, 4, 2)
        total = 0.0
        for k, g, q in itertools.product(range(3), range(4), range(2)):
            lp, _ = dist.log_prob_entropy(torch.tensor([[[k, g, q]]]))
            total += float(lp.exp())
        # A head with no legal choice puts all its mass on 0.
        assert total == pytest.approx(1.0, abs=1e-5)


def test_rts_policy_log_prob_matches_the_sampled_heads():
    gen = np.random.default_rng(3)
    dist, masks = random_dist(gen, 3, 4, 6, 9, 4)
    a = dist.sample()
    lp, ent = dist.log_prob_entropy(a)
    # Recompute slot by slot from the masked softmax of each head.
    tm, gm, pm = masks
    want = torch.zeros(3)
    for i, j in itertools.product(range(3), range(4)):
        k, g, q = (int(x) for x in a[i, j])
        for logits, row, choice in (
            (dist.type_logits[i, j], tm[i, j], k),
            (dist.target_logits[i, j, k], gm[i, j, k], g),
            (dist.product_logits[i, j, k], pm[i, j, k], q),
        ):
            if row.any():
                want[i] += torch.log_softmax(logits.masked_fill(~row, -1e9), -1)[choice]
    assert torch.allclose(lp, want, atol=1e-5)
    assert (ent >= 0).all()


def test_rts_policy_empty_rows_play_zero_with_no_mass():
    z = lambda *s: torch.zeros(*s, dtype=torch.bool)  # noqa: E731
    dist = Conditional(torch.randn(2, 3, 6), torch.randn(2, 3, 6, 9), torch.randn(2, 3, 6, 4), z(2, 3, 6), z(2, 3, 6, 9), z(2, 3, 6, 4))
    a = dist.sample()
    assert (a == 0).all()
    lp, ent = dist.log_prob_entropy(a)
    assert (lp == 0).all() and (ent == 0).all()


def test_rts_policy_plays_a_game_with_no_illegal_order():
    # Both agents sample from an untrained network under the full masks;
    # the environment's illegal counters stay at zero. No weight changes.
    torch.manual_seed(0)
    size, units = 8, 8
    with make("tessera/RTS-v1", num_envs=2, size=size, max_units=units, outputs="owned") as e:
        agent = Agent(11, size, 2, units, 6, 81, 4)
        step = e.reset(seed=5)
        for _ in range(40):
            planes = torch.as_tensor(np.ascontiguousarray(step.observations["planes"])).reshape(-1, 11, size, size)
            globals_ = torch.as_tensor(np.ascontiguousarray(step.observations["globals"])).reshape(-1, 2)
            t, g, p = (np.ascontiguousarray(v).reshape(-1, *v.shape[2:]) for v in head_views(step.masks, e.mask_heads, e.entity_group_heads))
            with torch.no_grad():
                dist, _ = agent.distribution(planes, globals_, t, g, p)
                actions = flat_actions(dist.sample())
            step = e.step(actions.numpy().astype(np.int32).reshape(2, 2, -1))
        assert [e.illegal_actions(i) for i in range(2)] == [[0, 0], [0, 0]]


def test_rts_policy_game_check_detects_an_unmasked_target():
    # Negative control for the game test: the same play with the target
    # head drawn ignoring its mask is counted illegal by the environment.
    torch.manual_seed(0)
    size, units = 8, 8
    with make("tessera/RTS-v1", num_envs=2, size=size, max_units=units, outputs="owned") as e:
        agent = Agent(11, size, 2, units, 6, 81, 4)
        step = e.reset(seed=5)
        for _ in range(40):
            planes = torch.as_tensor(np.ascontiguousarray(step.observations["planes"])).reshape(-1, 11, size, size)
            globals_ = torch.as_tensor(np.ascontiguousarray(step.observations["globals"])).reshape(-1, 2)
            t, g, p = (np.ascontiguousarray(v).reshape(-1, *v.shape[2:]) for v in head_views(step.masks, e.mask_heads, e.entity_group_heads))
            with torch.no_grad():
                dist, _ = agent.distribution(planes, globals_, t, np.ones_like(g), p)
                actions = flat_actions(dist.sample())
            step = e.step(actions.numpy().astype(np.int32).reshape(2, 2, -1))
        assert sum(sum(e.illegal_actions(i)) for i in range(2)) > 0

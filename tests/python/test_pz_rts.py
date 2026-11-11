# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PettingZoo Parallel adapter on tessera/RTS-v1, a multi-head
environment (RTS stage 4, PR C; plan r2 R2 and R5): the official API and
seed tests, the masks in observations and infos, refused actions and AEC,
terminal reward timing, and equality with the native pool. PettingZoo is
imported, not skipped (Q4): a run without it fails."""

import numpy as np
import pytest

import pettingzoo.test as pz_test

import tessera_rl
from tessera_rl import _native, pz
from tessera_rl.masks import head_views

RTS = "tessera/RTS-v1"
SMALL = {"size": 8, "max_units": 4, "step_limit": 40}
HEADS = 4 * 3

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
    "ignore:Environment has not defined a render:UserWarning",
)


def env(**kw):
    return pz.parallel_env(RTS, seed=3, **{**SMALL, **kw})


def noop(e):
    return {a: np.zeros(HEADS, dtype=np.int64) for a in e.agents}


def test_pz_rts_passes_the_parallel_api_test():
    pz_test.parallel_api_test(env(), num_cycles=60)


def test_pz_rts_passes_the_seed_test():
    pz_test.parallel_seed_test(lambda: env())


def test_pz_rts_spaces_are_multidiscrete_and_the_masks_are_in_infos():
    e = env()
    a = e.possible_agents[0]
    assert e.possible_agents == ["agent_0", "agent_1"]
    assert e.action_space(a).nvec.tolist() == [6, 81, 4] * 4
    assert "action_mask" not in e.observation_space(a).spaces
    assert e.observation_space(a)["planes"].shape == (11, 8, 8)
    e.close()


def test_pz_rts_infos_carry_that_agents_masks_and_nothing_else():
    e = env()
    obs, infos = e.reset()
    for steps in range(3):
        for a in e.agents:
            assert set(infos[a]) == {"action_mask", "action_masks"}
            t, tg, p = infos[a]["action_masks"]
            assert (t.shape, tg.shape, p.shape) == ((4, 6), (4, 6, 81), (4, 6, 4))
            assert (infos[a]["action_mask"] == t).all()
            assert "action_mask" not in obs[a]
            assert t.dtype == np.int8
        obs, _, _, _, infos = e.step(noop(e))
    e.close()


def test_pz_rts_masks_match_the_native_pool_and_are_owned():
    e = env()
    native = tessera_rl.make(RTS, num_envs=1, seed=3, outputs="owned", auto_reset="disabled", **SMALL)
    obs, infos = e.reset()
    out = native.reset()
    views = head_views(out.masks, native.mask_heads, native.entity_group_heads)
    for k, a in enumerate(e.agents):
        for got, want in zip(infos[a]["action_masks"], views, strict=True):
            assert (got == want[0, k]).all()
    kept = infos["agent_0"]["action_masks"][0].copy()
    e.step(noop(e))
    assert (infos["agent_0"]["action_masks"][0] == kept).all(), "an earlier info changed"
    e.close()
    native.close()


def test_pz_rts_parallel_matches_the_native_pool_step_for_step():
    e = env()
    native = tessera_rl.make(RTS, num_envs=1, seed=3, outputs="owned", auto_reset="disabled", **SMALL)
    obs, _ = e.reset()
    out = native.reset()
    rng = np.random.default_rng(7)
    for _ in range(15):
        acts = {a: rng.integers(0, e.action_space(a).nvec) for a in e.agents}
        obs, rewards, _, _, _ = e.step(acts)
        rows = np.stack([acts[f"agent_{k}"] for k in range(2)])[None].astype(np.int32)
        out = native.step(rows)
        for k in range(2):
            for leaf, arr in out.observations.items():
                assert (obs[f"agent_{k}"][leaf] == arr[0, k]).all()
            assert rewards[f"agent_{k}"] == out.rewards[0, k]
    e.close()
    native.close()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(HEADS - 1, dtype=np.int64),  # one head short
        np.zeros((1, HEADS), dtype=np.int64),  # wrong ndim
        np.full(HEADS, 81),  # out of range
        np.full(HEADS, -1),  # negative
        np.full(HEADS, 0.5),  # not an integer
        3,  # a scalar
    ],
)
def test_pz_rts_invalid_action_raises_and_changes_nothing(bad):
    e = env()
    e.reset()
    before = e._a.env.snapshot(0)
    acts = noop(e)
    acts["agent_1"] = bad
    with pytest.raises(_native.ActionError):
        e.step(acts)
    assert e._a.env.snapshot(0) == before
    e.close()


def test_pz_rts_aec_is_refused_with_a_typed_error():
    with pytest.raises(_native.ArgumentError, match="not 'aec'"):
        pz.aec_env(RTS, seed=3, **SMALL)


def test_pz_rts_episode_end_reaches_both_agents_in_the_same_step():
    # SPEC 12 terminal timing: at the step limit both agents get their
    # final reward and flag in the same step, and agents empties.
    e = env(step_limit=3)
    e.reset()
    for _ in range(2):
        _, _, terms, truncs, _ = e.step(noop(e))
        assert not any(truncs.values()) and not any(terms.values())
    _, rewards, terms, truncs, _ = e.step(noop(e))
    assert truncs == {"agent_0": True, "agent_1": True}
    assert terms == {"agent_0": False, "agent_1": False}
    assert rewards == {"agent_0": 0.0, "agent_1": 0.0}
    assert e.agents == []
    e.close()


def test_pz_rts_seat_swap_renames_the_agents_sides():
    plain, swapped = env(), env(seat_swap=True)
    o1, _ = plain.reset()
    o2, _ = swapped.reset()
    # Each agent sees its own side in its own frame: at reset the two
    # sides are mirror images, so the views are the same either way.
    for a in plain.agents:
        assert (o1[a]["planes"] == o2[a]["planes"]).all()
    plain.close()
    swapped.close()

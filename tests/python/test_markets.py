# SPDX-License-Identifier: MIT OR Apache-2.0
"""tessera/LimitOrderBook-v1 from Python (#381, PR 4): ``make``, the layout
with its two ``Set`` leaves (ADR-0082), the PettingZoo Parallel adapter's
official API and seed tests, the conditional ``MultiDiscrete`` masks in
infos, the set masks as ``MultiBinary``, AEC refused, and step-for-step
equality with the native pool under legal play. PettingZoo is imported,
not skipped: a run without it fails."""

import gymnasium
import numpy as np
import pytest

import pettingzoo.test as pz_test

import tessera_rl
from tessera_rl import _native, pz
from tessera_rl.masks import head_views

ID = "tessera/LimitOrderBook-v1"
SMALL = {"agents": 4, "steps": 60, "latency": 1}
NVEC = [6, 9, 4, 8]

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
    "ignore:Environment has not defined a render:UserWarning",
)


def env(**kw):
    return pz.parallel_env(ID, seed=3, **{**SMALL, **kw})


def native(**kw):
    return tessera_rl.make(ID, num_envs=1, seed=3, outputs="owned", auto_reset="disabled", **{**SMALL, **kw})


def legal(rng, masks):
    """A legal action drawn head by head through the conditional masks
    (``kind``; ``offset`` under it; ``size`` under both; ``cancel`` under
    ``kind``)."""
    kind, offset, size, cancel = (np.asarray(m) for m in masks)
    kind = kind.reshape(6)
    offset = offset.reshape(6, 9)
    size = size.reshape(6, 9, 4)
    cancel = cancel.reshape(6, 8)
    k = int(rng.choice(np.flatnonzero(kind)))
    o = int(rng.choice(np.flatnonzero(offset[k])))
    s = int(rng.choice(np.flatnonzero(size[k, o])))
    c = int(rng.choice(np.flatnonzero(cancel[k])))
    return np.asarray([k, o, s, c], dtype=np.int64)


def test_markets_make_layout_has_two_sets_and_their_masks():
    e = native()
    layout = e._native.layout()
    names = [(name, dtype, tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]
    assert names == [
        ("bids", "int32", (10, 2)),
        ("asks", "int32", (10, 2)),
        ("own", "int32", (6,)),
        ("own_orders", "int32", (8, 3)),
        ("own_orders_mask", "bool", (8,)),
        ("trades", "int32", (16, 3)),
        ("trades_mask", "bool", (16,)),
        ("signal", "int32", (1,)),
    ]
    assert [(p, m) for p, m, _ in layout["sets"]] == [("own_orders", 8), ("trades", 16)]
    assert tuple(e.action_nvec) == tuple(NVEC)
    e.close()


@pytest.mark.parametrize("kw", [{"agents": 1}, {"agents": 17}, {"latency": 4}, {"steps": 0}])
def test_markets_make_rejects_config_out_of_range(kw):
    with pytest.raises(tessera_rl.ConfigError):
        native(**kw)


def test_markets_bool_option_must_be_a_bool():
    with pytest.raises(TypeError, match="allow_short_cash must be a bool"):
        native(allow_short_cash=1)


def test_markets_pz_passes_the_parallel_api_test():
    pz_test.parallel_api_test(env(), num_cycles=60)


def test_markets_pz_passes_the_seed_test():
    pz_test.parallel_seed_test(lambda: env())


def test_markets_pz_spaces_multidiscrete_and_set_masks_multibinary():
    e = env()
    a = e.possible_agents[0]
    assert e.possible_agents == [f"agent_{k}" for k in range(4)]
    assert e.action_space(a).nvec.tolist() == NVEC
    space = e.observation_space(a)
    for leaf, n in (("own_orders_mask", 8), ("trades_mask", 16)):
        assert isinstance(space[leaf], gymnasium.spaces.MultiBinary)
        assert space[leaf].n == n
    obs, _ = e.reset()
    assert obs[a]["own_orders_mask"].dtype == np.int8
    assert "action_mask" not in space.spaces
    e.close()


def test_markets_pz_infos_carry_the_conditional_masks():
    e = env()
    _, infos = e.reset()
    for a in e.agents:
        assert set(infos[a]) == {"action_mask", "action_masks"}
        kind, offset, size, cancel = infos[a]["action_masks"]
        assert kind.reshape(-1).shape == (6,)
        assert offset.reshape(-1).shape == (6 * 9,)
        assert size.reshape(-1).shape == (6 * 9 * 4,)
        assert cancel.reshape(-1).shape == (6 * 8,)
        # Before any order: no-op and limits only.
        assert kind.reshape(6).tolist() == [1, 0, 0, 1, 1, 0]
        relaxed = infos[a]["action_mask"]
        assert isinstance(relaxed, tuple) and len(relaxed) == 4
    e.close()


def test_markets_pz_aec_is_refused_with_a_capability_error():
    with pytest.raises(_native.CapabilityError, match="not 'aec'"):
        pz.aec_env(ID, seed=3, **SMALL)


def test_markets_pz_legal_play_matches_the_native_pool_step_for_step():
    e = env()
    n = native()
    _, infos = e.reset()
    out = n.reset()
    rngs = {a: np.random.default_rng(k + 1) for k, a in enumerate(e.possible_agents)}
    traded = False
    for step in range(50):
        views = head_views(out.masks, n.mask_heads, n.entity_group_heads)
        for k, a in enumerate(e.agents):
            for got, want in zip(infos[a]["action_masks"], views, strict=True):
                assert (got == want[0, k]).all(), f"step {step} {a}"
        acts = {a: legal(rngs[a], infos[a]["action_masks"]) for a in e.agents}
        obs, rewards, terms, truncs, infos = e.step(acts)
        # The native pool takes every slot (MAX_AGENTS = 16); the unused
        # slots are padded with the no-op, as the adapter does.
        rows = np.zeros((1, 16, 4), dtype=np.int32)
        for k in range(4):
            rows[0, k] = acts[f"agent_{k}"]
        out = n.step(rows)
        assert e._a.env.state_hash(0) == n.state_hash(0), f"step {step}"
        for k in range(4):
            name = f"agent_{k}"
            for leaf, arr in out.observations.items():
                assert (np.asarray(obs[name][leaf]) == np.asarray(arr[0, k], dtype=obs[name][leaf].dtype)).all()
            assert rewards[name] == out.rewards[0, k]
        traded |= bool(out.observations["trades_mask"][0].any())
    assert traded, "no trade in 50 steps of legal play: the test cannot see the trades set"
    e.close()
    n.close()

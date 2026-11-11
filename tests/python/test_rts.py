"""tessera/RTS-v1 from Python (RTS stage 4, PR B): the layout, the mask head
views of SPEC 11.3, the Rust goldens replayed across the boundary, empty
slots, and the adapters that refuse the environment."""

import numpy as np
import pytest

from tessera_rl import _native
from tessera_rl._env import is_stale, make
from tessera_rl.masks import head_views

ID = "tessera/RTS-v1"
TYPES, WINDOW, PRODUCTS = 6, 81, 4
BLOCK = TYPES + TYPES * WINDOW + TYPES * PRODUCTS  # 516


def rts(n=2, **kw):
    return make(ID, num_envs=n, **kw)


@pytest.mark.parametrize("size", [8, 32])
def test_layout_heads_masks_and_observation(size):
    with rts(size=size, max_units=8) as e:
        assert e.action_nvec == (TYPES, WINDOW, PRODUCTS) * 8
        assert e.entity_group_heads == 3
        assert e._mask_len == 8 * BLOCK
        assert e.mask_heads[:3] == ((0, (), TYPES), (TYPES, (TYPES,), WINDOW), (TYPES + TYPES * WINDOW, (TYPES,), PRODUCTS))
        assert e.mask_heads[3][0] == BLOCK
        assert [(n, d, s) for n, d, s in e._leaves] == [
            ("planes", np.dtype("uint8"), (11, size, size)),
            ("globals", np.dtype("uint16"), (2,)),
        ]
        assert e.agents == 2 and e.turns == "simultaneous"


def test_head_views_are_read_only_views_in_the_spec_shapes():
    with rts(max_units=8, outputs="owned") as e:
        m = e.reset().masks
        t, tg, p = head_views(m, e.mask_heads, e.entity_group_heads)
        assert (t.shape, tg.shape, p.shape) == ((2, 2, 8, 6), (2, 2, 8, 6, 81), (2, 2, 8, 6, 4))
        flat = m.reshape(2, 2, 8, BLOCK)
        assert (t == flat[..., :TYPES]).all()
        assert (tg == flat[..., TYPES : TYPES + TYPES * WINDOW].reshape(2, 2, 8, 6, 81)).all()
        assert (p == flat[..., TYPES + TYPES * WINDOW :].reshape(2, 2, 8, 6, 4)).all()
        for v in (t, tg, p):
            assert np.shares_memory(v, m)
            assert not v.flags.writeable
            assert not v.flags.c_contiguous


def test_head_views_follow_the_borrow_window_of_the_flat_mask():
    with rts(max_units=8) as e:
        m = e.reset().masks
        views = head_views(m, e.mask_heads, e.entity_group_heads)
        assert not is_stale(m) and not any(is_stale(v) for v in views)
        e.step(np.zeros((2, 2, 24), dtype=np.int32))
        assert is_stale(m) and all(is_stale(v) for v in views)


# -- the Rust goldens, replayed (plan r2, R3) --------------------------------

M64 = (1 << 64) - 1


def fnv(h, word):
    for b in int(word).to_bytes(8, "little"):
        h = ((h ^ b) * 0x0100_0000_01B3) & M64
    return h


class Rng:
    """The xorshift of crates/tessera-rl-rts/tests/golden.rs."""

    def __init__(self, s):
        self.s = s

    def below(self, n):
        s = self.s
        s ^= (s << 13) & M64
        s ^= s >> 7
        s ^= (s << 17) & M64
        self.s = s
        return s % n

    def pick(self, row):
        legal = np.flatnonzero(row)
        return int(legal[self.below(len(legal))])


def legal_actions(rng, masks, units):
    """golden.rs's `legal_actions`: per agent row, per unit, type, then
    target and product under it."""
    n, a, _ = masks.shape
    out = np.zeros((n, a, units * 3), dtype=np.int32)
    for i in range(n):
        for j in range(a):
            row = masks[i, j]
            for u in range(units):
                b = row[u * BLOCK : (u + 1) * BLOCK]
                t = rng.pick(b[:TYPES])
                target = rng.pick(b[TYPES + t * WINDOW : TYPES + (t + 1) * WINDOW])
                base = TYPES + TYPES * WINDOW
                product = rng.pick(b[base + t * PRODUCTS : base + (t + 1) * PRODUCTS])
                out[i, j, 3 * u : 3 * u + 3] = (t, target, product)
    return out


def golden_env():
    return make(ID, num_envs=2, seed=2026, outputs="owned", size=12, max_units=6, step_limit=400)


def test_rts_golden_state_hashes_match_rust():
    """RTS_GOLDEN of golden.rs: rewards, masks and state hashes folded."""
    with golden_env() as e:
        rng = Rng(0x7E55)
        masks = e.reset().masks.copy()
        h = 0xCBF2_9CE4_8422_2325
        for _ in range(96):
            out = e.step(legal_actions(rng, masks, 6))
            for r in out.rewards.reshape(-1):
                h = fnv(h, np.float32(r).view(np.uint32))
            masks = out.masks.copy()
            for m in masks.reshape(-1):
                h = fnv(h, int(m))
            for i in range(2):
                h = fnv(h, e.state_hash(i))
    assert h == 0x7DCD_1ABC_14AA_2177, hex(h)


def test_rts_observation_golden_matches_rust():
    """OBSERVATION_GOLDEN of golden.rs: both leaves' bytes, every step."""
    with golden_env() as e:
        rng = Rng(0x0B5E)
        masks = e.reset().masks.copy()
        h = 0xCBF2_9CE4_8422_2325
        for _ in range(300):
            out = e.step(legal_actions(rng, masks, 6))
            for leaf in ("planes", "globals"):
                for b in np.ascontiguousarray(out.observations[leaf]).view(np.uint8).reshape(-1):
                    h = fnv(h, int(b))
            masks = out.masks.copy()
    assert h == 0xE4D2_FA41_D533_CC23, hex(h)


def test_borrowed_and_owned_outputs_are_equal():
    with rts(seed=3, max_units=8) as b, rts(seed=3, max_units=8, outputs="owned") as o:
        rb, ro = b.reset(), o.reset()
        acts = np.zeros((2, 2, 24), dtype=np.int32)
        for _ in range(5):
            rb, ro = b.step(acts), o.step(acts)
            for k in ro.observations:
                assert (np.asarray(rb.observations[k]) == ro.observations[k]).all()
            assert (np.asarray(rb.masks) == ro.masks).all()


# -- empty slots (plan r2, R1) ------------------------------------------------


def test_empty_slots_count_only_non_noop_orders():
    with rts(max_units=8, outputs="owned") as e:
        e.reset()
        # A fresh game holds a Core and one Drone per side: slots 2..7 empty.
        e.step(np.zeros((2, 2, 24), dtype=np.int32))
        assert e.illegal_actions(0) == [0, 0]
        acts = np.zeros((2, 2, 24), dtype=np.int32)
        acts[:, 0, 3 * 5] = 1  # unit slot 5 of agent 0: a move, on an empty slot
        acts[:, 0, 3 * 6 + 2] = 1  # slot 6: a product on the no-op type
        e.step(acts)
        assert e.illegal_actions(0) == [2, 0]


# -- configuration and adapters ----------------------------------------------


def test_config_fields_and_errors():
    with rts(size=10, decision_ticks=2, max_units=4, step_limit=5, seat_swap=True) as e:
        assert e._leaves[0][2] == (11, 10, 10)
    for bad in ({"size": 7}, {"size": 33}, {"decision_ticks": 0}, {"max_units": 2}):
        with pytest.raises(_native.ConfigError):
            rts(**bad)
    with pytest.raises(TypeError):
        rts(seat_swap=1)
    with pytest.raises(_native.ArgumentError):
        rts(colour=1)


def test_actions_are_refused_before_the_pool_is_touched():
    with rts(max_units=4) as e:
        e.reset()
        before = e.snapshot(0)
        for bad in (np.zeros((2, 2), dtype=np.int32), np.zeros((2, 2, 11), dtype=np.int32), np.full((2, 2, 12), 81)):
            with pytest.raises(_native.ActionError):
                e.step(bad)
        assert e.snapshot(0) == before


def test_search_refuses_rts():
    search = pytest.importorskip("tessera_rl.search")
    with pytest.raises(_native.ArgumentError):
        search.Search(ID)


def test_pufferlib_refuses_rts():
    try:
        from tessera_rl import puffer
    except ImportError:
        pytest.skip("needs PufferLib (the puffer group)")
    with pytest.raises(_native.ArgumentError):
        puffer.make_puffer(ID)

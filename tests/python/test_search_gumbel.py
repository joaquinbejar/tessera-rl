# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Gumbel root from Python (#69): ``root="gumbel"`` on ``Search`` and
``SearchGroup``, the ``improved_policy`` and ``considered`` readers, the
refusals, rollback, and PUCT unchanged next to it."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl.search import Search, SearchGroup

# The Rust PUCT golden (crates/tessera-rl-search/tests/mcts.rs).
GOLDEN_VISITS = [1, 142, 18, 3, 69, 3, 20]


def root(moves, seed=21):
    env = t.make("tessera/Connect4-v1", seed=seed, auto_reset="disabled", outputs="owned")
    env.reset()
    for c in moves:
        env.step(np.array([[int(c), int(c)]]))
    return env.snapshot(0)


def canned(obs, legal, hashes):
    w = np.stack([1 + ((hashes >> np.uint64(8 * a)) & np.uint64(0xFF)).astype(np.float64) for a in range(7)], axis=1)
    priors = w / w.sum(axis=1, keepdims=True)
    values = ((hashes >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.float64) / 32767.5 - 1.0
    return priors, values


def gumbel(moves="33", seed=21, **options):
    options.setdefault("simulations", 64)
    s = Search(root="gumbel", **options)
    s.set_root(root(moves, seed))
    return s


def result(s):
    return (
        s.root_visits().tolist(),
        s.improved_policy().tobytes(),
        s.considered().tolist(),
        s.select_action(0.0),
        s.tree_hash(),
        s.calls,
    )


def test_gumbel_search_connect4_runs_and_reads_policy_and_considered():
    s = gumbel(gumbel_m=4)
    assert s.improved_policy() is None and s.considered() is None
    s.run(canned)
    pi = s.improved_policy()
    assert pi.dtype == np.float64 and pi.shape == (7,)
    considered = s.considered()
    assert considered.dtype == np.uint32 and considered.shape == (4,)
    assert s.root_visits().sum() == 64
    assert s.select_action(0.0) in considered.tolist()
    # A fresh array each time: changing one leaves the search's own alone.
    pi[:] = 0
    assert s.improved_policy().sum() > 0


@pytest.mark.parametrize("m", [1, 3, 16])
def test_gumbel_policy_on_full_column_sums_to_one_over_legal_and_zero_on_illegal(m):
    # Column 3 is full at the root: six legal actions.
    s = gumbel("333333", gumbel_m=m)
    s.run(canned)
    pi = s.improved_policy()
    legal = np.ones(7, dtype=bool)
    legal[3] = False
    assert pi[3] == 0.0
    assert (pi[legal] > 0).all()
    assert abs(pi.sum() - 1.0) < 1e-12
    considered = s.considered().tolist()
    assert len(considered) == min(m, 6) == len(set(considered))
    assert set(considered) <= {0, 1, 2, 4, 5, 6}


def test_gumbel_same_seed_twice_is_bit_identical():
    a, b = gumbel(gumbel_m=5), gumbel(gumbel_m=5)
    for _ in range(2):
        a.run(canned)
        b.run(canned)
        assert result(a) == result(b)


def test_gumbel_group_slots_equal_independent_searches():
    slots = [{"gumbel_m": 1}, {"gumbel_m": 16}, {"root": "puct"}]
    moves = ["33", "333333", "2"]
    g = SearchGroup(3, root="gumbel", simulations=64, slot_options=slots)
    solo = []
    for i, (m, o) in enumerate(zip(moves, slots)):
        g.set_root(i, root(m))
        options = {"simulations": 64, "root": "gumbel", **o}
        s = Search(**options)
        s.set_root(root(m))
        solo.append(s)
    for _ in range(2):
        g.run(canned)
        for s in solo:
            s.run(canned)
        for i, s in enumerate(solo):
            assert g.root_visits(i).tolist() == s.root_visits().tolist()
            assert g.tree_hash(i) == s.tree_hash()
            if i < 2:
                assert g.improved_policy(i).tobytes() == s.improved_policy().tobytes()
                assert g.considered(i).tolist() == s.considered().tolist()
                assert g.select_action(i, 0.0) == s.select_action(0.0)
    assert g.improved_policy(2) is None and g.considered(2) is None
    assert len(g.considered(0)) == 1


def test_gumbel_refusals_raise_the_existing_error_types():
    with pytest.raises(t.SearchError):
        Search(root="gumbel", noise_alpha=0.3, noise_epsilon=0.25)
    with pytest.raises(t.SearchError):
        Search(root="gumbel", gumbel_m=0)
    for key in ("c_visit", "c_scale"):
        for bad in (-1.0, float("nan"), float("inf")):
            with pytest.raises(t.SearchError):
                Search(root="gumbel", **{key: bad})
    with pytest.raises(t.ArgumentError):
        Search(root="alphazero")
    for key, value in (("gumbel_m", 4), ("c_visit", 50.0), ("c_scale", 1.0)):
        with pytest.raises(t.ArgumentError):
            Search(**{key: value})
        with pytest.raises(t.ArgumentError):
            SearchGroup(2, slot_options=[{key: value}, {}])
    with pytest.raises(t.SearchError):
        SearchGroup(2, slot_options=[{"root": "gumbel", "gumbel_m": 0}, {}])
    s = gumbel()
    s.run(canned)
    with pytest.raises(t.SearchError):
        s.select_action(1.0)
    assert s.select_action(0.0) == s.select_action(0.0)


def test_gumbel_evaluator_exception_mid_call_rolls_back_everything():
    s, fresh = gumbel(gumbel_m=4), gumbel(gumbel_m=4)
    s.run(canned)
    fresh.run(canned)
    before = result(s)
    batches = []
    boom = ValueError("boom")

    def flaky(obs, legal, hashes):
        batches.append(1)
        if len(batches) == 3:
            raise boom
        return canned(obs, legal, hashes)

    with pytest.raises(t.EvaluatorError) as info:
        s.run(flaky)
    assert info.value.__cause__ is boom
    assert result(s) == before
    # The retry equals a fresh second call.
    s.run(canned)
    fresh.run(canned)
    assert result(s) == result(fresh)


def test_gumbel_group_exception_rolls_back_every_slot():
    def make():
        g = SearchGroup(2, root="gumbel", simulations=64, gumbel_m=4)
        g.set_root(0, root("33"))
        g.set_root(1, root("2"))
        return g

    g, fresh = make(), make()
    g.run(canned)
    fresh.run(canned)
    before = [(g.tree_hash(i), g.improved_policy(i).tobytes(), g.considered(i).tolist()) for i in range(2)]

    def fail(obs, legal, hashes):
        raise RuntimeError("late")

    with pytest.raises(t.EvaluatorError):
        g.run(fail)
    assert [(g.tree_hash(i), g.improved_policy(i).tobytes(), g.considered(i).tolist()) for i in range(2)] == before
    g.run(canned)
    fresh.run(canned)
    for i in range(2):
        assert g.improved_policy(i).tobytes() == fresh.improved_policy(i).tobytes()
        assert g.tree_hash(i) == fresh.tree_hash(i)


def test_gumbel_set_root_clears_the_readers():
    s = gumbel()
    s.run(canned)
    s.set_root(root("2"))
    assert s.improved_policy() is None and s.considered() is None


def test_puct_explicit_root_equals_default_and_the_rust_golden():
    default, explicit = Search(simulations=256), Search(simulations=256, root="puct")
    for s in (default, explicit):
        s.set_root(root("33"))
        s.run(canned)
        assert s.root_visits().tolist() == GOLDEN_VISITS
        assert s.improved_policy() is None and s.considered() is None
    assert default.tree_hash() == explicit.tree_hash()
    assert [default.select_action(1.0) for _ in range(8)] == [explicit.select_action(1.0) for _ in range(8)]

# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PUCT search from Python (P2-M2, ADR-0043): the Python evaluator gets
each leaf's observation, legal actions and record hash; a canned evaluator
reproduces the Rust golden visits; every invalid response or exception is
an EvaluatorError (the evaluator's exception as its cause) with the tree
and the call counter unchanged."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl.search import Search

# The Rust golden (crates/tessera-rl-search/tests/mcts.rs): 256 simulations
# from moves "33" with the canned evaluator.
GOLDEN_VISITS = [1, 142, 18, 3, 69, 3, 20]


def root(moves, seed=21):
    env = t.make("tessera/Connect4-v1", seed=seed, auto_reset="disabled", outputs="owned")
    env.reset()
    for c in moves:
        env.step(np.array([[int(c), int(c)]]))
    return env.snapshot(0)


def canned(obs, legal, hashes):
    """Priors and value from the record hash, exactly as the Rust canned
    evaluator computes them."""
    w = np.stack([1 + ((hashes >> np.uint64(8 * a)) & np.uint64(0xFF)).astype(np.float64) for a in range(7)], axis=1)
    priors = w / w.sum(axis=1, keepdims=True)
    values = ((hashes >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.float64) / 32767.5 - 1.0
    return priors, values


def uniform(obs, legal, hashes):
    n = len(hashes)
    return np.full((n, 7), 1 / 7), np.zeros(n)


def search(moves="33", **options):
    s = Search(simulations=256, **options)
    s.set_root(root(moves))
    return s


def test_search_python_canned_evaluator_reproduces_the_rust_golden():
    s = search()
    s.run(canned)
    assert s.root_visits().tolist() == GOLDEN_VISITS
    assert s.calls == 1
    assert s.select_action(0.0) == 1


def test_search_evaluator_gets_the_movers_planes_legal_actions_and_hashes():
    seen = []

    def inspect(obs, legal, hashes):
        assert obs.shape == (len(hashes), 2, 6, 7) and obs.dtype == np.uint8
        assert legal.shape == (len(hashes), 7) and legal.dtype == np.bool_
        assert hashes.dtype == np.uint64
        for i in range(len(hashes)):
            mine, theirs = int(obs[i, 0].sum()), int(obs[i, 1].sum())
            # The mover has as many stones as the opponent, or one fewer.
            assert theirs - mine in (0, 1)
            # A column is legal exactly when its top cell is empty.
            top = obs[i, :, 0, :].sum(axis=0)
            assert (legal[i] == (top == 0)).all()
        seen.append(len(hashes))
        return uniform(obs, legal, hashes)

    s = search("3333332")
    s.run(inspect)
    assert sum(seen) > 50
    # Column 3 is full at the root: never visited.
    assert s.root_visits()[3] == 0


def bad(priors=None, values=None, exc=None):
    def evaluator(obs, legal, hashes):
        if exc is not None:
            raise exc
        n = len(hashes)
        p = np.full((n, 7), 1 / 7) if priors is None else priors(n)
        v = np.zeros(n) if values is None else values(n)
        return p, v

    return evaluator


@pytest.mark.parametrize(
    "evaluator",
    [
        bad(priors=lambda n: np.full((n, 6), 1 / 6)),  # wrong shape
        bad(values=lambda n: np.zeros(n + 1)),  # wrong length
        bad(values=lambda n: np.full(n, np.nan)),
        bad(values=lambda n: np.full(n, 2.0)),
        bad(priors=lambda n: np.tile([-0.1, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1], (n, 1))),
        bad(priors=lambda n: np.full((n, 7), 0.2)),  # sum 1.4
        bad(priors=lambda n: np.tile([0, 0, 0, 1.0, 0, 0, 0], (n, 1))),  # all mass on the full column
    ],
    ids=["shape", "length", "nan", "range", "negative", "sum", "no_legal_mass"],
)
def test_search_invalid_responses_raise_with_the_tree_unchanged(evaluator):
    s = search("333333")
    before = s.tree_hash()
    with pytest.raises(t.EvaluatorError):
        s.run(evaluator)
    assert s.tree_hash() == before
    assert s.calls == 0
    # The next call with a valid evaluator works.
    s.run(uniform)
    assert s.calls == 1


def test_search_evaluator_exception_is_the_cause_and_rolls_back():
    s = search()
    s.run(canned)
    before = s.tree_hash()
    boom = ValueError("boom")
    calls = []

    def flaky(obs, legal, hashes):
        calls.append(1)
        if len(calls) == 3:
            raise boom
        return canned(obs, legal, hashes)

    with pytest.raises(t.EvaluatorError) as info:
        s.run(flaky)
    assert info.value.__cause__ is boom
    assert s.tree_hash() == before
    assert s.calls == 1


def test_search_nested_call_from_the_evaluator_is_a_concurrency_error():
    s = search()

    def nested(obs, legal, hashes):
        s.root_visits()
        return uniform(obs, legal, hashes)

    before = s.tree_hash()
    with pytest.raises(t.EvaluatorError) as info:
        s.run(nested)
    assert isinstance(info.value.__cause__, t.ConcurrencyError)
    assert s.tree_hash() == before


def test_search_options_are_validated():
    with pytest.raises(t.ArgumentError):
        Search(env_id="tessera/Tag-v1")
    with pytest.raises(t.ArgumentError):
        Search(eviction="newest")
    with pytest.raises(t.ArgumentError):
        Search(noise_alpha=0.3)
    with pytest.raises(t.SearchError):
        Search(batch_size=99)
    with pytest.raises(t.RecordError):
        Search().set_root(b"not a record")

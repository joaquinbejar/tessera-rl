# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PUCT search from Python (P2-M2, ADR-0043): the Python evaluator gets
each leaf's observation, legal actions and record hash; a canned evaluator
reproduces the Rust golden visits; every invalid response or exception is
an EvaluatorError (the evaluator's exception as its cause) with the tree
and the call counter unchanged."""

import signal

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


@pytest.mark.parametrize(
    "response",
    [
        lambda n: (np.full((n, 7), 1 / 7, dtype=np.float32), np.zeros(n)),
        lambda n: (np.full((n, 7), 1 / 7), np.zeros(n, dtype=np.complex128)),
        lambda n: (np.full((7, n), 1 / 7).T, np.zeros(n)),  # not C-contiguous
        lambda n: ([[1 / 7] * 7] * n, np.zeros(n)),  # a list, no buffer
        lambda n: (np.full((n, 7), 1 / 7), np.zeros(n), None),  # arity 3
        lambda n: np.full((n, 7), 1 / 7),  # not a tuple
        lambda n: (np.full((1, 7), 1 / 7), np.zeros(1)),  # no broadcasting
    ],
    ids=["float32", "complex", "transposed", "list", "arity", "not_tuple", "broadcast"],
)
def test_search_wire_format_is_strict_and_never_casts(response):
    s = search("33")
    before = s.tree_hash()

    def evaluator(obs, legal, hashes):
        return response(len(hashes))

    with pytest.raises(t.EvaluatorError):
        s.run(evaluator)
    assert (s.tree_hash(), s.calls) == (before, 0)


def test_search_inputs_are_owned_and_keep_their_values():
    kept = []

    def keep(obs, legal, hashes):
        kept.append((obs, obs.copy(), legal, legal.copy(), hashes, hashes.copy()))
        return uniform(obs, legal, hashes)

    s = search("33")
    s.run(keep)
    s.run(keep)
    s.set_root(root("0"))
    s.run(uniform)
    assert len(kept) > 2
    for obs, obs0, legal, legal0, hashes, hashes0 in kept:
        assert np.array_equal(obs, obs0) and np.array_equal(legal, legal0) and np.array_equal(hashes, hashes0)
        assert obs.base is None or obs.flags.owndata or not isinstance(obs.base, memoryview)


def test_search_keyboard_interrupt_is_raised_unchanged_and_rolls_back():
    s = search()
    before = s.tree_hash()

    def interrupt(obs, legal, hashes):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as info:
        s.run(interrupt)
    assert type(info.value) is KeyboardInterrupt
    assert (s.tree_hash(), s.calls) == (before, 0)


def test_search_system_exit_is_raised_unchanged_and_rolls_back():
    s = search()
    before = s.tree_hash()

    def leave(obs, legal, hashes):
        raise SystemExit(3)

    with pytest.raises(SystemExit) as info:
        s.run(leave)
    assert info.value.code == 3
    assert (s.tree_hash(), s.calls) == (before, 0)


def test_search_exception_with_failing_str_is_chained():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    s = search()
    before = s.tree_hash()

    def fail(obs, legal, hashes):
        raise Unprintable

    with pytest.raises(t.EvaluatorError) as info:
        s.run(fail)
    assert isinstance(info.value.__cause__, Unprintable)
    assert (s.tree_hash(), s.calls) == (before, 0)


# 41 plies of a drawn game: one legal move, into a terminal draw, so after
# the root every batch is made of terminal leaves and calls no evaluator.
DRAW_41 = "545062455041104565311226266362030334314210"[:41]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs setitimer")
def test_search_signal_during_terminal_only_batches_interrupts_and_rolls_back():
    s = Search(simulations=2_000_000)
    s.set_root(root(DRAW_41))
    before = s.tree_hash()
    evaluations = []

    def counted(obs, legal, hashes):
        evaluations.append(len(hashes))
        return canned(obs, legal, hashes)

    def alarm(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        with pytest.raises(KeyboardInterrupt):
            s.run(counted)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert evaluations == [1]  # the root only
    assert (s.tree_hash(), s.calls) == (before, 0)
    assert s.root_visits().sum() == 0


def test_search_exception_after_commits_and_evictions_rolls_back_tree_calls_and_stream():
    def make():
        s = Search(simulations=200, capacity=48, eviction="oldest")
        s.set_root(root("33"))
        return s

    s, fresh = make(), make()
    s.run(canned)
    fresh.run(canned)
    before = s.tree_hash()
    batches = []

    def late_failure(obs, legal, hashes):
        batches.append(1)
        if len(batches) == 12:  # well after commits and evictions
            raise RuntimeError("late")
        return canned(obs, legal, hashes)

    with pytest.raises(t.EvaluatorError) as info:
        s.run(late_failure)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert len(batches) == 12
    assert (s.tree_hash(), s.calls) == (before, 1)
    # The retry equals a fresh second call, tree and search stream.
    s.run(canned)
    fresh.run(canned)
    assert s.tree_hash() == fresh.tree_hash()
    assert [s.select_action(1.0) for _ in range(16)] == [fresh.select_action(1.0) for _ in range(16)]


def test_search_first_call_failure_then_retry_equals_a_fresh_call():
    s, fresh = search(), search()

    def fail(obs, legal, hashes):
        raise ValueError("first")

    with pytest.raises(t.EvaluatorError):
        s.run(fail)
    s.run(canned)
    fresh.run(canned)
    assert s.root_visits().tolist() == fresh.root_visits().tolist() == GOLDEN_VISITS
    assert s.tree_hash() == fresh.tree_hash()


@pytest.mark.parametrize("call", ["run", "set_root", "select_action", "root_visits", "calls", "tree_hash"])
def test_search_nested_calls_are_concurrency_errors(call):
    s = search()
    record = root("4")
    calls = {
        "run": lambda: s.run(uniform),
        "set_root": lambda: s.set_root(record),
        "select_action": lambda: s.select_action(0.0),
        "root_visits": lambda: s.root_visits(),
        "calls": lambda: s.calls,
        "tree_hash": lambda: s.tree_hash(),
    }

    def nested(obs, legal, hashes):
        calls[call]()
        return uniform(obs, legal, hashes)

    with pytest.raises(t.EvaluatorError) as info:
        s.run(nested)
    assert isinstance(info.value.__cause__, t.ConcurrencyError)
    assert s.calls == 0


def test_search_call_from_another_thread_during_run_is_a_concurrency_error():
    import threading

    s = search()
    seen = []

    def evaluator(obs, legal, hashes):
        if not seen:
            def other():
                try:
                    s.root_visits()
                    seen.append("ok")
                except t.ConcurrencyError:
                    seen.append("concurrency")

            th = threading.Thread(target=other)
            th.start()
            th.join()
        return uniform(obs, legal, hashes)

    s.run(evaluator)
    assert seen == ["concurrency"]
    assert s.calls == 1
    assert s.root_visits().sum() == 256


def test_search_mixed_terminal_and_nonterminal_batches():
    # Slot 0 to move wins in column 0: batches mix terminal leaves (the
    # win) with evaluated ones.
    s = Search(simulations=200)
    s.set_root(root("010101"))
    sizes = []

    def count(obs, legal, hashes):
        sizes.append(len(hashes))
        return uniform(obs, legal, hashes)

    s.run(count)
    assert s.root_visits().sum() == 200
    assert sum(sizes) < 200, "terminal leaves are not sent to the evaluator"
    assert s.select_action(0.0) == 0

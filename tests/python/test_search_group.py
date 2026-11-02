# SPDX-License-Identifier: MIT OR Apache-2.0
"""SearchGroup from Python (ADR-0045).

Covered:
- with the canned evaluator, every slot equals an independent Search
  (visits, tree and stream position) and reproduces the Rust golden;
- one evaluator call serves every slot's leaves;
- a call is all or nothing, with a successful retry;
- the user's exceptions propagate untouched, even with a hostile
  __setattr__ or __str__, and signal-handler exceptions as themselves;
- group-created errors carry .slot, and last_failure() is reset per call;
- the whole group is guarded during a call (ConcurrencyError).
"""

import signal

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl.search import Search, SearchGroup

GOLDEN_VISITS = [1, 142, 18, 3, 69, 3, 20]
DRAW_41 = "545062455041104565311226266362030334314210"[:41]


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


def group(roots, **options):
    g = SearchGroup(len(roots), **options)
    for i, moves in enumerate(roots):
        g.set_root(i, root(moves))
    return g


def state(g, i):
    return g.root_visits(i).tolist(), g.tree_hash(i), g.calls(i)


def test_search_group_canned_slots_reproduce_the_golden_in_one_call_per_round():
    g = group(["33", "33", "33"], simulations=256)
    sizes = []

    def counted(obs, legal, hashes):
        sizes.append(len(hashes))
        return canned(obs, legal, hashes)

    g.run(counted)
    for i in range(3):
        assert g.root_visits(i).tolist() == GOLDEN_VISITS
    # The root batch has one row per slot; rounds carry up to 3 x 8 rows.
    assert sizes[0] == 3 and max(sizes) > 8
    alone = Search(simulations=256)
    alone.set_root(root("33"))
    calls = []
    alone.run(lambda o, l, h: calls.append(1) or canned(o, l, h))
    assert len(sizes) <= len(calls)


def test_search_group_slots_equal_independent_searches_with_slot_options_and_noise():
    roots = ["", "33", "3412"]
    per = [{"simulations": 64}, {"simulations": 100, "batch_size": 3}, {"simulations": 37, "batch_size": 1}]
    g = group(roots, slot_options=per, noise_alpha=0.3, noise_epsilon=0.25)
    solo = []
    for moves, o in zip(roots, per, strict=True):
        s = Search(noise_alpha=0.3, noise_epsilon=0.25, **o)
        s.set_root(root(moves))
        solo.append(s)
    for _ in range(2):
        g.run(canned)
        for s in solo:
            s.run(canned)
        for i, s in enumerate(solo):
            assert state(g, i) == (s.root_visits().tolist(), s.tree_hash(), s.calls)
            draws = [g.select_action(i, 1.0) for _ in range(5)]
            assert draws == [s.select_action(1.0) for _ in range(5)], f"slot {i}"


class Hostile(Exception):
    def __setattr__(self, name, value):
        raise AssertionError(f"__setattr__({name!r}) ran on a user exception")

    def __str__(self):
        raise AssertionError("__str__ ran on a user exception")


class HostileBase(BaseException):
    def __setattr__(self, name, value):
        raise AssertionError("__setattr__ ran on a user BaseException")


@pytest.mark.parametrize("kind", ["exception", "base"])
def test_search_group_user_exceptions_propagate_untouched_and_roll_back(kind):
    g = group(["", "33"], simulations=40)
    g.run(canned)
    before = [state(g, i) for i in range(2)]
    sent = Hostile() if kind == "exception" else HostileBase()
    seen = []

    def failing(obs, legal, hashes):
        seen.append(len(hashes))
        if len(seen) == 3:
            raise sent
        return canned(obs, legal, hashes)

    if kind == "exception":
        with pytest.raises(t.EvaluatorError) as info:
            g.run(failing)
        assert info.value.__cause__ is sent
        assert info.value.slot is None
    else:
        with pytest.raises(HostileBase) as info:
            g.run(failing)
        assert info.value is sent
    assert [state(g, i) for i in range(2)] == before
    f = g.last_failure()
    assert f is not None and f.slot is None and f.user_raised
    with pytest.raises(AttributeError):
        f.slot = 1  # frozen
    # A retry succeeds and resets last_failure.
    g.run(canned)
    assert g.last_failure() is None


def test_search_group_bad_row_names_its_slot_and_rolls_every_slot_back():
    g = group(["", "33", "44"], simulations=30)
    before = [state(g, i) for i in range(3)]
    calls = []

    def bad_last_row(obs, legal, hashes):
        calls.append(1)
        priors, values = canned(obs, legal, hashes)
        if len(calls) == 3:
            values[-1] = np.nan
        return priors, values

    with pytest.raises(t.EvaluatorError) as info:
        g.run(bad_last_row)
    assert info.value.slot == 2
    assert g.last_failure().slot == 2 and not g.last_failure().user_raised
    assert [state(g, i) for i in range(3)] == before


def test_search_group_terminal_root_names_its_slot_before_any_call():
    g = group(["", "1212121"], simulations=8)
    called = []
    with pytest.raises(t.TesseraError) as info:
        g.run(lambda o, l, h: called.append(1) or canned(o, l, h))
    assert info.value.slot == 1 and not called
    assert g.calls(0) == 0


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs setitimer")
@pytest.mark.parametrize("raised", [KeyboardInterrupt, ValueError, Hostile])
def test_search_group_signal_handler_exception_propagates_as_itself(raised):
    g = group([DRAW_41, DRAW_41], simulations=2_000_000)
    before = [state(g, i) for i in range(2)]
    sent = raised()

    def alarm(signum, frame):
        raise sent

    previous = signal.signal(signal.SIGALRM, alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.01)
        with pytest.raises(raised) as info:
            g.run(canned)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert info.value is sent
    assert [state(g, i) for i in range(2)] == before
    assert g.last_failure().user_raised


@pytest.mark.parametrize(
    "inside",
    [
        lambda g: g.run(canned),
        lambda g: g.set_root(1, root("4")),
        lambda g: g.set_active(1, False),
        lambda g: g.select_action(1, 0.0),
        lambda g: g.root_visits(0),
    ],
)
def test_search_group_is_guarded_during_a_call(inside):
    g = group(["", "33"], simulations=16)
    g.set_active(1, False)  # mutating an inactive slot is refused too
    caught = []

    def evaluator(obs, legal, hashes):
        try:
            inside(g)
        except t.ConcurrencyError as e:
            caught.append(e)
        return canned(obs, legal, hashes)

    g.run(evaluator)
    assert caught, "ConcurrencyError inside the call"
    assert g.active(1) is False


def test_search_group_inactive_slots_and_no_participants():
    g = group(["", "33"], simulations=16)
    g.set_active(0, False)
    h0 = g.tree_hash(0)
    g.run(canned)
    assert (g.calls(0), g.tree_hash(0), g.calls(1)) == (0, h0, 1)
    g.set_active(1, False)
    called = []
    g.run(lambda o, l, h: called.append(1) or canned(o, l, h))
    assert not called and g.calls(1) == 1


def test_search_group_options_and_bounds_are_validated():
    with pytest.raises(t.ArgumentError):
        SearchGroup(0)
    with pytest.raises(t.ArgumentError):
        SearchGroup(2, slot_options=[{}])
    with pytest.raises(t.ArgumentError):
        SearchGroup(1, slot_options=[{"env_id": "x"}])
    with pytest.raises(t.ArgumentError):
        SearchGroup(1).set_root(1, root(""))
    g = SearchGroup(4, capacity=1024)
    assert g.memory_bytes > 0 and g.trees == 4
    # Over the 16 GiB aggregate budget: refused from the options alone,
    # before any tree is allocated (4096 trees of 2^20 nodes would need
    # hundreds of GiB).
    import time

    start = time.monotonic()
    with pytest.raises(t.TesseraError, match="budget"):
        SearchGroup(4096, capacity=1 << 20)
    assert time.monotonic() - start < 1.0

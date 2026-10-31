# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native environment API (phase 1b, B2): construction, stepping, owned
outputs, records, errors and the reentrancy guard."""

import threading
import time

import numpy as np
import pytest

import tessera_rl as t


def cartpole(n=4, seed=7, **kw):
    return t.make("tessera/CartPole-v1", num_envs=n, seed=seed, **kw)


def test_cartpole_shapes_and_rewards():
    env = cartpole(3)
    s = env.reset()
    assert s.observations["obs"].shape == (3, 1, 4)
    assert s.observations["obs"].dtype == np.float32
    assert s.masks is None
    s = env.step(np.ones((3, 1), dtype=int))
    assert s.rewards.tolist() == [[1.0], [1.0], [1.0]]
    assert s.episode_terminated.shape == (3,)


def test_tag_shapes_masks_and_initial_agents():
    env = t.make("tessera/Tag-v1", num_envs=2, seed=3, pursuers=2, evaders=2, radius=1)
    s = env.reset()
    assert s.observations["window"].shape == (2, 8, 3, 3, 3)
    assert s.observations["window"].dtype == np.uint8
    assert s.masks.shape == (2, 8, 5)
    assert s.alive[0].tolist() == [True] * 4 + [False] * 4
    # Stay is always legal.
    assert s.masks[:, :4, 0].all()


def test_same_seed_same_trajectory_and_different_seed_differs():
    def run(seed):
        env = cartpole(4, seed)
        env.reset()
        out = []
        for step in range(50):
            s = env.step(np.full((4, 1), step % 2))
            out.append(s.observations["obs"].copy())
        return np.stack(out)

    assert np.array_equal(run(11), run(11))
    assert not np.array_equal(run(11), run(12))


def test_outputs_are_owned_and_survive_later_steps():
    env = cartpole(2)
    first = env.reset().observations["obs"]
    kept = first.copy()
    for _ in range(5):
        env.step(np.zeros((2, 1)))
    assert np.array_equal(first, kept)
    assert first.flags.writeable


def test_snapshot_import_round_trip_continues_identically():
    env = cartpole(2)
    env.reset()
    env.step(np.zeros((2, 1)))
    record = env.snapshot(0)
    expected = [env.step(np.ones((2, 1))).observations["obs"][0].copy() for _ in range(5)]
    env.import_record(1, record)
    got = [env.step(np.ones((2, 1))).observations["obs"][1].copy() for _ in range(5)]
    assert all(np.array_equal(a, b) for a, b in zip(expected, got, strict=True))


def test_fork_copies_instances():
    env = cartpole(3)
    env.reset()
    env.fork(0, [1, 2])
    s = env.step(np.zeros((3, 1)))
    assert np.array_equal(s.observations["obs"][0], s.observations["obs"][1])
    assert np.array_equal(s.observations["obs"][0], s.observations["obs"][2])


def test_errors_are_typed():
    with pytest.raises(t.ConfigError):
        t.make("tessera/Tag-v1", pursuers=5, evaders=4)
    with pytest.raises(t.ArgumentError):
        cartpole(bogus=1)
    with pytest.raises(t.ArgumentError):
        t.make("tessera/Nope-v0")
    env = cartpole(1)
    with pytest.raises(t.PendingResetError):
        env.step(np.zeros((1, 1)))
    env.reset()
    with pytest.raises(t.ActionError):
        env.step(np.full((1, 1), 7))
    with pytest.raises(t.ActionError):
        env.step(np.zeros((2, 1)))
    with pytest.raises(t.RecordError):
        env.import_record(0, b"not a record")
    with pytest.raises(t.ArgumentError):
        env.snapshot(9)
    assert issubclass(t.ActionError, t.TesseraError)


def test_close_is_idempotent_and_later_calls_fail():
    env = cartpole(1)
    obs = env.reset().observations["obs"]
    env.close()
    env.close()
    assert obs.shape == (1, 1, 4)
    with pytest.raises(t.ArgumentError):
        env.reset()


def test_concurrent_calls_raise_concurrency_error():
    # A large Tag pool makes each step long enough to overlap; the step
    # releases the interpreter, so the main thread runs meanwhile.
    env = t.make("tessera/Tag-v1", num_envs=20000, seed=1)
    env.reset()
    acts = np.zeros((20000, 8), dtype=int)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            env.step(acts)

    thread = threading.Thread(target=worker)
    thread.start()
    seen = False
    deadline = time.monotonic() + 20
    try:
        while not seen and time.monotonic() < deadline:
            try:
                env.is_poisoned
            except t.ConcurrencyError:
                seen = True
    finally:
        stop.set()
        thread.join()
    assert seen, "no overlapping call observed"
    assert not env.is_poisoned


def test_workers_match_inline():
    def run(workers):
        env = cartpole(16, 5, workers=workers)
        env.reset()
        for _ in range(30):
            s = env.step(np.ones((16, 1)))
        return s.observations["obs"].copy()

    assert np.array_equal(run(0), run(3))


def test_step_and_close_from_two_threads_never_crash():
    # TESSERA-REVIEWER R-PR17: closing while another thread steps must be
    # either rejected (ConcurrencyError) or happen between steps; a step on
    # a closed environment raises ArgumentError. Nothing else, no crash.
    for _ in range(20):
        env = t.make("tessera/Tag-v1", num_envs=2000, seed=2)
        env.reset()
        acts = np.zeros((2000, 8), dtype=int)
        errors = []

        def stepper():
            for _ in range(50):
                try:
                    env.step(acts)
                except (t.ConcurrencyError, t.ArgumentError, t.PendingResetError) as e:
                    errors.append(type(e))
                    if isinstance(e, t.ArgumentError):
                        return

        thread = threading.Thread(target=stepper)
        thread.start()
        while True:
            try:
                env.close()
                break
            except t.ConcurrencyError:
                time.sleep(0)
        thread.join()
        with pytest.raises(t.ArgumentError):
            env.step(acts)


def test_step_outputs_are_never_those_of_a_concurrent_reset():
    # codex R-PR17: a step must return its own outputs, never those of a
    # reset admitted between the mutation and the copy. Every step here is
    # at most 4 steps after a reset, so no CartPole instance has terminated
    # and every step reward is 1; a reset's rewards are all 0.
    env = cartpole(20000, 3)
    env.reset()
    acts = np.ones((20000, 1), dtype=np.int32)
    stop = threading.Event()
    bad = []

    def stepper():
        n = 0
        while not stop.is_set():
            try:
                if n % 4 == 0:
                    env.reset()
                s = env.step(acts)
                if not np.all(s.rewards == 1.0):
                    bad.append(s.rewards.min())
                n += 1
            except t.ConcurrencyError:
                pass

    thread = threading.Thread(target=stepper)
    thread.start()
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            try:
                env.reset()
            except t.ConcurrencyError:
                pass
    finally:
        stop.set()
        thread.join()
    assert not bad


@pytest.mark.parametrize(
    "actions",
    [
        np.full((2, 1), 0.9),
        np.full((2, 1), 2**32, dtype=np.int64),
        np.full((2, 1), -(2**31) - 1, dtype=np.int64),
        np.full((2, 1), np.nan),
        np.full((2, 1), np.inf),
        np.array([["a"], ["b"]]),
        [[0.5], [1]],
    ],
)
def test_step_inexact_actions_raise_action_error_before_mutation(actions):
    env = cartpole(2, 7)
    env.reset()
    before = env.snapshot(0)
    with pytest.raises(t.ActionError):
        env.step(actions)
    assert env.snapshot(0) == before


def test_step_exact_float_and_int64_actions_are_accepted():
    a = cartpole(2, 7)
    b = cartpole(2, 7)
    a.reset()
    b.reset()
    sa = a.step(np.ones((2, 1), dtype=np.float64))
    sb = b.step(np.ones((2, 1), dtype=np.int64))
    assert np.array_equal(sa.observations["obs"], sb.observations["obs"])

# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Pipelined mode from Python (ADR-0047 section 9): send and recv,
borrow windows, staleness, interruption and threads."""

import signal
import threading
import time

import numpy as np
import pytest

import tessera_rl as t


def pipelined(n=8, seed=7, env="tessera/CartPole-v1", **kw):
    kw.setdefault("chunk_size", 2)
    return t.make(env, num_envs=n, seed=seed, mode="pipelined", **kw)


def actions(k, step):
    return (np.arange(k) + step) % 2


def test_pipelined_cycle_is_bit_identical_to_sync_per_half():
    n = 8
    sync = t.make("tessera/CartPole-v1", num_envs=n, seed=7, outputs="owned")
    sync.reset()
    env = pipelined(n, outputs="owned")
    env.reset()
    a, b = env.halves
    assert (a.start, a.stop, b.start, b.stop) == (0, 4, 4, 8)
    for step in range(100):
        all_actions = actions(n, step)
        expected = sync.step(all_actions)
        env.send(0, all_actions[a.start : a.stop])
        env.send(1, all_actions[b.start : b.stop])
        for half in (0, 1):
            got = env.recv()
            assert got.half == half
            ids = got.ids
            assert got.step.observations["obs"].tobytes() == expected.observations["obs"][ids.start : ids.stop].tobytes()
            assert got.step.rewards.tobytes() == expected.rewards[ids.start : ids.stop].tobytes()
            assert np.array_equal(got.step.episode_terminated, expected.episode_terminated[ids.start : ids.stop])
    env.close()


def test_pipelined_borrowed_views_cover_their_half_and_match_owned():
    borrowed, owned = pipelined(), pipelined(outputs="owned")
    for got, want in zip(borrowed.reset(), owned.reset(), strict=True):
        assert got.step.observations["obs"].shape[0] == len(got.ids)
        assert got.step.observations["obs"].tobytes() == want.step.observations["obs"].tobytes()
    for env in (borrowed, owned):
        env.send(0, actions(4, 0))
    assert borrowed.recv().step.rewards.tobytes() == owned.recv().step.rewards.tobytes()


def test_pipelined_masks_and_agents_keep_their_shape():
    env = pipelined(4, env="tessera/Connect4-v1", chunk_size=1, workers=2)
    first, second = env.reset()
    assert first.step.masks.shape == (2, env.agents, env.action_choices)
    assert second.ids == range(2, 4)


def test_pipelined_view_goes_stale_at_its_own_send_only():
    env = pipelined()
    first, second = env.reset()
    a, b = first.step.rewards, second.step.rewards
    assert not t.is_stale(a) and not t.is_stale(b)
    env.send(0, actions(4, 0))
    assert t.is_stale(a)
    assert not t.is_stale(b)
    got = env.recv()
    assert not t.is_stale(got.step.rewards)
    env.send(1, actions(4, 0))
    assert t.is_stale(b)
    env.recv()


def test_pipelined_refused_send_keeps_the_view_fresh():
    env = pipelined()
    first, _ = env.reset()
    with pytest.raises(t.ActionError):
        env.send(0, actions(3, 0))
    assert not t.is_stale(first.step.rewards)
    env.send(0, actions(4, 0))
    env.recv()


def test_pipelined_step_and_sync_calls_raise_wrong_mode():
    env = pipelined()
    env.reset()
    for call in (lambda: env.step(actions(8, 0)), lambda: env.snapshot(0), lambda: env.fork(0, [1])):
        with pytest.raises(t.ArgumentError, match="not available in pipelined mode"):
            call()
    sync = t.make("tessera/CartPole-v1", num_envs=2)
    with pytest.raises(t.ArgumentError, match="not available in sync mode"):
        sync.send(0, [0])
    with pytest.raises(t.ArgumentError, match="not available in sync mode"):
        sync.recv()


def test_make_async_and_unknown_modes_are_refused():
    with pytest.raises(t.ArgumentError, match="not available yet"):
        t.make("tessera/CartPole-v1", mode="async")
    with pytest.raises(t.ArgumentError, match="mode must be"):
        t.make("tessera/CartPole-v1", mode="fast")


def test_pipelined_too_few_chunks_is_a_config_error():
    with pytest.raises(t.ConfigError):
        pipelined(1, chunk_size=1)


def test_pipelined_recv_with_nothing_in_flight_raises():
    env = pipelined()
    env.reset()
    with pytest.raises(t.ConcurrencyError):
        env.recv()


def test_pipelined_cycle_ahead_keeps_the_token():
    env = pipelined()
    first, _ = env.reset()
    env.send(0, actions(4, 0))
    env.recv()
    env.send(0, actions(4, 1))
    fresh = env.recv()
    with pytest.raises(t.ConcurrencyError, match="(?i)cycle"):
        env.send(0, actions(4, 2))
    assert not t.is_stale(fresh.step.rewards)
    env.send(1, actions(4, 0))
    env.recv()
    env.send(0, actions(4, 2))
    env.recv()
    assert t.is_stale(first.step.rewards)


def test_pipelined_reset_with_work_in_flight_is_refused_then_works():
    env = pipelined()
    env.reset()
    env.send(0, actions(4, 0))
    with pytest.raises(t.ConcurrencyError):
        env.reset()
    env.recv()
    assert len(env.reset()) == 2


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs setitimer")
def test_pipelined_interrupted_recv_leaves_the_step_in_flight():
    """A handler raising during the wait propagates, and the same step is
    received afterwards, identical to an uninterrupted reference."""

    class Stop(Exception):
        pass

    armed = [False]

    def handler(_signum, _frame):
        if armed[0]:
            armed[0] = False
            raise Stop

    n = 1 << 16
    env = pipelined(n, chunk_size=64, workers=1, outputs="owned")
    reference = pipelined(n, chunk_size=64, workers=1, outputs="owned")
    env.reset()
    reference.reset()
    half = len(env.halves[0])
    interrupted = False
    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 1e-5, 1e-5)
    try:
        for step in range(500):
            acts = actions(half, step)
            env.send(0, acts)
            reference.send(0, acts)
            want = reference.recv()
            armed[0] = True
            try:
                got = env.recv()
            except Stop:
                armed[0] = False
                assert not env.is_poisoned
                try:
                    got = env.recv()
                    interrupted = True
                except t.ConcurrencyError:
                    # The handler ran in Python after the native recv had
                    # returned: nothing to retry, so not an interrupted wait.
                    got = None
            armed[0] = False
            if got is not None:
                assert got.step.observations["obs"].tobytes() == want.step.observations["obs"].tobytes()
            if interrupted:
                break
            for pool in (env, reference):
                pool.send(1, actions(len(pool.halves[1]), step))
                pool.recv()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert interrupted
    env.close()


def test_pipelined_recv_from_another_thread_returns_the_sent_half():
    env = pipelined()
    env.reset()
    results = []
    env.send(0, actions(4, 0))
    thread = threading.Thread(target=lambda: results.append(env.recv().half))
    thread.start()
    thread.join(30)
    assert results == [0]


def test_pipelined_two_threads_send_and_recv():
    env = pipelined(256, chunk_size=64, workers=2)
    env.reset()
    rounds = 200
    errors = []
    sent = threading.Semaphore(0)
    room = threading.Semaphore(1)

    def sender():
        try:
            for step in range(rounds):
                room.acquire()
                env.send(0, actions(128, step))
                sent.release()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def receiver():
        try:
            for _ in range(rounds):
                sent.acquire()
                assert env.recv().half == 0
                # The other half advances too, so cycle 0 commits.
                env.send(1, actions(128, 0))
                assert env.recv().half == 1
                room.release()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=f) for f in (sender, receiver)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(60)
    assert not errors
    env.close()


def test_pipelined_two_concurrent_recv_one_is_refused():
    n = 1 << 16
    env = pipelined(n, chunk_size=64, workers=1)
    env.reset()
    half = len(env.halves[0])
    refused = 0
    for round_ in range(200):
        env.send(round_ % 2, actions(half, round_))
        start = threading.Barrier(2)
        outcomes = []

        def race():
            start.wait()
            try:
                env.recv()
                outcomes.append("received")
            except t.ConcurrencyError as e:
                outcomes.append("refused" if "ConcurrentRecv" in str(e) else "nothing")

        thread = threading.Thread(target=race)
        thread.start()
        race()
        thread.join(30)
        # One of them got the half; the other was refused, or came after.
        assert sorted(outcomes) in (["received", "refused"], ["nothing", "received"])
        if "refused" in outcomes:
            refused += 1
            break
    env.close()
    assert refused


def test_pipelined_close_with_work_in_flight_then_calls_refuse():
    env = pipelined()
    first, _ = env.reset()
    env.send(1, actions(4, 0))
    env.close()
    env.close()
    assert first.step.rewards.shape == (4, 1)
    with pytest.raises(t.ArgumentError, match="closed"):
        env.recv()
    with pytest.raises(t.ArgumentError, match="closed"):
        env.send(0, actions(4, 0))


def test_pipelined_views_outlive_a_dropped_environment():
    env = pipelined()
    first, _ = env.reset()
    obs = first.step.observations["obs"]
    before = obs.copy()
    env.send(1, actions(4, 0))
    del env
    assert np.array_equal(obs, before)

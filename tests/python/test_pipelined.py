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


#: Per-agent output keys, and the keys with one value per instance.
ROW_KEYS = ("rewards", "agent_terminated", "agent_truncated", "alive", "acting", "masks")
EPISODE_KEYS = ("episode_terminated", "episode_truncated")


def assert_rows_equal(sync_out, half_out, ids, n):
    """Every output field of ``half_out`` equals rows ``ids`` of ``sync_out``."""

    def check(key, whole, part):
        whole, part = bytes(whole), bytes(part)
        row = len(whole) // n
        assert part == whole[row * ids.start : row * ids.stop], key

    for key in (*ROW_KEYS, *EPISODE_KEYS):
        check(key, sync_out[key], half_out[key])
    for key in ("observations", "terminal_observations"):
        assert len(sync_out[key]) == len(half_out[key]), key
        for whole, part in zip(sync_out[key], half_out[key], strict=True):
            check(key, whole, part)


@pytest.mark.parametrize(
    ("env_id", "auto_reset"),
    [
        ("tessera/CartPole-v1", "same_step"),
        ("tessera/CartPole-v1", "next_step"),
        ("tessera/Tag-v1", "same_step"),
    ],
)
def test_pipelined_cycle_is_bit_identical_to_sync_in_every_field(env_id, auto_reset):
    n = 8
    common = dict(num_envs=n, seed=7, outputs="owned", auto_reset=auto_reset, max_episode_steps=5)
    sync = t.make(env_id, **common)
    env = t.make(env_id, mode="pipelined", chunk_size=2, **common)
    sync.reset()
    env.reset()
    agents = sync.agents
    ended = terminal_rows = 0
    rng = np.random.default_rng(3)
    for _ in range(40):
        # Action 0 is legal for every agent of both environments' rows when
        # sampled from {0}; CartPole also samples its second action.
        high = 2 if env_id.endswith("CartPole-v1") else 1
        acts = rng.integers(0, high, size=n * agents, dtype=np.int32)
        expected = sync._native.step(acts.tobytes())
        for half, ids in enumerate(env.halves):
            part = acts[ids.start * agents : ids.stop * agents]
            env._native.send(half, part.tobytes(), env._delivered)
        for half, ids in enumerate(env.halves):
            got = env._native.recv(env._delivered)
            env._delivered = got["seq"]
            assert got["half"] == half
            assert_rows_equal(expected, got["outputs"], ids, n)
        ended += sum(expected["episode_terminated"]) + sum(expected["episode_truncated"])
        terminal_rows += len(expected["terminal_observations"])
    assert ended, "no episode ended, so the end flags were not exercised"
    if auto_reset == "same_step":
        assert terminal_rows, "no terminal observations were compared"
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


def test_make_unknown_modes_are_refused():
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


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
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
    done = threading.Event()
    main = threading.main_thread().ident

    def ticks():
        while not done.is_set():
            signal.pthread_kill(main, signal.SIGUSR1)
            time.sleep(1e-4)

    previous = signal.signal(signal.SIGUSR1, handler)
    ticker = threading.Thread(target=ticks)
    ticker.start()
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
                got = env.recv()
                interrupted = True
            armed[0] = False
            assert got.step.observations["obs"].tobytes() == want.step.observations["obs"].tobytes()
            if interrupted:
                break
            for pool in (env, reference):
                pool.send(1, actions(len(pool.halves[1]), step))
                pool.recv()
    finally:
        armed[0] = False
        done.set()
        # Every tick has been sent before the handler goes back.
        ticker.join()
        signal.signal(signal.SIGUSR1, previous)
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


def test_pipelined_view_over_running_rows_outlives_the_dropped_environment():
    env = pipelined(outputs="borrowed")
    reference = pipelined(outputs="owned")
    first, _ = env.reset()
    reference.reset()
    obs = first.step.observations["obs"]
    env.send(0, actions(4, 0))
    assert t.is_stale(obs)
    # Half 0's rows are the ones running: dropping drains that step, and the
    # view then shows its result.
    del env, first
    reference.send(0, actions(4, 0))
    assert obs.tobytes() == reference.recv().step.observations["obs"].tobytes()


def test_pipelined_view_is_stale_before_any_worker_write():
    n = 1 << 14
    env = pipelined(n, chunk_size=64, workers=1)
    half = len(env.halves[0])
    obs = env.reset()[0].step.observations["obs"]
    other = env.halves[1]
    violations = []
    stop = threading.Event()

    def watcher(array, original):
        # Rows that changed while the view still reads fresh would be a
        # write ahead of the generation. Read first, then ask.
        while not stop.is_set():
            changed = array.tobytes() != original
            if changed and not t.is_stale(array):
                violations.append(True)
                return

    for step in range(30):
        current = env.reset()[0].step.observations["obs"] if step else obs
        original = current.tobytes()
        stop.clear()
        thread = threading.Thread(target=watcher, args=(current, original))
        thread.start()
        env.send(0, actions(half, step))
        env.recv()
        stop.set()
        thread.join(30)
    assert len(other) == n - half
    assert not violations


def test_pipelined_result_survives_an_exception_after_native_recv(monkeypatch):
    env = pipelined(outputs="owned")
    env.reset()
    env.send(0, actions(4, 0))
    real = env._half
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            # What a signal handler raising right after the native call does.
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_half", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    # Not yet seen by the caller: still in flight for send.
    with pytest.raises(t.ConcurrencyError, match="in flight"):
        env.send(0, actions(4, 1))
    got = env.recv()
    assert got.half == 0
    assert calls[0] == calls[1]
    assert not t.is_stale(got.step.rewards)
    # Delivered once: the next recv has nothing, and the half can go again.
    with pytest.raises(t.ConcurrencyError, match="(?i)nothing"):
        env.recv()
    env.send(0, actions(4, 1))
    env.recv()


def test_pipelined_snapshot_all_restores_identical_futures():
    env = pipelined(outputs="owned")
    env.reset()
    for half in (0, 1):
        env.send(half, actions(4, 0))
    env.recv()
    with pytest.raises(t.ConcurrencyError):
        env.snapshot_all()
    env.recv()
    taken = env.snapshot_all()

    def play():
        out = []
        for step in range(1, 6):
            for half in (0, 1):
                env.send(half, actions(4, step))
            out += [env.recv().step.observations["obs"].tobytes() for _ in (0, 1)]
        return out

    first = play()
    halves = env.restore_all(taken)
    assert [h.half for h in halves] == [0, 1]
    assert play() == first


def test_pipelined_restore_all_is_refused_with_work_or_a_foreign_snapshot():
    env = pipelined()
    env.reset()
    other = pipelined()
    other.reset()
    with pytest.raises(t.ArgumentError, match="only the pool"):
        env.restore_all(other.snapshot_all())
    with pytest.raises(TypeError):
        env.restore_all(b"bytes")
    taken = env.snapshot_all()
    env.send(0, actions(4, 0))
    with pytest.raises(t.ConcurrencyError):
        env.restore_all(taken)
    env.recv()
    assert len(env.restore_all(taken)) == 2
    sync = t.make("tessera/CartPole-v1", num_envs=2)
    with pytest.raises(t.ArgumentError, match="not available in sync mode"):
        sync.snapshot_all()


@pytest.mark.parametrize("writer", ["reset", "restore_all"])
def test_pipelined_failed_writer_leaves_a_pending_delivery_repeatable(writer, monkeypatch):
    env = pipelined(outputs="owned")
    env.reset()
    taken = env.snapshot_all()
    env.send(0, actions(4, 0))
    env.send(1, actions(4, 0))
    real = env._half
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_half", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    # Half 1 is unreceived, so the writer fails, after releasing the tokens.
    with pytest.raises(t.ConcurrencyError):
        env.reset() if writer == "reset" else env.restore_all(taken)
    with pytest.raises(t.ConcurrencyError, match="in flight"):
        env.send(0, actions(4, 1))
    got = env.recv()
    assert (got.half, calls[0]) == (0, calls[1])
    assert env.recv().half == 1
    env.send(0, actions(4, 1))
    assert env.recv().half == 0


@pytest.mark.parametrize("clear", ["reset", "restore_all", "close"])
def test_pipelined_dead_delivery_thread_refuses_recv_until_cleared(clear):
    env = pipelined()
    env.reset()
    taken = env.snapshot_all()
    env.send(0, actions(4, 0))
    thread = threading.Thread(target=lambda: env._native.recv(env._delivered))
    thread.start()
    thread.join(30)
    with pytest.raises(t.ConcurrencyError, match="ConcurrentRecv"):
        env.recv()
    if clear == "reset":
        env.reset()
    elif clear == "restore_all":
        env.restore_all(taken)
    else:
        env.close()
        with pytest.raises(t.ArgumentError, match="closed"):
            env.recv()
        return
    with pytest.raises(t.ConcurrencyError, match="(?i)nothing"):
        env.recv()
    env.send(0, actions(4, 0))
    assert env.recv().half == 0


def test_pipelined_half_view_exports_its_own_rows_through_dlpack():
    n = 8
    env = pipelined(n)
    reference = pipelined(n, outputs="owned")
    env.reset()
    reference.reset()
    second = env.halves[1]
    assert second.start > 0
    env.send(1, actions(len(second), 0))
    reference.send(1, actions(len(second), 0))
    want = reference.recv()
    out = env._native.recv(env._delivered)
    env._delivered = out["seq"]
    view = out["outputs"]["observations"][0]
    arr = np.from_dlpack(view)
    assert arr.shape == (len(second), 1, 4)
    assert not arr.flags.writeable
    assert arr.__array_interface__["data"][0] == np.asarray(view).__array_interface__["data"][0]
    assert arr.tobytes() == want.step.observations["obs"].tobytes()
    env.close()
    reference.close()


def test_pipelined_refused_reset_leaves_views_fresh_and_a_real_one_ends_them():
    env = pipelined()
    first, second = env.reset()
    with pytest.raises(t.ArgumentError):
        env._native.reset([])
    assert not t.is_stale(first.step.rewards)
    assert not t.is_stale(second.step.rewards)
    env.reset()
    assert t.is_stale(first.step.rewards)
    assert t.is_stale(second.step.rewards)
    env.close()


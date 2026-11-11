# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Async mode from Python (ADR-0047 section 9): per-instance identity
with Sync, batches, output windows, delivery, interruption and threads."""

import signal
import threading
import time

import numpy as np
import pytest

import tessera_rl as t

ROW_KEYS = ("rewards", "agent_terminated", "agent_truncated", "alive", "acting", "masks")
EPISODE_KEYS = ("episode_terminated", "episode_truncated")


def asy(n=8, batch=4, env="tessera/CartPole-v1", **kw):
    kw.setdefault("seed", 7)
    return t.make(env, num_envs=n, mode="async", batch_size=batch, **kw)


def zeros(ids, agents=1):
    return np.zeros(len(ids) * agents, dtype=np.int32)


def rows_of(buf, rows):
    buf = bytes(buf)
    size = len(buf) // rows
    return [buf[i * size : (i + 1) * size] for i in range(rows)]


@pytest.mark.parametrize("workers", [1, 2, 7])
@pytest.mark.parametrize(
    ("env_id", "n", "batch"),
    [
        ("tessera/CartPole-v1", 8, 4),
        ("tessera/CartPole-v1", 8, 3),
        ("tessera/Tag-v1", 6, 2),
        ("tessera/Tag-v1", 7, 4),
        ("tessera/Connect4-v1", 8, 8),
    ],
)
def test_async_instances_are_bit_identical_to_sync_in_every_field(workers, env_id, n, batch):
    common = dict(num_envs=n, seed=7, outputs="owned", auto_reset="same_step", max_episode_steps=5)
    sync = t.make(env_id, **common)
    sync.reset()
    env = t.make(env_id, mode="async", batch_size=batch, workers=workers, **common)
    env.reset()
    agents = sync.agents
    rng = np.random.default_rng(3)
    high = 2 if env_id.endswith("CartPole-v1") else 1
    legal_first = env_id.endswith("Connect4-v1")
    # Per instance, the rows each step must produce, as Sync produced them.
    expected = [[] for _ in range(n)]
    actions_of = [[] for _ in range(n)]
    masks = sync._native.outputs()["masks"] if legal_first else None
    ended = 0
    steps = 12
    for _ in range(steps):
        acts = rng.integers(0, high, size=n * agents, dtype=np.int32)
        if legal_first:
            # The first legal column of the acting agent.
            m = np.frombuffer(masks, dtype=bool).reshape(n, agents, -1)
            acts = m.argmax(axis=2).reshape(-1).astype(np.int32)
        out = sync._native.step(acts.tobytes())
        masks = out["masks"]
        fields = {k: rows_of(out[k], n) for k in (*ROW_KEYS, *EPISODE_KEYS)}
        for key in ("observations", "terminal_observations"):
            fields[key] = [rows_of(leaf, n) for leaf in out[key]]
        ended += sum(out["episode_terminated"]) + sum(out["episode_truncated"])
        for i in range(n):
            expected[i].append(fields)
            actions_of[i].append(acts[i * agents : (i + 1) * agents])
    assert ended
    # Instances advance out of step: each batch's ids go again at once, and
    # an instance out of steps just stops, until too few remain for a batch.
    done = [0] * n
    idle = list(range(n))
    out_now = 0
    compared = 0
    while True:
        go = [i for i in idle if done[i] < steps]
        idle = [i for i in idle if done[i] >= steps]
        if go:
            part = np.concatenate([actions_of[i][done[i]] for i in go])
            env._native.send(go, part.tobytes(), env._delivered)
            out_now += len(go)
        if out_now < batch:
            break
        got = env._native.recv(env._delivered)
        env._delivered = got["seq"]
        ids = got["ids"]
        assert len(ids) == batch
        out_now -= batch
        out = got["outputs"]
        for pos, i in enumerate(ids):
            want = expected[i][done[i]]
            for key in (*ROW_KEYS, *EPISODE_KEYS):
                assert rows_of(out[key], batch)[pos] == want[key][i], key
            for key in ("observations", "terminal_observations"):
                for leaf, whole in zip(out[key], want[key], strict=True):
                    assert rows_of(leaf, batch)[pos] == whole[i], key
            done[i] += 1
            compared += 1
            idle.append(i)
    assert compared >= n * (steps - 1) - batch
    env.close()


def test_async_batch_size_rules():
    with pytest.raises(t.ArgumentError, match="needs batch_size"):
        t.make("tessera/CartPole-v1", mode="async")
    with pytest.raises(t.ArgumentError, match="belong to"):
        t.make("tessera/CartPole-v1", batch_size=1)
    with pytest.raises(t.ConfigError):
        asy(4, 5)
    with pytest.raises(t.ArgumentError, match="at least 1"):
        asy(4, 0)
    with pytest.raises(t.ArgumentError, match="needs outputs"):
        asy(4, 2, outputs="owned", gather="indexed")


def test_async_batch_unreachable_and_nothing_in_flight():
    env = asy(8, 4)
    env.reset()
    with pytest.raises(t.ConcurrencyError, match="(?i)nothing"):
        env.recv()
    env.send([0, 1], zeros([0, 1]))
    with pytest.raises(t.ConcurrencyError):
        env.recv()
    env.send([2, 3], zeros([2, 3]))
    assert sorted(env.recv().ids.tolist()) == [0, 1, 2, 3]


def test_async_step_is_wrong_mode_and_sync_calls_refuse():
    env = asy()
    env.reset()
    for call in (lambda: env.step(zeros(range(8))), lambda: env.fork(0, [1])):
        with pytest.raises(t.ArgumentError, match="not available in async mode"):
            call()


def test_async_gathered_view_goes_stale_at_the_next_recv_only():
    env = asy(8, 4)
    first_all = env.reset()
    assert first_all.step.rewards.shape == (8, 1)
    env.send([0, 1, 2, 3], zeros(range(4)))
    assert not t.is_stale(first_all.step.rewards), "a send does not end the gathered window"
    first = env.recv()
    assert t.is_stale(first_all.step.rewards)
    assert first.step.rewards.shape == (4, 1)
    env.send(first.ids.tolist(), zeros(first.ids))
    assert not t.is_stale(first.step.rewards), "still the gathered window"
    second = env.recv()
    assert t.is_stale(first.step.rewards)
    assert not t.is_stale(second.step.rewards)


def test_async_indexed_view_goes_stale_at_a_send_of_its_ids_only():
    env = asy(8, 4, gather="indexed")
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    batch = env.recv()
    rewards = batch.step.rewards
    assert rewards.shape == (8, 1)
    # Other ids move: this batch's rows are untouched and the view is fresh.
    env.send([4, 5, 6, 7], zeros(range(4)))
    assert not t.is_stale(rewards)
    other = env.recv()
    assert not t.is_stale(rewards)
    env.send(batch.ids[:1].tolist(), zeros([0]))
    assert t.is_stale(rewards)
    assert not t.is_stale(other.step.rewards)


def test_async_indexed_refused_send_keeps_the_view_fresh():
    env = asy(8, 4, gather="indexed")
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    rewards = env.recv().step.rewards
    with pytest.raises(t.ActionError):
        env.send([0, 1], zeros([0]))
    assert not t.is_stale(rewards)


def test_async_owned_outputs_are_copies():
    env = asy(8, 4, outputs="owned")
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    first = env.recv()
    before = first.step.observations["obs"].copy()
    assert first.step.observations["obs"].flags.writeable
    env.send(first.ids.tolist(), zeros(first.ids))
    env.recv()
    assert np.array_equal(first.step.observations["obs"], before)
    assert t.is_stale(first.step.rewards) is False


def test_async_stored_tokens_never_block_ordinary_sends():
    """Batches overlap old tokens in every way: only part of a batch goes
    back, the rest is sent later, and nothing ever raises Borrowed."""
    env = asy(8, 3)
    env.reset()
    idle = list(range(8))
    for round_ in range(300):
        keep = round_ % 3
        go, idle = idle[keep:], idle[:keep]
        if go:
            env.send(go, zeros(go))
        idle += env.recv().ids.tolist()
    env.close()


def test_async_reset_ids_are_delivered_by_recv():
    env = asy(4, 2)
    env.reset()
    env.reset(ids=[1, 3])
    got = env.recv()
    assert sorted(got.ids.tolist()) == [1, 3]
    assert not env.is_poisoned


def test_async_failed_writer_leaves_a_pending_delivery_repeatable(monkeypatch):
    env = asy(8, 2, outputs="owned")
    env.reset()
    env.send([0, 1], zeros([0, 1]))
    real = env._batch
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    # Writers that fail deterministically, after the binding released its
    # stored tokens: a wrong seed count, and an id out of range.
    with pytest.raises(t.ArgumentError):
        env._native.reset([1])
    with pytest.raises(t.ConcurrencyError, match="(?i)in flight"):
        env.reset(ids=[5])
    # Ids of the pending batch stay refused, and it repeats.
    with pytest.raises(t.ConcurrencyError, match="(?i)in flight"):
        env.send([0], zeros([0]))
    got = env.recv()
    assert calls[0] == calls[1]
    assert sorted(got.ids.tolist()) == [0, 1]
    env.send([0, 1], zeros([0, 1]))
    assert sorted(env.recv().ids.tolist()) == [0, 1]


def test_async_signal_after_native_recv_keeps_the_batch(monkeypatch):
    env = asy(8, 4)
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    real = env._batch
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    assert sorted(env.recv().ids.tolist()) == [0, 1, 2, 3]
    with pytest.raises(t.ConcurrencyError, match="(?i)nothing"):
        env.recv()


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_async_interrupted_recv_keeps_the_work():
    class Stop(Exception):
        pass

    armed = [False]

    def handler(_signum, _frame):
        if armed[0]:
            armed[0] = False
            raise Stop

    n = 1 << 14
    env = asy(n, n, workers=1)
    env.reset()
    every = list(range(n))
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
        for _ in range(30):
            env.send(every, zeros(every))
            armed[0] = True
            try:
                got = env.recv()
            except Stop:
                interrupted = True
                armed[0] = False
                assert not env.is_poisoned
                got = env.recv()
            armed[0] = False
            assert len(got.ids) == n
            if interrupted:
                break
    finally:
        done.set()
        # Every tick has been sent before the handler goes back.
        ticker.join()
        signal.signal(signal.SIGUSR1, previous)
    assert interrupted
    env.close()


def test_async_two_threads_send_and_recv():
    n = 64
    env = asy(n, 8, workers=2)
    env.reset()
    errors = []
    sent = threading.Semaphore(0)
    returned = list(range(n))
    lock = threading.Lock()
    rounds = 200

    def sender():
        try:
            for _ in range(rounds):
                while True:
                    with lock:
                        ids, returned[:] = returned[:8], returned[8:]
                    if len(ids) == 8:
                        break
                    with lock:
                        returned[:0] = ids
                env.send(ids, zeros(ids))
                sent.release()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def receiver():
        try:
            for _ in range(rounds):
                sent.acquire()
                got = env.recv().ids.tolist()
                with lock:
                    returned.extend(got)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=f) for f in (sender, receiver)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(60)
    assert not errors
    env.close()


def test_async_two_concurrent_recv_one_is_refused():
    n = 1 << 14
    env = asy(n, n, workers=1)
    env.reset()
    every = list(range(n))
    refused = False
    for _ in range(200):
        env.send(every, zeros(every))
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
        assert sorted(outcomes) in (["received", "refused"], ["nothing", "received"])
        if "refused" in outcomes:
            refused = True
            break
    env.close()
    assert refused


@pytest.mark.parametrize("clear", ["reset", "restore_all", "close"])
def test_async_dead_delivery_thread_refuses_recv_until_cleared(clear):
    env = asy(8, 4)
    env.reset()
    taken = env.snapshot_all()
    env.send([0, 1, 2, 3], zeros(range(4)))
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
    env.send([0, 1, 2, 3], zeros(range(4)))
    assert len(env.recv().ids) == 4


def test_async_snapshot_and_restore_give_identical_futures():
    env = asy(8, 4, outputs="owned")
    env.reset()
    every = list(range(8))
    env.send(every, zeros(every))
    env.recv()
    env.recv()
    record = env.snapshot(0)
    taken = env.snapshot_all()

    def play():
        out = {}
        for _ in range(3):
            env.send(every, zeros(every))
            for _ in (0, 1):
                got = env.recv()
                for pos, i in enumerate(got.ids.tolist()):
                    out.setdefault(i, []).append(got.step.observations["obs"][pos].tobytes())
        return out

    first = play()
    env.restore_all(taken)
    assert play() == first
    env.restore([0], [record])
    env.reset(ids=[1, 2, 3])
    assert sorted(env.recv().ids.tolist()) == [0, 1, 2, 3]
    other = asy(8, 4)
    other.reset()
    with pytest.raises(t.ArgumentError, match="only the pool"):
        env.restore_all(other.snapshot_all())


def test_async_snapshot_of_an_idle_id_while_others_run_succeeds():
    # Issue #85: only the snapshotted id must be idle. The Async record is
    # opaque here; its byte identity with the quiescent one is tested in Rust.
    n = 1 << 14
    env = asy(n, n - 1, workers=1)
    env.reset()
    others = list(range(1, n))
    env.send(others, zeros(others))
    env.snapshot(0)
    with pytest.raises(t.ConcurrencyError, match="instance 1 is in flight"):
        env.snapshot(1)
    env.recv()
    env.snapshot(1)
    assert not env.is_poisoned


def test_async_close_with_work_in_flight_then_calls_refuse():
    env = asy()
    env.reset()
    env.send([0, 1], zeros([0, 1]))
    env.close()
    env.close()
    with pytest.raises(t.ArgumentError, match="closed"):
        env.recv()


def test_async_views_outlive_a_dropped_environment():
    env = asy(8, 8, gather="indexed")
    reference = asy(8, 8, outputs="owned")
    obs = env.reset().step.observations["obs"]
    reference.reset()
    every = list(range(8))
    env.send(every, zeros(every))
    # Dropping drains the step over these rows; the view then shows it.
    del env
    reference.send(every, zeros(every))
    got = reference.recv()
    order = np.argsort(got.ids)
    assert obs.tobytes() == got.step.observations["obs"][order].tobytes()


def test_async_gathered_view_outlives_a_dropped_environment():
    env = asy(8, 8)
    obs = env.reset().step.observations["obs"]
    before = obs.copy()
    del env
    assert np.array_equal(obs, before)


def test_async_refused_reset_leaves_views_fresh():
    for kw in ({}, {"gather": "indexed"}):
        env = asy(8, 4, **kw)
        env.reset()
        env.send([0, 1, 2, 3], zeros(range(4)))
        rewards = env.recv().step.rewards
        # A wrong seed count fails validation, after the tokens are released.
        with pytest.raises(t.ArgumentError):
            env._native.reset([1])
        assert not t.is_stale(rewards), kw
        # An id out of range fails too.
        with pytest.raises(t.ArgumentError):
            env.reset(ids=[99])
        assert not t.is_stale(rewards), kw
        env.close()


def test_async_successful_writer_makes_indexed_views_of_other_ids_stale():
    env = asy(8, 4, gather="indexed")
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    rewards = env.recv().step.rewards
    assert not t.is_stale(rewards)
    env.reset(ids=[6])
    assert t.is_stale(rewards), "the writer dropped this view's token"


def test_async_successful_writer_leaves_gathered_views_fresh():
    env = asy(8, 4)
    env.reset()
    env.send([0, 1, 2, 3], zeros(range(4)))
    rewards = env.recv().step.rewards
    env.reset(ids=[6])
    assert not t.is_stale(rewards), "a gathered copy ends only at the next recv"


def test_async_failed_restore_ids_leaves_pending_and_views_usable(monkeypatch):
    env = asy(8, 2, outputs="borrowed", gather="indexed")
    env.reset()
    record = env.snapshot(4)
    env.send([0, 1], zeros([0, 1]))
    rewards = env.recv().step.rewards
    env.send([2, 3], zeros([2, 3]))
    real = env._batch
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    # Any partial writer waits for the pending batch, and changes nothing.
    with pytest.raises(t.ConcurrencyError, match="(?i)in flight"):
        env.restore([4], [record])
    assert not t.is_stale(rewards)
    got = env.recv()
    assert calls[0] == calls[1]
    assert sorted(got.ids.tolist()) == [2, 3]
    # With nothing pending, an id count mismatch fails in the engine.
    with pytest.raises(t.ArgumentError):
        env.restore([4, 5], [record])
    assert not t.is_stale(rewards)
    env.send([2, 3], zeros([2, 3]))
    assert sorted(env.recv().ids.tolist()) == [2, 3]


@pytest.mark.parametrize("gather", ["gathered", "indexed"])
def test_async_view_exports_its_rows_through_dlpack(gather):
    env = asy(8, 4, gather=gather)
    env.reset()
    env.send([4, 5, 6, 7], zeros(range(4)))
    out = env._native.recv(env._delivered)
    env._delivered = out["seq"]
    view = out["outputs"]["observations"][0]
    arr = np.from_dlpack(view)
    assert arr.shape == ((4, 1, 4) if gather == "gathered" else (8, 1, 4))
    assert not arr.flags.writeable
    assert arr.__array_interface__["data"][0] == np.asarray(view).__array_interface__["data"][0]
    assert arr.tobytes() == np.asarray(view).tobytes()
    env.close()


@pytest.mark.parametrize("workers", [1, 2, 7])
@pytest.mark.parametrize(("env_id", "batch"), [("tessera/CartPole-v1", 3), ("tessera/Tag-v1", 2)])
def test_async_indexed_rows_are_bit_identical_to_sync_in_every_field(workers, env_id, batch):
    n = 6
    common = dict(num_envs=n, seed=7, auto_reset="same_step", max_episode_steps=5)
    sync = t.make(env_id, outputs="owned", **common)
    sync.reset()
    env = t.make(env_id, mode="async", batch_size=batch, gather="indexed", workers=workers, **common)
    env.reset()
    agents = sync.agents
    every = list(range(n))
    ended = 0
    rng = np.random.default_rng(5)
    high = 2 if env_id.endswith("CartPole-v1") else 1
    for _ in range(12):
        acts = rng.integers(0, high, size=n * agents, dtype=np.int32)
        want = sync._native.step(acts.tobytes())
        ended += sum(want["episode_terminated"]) + sum(want["episode_truncated"])
        # Every id once per round, so Sync's round is comparable; rows are
        # read per batch, from the batch's own ids, as the view contract says.
        env._native.send(every, acts.tobytes(), env._delivered)
        seen = []
        for _ in range(n // batch):
            got = env._native.recv(env._delivered)
            env._delivered = got["seq"]
            out = got["outputs"]
            for i in got["ids"]:
                seen.append(i)
                for key in (*ROW_KEYS, *EPISODE_KEYS):
                    assert rows_of(np.asarray(out[key]).tobytes(), n)[i] == rows_of(want[key], n)[i], key
                for key in ("observations", "terminal_observations"):
                    for leaf, whole in zip(out[key], want[key], strict=True):
                        assert rows_of(np.asarray(leaf).tobytes(), n)[i] == rows_of(whole, n)[i], key
        assert sorted(seen) == every
    assert ended
    env.close()


def test_async_successful_restore_ids_makes_indexed_views_stale():
    env = asy(8, 4, gather="indexed")
    env.reset()
    record = env.snapshot(6)
    env.send([0, 1, 2, 3], zeros(range(4)))
    rewards = env.recv().step.rewards
    assert not t.is_stale(rewards)
    env.restore([6], [record])
    assert t.is_stale(rewards), "the writer dropped this view's token"


def test_async_successful_restore_ids_leaves_gathered_views_fresh():
    env = asy(8, 4)
    env.reset()
    record = env.snapshot(6)
    env.send([0, 1, 2, 3], zeros(range(4)))
    rewards = env.recv().step.rewards
    env.restore([6], [record])
    assert not t.is_stale(rewards)


def test_async_interrupted_recv_refuses_a_disjoint_writer_and_replays_fresh(monkeypatch):
    env = asy(8, 2, gather="indexed")
    env.reset()
    env.send([0, 1], zeros([0, 1]))
    real = env._batch
    calls = []

    def interrupted(out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real(out)

    monkeypatch.setattr(env, "_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        env.recv()
    with pytest.raises(t.ConcurrencyError, match="(?i)in flight"):
        env.reset(ids=[5])
    got = env.recv()
    assert sorted(got.ids.tolist()) == [0, 1]
    assert not t.is_stale(got.step.rewards), "the replay is usable"
    env.reset(ids=[5])
    assert t.is_stale(got.step.rewards)


def test_async_batch_in_non_id_order_gathers_each_ids_own_rows():
    n = 8
    env = asy(n, n, outputs="owned", seed=9)
    env.reset()
    env.send(list(reversed(range(n))), zeros(range(n)))
    got = env.recv()
    # Workers claim in send order, so one worker delivers ids reversed.
    assert got.ids.tolist() == list(reversed(range(n)))
    sync = t.make("tessera/CartPole-v1", num_envs=n, seed=9, outputs="owned")
    sync.reset()
    want = sync.step(zeros(range(n))).observations["obs"]
    for pos, i in enumerate(got.ids.tolist()):
        assert got.step.observations["obs"][pos].tobytes() == want[i].tobytes()
    assert len({want[i].tobytes() for i in range(n)}) == n, "rows differ, so order matters"

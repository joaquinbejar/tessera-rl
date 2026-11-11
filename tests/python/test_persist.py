# SPDX-License-Identifier: MIT OR Apache-2.0
"""Persisted files through ``Env.encode`` and ``Env.load`` (ADR-0011), in every mode."""

import sys
import threading
import time

import numpy as np
import pytest

import tessera_rl as t

N = 8


def make(mode="sync", env="tessera/Tag-v1", **kwargs):
    if mode == "pipelined":
        kwargs.update(mode="pipelined", workers=2, chunk_size=2)
    elif mode == "async":
        kwargs.update(mode="async", workers=2, batch_size=N)
    return t.make(env, num_envs=N, seed=5, outputs="owned", **kwargs)


def sync_file(env_id="tessera/Tag-v1", steps=3):
    env = make(env=env_id)
    env.reset()
    for _ in range(steps):
        env.step(np.zeros(N * env.agents, dtype=np.int32))
    data = env.encode()
    env.close()
    return data


@pytest.mark.parametrize("env_id", ["tessera/Tag-v1", "tessera/Connect4-v1", "tessera/CartPole-v1"])
def test_persist_round_trip_in_sync_gives_the_same_file(env_id):
    data = sync_file(env_id)
    assert type(data) is bytes
    env = make(env=env_id)
    env.reset()
    env.load(data)
    assert env.encode() == data
    env.close()


@pytest.mark.parametrize("env_id", ["tessera/Tag-v1", "tessera/Connect4-v1"])
@pytest.mark.parametrize("mode", ["pipelined", "async"])
def test_persist_a_sync_file_loads_in_split_modes_and_back(mode, env_id):
    data = sync_file(env_id)
    env = make(mode, env=env_id)
    env.reset()
    out = env.load(data)
    if mode == "pipelined":
        assert len(out) == 2
    else:
        assert out is None
    assert env.encode() == data
    env.close()


def test_persist_partial_ids_are_sync_only():
    data = sync_file()
    env = make()
    env.reset()
    part = env.encode([3, 1])
    other = make()
    other.reset()
    other.load(part, ids=[0, 2])
    assert other.encode([0, 2]) == env.encode([3, 1])
    with pytest.raises(t.TesseraError):
        other.load(part, ids=[0])
    with pytest.raises(t.TesseraError):
        other.load(part)
    env.close()
    other.close()
    for mode in ("pipelined", "async"):
        split = make(mode)
        split.reset()
        with pytest.raises(t.ArgumentError):
            split.load(data, ids=[0])
        split.close()


@pytest.mark.parametrize("kind", [bytearray, memoryview])
def test_persist_load_takes_exactly_bytes(kind):
    data = sync_file()
    env = make()
    env.reset()
    with pytest.raises(TypeError):
        env.load(kind(data))
    env.load(bytes(kind(data)))
    env.close()


def damaged(data):
    bad = bytearray(data)
    bad[-20] ^= 1
    return bytes(bad)


@pytest.mark.parametrize("mode", ["sync", "pipelined", "async"])
def test_persist_a_damaged_file_raises_and_changes_nothing(mode):
    data = sync_file()
    env = make(mode)
    env.reset()
    before = env.encode()
    with pytest.raises(t.PersistError):
        env.load(damaged(data))
    with pytest.raises(t.PersistError):
        env.load(b"not a file")
    assert issubclass(t.PersistError, t.TesseraError)
    assert env.encode() == before
    env.close()


def test_persist_limits_are_passed_through():
    data = sync_file()
    env = make()
    env.reset()
    with pytest.raises(t.PersistError):
        env.load(data, max_instances=1)
    with pytest.raises(t.PersistError):
        env.load(data, max_decoded_bytes=8)
    env.load(data, max_instances=N, max_decoded_bytes=1 << 30)
    env.close()


def test_persist_another_config_is_refused():
    data = sync_file()
    env = t.make("tessera/Tag-v1", num_envs=N, seed=5, outputs="owned", width=32)
    env.reset()
    with pytest.raises(t.PersistError):
        env.load(data)
    env.close()


# A switch interval far longer than any measured region: the helper thread only gets
# the interpreter while the measured callable lets go of it (or the region blocks).
SWITCH_INTERVAL_S = 5.0
# Ticks the helper must make; more than one rules out a single stray switch.
REQUIRED_TICKS = 3


def ticks_while_calling(call, max_calls, budget_s):
    """Call ``call()`` until a helper thread has ticked ``REQUIRED_TICKS`` times, or
    until ``max_calls`` calls or ``budget_s`` seconds have run, whichever comes first.

    Returns ``(calls, ticks)``. The helper ticks in a ``time.sleep`` loop, so it can
    tick only when the main thread releases the interpreter. Stopping on ticks
    rather than after a fixed count keeps the result independent of how coarse
    ``time.sleep`` is on the platform (issue #179).
    ``budget_s`` must stay well below ``SWITCH_INTERVAL_S``.
    """
    assert budget_s <= SWITCH_INTERVAL_S / 2
    ticks = [0]
    stop = threading.Event()
    started = threading.Event()

    def other():
        started.set()
        while not stop.is_set():
            ticks[0] += 1
            time.sleep(0.0001)

    previous = sys.getswitchinterval()
    thread = threading.Thread(target=other)
    thread.start()
    try:
        started.wait()
        sys.setswitchinterval(SWITCH_INTERVAL_S)
        # A blocking sleep hands the interpreter over once, so that any wait the
        # helper began under the previous (short) interval is over before measuring.
        time.sleep(0.01)
        before = ticks[0]
        calls = 0
        deadline = time.monotonic() + budget_s
        while calls < max_calls and time.monotonic() < deadline:
            call()
            calls += 1
            if ticks[0] - before >= REQUIRED_TICKS:
                break
        moved = ticks[0] - before
    finally:
        sys.setswitchinterval(previous)
        stop.set()
        thread.join()
    return calls, moved


def test_persist_load_releases_the_interpreter():
    data = sync_file()
    env = make()
    env.reset()
    # The budget stays well below SWITCH_INTERVAL_S: past the interval the
    # interpreter switches even if the call holds it, and the test would lose its
    # teeth. On a native Windows host 2000 loads took about 8 ms with at least 10
    # ticks, so 2 s is a margin of hundreds of times.
    calls, moved = ticks_while_calling(lambda: env.load(data), max_calls=1_000_000, budget_s=2.0)
    env.close()
    assert moved >= REQUIRED_TICKS, f"{calls} loads ran and the other thread ticked {moved} times"


def test_persist_ticks_while_calling_python_holding_the_interpreter_sees_no_ticks():
    # Negative control for the method above: pure Python never releases the
    # interpreter before the switch interval, so the helper cannot tick. The region
    # (about 50 ms) is two orders of magnitude below the 5 s interval, so an
    # involuntary switch cannot happen within it.
    def busy():
        end = time.perf_counter() + 0.0005
        while time.perf_counter() < end:
            pass

    calls, moved = ticks_while_calling(busy, max_calls=1_000_000, budget_s=0.05)
    assert calls > 1
    assert moved == 0, f"{calls} calls ran and the other thread ticked {moved} times"


def test_persist_no_identity_override_reaches_python():
    names = dir(t) + dir(t.Env) + dir(t._native)
    assert not [n for n in names if "encode_with" in n or "decode_with" in n or "build_id" in n]


@pytest.mark.parametrize("mode", ["pipelined", "async"])
def test_persist_a_refused_file_leaves_views_fresh_and_a_good_one_stales_them(mode):
    data = sync_file()
    env = t.make(
        "tessera/Tag-v1",
        num_envs=N,
        seed=5,
        **(dict(mode="pipelined", workers=2, chunk_size=2) if mode == "pipelined" else dict(mode="async", workers=2, batch_size=N)),
    )
    out = env.reset()
    rewards = out[0].step.rewards if mode == "pipelined" else out.step.rewards
    with pytest.raises(t.PersistError):
        env.load(damaged(data))
    assert not t.is_stale(rewards)
    env.load(data)
    if mode == "pipelined":
        assert t.is_stale(rewards)
    # Async: a gathered view is its own window, stale at the next recv, as after restore_all.
    env.close()

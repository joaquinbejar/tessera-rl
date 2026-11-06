# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pipelined through the PufferLib adapter. Run through ``make puffer-test``."""

import signal
import subprocess
import sys

import numpy as np
import pufferlib.vector
import pytest
from test_adapter import CARTPOLE, TAG, assert_same, leaf, scenario

import tessera_rl as t
from tessera_rl.puffer import MAX_WORKERS, TesseraPufferPipelinedEnv, make_puffer, mix_seed


def env_of(env_id=CARTPOLE, n=8, chunk=2, **kw):
    return pufferlib.vector.make(make_puffer(env_id, num_envs=n, chunk_size=chunk, mode="pipelined", **kw))


@pytest.mark.parametrize(("env_id", "n", "chunk"), [(CARTPOLE, 8, 2), (TAG, 8, 4)])
@pytest.mark.parametrize("workers", [1, 2])
def test_half_local_identity_against_a_pipelined_reference(env_id, n, chunk, workers):
    env = env_of(env_id, n, chunk, workers=workers, log_interval=None)
    env.reset(seed=3)
    base = mix_seed(0, 3)
    ref = t.make(env_id, num_envs=n, seed=base, auto_reset="same_step", mode="pipelined", chunk_size=chunk,
                 workers=workers)  # fmt: skip
    want = ref.reset(base)
    rng = np.random.default_rng(2)
    agents = env.num_agents // n
    for h in want:
        got = env.recv()
        assert np.asarray(got[0]).tobytes() == leaf(h.step).tobytes()
        assert list(got[5]) == list(range(h.ids.start * agents, h.ids.stop * agents))
        acts = rng.integers(0, env.single_action_space.n, env.agents_per_batch, dtype=np.int32)
        env.send(acts)
        ref.send(h.half, acts)
    for _ in range(300):
        got = env.recv()
        h = ref.recv()
        assert_same(h.step, got)
        assert list(got[5]) == list(range(h.ids.start * agents, h.ids.stop * agents))
        acts = rng.integers(0, env.single_action_space.n, env.agents_per_batch, dtype=np.int32)
        env.send(acts)
        ref.send(h.half, acts)
    env.recv()
    ref.recv()
    env.close()
    ref.close()


def test_no_array_spans_the_running_half_under_load():
    hogs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(6)]
    try:
        env = env_of(n=1 << 14, chunk=64, workers=2, log_interval=None)
        env.reset(seed=1)
        half = env.agents_per_batch
        assert half == env.num_agents // 2
        first = env.recv()
        assert first[0].shape[0] == half
        env.send(np.zeros(half, dtype=np.int32))
        assert env.observations is None
        for _ in range(200):
            o, r, d, tr, _, ids, m = env.recv()
            assert all(a.shape[0] == half for a in (o, r, d, tr, ids, m))
            assert not any(a.flags.writeable for a in (o, r, d, tr))
            before = o.tobytes(), r.tobytes(), d.tobytes(), tr.tobytes()
            # The other half is running while these rows stay put.
            for _ in range(20):
                assert (o.tobytes(), r.tobytes(), d.tobytes(), tr.tobytes()) == before
            env.send(np.ones(half, dtype=np.int32))
            assert env.observations is None and env.rewards is None
        env.recv()
        env.close()
    finally:
        for hog in hogs:
            hog.send_signal(signal.SIGKILL)
            hog.wait()


def test_interrupted_recv_repeats_through_the_wrapper(monkeypatch):
    env = env_of(log_interval=None)
    env.reset(seed=1)
    for _ in range(2):
        env.recv()
        env.send(np.zeros(env.agents_per_batch, dtype=np.int32))
    native = env._env._native

    class Stop(Exception):
        pass

    real = type(env._env)._half
    calls = []

    def once(self, out):
        calls.append(out["seq"])
        if len(calls) == 1:
            raise Stop
        return real(self, out)

    monkeypatch.setattr(type(env._env), "_half", once)
    with pytest.raises(Stop):
        env.recv()
    assert env._current is None
    assert env.observations is None
    got = env.recv()
    assert calls[0] == calls[1], "the same delivery repeats"
    assert env._current == 0
    assert got[5][0] == 0
    env.send(np.zeros(env.agents_per_batch, dtype=np.int32))
    env.recv()
    assert native is env._env._native
    env.close()


def test_send_pairs_with_the_half_last_received():
    env = env_of(log_interval=None)
    acts = np.zeros(env.agents_per_batch, dtype=np.int32)
    with pytest.raises(t.ArgumentError, match="before the first reset"):
        env.send(acts)
    with pytest.raises(t.ArgumentError, match="before the first reset"):
        env.recv()
    env.reset(seed=1)
    with pytest.raises(t.ArgumentError, match="recv first"):
        env.send(acts)
    assert env.recv()[5][0] == 0
    with pytest.raises(t.ArgumentError, match="before half 0 was sent"):
        env.recv()
    env.send(acts)
    with pytest.raises(t.ArgumentError, match="recv first"):
        env.send(acts)
    assert env.recv()[5][0] == env.agents_per_batch
    with pytest.raises(t.ActionError):
        env.send(acts[:-1])
    assert env._current == 1, "a refused send keeps its half"
    env.send(acts)
    assert env.recv()[5][0] == 0
    env.close()


def test_step_and_terminal_observations_are_refused():
    env = env_of()
    env.reset(seed=1)
    with pytest.raises(t.ArgumentError, match="no step"):
        env.step(np.zeros(env.agents_per_batch, dtype=np.int32))
    with pytest.raises(t.ArgumentError, match="not available"):
        _ = env.terminal_observations
    env.close()


def test_unequal_halves_are_refused_naming_the_sizes():
    with pytest.raises(t.ArgumentError, match=r"these have \d+ and \d+ instances"):
        env_of(n=66, chunk=4)


def test_a_chunk_that_cannot_split_raises_the_engines_config_error():
    with pytest.raises(t.ConfigError, match="cannot be split"):
        env_of(n=64, chunk=64)


@pytest.mark.parametrize("bad", [-1, "x", MAX_WORKERS + 1])
def test_invalid_workers_are_refused(bad):
    with pytest.raises(t.ArgumentError, match="workers"):
        TesseraPufferPipelinedEnv(CARTPOLE, num_envs=8, chunk_size=2, workers=bad)


def test_buffers_are_refused():
    with pytest.raises(t.ArgumentError, match="native backend"):
        TesseraPufferPipelinedEnv(CARTPOLE, num_envs=8, chunk_size=2, buf={})


@pytest.mark.parametrize("backend", ["Serial", "Multiprocessing"])
@pytest.mark.parametrize("start", ["spawn", "fork"] if sys.platform == "linux" else ["spawn"])
def test_vector_backends_refuse_in_the_parent(backend, start):
    assert scenario("pipelined_refused", backend, start, timeout=60) == {"refused": True}


def test_logging_only_counts_real_steps():
    env = env_of(n=8, chunk=2, log_interval=4, max_episode_steps=2)
    env.reset(seed=1)
    logs = []
    for _ in range(40):
        logs += env.recv()[4]
        env.send(np.zeros(env.agents_per_batch, dtype=np.int32))
    env.recv()
    assert logs
    assert all(1 <= log["episode_length"] <= 2 and log["n"] > 0 for log in logs)
    env.close()


@pytest.mark.parametrize("env_id", [CARTPOLE, TAG])
@pytest.mark.parametrize("rnn", ["rnn", "plain"])
def test_smoke_training_and_logging_through_the_trainer(env_id, rnn):
    out = scenario("smoke_pipelined", env_id, rnn, timeout=600)
    assert out["finite"]
    assert out["global_step"] >= 8192
    assert out["logged"] > 0, "no log reached the trainer"
    assert out["bounded"]


@pytest.mark.skipif(not hasattr(__import__("os"), "fork"), reason="needs fork")
@pytest.mark.parametrize("workers", [1, 2])
def test_fork_guard_with_both_halves_in_flight(workers):
    assert scenario("pipelined_fork_guard", workers, timeout=120) == {"child": 0}

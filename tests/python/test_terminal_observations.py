# SPDX-License-Identifier: MIT OR Apache-2.0
"""``Env.terminal_observations`` (#545): the final observations of the last
whole-pool step, before their same-step reset."""

import numpy as np
import pytest

import tessera_rl as t

CARTPOLE = "tessera/CartPole-v1"
GATHER = "tessera/Gather-v1"
LIMIT = 3


def pair(env_id, outputs, **kw):
    """A same_step pool and a twin that does not reset, at the same seeds."""
    env = t.make(env_id, num_envs=4, seed=6, outputs=outputs, auto_reset="same_step", max_episode_steps=LIMIT, **kw)
    twin = t.make(env_id, num_envs=4, seed=6, outputs="owned", auto_reset="disabled", max_episode_steps=LIMIT, **kw)
    return env, twin


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
@pytest.mark.parametrize("env_id", [CARTPOLE, GATHER])
def test_terminal_observations_are_the_observation_before_the_reset(env_id, outputs):
    env, twin = pair(env_id, outputs)
    try:
        s = env.reset()
        twin.reset()
        actions = np.zeros((4, 1), dtype=np.int32)
        for step in range(1, LIMIT + 1):
            s = env.step(actions)
            w = twin.step(actions)
            final = env.terminal_observations
            assert list(final) == list(s.observations)
            for name, rows in final.items():
                assert rows.shape == s.observations[name].shape and rows.dtype == s.observations[name].dtype
                if step < LIMIT:
                    # Nothing ended: the engine's rows are zero.
                    assert not np.asarray(rows).any(), (name, step)
                else:
                    # Every instance hit the step limit: the observation the
                    # twin, which does not reset, sees now.
                    assert (s.episode_truncated | s.episode_terminated).all()
                    np.testing.assert_array_equal(rows, w.observations[name])
    finally:
        env.close()
        twin.close()


def test_borrowed_terminal_observations_have_the_borrow_window():
    env, twin = pair(CARTPOLE, "borrowed")
    twin.close()
    try:
        env.reset()
        env.step(np.zeros((4, 1), dtype=np.int32))
        rows = env.terminal_observations["obs"]
        assert not rows.flags.writeable
        assert not t.is_stale(rows)
        env.step(np.zeros((4, 1), dtype=np.int32))
        assert t.is_stale(rows)
    finally:
        env.close()


def test_owned_terminal_observations_are_independent_copies():
    env, twin = pair(CARTPOLE, "owned")
    twin.close()
    try:
        env.reset()
        for _ in range(LIMIT):
            env.step(np.zeros((4, 1), dtype=np.int32))
        a = env.terminal_observations["obs"]
        b = env.terminal_observations["obs"]
        assert a is not b and a.flags.writeable
        before = b.copy()
        a[:] = 42.0
        np.testing.assert_array_equal(b, before)
        np.testing.assert_array_equal(env.terminal_observations["obs"], before)
    finally:
        env.close()


@pytest.mark.parametrize("auto_reset", ["next_step", "disabled"])
def test_terminal_observations_refused_without_same_step(auto_reset):
    env = t.make(CARTPOLE, num_envs=2, seed=1, auto_reset=auto_reset)
    try:
        env.reset()
        env.step(np.zeros((2, 1), dtype=np.int32))
        with pytest.raises(t.ArgumentError, match="same_step"):
            env.terminal_observations
    finally:
        env.close()


def test_terminal_observations_refused_before_the_first_call():
    env = t.make(CARTPOLE, num_envs=2, seed=1, auto_reset="same_step")
    try:
        with pytest.raises(t.ArgumentError, match="Sync mode first"):
            env.terminal_observations
    finally:
        env.close()


def test_terminal_observations_refused_after_a_pipelined_half():
    env = t.make(CARTPOLE, num_envs=4, seed=1, auto_reset="same_step", mode="pipelined", chunk_size=2, workers=1)
    try:
        halves = env.reset()
        with pytest.raises(t.ArgumentError, match="Sync mode"):
            env.terminal_observations
        env.send(halves[0].half, np.zeros((len(halves[0].ids), 1), dtype=np.int32))
        env.recv()
        with pytest.raises(t.ArgumentError, match="Sync mode"):
            env.terminal_observations
    finally:
        env.close()

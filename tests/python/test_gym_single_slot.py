# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Gymnasium adapter on two-slot environments built with one agent
(Snake-v1, PaddleDuel-v1): slot 0 of each native row, against the native
env bit for bit, with the unused slot never alive or acting."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import tessera_rl as t
import tessera_rl.gym as tg

SNAKE, PADDLE = "tessera/Snake-v1", "tessera/PaddleDuel-v1"
CASES = [(SNAKE, "pixels"), (SNAKE, "state"), (PADDLE, "pixels"), (PADDLE, "state")]
CHOICES = {SNAKE: 4, PADDLE: 3}
NATIVE_MODE = {AutoresetMode.NEXT_STEP: "next_step", AutoresetMode.SAME_STEP: "same_step", AutoresetMode.DISABLED: "disabled"}


def _leaf(obs):
    return obs["obs"] if isinstance(obs, dict) else obs


def _actions(env_id, steps, n, seed=11):
    rng = np.random.default_rng(seed)
    return rng.integers(0, CHOICES[env_id], size=(steps, n))


def _native(env_id, obs, n, seed, mode, outputs="owned", limit=0):
    return t.make(
        env_id, num_envs=n, seed=seed, outputs=outputs, agents=1, obs=obs, auto_reset=NATIVE_MODE[mode], max_episode_steps=limit
    )


@pytest.mark.parametrize(("env_id", "obs"), CASES)
def test_gym_single_slot_env_checker_passes(env_id, obs):
    check_env(gymnasium.make(env_id, obs=obs).unwrapped, skip_render_check=True)
    check_env(tg.TesseraEnv(env_id, obs=obs), skip_render_check=True)


@pytest.mark.parametrize(("env_id", "obs"), CASES)
def test_gym_single_slot_spaces_match_the_native_slot_zero(env_id, obs):
    env = tg.make(env_id, num_envs=2, seed=3, obs=obs)
    native = _native(env_id, obs, 2, 3, AutoresetMode.NEXT_STEP)
    first = _leaf(native.reset().observations)
    assert env.single_observation_space.shape == first.shape[2:]
    assert env.single_observation_space.dtype == first.dtype
    assert env.single_action_space == gymnasium.spaces.Discrete(CHOICES[env_id])
    env.close()
    native.close()


def _run(env_id, obs, mode, outputs, steps=300, n=3, seed=7, limit=40):
    """Steps the gym env and the native env side by side; checks slot 0
    of every native output and that slot 1 never lives or acts."""
    env = tg.make(env_id, num_envs=n, seed=seed, obs=obs, autoreset_mode=mode, outputs=outputs, max_episode_steps=limit)
    native = _native(env_id, obs, n, seed, mode, limit=limit)
    g_obs, _ = env.reset()
    s = native.reset()
    assert np.array_equal(np.asarray(g_obs), _leaf(s.observations)[:, 0])
    assert not s.alive[:, 1].any() and not s.acting[:, 1].any()
    ended = 0
    for a in _actions(env_id, steps, n):
        g_obs, rew, term, trunc, info = env.step(a)
        s = native.step(np.stack([a, np.zeros(n, dtype=a.dtype)], axis=1))
        assert np.array_equal(np.asarray(g_obs), _leaf(s.observations)[:, 0])
        assert np.array_equal(np.asarray(rew), s.rewards[:, 0])
        assert np.array_equal(np.asarray(term), s.episode_terminated)
        assert np.array_equal(np.asarray(trunc), s.episode_truncated)
        assert not s.alive[:, 1].any() and not s.acting[:, 1].any()
        done = np.asarray(term) | np.asarray(trunc)
        if mode is AutoresetMode.SAME_STEP and done.any():
            assert np.array_equal(info["_final_obs"], done)
            for i in np.flatnonzero(done):
                # The native same-step row is already reset; its terminal
                # observation is the gym final_obs.
                assert info["final_obs"][i].shape == env.single_observation_space.shape
        if mode is AutoresetMode.DISABLED and done.any():
            mask = done.copy()
            g_obs, _ = env.reset(options={"reset_mask": mask})
            out = native._native.reset(None, [int(i) for i in np.flatnonzero(mask)])
            dtype, shape = env._shapes[0]
            rows = np.frombuffer(out["observations"][0], dtype=dtype).reshape((n, 2, *shape))
            assert np.array_equal(_leaf(g_obs), rows[:, 0])
            assert not np.frombuffer(out["alive"], dtype=np.bool_).reshape(n, 2)[:, 1].any()
        ended += int(done.sum())
    assert ended > 0, "no episode ended; the run checks no reset"
    env.close()
    native.close()


@pytest.mark.parametrize(("env_id", "obs"), CASES)
@pytest.mark.parametrize("mode", list(AutoresetMode))
@pytest.mark.parametrize("outputs", ["owned", "borrowed"])
def test_gym_single_slot_matches_the_native_env(env_id, obs, mode, outputs):
    _run(env_id, obs, mode, outputs)


@pytest.mark.parametrize(("env_id", "obs"), CASES)
def test_gym_single_slot_same_step_final_obs_is_slot_zero(env_id, obs):
    n, seed, limit = 2, 9, 25
    env = tg.make(env_id, num_envs=n, seed=seed, obs=obs, autoreset_mode="same_step", max_episode_steps=limit)
    native = t.make(env_id, num_envs=n, seed=seed, outputs="owned", agents=1, obs=obs, auto_reset="same_step", max_episode_steps=limit)
    env.reset()
    native.reset()
    raw = native._native  # the terminal observations are not in Step
    seen = False
    for a in _actions(env_id, 200, n):
        _, _, term, trunc, info = env.step(a)
        out = raw.step(np.stack([a, np.zeros(n, dtype=a.dtype)], axis=1).astype("<i4").tobytes())
        done = term | trunc
        if done.any():
            dtype, shape = env._shapes[0]
            final = np.frombuffer(out["terminal_observations"][0], dtype=dtype).reshape((n, 2, *shape))
            for i in np.flatnonzero(done):
                assert np.array_equal(_leaf(info["final_obs"][i]), final[i, 0])
            seen = True
    assert seen
    env.close()
    native.close()


@pytest.mark.parametrize(("env_id", "obs"), CASES)
def test_gym_single_slot_owned_outputs_do_not_alias(env_id, obs):
    env = tg.make(env_id, num_envs=2, seed=4, obs=obs, autoreset_mode="same_step", max_episode_steps=5)
    env.reset()
    a = np.zeros(2, dtype=np.int64)
    o1, r1, _, _, info = env.step(a)
    finals = [info["final_obs"][i] for i in np.flatnonzero(info.get("_final_obs", np.zeros(2, bool)))]
    keep = (np.array(o1), np.array(r1), [np.array(f) for f in finals])
    for _ in range(6):
        o2, r2, _, _, _ = env.step(a + 1)
        assert not np.shares_memory(o1, o2) and not np.shares_memory(r1, r2)
    assert o1.base is None and r1.base is None
    assert np.array_equal(o1, keep[0]) and np.array_equal(r1, keep[1])
    for f, k in zip(finals, keep[2], strict=True):
        assert np.array_equal(f, k)
    env.close()


@pytest.mark.parametrize("env_id", [SNAKE, PADDLE])
def test_gym_single_slot_rejects_two_agents(env_id):
    with pytest.raises(t.ArgumentError, match="PettingZoo"):
        tg.make(env_id, agents=2)
    with pytest.raises(t.ArgumentError, match="PettingZoo"):
        gymnasium.make_vec(env_id, num_envs=2, agents=2)


def test_gym_single_slot_only_allowlisted_ids_are_admitted(monkeypatch):
    # KeyDoor has four slots and accepts one agent, but it is not in the
    # slot-0 allowlist: one agent at the start is not admission.
    monkeypatch.setitem(tg._MAKERS, "tessera/GridKeyDoor-v1", t._native.make_keydoor)
    with pytest.raises(t.ArgumentError, match="PettingZoo"):
        tg.make("tessera/GridKeyDoor-v1", agents=1)


@pytest.mark.parametrize(("env_id", "limit"), [(SNAKE, 1000), (PADDLE, 5000)])
def test_gym_single_slot_time_limits_follow_the_registration(env_id, limit):
    assert gymnasium.spec(env_id).max_episode_steps == limit
    assert gymnasium.spec(env_id).kwargs["agents"] == 1


def test_gym_snake_reverse_action_is_a_noop():
    nexts = []
    for a in range(4):
        env = tg.make(SNAKE, num_envs=1, seed=21, obs="state")
        env.reset()
        o, *_ = env.step(np.array([a]))
        native = _native(SNAKE, "state", 1, 21, AutoresetMode.NEXT_STEP)
        native.reset()
        s = native.step(np.array([[a, 0]]))
        assert np.array_equal(o, _leaf(s.observations)[:, 0])
        nexts.append(o.tobytes())
        env.close()
        native.close()
    # Four moves from the spawn: the reverse is the heading kept, so exactly
    # two actions give the same next observation.
    assert len(set(nexts)) == 3


@pytest.mark.parametrize(("env_id", "agents"), [(SNAKE, 1), (SNAKE, 2), (PADDLE, 1), (PADDLE, 2)])
def test_layout_initial_agents_is_the_configured_population(env_id, agents):
    env = t.make(env_id, num_envs=1, seed=1, agents=agents)
    layout = env._native.layout()
    assert layout["agents"] == 2 and layout["initial_agents"] == agents
    env.close()


@pytest.mark.parametrize("env_id", ["tessera/CartPole-v1", "tessera/BrickWall-v1"])
def test_layout_initial_agents_equals_capacity_for_fixed_populations(env_id):
    env = t.make(env_id, num_envs=1, seed=1)
    layout = env._native.layout()
    assert layout["initial_agents"] == layout["agents"]
    env.close()

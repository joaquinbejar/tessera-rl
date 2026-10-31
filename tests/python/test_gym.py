# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Gymnasium adapter (SPEC 15): registration, the env checker, the
vector API in each autoreset mode, owned outputs by default, seeding and
compatibility with Gymnasium's vector wrappers."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from gymnasium.vector import AutoresetMode

import tessera_rl as t
import tessera_rl.gym as tg

ID = "tessera/CartPole-v1"


def run_until_done(env, actions=None, limit=600):
    """Steps until some instance ends; returns the step outputs and the
    outputs of the step before it."""
    n = env.num_envs
    prev = env.reset(seed=3)
    for _ in range(limit):
        out = env.step(np.ones(n, dtype=np.int64) if actions is None else actions)
        if (out[2] | out[3]).any():
            return prev, out
        prev = out
    raise AssertionError("no episode ended")


def test_gym_env_checker_passes_on_make_and_on_the_class():
    # The checker warns about the infinite velocity bounds, as it does for
    # Gymnasium's own CartPole-v1, whose space this matches.
    check_env(gymnasium.make(ID).unwrapped, skip_render_check=True)
    check_env(tg.TesseraEnv(), skip_render_check=True)


def test_gym_make_vec_uses_the_vector_entry_point():
    env = gymnasium.make_vec(ID, num_envs=4)
    assert isinstance(env, tg.TesseraVectorEnv)
    assert env.num_envs == 4
    assert env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
    env.close()


def test_gym_spaces_match_gymnasium_cartpole():
    ours = tg.make(ID, num_envs=3)
    ref = gymnasium.make("CartPole-v1")
    assert ours.single_observation_space == ref.observation_space
    assert ours.single_action_space == ref.action_space
    assert ours.observation_space.shape == (3, 4)
    assert ours.action_space == gymnasium.spaces.MultiDiscrete([2, 2, 2])


def test_gym_outputs_are_owned_by_default_and_typed():
    env = tg.make(ID, num_envs=3)
    obs, info = env.reset(seed=1)
    assert info == {}
    kept = obs.copy()
    assert obs.flags.writeable and obs.shape == (3, 4) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    for _ in range(5):
        o, r, term, trunc, info = env.step(np.zeros(3, dtype=np.int64))
    assert np.array_equal(obs, kept)
    assert r.shape == (3,) and r.dtype == np.float32
    assert term.dtype == np.bool_ and trunc.dtype == np.bool_
    assert not t.is_stale(obs)  # owned: never stale


def test_gym_next_step_resets_on_the_call_after_the_end():
    # Gymnasium NEXT_STEP: the step after an episode ends resets that
    # instance and returns its first observation, reward 0, no flags.
    env = tg.make(ID, num_envs=4)
    _, (obs, r, term, trunc, info) = run_until_done(env)
    done = term | trunc
    assert "final_obs" not in info
    obs2, r2, term2, trunc2, _ = env.step(np.ones(4, dtype=np.int64))
    assert (r2[done] == 0).all()
    assert not term2[done].any() and not trunc2[done].any()
    assert (np.abs(obs2[done]) <= 0.05).all()  # fresh reset state
    assert (r2[~done] == 1).all()


def test_gym_same_step_reports_final_obs_in_info():
    env = tg.make(ID, num_envs=4, autoreset_mode="SameStep")
    assert env.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
    (prev_obs, *_), (obs, r, term, trunc, info) = run_until_done(env)
    done = term | trunc
    assert info["_final_obs"].tolist() == done.tolist()
    assert info["_final_info"].tolist() == done.tolist()
    for i in range(4):
        if done[i]:
            final = info["final_obs"][i]
            assert final.shape == (4,) and final.dtype == np.float32
            # The returned observation is already the next episode's.
            assert (np.abs(obs[i]) <= 0.05).all()
            assert not np.array_equal(final, obs[i])
        else:
            assert info["final_obs"][i] is None
    assert (r[done] == 1).all()


def test_gym_disabled_needs_a_masked_reset():
    env = tg.make(ID, num_envs=3, autoreset_mode=AutoresetMode.DISABLED)
    _, (obs, _, term, trunc, _) = run_until_done(env)
    done = term | trunc
    with pytest.raises(t.PendingResetError):
        env.step(np.ones(3, dtype=np.int64))
    obs2, _ = env.reset(options={"reset_mask": done})
    assert np.array_equal(obs2[~done], obs[~done])
    assert (np.abs(obs2[done]) <= 0.05).all()
    env.step(np.ones(3, dtype=np.int64))
    with pytest.raises(t.ArgumentError):
        env.reset(options={"unknown": 1})


def test_gym_seeds_follow_the_vector_convention():
    a = tg.make(ID, num_envs=3)
    b = tg.make(ID, num_envs=3)
    c = tg.make(ID, num_envs=1)
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=[7, 8, 9])
    obs_c, _ = c.reset(seed=8)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(obs_a[1], obs_c[0])
    assert np.array_equal(a.reset(seed=7)[0], obs_a)
    with pytest.raises(ValueError):
        a.reset(seed=[1, 2])


def test_gym_borrowed_mode_is_available_and_read_only():
    env = tg.make(ID, num_envs=2, outputs="borrowed")
    obs, _ = env.reset(seed=1)
    assert not obs.flags.writeable
    env.step(np.zeros(2, dtype=np.int64))
    assert t.is_stale(obs)


def test_gym_rejects_multi_agent_environments_and_bad_actions():
    with pytest.raises(t.ArgumentError):
        tg.make("tessera/Tag-v1")
    env = tg.make(ID, num_envs=2)
    env.reset()
    with pytest.raises(t.ActionError):
        env.step(np.array([0.5, 1.0]))
    with pytest.raises(t.ActionError):
        env.step(np.zeros(3, dtype=np.int64))


def test_gym_vector_wrappers_record_episode_statistics():
    env = gymnasium.wrappers.vector.RecordEpisodeStatistics(tg.make(ID, num_envs=4))
    env.reset(seed=0)
    seen = 0
    for _ in range(600):
        _, _, term, trunc, info = env.step(np.ones(4, dtype=np.int64))
        if "episode" in info:
            mask = info["_episode"]
            assert (info["episode"]["l"][mask] > 0).all()
            seen += int(mask.sum())
        if seen:
            break
    assert seen


def _reference(n):
    """Gymnasium's own vectorization of the registered single env."""
    return gymnasium.vector.SyncVectorEnv([lambda: tg.TesseraEnv() for _ in range(n)])


@pytest.mark.parametrize("seed", [7, [7, 8, 9], [None, None, 9], None])
def test_gym_masked_reset_seeds_match_sync_vector_env(seed):
    # codex R-PR22: the full seed vector is formed before the mask.
    ours, ref = tg.make(ID, num_envs=3), _reference(3)
    mask = np.array([False, False, True])
    for env in (ours, ref):
        env.reset(seed=3)
        env.step(np.ones(3, dtype=np.int64))
    a, _ = ours.reset(seed=seed, options={"reset_mask": mask.copy()})
    b, _ = ref.reset(seed=seed, options={"reset_mask": mask.copy()})
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    ("mask", "error"),
    [
        (np.zeros(3, dtype=bool), ValueError),
        (np.array([0, 0, 1]), TypeError),
        ([False, False, True], TypeError),
        (np.ones(2, dtype=bool), ValueError),
    ],
)
def test_gym_reset_mask_is_validated_like_sync_vector_env(mask, error):
    ours, ref = tg.make(ID, num_envs=3), _reference(3)
    for env in (ours, ref):
        env.reset(seed=3)
        with pytest.raises(error):
            env.reset(options={"reset_mask": mask})


def _balance(obs):
    """A controller that keeps the pole up far beyond 500 steps."""
    return int(obs[2] + 0.25 * obs[3] + 0.05 * obs[0] + 0.1 * obs[1] > 0)


@pytest.mark.parametrize(("limit", "ends_at"), [(None, 500), (600, 600), (-1, None)])
def test_gym_time_limit_is_owned_by_the_gymnasium_wrapper(limit, ends_at):
    # codex R-PR22: no hidden native limit under gymnasium.make.
    env = gymnasium.make(ID) if limit is None else gymnasium.make(ID, max_episode_steps=limit)
    obs, _ = env.reset(seed=3)
    end = None
    for k in range(650):
        obs, _, term, trunc, _ = env.step(_balance(obs))
        assert not term, "the controller should keep the pole up"
        if trunc:
            end = k + 1
            break
    assert end == ends_at
    env.close()


@pytest.mark.parametrize(("limit", "ends_at"), [(None, 500), (600, 600), (-1, None)])
def test_gym_make_vec_time_limit_follows_the_registration(limit, ends_at):
    kwargs = {} if limit is None else {"max_episode_steps": limit}
    env = gymnasium.make_vec(ID, num_envs=1, **kwargs)
    obs, _ = env.reset(seed=3)
    end = None
    for k in range(650):
        obs, _, term, trunc, _ = env.step(np.array([_balance(obs[0])]))
        assert not term[0]
        if trunc[0]:
            end = k + 1
            break
    assert end == ends_at
    env.close()


def test_gym_seed_none_draws_and_exposes_a_reproducible_seed():
    a, b = tg.make(ID, num_envs=2), tg.make(ID, num_envs=2)
    assert 0 <= a.construction_seed < 2**64
    assert a.construction_seed != b.construction_seed
    again = tg.make(ID, num_envs=2, seed=a.construction_seed)
    assert again.construction_seed == a.construction_seed
    assert np.array_equal(a.reset()[0], again.reset()[0])
    assert tg.make(ID, num_envs=2, seed=5).construction_seed == 5
    assert gymnasium.make(ID).unwrapped.construction_seed >= 0

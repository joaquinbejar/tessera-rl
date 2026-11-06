# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PufferLib adapter at the pins of compat/targets.toml. Run through
``make puffer-test`` from a temporary directory: importing PufferLib creates a
``resources`` symlink in the current directory.

The first test is the identity test. A throwaway spike once reported a mismatch
under Multiprocessing: its reference was wrong, using
``(seed + w) * num_envs + j``. ``vector.make`` drops ``seed``, and the multiplier
is the worker's env count, so the true constructor seed is
``w * envs_per_worker + j``."""

import json
import os
import signal
import subprocess
import sys

import numpy as np
import pufferlib
import pufferlib.vector
import pytest

import tessera_rl as t
from tessera_rl import puffer
from tessera_rl.puffer import TesseraPufferEnv, make_puffer, mix_seed

CARTPOLE = "tessera/CartPole-v1"
TAG = "tessera/Tag-v1"
SCENARIOS = os.path.join(os.path.dirname(__file__), "scenarios.py")


def scenario(*args, timeout=300):
    """Runs a scenario; on a timeout, kills the recorded process group."""
    p = subprocess.Popen(
        [sys.executable, SCENARIOS, *map(str, args)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )  # fmt: skip
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        pytest.fail(f"{args} timed out\n{out[-800:]}\n{err[-2000:]}")
    # A scenario exits at once; its Multiprocessing children must go with it.
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"{args} rc={p.returncode}\n{out[-800:]}\n{err[-3000:]}"
    return json.loads(lines[-1][7:])


def leaf(step):
    return next(iter(step.observations.values()))


def assert_same(want, got):
    o, r, d, tr = got[:4]
    assert np.asarray(o).tobytes() == leaf(want).tobytes()
    assert np.asarray(r).tobytes() == want.rewards.tobytes()
    assert np.asarray(d).tobytes() == want.agent_terminated.tobytes()
    assert np.asarray(tr).tobytes() == want.agent_truncated.tobytes()


def actions(rng, vec_or_env, n):
    return rng.integers(0, vec_or_env.single_action_space.n, n, dtype=np.int32)


@pytest.mark.parametrize("start", ["spawn", "fork"] if sys.platform == "linux" else ["spawn"])
@pytest.mark.parametrize(("workers", "envs", "procs"), [(0, 4, 2), (2, 4, 2), (0, 4, 4), (0, 2, 2)])
def test_identity_multiprocessing_matches_the_seeds_the_creators_received(start, workers, envs, procs):
    out = scenario("multiprocessing_identity", start, workers, envs, procs, 1)
    assert out["diffs"] == 0
    assert out["distinct"] == out["total"], "reset seeds repeat across workers; the bases must not"
    assert out["driver_has_pool"] is False


def test_identity_check_rejects_the_spikes_wrong_reference():
    start = "fork" if sys.platform == "linux" else "spawn"
    assert scenario("multiprocessing_identity", start, 0, 4, 2, 1, "spike")["diffs"] > 0


@pytest.mark.parametrize("env_id", [CARTPOLE, TAG])
@pytest.mark.parametrize("workers", [0, 2])
def test_identity_native_backend(env_id, workers):
    n = 8
    vec = pufferlib.vector.make(make_puffer(env_id, num_envs=n, workers=workers, chunk_size=4, log_interval=None))
    obs, _ = vec.reset(seed=9)
    base = mix_seed(0, 9)
    assert vec.instance_seed_base == base
    ref = t.make(env_id, num_envs=n, seed=base, auto_reset="same_step", outputs="owned", chunk_size=4)
    first = ref.reset(base)
    assert np.asarray(obs).tobytes() == leaf(first).tobytes()
    rng = np.random.default_rng(3)
    for _ in range(300):
        acts = actions(rng, vec, vec.num_agents)
        assert_same(ref.step(acts), vec.step(acts))
    vec.close()
    ref.close()


@pytest.mark.parametrize("env_id", [CARTPOLE, TAG])
def test_identity_serial_per_env_and_seed_arithmetic(env_id):
    n, envs = 4, 3
    vec = pufferlib.vector.make(
        make_puffer(env_id, num_envs=n, log_interval=None), backend=pufferlib.vector.Serial, num_envs=envs
    )
    vec.reset(seed=5)
    per = vec.num_agents // envs
    refs = []
    for j, env in enumerate(vec.envs):
        base = mix_seed(j, 5 + j)
        assert env.instance_seed_base == base
        ref = t.make(env_id, num_envs=n, seed=base, auto_reset="same_step", outputs="owned")
        ref.reset(base)
        refs.append(ref)
    rng = np.random.default_rng(4)
    for _ in range(200):
        acts = actions(rng, vec, vec.num_agents)
        got = vec.step(acts)
        for j, ref in enumerate(refs):
            part = slice(j * per, (j + 1) * per)
            assert_same(ref.step(acts[part]), [a[part] for a in got[:4]])
    vec.close()
    for ref in refs:
        ref.close()


GOLDEN = [
    (0, 0, 0x0),
    (0, 1, 0x5692161D100B05E5),
    (1, 0, 0xD820B7E910B0F93F),
    (1, 42, 0x03E937E4DF4F4021),
    (123, 42, 0x6F32F92746FDA654),
    (2**32 - 1, 2**32 - 1, 0xB4D055FCF2CBBD7B),
]


@pytest.mark.parametrize(("c", "r", "base"), GOLDEN)
def test_mix_seed_golden_vectors(c, r, base):
    assert mix_seed(c, r) == base


@pytest.mark.parametrize("bad", [-1, 2**32, 2**63])
def test_seeds_out_of_range_are_refused_never_truncated(bad):
    with pytest.raises(t.ArgumentError):
        mix_seed(bad, 0)
    with pytest.raises(t.ArgumentError):
        mix_seed(0, bad)
    with pytest.raises(t.ArgumentError):
        TesseraPufferEnv(CARTPOLE, seed=bad)
    env = TesseraPufferEnv(CARTPOLE)
    with pytest.raises(t.ArgumentError):
        env.reset(seed=bad)
    assert env._env is None
    env.close()


def test_seed_boundaries_are_accepted():
    env = TesseraPufferEnv(CARTPOLE, seed=2**32 - 1)
    env.reset(seed=2**32 - 1)
    assert env.instance_seed_base == GOLDEN[-1][2]
    env.close()


def test_reset_none_first_uses_seed_zero_then_keeps_seeds():
    a, b = TesseraPufferEnv(CARTPOLE, num_envs=4), TesseraPufferEnv(CARTPOLE, num_envs=4)
    obs_a, _ = a.reset()
    b.reset(seed=0)
    assert a.instance_seed_base == mix_seed(0, 0)
    assert obs_a.tobytes() == b.observations.tobytes()
    before = a.instance_seed_base
    a.reset()
    assert a.instance_seed_base == before
    a.close()
    b.close()


def test_same_seed_same_bytes_and_different_constructor_differs():
    runs = []
    for c in (0, 0, 1):
        env = TesseraPufferEnv(CARTPOLE, num_envs=4, seed=c)
        env.reset(seed=7)
        env.step(np.zeros(4, dtype=np.int32))
        runs.append(env.observations.tobytes())
        env.close()
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


def test_vector_make_seed_has_no_effect():
    a = pufferlib.vector.make(make_puffer(CARTPOLE, num_envs=2), seed=123)
    b = pufferlib.vector.make(make_puffer(CARTPOLE, num_envs=2))
    assert a.reset(seed=1)[0].tobytes() == b.reset(seed=1)[0].tobytes()
    a.close()
    b.close()


def test_native_backend_shares_memory_with_the_engine():
    env = TesseraPufferEnv(CARTPOLE, num_envs=4)
    obs, _ = env.reset(seed=1)
    out = env._env._native.outputs()
    for arr, key in [(obs, None), (env.rewards, "rewards"), (env.terminals, "agent_terminated")]:
        view = np.asarray(out["observations"][0] if key is None else out[key])
        assert np.shares_memory(arr, view)
        assert not arr.flags.writeable
    before = obs.tobytes()
    got = env.step(np.ones(4, dtype=np.int32))[0]
    assert got is obs
    assert obs.tobytes() != before
    env.reset(seed=2)
    assert env.observations is obs
    env.close()


def test_vector_backend_copies_into_its_own_buffers():
    vec = pufferlib.vector.make(make_puffer(CARTPOLE, num_envs=2), backend=pufferlib.vector.Serial, num_envs=2)
    vec.reset(seed=1)
    for env in vec.envs:
        view = np.asarray(env._env._native.outputs()["observations"][0])
        assert not np.shares_memory(env.observations, view)
        assert np.shares_memory(env.observations, vec.observations)
    vec.close()


def test_dtypes_and_done_match_serial():
    native = TesseraPufferEnv(TAG, num_envs=2)
    native.reset(seed=1)
    vec = pufferlib.vector.make(make_puffer(TAG, num_envs=2), backend=pufferlib.vector.Serial, num_envs=1)
    vec.reset(seed=1)
    for a, b in [(native.observations, vec.observations), (native.rewards, vec.rewards),
                 (native.terminals, vec.terminals), (native.truncations, vec.truncations)]:  # fmt: skip
        assert a.dtype == b.dtype
        assert a.shape == b.shape
    assert native.terminals.dtype == np.bool_
    assert native.masks.all() and native.masks.shape == (native.num_agents,)
    native.close()
    vec.close()


def test_conformance_shapes_and_seven_tuple():
    env = TesseraPufferEnv(TAG, num_envs=3)
    assert env.num_agents == 3 * 8
    assert env.agents_per_batch == env.num_agents
    assert env.action_space.shape == (env.num_agents,)
    obs, infos = env.reset(seed=1)
    assert env.observation_space.contains(obs.astype(env.observation_space.dtype)) or obs.shape[0] == env.num_agents
    assert infos == []
    env.async_reset(seed=1)
    env.send(np.zeros(env.num_agents, dtype=np.int32))
    o, r, d, tr, i, ids, m = env.recv()
    assert o is env.observations
    assert (r.shape, d.shape, tr.shape, ids.shape, m.shape) == ((env.num_agents,),) * 5
    assert isinstance(i, list)
    assert env.action_masks.shape == (env.num_agents, env.single_action_space.n)
    assert env.terminal_observations.shape == o.shape
    env.close()
    env.close()


@pytest.mark.parametrize("call", ["step", "send", "recv", "masks"])
def test_calls_before_the_first_reset_raise(call):
    env = TesseraPufferEnv(CARTPOLE)
    acts = np.zeros(1, dtype=np.int32)
    with pytest.raises(t.ArgumentError, match="before the first reset"):
        {"step": lambda: env.step(acts), "send": lambda: env.send(acts), "recv": env.recv,
         "masks": lambda: env.terminal_observations}[call]()  # fmt: skip
    assert env._env is None


def test_no_pool_exists_until_the_first_reset():
    env = TesseraPufferEnv(CARTPOLE, workers=2)
    assert env._env is None
    env.reset(seed=1)
    assert env._env is not None
    env.close()


@pytest.mark.parametrize("env_id", ["tessera/Connect4-v1", "tessera/Leduc-v1", "tessera/Othello-v1", "nope"])
def test_sequential_and_unknown_environments_are_refused(env_id):
    with pytest.raises(t.ArgumentError, match="not supported"):
        make_puffer(env_id)
    with pytest.raises(t.ArgumentError, match="not supported"):
        TesseraPufferEnv(env_id)


@pytest.mark.parametrize("kw", [{"auto_reset": "next_step"}, {"mode": "pipelined"}, {"outputs": "owned"}, {"log_interval": 0}])
def test_other_settings_are_refused(kw):
    with pytest.raises(t.ArgumentError):
        TesseraPufferEnv(CARTPOLE, **kw)


def test_several_leaves_are_refused():
    with pytest.raises(t.ArgumentError, match="a.*b"):
        puffer._single_leaf([("a", "f4", [1], [0], [1]), ("b", "f4", [1], [0], [1])])


def test_invalid_pool_options_fail_at_construction():
    with pytest.raises(t.ArgumentError):
        TesseraPufferEnv(CARTPOLE, num_envs=0)


def test_out_of_range_action_raises_and_illegal_is_a_noop():
    env = TesseraPufferEnv(TAG, num_envs=1)
    env.reset(seed=1)
    with pytest.raises(t.ActionError):
        env.step(np.full(env.num_agents, 99, dtype=np.int32))
    env.close()


def test_logging_means_over_finished_episodes():
    env = TesseraPufferEnv(CARTPOLE, num_envs=8, log_interval=64, max_episode_steps=10)
    env.reset(seed=1)
    logs = []
    for _ in range(64):
        logs += env.step(np.zeros(8, dtype=np.int32))[4]
    assert len(logs) == 1
    assert logs[0]["n"] >= 8
    assert 1 <= logs[0]["episode_length"] <= 10
    assert logs[0]["episode_return"] == pytest.approx(logs[0]["episode_length"])
    off = TesseraPufferEnv(CARTPOLE, num_envs=8, log_interval=None, max_episode_steps=2)
    off.reset(seed=1)
    assert off.step(np.zeros(8, dtype=np.int32))[4] == []
    env.close()
    off.close()


def test_adapter_arrays_do_not_grow():
    env = TesseraPufferEnv(CARTPOLE, num_envs=4)
    env.reset(seed=1)
    arrays = [id(a) for a in (env.observations, env.rewards, env.terminals, env.truncations, env._episode_return)]
    for _ in range(1000):
        env.step(np.zeros(4, dtype=np.int32))
    assert arrays == [id(a) for a in (env.observations, env.rewards, env.terminals, env.truncations, env._episode_return)]
    env.close()


def test_import_failure_names_the_cause(tmp_path):
    code = "import sys; sys.modules['pufferlib'] = None\ntry:\n import tessera_rl.puffer\nexcept ImportError as e:\n print('symlink' in str(e))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert out.stdout.strip() == "True", out.stderr


@pytest.mark.parametrize("env_id", [CARTPOLE, TAG])
@pytest.mark.parametrize("rnn", ["rnn", "plain"])
def test_smoke_training(env_id, rnn):
    out = scenario("smoke", env_id, rnn, timeout=600)
    assert out["finite"]
    assert out["global_step"] >= 8192
    assert out["checks"] >= 1

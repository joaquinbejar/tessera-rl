# SPDX-License-Identifier: MIT OR Apache-2.0
"""The out-of-tree example wheel examples/host_orderbook (#272 step 6,
ADR-0062 item 11), installed next to the first-party wheel
(`make py-test-orderbook`): tessera_rl.make, the Gymnasium adapter and env
checker, the PettingZoo adapters and API tests, the borrowed-output rules
against the host object, cross-wheel exception classes, the floating-point
boundary under hostile caller controls, and determinism across two runs.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import tessera_rl

pytest.importorskip(
    "tessera_example_orderbook._tessera_host", reason="needs the example wheel (make py-test-orderbook)"
)

ID = "example/OrderBook-v1"
CHILD = pathlib.Path(__file__).with_name("fp_child.py")


def acts(n, k):
    """A fixed action script, ``side * 4 + level``: hold, buy and sell in
    turn, at levels 0 to 3. Shape ``(n, 1)``."""
    side = (k + np.arange(n)) % 3
    level = (k + np.arange(n)) % 4
    return (side * 4 + level).astype(np.int32).reshape(n, 1)


def test_host_orderbook_make_finds_the_example_through_its_entry_point():
    env = tessera_rl.make(ID, num_envs=3, seed=4, episode=5)
    assert isinstance(env, tessera_rl.HostEnv)
    assert (env.env_id, env.agents, env.turns) == (ID, 1, "simultaneous")
    step = env.reset()
    obs = step.observations["obs"]
    assert obs.shape == (3, 1, 4) and obs.dtype == np.int32
    # Seeds 4, 5 and 6: the opening price is the seed modulo 97.
    assert obs[:, 0, 2].tolist() == [4, 5, 6]
    for k in range(5):
        step = env.step(acts(3, k))
    assert step.episode_terminated.tolist() == [True] * 3
    env.close()


def test_host_orderbook_gymnasium_vector_env_and_env_checker():
    gym = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium")
    from gymnasium.utils.env_checker import check_env

    vec = gym.make(ID, num_envs=4, seed=2, episode=6)
    assert vec.metadata["tessera_kind"] == "host"
    obs, info = vec.reset(seed=2)
    assert obs.shape == (4, 4)
    for k in range(8):
        obs, rewards, terminated, truncated, info = vec.step(acts(4, k)[:, 0])
        assert rewards.dtype == np.float32
    vec.close()
    single = gym.TesseraEnv(ID, seed=3)
    check_env(single, skip_render_check=True)
    single.close()


def test_host_orderbook_pettingzoo_api_tests():
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    from pettingzoo.test import api_test, parallel_api_test

    par = pz.parallel_env(ID, seed=3, episode=10)
    assert par.metadata["tessera_kind"] == "host"
    parallel_api_test(par, num_cycles=30)
    par.close()
    aec = pz.aec_env(ID, seed=3, episode=10)
    api_test(aec, num_cycles=30)
    aec.close()


def test_host_orderbook_borrowed_views_follow_the_borrow_window():
    # SPEC 5.3 rule 7 against the host object: read-only views of the
    # host's storage, stale and overwritten after the next call, readable
    # after close; owned copies never stale.
    env = tessera_rl.make(ID, num_envs=2, seed=1)
    step = env.reset()
    obs, rewards = step.observations["obs"], step.rewards
    assert not obs.flags.writeable and not rewards.flags.writeable
    with pytest.raises(ValueError):
        obs[0, 0, 0] = 1
    kept = obs.copy()
    assert not tessera_rl.is_stale(obs) and not tessera_rl.is_stale(memoryview(rewards))
    after = env.step(acts(2, 1))
    assert tessera_rl.is_stale(obs) and tessera_rl.is_stale(rewards)
    assert np.array_equal(obs, after.observations["obs"])
    assert not np.array_equal(obs, kept)
    final = after.observations["obs"].copy()
    env.close()
    assert np.array_equal(after.observations["obs"], final), "views stay readable after close"
    owned = tessera_rl.make(ID, num_envs=2, seed=1, outputs="owned")
    first = owned.reset().observations["obs"]
    owned.step(acts(2, 1))
    assert not tessera_rl.is_stale(first)
    owned.close()


def test_host_orderbook_errors_are_the_first_party_classes():
    env = tessera_rl.make(ID, num_envs=2, seed=1)
    env.reset()
    view = env.step(acts(2, 0)).observations["obs"]
    before = view.copy()
    bad = acts(2, 1)
    bad[0, 0] = 12
    try:
        env.step(bad)
    except tessera_rl.ActionError as e:
        assert type(e) is tessera_rl.ActionError
    else:
        pytest.fail("an out-of-range action is an ActionError")
    # Refused before the pool is touched: the last window is still open.
    assert not tessera_rl.is_stale(view) and np.array_equal(view, before)
    with pytest.raises(tessera_rl.CapabilityError):
        env.snapshot(0)
    with pytest.raises(tessera_rl.ConfigError):
        tessera_rl.make(ID, num_envs=1, episode=0)
    env.close()


def episode_hashes(seed):
    """Hashes of the observations and the rewards of one Gymnasium episode
    and one PettingZoo episode, each with the same seed and actions."""
    gym = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium")
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    obs_h, rew_h = hashlib.sha256(), hashlib.sha256()
    vec = gym.make(ID, num_envs=3, seed=seed, episode=12)
    obs, _ = vec.reset(seed=seed)
    obs_h.update(np.ascontiguousarray(obs).tobytes())
    for k in range(12):
        obs, rewards, *_ = vec.step(acts(3, k)[:, 0])
        obs_h.update(np.ascontiguousarray(obs).tobytes())
        rew_h.update(np.ascontiguousarray(rewards).tobytes())
    vec.close()
    par = pz.parallel_env(ID, seed=seed, episode=12)
    observations, _ = par.reset(seed=seed)
    k = 0
    while par.agents:
        actions = {a: int(acts(1, k)[0, 0]) for a in par.agents}
        observations, rewards, *_ = par.step(actions)
        for a in sorted(observations):
            # A PettingZoo observation is a dict of leaves and the mask.
            for key in sorted(observations[a]):
                obs_h.update(key.encode() + np.ascontiguousarray(observations[a][key]).tobytes())
            rew_h.update(np.float32(rewards[a]).tobytes())
        k += 1
    par.close()
    return obs_h.hexdigest(), rew_h.hexdigest()


def test_host_orderbook_two_runs_with_one_seed_hash_equal():
    first, second = episode_hashes(11), episode_hashes(11)
    assert first == second
    assert episode_hashes(12) != first, "the seed reaches the outputs"


def fp_child(scenario):
    env = {**os.environ, "TESSERA_FP_CHILD_ENV": "orderbook"}
    proc = subprocess.run([sys.executable, str(CHILD), scenario], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    if "skipped" not in r:
        assert r["ran"] == [ID], "the child ran the example's episode only"
    return r


def test_host_orderbook_results_identical_under_hostile_rounding():
    r = fp_child("identical")
    assert r["identical"], "the host wheel returns the same bytes under hostile rounding"
    assert r["round_restored"], "the caller's rounding mode survives every call"


def test_host_orderbook_unmasked_exceptions_never_trap():
    from test_fp_boundary import check_unmasked

    check_unmasked(fp_child("unmasked"))


def test_host_orderbook_torch_flush_denormal_is_invisible():
    pytest.importorskip("torch")
    r = fp_child("torch_ftz")
    assert r["supported"], "torch can enable flush-to-zero on this CPU"
    assert r["identical"]

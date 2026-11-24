"""Adapter conformance of ``{{env_id}}`` (SPEC 10.2), run by
``cargo tessera check --python`` against the installed wheel: discovery
through ``tessera_rl.make``, the Gymnasium env checker (one agent,
simultaneous turns), PettingZoo's API tests for the interfaces the turns
declare, and equal hashes across two runs with one seed."""

import hashlib

import numpy as np
import pytest

import tessera_rl

pytest.importorskip("{{py_package}}._tessera_host", reason="needs this crate's wheel (cargo tessera check --python)")

ID = "{{env_id}}"
AGENTS = {{agents}}


def test_make_finds_the_environment_through_its_entry_point():
    env = tessera_rl.make(ID, num_envs=3, seed=4)
    assert isinstance(env, tessera_rl.HostEnv)
    assert (env.env_id, env.agents, env.turns) == (ID, AGENTS, "{{turns}}")
    step = env.reset()
    obs = step.observations["obs"]
    assert obs.shape == (3, AGENTS, 4) and obs.dtype == np.int32
    env.close()
{{#gym}}


def test_gymnasium_env_checker():
    gym = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium")
    from gymnasium.utils.env_checker import check_env

    single = gym.TesseraEnv(ID, seed=3)
    check_env(single, skip_render_check=True)
    single.close()
{{/gym}}
{{#parallel}}


def test_pettingzoo_parallel_api():
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    from pettingzoo.test import parallel_api_test

    par = pz.parallel_env(ID, seed=3)
    parallel_api_test(par, num_cycles=30)
    par.close()
{{/parallel}}
{{#aec}}


def test_pettingzoo_aec_api():
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    from pettingzoo.test import api_test

    aec = pz.aec_env(ID, seed=3)
    api_test(aec, num_cycles=30)
    aec.close()
{{/aec}}


def run_hash(seed):
    """The hash of the observations and rewards of 40 steps."""
    h = hashlib.sha256()
    env = tessera_rl.make(ID, num_envs=3, seed=seed)
    step = env.reset()
    h.update(np.ascontiguousarray(step.observations["obs"]).tobytes())
    actions = np.zeros((3, AGENTS), dtype=np.int32)
    for k in range(40):
        actions[:] = k % 3
        step = env.step(actions)
        h.update(np.ascontiguousarray(step.observations["obs"]).tobytes())
        h.update(np.ascontiguousarray(step.rewards).tobytes())
    env.close()
    return h.hexdigest()


def test_two_runs_with_one_seed_hash_equal():
    assert run_hash(5) == run_hash(5)
    assert run_hash(5) != run_hash(6)

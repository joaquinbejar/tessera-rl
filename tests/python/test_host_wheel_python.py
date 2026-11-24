# SPDX-License-Identifier: MIT OR Apache-2.0
"""tessera_rl over the compiled test host wheel (#272 step 5, SPEC 14.4
draft, `make py-test-host`): the entry point of tests/host-wheel, the
HostEnv wrapper, is_stale over real host views, set_fp_policy reaching the
host module, cross-wheel exception classes, and the PettingZoo API test."""

import numpy as np
import pytest

import tessera_rl

_package = pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")
host = getattr(_package, "_tessera_host", _package)

ID = "test/FactoryCounter-v1"


def test_host_wheel_is_found_through_its_entry_point():
    env = tessera_rl.make(ID, num_envs=3, seed=4, offset=1)
    assert isinstance(env, tessera_rl.HostEnv)
    assert (env.env_id, env.agents, env.turns) == (ID, 2, "mixed")
    step = env.reset()
    assert step.observations["obs"][:, 0].tolist() == [[1, 4], [1, 5], [1, 6]]
    env.close()


def test_host_wheel_is_stale_follows_host_views():
    env = tessera_rl.make(ID, num_envs=2)
    step = env.reset()
    rewards, obs = step.rewards, step.observations["obs"]
    view = memoryview(rewards)
    assert not tessera_rl.is_stale(rewards) and not tessera_rl.is_stale(obs[:, 0])
    assert not tessera_rl.is_stale(view)
    env.step(np.zeros((2, 2)))
    assert tessera_rl.is_stale(rewards) and tessera_rl.is_stale(obs[:, 0]) and tessera_rl.is_stale(view)
    owned = tessera_rl.make(ID, num_envs=2, outputs="owned")
    assert not tessera_rl.is_stale(owned.reset().rewards)
    env.close()
    owned.close()


def test_host_wheel_set_fp_policy_reaches_the_host_module():
    env = tessera_rl.make(ID, num_envs=1)
    try:
        tessera_rl.set_fp_policy("reject")
        assert host._fp_policy() == "reject"
    finally:
        tessera_rl.set_fp_policy("canonicalize")
    assert host._fp_policy() == "canonicalize"
    env.close()


def test_host_wheel_errors_are_tessera_rl_classes():
    env = tessera_rl.make(ID, num_envs=2)
    env.reset()
    try:
        env.step(np.full((2, 2), 7))
    except tessera_rl.ActionError as e:
        assert type(e) is tessera_rl.ActionError
    else:
        pytest.fail("an out-of-range action is an ActionError")
    # The test counter has snapshots (ADR-0073); snapshot_all stays
    # first-party only.
    with pytest.raises(tessera_rl.CapabilityError):
        env.snapshot_all()
    env.close()


def test_host_wheel_pettingzoo_parallel_api():
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    from pettingzoo.test import parallel_api_test

    env = pz.parallel_env(ID, seed=3)
    parallel_api_test(env, num_cycles=20)
    env.close()


def test_host_wheel_int32_ndarray_actions_step_as_bytes():
    # Env.step passes an int32 C-contiguous array on as a memoryview (#152
    # E2); the host backend turns it into the exact bytes protocol 1 takes,
    # so the result equals the converting path's.
    a = tessera_rl.make(ID, num_envs=2, seed=4, offset=1, outputs="owned")
    b = tessera_rl.make(ID, num_envs=2, seed=4, offset=1, outputs="owned")
    a.reset()
    b.reset()
    for k in range(3):
        acts = np.full((2, 2), k % 2, dtype=np.int32)
        sa, sb = a.step(acts), b.step(acts.astype(np.int64))
        assert np.array_equal(sa.observations["obs"], sb.observations["obs"])
        assert np.array_equal(sa.rewards, sb.rewards)
    a.close()
    b.close()

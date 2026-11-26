# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ``agent_lanes`` pool option (ADR-0100 decision 1.6, #759): ``"auto"``
(the default) or ``"reference"``. It changes which step runs, never what it
computes, so both give identical outputs; any other value is refused. No
shipped environment sets the agent-lane hook yet, so here both run the
reference step: the test pins the option's plumbing and its contract."""

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _env

ID = "tessera/Survival-v1"
SMALL = {"agents": 16, "size": 32, "steps": 64, "respawn": 2}


def run(agent_lanes):
    env = tessera_rl.make(ID, num_envs=2, seed=7, outputs="owned", agent_lanes=agent_lanes, **SMALL)
    out = [env.reset()]
    rng = np.random.default_rng(1)
    for _ in range(20):
        actions = np.stack([rng.integers(0, 5, (2, 128)), rng.integers(0, 7, (2, 128))], axis=-1).astype(np.int32)
        out.append(env.step(actions))
    snaps = [env.snapshot(i) for i in range(2)]
    env.close()
    return out, snaps


@pytest.mark.parametrize("value", ["auto", "reference"])
def test_agent_lanes_auto_and_reference_give_identical_outputs(value):
    base, base_snaps = run(None)
    got, snaps = run(value)
    assert snaps == base_snaps
    for a, b in zip(base, got):
        assert np.array_equal(a.rewards, b.rewards)
        for k in a.observations:
            assert np.array_equal(a.observations[k], b.observations[k]), k


def test_agent_lanes_bad_value_is_argument_error():
    with pytest.raises(tessera_rl.ArgumentError, match="agent_lanes"):
        tessera_rl.make(ID, num_envs=1, agent_lanes="fast", **SMALL)


def test_agent_lanes_is_an_execution_option():
    # A shadow pool (viewer.watch) drops it, as it drops the other options
    # that change how a pool runs, not what its records are.
    assert "agent_lanes" in _env._EXECUTION_OPTIONS
    assert "agent_lanes" in _env._STRING_OPTIONS

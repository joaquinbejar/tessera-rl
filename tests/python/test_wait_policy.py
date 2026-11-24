# SPDX-License-Identifier: MIT OR Apache-2.0
"""The worker wait policy (ADR-0074): ``wait="adaptive"`` with
``wait_cap_us`` on ``make`` and the PufferLib adapter. The policy changes
only how workers wait, never what a step computes."""

import numpy as np
import pytest

import tessera_rl

ID = "tessera/CartPole-v1"


def trajectory(**kwargs):
    env = tessera_rl.make(ID, num_envs=8, seed=3, outputs="owned", **kwargs)
    env.reset()
    rng = np.random.default_rng(0)
    out = []
    for _ in range(200):
        step = env.step(rng.integers(0, 2, size=(8, 1), dtype=np.int32))
        out.append(np.asarray(step.observations["obs"]).copy())
    env.close()
    return np.stack(out)


def test_wait_policy_adaptive_steps_bit_identical_to_fixed():
    fixed = trajectory(workers=2)
    adaptive = trajectory(workers=2, wait="adaptive", wait_cap_us=5_000)
    assert fixed.tobytes() == adaptive.tobytes()


@pytest.mark.parametrize("extra", [{"mode": "pipelined", "chunk_size": 2}, {"mode": "async", "batch_size": 2}])
def test_wait_policy_split_phase_modes_accept_adaptive(extra):
    # Their workers wait in the same gang (ADR-0074 item 5); workers default to 1.
    tessera_rl.make(ID, num_envs=8, wait="adaptive", wait_cap_us=2_000, **extra).close()


def test_wait_policy_fixed_and_default_cap_accepted():
    tessera_rl.make(ID, num_envs=2, workers=1, wait="fixed").close()
    tessera_rl.make(ID, num_envs=2, workers=1, wait="adaptive").close()
    tessera_rl.make(ID, num_envs=2, workers=1, wait="adaptive", wait_cap_us=1).close()
    tessera_rl.make(ID, num_envs=2, workers=1, wait="adaptive", wait_cap_us=1_000_000).close()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"workers": 0, "wait": "adaptive"}, "wait needs workers"),
        ({"workers": 1, "wait": "eager"}, "wait must be"),
        ({"workers": 1, "wait": "adaptive", "wait_cap_us": 0}, "wait_cap_us must be in"),
        ({"workers": 1, "wait": "adaptive", "wait_cap_us": 1_000_001}, "wait_cap_us must be in"),
        ({"workers": 1, "wait_cap_us": 100}, "wait_cap_us needs"),
        ({"workers": 1, "wait": "fixed", "wait_cap_us": 100}, "wait_cap_us needs"),
    ],
)
def test_wait_policy_bad_arguments_raise_argument_error(kwargs, match):
    with pytest.raises(tessera_rl.ArgumentError, match=match):
        tessera_rl.make(ID, num_envs=2, **kwargs)


def test_wait_policy_puffer_passes_wait_through():
    pytest.importorskip("pufferlib")
    from tessera_rl.puffer import TesseraPufferEnv, make_puffer

    env = TesseraPufferEnv(ID, num_envs=4, workers=1, wait="adaptive", wait_cap_us=2_000)
    env.reset(seed=0)
    env.step(np.zeros(4, dtype=np.int32))
    env.close()
    # The creator form carries the keywords too.
    creator = make_puffer(ID, num_envs=4, workers=1, wait="adaptive")
    env = creator()
    env.reset(seed=0)
    env.close()

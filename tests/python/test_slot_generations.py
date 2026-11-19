# SPDX-License-Identifier: MIT OR Apache-2.0
"""``slot_generations`` from the first-party ``_native.Env`` (issue #273):
the same name, dtype and shape as host protocol 1's output, in borrowed and
owned mode. No first-party environment spawns agents (none calls
``agent_spawned``), so every slot keeps generation 0; the increment path is
covered by the engine's Rust tests."""

import numpy as np
import pytest

import tessera_rl as t

N = 3
STEPS = 40

CASES = [
    ("tessera/Tag-v1", {"pursuers": 2, "evaders": 2, "radius": 1}),
    ("tessera/Snake-v1", {}),
]


def generations(env) -> np.ndarray:
    """The last whole-pool call's ``slot_generations``, copied at once (a
    borrowed view is valid only until the next call)."""
    raw = env._last_out
    assert raw is not None
    return np.frombuffer(bytes(raw["slot_generations"]), dtype="<u4").reshape(env.num_envs, -1).copy()


def run(env_id, kwargs, outputs):
    env = t.make(env_id, num_envs=N, seed=7, outputs=outputs, **kwargs)
    env.reset()
    agents = env._native.layout()["agents"]
    rng = np.random.default_rng(5)
    seen = [generations(env)]
    alive = []
    for _ in range(STEPS):
        s = env.step(rng.integers(0, 2, size=(N, agents)))
        seen.append(generations(env))
        alive.append(np.array(s.alive, copy=True))
    env.close()
    return np.stack(seen), np.stack(alive), agents


@pytest.mark.parametrize(("env_id", "kwargs"), CASES)
@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_slot_generations_shape_dtype_and_name(env_id, kwargs, outputs):
    env = t.make(env_id, num_envs=N, seed=7, outputs=outputs, **kwargs)
    env.reset()
    raw = env._last_out
    agents = env._native.layout()["agents"]
    assert "slot_generations" in raw
    view = memoryview(raw["slot_generations"])
    assert view.nbytes == N * agents * 4
    if outputs == "borrowed":
        # The borrowed view carries the host protocol's dtype and shape.
        assert np.asarray(raw["slot_generations"]).dtype == np.uint32
        assert np.asarray(raw["slot_generations"]).shape == (N, agents)
    env.close()


@pytest.mark.parametrize(("env_id", "kwargs"), CASES)
def test_slot_generations_borrowed_equals_owned_and_stays_zero(env_id, kwargs):
    borrowed, alive_b, agents = run(env_id, kwargs, "borrowed")
    owned, alive_o, _ = run(env_id, kwargs, "owned")
    assert borrowed.shape == (STEPS + 1, N, agents)
    assert np.array_equal(borrowed, owned)
    assert np.array_equal(alive_b, alive_o)
    # Agents die (Tag catches evaders, snakes crash), but nothing respawns.
    assert not alive_b.all()
    assert not borrowed.any()

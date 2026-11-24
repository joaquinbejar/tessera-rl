# SPDX-License-Identifier: MIT OR Apache-2.0
"""A recorded poisoning step replays to the same poison (ADR-0072 section 3,
``Poisoned``): ``summary.poisoned`` end to end. Needs the test-hooks wheel
(``make py-test-hooks``), whose panic counter panics in its first step."""

import pytest

import tessera_rl
from tessera_rl import _native, replay

if not hasattr(_native, "_make_panic_counter"):
    pytest.skip("needs the test-hooks wheel: make py-test-hooks", allow_module_level=True)


def panic_counter():
    return _native._make_panic_counter(num_envs=4, seed=1, outputs="owned")


def test_replay_poisoning_step_replays_to_summary_poisoned(tmp_path):
    path = tmp_path / "poison.trpl"
    env = panic_counter()
    with replay.record(env, path):
        env.reset()
        agents = len(bytes(env.outputs()["rewards"])) // 4
        with pytest.raises(tessera_rl.PoisonedError):
            env.step(b"\0" * (4 * agents))
        assert env.is_poisoned()
    assert path.exists()
    fresh = panic_counter()
    summary = replay.run(path, fresh)
    assert summary.poisoned is True
    assert summary.unverified_poison is False
    assert fresh.is_poisoned()

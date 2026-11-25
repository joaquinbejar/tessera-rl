# SPDX-License-Identifier: MIT OR Apache-2.0
"""Curricula from Python (ADR-0081 items 3 and 5.5, #613 head B):
``targets=True``, ``targets()``, ``set_targets()`` and
``epoch_barrier(curriculum=...)`` on Gather, whose ``items`` is adaptive,
and the replay of a Python-curriculum run without the callable."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import replay

GATHER = "tessera/Gather-v1"
N = 4


def make(**kwargs):
    kwargs.setdefault("targets", True)
    kwargs.setdefault("auto_reset", "disabled")
    return t.make(GATHER, num_envs=N, seed=7, outputs="owned", **kwargs)


def reset_counts(env):
    out = env.reset()
    mask = np.asarray(out.observations["items_mask"])
    return [int(mask[i].sum()) for i in range(N)]


def test_targets_default_to_the_config_and_round_trip():
    env = make(items=5)
    assert env.adaptive_fields == ("items",)
    assert env.targets() == [{"items": 5}] * N
    env.set_targets({1: {"items": 2}, 3: {"items": 8}})
    assert env.targets() == [{"items": 5}, {"items": 2}, {"items": 5}, {"items": 8}]
    # A copy each call: mutating it writes nothing.
    got = env.targets()
    got[0]["items"] = 1
    assert env.targets()[0] == {"items": 5}
    env.close()


def test_set_targets_applies_at_the_next_reset_only():
    env, twin = make(items=5), make(items=5)
    assert reset_counts(env) == [5] * N
    twin.reset()
    env.set_targets({0: {"items": 2}})
    # Mid-episode: the running episodes are those of a pool without the
    # change, step for step.
    act = np.zeros((N, 1), dtype=np.int32)
    for _ in range(3):
        a, b = env.step(act), twin.step(act)
        for leaf in ("agent", "items", "items_mask"):
            assert np.array_equal(a.observations[leaf], b.observations[leaf])
    assert [env.state_hash(i) for i in range(N)] == [twin.state_hash(i) for i in range(N)]
    # The next reset uses the target.
    assert reset_counts(env) == [2, 5, 5, 5]
    env.close()
    twin.close()


def test_set_targets_refusals_change_nothing():
    env = make(items=5)
    before = env.targets()
    with pytest.raises(t.ArgumentError):
        env.set_targets({0: {"items": 2}, N: {"items": 3}})
    with pytest.raises(t.ArgumentError):
        env.set_targets({0: {"size": 4}})
    with pytest.raises(t.ArgumentError):
        env.set_targets({0: {"items": 2.5}})
    with pytest.raises(t.ArgumentError):
        env.set_targets({0: {"items": True}})
    with pytest.raises(t.ConfigError):
        env.set_targets({0: {"items": 2}, 1: {"items": 9}})
    with pytest.raises(t.ConfigError):
        env.set_targets({0: {"items": 0}})
    assert env.targets() == before
    env.close()


def test_targets_need_an_adaptive_environment_and_sync_mode():
    with pytest.raises(t.ArgumentError):
        t.make("tessera/CartPole-v1", targets=True)
    with pytest.raises(t.ArgumentError):
        t.make(GATHER, num_envs=N, targets=True, mode="pipelined")
    plain = t.make(GATHER, num_envs=N)
    assert plain.adaptive_fields == ()
    with pytest.raises(t.ArgumentError):
        plain.targets()
    with pytest.raises(t.ArgumentError):
        plain.set_targets({0: {"items": 2}})
    plain.close()


def raise_on_return(report, targets):
    """Raises every slot's ``items`` by one after a barrier that saw
    episodes end, up to 8."""
    if report["episodes"]["count"] == 0:
        return None
    return {i: {"items": min(8, row["items"] + 1)} for i, row in enumerate(targets)}


def run_curriculum(env, steps=60):
    env.reset()
    rng = np.random.default_rng(1)
    for k in range(steps):
        env.step(rng.integers(0, 4, size=(N, 1)).astype(np.int32))
        if k % 10 == 9:
            env.epoch_barrier(curriculum=raise_on_return)
            env.reset()
    return [env.state_hash(i) for i in range(N)], env.targets()


def test_epoch_barrier_curriculum_changes_targets_only_at_barriers():
    env = make(items=1, max_steps=8, auto_reset="next_step")
    seen = []

    def spy(report, targets):
        seen.append([row["items"] for row in targets])
        # Mutating the copy writes nothing.
        targets[0]["items"] = 7
        return raise_on_return(report, targets)

    env.reset()
    env.step(np.zeros((N, 1), dtype=np.int32))
    assert env.epoch_barrier(curriculum=spy)["episodes"]["count"] >= 0
    assert seen == [[1, 1, 1, 1]]
    hashes, targets = run_curriculum(env)
    again = make(items=1, max_steps=8, auto_reset="next_step")
    again.reset()
    again.step(np.zeros((N, 1), dtype=np.int32))
    again.epoch_barrier(curriculum=raise_on_return)
    assert run_curriculum(again) == (hashes, targets)
    assert all(row["items"] > 1 for row in targets)
    env.close()
    again.close()


def test_epoch_barrier_curriculum_exception_propagates_and_writes_nothing():
    env = make(items=3)
    env.reset()
    before = env.targets()

    class Boom(Exception):
        pass

    def boom(report, targets):
        raise Boom

    with pytest.raises(Boom):
        env.epoch_barrier(curriculum=boom)
    assert env.targets() == before
    assert env.epoch_barrier(curriculum=lambda r, tg: {})["curriculum"] is None
    env.close()


def test_python_curriculum_run_replays_without_the_callable(tmp_path):
    path = tmp_path / "curriculum.trpl"
    env = make(items=1, max_steps=8, auto_reset="next_step")
    with replay.record(env, path, checkpoint_every=4):
        hashes, targets = run_curriculum(env)
    env.close()
    target_env = make(items=1, max_steps=8, auto_reset="next_step")
    summary = replay.run(path, target_env)
    assert not summary.poisoned
    assert [target_env.state_hash(i) for i in range(N)] == hashes
    assert target_env.targets() == targets
    target_env.close()
    plain = t.make(GATHER, num_envs=N, seed=7, outputs="owned", auto_reset="next_step", items=1, max_steps=8)
    with pytest.raises(t.PersistError):
        replay.run(path, plain)
    plain.close()

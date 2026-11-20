# SPDX-License-Identifier: MIT OR Apache-2.0
"""The view set is built once per environment (plan #152 E3), but every call
still makes new views stamped with that call's window: a view of an earlier
call is stale after each writer (step, reset, restore, fork, load) and the
new views are fresh, with the layout's shapes and the current values."""

import numpy as np
import pytest

import tessera_rl as t

N = 4


def cartpole(**kw):
    return t.make("tessera/CartPole-v1", num_envs=N, seed=3, **kw)


def ones():
    return np.ones((N, 1), dtype=np.int32)


def check_fresh(env, step):
    # Fresh, the layout's shapes, and the values the owned path reports.
    for view in (step.observations["obs"], step.rewards, step.alive, step.agent_terminated):
        assert not t.is_stale(view)
    assert step.observations["obs"].shape == (N, 1, 4)
    assert step.rewards.shape == (N, 1)
    raw = env._native.outputs()
    assert step.observations["obs"].tobytes() == bytes(raw["observations"][0])
    assert step.rewards.tobytes() == bytes(raw["rewards"])


def test_viewset_cache_many_steps_each_window_is_fresh_then_stale():
    env = cartpole()
    prev = env.reset()
    check_fresh(env, prev)
    for _ in range(5):
        cur = env.step(ones())
        assert t.is_stale(prev.observations["obs"]) and t.is_stale(prev.rewards)
        check_fresh(env, cur)
        prev = cur
    env.close()


@pytest.mark.parametrize("writer", ["reset", "import_record", "fork", "load"])
def test_viewset_cache_writer_stales_old_views_and_new_ones_are_fresh(writer):
    env = cartpole()
    env.reset()
    old = env.step(ones())
    record, data = env.snapshot(0), env.encode()
    env.step(ones())
    old = env.step(ones())
    if writer == "reset":
        env.reset()
    elif writer == "import_record":
        env.import_record(1, record)
    elif writer == "fork":
        env.fork(0, [1, 2])
    else:
        env.load(data)
    assert t.is_stale(old.observations["obs"]) and t.is_stale(old.rewards)
    check_fresh(env, env._wrap(env._native.outputs()))
    after = env.step(ones())
    check_fresh(env, after)
    env.close()


def test_viewset_cache_close_keeps_views_readable_and_refuses_new_ones():
    env = cartpole()
    env.reset()
    s = env.step(ones())
    kept = s.observations["obs"].copy()
    env.close()
    assert not t.is_stale(s.observations["obs"])
    assert np.array_equal(s.observations["obs"], kept)
    with pytest.raises(t.ArgumentError):
        env._native.outputs()


def test_viewset_cache_two_environments_do_not_share_views():
    a, b = cartpole(), t.make("tessera/CartPole-v1", num_envs=N, seed=9)
    sa, sb = a.reset(), b.reset()
    assert sa.observations["obs"].__array_interface__["data"][0] != sb.observations["obs"].__array_interface__["data"][0]
    b.step(ones())
    assert not t.is_stale(sa.observations["obs"])
    a.close()
    b.close()


def test_viewset_cache_pipelined_halves_keep_their_own_rows():
    env = t.make("tessera/Tag-v1", num_envs=N, seed=5, mode="pipelined", workers=2, chunk_size=1)
    owned = t.make("tessera/Tag-v1", num_envs=N, seed=5, mode="pipelined", workers=2, chunk_size=1, outputs="owned")
    for _ in range(3):
        got, want = env.reset(), owned.reset()
        for g, w in zip(got, want, strict=True):
            assert g.step.rewards.shape[0] == len(g.ids) == 2
            for key, view in g.step.observations.items():
                assert view.tobytes() == w.step.observations[key].tobytes()
            assert not t.is_stale(g.step.rewards)
        first, second = got
        zeros = np.zeros(2 * env.agents, dtype=np.int32)
        for e in (env, owned):
            e.send(1, zeros)
        assert t.is_stale(second.step.rewards) and not t.is_stale(first.step.rewards)
        r, ro = env.recv(), owned.recv()
        assert r.half == 1 and list(r.ids) == [2, 3]
        assert r.step.rewards.tobytes() == ro.step.rewards.tobytes()
    env.close()
    owned.close()


def test_viewset_cache_async_indexed_views_are_fresh_per_batch():
    env = t.make("tessera/CartPole-v1", num_envs=8, seed=5, mode="async", workers=2, batch_size=4, gather="indexed")
    env.reset()
    zeros = np.zeros(4, dtype=np.int32)
    env.send([0, 1, 2, 3], zeros)
    first = env.recv()
    assert first.step.rewards.shape == (8, 1) and not t.is_stale(first.step.rewards)
    env.send(first.ids.tolist(), zeros)
    second = env.recv()
    assert t.is_stale(first.step.rewards) and not t.is_stale(second.step.rewards)
    env.close()

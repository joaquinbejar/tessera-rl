# SPDX-License-Identifier: MIT OR Apache-2.0
"""tessera_rl.jax: the ordered-callback JAX handles (SPEC 15.2, ADR-0031,
docs/reviews/jax-ordered-callback-design.md r3, section 7). CPU only.

Multi-device cases run in a subprocess with two forced CPU devices, since
XLA reads its flags once per process.
"""

import gc
import json
import os
import subprocess
import sys
import textwrap
import warnings
import weakref

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.jax

import jax.numpy as jnp  # noqa: E402
from jax.experimental import io_callback  # noqa: E402

import tessera_rl as t  # noqa: E402
import tessera_rl.jax as tjax  # noqa: E402

JaxRuntimeError = jax.errors.JaxRuntimeError
FIELDS = [f for f in t.Step._fields if f != "observations"]


# ---------------------------------------------------------------- helpers


def native(env_id, n, seed, **kw):
    return t.make(env_id, num_envs=n, seed=seed, outputs="owned", **kw)


def assert_same(jstep, nstep):
    """Bit-for-bit equality of a tjax.Step and a native Step."""
    assert set(jstep.observations) == set(nstep.observations)
    for k, v in nstep.observations.items():
        got = np.asarray(jstep.observations[k])
        assert got.dtype == v.dtype and got.shape == v.shape and got.tobytes() == v.tobytes(), k
    for f in FIELDS:
        a, b = getattr(jstep, f), getattr(nstep, f)
        if b is None:
            assert a is None, f
            continue
        a = np.asarray(a)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), f


def index(stacked, i):
    return jax.tree.map(lambda x: x[i], stacked)


def legal_actions(step):
    """First legal choice per agent row, 0 where there is no mask."""
    if step.masks is None:
        return np.zeros(step.rewards.shape, np.int32)
    return np.argmax(np.asarray(step.masks), axis=-1).astype(np.int32)


CASES = [
    ("tessera/CartPole-v1", {}, "cartpole"),
    ("tessera/Tag-v1", {"pursuers": 2, "evaders": 2, "radius": 1}, "tag"),
    ("tessera/Connect4-v1", {}, "connect4"),
]


def native_rollout(env_id, n, seed, kw, steps):
    """Actions chosen from the native trajectory, and its steps."""
    env = native(env_id, n, seed, **kw)
    s = env.reset()
    out, acts = [s], []
    for _ in range(steps):
        a = legal_actions(s) if env_id != "tessera/CartPole-v1" else np.ones((n, env.agents), np.int32)
        acts.append(a)
        s = env.step(a)
        out.append(s)
    env.close()
    return np.stack(acts), out


def run_py(code, devices=1):
    envs = dict(os.environ)
    if devices > 1:
        envs["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=envs, timeout=300)
    return p.returncode, p.stdout, p.stderr


# ------------------------------------------------------------ 1. equivalence


@pytest.mark.parametrize(("env_id", "kw", "name"), CASES, ids=[c[2] for c in CASES])
def test_jax_scan_trajectory_equals_native_bit_for_bit(env_id, kw, name):
    acts, ref = native_rollout(env_id, 3, 11, kw, 20)
    with tjax.make(env_id, num_envs=3, seed=11, **kw) as env:

        def rollout(actions):
            first = env.reset()

            def body(c, a):
                return c, env.step(a)

            return first, jax.lax.scan(body, 0, actions)[1]

        first, steps = jax.jit(rollout)(jnp.asarray(acts))
        assert_same(first, ref[0])
        for i in range(len(acts)):
            assert_same(index(steps, i), ref[i + 1])
        assert (env._spec.masks is None) == (ref[0].masks is None)


# ------------------------------------------------- 2. repeated dispatches


def test_jax_repeated_dispatches_and_a_recompile_keep_program_order():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 5, {}, 12)
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=5) as env:
        env.reset()
        f = jax.jit(lambda a: env.step(a))
        outs = [f(jnp.asarray(acts[i])) for i in range(6)]  # no block in between
        # The same body under a new input shape: a recompile mid-stream.
        g = jax.jit(lambda a, pad: (env.step(a), pad.sum()))
        for i, pad in zip(range(6, 12), [1, 2, 1, 3, 2, 1], strict=True):
            outs.append(g(jnp.asarray(acts[i]), jnp.zeros(pad))[0])
        jax.effects_barrier()
        for i, o in enumerate(outs):
            assert_same(o, ref[i + 1])


# ------------------------------------------------------ 3. across handles


def test_jax_two_handles_interleave_in_program_order():
    acts_a, ref_a = native_rollout("tessera/CartPole-v1", 2, 1, {}, 8)
    acts_b, ref_b = native_rollout("tessera/CartPole-v1", 2, 2, {}, 8)
    a = tjax.make("tessera/CartPole-v1", num_envs=2, seed=1)
    b = tjax.make("tessera/CartPole-v1", num_envs=2, seed=2)
    log = []
    for name, h in (("a", a), ("b", b)):
        inner = h._host_step

        def logged(x, inner=inner, name=name):
            log.append(name)
            return inner(x)

        h._host_step = logged
    a.reset()
    b.reset()

    @jax.jit
    def pair(x, y):
        return a.step(x), b.step(y), a.step(x), b.step(y)

    sa1, sb1, sa2, sb2 = pair(jnp.asarray(acts_a[0]), jnp.asarray(acts_b[0]))
    assert_same(sa1, ref_a[1])
    assert_same(sb1, ref_b[1])
    assert_same(sa2, ref_a[2])
    assert_same(sb2, ref_b[2])

    def body(c, xy):
        return c, (b.step(xy[1]), a.step(xy[0]))

    _, (sb, sa) = jax.jit(lambda xs, ys: jax.lax.scan(body, 0, (xs, ys)))(jnp.asarray(acts_a[2:6]), jnp.asarray(acts_b[2:6]))
    jax.effects_barrier()
    for i in range(4):
        assert_same(index(sa, i), ref_a[i + 3])
        assert_same(index(sb, i), ref_b[i + 3])
    assert log == ["a", "b", "a", "b"] + ["b", "a"] * 4
    a.close()
    b.close()


# --------------------------------------------------------- 4. unused result


def test_jax_unused_result_still_steps():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 3, {}, 2)
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=3) as env:
        env.reset()

        @jax.jit
        def discard(a):
            env.step(a)
            return jnp.int32(0)

        discard(jnp.asarray(acts[0]))
        assert_same(env.step(jnp.asarray(acts[1])), ref[2])


# ------------------------------------------------------------- 5. refusals


def test_jax_vmap_is_refused_and_leaves_the_pool_untouched():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 4, {}, 1)
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=4) as env:
        env.reset()
        batch = jnp.stack([jnp.asarray(acts[0])] * 3)
        for fn in (jax.vmap(env.step), jax.jit(jax.vmap(env.step))):
            with pytest.raises(ValueError, match=r"Cannot .vmap. ordered IO callback"):
                fn(batch)
        assert_same(env.step(jnp.asarray(acts[0])), ref[1])


MAPPED = """
import json, numpy as np, jax, jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import tessera_rl as t, tessera_rl.jax as tjax
assert jax.device_count() == 2
mesh = Mesh(np.array(jax.devices()), ("d",))
env = tjax.make("tessera/CartPole-v1", num_envs=2, seed=9)
calls = []
inner = env._host_step
def counted(x):
    calls.append(1)
    return inner(x)
env._host_step = counted
env.reset()
acts = jnp.ones((2, 1), jnp.int32)
pair = jnp.ones((2, 2, 1), jnp.int32)
cases = {
    "pmap step": lambda: jax.pmap(lambda a: env.step(a).rewards)(pair),
    "pmap reset": lambda: jax.pmap(lambda a: env.reset().rewards)(jnp.zeros(2)),
    "shard_map step": lambda: jax.jit(shard_map(lambda a: env.step(acts).rewards[:1], mesh=mesh, in_specs=P("d"), out_specs=P("d")))(jnp.zeros(2)),
    "shard_map reset": lambda: jax.jit(shard_map(lambda a: env.reset().rewards[:1], mesh=mesh, in_specs=P("d"), out_specs=P("d")))(jnp.zeros(2)),
    "jit in pmap": lambda: jax.pmap(jax.jit(lambda a: env.step(a).rewards))(pair),
    "jit in shard_map": lambda: jax.jit(shard_map(jax.jit(lambda a: env.step(acts).rewards[:1]), mesh=mesh, in_specs=P("d"), out_specs=P("d")))(jnp.zeros(2)),
    "named vmap": lambda: jax.vmap(lambda a: env.step(a).rewards, axis_name="i")(pair),
}
out = {}
for name, fn in cases.items():
    try:
        jax.block_until_ready(fn()); out[name] = "accepted"
    except t.ArgumentError as e:
        out[name] = "ArgumentError" if "mapped axis" in str(e) else "other: " + str(e)
    except Exception as e:
        out[name] = type(e).__name__ + ": " + str(e)[:100]
refused_calls = len(calls)
ref = t.make("tessera/CartPole-v1", num_envs=2, seed=9, outputs="owned"); ref.reset()
same = np.asarray(env.step(acts).observations["obs"]).tobytes() == ref.step(np.ones((2, 1), np.int32)).observations["obs"].tobytes()
calls.clear()
xs = jax.device_put(jnp.ones((2, 1), jnp.int32), NamedSharding(mesh, P("d")))
r2 = ref.step(np.ones((2, 1), np.int32))
sh = jax.jit(lambda a: env.step(a))(xs); jax.effects_barrier()
sharded_same = np.asarray(sh.observations["obs"]).tobytes() == r2.observations["obs"].tobytes()
print(json.dumps({"cases": out, "refused_calls": refused_calls, "untouched": same, "sharded_calls": len(calls), "sharded_same": sharded_same}))
"""


def test_jax_mapped_axes_are_refused_and_sharded_jit_steps_once():
    rc, out, err = run_py(MAPPED, devices=2)
    assert rc == 0, err[-3000:]
    r = json.loads(out.strip().splitlines()[-1])
    assert r["cases"] == {k: "ArgumentError" for k in r["cases"]}, r["cases"]
    assert len(r["cases"]) == 7
    assert r["refused_calls"] == 0 and r["untouched"]
    assert r["sharded_calls"] == 1 and r["sharded_same"]


class NotAMapping:
    axis_sizes = ["d"]


@pytest.mark.parametrize("broken", ["missing", "not-a-mapping"])
def test_jax_axis_env_check_fails_closed(monkeypatch, broken):
    from jax._src import core as jcore

    if broken == "missing":
        monkeypatch.delattr(jcore, "get_axis_env")
    else:
        monkeypatch.setattr(jcore, "get_axis_env", lambda: NotAMapping())
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=0) as env:
        with pytest.raises(t.ArgumentError, match="axis environment"):
            env.reset()
        with pytest.raises(t.ArgumentError, match="axis environment"):
            env.step(jnp.ones((1, 1), jnp.int32))


# ------------------------------------------------------------ 6. no aliasing


def test_jax_outputs_do_not_alias_engine_buffers():
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=6) as env:
        first = env.reset()
        kept = np.asarray(first.observations["obs"]).copy()
        for _ in range(5):
            env.step(jnp.ones((2, 1), jnp.int32))
        assert np.asarray(first.observations["obs"]).tobytes() == kept.tobytes()
        # The callback's own arrays sit on fresh bytes, never a native view.
        host = env._host_step(np.ones((2, 1), np.int32))
        for leaf in jax.tree.leaves(host):
            base = leaf
            while isinstance(base, (np.ndarray, memoryview)):
                base = base.obj if isinstance(base, memoryview) else base.base
            # Owned outputs: a fresh bytearray per call, never a native view.
            assert isinstance(base, (bytes, bytearray)), type(base)


# ------------------------------------------------- 7. flags and auto reset


@pytest.mark.parametrize("mode", ["next_step", "same_step"])
def test_jax_auto_reset_modes_match_native(mode):
    acts, ref = native_rollout("tessera/CartPole-v1", 1, 2, {"auto_reset": mode}, 60)
    assert any(bool(s.episode_terminated[0]) for s in ref)
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=2, auto_reset=mode) as env:
        env.reset()
        steps = jax.jit(lambda xs: jax.lax.scan(lambda c, a: (c, env.step(a)), 0, xs)[1])(jnp.asarray(acts))
        for i in range(len(acts)):
            assert_same(index(steps, i), ref[i + 1])


def test_jax_disabled_auto_reset_reports_pending_reset():
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=2, auto_reset="disabled") as env:
        env.reset()
        one = jnp.ones((1, 1), jnp.int32)
        for _ in range(200):
            if bool(env.step(one).episode_terminated[0]):
                break
        else:
            pytest.fail("CartPole did not terminate")
        with pytest.raises(JaxRuntimeError):
            env.step(one)
        assert isinstance(env.last_error, t.PendingResetError)
        env.reset()
        env.step(one)


# ------------------------------------------------------ 8. actions, seeds


def test_jax_actions_must_be_exact_int32():
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=0) as env:
        env.reset()
        for bad in (np.ones((2, 1), np.uint32), np.ones((2, 1), np.int64), np.ones((2, 1), np.float32), [[1], [1]]):
            with pytest.raises(t.ActionError, match="int32"):
                env.step(bad)
        with pytest.raises(t.ActionError, match="shape"):
            env.step(jnp.ones((2, 2), jnp.int32))
        # With x64 off (the default) JAX itself canonicalises an int64 input
        # to int32 when it becomes a JAX array: recorded, not ours.
        assert not jax.config.jax_enable_x64
        assert jnp.asarray(np.ones((2, 1), np.int64)).dtype == jnp.int32
        with jax.enable_x64(True):
            with pytest.raises(t.ActionError, match="int32"):
                env.step(jnp.ones((2, 1), jnp.int64))


def test_jax_out_of_range_actions_fail_in_the_callback_without_touching_the_pool():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 8, {}, 1)
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=8) as env:
        env.reset()
        with pytest.raises(JaxRuntimeError):
            env.step(jnp.full((2, 1), 7, jnp.int32))
        assert isinstance(env.last_error, t.ActionError)
        assert_same(env.step(jnp.asarray(acts[0])), ref[1])


@pytest.mark.parametrize(
    ("make_seed", "value"),
    [
        (lambda: 2**40 + 5, 2**40 + 5),
        (lambda: jnp.uint32(3_000_000_000), 3_000_000_000),
        (lambda: jnp.array([2**8 + 1, 7], jnp.uint32), ((2**8 + 1) << 32) | 7),
        (lambda: 2**64 - 1, 2**64 - 1),
    ],
    ids=["static-64bit", "traced-uint32", "traced-pair", "static-max"],
)
def test_jax_seeds_match_native(make_seed, value):
    ref = native("tessera/CartPole-v1", 2, 0)
    want = ref.reset(value)
    with tjax.make("tessera/CartPole-v1", num_envs=2, seed=0) as env:
        assert_same(env.reset(make_seed()), want)
        assert_same(jax.jit(lambda: env.reset(make_seed()))(), want)


def test_jax_static_seeds_share_one_callback(monkeypatch):
    callbacks = []
    real = tjax.io_callback

    def recording(cb, *a, **k):
        callbacks.append(cb)
        return real(cb, *a, **k)

    monkeypatch.setattr(tjax, "io_callback", recording)
    ref = native("tessera/CartPole-v1", 1, 0)
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=0) as env:
        for seed in (1, 2**33 + 1, 12345):
            assert_same(env.reset(seed), ref.reset(seed))
    assert len(callbacks) == 3 and all(cb == callbacks[0] for cb in callbacks)


def test_jax_bad_seeds_are_refused():
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=0) as env:
        for bad in (jnp.int32(3), jnp.array([1, 2, 3], jnp.uint32), np.float32(1)):
            with pytest.raises(t.ArgumentError, match="uint32"):
                env.reset(bad)
        for bad in (-1, 2**64):
            with pytest.raises(t.ArgumentError, match="64-bit"):
                env.reset(bad)
        with pytest.raises(TypeError):
            env.reset(True)


# ---------------------------------------------------- 9. errors and handles


def test_jax_a_failure_aborts_the_computation_and_spreads_no_further():
    acts, ref_b = native_rollout("tessera/CartPole-v1", 1, 21, {}, 3)
    _, ref_c = native_rollout("tessera/CartPole-v1", 1, 22, {}, 3)
    a = tjax.make("tessera/CartPole-v1", num_envs=1, seed=20)
    b = tjax.make("tessera/CartPole-v1", num_envs=1, seed=21)
    c = tjax.make("tessera/CartPole-v1", num_envs=1, seed=22)
    for h in (a, b, c):
        h.reset()
    good, bad = jnp.ones((1, 1), jnp.int32), jnp.full((1, 1), 9, jnp.int32)

    @jax.jit
    def mixed(x, y):
        return c.step(x), a.step(y), b.step(x)

    with pytest.raises(JaxRuntimeError):
        jax.block_until_ready(mixed(good, bad))
    assert isinstance(a.last_error, t.ActionError)
    assert b.last_error is None and c.last_error is None
    # c was ordered before the failure and advanced; b after it, and did not.
    assert_same(c.step(good), ref_c[2])
    assert_same(b.step(good), ref_b[1])
    # The runtime token is not left failed.
    out = jax.jit(lambda x: (a.step(x), b.step(x)))(good)
    jax.effects_barrier()
    assert_same(out[1], ref_b[2])
    assert isinstance(a.last_error, t.ActionError)  # a success does not clear it
    a.clear_error()
    assert a.last_error is None
    for h in (a, b, c):
        h.close()


# ---------------------------------------------------------------- 10. shutdown


def test_jax_close_waits_for_dispatched_work_and_is_idempotent():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 30, {}, 8)
    env = tjax.make("tessera/CartPole-v1", num_envs=2, seed=30)
    env.reset()
    f = jax.jit(lambda a: env.step(a))
    outs = [f(jnp.asarray(acts[i])) for i in range(8)]
    env.close()
    assert env.closed
    for i, o in enumerate(outs):
        assert_same(o, ref[i + 1])
    env.close()
    with pytest.raises(JaxRuntimeError):
        env.step(jnp.ones((2, 1), jnp.int32))
    assert env.last_error is not None


def test_jax_finalizer_closes_without_a_barrier(monkeypatch):
    barriers = []
    real = jax.effects_barrier
    monkeypatch.setattr(jax, "effects_barrier", lambda: barriers.append(1) or real())
    env = tjax.make("tessera/CartPole-v1", num_envs=1, seed=0)
    env.reset()
    pool = env._pool
    ref = weakref.ref(env)
    del env
    # Even an eager call leaves the callback, and so the handle, in JAX's
    # caches: collection is not a reliable shutdown (use close).
    gc.collect()
    assert ref() is not None
    jax.clear_caches()
    gc.collect()
    assert ref() is None
    assert barriers == []
    with pytest.raises(t.TesseraError):
        pool.step(np.ones((1, 1), np.int32))


def test_jax_a_captured_handle_lives_as_long_as_the_jitted_function():
    env = tjax.make("tessera/CartPole-v1", num_envs=1, seed=0)
    env.reset()
    f = jax.jit(lambda a: env.step(a))
    f(jnp.ones((1, 1), jnp.int32))
    jax.effects_barrier()
    ref = weakref.ref(env)
    del env
    gc.collect()
    assert ref() is not None
    del f
    jax.clear_caches()
    gc.collect()
    assert ref() is None


# --------------------------------------------------------------- 11. gradients


def test_jax_gradients_treat_env_outputs_as_constants():
    ref = native("tessera/CartPole-v1", 3, 40)
    ref.reset()
    rewards = ref.step(np.ones((3, 1), np.int32)).rewards
    with tjax.make("tessera/CartPole-v1", num_envs=3, seed=40) as env:
        env.reset()

        def loss(p):
            s = env.step(jnp.ones((3, 1), jnp.int32))
            return (s.rewards * p).sum()

        g = jax.grad(loss)(jnp.ones((3, 1)))
        assert np.asarray(g).tobytes() == rewards.tobytes()


def test_jax_float_callback_inputs_have_no_derivative():
    with tjax.make("tessera/CartPole-v1", num_envs=1, seed=0) as env:
        env.reset()

        def through(x):
            return io_callback(env._host_step, env._spec, x, ordered=True).rewards.sum()

        with pytest.raises(ValueError, match="IO callbacks do not support JVP"):
            jax.grad(through)(jnp.ones((1, 1)))
        with pytest.raises(ValueError, match="IO callbacks do not support JVP"):
            jax.jvp(through, (jnp.ones((1, 1)),), (jnp.ones((1, 1)),))


# ----------------------------------------------------- 12, 13. import guards


def test_jax_version_guard_refuses_and_override_warns():
    with pytest.raises(ImportError, match="jax==0.10.2"):
        tjax._check_versions("0.10.3", "0.10.2", {})
    with pytest.raises(ImportError, match="jaxlib==0.10.2"):
        tjax._check_versions("0.10.2", "0.10.1", {})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tjax._check_versions("0.10.3", "0.10.3", {tjax.UNPINNED_ENV: "1"})
    assert [x.category for x in w] == [RuntimeWarning]
    tjax._check_versions("0.10.2", "0.10.2", {})


def test_jax_import_with_another_version_fails_and_override_keeps_the_axis_check():
    code = """
    import jax, jaxlib
    jax.__version__ = "0.0.1"
    try:
        import tessera_rl.jax
    except ImportError as e:
        print("refused" if "jax==0.10.2" in str(e) else "other")
    """
    rc, out, err = run_py(code)
    assert rc == 0 and out.strip() == "refused", err
    code = """
    import os, warnings
    os.environ["TESSERA_JAX_UNPINNED"] = "1"
    import jax
    jax.__version__ = "0.0.1"
    from jax._src import core as jcore
    del jcore.get_axis_env
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        try:
            import tessera_rl.jax
        except ImportError as e:
            print("refused" if "axis environment" in str(e) else "other")
    """
    rc, out, err = run_py(code)
    assert rc == 0 and out.strip() == "refused", err


def test_jax_tessera_imports_without_jax_and_the_handle_names_the_extra():
    code = """
    import sys
    sys.modules["jax"] = None
    import tessera_rl
    assert "jax" not in [m for m in sys.modules if sys.modules[m] is not None]
    try:
        import tessera_rl.jax
    except ImportError as e:
        print("refused" if "`jax` extra" in str(e) else "other: " + str(e))
    """
    rc, out, err = run_py(code)
    assert rc == 0 and out.strip() == "refused", err


def test_jax_split_phase_modes_and_outputs_are_refused():
    for kw in ({"mode": "pipelined"}, {"mode": "async", "batch_size": 1}, {"outputs": "borrowed"}, {"batch_size": 1}):
        with pytest.raises(t.ArgumentError):
            tjax.make("tessera/CartPole-v1", num_envs=2, seed=0, **kw)

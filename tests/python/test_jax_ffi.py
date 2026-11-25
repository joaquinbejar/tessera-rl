# SPDX-License-Identifier: MIT OR Apache-2.0
"""tessera_rl.jax.ffi_handle: the experimental XLA FFI token path (ADR-0095),
against the native API. The ADR-0031 design tests 2, 3, 4, 9 and 10 adapted
to the token API, plus token refusals, equivalence and the refusals of the
handle. CPU, x86-64 only (elsewhere the handle is refused)."""

import platform
import subprocess
import sys
import textwrap

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytestmark = pytest.mark.jax

import jax.numpy as jnp  # noqa: E402

import tessera_rl as t  # noqa: E402
import tessera_rl.jax as tjax  # noqa: E402

JaxRuntimeError = jax.errors.JaxRuntimeError
X86 = platform.machine().lower() in ("x86_64", "amd64")
needs_x86 = pytest.mark.skipif(not X86, reason="the XLA FFI path is x86-64 only for now")
FIELDS = [f for f in t.Step._fields if f != "observations"]


def native(env_id, n, seed, **kw):
    return t.make(env_id, num_envs=n, seed=seed, outputs="owned", workers=0, **kw)


def assert_same(jstep, nstep):
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


def legal(step):
    if step.masks is None:
        return np.zeros(step.rewards.shape, np.int32)
    return np.argmax(np.asarray(step.masks), axis=-1).astype(np.int32)


def native_rollout(env_id, n, seed, kw, steps):
    env = native(env_id, n, seed, **kw)
    s = env.reset()
    out, acts = [s], []
    for _ in range(steps):
        a = legal(s) if env_id != "tessera/CartPole-v1" else np.ones((n, env.agents), np.int32)
        acts.append(a)
        s = env.step(a)
        out.append(s)
    env.close()
    return np.stack(acts), out


CASES = [
    ("tessera/CartPole-v1", {}),
    ("tessera/Tag-v1", {"pursuers": 2, "evaders": 2, "radius": 1}),
]


@needs_x86
@pytest.mark.parametrize(("env_id", "kw"), CASES, ids=["cartpole", "tag"])
def test_jax_ffi_scan_trajectory_equals_native_bit_for_bit(env_id, kw):
    acts, ref = native_rollout(env_id, 3, 11, kw, 20)
    env = native(env_id, 3, 11, **kw)
    with tjax.ffi_handle(env) as h:

        def rollout(token, actions):
            token, first = h.reset(token)

            def body(tok, a):
                tok, s = h.step(tok, a)
                return tok, s

            token, steps = jax.lax.scan(body, token, actions)
            return token, first, steps

        token, first, steps = jax.jit(rollout)(h.token(), jnp.asarray(acts))
        assert int(token) == 21
        assert_same(first, ref[0])
        for i in range(len(acts)):
            assert_same(index(steps, i), ref[i + 1])
    env.close()


@needs_x86
def test_jax_ffi_repeated_dispatches_and_a_recompile_keep_program_order():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 5, {}, 12)
    env = native("tessera/CartPole-v1", 2, 5)
    with tjax.ffi_handle(env) as h:
        tok, _ = h.reset(h.token())
        f = jax.jit(h.step)
        outs = []
        for i in range(6):  # no block in between
            tok, s = f(tok, jnp.asarray(acts[i]))
            outs.append(s)
        g = jax.jit(lambda tok, a, pad: (h.step(tok, a), pad.sum()))
        for i, pad in zip(range(6, 12), [1, 2, 1, 3, 2, 1], strict=True):
            (tok, s), _ = g(tok, jnp.asarray(acts[i]), jnp.zeros(pad))
            outs.append(s)
        jax.effects_barrier()
        for i, o in enumerate(outs):
            assert_same(o, ref[i + 1])
    env.close()


@needs_x86
def test_jax_ffi_two_handles_interleave_each_in_its_own_order():
    acts_a, ref_a = native_rollout("tessera/CartPole-v1", 2, 1, {}, 6)
    acts_b, ref_b = native_rollout("tessera/CartPole-v1", 2, 2, {}, 6)
    ea, eb = native("tessera/CartPole-v1", 2, 1), native("tessera/CartPole-v1", 2, 2)
    a, b = tjax.ffi_handle(ea), tjax.ffi_handle(eb)
    ta, _ = a.reset(a.token())
    tb, _ = b.reset(b.token())

    def body(c, xy):
        ta, tb = c
        tb, sb = b.step(tb, xy[1])
        ta, sa = a.step(ta, xy[0])
        return (ta, tb), (sa, sb)

    (ta, tb), (sa, sb) = jax.jit(lambda c, xs, ys: jax.lax.scan(body, c, (xs, ys)))((ta, tb), jnp.asarray(acts_a), jnp.asarray(acts_b))
    jax.effects_barrier()
    for i in range(6):
        assert_same(index(sa, i), ref_a[i + 1])
        assert_same(index(sb, i), ref_b[i + 1])
    assert (int(ta), int(tb)) == (7, 7)
    a.close()
    b.close()
    ea.close()
    eb.close()


@needs_x86
def test_jax_ffi_unused_results_still_step():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 3, {}, 2)
    env = native("tessera/CartPole-v1", 2, 3)
    with tjax.ffi_handle(env) as h:
        h.reset(h.token())

        @jax.jit
        def discard(tok, a):
            h.step(tok, a)
            return jnp.int32(0)

        discard(h.token(), jnp.asarray(acts[0]))
        jax.effects_barrier()
        assert int(h.token()) == 2, "the discarded step ran"
        _, s = h.step(h.token(), jnp.asarray(acts[1]))
        assert_same(s, ref[2])
    env.close()


@needs_x86
def test_jax_ffi_stale_and_replayed_tokens_are_refused_without_stepping():
    acts, ref = native_rollout("tessera/CartPole-v1", 2, 4, {}, 2)
    env = native("tessera/CartPole-v1", 2, 4)
    with tjax.ffi_handle(env) as h:
        t0 = h.token()
        t1, _ = h.reset(t0)
        t2, _ = h.step(t1, jnp.asarray(acts[0]))
        for bad in (t1, t0, jnp.uint32(99)):  # replayed, stale, invented
            with pytest.raises(JaxRuntimeError):
                jax.block_until_ready(h.step(bad, jnp.asarray(acts[1])))
            assert isinstance(h.last_error, t.OrderError), h.last_error
            h.clear_error()
        _, s = h.step(t2, jnp.asarray(acts[1]))
        assert_same(s, ref[2])
    env.close()


@needs_x86
def test_jax_ffi_errors_stay_on_the_failing_handle():
    ea, eb = native("tessera/CartPole-v1", 2, 1), native("tessera/CartPole-v1", 2, 2)
    a, b = tjax.ffi_handle(ea), tjax.ffi_handle(eb)
    ta, _ = a.reset(a.token())
    tb, _ = b.reset(b.token())
    ones = jnp.ones((2, 1), jnp.int32)
    with pytest.raises(JaxRuntimeError):
        jax.block_until_ready(a.step(jnp.uint32(42), ones))
    tb, _ = b.step(tb, ones)
    ta, _ = a.step(ta, ones)
    jax.effects_barrier()
    assert isinstance(a.last_error, t.OrderError)
    assert b.last_error is None
    assert (int(ta), int(tb)) == (2, 2)
    for h in (a, b):
        h.close()
    ea.close()
    eb.close()


@needs_x86
def test_jax_ffi_close_waits_is_idempotent_and_refuses_later_calls():
    env = native("tessera/CartPole-v1", 2, 6)
    h = tjax.ffi_handle(env)
    tok, _ = h.reset(h.token())
    f = jax.jit(h.step)
    for _ in range(4):
        tok, _ = f(tok, jnp.ones((2, 1), jnp.int32))
    h.close()
    h.close()
    with pytest.raises(JaxRuntimeError):
        jax.block_until_ready(f(tok, jnp.ones((2, 1), jnp.int32)))
    # The Env stays open and usable.
    env.step(np.ones((2, 1), np.int32))
    env.close()


@needs_x86
def test_jax_ffi_shares_the_env_guard_and_fails_on_a_closed_env():
    env = native("tessera/CartPole-v1", 2, 7)
    h = tjax.ffi_handle(env)
    tok, _ = h.reset(h.token())
    env.close()
    with pytest.raises(JaxRuntimeError):
        jax.block_until_ready(h.step(tok, jnp.ones((2, 1), jnp.int32)))
    h.close()


@needs_x86
def test_jax_ffi_refuses_privileged_state_and_split_phase_modes():
    env = t.make("tessera/Leduc-v1", num_envs=1, privileged_state=True, workers=0)
    with pytest.raises(t.CapabilityError):
        tjax.ffi_handle(env)
    env.close()
    env = t.make("tessera/CartPole-v1", num_envs=4, mode="pipelined", workers=1, chunk_size=2)
    with pytest.raises(t.CapabilityError):
        tjax.ffi_handle(env)
    env.close()


@needs_x86
def test_jax_ffi_vmap_is_refused():
    env = native("tessera/CartPole-v1", 2, 8)
    with tjax.ffi_handle(env) as h:
        tok, _ = h.reset(h.token())
        with pytest.raises(Exception, match="vmap"):
            jax.vmap(lambda a: h.step(tok, a))(jnp.ones((3, 2, 1), jnp.int32))
        assert int(h.token()) == 1, "nothing stepped"
    env.close()


@pytest.mark.skipif(X86, reason="the refusal applies off x86-64")
def test_jax_ffi_is_refused_off_x86_64():
    env = native("tessera/CartPole-v1", 2, 9)
    with pytest.raises(t.CapabilityError, match="x86-64"):
        tjax.ffi_handle(env)
    env.close()


@needs_x86
def test_jax_ffi_deferred_fp_warning_is_emitted_exactly_once():
    """ADR-0095 with REVIEW-plan-B answer 1: a handler under non-default
    controls defers the once-per-process warning to the next Python call.
    Run in a fresh process so the process-wide warning has not fired."""
    code = """
        import warnings
        import numpy as np
        import jax, jax.numpy as jnp
        import tessera_rl as t, tessera_rl.jax as tjax
        from tessera_rl import _native
        env = t.make("tessera/CartPole-v1", num_envs=2, seed=1, outputs="owned", workers=0)
        h = tjax.ffi_handle(env)
        before = _native.fp_non_canonical_entries()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tok, _ = h.reset(h.token())
            for _ in range(3):
                tok, _ = h.step(tok, jnp.ones((2, 1), jnp.int32))
            jax.effects_barrier()
            ffi_entries = _native.fp_non_canonical_entries() - before
            for _ in range(3):
                env.step(np.ones((2, 1), np.int32))
        fp = [x for x in w if "floating-point controls" in str(x.message)]
        print(ffi_entries, len(fp))
    """
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    entries, warned = map(int, p.stdout.split())
    # XLA's CPU threads may or may not run under non-default controls: when
    # they did, the warning appears exactly once; when not, never.
    assert warned == (1 if entries > 0 else 0), (entries, warned)

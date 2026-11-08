# SPDX-License-Identifier: MIT OR Apache-2.0
"""JAX handles over native pools: the ordered callback path (SPEC 15.2,
ADR-0031; design ``docs/reviews/jax-ordered-callback-design.md`` r3).

``reset`` and ``step`` call ``jax.experimental.io_callback(..., ordered=True)``,
so they work eagerly, under ``jax.jit`` and inside ``jax.lax.scan``, and JAX
runs them in program order. ``jax.pure_callback`` is never used.

- **Copies.** The pool runs with ``outputs="owned"``: every result is fresh
  memory the engine never writes again, and JAX copies it into its own
  arrays. JAX never aliases an engine buffer.
- **One handle, one pool.** The handle owns its pool and does not expose it.
  Only ``mode="sync"`` is supported.
- **Refused:** ``vmap`` (by JAX: ``ValueError``), and ``pmap``, ``shard_map``
  or a named ``vmap`` (by the handle, at trace time: ``ArgumentError``),
  since those would step one pool several times per logical step, in no
  defined order. ``jit`` over sharded inputs is allowed: its callback runs
  once.
- **Ordering** across separate dispatches holds for one dispatching thread.
  Dispatching computations that use the same handles from several Python
  threads is unsupported.
- **Errors.** A native error inside a callback reaches the caller as
  ``JaxRuntimeError``, with its type lost; the original exception is kept
  in ``last_error``. The failing computation is aborted: callbacks ordered
  before the failure did step their pools, those after it did not run. So
  after a failed computation, ``reset`` every handle it used.
- **Gradients.** Callback inputs are integers only, so no tangent reaches
  the environment and its outputs act as constants in a loss.
- **Shutdown.** ``close`` (or the context manager) waits for dispatched work
  with ``jax.effects_barrier`` and then closes the pool. Collection closes
  the pool too, best effort and without waiting on JAX. But any call, even
  an eager one, leaves the callback and so the handle in JAX's caches until
  the jitted function is dropped or ``jax.clear_caches`` runs: call
  ``close``.
- **Floating point.** XLA's CPU threads may run with flush-to-zero set. The
  native boundary computes under canonical controls anyway (ADR-0041) and
  warns once per process. Under JAX that ``RuntimeWarning`` is expected and
  harmless: trajectories are bit-identical to the native API.

JAX and jaxlib are pinned exactly (the ``jax`` extra). Another version raises
``ImportError``; ``TESSERA_JAX_UNPINNED=1`` overrides that check at your own
risk, with a ``RuntimeWarning``.
"""

from __future__ import annotations

import os
import warnings
import weakref
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from tessera_rl import _env, _native

try:
    import jax
    import jaxlib
    from jax.experimental import io_callback
except ImportError as e:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "tessera_rl.jax needs JAX: install the `jax` extra (jax==0.10.2, jaxlib==0.10.2)"
    ) from e

#: The exact JAX and jaxlib version the ordering guarantees are tested on.
JAX_VERSION = "0.10.2"
#: The environment variable that overrides the version check.
UNPINNED_ENV = "TESSERA_JAX_UNPINNED"

_U32 = np.dtype(np.uint32)
_I32 = np.dtype(np.int32)


def _check_versions(jax_version: str, jaxlib_version: str, environ: Mapping[str, str]) -> None:
    if jax_version == JAX_VERSION and jaxlib_version == JAX_VERSION:
        return
    found = f"jax {jax_version}, jaxlib {jaxlib_version}"
    if environ.get(UNPINNED_ENV) == "1":
        warnings.warn(
            f"tessera_rl.jax is tested on jax and jaxlib {JAX_VERSION}, found {found}; "
            f"{UNPINNED_ENV}=1 overrides the check and ordering is not guaranteed",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    raise ImportError(
        f"tessera_rl.jax needs jax=={JAX_VERSION} and jaxlib=={JAX_VERSION} (the `jax` extra), "
        f"found {found}; set {UNPINNED_ENV}=1 to override at your own risk"
    )


_check_versions(jax.__version__, jaxlib.__version__, os.environ)


# ------------------------------------------------------------- mapped axes


def _axis_sizes() -> Mapping:
    """The named axes of the current trace. Internal JAX API, acceptable only
    under the exact pin; anything unexpected fails closed."""
    try:
        from jax._src import core as jcore

        sizes = jcore.get_axis_env().axis_sizes
    except Exception as e:
        raise _native.ArgumentError(
            f"tessera_rl.jax cannot read JAX's axis environment ({type(e).__name__}: {e}); refusing the call"
        ) from None
    if not isinstance(sizes, Mapping):
        raise _native.ArgumentError(
            f"tessera_rl.jax cannot read JAX's axis environment (got {type(sizes).__name__}); refusing the call"
        )
    return sizes


def _refuse_mapped(what: str) -> None:
    sizes = _axis_sizes()
    if sizes:
        names = ", ".join(sorted(map(str, sizes)))
        raise _native.ArgumentError(
            f"{what} under a mapped axis ({names}): pmap, shard_map and named vmap would step the pool "
            "once per device or shard, in no defined order; call it outside the map"
        )


# The API must exist at import, before any handle is made.
try:
    _axis_sizes()
except _native.ArgumentError as e:
    raise ImportError(str(e)) from None


# ------------------------------------------------------------------ handle


class Step(NamedTuple):
    """The outputs of one call as JAX arrays; fields as ``tessera_rl.Step``.

    ``masks`` is ``None`` for environments without action masks, decided at
    construction.
    """

    observations: dict[str, jax.Array]
    rewards: jax.Array
    agent_terminated: jax.Array
    agent_truncated: jax.Array
    alive: jax.Array
    episode_terminated: jax.Array
    episode_truncated: jax.Array
    masks: jax.Array | None
    acting: jax.Array


def _close_pool(native) -> None:
    """The finalizer: holds the native pool only, never the handle, and does
    not wait on JAX."""
    native.close()


class Env:
    """A JAX handle over one native pool (see the module notes).

    Use ``make``. ``reset(seed=None)`` and ``step(actions)`` return a
    ``Step`` of JAX arrays.
    """

    def __init__(self, pool: _env.Env):
        if pool.mode != "sync" or pool.outputs != "owned":
            raise _native.ArgumentError("tessera_rl.jax needs a sync pool with owned outputs")
        self._pool = pool
        self.num_envs: int = pool.num_envs
        self.agents: int = pool.agents
        self.action_choices: int = pool.action_choices
        self.turns: str = pool.turns
        #: The original exception of this handle's most recent failed
        #: callback; a success does not clear it (``clear_error`` does).
        self.last_error: BaseException | None = None
        self._closed = False
        n, a = self.num_envs, self.agents

        def spec(shape, dtype):
            return jax.ShapeDtypeStruct(shape, np.dtype(dtype))

        self._spec = Step(
            observations={name: spec((n, a, *shape), dtype) for name, dtype, shape in pool._leaves},
            rewards=spec((n, a), np.float32),
            agent_terminated=spec((n, a), np.bool_),
            agent_truncated=spec((n, a), np.bool_),
            alive=spec((n, a), np.bool_),
            episode_terminated=spec((n,), np.bool_),
            episode_truncated=spec((n,), np.bool_),
            masks=spec((n, a, pool._mask_len), np.bool_) if pool._mask_len else None,
            acting=spec((n, a), np.bool_),
        )
        self._finalizer = weakref.finalize(self, _close_pool, pool._native)

    # -------------------------------------------------------------- calls

    def reset(self, seed=None) -> Step:
        """Resets every instance, ordered with every other call.

        ``seed``: ``None`` keeps the seed base; a Python ``int`` is static
        (the full 64-bit range; a new value retraces); a traced ``uint32``
        scalar is zero-extended; a traced ``uint32[2]`` pair ``(hi, lo)``
        is ``hi << 32 | lo``. Other dtypes, signed included, raise
        ``ArgumentError``.
        """
        _refuse_mapped("reset")
        if seed is None:
            return io_callback(self._host_reset_keep, self._spec, ordered=True)
        if isinstance(seed, bool):
            raise TypeError("seed must be an integer, not a bool")
        if isinstance(seed, int):
            value = _env._index(seed, "seed")
            if not 0 <= value < 2**64:
                raise _native.ArgumentError(f"seed {value} is outside the 64-bit seed range")
            # As a (hi, lo) operand of the one traced-seed callback: a new
            # closure per seed would grow JAX's callback caches without bound.
            pair = np.array([value >> 32, value & 0xFFFFFFFF], dtype=_U32)
            return io_callback(self._host_reset_traced, self._spec, pair, ordered=True)
        dtype = getattr(seed, "dtype", None)
        shape = tuple(getattr(seed, "shape", ()) or ())
        if dtype is None or np.dtype(dtype) != _U32 or shape not in ((), (2,)):
            raise _native.ArgumentError(
                "a traced seed must be a uint32 scalar or a uint32[2] (hi, lo) pair, "
                f"got {getattr(dtype, 'name', type(seed).__name__)}{list(shape)}"
            )
        return io_callback(self._host_reset_traced, self._spec, seed, ordered=True)

    def step(self, actions) -> Step:
        """One step of every instance, ordered with every other call.

        ``actions`` must be exactly ``int32`` with shape ``(num_envs,
        agents)``; nothing is cast, so out-of-range values of wider types
        cannot wrap into valid actions. Value checks run in the callback, at
        the native validation boundary, before the pool is touched.
        """
        _refuse_mapped("step")
        dtype = getattr(actions, "dtype", None)
        shape = tuple(getattr(actions, "shape", ()) or ())
        expected = (self.num_envs, self.agents)
        if dtype is None or np.dtype(dtype) != _I32:
            raise _native.ActionError(
                f"actions must be int32 {list(expected)}, got "
                f"{getattr(dtype, 'name', type(actions).__name__)}; convert explicitly"
            )
        if shape != expected:
            raise _native.ActionError(f"actions must have shape {list(expected)}, got {list(shape)}")
        return io_callback(self._host_step, self._spec, actions, ordered=True)

    # --------------------------------------------------- host-side callbacks

    def _guarded(self, call):
        try:
            return self._to_step(call())
        except BaseException as e:
            self.last_error = e
            raise

    def _host_reset_keep(self):
        return self._guarded(lambda: self._pool.reset())

    def _host_reset(self, seed: int):
        return self._guarded(lambda: self._pool.reset(seed))

    def _host_reset_traced(self, seed):
        v = np.asarray(seed)
        value = int(v) if v.shape == () else (int(v[0]) << 32) | int(v[1])
        return self._host_reset(value)

    def _host_step(self, actions):
        return self._guarded(lambda: self._pool.step(np.asarray(actions)))

    @staticmethod
    def _to_step(out: _env.Step) -> Step:
        # Owned outputs: arrays over fresh memory per call, never an engine view.
        return Step(**out._asdict())

    # ----------------------------------------------------------- lifecycle

    def clear_error(self) -> None:
        """Resets ``last_error`` to ``None``."""
        self.last_error = None

    @property
    def is_poisoned(self) -> bool:
        return self._pool.is_poisoned

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Waits for dispatched work (``jax.effects_barrier``), then closes
        the pool; idempotent. Later calls raise the native closed error."""
        if self._closed:
            return
        jax.effects_barrier()
        self._closed = True
        self._finalizer()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make(env_id: str, num_envs: int = 1, seed: int = 0, **kwargs) -> Env:
    """A JAX handle over a new pool of ``env_id``.

    Takes the pool and config options of ``tessera_rl.make``, except
    ``outputs`` (JAX always receives copies) and split-phase modes.
    """
    if "outputs" in kwargs:
        raise _native.ArgumentError("tessera_rl.jax always copies outputs; `outputs` is not an option")
    mode = kwargs.pop("mode", None)
    if mode not in (None, "sync"):
        raise _native.ArgumentError(f"tessera_rl.jax supports mode='sync' only, got {mode!r}")
    for key in ("batch_size", "gather"):
        if key in kwargs:
            raise _native.ArgumentError(f"tessera_rl.jax supports mode='sync' only; `{key}` is not an option")
    return Env(_env.make(env_id, num_envs, seed, outputs="owned", **kwargs))


__all__ = ["JAX_VERSION", "Env", "Step", "make"]

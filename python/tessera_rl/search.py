# SPDX-License-Identifier: MIT OR Apache-2.0
"""PUCT search over a Tessera environment (ADR-0043), with a Python
evaluator.

``Search.run(evaluator)`` performs one search call. The evaluator is called
once per batch of leaves as ``evaluator(obs, legal, hashes)``:

- ``obs``: the observation of the player to move at each leaf, an array of
  shape ``(n, *obs_shape)`` (a dict of arrays for environments with several
  observation leaves);
- ``legal``: ``(n, actions)`` bool, that player's legal actions;
- ``hashes``: ``(n,)`` uint64, each leaf's record hash (for canned or
  table evaluators).

It returns ``(priors, values)``: priors ``(n, actions)`` over **all** actions,
each row summing to 1, and values ``(n,)`` in ``[-1, 1]`` from the view of
the player to move at the leaf, both C-contiguous float64 in native byte
order (the wire format: nothing is cast or broadcast, see
``_wire_problem``). The engine validates them, drops the illegal priors
and renormalizes over the legal ones.

The inputs are fresh arrays owned by Python for every batch: an evaluator
may keep them, and they never change afterwards.

The record hashes identify perfect-information states only. They are
exposed for Connect Four; an imperfect-information environment would leak
hidden state through them, and is not searchable from here without
``Determinize`` (ADR-0035).

A call is all or nothing: if the evaluator raises an ``Exception`` or
returns an invalid response, the call raises ``EvaluatorError`` (with the
evaluator's own exception as its ``__cause__``), and the tree and the call
counter are as before the call. A ``BaseException`` that is not an
``Exception`` (``KeyboardInterrupt``, ``SystemExit``) rolls the call back
the same way and is raised again unchanged.

The evaluator runs under the caller's floating-point controls (ADR-0041:
it is called through a callout); the search arithmetic, including the
validation of each response, runs under the canonical ones, and the
caller's controls are restored when ``run`` returns or raises. The
interpreter stays attached for the whole call, which holds the search
object: a nested or concurrent call on it raises ``ConcurrencyError``.
Pending signals are checked before every batch, including batches whose
leaves are all terminal and call no evaluator: a ``KeyboardInterrupt``
from a signal handler rolls the call back and is raised unchanged. Building the evaluator's
arguments and reading its response allocate: the engine's zero-allocation
guarantee covers the search itself, not the Python boundary.
"""

from __future__ import annotations

import numpy as np

from tessera_rl import _native
from tessera_rl._env import make


def _wire_problem(response, n: int, actions: int) -> str | None:
    """Why ``response`` is not the wire format, or ``None``.

    The wire format is strict: a 2-tuple of buffer-protocol objects, each C
    contiguous, of native-byte-order float64 (format ``"d"``), with shapes
    exactly ``(n, actions)`` and ``(n,)``. Nothing is converted, cast or
    broadcast: a float32 or complex array, a list, a transposed view or a
    wrong shape is rejected (``EvaluatorError``).
    """
    if not isinstance(response, tuple) or len(response) != 2:
        return "the evaluator must return a (priors, values) tuple"
    for name, obj, shape in (("priors", response[0], (n, actions)), ("values", response[1], (n,))):
        try:
            view = memoryview(obj)
        except TypeError:
            return f"{name} must support the buffer protocol (a NumPy array), not {type(obj).__name__}"
        if view.format != "d":
            return f"{name} must be float64 in native byte order, not format {view.format!r}"
        if not view.c_contiguous:
            return f"{name} must be C-contiguous"
        if tuple(view.shape) != shape:
            return f"{name} must have shape {shape}, not {tuple(view.shape)}"
    return None


class Search:
    """A PUCT search. ``env_id`` is the environment (``tessera/Connect4-v1``);
    options: ``capacity`` (nodes, default 4096), ``batch`` (children stepped
    per batch, default 8), ``eviction`` (``"strict"`` or ``"oldest"``),
    ``c_puct``, ``fpu``, ``virtual_loss``, ``batch_size``, ``simulations``,
    ``noise_alpha`` with ``noise_epsilon`` (root Dirichlet noise), ``seed``.
    The search starts at a fresh game; ``set_root`` moves it."""

    def __init__(self, env_id: str = "tessera/Connect4-v1", **options):
        self._native = _native.make_search(env_id=env_id, **options)
        layout = self._native.layout()
        self.actions: int = layout["actions"]
        self._obs_bytes: int = layout["obs_bytes"]
        probe = make(env_id)
        self._leaves = list(probe._leaves)
        probe.close()

    def set_root(self, record: bytes) -> None:
        """Replaces the tree by the root ``record`` (``Env.snapshot`` bytes)."""
        self._native.set_root(bytes(record))

    def run(self, evaluator) -> None:
        """One search call with ``evaluator`` (see the module docs)."""
        raised = []
        actions = self.actions
        leaves = self._leaves
        obs_bytes = self._obs_bytes

        def call(obs: bytes, legal: bytes, hashes: bytes, n: int):
            try:
                # Each row is one leaf's observation leaves, concatenated.
                rows = np.frombuffer(obs, dtype=np.uint8).reshape((n, obs_bytes))
                arrays = {}
                at = 0
                for name, dtype, shape in leaves:
                    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    leaf = np.ascontiguousarray(rows[:, at : at + size])
                    arrays[name] = leaf.view(dtype).reshape((n, *shape))
                    at += size
                o = next(iter(arrays.values())) if len(arrays) == 1 else arrays
                m = np.frombuffer(legal, dtype=np.bool_).reshape((n, actions))
                h = np.frombuffer(hashes, dtype="<u8")
                response = evaluator(o, m, h)
                problem = _wire_problem(response, n, actions)
                if problem is not None:
                    # A malformed response: native reports it as Shape.
                    return (problem.encode(),)
                priors, values = response
                return (bytes(memoryview(priors)), bytes(memoryview(values)))
            except Exception as exc:
                raised.append(exc)
                raise

        try:
            self._native.run(call)
        except Exception as exc:
            if raised and exc is raised[0]:
                # The type name only: formatting the exception runs its
                # ``__str__``, which may itself raise.
                raise _native.EvaluatorError(f"the evaluator raised {type(exc).__name__}") from exc
            raise

    def root_visits(self) -> np.ndarray:
        """The visit count of every root action."""
        return np.asarray(self._native.root_visits(), dtype=np.uint32)

    def select_action(self, temperature: float = 0.0) -> int:
        """The most visited action at 0, else a draw proportional to
        ``N^(1/T)`` from the last call's search stream."""
        return self._native.select_action(float(temperature))

    @property
    def calls(self) -> int:
        """Completed search calls."""
        return self._native.calls()

    def tree_hash(self) -> int:
        """A digest of the whole tree (for tests of determinism)."""
        return self._native.tree_hash()

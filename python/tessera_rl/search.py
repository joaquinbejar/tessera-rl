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
leaves are all terminal and call no evaluator. That poll is not an
evaluator failure: whatever a signal handler raises, ``KeyboardInterrupt``
or an ``Exception`` subclass alike, rolls the call back and is raised
unchanged. Building the evaluator's arguments and reading its response
allocate: the engine's zero-allocation guarantee covers the search itself,
not the Python boundary.
"""

from __future__ import annotations

from typing import NamedTuple

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


def _bridge(evaluator, actions: int, leaves, obs_bytes: int, raised: list):
    """The native callback around ``evaluator``: it decodes one batch,
    calls the evaluator and encodes its response, keeping the evaluator's
    own exception in ``raised``."""

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

    return call


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
        call = _bridge(evaluator, self.actions, self._leaves, self._obs_bytes, raised)
        try:
            self._native.run(call)
        except Exception as exc:
            if raised and exc is raised[0]:
                # A fixed message: formatting the exception, or even naming
                # its type, can run user code that raises. The cause
                # carries both.
                raise _native.EvaluatorError("the evaluator raised") from exc
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

    @property
    def is_privileged(self) -> bool:
        """Whether the search reads the full state of an
        imperfect-information environment, hidden cards included. Always
        ``False`` for perfect-information environments."""
        return self._native.is_privileged()


class GroupFailure(NamedTuple):
    """The last failed group call: the slot it belongs to (``None`` for an
    evaluator failure or an interruption), its message, and whether the
    exception raised was the user's own (the evaluator's or a signal
    handler's), which is never modified."""

    slot: int | None
    message: str
    user_raised: bool


class SearchGroup:
    """Independent PUCT searches whose leaves share one evaluator call per
    round (ADR-0045), for self-play throughput.

    ``trees`` slots, each a full ``Search`` with its own tree, counter and
    search stream. ``options`` are shared, and ``slot_options``, a list of
    one dict per slot, updates them per slot. A group call runs one search
    call on every active slot, interleaved in slot order. There is one
    root batch, then each slot's root noise. Each round then collects every
    slot's leaves, makes one evaluator call on the concatenated batch
    (rows in slot order), validates every slot's rows, and backs every slot
    up. The evaluator is the ``Search`` one, and sees the concatenated
    batch. A round of terminal leaves only makes no evaluator call.

    With an evaluator whose answer for a leaf depends on that leaf alone
    (a canned one), every slot's result equals a ``Search`` started from
    the same state. A learned network is in general not batch-invariant,
    so its results can depend on the other slots in the batch. They are
    repeatable only for the identical group on the same machine. Gate and
    held-out evaluation use ``Search`` (ADR-0042).

    A call is all or nothing: any failure rolls every participating slot
    back. The evaluator's own exception is raised as ``EvaluatorError``
    with it as ``__cause__``, and ``.slot`` is ``None``. A ``BaseException``
    that is not an ``Exception``, and anything a signal handler raises, is
    raised unchanged. Errors the group creates itself carry ``.slot``:
    a bad row, an engine error, or a missing or terminal root. The
    exceptions of user code are never modified; ``last_failure()`` gives
    the slot of any failure. While a call runs, every other method raises
    ``ConcurrencyError``.
    """

    def __init__(self, trees: int, env_id: str = "tessera/Connect4-v1", slot_options=None, **options):
        if slot_options is not None:
            slot_options = [dict(o) for o in slot_options]
        self._native = _native.make_search_group(trees=trees, env_id=env_id, slot_options=slot_options, **options)
        layout = self._native.layout()
        self.trees: int = layout["trees"]
        self.actions: int = layout["actions"]
        self.memory_bytes: int = layout["memory_bytes"]
        self._obs_bytes: int = layout["obs_bytes"]
        probe = make(env_id)
        self._leaves = list(probe._leaves)
        probe.close()

    def set_root(self, i: int, record: bytes) -> None:
        """Replaces slot ``i``'s tree by the root ``record``."""
        self._native.set_root(i, bytes(record))

    def set_active(self, i: int, flag: bool) -> None:
        """Whether slot ``i`` takes part in the next call."""
        self._native.set_active(i, bool(flag))

    def active(self, i: int) -> bool:
        return self._native.active(i)

    def run(self, evaluator) -> None:
        """One group call with ``evaluator``."""
        raised = []
        call = _bridge(evaluator, self.actions, self._leaves, self._obs_bytes, raised)
        try:
            self._native.run(call)
        except Exception as exc:
            if raised and exc is raised[0]:
                err = _native.EvaluatorError("the evaluator raised")
                err.slot = None
                raise err from exc
            failure = self._native.last_failure()
            if failure is not None and not failure[2] and isinstance(exc, _native.TesseraError):
                # An error the group created itself (never the user's).
                exc.slot = failure[0]
            raise

    def last_failure(self) -> GroupFailure | None:
        """The last failed call, or ``None``; cleared when a call starts."""
        f = self._native.last_failure()
        return None if f is None else GroupFailure(*f)

    def root_visits(self, i: int) -> np.ndarray:
        return np.asarray(self._native.root_visits(i), dtype=np.uint32)

    def select_action(self, i: int, temperature: float = 0.0) -> int:
        return self._native.select_action(i, float(temperature))

    def calls(self, i: int) -> int:
        return self._native.calls(i)

    def tree_hash(self, i: int) -> int:
        return self._native.tree_hash(i)

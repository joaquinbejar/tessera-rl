# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native environment API: a pool of instances with NumPy outputs.

Output modes: ``outputs="borrowed"`` (the default here) returns
read-only arrays that view the engine's buffers without copying. They stay
readable for as long as they are referenced, even after ``close``, but
their contents are only guaranteed until the next ``reset`` or ``step`` on
the same environment: afterwards they show the values of later calls, and
``is_stale`` reports it. ``outputs="owned"`` returns fresh writable arrays
the caller keeps. Finish reading borrowed arrays, including asynchronous
work you started on them, before the next call.

Native views export through the buffer protocol (read-only) and DLPack
(``__dlpack__``: DLPack 1.0 with the read-only flag when the consumer asks
for ``max_version >= (1, 0)``, a legacy capsule otherwise). A legacy
capsule cannot be marked read-only, so legacy consumers get a writable
alias. NumPy honours the read-only flag; PyTorch 2.15 asks for DLPack 1.0
and accepts the flag without warnings but returns a writable tensor.

The read-only flag is a signal, not an enforcement: no
tensor is promised to be immutable, and writing is the caller's
responsibility. A write through any writable alias reaches the engine
buffer and corrupts the outputs until the next call overwrites them. To
modify outputs, use ``outputs="owned"`` or copy first (``tensor.clone()``,
``array.copy()``).
"""

from __future__ import annotations

import operator
import typing
from typing import NamedTuple

import numpy as np

from tessera_rl import _envs, _native, _render


class Step(NamedTuple):
    """The outputs of one call: borrowed read-only views or owned arrays.

    Per-agent arrays have shape ``(num_envs, agents, ...)``; episode flags
    have shape ``(num_envs,)``. ``masks`` is ``None`` for environments
    without action masks. ``acting`` marks the agents whose action the
    next step reads: every alive agent in a simultaneous
    environment, exactly one in a sequential one while its game runs, and
    none on a finished board.

    ``slot_generations`` (``uint32``, ``(num_envs, agents)``) is the
    generation of each slot's occupant (an agent is identified by its
    (slot, generation) pair); it starts at 0 and grows by one each time an agent
    spawns into a slot that died. It has the same name, dtype and shape as
    host protocol 1's output.

    Breaking change in P2-M1b: ``acting`` is the ninth field, so unpacking
    a ``Step`` into 8 names fails; access by name is stable.
    ``slot_generations`` is the tenth field (issue #273), so unpacking into
    9 names fails as well.
    """

    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    agent_terminated: np.ndarray
    agent_truncated: np.ndarray
    alive: np.ndarray
    episode_terminated: np.ndarray
    episode_truncated: np.ndarray
    masks: np.ndarray | None
    acting: np.ndarray
    slot_generations: np.ndarray


class HalfStep(NamedTuple):
    """One half's outputs in Pipelined mode.

    ``ids`` are the instances of the half; ``step`` holds their rows only,
    so its first dimension is ``len(ids)``. In ``borrowed`` mode the arrays
    stay valid until this half's next ``send``, or a ``reset``: the other
    half's ``send`` does not end them.
    """

    half: int
    ids: range
    step: Step


class AsyncStep(NamedTuple):
    """One batch in Async mode.

    ``ids`` lists the batch's instances in the order they completed. With the
    default ``gather="gathered"`` (and in ``owned`` mode) ``step`` has one row
    per entry of ``ids``, in that order. With ``gather="indexed"`` every array
    of ``step`` has all ``num_envs`` rows and only the rows of ``ids`` belong
    to this batch: read those, the others may be running.
    """

    ids: np.ndarray
    step: Step


def _resolve_constructor_annotations(*classes: type) -> None:
    """Gives each NamedTuple's generated constructor the resolved field
    types (#525).

    With ``from __future__ import annotations`` the fields are strings, and
    the ``__new__`` that ``NamedTuple`` generates keeps them as strings to be
    evaluated in the ``collections`` namespace, where ``np`` and ``Step`` do
    not exist: ``inspect.signature`` users such as pdoc then cannot resolve
    them. ``typing.get_type_hints`` resolves them in this module's namespace
    once, at import.
    """
    for cls in classes:
        cls.__new__.__annotations__ = dict(typing.get_type_hints(cls))


_resolve_constructor_annotations(Step, HalfStep, AsyncStep)


class Env:
    """A pool of ``num_envs`` instances of one environment.

    ``outputs`` is ``"borrowed"`` or ``"owned"`` (see the module notes).

    Concurrency: each native call is admitted on its own. A native call
    that starts while another native call on the same ``Env`` is running
    raises ``ConcurrencyError``. This is not a guarantee about whole Python
    method calls: the Python work around the native call (converting
    actions, wrapping outputs) can interleave with other threads, so
    serialize calls on one ``Env`` yourself.

    Split-phase modes: a step stays in flight after ``send``
    returns, and ``step`` raises ``ArgumentError``. One thread may ``send``
    while another waits in ``recv``; a second concurrent ``recv`` raises
    ``ConcurrencyError`` naming ``ConcurrentRecv``. Writers (``reset``,
    ``restore``, ``restore_all``) raise ``ConcurrencyError`` while sent work
    still runs, and that error is retryable.

    ``mode="pipelined"``: two fixed halves, ``halves``. ``send(half, actions)``
    and ``recv() -> HalfStep``, oldest sent half first. A half's arrays stay
    valid until its next ``send`` or a ``reset``. ``snapshot``,
    ``import_record``, ``fork`` and ``reset`` with ids are refused;
    ``snapshot_all`` and ``restore_all`` work.

    ``mode="async"`` with ``batch_size``: ``send(ids, actions)`` and
    ``recv() -> AsyncStep``, the first ``batch_size`` ids to finish. Batch
    membership depends on timing; each instance's trajectory does not.
    By default a batch is gathered into one buffer reused at every ``recv``,
    so its arrays go stale at the next ``recv``, not the next ``send``: a
    deliberate exception to the rule that outputs stay valid until the
    next ``send`` for their instances. Pass
    ``gather="indexed"`` for arrays over all rows, which go stale when a
    ``send`` or mutation of the batch's ids succeeds, or when ``reset(ids=...)``
    or ``restore`` succeeds on any ids: those writers release every batch's
    borrow, even over rows they do not write. A refused call leaves all arrays
    fresh, and a poisoned pool makes all of them stale, for good.

    In both split-phase modes, every array reads stale while a writer
    (``reset``, ``restore``, ``restore_all``) runs, gathered ones included. After
    a refused writer they read fresh again.

    In Async, ``reset(ids=...)`` and ``restore`` also raise a retryable
    ``ConcurrencyError`` while a ``recv`` result has not reached its caller
    (an interrupted ``recv``): call ``recv`` again first. The error may name an
    id outside the call, an id of that result. If the thread holding that result
    has exited, ``recv`` cannot clear it: only ``reset`` of all instances,
    ``restore_all`` or ``close`` can. ``reset()`` returns all
    instances as one batch; ``reset(ids=...)`` and ``restore`` return nothing
    and the next ``recv`` delivers those ids. ``snapshot(id)`` needs only ``id``
    to be idle (issue #85): other ids may run meanwhile, and it raises a
    retryable ``ConcurrencyError`` while ``id`` itself is sent and not yet
    received. It returns an opaque record for ``restore``, not bytes. ``import_record`` and ``fork`` are refused.

    Async is currently much slower per round than Sync: workers claim one
    instance at a time and waits poll. Issue #83 tracks batch claiming and
    wake-based waits. Prefer Sync or Pipelined unless you need out-of-order
    batches.

    Fork: an ``Env`` does not survive ``os.fork``, whatever its mode or
    ``workers``. In the child, every call raises ``ForkError``, except ``close``,
    which does nothing. Dropping it leaks the pool in the child and never joins.
    Arrays taken before the fork stay readable, always read stale, and may hold
    rows of a step that was running at the fork. Create a new ``Env`` in the
    child, or use the spawn start method.

    On macOS before 14.4, creating an ``Env`` with worker threads in a forked
    child raises ``ForkError``, and ``make`` starts worker threads by default:
    pass ``workers=0`` in Sync mode, or use spawn.

    On macOS, a child that first imports ``tessera_rl`` after the fork is not
    known to be one, so its new pools use the default blocking wait, which can
    trap with ``SIGTRAP`` if the parent's libdispatch state marks it unsafe. A
    parent with plain Python threads did not reproduce this. Import before
    forking, or use the spawn start method.

    Snapshots of split-phase modes are opaque, restore only the environment
    that took them, and cannot be saved; ``encode`` and ``load`` make files that
    can, in every mode.
    """

    def __init__(self, native: _native.Env, mode: str = "sync", indexed: bool = False):
        self._native = native
        #: ``"sync"``, ``"pipelined"`` or ``"async"``.
        self.mode: str = mode
        #: The last split-phase delivery this object handed out.
        self._delivered = 0
        #: The last whole-pool call's terminal-observation buffers (#545).
        self._last_terminal: list | None = None
        layout = native.layout()
        self.num_envs: int = layout["num_envs"]
        self.agents: int = layout["agents"]
        self.action_choices: int = layout["action_choices"]
        #: Inclusive ``(low, high)`` of a continuous environment's one float32
        #: head; ``None`` for a discrete one (``action_choices`` is then > 0).
        self.action_bounds: tuple[float, float] | None = layout["action_bounds"]
        #: Choices per discrete head of one agent (SPEC 11.1): ``(n,)`` for
        #: ``Discrete(n)``, ``nvec`` for ``MultiDiscrete(nvec)``, ``()`` for a
        #: continuous environment. With more than one head, actions have
        #: shape ``(rows, agents, heads)``.
        self.action_nvec: tuple[int, ...] = tuple(layout["action_nvec"])
        self._mask_len: int = layout["mask_len"]
        #: Each discrete head's block in an agent's mask row (SPEC 11.3):
        #: ``(offset, parent sizes, choices)``; empty without masks.
        self.mask_heads: tuple[tuple[int, tuple[int, ...], int], ...] = tuple(
            (offset, tuple(parents), size) for offset, parents, size in layout["mask_heads"]
        )
        #: Discrete heads per entity group, 0 when none.
        self.entity_group_heads: int = layout["entity_group_heads"]
        #: ``"simultaneous"``, ``"sequential"`` or ``"mixed"``.
        self.turns: str = layout["turns"]
        #: The policy group of each agent slot: ``"learner"``,
        #: ``"external"``, ``"heuristic"`` or ``"privileged_heuristic"``.
        self.policy_groups: tuple[str, ...] = tuple(layout.get("policy_groups", ()))
        self.outputs: str = layout["outputs"]
        #: The instance ids of the two halves, in Pipelined mode; else ``None``.
        self.halves: tuple[range, range] | None = None
        if "halves" in layout:
            (a, b), (c, d) = layout["halves"]
            self.halves = (range(a, b), range(c, d))
        self._indexed: bool = indexed
        #: The id given to `make`, for the Viewer; `None` otherwise.
        self.env_id: str | None = None
        #: The `make` options that shape the records (for a shadow pool).
        self._config: dict | None = None
        self._leaves = [(name or "obs", np.dtype(dtype), tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]
        #: Whether the pool has the privileged state buffer
        #: (``make(..., privileged_state=True)``); see ``state``.
        self.privileged_state: bool = bool(layout.get("privileged_state", False))
        state_leaves = layout.get("state_leaves", ())
        self._state_leaves = [(name or "state", np.dtype(dtype), tuple(shape)) for name, dtype, shape, *_ in state_leaves]
        self._state: dict[str, np.ndarray] | None = None

    def reset(self, seed: int | None = None, ids=None) -> Step | list[HalfStep] | AsyncStep | None:
        """Resets every instance; a new seed base if ``seed`` is given.

        Pipelined: one ``HalfStep`` per half. Async: one ``AsyncStep`` of all
        instances, or with ``ids`` nothing, as the next ``recv`` delivers
        them. Both need every sent id finished (``ConcurrencyError``),
        except to recover a poisoned pool.
        """
        seed = None if seed is None else _index(seed, "seed")
        if ids is not None:
            if self.mode != "async":
                raise _native.ArgumentError(f"reset with ids is not available in {self.mode} mode")
            self._native.reset(seed, [_index(i, "ids") for i in ids], self._delivered)
            return None
        out = self._native.reset() if seed is None else self._native.reset(seed)
        if self.mode == "pipelined":
            return [self._half(h) for h in out]
        if self.mode == "async":
            return self._batch(out)
        return self._wrap(out)

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Sets what resets start from: a :class:`tessera_rl.Archive`
        of this environment, with optional ``weights`` (one non-negative int
        per entry, not all zero; all ones by default), or ``None`` for
        ``Environment::reset``. It applies to every reset after the call,
        auto-resets included. An archive reset keeps the instance's own
        seed and lineage and copies an entry's state.

        In Pipelined and Async mode no step may be in flight
        (``ConcurrencyError``). An archive of another environment raises
        ``ArgumentError``; of another config, ``ConfigError``. While
        recording (``tessera_rl.replay.record``) the call is logged.
        """
        from tessera_rl._archive import Archive

        if archive is not None and not isinstance(archive, Archive):
            raise _native.ArgumentError(f"archive must be a tessera_rl.Archive or None, not {type(archive).__name__}")
        native = None if archive is None else archive._native
        if weights is not None:
            weights = [_index(w, "weights") for w in weights]
        self._native.set_reset_source(native, weights)

    def reset_from(self, ids, entries) -> Step | None:
        """Resets the instances ``ids`` from the archive entries ``entries``
        (one per id) of the attached archive. Sync: the outputs,
        as ``reset``; Async: nothing, the next ``recv`` delivers the ids.
        Pipelined mode has no partial reset (``CapabilityError``). Logged
        while recording.
        """
        ids = [_index(i, "ids") for i in ids]
        entries = [_index(e, "entries") for e in entries]
        if self.mode == "async":
            self._native.reset_from(ids, entries, self._delivered)
            return None
        out = self._native.reset_from(ids, entries)
        return self._wrap(out)

    def send(self, target, actions) -> None:
        """Starts steps. Pipelined: ``target`` is a half and ``actions`` has
        one int per agent row of it; it ends that half's arrays' window.
        Async: ``target`` is a list of ids and ``actions`` one int per agent
        row of them, in that order. Rules as ``step``.

        ``ConcurrencyError`` while the target is in flight or was delivered
        to a ``recv`` that was interrupted before returning, or, in
        Pipelined, two cycles ahead of the other half.
        """
        acts = self._actions(actions).tobytes()
        if self.mode == "async":
            target = [_index(i, "ids") for i in target]
        else:
            target = _index(target, "half")
        self._native.send(target, acts, self._delivered)

    def recv(self) -> HalfStep | AsyncStep:
        """Waits for the next result: the oldest sent half (Pipelined) or the
        first full batch (Async).

        The wait is interruptible: an exception from a signal handler
        (``KeyboardInterrupt``) leaves the work in flight, and a later
        ``recv`` returns it. An exception raised after the native call
        returned loses nothing either: the next ``recv`` returns that result
        again, and until then its ids count as in flight for ``send``.

        One limit remains: once this method acknowledges the result, a signal
        handler that raises in the caller's frame before the result is bound
        loses it. No Python function can close that. The same happens if a
        handler runs inside this method after the acknowledgement, which a
        trace function installed with ``sys.settrace`` makes possible on the
        return line. Whether CPython can run handlers there without one is not
        verified for any version; the code keeps that gap to one attribute
        store.

        If the thread that received a result exits before acknowledging it,
        ``recv`` on other threads stays refused with ``ConcurrencyError``
        until ``reset`` of all instances, ``restore_all`` or ``close``.
        """
        out = self._native.recv(self._delivered)
        result = self._batch(out) if self.mode == "async" else self._half(out)
        self._delivered = out["seq"]
        return result

    def snapshot_all(self) -> _native.PoolSnapshot:
        """An opaque snapshot of the whole pool, in every mode: every
        instance's record and frame stacks, and the pool context (fork
        branch counter, reset source, transform statistics). It restores
        only this environment and cannot be saved. All sent work must be
        finished (``ConcurrencyError``, retryable); arrays stay valid.
        """
        return self._native.snapshot_all()

    def restore_all(self, snapshot) -> list[HalfStep] | None:
        """Restores the instances of a snapshot of this environment
        (split-phase modes): records only (and Pipelined's cycle state); the
        pool context stays as it is, so use :meth:`restore_pool` to bring
        back the reset source, the fork branch counter and transform
        statistics too. Pipelined: one ``HalfStep`` per half. Async:
        nothing; send next. Needs all sent work finished, except to recover
        a poisoned pool.
        """
        out = self._native.restore_all(snapshot)
        return None if out is None else [self._half(h) for h in out]

    def restore_pool(self, snapshot, archive=None) -> Step | list[HalfStep] | None:
        """Restores a :meth:`snapshot_all` snapshot of this environment
        whole: records, frame stacks and the pool context,
        an exact continuation. If the context names an archive, ``archive``
        must be that :class:`tessera_rl.Archive` (else ``ConfigError``); for
        fresh resets it must be ``None`` (else ``ArgumentError``). Every check
        comes first: an error changes nothing. Sync: the outputs, as
        ``reset``; Pipelined: one ``HalfStep`` per half; Async: nothing.
        Refused while recording or replaying.
        """
        from tessera_rl._archive import Archive

        if archive is not None and not isinstance(archive, Archive):
            raise _native.ArgumentError(f"archive must be a tessera_rl.Archive or None, not {type(archive).__name__}")
        out = self._native.restore_pool(snapshot, None if archive is None else archive._native)
        if self.mode == "pipelined":
            return [self._half(h) for h in out]
        if self.mode == "async":
            return None
        return self._wrap(out)

    def restore(self, ids, snapshots) -> None:
        """Restores ``ids`` from one-instance ``snapshot`` records (Async
        mode); the next ``recv`` delivers them."""
        self._native.restore([_index(i, "ids") for i in ids], list(snapshots), self._delivered)

    def _batch(self, out: dict) -> AsyncStep:
        ids = np.array(out["ids"], dtype=np.uint32)
        rows = None if self._indexed else len(ids)
        return AsyncStep(ids, self._wrap(out["outputs"], rows))

    def _half(self, out: dict) -> HalfStep:
        ids = range(out["start"], out["end"])
        return HalfStep(out["half"], ids, self._wrap(out["outputs"], len(ids)))

    def step(self, actions) -> Step:
        """One step. ``actions`` has one int per agent row, ``(num_envs, agents)``;
        with several discrete heads (``MultiDiscrete``), one int per head,
        ``(num_envs, agents, heads)``, each in ``[0, action_nvec[k])``.

        Discrete actions: every value must be an exact integer in the int32
        range (integer dtypes, or floats with no fractional part). Continuous
        actions: real numbers, shape ``(num_envs,)`` or ``(num_envs, 1)``,
        converted to float32 (nearest) and refused, not clipped, if NaN,
        infinite or out of bounds. Anything else raises ``ActionError``
        before the pool is touched.
        """
        if _plain_int32(actions) and self.action_bounds is None and len(self.action_nvec) <= 1:
            # #152 E2: an int32 C-contiguous array is already the wire format,
            # and every int32 is in the int32 range, so no Python pass over it:
            # the native step copies it through a memoryview before the pool
            # is touched, and checks each value against the head (E1).
            if actions.size != self.num_envs * self.agents:
                raise _native.ActionError(f"expected {self.num_envs * self.agents} actions, got {actions.size}")
            return self._wrap(self._native.step(memoryview(actions)))
        acts = self._actions(actions)
        per_row = len(self.action_nvec) if len(self.action_nvec) > 1 else 1
        if acts.size != self.num_envs * self.agents * per_row:
            raise _native.ActionError(f"expected {self.num_envs * self.agents * per_row} actions, got {acts.size}")
        return self._wrap(self._native.step(acts.tobytes()))

    def _actions(self, actions) -> np.ndarray:
        if self.action_bounds is not None:
            return _continuous_actions(actions)
        if len(self.action_nvec) > 1:
            return _multi_actions(actions, self.agents, self.action_nvec)
        return _discrete_actions(actions)

    def _wrap(self, out: dict, rows: int | None = None) -> Step:
        n, a = self.num_envs if rows is None else rows, self.agents
        # The raw outputs of the last whole-pool call, for adapters that read
        # outputs ``Step`` does not carry (the episode statistics); ``None``
        # after any call on a subset (a half or an Async batch). Its arrays
        # are borrowed views: valid only until the next call on this
        # environment, so a reader takes what it needs at once and drops it.
        self._last_out = out if rows is None else None
        # Kept apart from `_last_out`, which an adapter may clear after
        # reading it (#545): `terminal_observations` reads this.
        self._last_terminal = out.get("terminal_observations") if rows is None else None
        self._state = self._wrap_state(out, n)
        if self.outputs == "borrowed":
            # Zero-copy, read-only arrays over the native views (the buffer
            # protocol carries dtype and shape).
            obs = {name: np.asarray(v) for (name, _, _), v in zip(self._leaves, out["observations"], strict=True)}
        else:
            obs = {
                name: np.frombuffer(buf, dtype=dtype).reshape((n, a, *shape))
                for (name, dtype, shape), buf in zip(self._leaves, out["observations"], strict=True)
            }

        def field(key, dtype, shape):
            if self.outputs == "borrowed":
                return np.asarray(out[key])
            return np.frombuffer(out[key], dtype=dtype).reshape(shape)

        masks = None
        if self._mask_len:
            masks = field("masks", np.bool_, (n, a, self._mask_len))
        return Step(
            observations=obs,
            rewards=field("rewards", "<f4", (n, a)),
            agent_terminated=field("agent_terminated", np.bool_, (n, a)),
            agent_truncated=field("agent_truncated", np.bool_, (n, a)),
            alive=field("alive", np.bool_, (n, a)),
            episode_terminated=field("episode_terminated", np.bool_, (n,)),
            episode_truncated=field("episode_truncated", np.bool_, (n,)),
            masks=masks,
            acting=field("acting", np.bool_, (n, a)),
            slot_generations=field("slot_generations", "<u4", (n, a)),
        )

    @property
    def terminal_observations(self) -> dict[str, np.ndarray]:
        """The final observations of the last whole-pool ``reset`` or ``step``,
        before their same-step reset (#545): a dict of leaves shaped as
        ``Step.observations``, ``(num_envs, agents, ...)``. Rows of instances
        that did not end in that call are zero.

        Only with ``auto_reset="same_step"`` and in Sync mode: otherwise, before
        the first call, and after a call on a subset (a Pipelined half, an Async
        batch), ``ArgumentError``. In ``borrowed`` mode the arrays are read-only
        views with the borrow window of ``Step``'s (``is_stale`` applies); in
        ``owned`` mode each read is a fresh copy."""
        bufs = self._last_terminal
        if self.mode != "sync" or bufs is None:
            raise _native.ArgumentError(
                "terminal_observations needs a whole-pool reset or step in Sync mode first"
            )
        if not bufs:
            raise _native.ArgumentError('terminal_observations needs auto_reset="same_step"')
        n, a = self.num_envs, self.agents
        if self.outputs == "borrowed":
            return {name: np.asarray(v) for (name, _, _), v in zip(self._leaves, bufs, strict=True)}
        return {
            name: np.array(np.frombuffer(buf, dtype=dtype).reshape((n, a, *shape)))
            for (name, dtype, shape), buf in zip(self._leaves, bufs, strict=True)
        }

    def _wrap_state(self, out: dict, n: int) -> dict[str, np.ndarray] | None:
        raw = out.get("state")
        if raw is None:
            return None
        if self.outputs == "borrowed":
            return {name: np.asarray(v) for (name, _, _), v in zip(self._state_leaves, raw, strict=True)}
        return {
            name: np.frombuffer(buf, dtype=dtype).reshape((n, *shape))
            for (name, dtype, shape), buf in zip(self._state_leaves, raw, strict=True)
        }

    @property
    def state(self) -> dict[str, np.ndarray] | None:
        """The privileged state of every instance after the last call, one
        array per leaf of shape ``(num_envs, ...)``: everything a centralized
        critic needs, hidden information included. ``None`` without
        ``make(..., privileged_state=True)`` or before the first call.

        It holds what no player observes: never feed it to a policy that
        acts. Search never reads it. A pending instance's row is zero. In
        ``borrowed`` mode the arrays are views with the same window as the
        observations of that call (``is_stale`` reports when it ended); in
        ``owned`` mode they are fresh copies. ``Step`` does not carry it.

        Cost: the rows are written in one serial pass after the workers'
        step, so every call adds ``num_envs`` state writes on the calling
        thread, whatever ``workers`` is.
        """
        return self._state

    def snapshot(self, index: int) -> bytes:
        """The canonical record of instance ``index``; in Async mode an opaque
        record for ``restore`` (see the class notes)."""
        return self._native.snapshot(_index(index, "index"))

    def render(self, ids=None) -> np.ndarray:
        """RGB8 frames of ``ids`` (default: every instance), shape
        ``(len(ids), height, width, 3)``, drawn from their snapshots. Sync mode, for an
        environment that implements
        ``Render``; ``CapabilityError`` otherwise. Owned arrays."""
        ids = range(self.num_envs) if ids is None else ids
        return _render.frames(self._native, [_index(i, "ids") for i in ids])

    def epoch_barrier(self) -> None:
        """Merges the normalization accumulators of a ``Normalize``
        transform into the pool statistics, in chunk order; later
        observations use them. Sync mode; a no-op without a
        ``Normalize``."""
        self._native.epoch_barrier()

    def freeze_normalization(self, frozen: bool = True) -> None:
        """Stops (``True``) or restarts (``False``) the accumulation of
        normalization statistics, as for evaluation. Sync mode."""
        self._native.freeze_normalization(bool(frozen))

    def scene(self, index: int) -> list[tuple]:
        """Instance ``index``'s vector scene: a list of
        primitives, coordinates in pixels; see ``tessera_rl._render.scene``."""
        return _render.scene(self._native, _index(index, "index"))

    @property
    def canvas(self) -> tuple[int, int] | None:
        """The frame size ``(width, height)``, or ``None`` when the pool
        cannot render."""
        return _render.canvas(self._native)

    def illegal_actions(self, index: int) -> list[int]:
        """Illegal-but-valid actions replaced in instance ``index``'s current
        episode, per agent slot (the record's statistic). Sync mode only.

        With entity groups (the RTS), each group counts on its
        own: an order other than the no-op on an empty or dead unit slot
        counts one."""
        return self._native.illegal_actions(_index(index, "index"))

    def state_hash(self, index: int) -> int:
        """Instance ``index``'s state hash, as Rust's goldens fold it. Sync
        mode only."""
        return self._native.state_hash(_index(index, "index"))

    def import_record(self, index: int, record: bytes) -> None:
        """Imports an (untrusted) record into instance ``index``."""
        self._native.import_record(_index(index, "index"), bytes(record))

    def encode(self, ids=None) -> bytes:
        """A persisted file of ``ids``, or of every instance: fresh ``bytes``.

        It can be saved and loaded by another process, into any mode, by an
        environment with the same id, version, config and step options. A file holds
        full state and is privileged: a Leduc file holds the hidden cards, so
        share it knowingly and never feed it to a player's view. It holds
        instance records only, no pool-level state.
        """
        if ids is None:
            ids = range(self.num_envs)
        return self._native.encode([_index(i, "ids") for i in ids])

    def load(self, data: bytes, ids=None, max_instances=None, max_decoded_bytes=None):
        """Loads a file from ``encode``. A file refused for its bytes, its records or
        work in flight changes nothing, and views stay readable. Past those checks only
        a panic while writing can fail, which poisons as any writer does.

        ``data`` must be exactly ``bytes``; copy a ``bytearray`` or ``memoryview``
        first with ``bytes(x)``, as other threads could write it during decoding.
        ``ids=None`` loads every instance, and the file must hold ``num_envs``
        records; it also recovers a poisoned environment. A list of ``ids`` is for
        Sync mode only. Returns what ``restore_all`` returns in this mode.
        Defaults: at most 2**24 instances and 1 GiB. Raises ``PersistError`` for a
        damaged, malformed or mismatched file.
        """
        if ids is not None:
            ids = [_index(i, "ids") for i in ids]
        return self._native.load(data, ids, max_instances, max_decoded_bytes)

    def fork(self, src: int, dst: list[int]) -> None:
        """Copies instance ``src`` into every instance of ``dst``."""
        self._native.fork(_index(src, "src"), [_index(d, "dst") for d in dst])

    @property
    def is_poisoned(self) -> bool:
        return self._native.is_poisoned()

    @property
    def assignment(self) -> np.ndarray | None:
        """Each instance's league opponent (ADR-0088 item 2.3), uint32
        ``(num_envs,)``: the population id its ``"opponent"`` slots play this
        episode, drawn at the instance's reset; ``None`` without a league.
        Each read is a fresh owned copy."""
        raw = self._native.assignment()
        if raw is None:
            return None
        return np.frombuffer(raw, dtype="<u4").copy()

    @property
    def heuristic_actions(self) -> np.ndarray | None:
        """The last step's heuristic actions, int32
        ``(num_envs, agents, heads)``: the action the engine computed for each
        heuristic slot that acted, zero elsewhere; ``None`` without a heuristic
        slot. ``Step`` does not carry it.

        Each read is a fresh owned copy of the engine's buffer: it never goes
        stale, and it does not change with later calls. The engine's buffer
        is written only by ``step`` (resets, restores and forks leave the last
        step's rows), so read it after the ``step`` whose opponents you log;
        a later ``step`` replaces what the next read returns.

        In ``mode="pipelined"`` and ``mode="async"`` the rows of an instance
        are written when ``recv`` returns it (its half, or its id in the
        batch); the other instances keep the rows of their own last step.
        """
        raw = self._native.heuristic_actions()
        if raw is None:
            return None
        heads = max(len(self.action_nvec), 1)
        return np.frombuffer(raw, dtype="<i4").reshape(self.num_envs, self.agents, heads)

    def watch(self) -> _native.WatchHandle:
        """Attaches a live watch and returns its handle.

        The handle is the watch's single consumer: ``request(instance)`` asks
        for that instance's record, and the pool answers at its next batch
        boundary where the instance is idle (after ``reset``, ``step``,
        ``restore`` or ``fork`` in sync mode; after the ``recv`` that returns
        it in the split-phase modes), on the training thread, by copying one
        record. ``take()`` returns ``(generation, record)`` once served, else
        ``None``; it never blocks and never takes this ``Env``'s guard, so
        another thread may call it while this one steps. ``generation`` counts
        the pool's batch boundaries since the watch was attached. ``close()``
        drops the handle. A poisoned pool serves nothing (``take`` raises
        ``PoisonedError``); after ``Env.close`` the handle raises
        ``ArgumentError`` (detached). A second watch while a handle is open
        raises ``ConcurrencyError``; host environments raise
        ``CapabilityError``. See :func:`tessera_rl.viewer.watch`.
        """
        return self._native.watch()

    def close(self) -> None:
        """Frees the pool; idempotent. Arrays already returned stay readable
        (borrowed views keep the engine's buffers alive) and never change."""
        self._native.close()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_I32 = np.iinfo(np.int32)


_I4 = np.dtype("<i4")


def _plain_int32(actions) -> bool:
    """Whether ``actions`` is an exact ndarray of C-contiguous little-endian
    int32, which the native step reads as is (#152 E2). Subclasses and
    every other dtype or layout take ``_discrete_actions``."""
    return type(actions) is np.ndarray and actions.dtype == _I4 and actions.flags.c_contiguous


def _discrete_actions(actions) -> np.ndarray:
    """``actions`` as contiguous little-endian int32, converted exactly."""
    try:
        arr = np.asarray(actions)
    except (TypeError, ValueError) as e:
        raise _native.ActionError(f"actions are not an array: {e}") from None
    if arr.dtype.kind in "iu":
        if arr.size and (arr.min() < _I32.min or arr.max() > _I32.max):
            raise _native.ActionError("actions out of the int32 range")
    elif arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise _native.ActionError("actions must be finite")
        if not np.all(arr == np.trunc(arr)):
            raise _native.ActionError("discrete actions must be integers")
        if arr.size and (arr.min() < _I32.min or arr.max() > _I32.max):
            raise _native.ActionError("actions out of the int32 range")
    elif arr.size:
        raise _native.ActionError(f"actions must be integers, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr, dtype="<i4")


def _multi_actions(actions, agents: int, nvec: tuple[int, ...]) -> np.ndarray:
    """Multi-head actions, shape ``(rows, agents, heads)``, as contiguous
    little-endian int32: integers (as for one head), each head in
    ``[0, nvec[k])``; ``ActionError`` otherwise, before the pool is touched."""
    try:
        arr = np.asarray(actions)
    except (TypeError, ValueError) as e:
        raise _native.ActionError(f"actions are not an array: {e}") from None
    heads = len(nvec)
    if arr.ndim != 3 or arr.shape[1:] != (agents, heads):
        raise _native.ActionError(f"actions have shape (rows, {agents}, {heads}), got {arr.shape}")
    acts = _discrete_actions(arr)
    if acts.size and ((acts < 0) | (acts >= np.asarray(nvec, dtype=np.int64))).any():
        raise _native.ActionError(f"an action is out of range for its head (choices {list(nvec)})")
    return acts


def _real_array(actions) -> np.ndarray:
    """``actions`` as an array of one real number per row, shape ``(n,)`` or
    ``(n, 1)``, flattened; ``ActionError`` otherwise."""
    try:
        arr = np.asarray(actions)
    except (TypeError, ValueError) as e:
        raise _native.ActionError(f"actions are not an array: {e}") from None
    if arr.dtype.kind not in "fiu":
        raise _native.ActionError(f"continuous actions must be real numbers, got dtype {arr.dtype}")
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    if arr.ndim != 1:
        raise _native.ActionError(f"continuous actions have shape (n,) or (n, 1), not {arr.shape}")
    return arr


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """Contiguous little-endian float32, rounded to nearest. A value beyond
    the float32 range becomes infinite, which the caller or the pool refuses."""
    with np.errstate(over="ignore"):
        return np.ascontiguousarray(arr, dtype="<f4")


def _continuous_actions(actions) -> np.ndarray:
    """``actions`` as flat little-endian float32 (design 4.3); bounds, NaN
    and infinities are the pool's to refuse."""
    return _to_float32(_real_array(actions))


def _index(value, what: str) -> int:
    """An integer argument as an exact ``int``: the native module accepts
    only exact built-in types, so user objects are
    converted here, under the caller's own floating-point controls."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, not a bool")
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {type(value).__name__}") from None


#: Keyword arguments the native constructors take as strings.
_STRING_OPTIONS = {"auto_reset", "outputs", "mode", "gather", "obs", "wait", "schedule"}


#: Constructor options passed as exact ``bool`` (the RTS ``seat_swap``, the
#: Lanterns ``zero_on_fuse``, the limit order book's ``allow_short_cash``).
_BOOL_OPTIONS = frozenset({"seat_swap", "zero_on_fuse", "allow_short_cash", "privileged_state"})

#: Asset options: a bundled asset's name (``str``) or its bytes
#: (``bytes``, or any buffer, copied to exact ``bytes``).
_ASSET_OPTIONS = frozenset({"layout"})


#: Policy group names.
_POLICY_GROUPS = ("learner", "external", "heuristic", "privileged_heuristic", "opponent")


def _policy_map(value) -> str | None:
    """A ``policy_map`` (a list of group names, one per agent slot, or a dict
    from slot to group name, other slots learners) as the string the native
    module parses: names separated by commas, or ``slot=name`` items. An
    empty dict is the default map."""

    def name(group) -> str:
        if not isinstance(group, str) or group not in _POLICY_GROUPS:
            raise _native.ArgumentError(f"policy_map groups are one of {_POLICY_GROUPS}, not {group!r}")
        return group

    if isinstance(value, dict):
        if not value:
            return None
        return ",".join(f"{_index(slot, 'policy_map slot')}={name(group)}" for slot, group in value.items())
    if isinstance(value, (list, tuple)):
        return ",".join(name(group) for group in value)
    raise TypeError(f"policy_map must be a list or a dict of group names, got {type(value).__name__}")

def native_kwargs(**kwargs) -> dict:
    """Constructor options as exact ``int``/``str``/``bool``/``None`` values for the
    native module (see ``_index``)."""
    out = {}
    for key, value in kwargs.items():
        if value is None:
            out[key] = None
        elif key == "policy_map":
            out[key] = _policy_map(value)
        elif key == "league":
            from tessera_rl._league import League

            if not isinstance(value, League):
                raise TypeError(f"league must be a tessera_rl.League, got {type(value).__name__}")
            out[key] = value._handle
        elif key in _STRING_OPTIONS:
            out[key] = str(value)
        elif key in _ASSET_OPTIONS:
            if isinstance(value, str):
                out[key] = str(value)
            elif isinstance(value, (bytes, bytearray, memoryview)):
                out[key] = bytes(value)
            else:
                raise TypeError(f"{key} must be a bundled asset name (str) or bytes, got {type(value).__name__}")
        elif key == "transforms":
            from tessera_rl import transforms as _transforms

            out[key] = _transforms.native_spec(value)
        elif key in _BOOL_OPTIONS:
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{key} must be a bool, got {type(value).__name__}")
            out[key] = bool(value)
        else:
            out[key] = _index(value, key)
    return out


def is_stale(array) -> bool:
    """Whether a borrowed output's borrow window has ended.

    ``array`` is an array returned in ``borrowed`` mode, or a NumPy view of
    one (slices, reshapes), or a ``memoryview`` of a native view. Owned
    arrays are never stale. Raw aliases NumPy does not link back to the
    exporter (for example a ``torch.from_dlpack`` tensor) cannot be
    tracked and raise ``TypeError``.
    """
    obj = array
    while obj is not None:
        if isinstance(obj, _native.OutputView):
            return obj.is_stale()
        # A host view (SPEC 14.4.4 item 3): its window's generation against
        # its pool's current one.
        view_generation = getattr(type(obj), "_tessera_view_generation", None)
        if view_generation is not None:
            return obj._tessera_view_generation() != obj._tessera_current_generation()
        if isinstance(obj, (bytes, bytearray)):
            return False
        if isinstance(obj, memoryview):
            obj = obj.obj
        elif isinstance(obj, np.ndarray):
            obj = obj.base
        else:
            break
    raise TypeError("not a Tessera output: its staleness cannot be tracked")


#: Pool options that change how a pool runs, not what its records are: a
#: shadow pool (`tessera_rl.viewer.watch`) drops them.
_EXECUTION_OPTIONS = frozenset(
    {"mode", "batch_size", "gather", "workers", "spin", "wait", "wait_cap_us", "chunk_size", "schedule"}
)


def make(env_id: str, num_envs: int = 1, seed: int = 0, outputs: str = "borrowed", **kwargs) -> Env:
    """Creates a pool of ``num_envs`` instances of ``env_id``.

    ``outputs`` is ``"borrowed"`` (default: read-only zero-copy views valid
    until the next call) or ``"owned"`` (fresh arrays). Pool options:
    ``max_episode_steps``, ``auto_reset`` (``"next_step"``,
    ``"same_step"`` or ``"disabled"``), ``chunk_size``, ``schedule``
    (``"static"``, the default, or ``"dynamic"``: workers claim chunks from a
    shared counter, which balances environments with variable step cost;
    outputs are identical either way), ``workers``
    (worker threads; by default one per physical core this process may use,
    at most one per chunk: ``physical_cores()``; ``workers=0`` steps
    inline on the caller in Sync mode, and one worker in the split-phase
    modes), ``spin`` (spins before a waiting worker or caller blocks, an int in
    ``0..2**32``; 2000 by default, nominally about 20 microseconds; a larger
    budget can cut wake-up latency at the cost of CPU time burnt while idle; it needs
    workers, so Sync mode with ``workers=0`` refuses it with
    ``ArgumentError``), ``wait`` (the worker wait policy:
    ``"fixed"``, the default, or ``"adaptive"``, where a worker keeps
    spinning through the caller's typical gap between steps, up to
    ``wait_cap_us`` microseconds, an int in ``1..=1000000``, 10000 by
    default; it keeps caches warm at the cost of about one core per worker
    for that gap; it needs workers, as ``spin`` does), ``mode`` (``"sync"``, the default; ``"pipelined"``, which needs at least
    two chunks; or ``"async"``, which needs ``batch_size`` and takes ``gather``,
    ``"gathered"`` by default or ``"indexed"``); see ``Env``. Tag also
    takes its config fields (``width``, ``height``, ``pursuers``,
    ``evaders``, ``slip``, ``radius``). The limit order book
    (``tessera/LimitOrderBook-v1``) takes every ``LobConfig`` field
    (``agents``, ``steps``, ``depth``, ``min_tick``, ``max_tick``,
    ``initial_price``, ``up_permille``, ``down_permille``, ``latency``,
    ``size_0`` to ``size_3``, ``max_position``, ``initial_cash``,
    ``allow_short_cash``, ``bankrupt``, ``noise``, ``reward_shift``); it
    truncates by itself after ``steps``. Connect Four takes no config;
    its ``max_episode_steps`` defaults to 64, above the 42 plies of a game.
    The reference grid world takes ``size``, ``max_steps`` and ``layout``
    (its wall map, a content-addressed asset: a bundled name, ``"open"``
    by default or ``"rooms8"``, or ``bytes`` with one ``.``/``#`` row per
    line; ``ConfigError`` if refused); it truncates by itself. Key and door takes ``size`` and ``agents``; its
    ``max_episode_steps`` defaults to 512. Snake takes ``size``, ``agents``
    and ``obs`` (``"pixels"`` or ``"state"``); its ``max_episode_steps`` defaults
    to 1000. Brick wall takes ``frameskip``, ``lives`` and ``obs``; its
    ``max_episode_steps`` defaults to 4000 decisions. Paddle duel takes
    ``agents``, ``frameskip``, ``max_score``, ``bounce`` and ``obs``; its
    ``max_episode_steps`` defaults to 5000 decisions. Mountain car takes no
    config; its ``max_episode_steps`` defaults to 200 (``MountainCar-v0``).
    Acrobot takes no config; its ``max_episode_steps`` defaults to 500. Pendulum takes no config; its ``max_episode_steps`` defaults to 200, and its action is one float32 in ``[-2, 2]``. Leduc hold'em takes no config
    either; its ``max_episode_steps`` defaults to 16, above the 8 actions of a hand. Othello takes no config
    either; its ``max_episode_steps`` defaults to 128, above the 119 plies
    of the longest game. Go 9x9 takes ``komi_half_points`` (default 15, that
    is 7.5); its ``max_episode_steps`` defaults to 200, above the cap of 162
    plies.

    ``transforms`` takes a list of ``tessera_rl.transforms`` objects
    (``FrameStack``, ``OneHot``, ``Normalize``), applied in the engine after
    ``observe``; ``layout()`` and the adapter spaces describe the
    transformed observation. Host environments refuse it with
    ``CapabilityError``.

    ``policy_map`` routes agent slots to policy groups: a list
    of group names, one per slot, or a dict from slot to name (other slots
    learners), each ``"learner"``, ``"external"`` (marked only: you still
    submit its actions), ``"heuristic"`` (the engine computes its action on
    the workers from the environment's scripted baseline; your rows for it
    are ignored) or ``"privileged_heuristic"`` (a full-state opponent for
    imperfect-information environments, for testing only). Heuristic slots
    run on Tag, Leduc, Lanterns, LimitOrderBook, Connect Four, Othello and
    Go 9x9, in every mode (``"sync"``, ``"pipelined"`` and ``"async"``); see
    ``Env.policy_groups`` and ``Env.heuristic_actions``. Host environments
    refuse it with ``CapabilityError``.

    Any other id is looked up among the host environments of installed
    third-party wheels (the ``tessera_rl.envs`` entry points, read once per
    process: restart after installing a wheel) and returns a ``HostEnv``.
    Host environments run in ``mode="sync"`` only.
    """
    _envs.warn_if_deprecated(env_id)
    makers = {
        "tessera/CartPole-v1": _native.make_cartpole,
        "tessera/MountainCar-v1": _native.make_mountaincar,
        "tessera/Acrobot-v1": _native.make_acrobot,
        "tessera/Pendulum-v1": _native.make_pendulum,
        "tessera/Tag-v1": _native.make_tag,
        "tessera/RTS-v1": _native.make_rts,
        "tessera/LimitOrderBook-v1": _native.make_lob,
        "tessera/Grid-v1": _native.make_grid,
        "tessera/GridKeyDoor-v1": _native.make_keydoor,
        "tessera/Gather-v1": _native.make_gather,
        "tessera/Snake-v1": _native.make_snake,
        "tessera/BrickWall-v1": _native.make_brickwall,
        "tessera/PaddleDuel-v1": _native.make_paddleduel,
        "tessera/Connect4-v1": _native.make_connect4,
        "tessera/Leduc-v1": _native.make_leduc,
        "tessera/Lanterns-v1": _native.make_lanterns,
        "tessera/Othello-v1": _native.make_othello,
        "tessera/Go9-v1": _native.make_go9,
        "tessera/JobShop-v1": _native.make_jobshop,
        "tessera/Vrp-v1": _native.make_vrp,
        "tessera/Survival-v1": _native.make_survival,
    }
    maker = makers.get(env_id)
    if maker is None:
        # A host environment of an installed third-party wheel (SPEC 14.4.1).
        from tessera_rl import _host

        return _host.make(env_id, makers, num_envs, seed, outputs, dict(kwargs))
    mode = str(kwargs.get("mode") or "sync")
    indexed = kwargs.get("gather") == "indexed"
    env = Env(maker(**native_kwargs(num_envs=num_envs, seed=seed, outputs=outputs, **kwargs)), mode, indexed)
    #: What `tessera_rl.viewer.watch` needs to build a one-instance shadow
    #: pool of the same configuration: the id and the options that are not
    #: about execution.
    env.env_id = env_id
    env._config = {k: v for k, v in kwargs.items() if k not in _EXECUTION_OPTIONS}
    return env

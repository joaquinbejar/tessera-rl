# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native environment API: a pool of instances with NumPy outputs.

Output modes (ADR-0003): ``outputs="borrowed"`` (the default here) returns
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

The read-only flag is a signal, not an enforcement (ADR-0003 rule 7): no
tensor is promised to be immutable, and writing is the caller's
responsibility. A write through any writable alias reaches the engine
buffer and corrupts the outputs until the next call overwrites them. To
modify outputs, use ``outputs="owned"`` or copy first (``tensor.clone()``,
``array.copy()``).
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import numpy as np

from tessera_rl import _native


class Step(NamedTuple):
    """The outputs of one call: borrowed read-only views or owned arrays.

    Per-agent arrays have shape ``(num_envs, agents, ...)``; episode flags
    have shape ``(num_envs,)``. ``masks`` is ``None`` for environments
    without action masks. ``acting`` marks the agents whose action the
    next step reads (ADR-0013): every alive agent in a simultaneous
    environment, exactly one in a sequential one while its game runs, and
    none on a finished board.

    Breaking change in P2-M1b: ``acting`` is the ninth field, so unpacking
    a ``Step`` into 8 names fails; access by name is stable.
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


class HalfStep(NamedTuple):
    """One half's outputs in Pipelined mode (ADR-0047).

    ``ids`` are the instances of the half; ``step`` holds their rows only,
    so its first dimension is ``len(ids)``. In ``borrowed`` mode the arrays
    stay valid until this half's next ``send``, or a ``reset``: the other
    half's ``send`` does not end them.
    """

    half: int
    ids: range
    step: Step


class AsyncStep(NamedTuple):
    """One batch in Async mode (ADR-0047).

    ``ids`` lists the batch's instances in the order they completed. With the
    default ``gather="gathered"`` (and in ``owned`` mode) ``step`` has one row
    per entry of ``ids``, in that order. With ``gather="indexed"`` every array
    of ``step`` has all ``num_envs`` rows and only the rows of ``ids`` belong
    to this batch: read those, the others may be running.
    """

    ids: np.ndarray
    step: Step


class Env:
    """A pool of ``num_envs`` instances of one environment.

    ``outputs`` is ``"borrowed"`` or ``"owned"`` (see the module notes).

    Concurrency: each native call is admitted on its own. A native call
    that starts while another native call on the same ``Env`` is running
    raises ``ConcurrencyError``. This is not a guarantee about whole Python
    method calls: the Python work around the native call (converting
    actions, wrapping outputs) can interleave with other threads, so
    serialize calls on one ``Env`` yourself.

    Split-phase modes (ADR-0047): a step stays in flight after ``send``
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
    deliberate deviation from ADR-0006 that ADR-0047 records. Pass
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
    and the next ``recv`` delivers those ids. ``snapshot(id)`` needs every sent
    id to have finished, not only ``id`` (issue #85): it raises a retryable
    ``ConcurrencyError`` until then, and returns an opaque record for
    ``restore``, not bytes. ``import_record`` and ``fork`` are refused.

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
    child raises ``ForkError``: use ``workers=0`` in Sync mode, or spawn.

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
        #: ``"simultaneous"``, ``"sequential"`` or ``"mixed"`` (ADR-0013).
        self.turns: str = layout["turns"]
        self.outputs: str = layout["outputs"]
        #: The instance ids of the two halves, in Pipelined mode; else ``None``.
        self.halves: tuple[range, range] | None = None
        if "halves" in layout:
            (a, b), (c, d) = layout["halves"]
            self.halves = (range(a, b), range(c, d))
        self._indexed: bool = indexed
        self._leaves = [(name or "obs", np.dtype(dtype), tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]

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
        """An opaque snapshot of every instance (split-phase modes). It
        restores only this environment and cannot be saved. All sent work must
        be finished (``ConcurrencyError``, retryable); arrays stay valid.
        """
        return self._native.snapshot_all()

    def restore_all(self, snapshot) -> list[HalfStep] | None:
        """Restores a snapshot of this environment. Pipelined: one
        ``HalfStep`` per half. Async: nothing; send next. Needs all sent work
        finished, except to recover a poisoned pool.
        """
        out = self._native.restore_all(snapshot)
        return None if out is None else [self._half(h) for h in out]

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
        )

    def snapshot(self, index: int) -> bytes:
        """The canonical record of instance ``index``; in Async mode an opaque
        record for ``restore`` (see the class notes)."""
        return self._native.snapshot(_index(index, "index"))

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

    def close(self) -> None:
        """Frees the pool; idempotent. Arrays already returned stay readable
        (borrowed views keep the engine's buffers alive) and never change."""
        self._native.close()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_I32 = np.iinfo(np.int32)


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
    only exact built-in types (ADR-0041 callouts), so user objects are
    converted here, under the caller's own floating-point controls."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, not a bool")
    try:
        return int(operator.index(value))
    except TypeError:
        raise TypeError(f"{what} must be an integer, got {type(value).__name__}") from None


#: Keyword arguments the native constructors take as strings.
_STRING_OPTIONS = {"auto_reset", "outputs", "mode", "gather", "obs"}


def native_kwargs(**kwargs) -> dict:
    """Constructor options as exact ``int``/``str``/``None`` values for the
    native module (see ``_index``)."""
    out = {}
    for key, value in kwargs.items():
        if value is None:
            out[key] = None
        elif key in _STRING_OPTIONS:
            out[key] = str(value)
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
        if isinstance(obj, (bytes, bytearray)):
            return False
        if isinstance(obj, memoryview):
            obj = obj.obj
        elif isinstance(obj, np.ndarray):
            obj = obj.base
        else:
            break
    raise TypeError("not a Tessera output: its staleness cannot be tracked")


def make(env_id: str, num_envs: int = 1, seed: int = 0, outputs: str = "borrowed", **kwargs) -> Env:
    """Creates a pool of ``num_envs`` instances of ``env_id``.

    ``outputs`` is ``"borrowed"`` (default: read-only zero-copy views valid
    until the next call) or ``"owned"`` (fresh arrays). Pool options:
    ``max_episode_steps``, ``auto_reset`` (``"next_step"``,
    ``"same_step"`` or ``"disabled"``), ``chunk_size``, ``workers``,
    ``mode`` (``"sync"``, the default; ``"pipelined"``, which needs at least
    two chunks; or ``"async"``, which needs ``batch_size`` and takes ``gather``,
    ``"gathered"`` by default or ``"indexed"``). Split-phase modes default to
    one worker; see ``Env``. Tag also
    takes its config fields (``width``, ``height``, ``pursuers``,
    ``evaders``, ``slip``, ``radius``). Connect Four takes no config;
    its ``max_episode_steps`` defaults to 64, above the 42 plies of a game.
    The reference grid world takes ``size`` and ``max_steps``; it truncates
    by itself. Key and door takes ``size`` and ``agents``; its
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
    """
    makers = {
        "tessera/CartPole-v1": _native.make_cartpole,
        "tessera/MountainCar-v1": _native.make_mountaincar,
        "tessera/Acrobot-v1": _native.make_acrobot,
        "tessera/Pendulum-v1": _native.make_pendulum,
        "tessera/Tag-v1": _native.make_tag,
        "tessera/Grid-v1": _native.make_grid,
        "tessera/GridKeyDoor-v1": _native.make_keydoor,
        "tessera/Snake-v1": _native.make_snake,
        "tessera/BrickWall-v1": _native.make_brickwall,
        "tessera/PaddleDuel-v1": _native.make_paddleduel,
        "tessera/Connect4-v1": _native.make_connect4,
        "tessera/Leduc-v1": _native.make_leduc,
        "tessera/Othello-v1": _native.make_othello,
        "tessera/Go9-v1": _native.make_go9,
    }
    try:
        maker = makers[env_id]
    except KeyError:
        raise _native.ArgumentError(f"unknown environment {env_id!r}; known: {sorted(makers)}") from None
    mode = str(kwargs.get("mode") or "sync")
    indexed = kwargs.get("gather") == "indexed"
    return Env(maker(**native_kwargs(num_envs=num_envs, seed=seed, outputs=outputs, **kwargs)), mode, indexed)

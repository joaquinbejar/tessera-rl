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


class Env:
    """A pool of ``num_envs`` instances of one environment.

    ``outputs`` is ``"borrowed"`` or ``"owned"`` (see the module notes).

    Concurrency: each native call is admitted on its own. A native call
    that starts while another native call on the same ``Env`` is running
    raises ``ConcurrencyError``. This is not a guarantee about whole Python
    method calls: the Python work around the native call (converting
    actions, wrapping outputs) can interleave with other threads, so
    serialize calls on one ``Env`` yourself.

    With ``mode="pipelined"`` (ADR-0047) the pool has two fixed halves,
    ``halves``, stepped with ``send`` and ``recv``; ``step`` raises
    ``ArgumentError``. Here one thread may ``send`` while another waits in
    ``recv``; a second concurrent ``recv`` raises ``ConcurrencyError``.
    ``snapshot``, ``import_record``, ``fork`` and ``reset`` with ids are
    not available in this mode yet; ``snapshot_all`` and ``restore_all``
    are.
    """

    def __init__(self, native: _native.Env):
        self._native = native
        #: The last split-phase delivery this object handed out.
        self._delivered = 0
        layout = native.layout()
        self.num_envs: int = layout["num_envs"]
        self.agents: int = layout["agents"]
        self.action_choices: int = layout["action_choices"]
        self._mask_len: int = layout["mask_len"]
        #: ``"simultaneous"``, ``"sequential"`` or ``"mixed"`` (ADR-0013).
        self.turns: str = layout["turns"]
        self.outputs: str = layout["outputs"]
        #: The instance ids of the two halves, in Pipelined mode; else ``None``.
        self.halves: tuple[range, range] | None = None
        if "halves" in layout:
            (a, b), (c, d) = layout["halves"]
            self.halves = (range(a, b), range(c, d))
        self._leaves = [(name or "obs", np.dtype(dtype), tuple(shape)) for name, dtype, shape, *_ in layout["leaves"]]

    def reset(self, seed: int | None = None) -> Step | list[HalfStep]:
        """Resets every instance; a new seed base if ``seed`` is given.

        In Pipelined mode it returns one ``HalfStep`` per half, and it needs
        every sent half received first (``ConcurrencyError``), except to
        recover a poisoned pool.
        """
        out = self._native.reset() if seed is None else self._native.reset(_index(seed, "seed"))
        if self.halves is not None:
            return [self._half(h) for h in out]
        return self._wrap(out)

    def send(self, half: int, actions) -> None:
        """Starts the next step of ``half`` (Pipelined mode). ``actions`` has
        one int per agent row of the half, ``(len(halves[half]), agents)``,
        under the rules of ``step``. Ends the borrow window of that half's
        arrays. ``ConcurrencyError`` if the half is two cycles ahead of what
        the other half has finished.
        """
        acts = _discrete_actions(actions)
        self._native.send(_index(half, "half"), acts.tobytes(), self._delivered)

    def recv(self) -> HalfStep:
        """Waits for the oldest sent half and returns its outputs.

        The wait is interruptible: an exception from a signal handler
        (``KeyboardInterrupt``) leaves the step in flight, and a later
        ``recv`` returns it. An exception raised after the native call
        returned loses nothing either: the next ``recv`` returns that half
        again, and until then it counts as in flight for ``send``.

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
        until ``reset``, ``restore_all`` or ``close``.
        """
        out = self._native.recv(self._delivered)
        step = self._half(out)
        self._delivered = out["seq"]
        return step

    def snapshot_all(self) -> _native.PoolSnapshot:
        """An opaque snapshot of every instance and the cycles (Pipelined
        mode). It restores only this environment and cannot be saved. No sent
        half may be unreceived (``ConcurrencyError``); arrays stay valid.
        """
        return self._native.snapshot_all()

    def restore_all(self, snapshot) -> list[HalfStep]:
        """Restores a snapshot of this environment; one ``HalfStep`` per half,
        as ``reset``. Needs every sent half received, except to recover a
        poisoned pool.
        """
        return [self._half(h) for h in self._native.restore_all(snapshot)]

    def _half(self, out: dict) -> HalfStep:
        ids = range(out["start"], out["end"])
        return HalfStep(out["half"], ids, self._wrap(out["outputs"], len(ids)))

    def step(self, actions) -> Step:
        """One step. ``actions`` has one int per agent row, ``(num_envs, agents)``.

        Actions are discrete: every value must be an exact integer in the
        int32 range (integer dtypes, or floats with no fractional part).
        Anything else raises ``ActionError`` before the pool is touched.
        """
        acts = _discrete_actions(actions)
        if acts.size != self.num_envs * self.agents:
            raise _native.ActionError(f"expected {self.num_envs * self.agents} actions, got {acts.size}")
        return self._wrap(self._native.step(acts.tobytes()))

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
        """The canonical record of instance ``index``."""
        return self._native.snapshot(_index(index, "index"))

    def import_record(self, index: int, record: bytes) -> None:
        """Imports an (untrusted) record into instance ``index``."""
        self._native.import_record(_index(index, "index"), bytes(record))

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
_STRING_OPTIONS = {"auto_reset", "outputs", "mode"}


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
    ``mode`` (``"sync"``, the default, or ``"pipelined"``, which takes
    ``workers``, 1 by default, and needs at least two chunks; ``"async"`` is
    not available yet). Tag also
    takes its config fields (``width``, ``height``, ``pursuers``,
    ``evaders``, ``slip``, ``radius``). Connect Four takes no config;
    its ``max_episode_steps`` defaults to 64, above the 42 plies of a game.
    Leduc hold'em takes no config either; its ``max_episode_steps``
    defaults to 16, above the 8 actions of a hand.
    """
    makers = {
        "tessera/CartPole-v1": _native.make_cartpole,
        "tessera/Tag-v1": _native.make_tag,
        "tessera/Connect4-v1": _native.make_connect4,
        "tessera/Leduc-v1": _native.make_leduc,
    }
    try:
        maker = makers[env_id]
    except KeyError:
        raise _native.ArgumentError(f"unknown environment {env_id!r}; known: {sorted(makers)}") from None
    return Env(maker(**native_kwargs(num_envs=num_envs, seed=seed, outputs=outputs, **kwargs)))

# SPDX-License-Identifier: MIT OR Apache-2.0
"""Outputs on a CUDA device: ``make(..., device="cuda:N")``.

The pool writes its outputs into pinned host memory (``PinnedAllocator``).
After every call that returns outputs, the outputs are copied to
preallocated device tensors on a dedicated stream, one asynchronous copy
per output field, and an event is recorded after the copies. The caller's
current stream waits on that event before the device tensors are
returned, so work queued on it later sees the copied values.

The borrow window: the host buffers must not change while a copy reads
them, so the next call into the pool first waits on the event
(``Event.synchronize``). In ``mode="sync"`` the copy therefore overlaps
the caller's own work between calls, not the pool's next step. In
``mode="pipelined"`` each half has its own event: ``send(h)`` waits only
on half ``h``'s copy, so one half's copy overlaps the other half's step.
``mode="async"`` is not supported.

The device tensors are overwritten by the next copy, exactly as borrowed
host arrays are by the next call. ``host_outputs()`` returns the host
arrays of the last result, under the same window.

Releasing the host memory waits on the last copy's event first, so the
memory is never freed while a copy reads it. That wait holds the
interpreter, and is bounded by one copy of the outputs.

Memory is bounded per pool: the pool allocates its pinned output storage
once, at construction, and the session allocates one device buffer of the
same size per pinned range, the first time an output in it is copied.
Every result, whatever rows it spans (a half, ``reset(ids=...)``), copies
into its own bytes of that buffer; nothing is allocated per call. The
pinned memory lives until the pool, every export and the session are gone.

Everything device-specific sits behind a small backend interface
(``Backend``, ``Stream``, ``Event``) so the window logic runs without CUDA;
``TorchBackend`` is the real one. ``torch`` is imported only when a
device is asked for.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

import numpy as np

from tessera_rl import _native
from tessera_rl._env import Env, HalfStep, Step

_DEVICE = re.compile(r"cuda:(\d+)")


class Stream(Protocol):
    """A device stream: copies queued on it run in order."""


class Event(Protocol):
    """A marker recorded on a stream after its queued copies."""

    def record(self, stream: Any) -> None:
        """Marks the point after everything queued on ``stream`` so far."""

    def synchronize(self) -> None:
        """Blocks until the recorded work has completed."""


class Backend(Protocol):
    """The device operations the copy logic needs."""

    def pinned(self, nbytes: int) -> tuple[int, Any]:
        """``nbytes`` of page-locked host memory: ``(address, handle)``;
        the memory lives as long as ``handle``."""

    def source(self, handle: Any, offset: int, nbytes: int) -> Any:
        """The bytes ``offset..offset + nbytes`` of a ``pinned`` handle, as a
        copy source."""

    def device_buffer(self, nbytes: int) -> Any:
        """A device buffer of ``nbytes`` bytes."""

    def device_view(self, raw: Any, offset: int, nbytes: int, dtype: np.dtype, shape: tuple[int, ...]) -> tuple[Any, Any]:
        """The bytes ``offset..offset + nbytes`` of a ``device_buffer``:
        ``(bytes, typed)``, the copy destination and the tensor of ``dtype``
        and ``shape`` over it that the caller receives."""

    def stream(self) -> Any:
        """A new dedicated stream."""

    def event(self) -> Any:
        """A new event."""

    def copy_async(self, dst: Any, src: Any, stream: Any) -> None:
        """Queues the copy of ``src`` into ``dst`` on ``stream``."""

    def stream_waits_current(self, stream: Any) -> None:
        """Makes ``stream`` wait for the work queued so far on the caller's
        current stream (which may still read the device tensors)."""

    def current_waits(self, event: Any) -> None:
        """Makes the caller's current stream wait on ``event``."""


_TORCH_DTYPES = {
    "bool": "bool",
    "uint8": "uint8",
    "int8": "int8",
    "uint16": "uint16",
    "int16": "int16",
    "uint32": "uint32",
    "int32": "int32",
    "uint64": "uint64",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


class TorchBackend:
    """The CUDA backend, over ``torch``."""

    def __init__(self, torch: Any, device: str):
        self._torch = torch
        self.device = torch.device(device)

    def pinned(self, nbytes: int) -> tuple[int, Any]:
        t = self._torch.empty(nbytes, dtype=self._torch.uint8, pin_memory=True)
        return int(t.data_ptr()), t

    def source(self, handle: Any, offset: int, nbytes: int) -> Any:
        return handle[offset : offset + nbytes]

    def device_buffer(self, nbytes: int) -> Any:
        return self._torch.empty(nbytes, dtype=self._torch.uint8, device=self.device)

    def device_view(self, raw: Any, offset: int, nbytes: int, dtype: np.dtype, shape: tuple[int, ...]) -> tuple[Any, Any]:
        part = raw[offset : offset + nbytes]
        return part, part.view(getattr(self._torch, _TORCH_DTYPES[np.dtype(dtype).name])).view(shape)

    def stream(self) -> Any:
        return self._torch.cuda.Stream(device=self.device)

    def event(self) -> Any:
        return self._torch.cuda.Event()

    def copy_async(self, dst: Any, src: Any, stream: Any) -> None:
        with self._torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)

    def stream_waits_current(self, stream: Any) -> None:
        stream.wait_stream(self._torch.cuda.current_stream(self.device))

    def current_waits(self, event: Any) -> None:
        self._torch.cuda.current_stream(self.device).wait_event(event)


def torch_backend(device: str) -> TorchBackend:
    """The backend of ``device`` (``"cuda:N"``): ``CapabilityError`` unless
    ``torch`` with CUDA is importable, never a silent fallback to unpinned
    memory."""
    match = _DEVICE.fullmatch(device)
    if match is None:
        raise _native.ArgumentError(f"device must be 'cuda:N', not {device!r}")
    message = f"device={device!r} needs torch with CUDA"
    try:
        import torch
    except ImportError:
        raise _native.CapabilityError(message) from None
    if not torch.cuda.is_available():
        raise _native.CapabilityError(message)
    if int(match.group(1)) >= torch.cuda.device_count():
        raise _native.ArgumentError(f"device={device!r}: only {torch.cuda.device_count()} CUDA devices")
    return TorchBackend(torch, device)


class _Pending:
    """The last event of each half (``None`` in Sync mode), shared by the
    session and every owner of its allocator."""

    def __init__(self) -> None:
        self.events: dict[int | None, Any] = {}

    def wait(self, key: int | None = None, every: bool = True) -> None:
        if every:
            keys = list(self.events)
        else:
            keys = [key] if key in self.events else []
        for k in keys:
            event = self.events.pop(k)
            event.synchronize()


class _Owner:
    """Keeps one pinned allocation alive. The pool releases it after its
    last export; ``wait`` first waits on the last copy that reads it."""

    __slots__ = ("_handle", "_pending", "__weakref__")

    def __init__(self, handle: Any, pending: _Pending):
        self._handle = handle
        self._pending = pending

    def wait(self) -> None:
        self._pending.wait()

    def __del__(self) -> None:
        # The pool calls `wait` before its last reference goes; this covers
        # an owner whose last reference is dropped elsewhere.
        try:
            self._pending.wait()
        except Exception:  # noqa: BLE001 - a finalizer must not raise
            pass


class PinnedAllocator:
    """The pool's output allocator: pinned host memory from ``backend``,
    over-allocated by ``align`` and offset to an ``align`` boundary (pinned
    pages are page-aligned, not necessarily 16 KiB-aligned). The backing
    handle stays in the owner and in this allocator, never in user code."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self.pending = _Pending()
        #: ``(start, length, handle, offset of start in handle)`` per allocation.
        self._ranges: list[tuple[int, int, Any, int]] = []

    def allocate(self, length: int, align: int) -> tuple[int, _Owner]:
        base, handle = self._backend.pinned(length + align)
        start = (base + align - 1) // align * align
        self._ranges.append((start, length, handle, start - base))
        return start, _Owner(handle, self.pending)

    def locate(self, address: int, nbytes: int) -> tuple[int, int]:
        """``(range index, offset in that range)`` of the host bytes at
        ``address``."""
        for index, (start, length, _handle, _offset) in enumerate(self._ranges):
            if start <= address and address + nbytes <= start + length:
                return index, address - start
        raise _native.TesseraError("an output is outside the pinned output storage")

    def range_length(self, index: int) -> int:
        return self._ranges[index][1]

    def source(self, index: int, offset: int, nbytes: int) -> Any:
        """The copy source of ``nbytes`` at ``offset`` in range ``index``."""
        _start, _length, handle, base = self._ranges[index]
        return self._backend.source(handle, base + offset, nbytes)


def _address(array: np.ndarray) -> int:
    return int(array.__array_interface__["data"][0])


class _Session:
    """The copies and the window of one pool."""

    def __init__(self, backend: Backend, allocator: PinnedAllocator):
        self._backend = backend
        self._allocator = allocator
        self._pending = allocator.pending
        self._stream = backend.stream()
        self._events: dict[int | None, Any] = {}
        #: One full-size device buffer per pinned range: an output copies
        #: into its own bytes of it, whatever rows it spans, so device memory
        #: is bounded by the pool's output storage (allocated once).
        self._buffers: dict[int, Any] = {}

    def before(self, name: str, args: tuple) -> None:
        """Ends the window before a pool call: waits on the copies that read
        what the call may write."""
        if name == "send" and args and isinstance(args[0], int):
            self._pending.wait(args[0], every=False)
        elif name == "recv":
            # recv writes no rows: only a send does, for its own half.
            return
        else:
            self._pending.wait()

    def _device(self, array: np.ndarray) -> Any:
        if not array.flags.c_contiguous:
            raise _native.TesseraError("a device output must be a contiguous host array")
        index, offset = self._allocator.locate(_address(array), array.nbytes)
        raw = self._buffers.get(index)
        if raw is None:
            raw = self._backend.device_buffer(self._allocator.range_length(index))
            self._buffers[index] = raw
        source = self._allocator.source(index, offset, array.nbytes)
        dst, typed = self._backend.device_view(raw, offset, array.nbytes, array.dtype, array.shape)
        self._backend.copy_async(dst, source, self._stream)
        return typed

    def to_device(self, step: Step, half: int | None = None) -> Step:
        """Queues the copies of ``step``'s outputs, records the half's
        event, makes the current stream wait on it, and only then returns
        the device outputs."""
        backend = self._backend
        backend.stream_waits_current(self._stream)
        obs = {name: self._device(v) for name, v in step.observations.items()}
        fields = {
            f: None if getattr(step, f) is None else self._device(getattr(step, f))
            for f in Step._fields
            if f != "observations"
        }
        event = self._events.get(half)
        if event is None:
            event = backend.event()
            self._events[half] = event
        event.record(self._stream)
        self._pending.events[half] = event
        backend.current_waits(event)
        return Step(observations=obs, **fields)

    def close(self) -> None:
        self._pending.wait()

    def __del__(self) -> None:
        try:
            self._pending.wait()
        except Exception:  # noqa: BLE001 - a finalizer must not raise
            pass


class _Gated:
    """The native pool behind the window: every call waits first on the
    copies that read what it may write."""

    __slots__ = ("_inner", "_session")

    def __init__(self, inner: Any, session: _Session):
        self._inner = inner
        self._session = session

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        session = self._session

        def call(*args: Any, **kwargs: Any) -> Any:
            session.before(name, args)
            return attr(*args, **kwargs)

        return call


class DeviceEnv(Env):
    """An :class:`~tessera_rl.Env` whose outputs are device tensors (see the
    module notes). ``host_outputs()`` returns the host arrays of the last
    result."""

    _session: _Session
    _host: Any

    def _attach(self, session: _Session) -> None:
        self._session = session
        self._host = None
        self._native = _Gated(self._native, session)  # type: ignore[assignment]

    def _out(self, result: Any) -> Any:
        self._host = result
        if result is None:
            return None
        if isinstance(result, list):
            return [self._out_half(h) for h in result]
        if isinstance(result, HalfStep):
            return self._out_half(result)
        return self._session.to_device(result)

    def _out_half(self, h: HalfStep) -> HalfStep:
        return HalfStep(h.half, h.ids, self._session.to_device(h.step, h.half))

    def host_outputs(self) -> Any:
        """The host arrays of the last result (``Step``, ``HalfStep`` or a
        list of them), valid under the same window as the device tensors."""
        return self._host

    def reset(self, seed: int | None = None, ids=None) -> Any:
        return self._out(super().reset(seed, ids))

    def reset_from(self, ids, entries) -> Any:
        return self._out(super().reset_from(ids, entries))

    def step(self, actions) -> Any:
        return self._out(super().step(actions))

    def recv(self) -> Any:
        return self._out(super().recv())

    def restore_all(self, snapshot) -> Any:
        return self._out(super().restore_all(snapshot))

    def restore_pool(self, snapshot, archive=None) -> Any:
        return self._out(super().restore_pool(snapshot, archive))

    def close(self) -> None:
        # Redundant today: the native close below waits through `_Gated`.
        # Kept as defence in depth, so close never frees under a pending
        # copy if a later change stops gating close.
        self._session.close()
        super().close()


def make(maker: Any, native_kwargs: dict, mode: str, indexed: bool, device: str, backend: Backend | None) -> DeviceEnv:
    """A device pool (``tessera_rl.make(..., device=...)``)."""
    if mode == "async":
        raise _native.CapabilityError("device= needs mode='sync' or mode='pipelined'")
    if native_kwargs.get("outputs", "borrowed") != "borrowed":
        raise _native.CapabilityError("device= needs outputs='borrowed'")
    if backend is None:
        backend = torch_backend(device)
    allocator = PinnedAllocator(backend)
    env = DeviceEnv(maker(**native_kwargs, output_allocator=allocator), mode, indexed)
    env._attach(_Session(backend, allocator))
    return env

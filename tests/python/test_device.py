# SPDX-License-Identifier: MIT OR Apache-2.0
"""``make(..., device=...)`` without CUDA: the fallback, the no-torch
default, the Python output allocator, and the copy window on a fake
backend whose events complete only when the test releases them."""

import gc
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _cuda

CARTPOLE = "tessera/CartPole-v1"
BLOCKED = 0.3


class FakeStream:
    def __init__(self):
        self.queue = []


class FakeEvent:
    def __init__(self, backend):
        self.backend = backend
        self.copies = []
        self.gate = threading.Event()
        self.gate.set()
        self.syncs = 0

    def record(self, stream):
        self.copies, stream.queue = stream.queue, []
        self.backend.log.append("record")
        if self.backend.auto:
            self._run()
        else:
            self.gate.clear()

    def _run(self):
        for dst, src in self.copies:
            dst[:] = src
        self.copies = []

    def release(self):
        self.gate.set()

    def synchronize(self):
        self.backend.log.append("sync")
        assert self.gate.wait(timeout=30), "the test never released the event"
        self._run()
        self.syncs += 1
        self.backend.log.append("synced")


class FakeBackend:
    """Host memory from NumPy (ctypes addresses), "device" buffers as NumPy
    arrays, copies run when their event completes."""

    def __init__(self, auto=True):
        self.auto = auto
        self.log = []
        self.handles = []

    def pinned(self, nbytes):
        arr = np.zeros(nbytes, dtype=np.uint8)
        self.handles.append(arr)
        return arr.ctypes.data, arr

    def source(self, handle, offset, nbytes):
        return handle[offset : offset + nbytes]

    def device_buffer(self, nbytes, dtype, shape):
        raw = np.zeros(nbytes, dtype=np.uint8)
        return raw, raw.view(dtype).reshape(shape)

    def stream(self):
        return FakeStream()

    def event(self):
        return FakeEvent(self)

    def copy_async(self, dst, src, stream):
        stream.queue.append((dst, src))
        self.log.append("copy")

    def stream_waits_current(self, stream):
        self.log.append("stream_waits_current")

    def current_waits(self, event):
        self.log.append("current_waits")


class NumpyAllocator:
    """A plain Python output allocator over a NumPy buffer."""

    def __init__(self, misalign=0, result=None):
        self.misalign = misalign
        self.result = result
        self.owners = []

    def allocate(self, length, align):
        if self.result is not None:
            return self.result
        buf = np.zeros(length + 2 * align, dtype=np.uint8)
        start = (buf.ctypes.data + align - 1) // align * align + self.misalign
        self.owners.append(buf)
        return start, buf


def _fields(step):
    out = {f"obs.{k}": np.asarray(v) for k, v in step.observations.items()}
    for f in step._fields:
        if f != "observations" and getattr(step, f) is not None:
            out[f] = np.asarray(getattr(step, f))
    return out


def _actions(env):
    return np.zeros((env.num_envs, env.agents), dtype=np.int32)


def _assert_same(a, b):
    fa, fb = _fields(a), _fields(b)
    assert fa.keys() == fb.keys()
    for k in fa:
        assert fa[k].dtype == fb[k].dtype, k
        assert fa[k].tobytes() == fb[k].tobytes(), k


def _blocked(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    time.sleep(BLOCKED)
    return t


ENVS = [
    (CARTPOLE, {}),
    ("tessera/Snake-v1", {"obs": "pixels"}),
    ("tessera/Connect4-v1", {}),
]


@pytest.mark.parametrize(("env_id", "cfg"), ENVS, ids=["cartpole", "snake-pixels", "connect4"])
def test_device_python_allocator_outputs_byte_identical(env_id, cfg):
    plain = tessera_rl.make(env_id, num_envs=8, seed=3, workers=0, **cfg)
    alloc = NumpyAllocator()
    ext = tessera_rl.make(env_id, num_envs=8, seed=3, workers=0, output_allocator=alloc, **cfg)
    assert len(alloc.owners) >= 1
    _assert_same(plain.reset(), ext.reset())
    for _ in range(5):
        _assert_same(plain.step(_actions(plain)), ext.step(_actions(ext)))
    plain.close()
    ext.close()


@pytest.mark.parametrize(("env_id", "cfg"), ENVS, ids=["cartpole", "snake-pixels", "connect4"])
def test_device_fake_backend_device_equals_host_and_default(env_id, cfg):
    plain = tessera_rl.make(env_id, num_envs=8, seed=5, workers=0, **cfg)
    dev = tessera_rl.make(env_id, num_envs=8, seed=5, workers=0, device="cuda:0", _device_backend=FakeBackend(), **cfg)
    pairs = [(plain.reset(), dev.reset())]
    _assert_same(pairs[0][0], dev.host_outputs())
    for _ in range(5):
        pairs.append((plain.step(_actions(plain)), dev.step(_actions(dev))))
        _assert_same(pairs[-1][0], dev.host_outputs())
    for p, d in pairs[-1:]:
        _assert_same(p, d)
    dev.close()
    plain.close()


def test_device_next_call_blocks_until_event_completes():
    backend = FakeBackend(auto=False)
    env = tessera_rl.make(CARTPOLE, num_envs=4, workers=0, device="cuda:0", _device_backend=backend)
    env.reset()
    event = env._session._events[None]
    t = _blocked(lambda: env.step(_actions(env)))
    assert t.is_alive(), "the next call ran while the copy was pending"
    assert backend.log.count("copy") == len(_fields(env.host_outputs()))
    event.release()
    t.join(timeout=30)
    assert not t.is_alive()
    # The step recorded the same event again; let it complete.
    event.release()
    env.close()


def test_device_tensors_returned_after_wait_queued():
    backend = FakeBackend()
    env = tessera_rl.make(CARTPOLE, num_envs=4, workers=0, device="cuda:0", _device_backend=backend)
    backend.log.clear()
    env.reset()
    n = len(_fields(env.host_outputs()))
    assert backend.log == ["stream_waits_current", *["copy"] * n, "record", "current_waits"]
    env.close()


def test_device_pipelined_one_event_per_half():
    backend = FakeBackend(auto=False)
    env = tessera_rl.make(
        CARTPOLE, num_envs=4, chunk_size=1, workers=1, mode="pipelined", device="cuda:0", _device_backend=backend
    )
    halves = env.reset()
    assert [h.half for h in halves] == [0, 1]
    events = env._session._events
    assert events[0] is not events[1]
    acts = np.zeros((len(env.halves[0]), env.agents), dtype=np.int32)
    t = _blocked(lambda: env.send(0, acts))
    assert t.is_alive(), "send(0) ran while half 0's copy was pending"
    events[0].release()
    t.join(timeout=30)
    # Half 1's copy is still pending: half 0 ran and is received meanwhile.
    assert not t.is_alive()
    assert not events[1].gate.is_set()
    h = env.recv()
    assert h.half == 0
    events[1].release()
    events[0].release()
    env.close()


def test_device_async_refused():
    with pytest.raises(tessera_rl.CapabilityError, match="device="):
        tessera_rl.make(CARTPOLE, num_envs=4, mode="async", batch_size=2, device="cuda:0", _device_backend=FakeBackend())


def test_device_native_async_with_allocator_refused():
    with pytest.raises(tessera_rl.CapabilityError, match="output allocator"):
        tessera_rl.make(CARTPOLE, num_envs=4, mode="async", batch_size=2, output_allocator=NumpyAllocator())


def test_device_owned_outputs_refused():
    with pytest.raises(tessera_rl.CapabilityError, match="borrowed"):
        tessera_rl.make(CARTPOLE, num_envs=2, outputs="owned", device="cuda:0", _device_backend=FakeBackend())


def test_device_host_environment_refused():
    with pytest.raises(tessera_rl.CapabilityError, match="host environments"):
        tessera_rl.make("nobody/NotInstalled-v1", device="cuda:0")


def test_device_close_waits_on_pending_event():
    backend = FakeBackend(auto=False)
    env = tessera_rl.make(CARTPOLE, num_envs=4, workers=0, device="cuda:0", _device_backend=backend)
    env.reset()
    event = env._session._events[None]
    t = _blocked(env.close)
    assert t.is_alive(), "close returned while the copy was pending"
    event.release()
    t.join(timeout=30)
    assert not t.is_alive()
    assert event.syncs == 1


def test_device_drop_waits_on_pending_event(monkeypatch):
    waits = []
    real_wait = _cuda._Owner.wait

    def wait(self):
        waits.append(self)
        real_wait(self)

    monkeypatch.setattr(_cuda._Owner, "wait", wait)
    backend = FakeBackend(auto=False)
    holder = [tessera_rl.make(CARTPOLE, num_envs=4, workers=0, device="cuda:0", _device_backend=backend)]
    holder[0].reset()
    event = holder[0]._session._events[None]

    def drop():
        holder.clear()
        gc.collect()

    t = _blocked(drop)
    assert t.is_alive(), "the pool was released while the copy was pending"
    event.release()
    t.join(timeout=30)
    assert not t.is_alive()
    assert event.syncs >= 1
    # The engine released its storage through the owner, which waited.
    gc.collect()
    assert waits


def test_device_owner_wait_waits_on_last_event():
    backend = FakeBackend(auto=False)
    alloc = _cuda.PinnedAllocator(backend)
    start, owner = alloc.allocate(64, 16384)
    assert start % 16384 == 0
    event = backend.event()
    event.record(FakeStream())
    alloc.pending.events[None] = event
    t = _blocked(owner.wait)
    assert t.is_alive()
    event.release()
    t.join(timeout=30)
    assert not t.is_alive()
    assert event.syncs == 1


@pytest.mark.parametrize(
    "alloc",
    [NumpyAllocator(misalign=64), NumpyAllocator(result=(0, None)), NumpyAllocator(result="no")],
    ids=["misaligned", "null", "not-a-tuple"],
)
def test_device_bad_allocation_raises_config_error(alloc):
    with pytest.raises(tessera_rl.ConfigError, match="output allocation refused"):
        tessera_rl.make(CARTPOLE, num_envs=2, workers=0, output_allocator=alloc)


def test_device_raising_allocator_raises_config_error():
    class Raising:
        def allocate(self, length, align):
            raise RuntimeError("no memory")

    with pytest.raises(tessera_rl.ConfigError, match="output allocation refused"):
        tessera_rl.make(CARTPOLE, num_envs=2, workers=0, output_allocator=Raising())


def test_device_without_cuda_raises_capability_error():
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(tessera_rl.CapabilityError, match=r"^device='cuda:1' needs torch with CUDA$"):
        tessera_rl.make(CARTPOLE, num_envs=2, device="cuda:1")


def test_device_bad_device_string_raises_argument_error():
    with pytest.raises(tessera_rl.ArgumentError, match="cuda:N"):
        tessera_rl.make(CARTPOLE, num_envs=2, device="gpu")


def test_device_default_does_not_import_torch():
    code = (
        "import sys, numpy as np, tessera_rl\n"
        "e = tessera_rl.make('tessera/CartPole-v1', num_envs=2, workers=0)\n"
        "e.reset(); e.step(np.zeros((2, 1), dtype=np.int32)); e.close()\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'tessera_rl._cuda' not in sys.modules, '_cuda imported'\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0, p.stderr


def test_device_allocator_source_picks_the_containing_allocation():
    backend = FakeBackend()
    alloc = _cuda.PinnedAllocator(backend)
    a, _owner_a = alloc.allocate(32768, 16384)
    b, _owner_b = alloc.allocate(32768, 16384)
    backend.handles[1][:] = 7
    src = alloc.source(b + 100, 16)
    assert src.tobytes() == bytes([7] * 16)
    assert alloc.source(a, 16).tobytes() == bytes(16)
    with pytest.raises(tessera_rl.TesseraError, match="outside"):
        alloc.source(b + 32768 - 8, 16)

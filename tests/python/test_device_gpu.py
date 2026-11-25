# SPDX-License-Identifier: MIT OR Apache-2.0
"""``make(..., device="cuda:0")`` on a real CUDA device (marker ``gpu``):
skipped without torch with CUDA. Run in a GPU-box window."""

import gc
import time

import numpy as np
import pytest

import tessera_rl

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

DEVICE = "cuda:0"


def _fields(step):
    out = {f"obs.{k}": v for k, v in step.observations.items()}
    for f in step._fields:
        if f != "observations" and getattr(step, f) is not None:
            out[f] = getattr(step, f)
    return out


def _actions(n, agents):
    return np.zeros((n, agents), dtype=np.int32)


@pytest.mark.parametrize(
    ("env_id", "cfg"),
    [("tessera/CartPole-v1", {}), ("tessera/Snake-v1", {"obs": "pixels"}), ("tessera/Go9-v1", {})],
    ids=["cartpole", "snake-pixels", "go9"],
)
def test_gpu_device_tensors_equal_host_outputs(env_id, cfg):
    env = tessera_rl.make(env_id, num_envs=16, seed=7, device=DEVICE, **cfg)
    for i in range(4):
        out = env.reset() if i == 0 else env.step(_actions(env.num_envs, env.agents))
        host = _fields(env.host_outputs())
        dev = _fields(out)
        assert host.keys() == dev.keys()
        torch.cuda.current_stream().synchronize()
        for k, d in dev.items():
            assert d.is_cuda, k
            assert d.cpu().numpy().tobytes() == np.asarray(host[k]).tobytes(), k
    env.close()


def _pipelined_loop(device, rounds):
    kwargs = {"device": device} if device else {}
    env = tessera_rl.make(
        "tessera/Snake-v1", num_envs=1024, chunk_size=64, workers=4, mode="pipelined", obs="pixels", **kwargs
    )
    env.reset()
    halves = env.halves
    for h in (0, 1):
        env.send(h, _actions(len(halves[h]), env.agents))
    start = time.perf_counter()
    for _ in range(rounds):
        r = env.recv()
        env.send(r.half, _actions(len(halves[r.half]), env.agents))
    env.recv()
    env.recv()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def _copy_time(rounds):
    env = tessera_rl.make("tessera/Snake-v1", num_envs=512, chunk_size=64, workers=4, obs="pixels", device=DEVICE)
    env.reset()
    total = 0.0
    for _ in range(rounds):
        out = env.step(_actions(env.num_envs, env.agents))
        t0 = time.perf_counter()
        env._session._pending.wait()
        total += time.perf_counter() - t0
        del out
    # Lower bound of one half's copy: the wait after the step returned.
    env.close()
    return total


def test_gpu_pipelined_copy_at_least_half_hidden():
    rounds = 200
    _pipelined_loop(DEVICE, 10)
    base = _pipelined_loop(None, rounds)
    with_device = _pipelined_loop(DEVICE, rounds)
    copy = _copy_time(rounds)
    exposed = max(0.0, with_device - base)
    hidden = 1.0 - exposed / copy if copy > 0 else 1.0
    print(f"base={base:.4f}s device={with_device:.4f}s copy={copy:.4f}s hidden={hidden:.3f}")
    assert hidden >= 0.5


def test_gpu_drop_with_copy_in_flight_keeps_device_data():
    env = tessera_rl.make("tessera/Snake-v1", num_envs=4096, workers=4, obs="pixels", device=DEVICE)
    env.reset()
    out = env.step(_actions(env.num_envs, env.agents))
    host = env.host_outputs()
    expected = {k: np.array(v, copy=True) for k, v in _fields(host).items()}
    nbytes = sum(v.nbytes for v in expected.values())
    dev = dict(_fields(out))
    # Drop the pool and every host export at once, before the copy can
    # have finished.
    del host, out, env
    gc.collect()
    junk = torch.empty(nbytes + 16384, dtype=torch.uint8, pin_memory=True)
    junk.fill_(0xFF)
    torch.cuda.synchronize()
    for k, d in dev.items():
        assert d.cpu().numpy().tobytes() == expected[k].tobytes(), k

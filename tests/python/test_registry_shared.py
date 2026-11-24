# SPDX-License-Identifier: MIT OR Apache-2.0
"""The engine registry shared across wheels (ADR-0005, SPEC 6.1, #349):
``tessera_rl`` and a host wheel (``make py-test-host``) count their active
engines in one counter at ``sys.modules["tessera_rl._registry"]``, so a pool
started while another wheel's pool is active runs unpinned. Pinning is read
back from the worker threads' affinity (Linux)."""

import ctypes
import gc
import os
import subprocess
import sys
import textwrap

import pytest

import tessera_rl

HOST_ID = "test/FactoryCounter-v1"
CAPSULE = b"tessera_rl._registry.counter_v1"
linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads thread affinity from /proc")


def _host():
    return pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")


def count() -> int:
    """The shared counter's value, read through the capsule."""
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    cap = sys.modules["tessera_rl._registry"].counter_v1
    return ctypes.c_size_t.from_address(get(cap, CAPSULE)).value


def worker_masks() -> list[frozenset[int]]:
    """The affinity of every Tessera worker thread of this process."""
    masks = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                if not f.read().startswith("tessera-worker"):
                    continue
            masks.append(frozenset(os.sched_getaffinity(int(tid))))
        except OSError:
            continue
    return masks


def pinned(masks):
    return sorted(next(iter(m)) for m in masks if len(m) == 1)


@pytest.fixture
def cores():
    _host()
    gc.collect()
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 4:
        pytest.skip("needs 4 cores")
    if count() != 0:
        pytest.skip("another engine is active in this process")
    return available


def test_registry_shared_both_wheels_use_one_capsule():
    _host()
    module = sys.modules["tessera_rl._registry"]
    assert type(module.counter_v1).__name__ == "PyCapsule"
    gc.collect()
    before = count()
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=4, workers=1)
    assert count() == before + 1
    env.close()
    del env
    gc.collect()
    assert count() == before


@linux
def test_registry_shared_pool_after_another_wheels_pool_runs_unpinned(cores):
    first = tessera_rl.make("tessera/CartPole-v1", num_envs=4, workers=2)
    assert count() == 1
    assert pinned(worker_masks()) == cores[:2]
    host = tessera_rl.make(HOST_ID, num_envs=2, workers=2, cpu_set=cores[-2:])
    assert count() == 2
    assert pinned(worker_masks()) == sorted(cores[:2] + cores[-2:])
    second = tessera_rl.make("tessera/CartPole-v1", num_envs=4, workers=2)
    assert count() == 3
    masks = worker_masks()
    # The second pool's two workers are unpinned: they keep the process mask.
    assert sum(1 for m in masks if m == frozenset(cores)) == 2
    assert pinned(masks) == sorted(cores[:2] + cores[-2:])
    for env in (second, host, first):
        env.close()
    del first, host, second, env
    gc.collect()
    assert count() == 0


@linux
def test_registry_shared_host_pool_first_unpins_a_tessera_pool(cores):
    host = tessera_rl.make(HOST_ID, num_envs=2, workers=2, cpu_set=cores[-2:])
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=4, workers=2)
    assert count() == 2
    masks = worker_masks()
    assert pinned(masks) == cores[-2:]
    assert sum(1 for m in masks if m == frozenset(cores)) == 2
    env.close()
    host.close()
    del env, host
    gc.collect()
    assert count() == 0


def test_registry_shared_unusable_entry_warns_and_counts_alone():
    code = textwrap.dedent(
        """
        import sys, types, warnings
        m = types.ModuleType("tessera_rl._registry")
        m.counter_v1 = 1
        sys.modules["tessera_rl._registry"] = m
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tessera_rl
            env = tessera_rl.make("tessera/CartPole-v1", num_envs=2, workers=1)
            env.close()
        print(sum("counts only its own engines" in str(w.message) for w in caught))
        print(sys.modules["tessera_rl._registry"].counter_v1)
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "1"], out.stderr

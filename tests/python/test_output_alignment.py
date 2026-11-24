"""Output buffer alignment (SPEC 5.2, #355): observation leaves start on a
16 KiB boundary (a multiple of the page size: 4 KiB x86-64, 16 KiB Apple
silicon), every other buffer on 64 bytes. Checked on the raw native outputs,
so every buffer is covered, terminal observations included (``same_step``).
"""

import numpy as np
import pytest

import tessera_rl

OUTPUT_ALIGN = 16 * 1024
OUTPUT_MIN_ALIGN = 64


def _buffers(out, prefix=""):
    if isinstance(out, dict):
        for key, value in out.items():
            yield from _buffers(value, f"{prefix}{key}.")
    elif isinstance(out, list | tuple):
        for k, value in enumerate(out):
            yield from _buffers(value, f"{prefix}{k}.")
    elif out is not None and not isinstance(out, int | float | str | bool):
        yield prefix.rstrip("."), np.asarray(out)


@pytest.mark.parametrize("auto_reset", ["next_step", "same_step"])
@pytest.mark.parametrize("env_id", ["tessera/CartPole-v1", "tessera/Tag-v1", "tessera/Connect4-v1"])
def test_outputs_aligned_to_spec(env_id, auto_reset):
    env = tessera_rl.make(env_id, num_envs=7, auto_reset=auto_reset)
    try:
        out = env._native.reset(0)
        seen = list(_buffers(out))
        assert len(seen) >= 8, [name for name, _ in seen]
        assert any(name.startswith("observations") for name, _ in seen), [n for n, _ in seen]
        if auto_reset == "same_step":
            assert any(name.startswith("terminal_observations") for name, _ in seen)
        for name, arr in seen:
            if arr.nbytes:
                obs = name.startswith(("observations", "terminal_observations"))
                align = OUTPUT_ALIGN if obs else OUTPUT_MIN_ALIGN
                assert arr.ctypes.data % align == 0, f"{env_id} {name} at {arr.ctypes.data:#x}, wanted {align}"
    finally:
        env.close()

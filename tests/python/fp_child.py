# SPDX-License-Identifier: MIT OR Apache-2.0
"""Child process for the floating-point boundary conformance tests
(ADR-0041): each scenario needs a fresh process, because the warning is
shown once per process and the policy is process-wide.

Usage: python fp_child.py SCENARIO. Prints one JSON object.
"""

import ctypes
import ctypes.util
import json
import platform
import sys
import warnings

import numpy as np

import tessera_rl as t

LIBM = ctypes.CDLL(ctypes.util.find_library("m") or None)
ARM = platform.machine() in ("arm64", "aarch64")
# <fenv.h> values: x86-64 (glibc and macOS) and aarch64 (glibc and macOS).
FE_TOWARDZERO = 0x00C00000 if ARM else 0x0C00
FE_TONEAREST = 0
FE_DIVBYZERO = 0x02 if ARM else 0x04


def set_round(mode):
    assert LIBM.fesetround(mode) == 0
    assert LIBM.fegetround() == mode


def run_episode(steps=60):
    """Every public binding once, plus a short episode; the bytes of all
    outputs and records."""
    env = t.make("tessera/CartPole-v1", num_envs=4, seed=5, outputs="owned")
    out = [env.reset().observations["obs"].tobytes()]
    for k in range(steps):
        s = env.step(np.full((4, 1), k % 2))
        out += [s.observations["obs"].tobytes(), s.rewards.tobytes(), s.episode_terminated.tobytes()]
    record = env.snapshot(0)
    env.import_record(1, record)
    env.fork(0, [2, 3])
    out += [env.snapshot(i) for i in range(4)]
    tag = t.make("tessera/Tag-v1", num_envs=2, seed=3)
    s = tag.reset()
    out.append(np.asarray(s.observations["window"]).tobytes())
    view = tag._native.outputs()["rewards"]
    out.append(bytes(memoryview(view)))
    cap = view.__dlpack__(max_version=(1, 0))
    del cap, view
    out += [t.version(), t.get_fp_policy(), env.outputs]
    env.close()
    tag.close()
    return [o if isinstance(o, str) else o.hex() for o in out]


def main(scenario):
    result = {"scenario": scenario}
    if scenario == "identical":
        default = run_episode()
        before = t.fp_non_canonical_entries()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_round(FE_TOWARDZERO)
            hostile = run_episode()
            result["round_after"] = LIBM.fegetround()
            set_round(FE_TONEAREST)
        result["identical"] = hostile == default
        result["counted"] = t.fp_non_canonical_entries() - before
        result["warnings"] = sum(issubclass(w.category, RuntimeWarning) and "tessera_rl" in str(w.message) for w in caught)
        result["round_restored"] = result["round_after"] == FE_TOWARDZERO
    elif scenario == "flags":
        # Only native calls between raising and testing the flag: NumPy
        # clears the status flags in many of its operations.
        env = t.make("tessera/CartPole-v1", num_envs=2, seed=1)
        native = env._native
        native.reset()
        actions = np.ones(2, dtype="<i4").tobytes()
        all_flags = 0x9F if ARM else 0x3F
        LIBM.feclearexcept(all_flags)
        LIBM.feraiseexcept(FE_DIVBYZERO)
        result["control_kept"] = bool(LIBM.fetestexcept(FE_DIVBYZERO))
        set_round(FE_TOWARDZERO)
        native.step(actions)
        native.step(actions)
        result["flag_kept"] = bool(LIBM.fetestexcept(FE_DIVBYZERO))
        set_round(FE_TONEAREST)
    elif scenario == "reject":
        env = t.make("tessera/CartPole-v1", num_envs=2, seed=1)
        env.reset()
        view = env._native.outputs()["rewards"]
        cap = view.__dlpack__(max_version=(1, 0))
        before = env.snapshot(0)
        t.set_fp_policy("reject")
        set_round(FE_TOWARDZERO)
        errors = {}
        for name, call in (
            ("method", lambda: env.step(np.ones((2, 1)))),
            ("getbuffer", lambda: memoryview(view)),
            ("module_function", t.version),
        ):
            try:
                call()
                errors[name] = None
            except t.FloatingPointEnvironmentError:
                errors[name] = "FloatingPointEnvironmentError"
            except Exception as e:  # noqa: BLE001
                errors[name] = type(e).__name__
        # Cleanup entries never reject: the capsule destructor, the view's
        # and the environment's deallocation.
        del cap, view
        env_ref = env
        del env
        round_after = LIBM.fegetround()
        set_round(FE_TONEAREST)
        result["errors"] = errors
        result["unchanged"] = env_ref.snapshot(0) == before
        del env_ref
        result["round_restored"] = round_after == FE_TOWARDZERO
        t.set_fp_policy("canonicalize")
        result["policy"] = t.get_fp_policy()
    elif scenario == "werror":
        env = t.make("tessera/CartPole-v1", num_envs=2, seed=1)
        env.reset()
        before = env.snapshot(0)
        warnings.simplefilter("error")
        set_round(FE_TOWARDZERO)
        try:
            env.step(np.ones((2, 1)))
            result["raised"] = None
        except RuntimeWarning as e:
            result["raised"] = "RuntimeWarning" if "tessera_rl" in str(e) else str(e)
        set_round(FE_TONEAREST)
        warnings.simplefilter("default")
        result["unchanged"] = env.snapshot(0) == before
    elif scenario == "unmasked":
        # glibc only: unmask divide-by-zero and invalid in the caller.
        if not hasattr(LIBM, "feenableexcept"):
            result["skipped"] = "no feenableexcept"
        else:
            default = run_episode()
            LIBM.feenableexcept(FE_DIVBYZERO | 0x01)
            hostile = run_episode()
            result["enabled_after"] = LIBM.fegetexcept()
            LIBM.fedisableexcept(FE_DIVBYZERO | 0x01)
            result["identical"] = hostile == default
    elif scenario == "torch_ftz":
        import torch

        default = run_episode()
        result["supported"] = torch.set_flush_denormal(True)
        hostile = run_episode()
        torch.set_flush_denormal(False)
        result["identical"] = hostile == default
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])

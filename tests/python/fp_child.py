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

WINDOWS = sys.platform == "win32"
ARM = platform.machine().lower() in ("arm64", "aarch64")


class _Ucrt:
    """The <fenv.h> calls this file uses, on Windows, through ucrtbase's
    ``_controlfp_s``, ``_clearfp`` and ``_statusfp`` (on x64 they act on
    MXCSR). Its values are the ``_RC_*``, ``_SW_*`` and ``_EM_*`` constants
    of <float.h>; an exception is enabled when its ``_EM_*`` mask bit is
    clear."""

    MCW_RC = 0x00000300
    MCW_EM = 0x0008001F
    MCW_DN = 0x03000000
    DN_FLUSH = 0x01000000
    ALL_FLAGS = 0x1F  # FE_ALL_EXCEPT: the _SW_* status flags
    EM = {0x01: 0x10, 0x04: 0x08, 0x08: 0x04, 0x10: 0x02, 0x20: 0x01}

    def __init__(self):
        self.crt = ctypes.CDLL("ucrtbase")
        self.crt._controlfp_s.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        self.crt._controlfp_s.restype = ctypes.c_int
        self.crt._clearfp.restype = ctypes.c_uint
        self.crt._statusfp.restype = ctypes.c_uint
        for name in ("fegetexceptflag", "fesetexceptflag"):
            getattr(self.crt, name).argtypes = [ctypes.POINTER(ctypes.c_ulong), ctypes.c_int]
            getattr(self.crt, name).restype = ctypes.c_int

    def _control(self, new=0, mask=0):
        # _controlfp_s clears the MXCSR status flags when it writes: keep
        # the caller's sticky flags, as <fenv.h> calls do, by saving them
        # with fegetexceptflag before and restoring them with
        # fesetexceptflag after (UCRT exports both; fexcept_t is opaque).
        saved = ctypes.c_ulong(0)
        if mask:
            assert self.crt.fegetexceptflag(ctypes.byref(saved), self.ALL_FLAGS) == 0
        cur = ctypes.c_uint(0)
        assert self.crt._controlfp_s(ctypes.byref(cur), new, mask) == 0
        if mask:
            assert self.crt.fesetexceptflag(ctypes.byref(saved), self.ALL_FLAGS) == 0
        return cur.value

    def fesetround(self, mode):
        self._control(mode, self.MCW_RC)
        return 0

    def fegetround(self):
        return self._control() & self.MCW_RC

    def feclearexcept(self, _flags):
        self.crt._clearfp()
        return 0

    def feraiseexcept(self, flags):
        # UCRT has no exported feraiseexcept (it is inline in <fenv.h>):
        # raise through a real operation, as the host would: 1/0 in SSE.
        if flags & 0x08:
            import numpy

            with numpy.errstate(divide="ignore"):
                numpy.float64(1.0) / numpy.float64(0.0)
        return 0

    def fetestexcept(self, flags):
        return self.crt._statusfp() & flags

    def _em(self, flags):
        return sum(em for sw, em in self.EM.items() if flags & sw)

    def feenableexcept(self, flags):
        self._control(0, self._em(flags))
        return 0

    def fedisableexcept(self, flags):
        em = self._em(flags)
        self._control(em, em)
        return 0

    def fegetexcept(self):
        masked = self._control() & self.MCW_EM
        return sum(sw for sw, em in self.EM.items() if not masked & em)

    def set_flush_denormals(self, on):
        self._control(self.DN_FLUSH if on else 0, self.MCW_DN)


if WINDOWS:
    LIBM = _Ucrt()
    # <float.h> values (_RC_*, _SW_*), x64.
    FE_TONEAREST = 0x000
    FE_UPWARD = 0x200
    FE_TOWARDZERO = 0x300
    FE_DIVBYZERO = 0x08
    FE_INVALID = 0x10
else:
    LIBM = ctypes.CDLL(ctypes.util.find_library("m") or None)
    # <fenv.h> values: x86-64 (glibc and macOS) and aarch64 (glibc and macOS).
    FE_TOWARDZERO = 0x00C00000 if ARM else 0x0C00
    FE_TONEAREST = 0
    FE_UPWARD = 0x00400000 if ARM else 0x0800
    FE_DIVBYZERO = 0x02 if ARM else 0x04
    FE_INVALID = 0x01


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
            LIBM.feenableexcept(FE_DIVBYZERO | FE_INVALID)
            hostile = run_episode()
            result["enabled_after"] = LIBM.fegetexcept()
            LIBM.fedisableexcept(FE_DIVBYZERO | FE_INVALID)
            result["identical"] = hostile == default
    elif scenario == "probe":
        # codex R-PR26: an __index__ that changes the controls. The native
        # module accepts only exact ints, so it never calls __index__: the
        # conversion fails with TypeError before any user code runs.
        from tessera_rl import _native

        observed = []

        class Change:
            def __index__(self):
                observed.append(LIBM.fegetround())
                LIBM.fesetround(FE_TOWARDZERO)
                return 1

        class Observe:
            def __index__(self):
                observed.append(LIBM.fegetround())
                return 1

        set_round(FE_TONEAREST)
        try:
            _native.make_cartpole(num_envs=Change(), seed=Observe())
            result["raised"] = None
        except TypeError as e:
            result["raised"] = "TypeError"
            result["message"] = str(e)
        result["user_code_ran"] = list(observed)
        result["round_after"] = LIBM.fegetround()
        set_round(FE_TONEAREST)
        # Through the Python layer, __index__ runs in Python, under the
        # caller's own controls, before the native call.
        env = t.make("tessera/CartPole-v1", num_envs=Observe(), seed=Observe())
        result["python_layer_ok"] = env.num_envs == 1
    elif scenario == "gc_finalizer":
        # A __del__ that changes the controls, run by gc.collect inside a
        # GC-tracked allocation of the native module (a callout): Rust must
        # not resume under it. Results identical to a run without it.
        import gc

        hits = []

        class Bomb:
            def __del__(self):
                hits.append(LIBM.fegetround())
                LIBM.fesetround(FE_TOWARDZERO)

        def episode(bombs):
            env = t.make("tessera/CartPole-v1", num_envs=8, seed=3, outputs="owned")
            native = env._native
            out = [bytes(native.reset()["observations"][0])]
            acts = np.ones(8, dtype="<i4").tobytes()
            for _ in range(40):
                if bombs:
                    gc.disable()
                    # Drain the dict, list and tuple free lists so the
                    # native allocations are real GC allocations that can
                    # trigger a collection.
                    hold = [({}, [], (i, i)) for i in range(400)]
                    for _ in range(50):
                        b = Bomb()
                        b.cycle = b
                        del b
                    gc.enable()
                    gc.set_threshold(1)
                o = native.step(acts)
                if bombs:
                    del hold
                    gc.set_threshold(700)
                    set_round(FE_TONEAREST)
                out.append(bytes(o["observations"][0]) + bytes(o["rewards"]))
            env.close()
            return out

        clean = episode(False)
        before = t.fp_non_canonical_entries()
        bombed = episode(True)
        result["finalizers_ran"] = len(hits)
        result["identical"] = bombed == clean
        result["counted"] = t.fp_non_canonical_entries() - before
        gc.collect()
    elif scenario in ("gc_inject", "gc_unwind"):
        # codex R-PR26 R2: a gc callback, run inside a native GC
        # allocation, finds the output dict under construction and stores
        # user objects in it and in its observations list. Every later drop
        # of those objects (a replaced dict value; on gc_unwind, the partly
        # built containers dropped by a failure) must run their finalizers
        # through a callout, under the caller's controls (FE_UPWARD here),
        # never under Rust's canonical ones.
        import gc

        from tessera_rl import _native

        warnings.simplefilter("ignore", RuntimeWarning)
        seen = []
        injected = []

        class Bomb:
            def __del__(self):
                seen.append(LIBM.fegetround())

        def callback(phase, info):
            if phase != "start":
                return
            for d in gc.get_objects():
                if type(d) is dict and "observations" in d and "masks" not in d and type(d["observations"]) is list:
                    d["rewards"] = Bomb()
                    d["agent_terminated"] = Bomb()
                    d["observations"].append(Bomb())
                    injected.append(id(d))

        env = _native.make_cartpole(num_envs=1, outputs="owned")
        env.reset()
        gc.collect()
        gc.disable()
        hold = [({}, [], (i, i)) for i in range(1000)]
        gc.callbacks.append(callback)
        gc.set_threshold(1, 1, 1)
        failed = 0
        try:
            for _ in range(30):
                if scenario == "gc_unwind":
                    # Fail at the 4th insertion: after observations and
                    # terminal_observations, before the remaining outputs.
                    _native._fail_after(4)
                set_round(FE_UPWARD)
                gc.enable()
                try:
                    env.outputs()
                except RuntimeError:
                    failed += 1
                gc.disable()
                set_round(FE_TONEAREST)
                if injected and seen:
                    break
        finally:
            gc.disable()
            gc.callbacks.remove(callback)
            gc.set_threshold(700, 10, 10)
            set_round(FE_TONEAREST)
            if scenario == "gc_unwind":
                _native._fail_after(0)
        del hold
        result["injected"] = len(injected)
        result["failed"] = failed
        result["finalizers"] = len(seen)
        result["all_caller_controls"] = all(m == FE_UPWARD for m in seen)
        result["modes"] = sorted(set(seen))
    elif scenario == "windows_ftz":
        # Windows: flush-to-zero and denormals-are-zero set through
        # _controlfp_s (_DN_FLUSH) by the caller; results identical.
        if not WINDOWS:
            result["skipped"] = "Windows only"
        else:
            clean = run_episode()
            LIBM.set_flush_denormals(True)
            try:
                hostile = run_episode()
                result["dn_after"] = LIBM._control() & LIBM.MCW_DN
            finally:
                LIBM.set_flush_denormals(False)
            result["identical"] = hostile == clean
            result["restored"] = result["dn_after"] == LIBM.DN_FLUSH
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

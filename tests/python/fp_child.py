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
    # On Windows FE_* already are the UCRT values, and <float.h> gives the
    # same bit to a status flag and to its mask: _SW_INEXACT/_EM_INEXACT
    # 0x01, UNDERFLOW 0x02, OVERFLOW 0x04, ZERODIVIDE 0x08, INVALID 0x10.
    # So an FE_* set is its _EM_* mask, with no translation.

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
        # _clearfp clears every status flag, not only ``_flags``: the
        # scenarios only clear all flags.
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
        assert not flags & ~self.ALL_FLAGS, f"not FE_* flags: {flags:#x}"
        return flags

    def feenableexcept(self, flags):
        self._control(0, self._em(flags))
        return 0

    def fedisableexcept(self, flags):
        em = self._em(flags)
        self._control(em, em)
        return 0

    def fegetexcept(self):
        # Enabled = mask bit clear.
        return ~self._control() & self.ALL_FLAGS

    def raw_control(self):
        """The raw control word (``_controlfp_s(&cur, 0, 0)``)."""
        return self._control()

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


# FPCR: RMode toward zero (bits 22-23), FZ (24), DN (25).
FPCR_HOSTILE = (0b11 << 22) | (1 << 24) | (1 << 25)
FE_OVERFLOW = 0x04 if ARM or WINDOWS else 0x08
ALL_FLAGS = 0x1F if WINDOWS else (0x9F if ARM else 0x3F)
GLIBC_ARM = ARM and sys.platform.startswith("linux")


def cpu_name():
    """The CPU as /proc/cpuinfo names it (model name, or the AArch64
    implementer and part), else platform.processor()."""
    try:
        with open("/proc/cpuinfo") as f:
            lines = f.read().splitlines()
    except OSError:
        return platform.processor() or platform.machine()
    keys = ("model name", "CPU implementer", "CPU part", "CPU variant", "CPU revision")
    found = {}
    for line in lines:
        k, _, v = line.partition(":")
        k = k.strip()
        if k in keys and k not in found:
            found[k] = v.strip()
    return "; ".join(f"{k}={found[k]}" for k in keys if k in found) or platform.machine()


def read_controls():
    """The raw control register, without status flags: FPCR on glibc
    AArch64, MXCSR controls and the x87 control word on glibc x86-64, the
    control word on Windows; None elsewhere."""
    if WINDOWS:
        return LIBM.raw_control()
    if not sys.platform.startswith("linux"):
        return None
    env = (ctypes.c_uint8 * 32)()
    assert LIBM.fegetenv(env) == 0
    if ARM:
        # glibc fenv_t on AArch64: { unsigned int __fpcr, __fpsr; }.
        return int.from_bytes(bytes(env[0:4]), "little")
    # glibc fenv_t on x86-64: x87 control word at 0, MXCSR at 28.
    cw = int.from_bytes(bytes(env[0:2]), "little")
    mxcsr = int.from_bytes(bytes(env[28:32]), "little") & ~0x3F
    return (cw << 32) | mxcsr


def write_fpcr(value):
    """Writes FPCR on glibc AArch64 through fesetenv, keeping FPSR."""
    assert GLIBC_ARM
    env = (ctypes.c_uint8 * 32)()
    assert LIBM.fegetenv(env) == 0
    env[0:4] = list(value.to_bytes(4, "little"))
    assert LIBM.fesetenv(env) == 0


def unmasked_skip_reason(enabled_set):
    """Why the unmasked scenario cannot run, or None. ``enabled_set`` is
    the trap enables read back right after the caller set them, before any
    call into the library. On AArch64 the FPCR trap enables are optional:
    a core without trapped floating-point exceptions ignores the write and
    they read back as zero (issue #225). Only that read-back decides; the
    value after the library call never does."""
    if enabled_set == 0:
        return "this CPU does not implement trapped floating-point exceptions (FPCR enables read back as zero)"
    return None


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
        # glibc and Windows: unmask divide-by-zero and invalid in the caller.
        # Always reported, also on a skip: the CPU and the read-back probe
        # (issue #225), so a CI log shows the mechanism.
        result["cpu"] = cpu_name()
        if not hasattr(LIBM, "feenableexcept"):
            result["skipped"] = "no feenableexcept"
        else:
            default = run_episode()
            original = read_controls()
            LIBM.feenableexcept(FE_DIVBYZERO | FE_INVALID)
            # Read the enables back before any call into the library: a CPU
            # may ignore the write (issue #225). Only this read-back decides
            # the skip.
            result["enabled_set"] = LIBM.fegetexcept()
            result["controls_with_enables"] = read_controls()
            result["enables_reason"] = unmasked_skip_reason(result["enabled_set"])
            if GLIBC_ARM:
                # The rest of FPCR hostile too: round toward zero, FZ, DN.
                write_fpcr(result["controls_with_enables"] | FPCR_HOSTILE)
            result["controls_before"] = read_controls()
            hostile = run_episode()
            result["controls_after"] = read_controls()
            result["enabled_after"] = LIBM.fegetexcept()
            # ADR-0041 / ADR-0037: a sticky flag the caller raised survives
            # native calls (only native calls in between: NumPy clears flags).
            env = t.make("tessera/CartPole-v1", num_envs=2, seed=1)
            native = env._native
            native.reset()
            acts = np.ones(2, dtype="<i4").tobytes()
            LIBM.feclearexcept(ALL_FLAGS)
            LIBM.feraiseexcept(FE_OVERFLOW)
            result["flag_set"] = bool(LIBM.fetestexcept(FE_OVERFLOW))
            native.step(acts)
            native.step(acts)
            result["flag_kept"] = bool(LIBM.fetestexcept(FE_OVERFLOW))
            LIBM.feclearexcept(ALL_FLAGS)
            if WINDOWS:
                # Independent of the shim's mapping: in the raw control word
                # exactly _EM_ZERODIVIDE and _EM_INVALID are clear.
                raw = LIBM.raw_control() & LIBM.ALL_FLAGS
                result["raw_em_exact"] = raw == LIBM.ALL_FLAGS & ~(0x08 | 0x10)
            LIBM.fedisableexcept(FE_DIVBYZERO | FE_INVALID)
            if GLIBC_ARM:
                write_fpcr(original)
            env.close()
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
    elif scenario == "search_callback":
        # ADR-0041 callouts and ADR-0043 section 3: the Python evaluator runs
        # under the caller's controls (here round toward zero), the search
        # arithmetic under the canonical ones, so the visits and the tree
        # equal those of a caller in the default mode.
        from tessera_rl.search import Search

        env = t.make("tessera/Connect4-v1", seed=21, auto_reset="disabled", outputs="owned")
        env.reset()
        for c in (3, 3):
            env.step(np.array([[c, c]]))
        record = env.snapshot(0)
        modes = []

        def evaluator(obs, legal, hashes):
            modes.append(LIBM.fegetround())
            # Responses exact in every rounding mode (powers of two summing
            # to 1, placed by the hash), so any difference in the result
            # would come from the engine's own arithmetic.
            n = len(hashes)
            priors = np.full((n, 7), 0.125)
            priors[np.arange(n), (hashes % np.uint64(7)).astype(np.int64)] = 0.25
            values = np.where(hashes % np.uint64(2) == 0, 0.5, -0.5)
            return priors, values

        def run():
            s = Search(simulations=128)
            s.set_root(record)
            s.run(evaluator)
            return s.root_visits().tolist(), s.tree_hash()

        clean = run()
        clean_modes = set(modes)
        modes.clear()
        before = t.fp_non_canonical_entries()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_round(FE_TOWARDZERO)
            try:
                hostile = run()
                result["round_after"] = LIBM.fegetround() == FE_TOWARDZERO
            finally:
                set_round(FE_TONEAREST)
        result["identical"] = hostile == clean
        result["clean_modes_default"] = clean_modes == {FE_TONEAREST}
        result["callback_saw_caller_mode"] = bool(modes) and all(m == FE_TOWARDZERO for m in modes)
        result["counted"] = t.fp_non_canonical_entries() - before

        # The evaluator ALTERS the controls (round upward) and leaves them
        # changed: the search arithmetic after it returns is canonical (the
        # tree is unchanged), the change is counted, and the caller's mode
        # is restored when run returns, and when it raises.
        def altering(obs, legal, hashes):
            LIBM.fesetround(FE_UPWARD)
            return evaluator(obs, legal, hashes)

        def failing(obs, legal, hashes):
            LIBM.fesetround(FE_UPWARD)
            raise RuntimeError("after altering the controls")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            set_round(FE_TOWARDZERO)
            try:
                before = t.fp_non_canonical_entries()
                s = Search(simulations=128)
                s.set_root(record)
                s.run(altering)
                altered = (s.root_visits().tolist(), s.tree_hash())
                result["altered_counted"] = t.fp_non_canonical_entries() - before
                result["restored_after_success"] = LIBM.fegetround() == FE_TOWARDZERO
                s2 = Search(simulations=128)
                s2.set_root(record)
                try:
                    s2.run(failing)
                    result["raised"] = False
                except t.EvaluatorError:
                    result["raised"] = True
                result["restored_after_exception"] = LIBM.fegetround() == FE_TOWARDZERO
                result["unchanged_after_exception"] = s2.calls == 0
            finally:
                set_round(FE_TONEAREST)
        result["altered_identical"] = altered == clean
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

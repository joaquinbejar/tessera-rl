# SPDX-License-Identifier: MIT OR Apache-2.0
"""Child process for the x87 control word tests (ADR-0044): each scenario
needs a fresh process, because the boundary's warning (the callout these
tests drive) is shown once per process, and the precondition scenarios
end in a hardware fault.

Usage: python x87_child.py SCENARIO. Prints one JSON object, unless the
scenario is expected to die with SIGFPE.

The x86-64 registers are read and written raw, through small native stubs
in an executable page (written while read-write, then made read-execute):
``fegetround`` reads only the x87 control word on glibc, so it cannot tell
the two registers apart. System V x86-64 only (Linux and macOS).
"""

import ctypes
import json
import mmap
import platform
import sys
import warnings

import numpy as np

import tessera_rl as t

# Masks: the x87 control word's defined bits (exception masks 0-5,
# precision 8-9, rounding 10-11, infinity 12) and MXCSR's control bits
# (everything but the six status flags).
X87_DEFINED = 0x1F3F
MXCSR_CONTROL = 0xFFC0
X87_DEFAULT = 0x037F
MXCSR_DEFAULT = 0x1F80
# A caller x87 word: double precision, round up, all masked.
X87_CALLER = 0x0A7F
# What host code writes: extended precision, toward zero, all masked.
X87_HOST = 0x0F7F
# MXCSR round toward zero, all masked.
MXCSR_TOWARDZERO = 0x7F80
X87_ZE = 0x04  # divide-by-zero: mask bit in the CW, flag bit in the SW

_STUBS = {
    # stmxcsr [rsp-4]; mov eax, [rsp-4]; ret
    "mxcsr": ("0fae5c24fc8b4424fcc3", ctypes.c_uint32, []),
    # mov [rsp-4], edi; ldmxcsr [rsp-4]; ret
    "set_mxcsr": ("897c24fc0fae5424fcc3", None, [ctypes.c_uint32]),
    # fnstcw [rsp-2]; movzx eax, word [rsp-2]; ret
    "x87": ("d97c24fe0fb74424fec3", ctypes.c_uint32, []),
    # mov [rsp-2], di; fldcw [rsp-2]; ret
    "set_x87": ("66897c24fed96c24fec3", None, [ctypes.c_uint32]),
    # fnstsw ax; movzx eax, ax; ret
    "x87_status": ("dfe00fb7c0c3", ctypes.c_uint32, []),
    # Sets the x87 divide-by-zero status flag without any arithmetic:
    # sub rsp, 28; fnstenv [rsp]; or word [rsp+4], 4; fldenv [rsp];
    # add rsp, 28; ret. fldenv reloads the stored control word, so if it
    # unmasks divide-by-zero the exception is now pending.
    "raise_x87_ze": ("4883ec1cd93424"
                     "66834c240404d92424"
                     "4883c41cc3", None, []),
}


class Raw:
    """The native stubs. Raises OSError where an executable mapping is
    refused (SELinux execmem, a hardened runtime)."""

    def __init__(self):
        code = b"".join(bytes.fromhex(h).ljust(32, b"\xcc") for h, _, _ in _STUBS.values())
        self._map = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self._map.write(code)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        if libc.mprotect(base, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
            raise OSError(ctypes.get_errno(), "mprotect to read-execute refused")
        for i, (name, (_, restype, argtypes)) in enumerate(_STUBS.items()):
            setattr(self, name, ctypes.CFUNCTYPE(restype, *argtypes)(base + 32 * i))

    def words(self):
        """[x87 control word, MXCSR control bits], masked to defined bits."""
        return [self.x87() & X87_DEFINED, self.mxcsr() & MXCSR_CONTROL]


def supported():
    if sys.platform == "win32" or platform.machine().lower() not in ("x86_64", "amd64"):
        return "the raw x87 probes run on System V x86-64 only"
    return None


def env_episode():
    """A short episode through several entries; its output bytes."""
    env = t.make("tessera/CartPole-v1", num_envs=2, seed=3, outputs="owned")
    out = [env.reset().observations["obs"].tobytes()]
    for k in range(20):
        out.append(env.step(np.full((2, 1), k % 2)).observations["obs"].tobytes())
    return out


def under(raw, x87, mxcsr, f):
    """Runs ``f`` with the given raw controls; returns its result and the
    raw words right after it, then restores the defaults."""
    raw.set_x87(x87)
    raw.set_mxcsr(mxcsr)
    try:
        r = f()
        return r, raw.words()
    finally:
        raw.set_x87(X87_DEFAULT)
        raw.set_mxcsr(MXCSR_DEFAULT)


def warn_hook(action):
    """Installs ``action`` as the warnings display hook: Python code the
    boundary runs through its warning callout (ADR-0041)."""

    def hook(*_args, **_kwargs):
        action()

    warnings.showwarning = hook
    warnings.simplefilter("always")


def main(scenario):
    result = {"scenario": scenario}
    why = supported()
    if why is None:
        try:
            raw = Raw()
        except OSError as e:
            why = f"no executable mapping: {e}"
    if why is not None:
        result["skipped"] = why
        print(json.dumps(result))
        return

    if scenario == "entry":
        # An x87-only caller change (MXCSR canonical): preserved, not
        # canonicalized, not counted, identical results.
        clean = env_episode()
        before = t.fp_non_canonical_entries()
        hostile, after = under(raw, X87_CALLER, MXCSR_DEFAULT, env_episode)
        result["identical"] = hostile == clean
        result["after"] = after
        result["want"] = [X87_CALLER & X87_DEFINED, MXCSR_DEFAULT & MXCSR_CONTROL]
        result["counted"] = t.fp_non_canonical_entries() - before
        # MXCSR-only: the x87 word is untouched, MXCSR restored.
        _, after = under(raw, X87_DEFAULT, MXCSR_TOWARDZERO, env_episode)
        result["mxcsr_only_after"] = after
        result["mxcsr_only_want"] = [X87_DEFAULT & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
    elif scenario == "fesetround":
        # The hostile-rounding case of fp_child.py, read raw: glibc
        # fesetround writes both registers; both are restored after every
        # binding, and the results do not change.
        import ctypes.util

        libm = ctypes.CDLL(ctypes.util.find_library("m"))
        clean = env_episode()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert libm.fesetround(0x0C00) == 0  # FE_TOWARDZERO
            try:
                want = raw.words()
                hostile = env_episode()
                result["after"] = raw.words()
            finally:
                libm.fesetround(0)
        result["want"] = want
        result["want_is_toward_zero"] = (want[0] >> 10) & 3 == 3 and (want[1] >> 13) & 3 == 3
        result["identical"] = hostile == clean
    elif scenario == "reject":
        # Strict policy with an x87-only change: never rejected (ADR-0044
        # keeps the x87 word out of the policy). Cleanup entries restore too.
        env = t.make("tessera/CartPole-v1", num_envs=2, seed=1)
        env.reset()
        t.set_fp_policy("reject")
        try:
            def step():
                try:
                    env.step(np.ones((2, 1)))
                    return "ran"
                except t.FloatingPointEnvironmentError:
                    return "rejected"

            result["x87_only"], result["x87_only_after"] = under(raw, X87_CALLER, MXCSR_DEFAULT, step)
            # Both registers hostile: rejected because of MXCSR, and the
            # x87 word is still restored on the error path.
            result["both"], result["both_after"] = under(raw, X87_CALLER, MXCSR_TOWARDZERO, step)

            def cleanup():
                nonlocal env
                del env
                return "freed"

            _, after = under(raw, X87_CALLER, MXCSR_DEFAULT, cleanup)
            result["cleanup_after"] = after
        finally:
            t.set_fp_policy("canonicalize")
        result["want"] = [X87_CALLER & X87_DEFINED, MXCSR_DEFAULT & MXCSR_CONTROL]
        result["want_both"] = [X87_CALLER & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
    elif scenario in ("callout", "callout_raises", "callout_interrupt", "callout_nested"):
        # The warning callout runs a Python hook. The hook changes both
        # registers (MXCSR toward zero is the caller's; the x87 word goes to
        # X87_HOST) and, per scenario, raises or enters Tessera again.
        seen = {}

        def action():
            seen["hook_entry"] = raw.words()
            raw.set_x87(X87_HOST)
            raw.set_mxcsr(MXCSR_DEFAULT)
            if scenario == "callout_nested":
                t.make("tessera/CartPole-v1", seed=2, outputs="owned").reset()
                seen["after_nested"] = raw.words()
            if scenario == "callout_raises":
                raise ValueError("a failing warning hook")
            if scenario == "callout_interrupt":
                raise KeyboardInterrupt

        warn_hook(action)

        def call():
            try:
                t.make("tessera/CartPole-v1", seed=1, outputs="owned").reset()
                return "ran"
            except (ValueError, KeyboardInterrupt) as e:
                return type(e).__name__

        outcome, after = under(raw, X87_CALLER, MXCSR_TOWARDZERO, call)
        result["outcome"] = outcome
        result["after"] = after
        result["hook_entry"] = seen.get("hook_entry")
        result["after_nested"] = seen.get("after_nested")
        result["want"] = [X87_CALLER & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
        result["want_nested"] = [X87_HOST & X87_DEFINED, MXCSR_DEFAULT & MXCSR_CONTROL]
    elif scenario in ("pending_supported", "pending_masked_newly_unmasked", "pending_current_unmasked"):
        # ADR-0044 precondition. The hook raises the x87 divide-by-zero flag:
        # - supported: under a masked word, and the caller masks it too;
        # - newly unmasked: masked when raised, the caller's word unmasks it;
        # - current unmasked (the caller masks everything): the hook's own
        #   word unmasks it, so it is pending when the callback returns.
        # The violating scenarios are expected to die with SIGFPE.
        caller = X87_DEFAULT
        hook_word = X87_DEFAULT
        if scenario == "pending_masked_newly_unmasked":
            caller = X87_DEFAULT & ~X87_ZE
        if scenario == "pending_current_unmasked":
            hook_word = X87_DEFAULT & ~X87_ZE

        def action():
            raw.set_x87(hook_word)
            raw.raise_x87_ze()

        warn_hook(action)

        def call():
            # Several entries and callouts: repeated restores.
            for _ in range(3):
                t.make("tessera/CartPole-v1", seed=1, outputs="owned").reset()
            status = raw.x87_status()
            return status

        status, after = under(raw, caller, MXCSR_TOWARDZERO, call)
        result["status_ze"] = bool(status & X87_ZE)
        result["after"] = after
        result["want"] = [caller & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])

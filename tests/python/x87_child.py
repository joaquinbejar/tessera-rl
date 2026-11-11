# SPDX-License-Identifier: MIT OR Apache-2.0
"""Child process for the x87 control word tests (ADR-0044): each scenario
needs a fresh process, because the boundary's warning (the callout these
tests drive) is shown once per process, and the precondition scenarios
end in a hardware fault.

Usage: python x87_child.py SCENARIO. Prints one JSON object, unless the
scenario is expected to die with a hardware fault (SIGFPE on System V, an
NTSTATUS float code on Win64).

The x86-64 registers are read and written raw, through small native stubs
in an executable page (written while read-write, then made read-execute):
``fegetround`` reads only the x87 control word on glibc, and the UCRT's
``_controlfp_s`` acts on MXCSR on x64, so neither can tell the two
registers apart. System V x86-64 (Linux and macOS) and Win64 (#163); the
Win64 stubs use the caller-allocated home space at ``[rsp+8]`` (Win64 has
no red zone) and take their argument in ``ecx``. Every stub is a leaf that
never moves ``rsp``. Encodings checked with ``llvm-objdump``.
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
# The System V process default (64-bit precision). On Win64 the CRT starts
# with 0x027F (53-bit precision); main() reads the start value there
# instead of assuming it.
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


_STUBS_WIN64 = {
    # stmxcsr [rsp+8]; mov eax, [rsp+8]; ret
    "mxcsr": ("0fae5c24088b442408c3", ctypes.c_uint32, []),
    # mov [rsp+8], ecx; ldmxcsr [rsp+8]; ret
    "set_mxcsr": ("894c24080fae542408c3", None, [ctypes.c_uint32]),
    # fnstcw [rsp+8]; movzx eax, word [rsp+8]; ret
    "x87": ("d97c24080fb7442408c3", ctypes.c_uint32, []),
    # mov [rsp+8], cx; fldcw [rsp+8]; ret
    "set_x87": ("66894c2408d96c2408c3", None, [ctypes.c_uint32]),
    # fnstsw ax; movzx eax, ax; ret
    "x87_status": ("dfe00fb7c0c3", ctypes.c_uint32, []),
    # fnstenv [rsp+8]; or word [rsp+12], 4; fldenv [rsp+8]; ret. The
    # 28-byte environment fits the 32 bytes of home space. fnstenv masks
    # every x87 exception after storing and fldenv reloads the stored
    # control word, so the word is unchanged across the stub; only the
    # divide-by-zero status flag is set, pending if the word unmasks it.
    "raise_x87_ze": ("d9742408"
                     "66834c240c04d9642408"
                     "c3", None, []),
}

WIN64 = sys.platform == "win32"


def _executable_page(code):
    """``code`` in a read-execute page; its base address and an owner that
    keeps the page alive."""
    if not WIN64:
        page = mmap.mmap(-1, mmap.PAGESIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        page.write(code)
        base = ctypes.addressof(ctypes.c_char.from_buffer(page))
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        if libc.mprotect(base, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_EXEC) != 0:
            raise OSError(ctypes.get_errno(), "mprotect to read-execute refused")
        return base, page
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_uint32)]
    k32.VirtualProtect.restype = ctypes.c_int
    k32.FlushInstructionCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    k32.FlushInstructionCache.restype = ctypes.c_int
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    mem_commit_reserve, page_rw, page_rx = 0x3000, 0x04, 0x20
    base = k32.VirtualAlloc(None, 4096, mem_commit_reserve, page_rw)
    if not base:
        raise ctypes.WinError(ctypes.get_last_error())
    ctypes.memmove(base, code, len(code))
    old = ctypes.c_uint32()
    # A refusal here (arbitrary code guard) becomes a named skip.
    if not k32.VirtualProtect(base, 4096, page_rx, ctypes.byref(old)):
        raise ctypes.WinError(ctypes.get_last_error())
    k32.FlushInstructionCache(k32.GetCurrentProcess(), base, 4096)
    # The page lives as long as the process: the child is short-lived.
    return base, None


class Raw:
    """The native stubs. Raises OSError where an executable mapping is
    refused (SELinux execmem, a hardened runtime, Windows ACG). ``last``
    names the stub called most recently, so a hardware exception that
    ctypes converts into OSError on Windows can be attributed."""

    def __init__(self):
        stubs = _STUBS_WIN64 if WIN64 else _STUBS
        code = b"".join(bytes.fromhex(h).ljust(32, b"\xcc") for h, _, _ in stubs.values())
        base, self._page = _executable_page(code)
        self.last = None
        for i, (name, (_, restype, argtypes)) in enumerate(stubs.items()):
            setattr(self, name, self._named(name, ctypes.CFUNCTYPE(restype, *argtypes)(base + 32 * i)))

    def _named(self, name, fn):
        def call(*args):
            self.last = name
            return fn(*args)

        return call

    def words(self):
        """[x87 control word, MXCSR control bits], masked to defined bits."""
        return [self.x87() & X87_DEFINED, self.mxcsr() & MXCSR_CONTROL]


def supported():
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return "the raw x87 probes run on x86-64 only"
    return None


def no_error_dialogs():
    """Win64: an unhandled exception ends the process without a Windows
    Error Reporting dialog, so a faulting child never hangs the session."""
    if WIN64:
        sem_failcriticalerrors, sem_nogpfaulterrorbox = 0x0001, 0x0002
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)


def ucrt_view():
    """Win64: what the UCRT reports (``_controlfp_s``, ``_statusfp``).
    Recorded beside the raw words, never asserted: ADR-0044 does not take
    a CRT abstraction as evidence of the physical x87 word."""
    if not WIN64:
        return None
    ucrt = ctypes.CDLL("ucrtbase")
    cw = ctypes.c_uint()
    err = ucrt._controlfp_s(ctypes.byref(cw), 0, 0)
    ucrt._statusfp.restype = ctypes.c_uint
    return {"controlfp_s": cw.value, "controlfp_s_err": err, "statusfp": ucrt._statusfp()}


def self_test(raw):
    """The stubs read back what they write, and the MXCSR stub leaves the
    x87 word alone. A failure fails the scenario, never skips it."""
    start = raw.x87() & X87_DEFINED
    seen = []
    for v in (X87_CALLER, X87_HOST):
        raw.set_x87(v)
        seen.append(raw.x87() & X87_DEFINED)
    raw.set_x87(X87_CALLER)
    raw.set_mxcsr(MXCSR_TOWARDZERO)
    seen.append(raw.x87() & X87_DEFINED)
    raw.set_mxcsr(MXCSR_DEFAULT)
    raw.set_x87(start)
    want = [X87_CALLER, X87_HOST, X87_CALLER]
    return {"ok": seen == want, "seen": seen, "want": want}


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
    global X87_DEFAULT
    result = {"scenario": scenario}
    no_error_dialogs()
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

    result["initial"] = raw.words()
    if WIN64:
        X87_DEFAULT = result["initial"][0]
        result["ucrt_initial"] = ucrt_view()
    result["self_test"] = self_test(raw)
    if not result["self_test"]["ok"]:
        print(json.dumps(result))
        raise SystemExit(3)

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

        clean = env_episode()
        if WIN64:
            # The UCRT's fesetround on x64 is not assumed to write the x87
            # word: whether it did is recorded, and the x87 word is then
            # set toward zero raw, so both registers are hostile as on
            # System V.
            libm = ctypes.CDLL("ucrtbase")
            toward_zero = 0x0300  # the UCRT's FE_TOWARDZERO
        else:
            libm = ctypes.CDLL(ctypes.util.find_library("m"))
            toward_zero = 0x0C00  # glibc's and macOS's FE_TOWARDZERO
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert libm.fesetround(toward_zero) == 0
            try:
                if WIN64:
                    result["crt_wrote_x87"] = (raw.x87() >> 10) & 3 == 3
                    raw.set_x87(raw.x87() | 0x0C00)
                want = raw.words()
                hostile = env_episode()
                result["after"] = raw.words()
            finally:
                libm.fesetround(0)
                if WIN64:
                    raw.set_x87(X87_DEFAULT)
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
            # The restore is shown, not only the end state: the hook's own
            # word is in force inside the callout.
            seen["hook_x87"] = raw.x87() & X87_DEFINED
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
        result["hook_x87"] = seen.get("hook_x87")
        result["want_hook_x87"] = X87_HOST & X87_DEFINED
        result["after_nested"] = seen.get("after_nested")
        result["want"] = [X87_CALLER & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
        result["want_nested"] = [X87_HOST & X87_DEFINED, MXCSR_DEFAULT & MXCSR_CONTROL]
    elif scenario in ("pending_supported", "pending_masked_newly_unmasked", "pending_current_unmasked"):
        # ADR-0044 precondition. The hook raises the x87 divide-by-zero flag:
        # - supported: under a masked word, and the caller masks it too;
        # - newly unmasked: masked when raised, the caller's word unmasks it;
        # - current unmasked (the caller masks everything): the hook's own
        #   word unmasks it, so it is pending when the callback returns.
        # The violating scenarios are expected to die with a hardware fault
        # in engine code: SIGFPE on System V, an NTSTATUS float code on
        # Win64. On Win64 ctypes turns a hardware exception inside a stub
        # into OSError; that is reported as its own outcome, with the stub.
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

        try:
            status, after = under(raw, caller, MXCSR_TOWARDZERO, call)
        except OSError as e:
            print(json.dumps({"scenario": scenario, "outcome": "ctypes_oserror",
                              "stub": raw.last, "winerror": str(e)}))
            raise SystemExit(4) from None
        result["status_ze"] = bool(status & X87_ZE)
        result["after"] = after
        result["want"] = [caller & X87_DEFINED, MXCSR_TOWARDZERO & MXCSR_CONTROL]
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    if WIN64:
        result["ucrt_final"] = ucrt_view()
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1])

# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0044: every System V x86-64 thunk of the built extension saves and
restores the x87 control word at the template's positions."""

import pathlib
import platform
import shutil
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).parents[2] / "scripts" / "check_x87_thunks.py"


def test_check_x87_thunks_on_the_built_extension():
    if not sys.platform.startswith("linux") or platform.machine() != "x86_64":
        pytest.skip("the disassembly check runs on Linux x86-64")
    objdump = shutil.which("objdump")
    if objdump is None:
        pytest.skip("objdump is not installed")
    import tessera_rl._native as native

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--so", native.__file__, "--objdump", objdump], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "entries" in proc.stdout


def _checker():
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_x87_thunks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENTRY = [
    "push rbp",
    "mov rbp,rsp",
    "sub rsp,0x50",
    "fnstcw WORD PTR [rbp-0x42]",
    "stmxcsr DWORD PTR [rbp-0x34]",
    "call 1000 <enter>",
    "call 2000 <target>",
    "call 3000 <leave>",
    "ldmxcsr DWORD PTR [rbp-0x38]",
    "fldcw WORD PTR [rbp-0x42]",
    "mov rax,QWORD PTR [rbp-0x8]",
    "leave",
    "ret",
]
CALLOUT = [
    "push rbp",
    "fnstcw WORD PTR [rbp-0x42]",
    "call 1000 <caller>",
    "call 2000 <host>",
    "fldcw WORD PTR [rbp-0x42]",
    "mov QWORD PTR [rbp-0x8],rax",
    "call 3000 <returned>",
    "mov rax,QWORD PTR [rbp-0x8]",
    "leave",
    "ret",
]


def test_check_x87_thunks_accepts_the_templates():
    m = _checker()
    assert m.check({"thunk_step": ENTRY, "c_decref": CALLOUT}) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e, c: (e[:3] + e[4:], c),  # entry without fnstcw
        lambda e, c: (e[:9] + e[10:], c),  # entry without fldcw
        lambda e, c: (e[:9] + ["mov rax,QWORD PTR [rbp-0x8]", e[9]] + e[11:], c),  # fldcw not right before leave
        lambda e, c: (e[:5] + [e[3]] + e[5:], c),  # a second fnstcw
        lambda e, c: (e[:3] + [e[5], e[3]] + e[4:5] + e[6:], c),  # fnstcw after the first call
        lambda e, c: (e, c[:4] + c[5:]),  # callout without fldcw
        lambda e, c: (e, c[:4] + [c[5], c[4]] + c[6:]),  # fldcw not right after the host call
        lambda e, c: (e, c[:1] + c[2:]),  # callout without fnstcw
    ],
)
def test_check_x87_thunks_rejects_each_broken_template(mutate):
    m = _checker()
    e, c = mutate(list(ENTRY), list(CALLOUT))
    assert m.check({"thunk_step": e, "c_decref": c}), "the checker reports the defect"


def test_check_x87_thunks_rejects_a_missing_kind():
    m = _checker()
    assert m.check({"thunk_step": ENTRY}), "no callout found"

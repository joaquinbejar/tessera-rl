# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks the x87 control word handling of every System V x86-64 thunk in a
built extension (ADR-0044), from its disassembly.

Each boundary entry thunk saves the caller's word with `fnstcw` before its
first call, and reloads it with `fldcw` right before its single epilogue
(`mov rax, [rbp-8]; leave; ret`). Each callout thunk saves it before its
first call, and reloads it right after the host call (its second call),
before saving the result.

Usage: python scripts/check_x87_thunks.py --so path/to/_native.so
[--objdump objdump]. Exits 1 and prints each problem when a thunk differs.
"""

import argparse
import re
import subprocess
import sys

# A function header in `objdump -d` output: "0000000000012340 <symbol>:".
HEADER = re.compile(r"^[0-9a-f]+ <(?P<sym>[^>]+)>:$")
# Rust v0 mangling of `thunks::<name>` ends in the identifier with its length.
THUNK = re.compile(r"6thunks(?P<len>\d+)(?P<name>[a-z0-9_]+)$")

SLOT = r"\[rbp-0x42\]"


def functions(dump: str) -> dict[str, list[str]]:
    """Every thunk's instructions (mnemonic and operands), keyed by name."""
    out: dict[str, list[str]] = {}
    current = None
    for line in dump.splitlines():
        m = HEADER.match(line.strip())
        if m:
            sym = m.group("sym")
            t = THUNK.search(sym)
            if sym == "PyInit__native":
                current = "PyInit__native"
            elif t and int(t.group("len")) == len(t.group("name")):
                current = t.group("name")
            else:
                current = None
            if current is not None:
                out[current] = []
            continue
        if current is None:
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[-1].strip():
            out[current].append(" ".join(parts[-1].split()))
    return out


def is_call(insn: str) -> bool:
    return insn.startswith("call ")


def check_entry(name: str, insns: list[str]) -> list[str]:
    problems = []
    saves = [i for i, x in enumerate(insns) if re.fullmatch(rf"fnstcw WORD PTR {SLOT}", x)]
    loads = [i for i, x in enumerate(insns) if re.fullmatch(rf"fldcw WORD PTR {SLOT}", x)]
    calls = [i for i, x in enumerate(insns) if is_call(x)]
    if len(saves) != 1 or len(loads) != 1 or not calls:
        return [f"{name}: {len(saves)} fnstcw, {len(loads)} fldcw, {len(calls)} calls"]
    if saves[0] > calls[0]:
        problems.append(f"{name}: fnstcw after the first call")
    rets = [i for i, x in enumerate(insns) if x == "ret"]
    if len(rets) != 1:
        problems.append(f"{name}: {len(rets)} ret, the template has one epilogue")
    elif insns[loads[0] : rets[0] + 1] != ["fldcw WORD PTR [rbp-0x42]", "mov rax,QWORD PTR [rbp-0x8]", "leave", "ret"]:
        problems.append(f"{name}: fldcw is not right before the epilogue: {insns[loads[0]:rets[0] + 1]}")
    return problems


def check_callout(name: str, insns: list[str]) -> list[str]:
    saves = [i for i, x in enumerate(insns) if re.fullmatch(rf"fnstcw WORD PTR {SLOT}", x)]
    loads = [i for i, x in enumerate(insns) if re.fullmatch(rf"fldcw WORD PTR {SLOT}", x)]
    calls = [i for i, x in enumerate(insns) if is_call(x)]
    if len(saves) != 1 or len(loads) != 1 or len(calls) < 2:
        return [f"{name}: {len(saves)} fnstcw, {len(loads)} fldcw, {len(calls)} calls"]
    problems = []
    if saves[0] > calls[0]:
        problems.append(f"{name}: fnstcw after the first call")
    host = calls[1]
    if loads[0] != host + 1 or insns[host + 2] != "mov QWORD PTR [rbp-0x8],rax":
        problems.append(f"{name}: fldcw is not right after the host call: {insns[host:host + 3]}")
    return problems


def check(funcs: dict[str, list[str]]) -> list[str]:
    entries = {n: i for n, i in funcs.items() if n == "PyInit__native" or n.startswith("thunk_")}
    callouts = {n: i for n, i in funcs.items() if n.startswith("c_")}
    if not entries or not callouts:
        return [f"found {len(entries)} entry and {len(callouts)} callout thunks"]
    problems = []
    for name, insns in sorted(entries.items()):
        problems += check_entry(name, insns)
    for name, insns in sorted(callouts.items()):
        problems += check_callout(name, insns)
    return problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--so", required=True)
    p.add_argument("--objdump", default="objdump")
    args = p.parse_args()
    dump = subprocess.run(
        [args.objdump, "-d", "-M", "intel", "--no-show-raw-insn", args.so], capture_output=True, text=True, check=True
    ).stdout
    funcs = functions(dump)
    problems = check(funcs)
    for line in problems:
        print(line)
    if problems:
        return 1
    n_entries = sum(1 for n in funcs if n == "PyInit__native" or n.startswith("thunk_"))
    n_callouts = sum(1 for n in funcs if n.startswith("c_"))
    print(f"x87 thunks: {n_entries} entries and {n_callouts} callouts, each saves and restores the control word")
    return 0


if __name__ == "__main__":
    sys.exit(main())

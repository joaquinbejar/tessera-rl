# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Win64 unwind-coverage checker (scripts/check_win64_unwind.py) on
synthetic linker-map and llvm-readobj text."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from check_win64_unwind import check, parse_map, parse_unwind  # noqa: E402

MAP = """
 _native

 Preferred load address is 0000000180000000

  Address         Publics by Value              Rva+Base               Lib:Object

 0001:00001000       _RNvNtNtCs1_7__native3ffi6thunks13thunk_version 0000000180002000 f   _native.o
 0001:00001100       _RNvNtNtCs1_7__native3ffi6thunks10c_dict_new 0000000180002100 f   _native.o
 0001:00001200       PyInit__native             0000000180002200 f   _native.o
 0001:00001300       _RNvNtNtCs1_7__native3ffi3api4step 0000000180002300 f   _native.o
"""


def record(addr, codes=("0x04: ALLOC_SMALL size=88",), frame="-"):
    lines = "\n".join(f"        {c}" for c in codes)
    return f"""  RuntimeFunction {{
    StartAddress: (0x{addr:x})
    EndAddress: (0x{addr + 0x94:x})
    UnwindInfo {{
      Version: 1
      PrologSize: 4
      FrameRegister: {frame}
      UnwindCodes [
{lines}
      ]
    }}
  }}
"""


def test_check_win64_unwind_full_coverage_passes():
    thunks = parse_map(MAP)
    assert sorted(thunks.values()) == [0x2000, 0x2100, 0x2200]
    dump = "".join(record(a) for a in (0x180002000, 0x180002100, 0x180002200))
    assert check(thunks, parse_unwind(dump, 0x180000000)) == []


def test_check_win64_unwind_rva_addresses_pass():
    dump = "".join(record(a) for a in (0x2000, 0x2100, 0x2200))
    assert check(parse_map(MAP), parse_unwind(dump, 0x180000000)) == []


def test_check_win64_unwind_missing_record_fails():
    dump = "".join(record(a) for a in (0x2000, 0x2200))
    problems = check(parse_map(MAP), parse_unwind(dump))
    assert len(problems) == 1 and "c_dict_new" in problems[0]


def test_check_win64_unwind_other_record_fails():
    dump = record(0x2000) + record(0x2100, codes=("0x05: PUSH_NONVOL reg=RBP", "0x01: ALLOC_SMALL size=80")) + record(0x2200, frame="RBP")
    assert len(check(parse_map(MAP), parse_unwind(dump))) == 2


def test_check_win64_unwind_no_thunks_fails():
    assert check(parse_map("Preferred load address is 0000000180000000\n"), {})

"""The disassembly checker reports what it must, on synthetic assembly."""

import importlib.util
import pathlib

_TOOL = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_simd_disasm.py"
_spec = importlib.util.spec_from_file_location("check_simd_disasm", _TOOL)
disasm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disasm)

GOOD = """_ZN5lanes8run_neon17h1E:
\tfmul\tv0.2d, v1.2d, v2.2d
_ZN5lanes10run_scalar17h2E:
\tfmul\td0, d1, d2
"""


def test_good_assembly_passes():
    assert disasm.problems("aarch64", GOOD) == []


def test_fused_operation_is_reported():
    assert any("fused" in p for p in disasm.problems("aarch64", GOOD + "\tfmla\tv0.2d, v1.2d, v2.2d\n"))
    assert any("fused" in p for p in disasm.problems("aarch64", GOOD + "\tfmadd\td0, d1, d2, d3\n"))


def test_x86_fused_operation_is_reported():
    text = "run_avx2:\n\tvmulpd %ymm0, %ymm1, %ymm2\nrun_sse42:\n\tmulpd %xmm0, %xmm1\nrun_scalar:\n\tmulsd %xmm0, %xmm1\n\tvfmadd231pd %ymm0, %ymm1, %ymm2\n"
    assert any("fused" in p for p in disasm.problems("x86_64", text))


def test_missing_entry_point_is_reported():
    assert any("no entry point" in p for p in disasm.problems("aarch64", "_ZN3foo3bar:\n\tret\n"))


def test_call_to_a_lane_operation_is_reported():
    bad = GOOD.replace("\tfmul\td0", "\tbl\t_ZN15tessera_rl_simd3add\n\tfmul\td0")
    assert any("not inlined" in p for p in disasm.problems("aarch64", bad))


INT_GOOD_X86 = """_ZN15tessera_rl_simd9int_lanes14run_avx2_lanes17h1E:
\tvpmuldq\t%ymm1, %ymm0, %ymm2
_ZN15tessera_rl_simd9int_lanes15run_sse42_lanes17h2E:
\tpmuldq\t%xmm1, %xmm0
"""
INT_GOOD_NEON = """_ZN15tessera_rl_simd9int_lanes14run_neon_lanes17h3E:
\tsmull\tv0.2d, v1.2s, v2.2s
\tsmull2\tv3.2d, v1.4s, v2.4s
"""


def test_int_lanes_good_assembly_passes():
    assert disasm.int_problems("x86_64", INT_GOOD_X86) == []
    assert disasm.int_problems("aarch64", INT_GOOD_NEON) == []


def test_int_lanes_missing_widening_multiply_is_reported():
    no_mul = INT_GOOD_X86.replace("\tvpmuldq\t%ymm1, %ymm0, %ymm2", "\tvpmulld\t%ymm1, %ymm0, %ymm2")
    assert any("widening multiply missing" in p for p in disasm.int_problems("x86_64", no_mul))
    no_smull = INT_GOOD_NEON.replace("smull", "mul")
    assert any("widening multiply missing" in p for p in disasm.int_problems("aarch64", no_smull))


def test_int_lanes_missing_entry_point_is_reported():
    assert any("no probe entry point" in p for p in disasm.int_problems("aarch64", "_ZN3foo3bar:\n\tret\n"))


def test_int_lanes_call_to_a_lane_operation_is_reported():
    bad = INT_GOOD_NEON.replace("\tsmull2", "\tbl\t_ZN10tessera_rl_num4math8lane_int3mul\n\tsmull2")
    assert any("not inlined" in p for p in disasm.int_problems("aarch64", bad))


def test_int_lanes_apple_neon_syntax_passes():
    apple = "__RINvNtCs4_15tessera_rl_simd9int_lanes14run_neon_lanesNtCs7_5probeE:\n\tsmull.2d\tv4, v2, v0\n\tsmull2.2d\tv0, v2, v0\n"
    assert disasm.int_problems("aarch64", apple) == []


def test_int_lanes_scalar_smull_is_not_the_vector_form():
    scalar = "__RINvNtCs4_15tessera_rl_simd9int_lanes14run_neon_lanesNtCs7_5probeE:\n\tsmull\tx8, w12, w8\n"
    assert any("widening multiply missing" in p for p in disasm.int_problems("aarch64", scalar))

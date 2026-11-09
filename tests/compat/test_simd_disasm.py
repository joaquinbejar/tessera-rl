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

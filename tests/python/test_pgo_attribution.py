# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo/attribution.py (#171, P1a) on a hand-written Callgrind sample: the
native share both ways, report only."""

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools" / "pgo"))
_spec = importlib.util.spec_from_file_location("attribution", ROOT / "tools" / "pgo" / "attribution.py")
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

EVENTS = "Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw Bc Bcm Bi Bim"
Z = " ".join(["0"] * 12)


def sample():
    """_native: 100 own Ir plus its own internal `memset` (70, as
    compiler-builtins would emit; a call with no `cob=` stays in the caller's
    object, as in real Callgrind files, so it is not a libc primitive). It
    calls libc memset (inclusive 900), memcpy (60) and malloc (40, not a
    primitive). libc's own exclusive cost is 1000; python calls memset for
    500 (not from _native)."""
    return f"""# callgrind format
version: 1
positions: line
events: {EVENTS}
ob=(1) /venv/lib/tessera_rl/_native.abi3.so
fl=(1) pool.rs
fn=(1) tessera_rl_engine::pool::clear_chunk
10 80 {Z}
cob=(2) /usr/lib/x86_64-linux-gnu/libc.so.6
cfn=(2) __memset_avx2_unaligned_erms
calls=3 0
11 900 {Z}
cob=(2)
cfn=(3) __memcpy_avx_unaligned_erms
calls=1 0
12 60 {Z}
cob=(2)
cfn=(4) malloc
calls=1 0
13 40 {Z}
cfn=(7) memset
calls=1 0
14 70 {Z}
fn=(7)
1 70 {Z}
fn=(5) tessera_rl_engine::pool::step_chunk
cfn=(1)
calls=1 0
20 1000 {Z}
21 20 {Z}
ob=(2)
fl=(2) memset.S
fn=(2)
1 900 {Z}
fn=(3)
1 60 {Z}
fn=(4)
1 40 {Z}
ob=(3) /usr/bin/python3.11
fl=(3) ceval.c
fn=(6) _PyEval_EvalFrameDefault
5 300 {Z}
cob=(2)
cfn=(2)
calls=1 0
6 500 {Z}
"""


def test_pgo_attribution_reports_both_native_shares():
    r = at.attribution(sample())
    # Exclusive per object: _native 80 + 70 + 20; libc 900 + 60 + 40; python 300.
    assert r["total_ir"] == 170 + 1000 + 300
    assert r["native_exclusive_ir"] == 170
    # Only _native's calls into libc memory primitives: memset 900, memcpy 60.
    assert r["primitives_from_native_ir"] == 960
    assert r["primitives_by_function"] == {"__memset_avx2_unaligned_erms": 900, "__memcpy_avx_unaligned_erms": 60}
    assert r["native_exclusive_share"] == pytest.approx(170 / 1470)
    assert r["native_with_primitives_share"] == pytest.approx(1130 / 1470)


def test_pgo_attribution_exclusive_equals_the_gate_statistic():
    sys.path.insert(0, str(ROOT / "tools" / "pgo"))
    import evidence

    assert at.attribution(sample())["native_exclusive_ir"] == evidence.native_cost(evidence.parse_callgrind(sample())).ir


@pytest.mark.parametrize(
    ("obj", "fn", "want"),
    [
        ("/lib/libc.so.6", "__memset_avx2_unaligned_erms", True),
        ("/lib/libc.so.6", "memmove", True),
        ("/lib/libc.so.6", "memcpy@GLIBC_2.14", True),
        ("/usr/lib/system/libsystem_platform.dylib", "__bzero", True),
        ("/usr/lib/system/libsystem_platform.dylib", "_platform_memset", True),
        ("/lib/libc.so.6", "malloc", False),
        ("/lib/libc.so.6", "memchr", False),
        ("/venv/_native.abi3.so", "memset", False),
        ("/lib/libm.so.6", "memset", False),
    ],
)
def test_pgo_attribution_memory_primitives(obj, fn, want):
    assert at.is_memory_primitive(obj, fn) is want


def test_pgo_attribution_cli_json(tmp_path, capsys):
    p = tmp_path / "s.callgrind"
    p.write_text(sample())
    assert at.main([str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[str(p)]["primitives_from_native_ir"] == 960


def test_pgo_attribution_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="no events"):
        at.primitive_calls("version: 1\n")
    with pytest.raises(ValueError, match="unknown id"):
        at.primitive_calls(f"events: {EVENTS}\nob=(9)\n")

# SPDX-License-Identifier: MIT OR Apache-2.0
"""The shared CPython boundary crate, tessera-rl-pyffi (ADR-0062 item 8,
#272 step 3): no public unsafe function, and the fixed initializers."""

import pathlib
import shutil
import subprocess
import sys

import tessera_rl._native as native

ROOT = pathlib.Path(__file__).resolve().parents[2]
LINT = ROOT / "scripts" / "check_py_entries.py"


def lint(*srcs):
    return subprocess.run(
        [sys.executable, str(LINT), *map(str, srcs)], capture_output=True, text=True, cwd=ROOT
    )


def test_pyffi_entries_lint_passes_on_the_tree():
    ok = lint()
    assert ok.returncode == 0, ok.stdout + ok.stderr


def test_pyffi_entries_lint_rejects_a_public_unsafe_function(tmp_path):
    py, ffi = tmp_path / "py", tmp_path / "pyffi"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", py)
    shutil.copytree(ROOT / "crates" / "tessera-rl-pyffi" / "src", ffi)
    capi = ffi / "ffi" / "capi.rs"
    clean = capi.read_text()
    assert lint(py, ffi).returncode == 0
    for sneaky in (
        "pub unsafe fn sneaky() {}",
        "pub const unsafe fn sneaky() {}",
        'pub unsafe extern "C" fn sneaky() {}',
        "impl Ref<'_> {\n    pub unsafe fn sneaky(&self) {}\n}",
        "pub unsafe trait Sneaky {}",
    ):
        capi.write_text(clean + "\n" + sneaky + "\n")
        bad = lint(py, ffi)
        assert bad.returncode != 0 and "public unsafe" in bad.stdout, sneaky
    capi.write_text(clean + "\n#![expect(clippy::not_unsafe_ptr_arg_deref, reason = \"no\")]\n")
    bad = lint(py, ffi)
    assert bad.returncode != 0 and "not_unsafe_ptr_arg_deref" in bad.stdout
    for fine in ("pub(crate) unsafe fn quiet() {}", "// pub unsafe fn commented() {}"):
        capi.write_text(clean + "\n" + fine + "\n")
        assert lint(py, ffi).returncode == 0, fine


def test_native_exports_its_initializer_and_not_the_host_one():
    # `_native` enables pyffi's `native-module` feature only, never `host`
    # (ADR-0062 D3): the host initializer is not in the extension at all.
    data = pathlib.Path(native.__file__).read_bytes()
    assert b"PyInit__native" in data
    assert b"PyInit__tessera_host" not in data


SECOND_INIT = """
import gc, importlib.machinery, importlib.util
import tessera_rl
import tessera_rl._native as native

loader = importlib.machinery.ExtensionFileLoader("_native", native.__file__)
spec = importlib.util.spec_from_file_location("_native", native.__file__, loader=loader)
try:
    second = importlib.util.module_from_spec(spec)
    loader.exec_module(second)
    print("second-init-ok")
except RuntimeError as e:
    print("second-init-refused:", e)
second = None
gc.collect()
try:
    native.set_fp_policy("nonsense")
except tessera_rl.ArgumentError:
    print("caught-argument-error")
"""


def test_native_second_init_neither_crashes_nor_breaks_the_exception_classes():
    # A second PyInit__native fails after the exception classes were
    # registered and frees its module; the classes must outlive it.
    run = subprocess.run([sys.executable, "-I", "-c", SECOND_INIT], capture_output=True, text=True)
    assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
    assert "second-init-refused: Env type already created" in run.stdout, run.stdout
    assert "caught-argument-error" in run.stdout, run.stdout

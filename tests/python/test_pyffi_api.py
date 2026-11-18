# SPDX-License-Identifier: MIT OR Apache-2.0
"""The shared CPython boundary crate, tessera-rl-pyffi (ADR-0062 item 8,
#272 step 3): no public unsafe function, and the fixed initializers."""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

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
        "pub trait Sneaky {\n    unsafe fn sneaky(&self);\n}",
        'pub trait Sneaky {\n    fn fine(&self) {}\n    unsafe extern "C" fn sneaky();\n}',
        "pub unsafe extern fn sneaky() {}",
        "pub async unsafe fn sneaky() {}",
        "pub const async unsafe fn sneaky() {}",
        "pub trait Sneaky<T = [u8; 4]> {\n    unsafe fn sneaky(&self);\n}",
        'pub trait Sneaky {\n    fn fine(&self) -> &str { "}" }\n    unsafe fn sneaky(&self);\n}',
        "pub trait Sneaky {\n    fn fine(&self) -> char { '}' }\n    unsafe fn sneaky(&self);\n}",
        "pub trait Sneaky {\n    /* } */\n    unsafe fn sneaky(&self);\n}",
    ):
        capi.write_text(clean + "\n" + sneaky + "\n")
        bad = lint(py, ffi)
        assert bad.returncode != 0 and "public unsafe" in bad.stdout, sneaky
    capi.write_text(clean + "\n#![expect(clippy::not_unsafe_ptr_arg_deref, reason = \"no\")]\n")
    bad = lint(py, ffi)
    assert bad.returncode != 0 and "not_unsafe_ptr_arg_deref" in bad.stdout
    for fine in (
        "pub(crate) unsafe fn quiet() {}",
        "// pub unsafe fn commented() {}",
        "pub(crate) trait Quiet {\n    unsafe fn quiet(&self);\n}",
        "pub trait Fine {\n    fn fine(&self);\n}",
        'const FINE: &str = "pub unsafe fn quoted() {}";',
        "/* pub unsafe fn commented() {} */",
        "pub trait Fine<T = [u8; 4]> {\n    fn fine(&self);\n}\nunsafe fn private_after() {}",
    ):
        capi.write_text(clean + "\n" + fine + "\n")
        assert lint(py, ffi).returncode == 0, fine


API_LINT = ROOT / "scripts" / "check_pyffi_api.py"
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "pyffi_api"


def test_pyffi_api_lint_flags_public_unsafe_items_of_a_rustdoc_json():
    # A rustdoc-shaped JSON (format 61): a pub unsafe fn and an unsafe fn in
    # a pub trait are flagged; a pub(crate) unsafe
    # fn, safe functions and another crate's items are not.
    run = subprocess.run(
        [sys.executable, str(API_LINT), str(FIXTURES / "rustdoc.json")], capture_output=True, text=True, cwd=ROOT
    )
    assert run.returncode == 1, (run.stdout, run.stderr)
    assert run.stdout.splitlines() == [
        "src/lib.rs:1: pub unsafe fn sneaky",
        "src/lib.rs:3: pub unsafe fn in_trait",
    ], run.stdout


def test_native_exports_its_initializer_and_not_the_host_one():
    # `_native` enables pyffi's `native-module` feature only, never `host`
    # (ADR-0062 D3): the host initializer is not in the extension at all.
    data = pathlib.Path(native.__file__).read_bytes()
    assert b"PyInit__native" in data
    assert b"PyInit__tessera_host" not in data


SECOND_INIT = """
import gc, importlib.machinery, importlib.util
import tessera_rl
import pytest

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


INIT_FAILS_AFTER_REGISTER = """
import gc
import tessera_rl
import pytest

import tessera_rl._native as native

for _ in range(2):
    try:
        native._init_fails_after_register()
    except RuntimeError as e:
        print("init-failed:", e)
gc.collect()
try:
    native.set_fp_policy("nonsense")
except tessera_rl.ArgumentError:
    print("caught-argument-error")
"""


@pytest.mark.skipif(
    not hasattr(native, "_init_fails_after_register"), reason="needs a test-hooks build (make py-test-hooks)"
)
def test_native_init_failing_after_register_leaves_the_exception_classes_alive():
    # A ModuleInit::init that fails after errors::register frees its module;
    # the classes it registered must outlive it.
    run = subprocess.run([sys.executable, "-I", "-c", INIT_FAILS_AFTER_REGISTER], capture_output=True, text=True)
    assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
    assert run.stdout.count("init-failed: injected: initialization fails after register") == 2, run.stdout
    assert "caught-argument-error" in run.stdout, run.stdout


FIRST_INIT_FAILS = """
import gc

for attempt in range(2):
    try:
        import tessera_rl._native
        print(f"import-{attempt}: ok")
    except Exception as e:
        print(f"import-{attempt}: {type(e).__name__}: {e}")
gc.collect()
# The package never imported, so `tessera_rl.ArgumentError` is not
# reachable by name; the class the failed import registered is found
# through the collector, and must still be alive and usable.
classes = [
    o for o in gc.get_objects()
    if isinstance(o, type) and o.__module__ == "tessera_rl" and o.__name__ == "ArgumentError"
]
print("argument-error-classes:", len(classes))
for cls in classes:
    try:
        raise cls("x")
    except cls:
        print("caught-argument-error")
"""


@pytest.mark.skipif(not hasattr(native, "_fail_after"), reason="needs a test-hooks build (make py-test-hooks)")
def test_native_first_init_failing_after_register_refuses_later_imports_with_its_cause():
    # The real first import fails after errors::register and after the Env
    # type was created (15 exception classes, Env, then the View attribute
    # fails). A later import must not rebuild the type objects ("Env type
    # already created"): it raises ImportError naming the first cause.
    env = dict(os.environ, TESSERA_TEST_INIT_FAIL_AFTER="17")
    run = subprocess.run(
        [sys.executable, "-I", "-c", FIRST_INIT_FAILS], capture_output=True, text=True, env=env
    )
    assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
    assert "import-0: RuntimeError: test hook: injected failure" in run.stdout, run.stdout
    assert (
        "import-1: ImportError: tessera_rl._native: an earlier initialization in this process failed "
        "(test hook: injected failure); start a new interpreter"
    ) in run.stdout, run.stdout
    assert "argument-error-classes: 1" in run.stdout, run.stdout
    assert "caught-argument-error" in run.stdout, run.stdout

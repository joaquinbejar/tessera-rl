# SPDX-License-Identifier: MIT OR Apache-2.0
"""#286: the gaps of the pyffi source lints, each with a case, and the
lexer cases the earlier tests left to chance (the surviving mutants of
#272 step 3's follow-up). These tests run the scripts only: no extension
module is needed."""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_py_callouts  # noqa: E402
import check_pyffi_api  # noqa: E402
import rust_lex  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "pyffi_api"


# -- rust_lex ---------------------------------------------------------------


def test_lex_strip_nested_block_comment_hides_everything_inside():
    code = rust_lex.strip("a /* x /* y */ pub unsafe fn f() {} */ b")
    assert "unsafe" not in code and "a" in code and code.rstrip().endswith("b")


def test_lex_strip_nested_block_comment_keeps_line_breaks():
    assert rust_lex.strip("a /* 1\n /* 2\n */ 3\n */ b").count("\n") == 3


@pytest.mark.parametrize(
    "lit",
    ['r"pub unsafe fn f() {}"', 'r#"a "quoted" pub unsafe fn"#', 'br##"x "# y"##', 'cr"pub unsafe fn"'],
)
def test_lex_strip_raw_strings_are_emptied(lit):
    code = rust_lex.strip(f"let s = {lit}; pub fn g() {{}}")
    assert "unsafe" not in code and '""' in code and "pub fn g()" in code


@pytest.mark.parametrize(
    "text, want",
    [
        ("a /* b", "a  "),  # an unterminated comment ends the text
        ("/**/x", " x"),
        ("/* */x", " x"),
        ("/*/**/*/x", " x"),
        ('"abc', '""'),  # an unterminated string ends the text
        ('"a\\"b" unsafe', '"" unsafe'),  # an escaped quote stays inside
        ('"\\\\" unsafe "x"', '"" unsafe ""'),  # an escaped backslash does not
        ('"a\nb"', '""\n'),
        ("ab'}'", "ab' '"),  # `b'` after an identifier is not a byte literal
        ("x'}'", "x' '"),
        ("let c = b'}'; {", "let c = ' '; {"),
        ('b"x" y', '"" y'),  # the prefix goes with the literal
        ('br"x" y', '"" y'),
    ],
)
def test_lex_strip_edges(text, want):
    assert rust_lex.strip(text) == want


def test_lex_strip_raw_string_needs_a_word_start():
    # `r"..."` right after an identifier character is not a raw string, so
    # its `\"` is an escape and the string runs on.
    assert "unsafe" not in rust_lex.strip('a_r"x\\" unsafe "')
    # At a word start it is raw: the string ends at the first `"`.
    assert "unsafe" in rust_lex.strip('r"a\\" pub unsafe fn x(){} "b"z')


def test_lex_block_end_unbalanced_is_the_end_of_the_text():
    assert rust_lex.block_end("{ a { b }", 0) == 9
    assert rust_lex.block_end("x { a } y", 2) == 7


def test_lex_strip_raw_string_keeps_line_breaks():
    assert rust_lex.strip('r#"a\nb\n"# c').count("\n") == 2


def test_lex_strip_an_identifier_ending_in_r_is_not_a_raw_string():
    code = rust_lex.strip('let ptr"x" = 1;')
    assert code.startswith("let ptr")


def test_lex_strip_char_and_lifetime():
    assert rust_lex.strip("fn f<'a>(x: &'a u8) -> char { '}' }") == "fn f<'a>(x: &'a u8) -> char { ' ' }"


def _brace_of(header):
    code = rust_lex.strip(header)
    i = rust_lex.item_brace(code, 0)
    return code[i:] if i >= 0 else None


@pytest.mark.parametrize(
    "header, body",
    [
        ("pub trait T<const B: bool = { 1 < 2 }> { fn f(); }", "{ fn f(); }"),
        ("pub trait T where [u8; { 1 << 3 }]: Sized { fn f(); }", "{ fn f(); }"),
        ("pub trait T where [u8; 1 << 3]: Sized { fn f(); }", "{ fn f(); }"),
        ("pub trait T<X = [u8; 4]> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<X: Fn([u8; 2]) -> u8> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<const N: usize = { 3 > 2 } as usize> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<F: Fn() -> u8, const B: bool = { true }> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<X = (u8,), const Y: u8 = { 1 }> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<X = [u8; 2], const Y: u8 = { 1 }> { fn f(); }", "{ fn f(); }"),
        ("pub trait T<const A: u8 = { 1 }, const B: u8 = { 2 }> { fn f(); }", "{ fn f(); }"),
        # A `>` inside brackets closes nothing outside them.
        ("pub trait T<X = [u8; 3 > 2], const Y: u8 = { 1 }> { fn f(); }", "{ fn f(); }"),
        ("fn f(a: [u8; 4]) -> Vec<u8> { 0 }", "{ 0 }"),
    ],
)
def test_lex_item_brace_finds_the_body_after_brackets_braces_and_generics(header, body):
    assert _brace_of(header) == body


@pytest.mark.parametrize("header", ["pub trait T;", "pub fn f(x: [u8; 4]) -> u8;", "pub type X<T> = Vec<T>; { }"])
def test_lex_item_brace_a_depth_0_semicolon_ends_the_item(header):
    assert _brace_of(header) is None


def test_lex_item_brace_semicolon_inside_brackets_does_not_end_the_item():
    assert _brace_of("pub trait T where [u8; 3]: Sized { }") == "{ }"


def test_lex_item_brace_unbalanced_is_not_found():
    assert _brace_of("pub trait T<X = [u8; 4") is None


# -- check_py_callouts.py ---------------------------------------------------


def _calls(code):
    return check_py_callouts.c_api_calls(rust_lex.strip(code))


@pytest.mark.parametrize(
    "code, name",
    [
        ("use pyo3::ffi::PyList_New;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("use pyo3::ffi::*;\nfn f() { unsafe { PyDict_New(); } }", "PyDict_New"),
        ("use pyo3::ffi::{PyTuple_New, PyObject};\nfn f() { unsafe { PyTuple_New(0); } }", "PyTuple_New"),
        ("use pyo3::ffi::{PyObject, PyTuple_New as make};\nfn f() { unsafe { make(0); } }", "PyTuple_New"),
        ("use pyo3::{ffi::{PyList_New}};\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("fn f() { let g = pyo3::ffi::PyErr_Clear; unsafe { g() } }", "PyErr_Clear"),
        ("use pyo3::ffi;\nfn f() { let g = ffi::PyErr_Clear; }", "PyErr_Clear"),
        ("use pyo3::ffi::PyErr_Clear;\nfn f() { let g = PyErr_Clear; }", "PyErr_Clear"),
        ("use pyo3::ffi::{self as raw};\nfn f() { unsafe { raw::PyDict_New(); } }", "PyDict_New"),
        ("use pyo3::ffi as raw;\nfn f() { unsafe { raw::PyDict_New(); } }", "PyDict_New"),
        ("use ::pyo3::ffi::PyList_New;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("use pyo3 :: ffi :: PyList_New;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("use pyo3::{ffi::{PyList_New, PyDict_New}, types};\nfn f() { unsafe { PyDict_New(); } }", "PyDict_New"),
        # Allowlisted names of any shape, and unlisted names of a function's
        # shape, taken as values.
        ("use pyo3::ffi;\nfn f() { let g = ffi::Py_TYPE; }", "Py_TYPE"),
        ("use pyo3::ffi;\nfn f() { let g = ffi::PyFoo_Bar; }", "PyFoo_Bar"),
        ("use pyo3::ffi;\nfn f() { let g = ffi::_PyFoo_Bar; }", "_PyFoo_Bar"),
    ],
)
def test_callouts_c_api_calls_sees_every_import_form(code, name):
    assert name in _calls(code)


@pytest.mark.parametrize(
    "code",
    [
        "use crate::ffi::capi::{list as make};\nfn f() { make(); }",
        "use crate::ffi::capi::{self, Mine};\nfn f() { capi::list(); }",
        "use core::ffi::{CStr, c_int};\nfn f() { let _ = c_int::MAX; }",
        "use pyo3::ffi;\nfn f(t: *mut ffi::PyTypeObject) { let _ = unsafe { ffi::PyExc_TypeError }; let _ = &raw mut ffi::PyList_Type; }",
        "use pyo3::ffi;\nconst F: u64 = ffi::Py_TPFLAGS_DEFAULT; const S: i32 = ffi::Py_tp_dealloc;",
        "use pyo3::ffi::PyObject;\nfn f(p: *mut PyObject) {}",
        "use pyo3::ffi::PyList_New as _;\nfn f() { let _ = 1; }",
        "use pyo3::ffi;\nfn f() { let t = &raw mut ffi::PyFoo_Type; }",
        "use pyo3::ffi;\nfn f() { let t = ffi::PyFoo_Bar::X; }",  # a path through it, not a value
        "fn f() { let s: ffi::PyType_Spec; }",
    ],
)
def test_callouts_c_api_calls_ignores_non_calls_and_other_ffi_paths(code):
    assert _calls(code) == set()


@pytest.mark.parametrize(
    "code, name",
    [
        # #310: a crate alias of pyo3.
        ("use pyo3 as p;\nuse p::ffi::PyList_New;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("use pyo3 as p;\nuse p::ffi as raw;\nfn f() { unsafe { raw::PyDict_New(); } }", "PyDict_New"),
        ("use pyo3::{self as p};\nuse p::ffi::{PyTuple_New as make};\nfn f() { unsafe { make(0); } }", "PyTuple_New"),
        ("use ::pyo3 as p;\nuse p::ffi::PyList_New;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        # #310: a module alias used before its `use` line.
        ("use raw::PyList_New;\nuse pyo3::ffi as raw;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        ("use q::PyList_New;\nuse p::ffi as q;\nuse pyo3 as p;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        # An alias of a crate alias, in reverse order: one pass over the
        # `use` lines is not enough (#310 follow-up review N1).
        ("use q::ffi::PyList_New;\nuse p as q;\nuse pyo3 as p;\nfn f() { unsafe { PyList_New(0); } }", "PyList_New"),
        # #310: a value after a glob import.
        ("use pyo3::ffi::*;\nfn f() { let g = PyList_New; }", "PyList_New"),
        ("use pyo3::ffi as raw;\nuse raw::*;\nfn f() { let g = PyErr_Clear; }", "PyErr_Clear"),
        # #310: a re-export, or any import of a C-API function.
        ("pub use pyo3::ffi::PyList_New;", "PyList_New"),
        ("pub(crate) use pyo3::ffi::{PyObject, PyErr_Clear as clear};", "PyErr_Clear"),
        ("use pyo3::ffi::PyErr_Clear;", "PyErr_Clear"),
    ],
)
def test_callouts_c_api_calls_sees_the_310_import_forms(code, name):
    assert name in _calls(code)


@pytest.mark.parametrize(
    "code",
    [
        "use pyo3::ffi::*;\nfn f() { let t: *mut PyObject = x; let k = Py_TPFLAGS_DEFAULT; }",
        "use pyo3 as p;\nuse p::types::PyList;\nfn f() {}",
        "use other as p;\nuse p::ffi::PyList_New;\nfn f() { PyList_New(0); }",
        "pub use pyo3::ffi::PyObject;",
        "use pyo3::ffi::PyErr_Clear as _;",
    ],
)
def test_callouts_c_api_calls_310_forms_leave_other_names_alone(code):
    assert _calls(code) == set()


def test_callouts_c_api_calls_thunks_may_still_import_its_targets():
    code = rust_lex.strip("use pyo3::ffi::PyList_New;\ngeneric_thunk!(c_list_new, PyList_New);")
    assert check_py_callouts.c_api_calls(code, values=False) == set()


@pytest.mark.parametrize(
    "code",
    [
        "use pyo3::ffi::{PyTuple_New as make};\nfn f() { remake(0); }",
        "use pyo3::ffi::PyErr_Clear;\nfn f() { let my_PyErr_Clear = 1; PyErr_ClearAll(); }",
    ],
)
def test_callouts_c_api_calls_an_imported_name_is_called_only_as_itself(code):
    # Calls only (as in thunks.rs): the import itself does not count there,
    # and a longer name or a name ending in it is not a call of it.
    assert check_py_callouts.c_api_calls(rust_lex.strip(code), values=False) == set()


def test_callouts_c_api_calls_thunks_may_name_targets_as_values():
    code = rust_lex.strip("use pyo3::ffi;\ngeneric_thunk!(c_list_new, ffi::PyList_New);")
    assert check_py_callouts.c_api_calls(code, values=False) == set()
    assert check_py_callouts.c_api_calls(code) == {"PyList_New"}


def _leaf(code):
    return check_py_callouts.leaf_owned(rust_lex.strip(code))


@pytest.mark.parametrize(
    "code",
    [
        "fn f(o: Obj) { let _ = Mine::new(unsafe { thunks::c_object_str(o) }); }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) };\n    let _ = Mine::new(s); }",
        "fn f(o: Obj) { let s: *mut ffi::PyObject = unsafe { thunks::c_object_str(o) };\n    let _ = Mine::new(s); }",
        "fn f(o: Obj) { let mut s = unsafe { thunks::c_object_str(o) }; let t = s; let u: *mut ffi::PyObject = t; Mine::new(u) }",
        "impl Mine { fn f(o: Obj) -> Self { let s = unsafe { thunks::c_object_str(o) }; Self::new(s) } }",
        "impl Mine { fn f(o: Obj) -> Self { Self::new(unsafe { thunks::c_object_str(o) }) } }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o; 1) }; Mine::new(s) }",
        # `let st` does not shadow `s`.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let st = 1; Mine::new(s) }",
        # Shadowed only inside an inner block or a closure body, used after it.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; { let s = 1; use_it(s); } Mine::new(s) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let g = || { let s = 1; s }; Mine::new(s) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; { { let s = 1; } } Mine::new(s) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let mut t = s; Mine::new(t) }",
        # Shadowed in an inner block, then rebound after it.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; { let s = 1; use_it(s); } let t = s; Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let g = || { let s = 1; s }; let t = s; Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; { let s = 1; } let t = s; let u = t; Mine::new(u) }",
        # #310: a cast at the call or in the rebinding, and a plain assignment.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::new(s.cast()) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::new(s.cast::<ffi::PyObject>()) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::new(s as *mut ffi::PyObject) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let t = s.cast(); Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let t = s as *mut _; Mine::new(t) }",
        "fn f(o: Obj) { let t; let s = unsafe { thunks::c_object_str(o) }; t = s; Mine::new(t) }",
        # A typed rebinding without a space before `=` (#310 review N2).
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let t: Option<P>= s; Mine::new(t) }",
        # A long hidden block: the rebinding would land inside it if offsets moved.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; { let s = 1; " + "a" * 80 + "(s); } let t = s; Mine::new(t) }",
    ],
)
def test_callouts_leaf_owned_follows_typed_and_rebound_bindings(code):
    assert _leaf(code) == {"c_object_str"}


@pytest.mark.parametrize(
    "code",
    [
        # A later, unrelated variable of the same name, in another function.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); }\nfn g(s: P) { Mine::new(s); }",
        # Shadowed before the `Mine::new`, with or without `mut`, and inside
        # the inner block that shadows it.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); let s = 1; Mine::new(s); }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); let mut s = 1; Mine::new(s); }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); { let s = 1; Mine::new(s); } }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); { let s = 1; { x(); } Mine::new(s); } }",
        # The binding's block ended.
        "fn f(o: Obj) { { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); } Mine::new(s); }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; let t = s; Mine::container(t); }",
        # Not a plain rebinding, and not the same name.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); let t = s.len(); Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); Mine::new(st) }",
        # Not a rebinding: a comparison, a compound assignment, a field.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); let b = t == s; Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); t += s; Mine::new(t) }",
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); x.t = s; Mine::new(t) }",
        # A method call that is not a cast.
        "fn f(o: Obj) { let s = unsafe { thunks::c_object_str(o) }; Mine::container(s); Mine::new(s.other()) }",
    ],
)
def test_callouts_leaf_owned_ignores_other_bindings_of_the_same_name(code):
    assert _leaf(code) == set()


def test_callouts_lint_passes_on_the_tree():
    run = subprocess.run([sys.executable, str(SCRIPTS / "check_py_callouts.py")], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_callouts_lint_reports_a_typed_leaf_binding_in_capi(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", src)
    capi = src / "ffi" / "capi.rs"
    shutil.copy(ROOT / "crates" / "tessera-rl-pyffi" / "src" / "ffi" / "capi.rs", capi)
    capi.write_text(capi.read_text() + "\nfn sneaky(o: Obj) { let s: *mut ffi::PyObject = unsafe { thunks::c_object_str(o) }; let t = s; let _ = Self::new(t); }\n")
    run = subprocess.run([sys.executable, str(SCRIPTS / "check_py_callouts.py"), str(src)], capture_output=True, text=True)
    assert run.returncode != 0 and "c_object_str returns a GC-tracked object" in run.stdout, run.stdout


def test_callouts_safe_foreign_looks_inside_the_extern_block_only():
    code = rust_lex.strip('unsafe extern "C" { safe fn q(); }\nmod m { pub safe fn not_in_block(); }')
    assert not check_py_callouts.safe_foreign(code)
    assert check_py_callouts.safe_foreign(rust_lex.strip('unsafe extern "C" { pub safe fn q(); }'))


def test_callouts_lint_checks_the_files_after_capi(tmp_path):
    # `split.rs` sorts after `ffi/capi.rs`: the scan goes on past capi.
    src = tmp_path / "src"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", src)
    shutil.copy(ROOT / "crates" / "tessera-rl-pyffi" / "src" / "ffi" / "capi.rs", src / "ffi" / "capi.rs")
    late = src / "split.rs"
    assert late.exists()
    late.write_text(late.read_text() + "\nfn sneaky() { unsafe { pyo3::ffi::PyList_New(0); } }\n")
    run = subprocess.run([sys.executable, str(SCRIPTS / "check_py_callouts.py"), str(src)], capture_output=True, text=True)
    assert run.returncode != 0 and "split.rs: calls ffi::PyList_New directly" in run.stdout, run.stdout


def _callouts_tree(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", src)
    shutil.copy(ROOT / "crates" / "tessera-rl-pyffi" / "src" / "ffi" / "capi.rs", src / "ffi" / "capi.rs")
    return src


def _callouts_run(src):
    return subprocess.run([sys.executable, str(SCRIPTS / "check_py_callouts.py"), str(src)], capture_output=True, text=True)


def test_callouts_lint_reports_a_function_value_outside_capi(tmp_path):
    src = _callouts_tree(tmp_path)
    api = src / "api.rs"
    api.write_text(api.read_text() + "\nfn sneaky() { let f = pyo3::ffi::PyErr_Clear; }\n")
    run = _callouts_run(src)
    assert run.returncode != 0 and "api.rs: calls ffi::PyErr_Clear directly" in run.stdout, run.stdout


def test_callouts_lint_reports_a_function_value_in_capi(tmp_path):
    src = _callouts_tree(tmp_path)
    capi = src / "ffi" / "capi.rs"
    capi.write_text(capi.read_text() + "\nfn sneaky() { let f = ffi::PyList_New; }\n")
    run = _callouts_run(src)
    assert run.returncode != 0 and "PyList_New can run Python code" in run.stdout, run.stdout


def test_callouts_lint_reports_a_re_export_outside_capi(tmp_path):
    # #310: a re-export in one file, used through `crate::` in another: the
    # import itself is reported, in the file that makes it.
    src = _callouts_tree(tmp_path)
    (src / "reexport.rs").write_text("pub use pyo3::ffi::PyList_New;\n")
    api = src / "api.rs"
    api.write_text(api.read_text() + "\nuse crate::reexport::PyList_New;\nfn sneaky() { unsafe { PyList_New(0); } }\n")
    run = _callouts_run(src)
    assert run.returncode != 0 and "reexport.rs: calls ffi::PyList_New directly" in run.stdout, run.stdout


@pytest.mark.parametrize(
    "code, want",
    [
        ("pub use pyo3::ffi::PyList_New;", {"PyList_New"}),
        ("pub(crate) use pyo3::ffi::{PyObject, PyErr_Clear as clear};", {"PyErr_Clear"}),
        ("use pyo3::ffi as raw;\npub(super) use raw::Py_TYPE;", {"Py_TYPE"}),
        ("pub(in crate::ffi) use pyo3::ffi::PyErr_Clear;", {"PyErr_Clear"}),
        ("pub use pyo3::ffi::*;", {"*"}),
        # The module itself, renamed or not (#310 follow-up review N2).
        ("pub use pyo3::ffi as raw;", {"*"}),
        ("pub(crate) use pyo3::{ffi};", {"*"}),
        ("use pyo3::ffi as raw;\npub use raw as raw2;", {"*"}),
        ("use pyo3::ffi;\nuse pyo3::ffi as raw;", set()),
        ("use pyo3::ffi::PyList_New;", set()),
        ("pub use pyo3::ffi::PyObject;", set()),
        ("pub use crate::ffi::capi::list;", set()),
    ],
)
def test_callouts_reexports(code, want):
    assert check_py_callouts.reexports(rust_lex.strip(code)) == want


@pytest.mark.parametrize("rel, line", [("ffi/thunks.rs", "pub use pyo3::ffi::PyList_New;"), ("ffi/capi.rs", "pub(crate) use pyo3::ffi::Py_IncRef;")])
def test_callouts_lint_reports_a_re_export_in_thunks_and_capi(tmp_path, rel, line):
    # #310 review N1: a re-export is reported in every file, the two files
    # that may name C-API functions included (here a callout target, and
    # an allowlisted function).
    src = _callouts_tree(tmp_path)
    f = src / rel
    if not f.exists():
        shutil.copy(ROOT / "crates" / "tessera-rl-pyffi" / "src" / rel, f)
    assert _callouts_run(src).returncode == 0
    f.write_text(f.read_text() + "\n" + line + "\n")
    run = _callouts_run(src)
    name = line.rsplit("::", 1)[1].rstrip(";")
    assert run.returncode != 0 and f"{rel}: re-exports ffi::{name}" in run.stdout, run.stdout


def test_callouts_lint_reports_a_renamed_module_re_export(tmp_path):
    # #310 follow-up review N2: the C-API module re-exported under another
    # name, then called through it from another file.
    src = _callouts_tree(tmp_path)
    (src / "reexport.rs").write_text("pub use pyo3::ffi as raw;\n")
    api = src / "api.rs"
    api.write_text(api.read_text() + "\nuse crate::reexport::raw;\nfn sneaky() { unsafe { raw::PyList_New(0); } }\n")
    run = _callouts_run(src)
    assert run.returncode != 0 and "reexport.rs: re-exports ffi::*" in run.stdout, run.stdout


def test_callouts_lint_reports_a_glob_import_outside_capi(tmp_path):
    src = tmp_path / "src"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", src)
    shutil.copy(ROOT / "crates" / "tessera-rl-pyffi" / "src" / "ffi" / "capi.rs", src / "ffi" / "capi.rs")
    api = src / "api.rs"
    api.write_text(api.read_text() + "\nuse pyo3::ffi::*;\nfn sneaky() { unsafe { PyList_New(0); } }\n")
    run = subprocess.run([sys.executable, str(SCRIPTS / "check_py_callouts.py"), str(src)], capture_output=True, text=True)
    assert run.returncode != 0 and "calls ffi::PyList_New directly" in run.stdout, run.stdout


# -- check_py_entries.py ----------------------------------------------------


def _entries_problem(tmp_path, extra):
    py, ffi = tmp_path / "py", tmp_path / "pyffi"
    shutil.copytree(ROOT / "crates" / "tessera-rl-py" / "src", py)
    shutil.copytree(ROOT / "crates" / "tessera-rl-pyffi" / "src", ffi)
    capi = ffi / "ffi" / "capi.rs"
    capi.write_text(capi.read_text() + "\n" + extra + "\n")
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "check_py_entries.py"), str(py), str(ffi)], capture_output=True, text=True, cwd=ROOT
    )


@pytest.mark.parametrize(
    "extra",
    [
        'unsafe extern "C" {\n    pub fn sneaky();\n}',
        'unsafe extern "C" {\n    pub unsafe fn sneaky();\n}',
        "unsafe extern {\n    pub static SNEAKY: u8;\n}",
        "unsafe extern {\n    pub unsafe static SNEAKY: u8;\n}",
        'unsafe extern "C" {\n    safe fn quiet();\n    pub fn sneaky();\n}',
        "pub trait T<const B: bool = { 1 < 2 }> {\n    unsafe fn sneaky(&self);\n}",
        "pub trait T where [u8; { 1 << 3 }]: Sized {\n    unsafe fn sneaky(&self);\n}",
    ],
)
def test_entries_lint_flags_implicitly_unsafe_foreign_items_and_generic_traits(tmp_path, extra):
    run = _entries_problem(tmp_path, extra)
    assert run.returncode != 0 and "public unsafe" in run.stdout, (extra, run.stdout)


@pytest.mark.parametrize(
    "extra",
    [
        'unsafe extern "C" {\n    fn quiet();\n    pub(crate) fn quieter();\n}',
        "pub trait T<const B: bool = { 1 < 2 }> {\n    fn fine(&self);\n}\nunsafe fn private_after() {}",
    ],
)
def test_entries_lint_passes_private_foreign_items_and_safe_traits(tmp_path, extra):
    run = _entries_problem(tmp_path, extra)
    assert run.returncode == 0, (extra, run.stdout)


# -- check_pyffi_api.py -----------------------------------------------------


def test_pyffi_api_lint_on_a_real_rustdoc_json():
    # FIXTURES/rustdoc-real.json is rustdoc's own JSON (format 61, nightly
    # 2026-11-15, `--document-hidden-items`) for the crate in
    # FIXTURES/rustdoc-real.rs, trimmed to that crate's own items. Its
    # `CAbi::required` (a safe C-ABI trait method) and `PLAIN` (a static
    # with a body) are not flagged.
    found = check_pyffi_api.public_unsafe(json.loads((FIXTURES / "rustdoc-real.json").read_text()))
    assert found == sorted([
        "src/lib.rs:1: pub unsafe fn sneaky",
        "src/lib.rs:11: pub unsafe trait UnsafeTrait",
        "src/lib.rs:4: pub safe foreign fn foreign_safe",
        "src/lib.rs:5: pub unsafe fn foreign_implicit",
        "src/lib.rs:6: pub unsafe fn foreign_unsafe",
        "src/lib.rs:7: pub safe foreign static SAFE_STATIC",
        "src/lib.rs:8: pub unsafe static IMPLICIT_STATIC",
    ]), found

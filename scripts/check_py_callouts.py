#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ADR-0041 callouts: every C-API call of tessera-rl-py that can run Python
code goes through a callout thunk, and only `ffi::capi` calls the C API.

Fails when, outside `ffi/capi.rs`:
  - PyO3's high-level API is used (`pyo3::types`, `pyo3::prelude`, `Bound`,
    `.extract`, `PyErr::`, `Python::`), or
  - a `pyo3::ffi` function is called (types, constants and table rows are
    fine; `thunks.rs` names the callout targets without calling them);
and when, inside `ffi/capi.rs`, a C-API function is called directly that is
not on the allowlist below (functions that cannot run Python code), in
particular any of the functions that must go through a callout.

A C-API call is `ffi::f(...)`, `X::f(...)` after `use ... ffi as X`, or
`g(...)` after `use ...::ffi::{f as g}`; a `pub safe fn` in an `unsafe
extern` block fails anywhere. String literals and comments are ignored
(scripts/rust_lex.py), so a `//` inside a string hides nothing. It is a
lint over the source text, not a proof.

Usage: check_py_callouts.py [SRC_DIR]
"""

import pathlib
import re
import sys

import rust_lex

_ROOT = pathlib.Path(__file__).resolve().parent.parent
# The bindings and the shared boundary crate (ADR-0062 item 8), which holds
# `ffi/capi.rs`.
DEFAULT_SRCS = [_ROOT / "crates" / "tessera-rl-py" / "src", _ROOT / "crates" / "tessera-rl-pyffi" / "src"]

# Direct calls `capi` may make: none of them can run Python code (the
# reasons are in capi's module docs). `Py_DecRef` is direct only for leaf
# objects; `Mine::drop` sends container decrefs through `c_decref`.
ALLOWED = {
    "Py_TYPE", "Py_None", "Py_True", "Py_False", "Py_IncRef", "Py_DecRef",
    "PyEval_SaveThread", "PyEval_RestoreThread",
    "PyErr_Fetch", "PyErr_Occurred", "PyFloat_AsDouble",
    "PyLong_FromUnsignedLongLong", "PyFloat_FromDouble", "PyUnicode_FromStringAndSize",
    "PyBytes_FromStringAndSize", "PyByteArray_FromStringAndSize",
    "PyTuple_Size", "PyTuple_GetItem", "PyList_Size", "PyList_GetItem", "PyDict_Next",
    "PyBytes_AsString", "PyBytes_Size",
    "PyType_GenericAlloc", "PyType_GetSlot",
    "PyCapsule_New", "PyCapsule_IsValid", "PyCapsule_GetPointer",
}

# Functions that can run Python code: callouts only (GC-tracked
# allocation, setting an exception, conversions that may raise, warnings,
# module attributes).
CALLOUT_ONLY = {
    "PyList_New", "PyTuple_New", "PyDict_New",
    "PyList_Append", "PyList_SetItem", "PyTuple_SetItem", "PyDict_SetItem", "PyDict_SetItemString",
    "PyModule_Create2", "PyType_FromSpec",
    "PyErr_NewException", "PyErr_NewExceptionWithDoc", "PyErr_SetString", "PyErr_SetObject",
    "PyErr_SetNone", "PyErr_Format", "PyErr_Clear", "PyErr_Restore", "PyErr_WarnEx", "PyModule_AddObjectRef",
    "PyLong_AsLongLong", "PyLong_AsUnsignedLongLong", "PyLong_AsLong", "PyNumber_Index",
    "PyUnicode_AsUTF8AndSize", "PyObject_IsTrue", "PyObject_GetIter", "PyIter_Next",
    "PyObject_GetItem", "PyObject_GetAttr", "PyObject_Str", "PyObject_Repr",
    "PyObject_Call", "PyObject_CallObject", "PyObject_CallNoArgs", "PyObject_Vectorcall",
    "PyErr_CheckSignals",
}

# Callouts that return a new GC-tracked object: it must be owned as a
# container (`Mine::container`), whose decref is a callout.
GC_ALLOCATORS = {
    "c_list_new", "c_tuple_new", "c_dict_new", "c_type_from_spec", "c_module_create",
    "c_err_new_exception", "c_object_call",
    # `str()` may return a GC-tracked `str` subclass with a `__del__`.
    "c_object_str",
}
LEAF_OWNED = re.compile(r"Mine::new\(\s*unsafe\s*\{\s*thunks::(\w+)\s*\(")
# The two-statement form: `let x = unsafe { thunks::f(..) };` then `Mine::new(x)`.
LET_THUNK = re.compile(r"\blet\s+(?:mut\s+)?(\w+)\s*=\s*unsafe\s*\{\s*thunks::(\w+)\s*\(")


def leaf_owned(code: str) -> set[str]:
    """The thunks whose result `code` owns with `Mine::new` (a leaf): inline,
    or through a `let` binding passed to `Mine::new` later in the file."""
    names = set(LEAF_OWNED.findall(code))
    for m in LET_THUNK.finditer(code):
        var, name = m.groups()
        if re.search(r"Mine::new\(\s*" + re.escape(var) + r"\s*\)", code[m.end():]):
            names.add(name)
    return names

HIGH_LEVEL = re.compile(r"pyo3::(types|prelude|sync|exceptions)\b|\bBound<|\.extract\s*::|\.extract\(|\bPyErr::|\bPython::")

# `use` declarations that rename the C API: `ffi as X` (a module alias) and
# `{f as g}` / `f as g` imported from an `ffi` path (a function alias).
USE = re.compile(r"\buse\s+([^;]*);")
MODULE_ALIAS = re.compile(r"\bffi\s+as\s+(\w+)")
NAME_ALIAS = re.compile(r"\b(\w+)\s+as\s+(\w+)")

# A `pub safe fn` (or `static`) in an `unsafe extern` block: a C function
# made safe to call, outside every callout.
UNSAFE_EXTERN = re.compile(r'\bunsafe\s+extern\s*(?:""\s*)?\{')
PUB_SAFE = re.compile(r"\bpub\s+safe\s+(?:fn|static)\b")


def strip_comments(text: str) -> str:
    """Comments removed and literals emptied (scripts/rust_lex.py), so a
    `//` inside a string hides nothing after it."""
    return rust_lex.strip(text)


def c_api_calls(code: str) -> set[str]:
    """The C-API functions `code` calls: through `ffi::`, through a module
    alias of `ffi`, or through a function imported from an `ffi` path under
    another name (reported by its C name)."""
    modules = {"ffi"}
    renamed = {}
    for m in USE.finditer(code):
        decl = re.sub(r"\s+", " ", m.group(1))
        modules.update(MODULE_ALIAS.findall(decl))
        if re.search(r"\bffi\s*::", decl):
            tail = decl.split("ffi", 1)[1]
            for name, alias in NAME_ALIAS.findall(tail):
                renamed[alias] = name
    names = set()
    call = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, modules))) + r")\s*::\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    names.update(call.findall(code))
    for alias, name in renamed.items():
        if re.search(r"(?<![\w:])" + re.escape(alias) + r"\s*\(", code):
            names.add(name)
    return names


def safe_foreign(code: str) -> bool:
    for m in UNSAFE_EXTERN.finditer(code):
        body = code[m.end() - 1:rust_lex.block_end(code, m.end() - 1)]
        if PUB_SAFE.search(body):
            return True
    return False


def check(src: pathlib.Path) -> list[str]:
    problems = []
    for path in sorted(src.rglob("*.rs")):
        rel = path.relative_to(src)
        code = strip_comments(path.read_text())
        if safe_foreign(code):
            problems.append(f"{rel}: a public safe foreign function (`pub safe fn` in an `unsafe extern` block)")
        if rel.as_posix() == "ffi/capi.rs":
            for name in sorted(leaf_owned(code) & GC_ALLOCATORS):
                problems.append(f"{rel}: {name} returns a GC-tracked object: own it with Mine::container")
            # The one direct decref is the leaf branch of `Mine::drop`.
            if len(re.findall(r"\bffi::Py_DecRef\s*\(", code)) > 1:
                problems.append(f"{rel}: Py_DecRef is called directly outside the leaf branch of Mine::drop")
            for name in sorted(c_api_calls(code)):
                if name in CALLOUT_ONLY:
                    problems.append(f"{rel}: {name} can run Python code: call it through a callout")
                elif name not in ALLOWED:
                    problems.append(f"{rel}: {name} is not on the direct-call allowlist")
            continue
        if HIGH_LEVEL.search(code):
            problems.append(f"{rel}: PyO3's high-level API is used outside ffi/capi.rs")
        for name in sorted(c_api_calls(code)):
            problems.append(f"{rel}: calls ffi::{name} directly (only ffi/capi.rs calls the C API)")
    return problems


def main() -> int:
    srcs = [pathlib.Path(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_SRCS
    problems = [p for src in srcs for p in check(src)]
    for p in problems:
        print(p)
    if problems:
        return 1
    print("py callouts: only ffi::capi calls the C API, and every call that can run Python code is a callout")
    return 0


if __name__ == "__main__":
    sys.exit(main())

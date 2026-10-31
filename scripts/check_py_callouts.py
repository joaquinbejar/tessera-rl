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

Usage: check_py_callouts.py [SRC_DIR]
"""

import pathlib
import re
import sys

DEFAULT_SRC = pathlib.Path(__file__).resolve().parent.parent / "crates" / "tessera-rl-py" / "src"

# Direct calls `capi` may make: none of them can run Python code (the
# reasons are in capi's module docs). `Py_DecRef` is direct only for leaf
# objects; `Mine::drop` sends container decrefs through `c_decref`.
ALLOWED = {
    "Py_TYPE", "Py_None", "Py_True", "Py_False", "Py_IncRef", "Py_DecRef",
    "PyEval_SaveThread", "PyEval_RestoreThread",
    "PyErr_Fetch", "PyErr_Restore", "PyErr_Occurred",
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
    "PyErr_SetNone", "PyErr_Format", "PyErr_Clear", "PyErr_WarnEx", "PyModule_AddObjectRef",
    "PyLong_AsLongLong", "PyLong_AsUnsignedLongLong", "PyLong_AsLong", "PyNumber_Index",
    "PyUnicode_AsUTF8AndSize", "PyObject_IsTrue", "PyObject_GetIter", "PyIter_Next",
    "PyObject_GetItem", "PyObject_GetAttr", "PyObject_Str", "PyObject_Repr",
}

HIGH_LEVEL = re.compile(r"pyo3::(types|prelude|sync|exceptions)\b|\bBound<|\.extract\s*::|\.extract\(|\bPyErr::|\bPython::")
CALL = re.compile(r"\bffi::([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def strip_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def check(src: pathlib.Path) -> list[str]:
    problems = []
    for path in sorted(src.rglob("*.rs")):
        rel = path.relative_to(src)
        code = strip_comments(path.read_text())
        if rel.as_posix() == "ffi/capi.rs":
            # The one direct decref is the leaf branch of `Mine::drop`.
            if len(re.findall(r"\bffi::Py_DecRef\s*\(", code)) > 1:
                problems.append(f"{rel}: Py_DecRef is called directly outside the leaf branch of Mine::drop")
            for name in sorted(set(CALL.findall(code))):
                if name in CALLOUT_ONLY:
                    problems.append(f"{rel}: {name} can run Python code: call it through a callout")
                elif name not in ALLOWED:
                    problems.append(f"{rel}: {name} is not on the direct-call allowlist")
            continue
        if HIGH_LEVEL.search(code):
            problems.append(f"{rel}: PyO3's high-level API is used outside ffi/capi.rs")
        for name in sorted(set(CALL.findall(code))):
            problems.append(f"{rel}: calls ffi::{name} directly (only ffi/capi.rs calls the C API)")
    return problems


def main() -> int:
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    problems = check(src)
    for p in problems:
        print(p)
    if problems:
        return 1
    print("py callouts: only ffi::capi calls the C API, and every call that can run Python code is a callout")
    return 0


if __name__ == "__main__":
    sys.exit(main())

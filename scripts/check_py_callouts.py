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

A C-API call is `ffi::f(...)`, `X::f(...)` after `use pyo3::ffi as X` or
`use pyo3::ffi::{self as X}`, or `g(...)` after `use pyo3::ffi::f`, `use
pyo3::ffi::f as g` or a group import of either; after a glob import
(`use pyo3::ffi::*`) any `Py...(...)` call counts. A function taken as a
value (`let f = ffi::PyList_New;`), imported at all (a `use` or `pub use`
of it, so a re-export is reported where it is made) or named bare after a
glob import counts as a call, except in `thunks.rs`, which names the
callout targets that way. Only `pyo3::ffi` paths (also through a crate
alias, `use pyo3 as p;`), or `ffi` and its aliases, are the C API:
`crate::ffi::capi::{list as make}` is not. The `use` lines are read in any
order.
A `pub safe fn` in an `unsafe extern` block fails anywhere. String literals and comments are ignored
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
    "PyBuffer_IsContiguous",  # reads a filled Py_buffer's fields only (#152 E2)
    "PyType_GenericAlloc", "PyType_GetSlot",
    "PyCapsule_New", "PyCapsule_IsValid", "PyCapsule_GetPointer",
    "PyType_Check", "PyType_IsSubtype", "PyErr_ExceptionMatches",
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
    "PyErr_CheckSignals", "PyImport_ImportModule", "PyObject_GetAttrString",
    "PyObject_GetBuffer", "PyBuffer_ToContiguous", "PyBuffer_Release",  # exporter code; a failed request sets an exception
}

# Callouts that return a new GC-tracked object: it must be owned as a
# container (`Mine::container`), whose decref is a callout.
GC_ALLOCATORS = {
    "c_list_new", "c_tuple_new", "c_dict_new", "c_type_from_spec", "c_module_create",
    "c_err_new_exception", "c_object_call", "c_import_module", "c_get_attr_string",
    # `str()` may return a GC-tracked `str` subclass with a `__del__`.
    "c_object_str",
}
LEAF_OWNED = re.compile(r"\b(?:Mine|Self)::new\(\s*unsafe\s*\{\s*thunks::(\w+)\s*\(")
# The two-statement form: `let x [: T] = unsafe { thunks::f(..) };`, then
# `Mine::new(x)` (or `Self::new(x)` inside `impl Mine`), directly or through
# plain rebindings `let y = x;`, while `x` is in scope and not shadowed.
LET_THUNK = re.compile(r"\blet\s+(?:mut\s+)?(\w+)\s*(?::[^=;]*)?=\s*unsafe\s*\{\s*thunks::(\w+)\s*\(")


# A value passed on unchanged but for its pointer type: `x`, `x.cast()`,
# `x.cast::<T>()`, `x as *mut T`.
CAST = r"\b(?:\s*\.\s*cast\s*(?:::\s*<[^;()]*>)?\s*\(\s*\)|\s+as\s+[^;(){}]+?)?"
# A rebinding of a name: `let y = x;` (typed or not, `mut` or not) or a
# plain assignment `y = x;` (not `==`, `<=`, `+=` and the like).
REBIND_HEAD = r"(?:\blet\s+(?:mut\s+)?(\w+)\s*(?::[^=;]*)?|(?<![\w.])(\w+)\s*)=(?!=)\s*"


def enclosing_end(code: str, pos: int) -> int:
    """The end of the innermost `{...}` block holding `pos` (`len(code)` at
    the top level), in stripped code."""
    depth = 0
    for i in range(pos - 1, -1, -1):
        if code[i] == "}":
            depth += 1
        elif code[i] == "{":
            if depth == 0:
                return rust_lex.block_end(code, i)
            depth -= 1
    return len(code)


def scope_of(code: str, var: str, start: int) -> str:
    """The code where the binding of `var` made just before `start` is
    visible: up to the end of its block, or to the next `let` of the same
    name in that block. A `let` inside a nested block (or a closure's body)
    shadows it only there, so the rest of that block is blanked (spaces, so
    offsets into the scope stay offsets into `code`) and the scan goes on
    after it."""
    end = enclosing_end(code, start)
    shadow = re.compile(r"\blet\s+(?:mut\s+)?" + re.escape(var) + r"\b")
    out, i, depth, hide = [], start, 0, None  # hide: the depth of the hiding block
    while i < end:
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if hide is not None and depth < hide:
                hide = None
        elif hide is None and shadow.match(code, i):
            if depth == 0:
                break
            # Shadowed in a nested block: hidden from here to its end.
            hide = depth
        out.append(ch if hide is None else " ")
        i += 1
    return "".join(out)


def bindings(code: str, var: str, start: int) -> list[tuple[str, str]]:
    """`var` and every plain rebinding of it (`let y = var;`), transitively,
    each with the code where it is visible."""
    out, todo, seen = [], [(var, start)], {var}
    while todo:
        name, at = todo.pop()
        scope = scope_of(code, name, at)
        out.append((name, scope))
        rebind = re.compile(REBIND_HEAD + re.escape(name) + CAST + r"\s*;")
        for m in rebind.finditer(scope):
            new = m.group(1) or m.group(2)
            if new not in seen:
                seen.add(new)
                todo.append((new, at + m.end()))
    return out


def leaf_owned(code: str) -> set[str]:
    """The thunks whose result `code` owns with `Mine::new` (a leaf): inline,
    or through a `let` binding (typed or not, rebound or not) passed to
    `Mine::new` or `Self::new` while the binding is in scope."""
    names = set(LEAF_OWNED.findall(code))
    for m in LET_THUNK.finditer(code):
        var, name = m.groups()
        # The binding is visible from the end of its `unsafe { .. }` on.
        after = rust_lex.block_end(code, code.rfind("{", m.start(), m.end()))
        for alias, scope in bindings(code, var, after):
            if re.search(r"\b(?:Mine|Self)::new\(\s*" + re.escape(alias) + CAST + r"\s*\)", scope):
                names.add(name)
    return names

HIGH_LEVEL = re.compile(r"pyo3::(types|prelude|sync|exceptions)\b|\bBound<|\.extract\s*::|\.extract\(|\bPyErr::|\bPython::")

# `use` declarations that bring in the C API. A path names the C API when
# it is `pyo3::ffi` (or `::pyo3::ffi`), or `ffi` or a module alias of it on
# its own; `crate::ffi::capi::...`, `core::ffi::...` and the like do not.
USE = re.compile(r"\buse\s+([^;]*);")
def is_c_api_path(path: str, crates: set[str], modules: set[str]) -> bool:
    """`[::]C::ffi` for `pyo3` or a crate alias of it (`use pyo3 as p;`), or
    `ffi` or another module alias of the C API (`modules` holds `ffi`)."""
    path = path.removeprefix("::")
    return path in modules or any(path == c + "::ffi" for c in crates)
MODULE_ALIAS = re.compile(r"\bffi\s+as\s+(\w+)")
# A C-API function's name: `PyList_New`, `Py_DecRef`, `_PyObject_GetX`.
# Statics, constants and types mostly have other shapes (`PyExc_TypeError`,
# `PyList_Type`, `Py_TPFLAGS_DEFAULT`, `Py_tp_dealloc`, `PyObject`). Some
# non-functions of pyo3-ffi do have this shape (`PyModuleDef_Slot`,
# `PyCapsule_Destructor`, `Py_Version`, ...): taken as a value, such a name
# is reported, which fails closed; the two the tree could meet are named.
# The shape also misses some functions (`PyObject_GC_UnTrack`,
# `PyList_GET_ITEM`) taken as values; their calls are still seen.
FN_NAME = re.compile(r"^_?Py[A-Za-z0-9]*_[A-Z][a-z][A-Za-z0-9]*$")
NOT_FN = {"PyType_Spec", "PyType_Slot"}


def is_fn_name(name: str) -> bool:
    if name in ALLOWED or name in CALLOUT_ONLY:
        return True
    return bool(FN_NAME.match(name)) and not name.startswith("PyExc_") and not name.endswith("_Type") and name not in NOT_FN


def _split_group(body: str) -> list[str]:
    """The top-level items of a `use` group's body (`a, b as c, d::{e, f}`)."""
    items, depth, cur = [], 0, ""
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip())
    return items


def use_tree(tree: str, prefix: str, crates: set[str], modules: set[str], out: dict, globs: list) -> None:
    """Walks a `use` tree under `prefix` (the path so far). Records crate
    aliases of `pyo3` (`pyo3 as p`) and, for a C-API path: module aliases
    (`ffi as X`, `ffi::{self as X}`), imported functions by local name (`f`,
    `f as g`, inside groups) and glob imports (`ffi::*`)."""
    tree = tree.strip()
    if tree.endswith("}"):
        head, body = tree.split("{", 1)
        path = "::".join(p for p in (prefix, head.strip().strip(":")) if p)
        for item in _split_group(body[:-1]):
            use_tree(item, path, crates, modules, out, globs)
        return
    alias = None
    m = re.match(r"^(.*?)\s+as\s+(\w+)$", tree)
    if m:
        tree, alias = m.group(1).strip(), m.group(2)
    full = (prefix + "::" + tree if prefix else tree).replace(" ", "")
    parent, _, last = full.rpartition("::")
    if last == "self":
        full, (parent, _, last) = parent, parent.rpartition("::")
    if full.removeprefix("::") in crates:
        if alias and alias != "_":
            crates.add(alias)
        return
    if is_c_api_path(full, crates, modules):
        if alias and alias != "_":
            modules.add(alias)
        return
    if not is_c_api_path(parent, crates, modules):
        return
    if last == "*":
        globs.append(full)
    elif alias != "_":
        out[alias or last] = last


def imports(code: str, pattern: re.Pattern = USE) -> tuple[set, set, dict, list]:
    """The crate and module aliases of the C API in `code`, and the C-API
    names and globs imported by the declarations `pattern` matches. The
    aliases come from every `use` line, read until nothing new is learnt,
    so their order does not matter."""
    crates, modules = {"pyo3"}, {"ffi"}
    decls = [re.sub(r"\s+", " ", m.group(1)).strip() for m in USE.finditer(code)]
    while True:
        size = (len(crates), len(modules))
        for decl in decls:
            modules.update(MODULE_ALIAS.findall(decl))
            use_tree(decl, "", crates, modules, {}, [])
        if (len(crates), len(modules)) == size:
            break
    imported, globs = {}, []
    for m in pattern.finditer(code):
        use_tree(re.sub(r"\s+", " ", m.group(1)).strip(), "", set(crates), set(modules), imported, globs)
    return crates, modules, imported, globs


# A `use` with any visibility: `pub use`, `pub(crate) use`, `pub(in ..) use`.
PUB_USE = re.compile(r"\bpub(?:\s*\([^)]*\))?\s+use\s+([^;]*);")


def reexports(code: str) -> set[str]:
    """The C-API functions `code` re-exports (`pub use pyo3::ffi::X`), and
    `*` for a re-exported glob of the C API. A C-API function leaves its
    file only through a callout, in every file, `capi.rs` and `thunks.rs`
    included."""
    _, _, imported, globs = imports(code, PUB_USE)
    return {name for name in imported.values() if is_fn_name(name)} | ({"*"} if globs else set())


def c_api_calls(code: str, values: bool = True) -> set[str]:
    """The C-API functions `code` calls or takes as a value: through
    `ffi::`, through a module alias of `ffi`, or through a function imported
    from the C API (plain, renamed, in a group; reported by its C name), or
    any `Py...(...)` call after a glob import of the C API. With `values`
    (every file but `thunks.rs`, which names its targets as values), a
    function of the C API also counts when taken as a value, when imported
    at all (a `use` of it, `pub use` re-exports included), and as a bare
    name after a glob import. The `use` lines are read until nothing new
    is learnt, so their order does not matter."""
    crates, modules, imported, globs = imports(code)
    # The `use` declarations themselves are not uses of the names.
    body = USE.sub(" ", code)
    names = set()
    path = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, modules))) + r")\s*::\s*([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*::)(\s*\()?")
    for m in path.finditer(body):
        name, called = m.group(1), m.group(2)
        # A call, or a function taken as a value (`let f = ffi::PyList_New;`).
        if called or (values and is_fn_name(name)):
            names.add(name)
    for local, name in imported.items():
        if values and is_fn_name(name):
            names.add(name)
        elif re.search(r"(?<![\w:.])" + re.escape(local) + r"\s*\(", body):
            names.add(name)
    if globs:
        names.update(re.findall(r"(?<![\w:.])(_?Py[A-Za-z0-9_]*)\s*\(", body))
        if values:
            bare = re.findall(r"(?<![\w:.])(_?Py[A-Za-z0-9_]*)\b(?!\s*::)", body)
            names.update(n for n in bare if is_fn_name(n))
    return names


# A `pub safe fn` (or `static`) in an `unsafe extern` block: a C function
# made safe to call, outside every callout.
UNSAFE_EXTERN = re.compile(r'\bunsafe\s+extern\s*(?:""\s*)?\{')
PUB_SAFE = re.compile(r"\bpub\s+safe\s+(?:fn|static)\b")


def strip_comments(text: str) -> str:
    """Comments removed and literals emptied (scripts/rust_lex.py), so a
    `//` inside a string hides nothing after it."""
    return rust_lex.strip(text)


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
        for name in sorted(reexports(code)):
            problems.append(f"{rel}: re-exports ffi::{name} (a C-API function is reached only through a callout)")
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
        for name in sorted(c_api_calls(code, values=rel.as_posix() != "ffi/thunks.rs")):
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

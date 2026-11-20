#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""#272 step 3: tessera-rl-pyffi exposes no public unsafe function.

The source lint of scripts/check_py_entries.py is the first; this is the
second, from what rustdoc sees: every function or method of the crate
that is public and `unsafe`, hidden items included, under every feature,
including a method declared in a public trait (rustdoc gives it the
visibility "default"), every public foreign function declared
`pub safe fn` in an `unsafe extern` block, every public `unsafe trait`,
and every public foreign static: `pub static` (unsafe to read) or
`pub safe static` in an extern block. Like the source scan, it is a lint,
not a proof: it sees what rustdoc documents.
It needs a nightly toolchain (rustdoc's JSON output is unstable).

Usage: check_pyffi_api.py [JSON]   (default: build it with cargo +nightly)
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FEATURES = "pyo3/extension-module,native-module,host,test-hooks"


def build() -> pathlib.Path:
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    subprocess.run(
        [
            "cargo", "+nightly", "rustdoc", "-q", "-p", "tessera-rl-pyffi", "--lib", "--locked",
            "--features", FEATURES, "--",
            "-Z", "unstable-options", "--output-format", "json", "--document-hidden-items",
        ],
        cwd=ROOT, check=True,
    )
    return target / "doc" / "tessera_rl_pyffi.json"


def public_unsafe(doc: dict) -> list[str]:
    found = []
    index = doc["index"]
    # Methods declared in a public trait have visibility "default": collect
    # the ids of every public trait's items, which are as public as it is.
    in_pub_trait = set()
    for item in index.values():
        trait = item.get("inner", {}).get("trait")
        if trait and item.get("crate_id") == 0 and item.get("visibility") == "public":
            in_pub_trait.update(str(i) for i in trait.get("items", []))
    for key, item in index.items():
        public = item.get("visibility") == "public" or str(key) in in_pub_trait
        if item.get("crate_id") != 0 or not public:
            continue
        inner = item.get("inner", {})
        span = item.get("span") or {}
        where = f'{span.get("filename", "?")}:{(span.get("begin") or ["?"])[0]}'
        name = item.get("name")
        trait = inner.get("trait")
        if trait is not None and trait.get("is_unsafe"):
            found.append(f"{where}: pub unsafe trait {name}")
        static = inner.get("static")
        if static is not None:
            # A foreign static has no initializer: rustdoc gives it an
            # empty `expr`; `is_unsafe` is set unless it is declared `safe`.
            if static.get("is_unsafe"):
                found.append(f"{where}: pub unsafe static {name}")
            elif static.get("expr", "") == "":
                found.append(f"{where}: pub safe foreign static {name}")
        function = inner.get("function")
        if not function:
            continue
        header = function.get("header", {})
        if header.get("is_unsafe"):
            found.append(f"{where}: pub unsafe fn {name}")
        elif header.get("abi", "Rust") != "Rust" and not function.get("has_body", True) and str(key) not in in_pub_trait:
            # A foreign function declared `pub safe fn` in an `unsafe
            # extern` block: safe to call, with no Rust body to check. A
            # required trait method with a C ABI has no body either, but
            # it is not foreign.
            found.append(f"{where}: pub safe foreign fn {name}")
    return sorted(found)


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else build()
    found = public_unsafe(json.loads(path.read_text()))
    for line in found:
        print(line)
    if found:
        return 1
    print("pyffi api: no public unsafe function")
    return 0


if __name__ == "__main__":
    sys.exit(main())

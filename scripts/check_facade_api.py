#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The public-API snapshot of the `tessera-rl` facade (ADR-0094 item 7, #642).

Every public item reachable from `tessera_rl` and its `prelude` with the
default features, plus the `onnx` feature's re-exports, is written as one
line per item under its facade path, with its signature, to
`api/tessera-rl.api`. `check` rebuilds it and fails on any difference;
`bless` rewrites the file after a reviewed change.

The items come from rustdoc's JSON, built with the pinned toolchain
(`rust-toolchain.toml`). The JSON output is unstable, so `RUSTC_BOOTSTRAP=1`
is set for the rustdoc subprocess only, and the snapshot header records the
JSON `format_version`. A different version is refused with a message to
re-bless after a toolchain bump (and to check the diff is format-only).

Left out, as ADR-0094 item 2 does: `#[doc(hidden)]` items (rustdoc omits
them), names starting with `_`, test features (never enabled here), blanket
impls, and the experimental surfaces of ADR-0094 Q1 (the host environment
API) while `INCLUDE_EXPERIMENTAL` is false.

Usage:
  check_facade_api.py check [--json DIR]   diff against api/tessera-rl.api
  check_facade_api.py bless [--json DIR]   rewrite api/tessera-rl.api
  check_facade_api.py print [--json DIR]   print the snapshot
`--json DIR` reads already built JSON files instead of running rustdoc.
"""

from __future__ import annotations

import difflib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "api" / "tessera-rl.api"

# The JSON format this script reads (rustdoc of the pinned toolchain).
FORMAT_VERSION = 61

# ADR-0094 Q1, the owner's call: the host environment API is experimental
# until decided. Flip to True (and re-bless) to put it in the snapshot.
INCLUDE_EXPERIMENTAL = False

# The experimental surface (ADR-0094 item 3), by **defining** path: an item
# defined under one of these is left out under every facade path that
# reaches it (`tessera_rl::core::HostOut` and `tessera_rl::core::host_out::
# HostOut` alike), with its fields, methods and impls. Each must match at
# least one item (a test checks it), so a rename cannot slip into the stable
# snapshot.
EXPERIMENTAL = (
    "tessera_rl_core::host",
    "tessera_rl_core::host_out",
    "tessera_rl_core::host_snapshot",
    "tessera_rl_core::envs::host_record::HostRegistration",
    "tessera_rl_core::envs::lookup::registered_host",
    "tessera_rl_core::register_host",
)

# The facade's crates: each is documented with the facade's feature set for
# it; `tessera_rl` itself with `onnx` on (ADR-0094 item 2).
FACADE = "tessera-rl"
FACADE_FEATURES = "onnx"
MEMBERS = (
    "tessera-rl-core",
    "tessera-rl-num",
    "tessera-rl-macros",
    "tessera-rl-engine",
    "tessera-rl-classic",
    "tessera-rl-games",
    "tessera-rl-markets",
    "tessera-rl-ops",
    "tessera-rl-world",
    "tessera-rl-search",
    "tessera-rl-simd",
    "tessera-rl-onnx",
)

# Auto traits whose loss breaks users; other synthetic impls are noise.
AUTO_TRAITS = ("Send", "Sync", "Unpin", "UnwindSafe", "RefUnwindSafe")


def build(out: pathlib.Path) -> pathlib.Path:
    """Runs rustdoc's JSON output for the facade and every member crate."""
    target = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    env = dict(os.environ, RUSTC_BOOTSTRAP="1")
    for krate in (FACADE, *MEMBERS):
        cmd = ["cargo", "rustdoc", "-q", "-p", krate, "--lib", "--locked"]
        if krate == FACADE:
            cmd += ["--features", FACADE_FEATURES]
        cmd += ["--", "-Z", "unstable-options", "--output-format", "json"]
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    return target / "doc" if out is None else out


class Crates:
    """Every crate's JSON, and lookup of an item by crate and canonical path."""

    def __init__(self, docs: dict[str, dict]):
        self.docs = docs
        # crate name -> canonical path -> local id
        self.by_path: dict[str, dict[tuple, str]] = {}
        for name, doc in docs.items():
            fmt = doc.get("format_version")
            if fmt != FORMAT_VERSION:
                raise SystemExit(
                    f"rustdoc JSON format {fmt} for {name}, this script reads {FORMAT_VERSION}: "
                    "after a toolchain bump, update FORMAT_VERSION, re-bless, and check the diff is format-only"
                )
            table = {}
            for key, entry in doc["paths"].items():
                if entry["crate_id"] == 0:
                    table[tuple(entry["path"])] = key
            self.by_path[name] = table

    def item(self, crate: str, key) -> dict | None:
        return self.docs[crate]["index"].get(str(key))

    def resolve(self, crate: str, key) -> tuple[str, str] | None:
        """The (crate, local id) an id of `crate` names, across crates."""
        key = str(key)
        if key in self.docs[crate]["index"]:
            return crate, key
        entry = self.docs[crate]["paths"].get(key)
        if entry is None:
            return None
        ext = self.docs[crate]["external_crates"].get(str(entry["crate_id"]))
        if ext is None or ext["name"] not in self.docs:
            return None
        local = self.by_path[ext["name"]].get(tuple(entry["path"]))
        if local is None:
            return None
        return ext["name"], local

    def canonical(self, crate: str, key) -> str | None:
        """The canonical `crate::path` of an id seen from `crate`."""
        entry = self.docs[crate]["paths"].get(str(key))
        if entry is None:
            return None
        if entry["crate_id"] == 0:
            return "::".join(entry["path"])
        ext = self.docs[crate]["external_crates"].get(str(entry["crate_id"]))
        return "::".join(entry["path"]) if ext else None


class Printer:
    """Renders rustdoc JSON types, generics and signatures as text."""

    def __init__(self, crates: Crates):
        self.crates = crates
        self.crate = ""

    def path(self, p: dict) -> str:
        name = self.crates.canonical(self.crate, p.get("id")) or p.get("path", "?")
        return name + self.args(p.get("args"))

    def args(self, a: dict | None) -> str:
        if not a:
            return ""
        if "angle_bracketed" in a:
            ab = a["angle_bracketed"]
            parts = [self.arg(x) for x in ab.get("args", [])]
            parts += [self.constraint(c) for c in ab.get("constraints", [])]
            return "<" + ", ".join(parts) + ">" if parts else ""
        if "parenthesized" in a:
            p = a["parenthesized"]
            out = "(" + ", ".join(self.ty(t) for t in p.get("inputs", [])) + ")"
            if p.get("output") is not None:
                out += " -> " + self.ty(p["output"])
            return out
        return ""

    def arg(self, x: dict) -> str:
        if "lifetime" in x:
            return x["lifetime"]
        if "type" in x:
            return self.ty(x["type"])
        if "const" in x:
            return str(x["const"].get("expr", x["const"]))
        return "_"

    def constraint(self, c: dict) -> str:
        name = c["name"] + self.args(c.get("args"))
        b = c.get("binding", {})
        if "equality" in b:
            eq = b["equality"]
            return name + " = " + (self.ty(eq["type"]) if "type" in eq else str(eq.get("constant", "")))
        if "constraint" in b:
            return name + ": " + self.bounds(b["constraint"])
        return name

    def bounds(self, bs: list) -> str:
        return " + ".join(self.bound(b) for b in bs)

    def bound(self, b: dict) -> str:
        if "trait_bound" in b:
            tb = b["trait_bound"]
            mod = {"maybe": "?", "maybe_const": "~const "}.get(tb.get("modifier", "none"), "")
            hr = self.hrtb(tb.get("generic_params", []))
            return hr + mod + self.path(tb["trait"])
        if "outlives" in b:
            return b["outlives"]
        if "use" in b:
            return "use<" + ", ".join(str(u) for u in b["use"]) + ">"
        return "?"

    def hrtb(self, params: list) -> str:
        names = [p["name"] for p in params]
        return "for<" + ", ".join(names) + "> " if names else ""

    def ty(self, t) -> str:
        if t is None:
            return "()"
        if isinstance(t, str):
            return t
        (kind, v), = t.items()
        if kind == "resolved_path":
            return self.path(v)
        if kind == "primitive":
            return v
        if kind == "generic":
            return v
        if kind == "tuple":
            return "(" + ", ".join(self.ty(x) for x in v) + ("," if len(v) == 1 else "") + ")"
        if kind == "slice":
            return "[" + self.ty(v) + "]"
        if kind == "array":
            return "[" + self.ty(v["type"]) + "; " + str(v["len"]) + "]"
        if kind == "borrowed_ref":
            lt = (v["lifetime"] + " ") if v.get("lifetime") else ""
            return "&" + lt + ("mut " if v.get("is_mutable") else "") + self.ty(v["type"])
        if kind == "raw_pointer":
            return ("*mut " if v.get("is_mutable") else "*const ") + self.ty(v["type"])
        if kind == "impl_trait":
            return "impl " + self.bounds(v)
        if kind == "dyn_trait":
            traits = [self.hrtb(p.get("generic_params", [])) + self.path(p["trait"]) for p in v["traits"]]
            if v.get("lifetime"):
                traits.append(v["lifetime"])
            return "dyn " + " + ".join(traits)
        if kind == "qualified_path":
            trait = self.path(v["trait"]) if v.get("trait") else "?"
            return f"<{self.ty(v['self_type'])} as {trait}>::{v['name']}" + self.args(v.get("args"))
        if kind == "function_pointer":
            return self.fn_pointer(v)
        if kind == "pat":
            return self.ty(v["type"])
        return kind

    def fn_pointer(self, v: dict) -> str:
        h = v.get("header", {})
        prefix = ("unsafe " if h.get("is_unsafe") else "") + self.abi(h.get("abi"))
        sig = v["sig"]
        ins = ", ".join(self.ty(t) for _, t in sig["inputs"])
        out = "fn(" + ins + ")"
        if sig.get("output") is not None:
            out += " -> " + self.ty(sig["output"])
        return self.hrtb(v.get("generic_params", [])) + prefix + out

    def abi(self, abi) -> str:
        if abi in (None, "Rust"):
            return ""
        if isinstance(abi, dict):
            (name, _), = abi.items()
            return f'extern "{name}" '
        return f'extern "{abi}" '

    def generics(self, g: dict | None) -> tuple[str, str]:
        """`<...>` and the `where` clause."""
        if not g:
            return "", ""
        params = []
        for p in g.get("params", []):
            k = p["kind"]
            if "lifetime" in k:
                outl = k["lifetime"].get("outlives", [])
                params.append(p["name"] + (": " + " + ".join(outl) if outl else ""))
            elif "type" in k:
                if k["type"].get("is_synthetic"):
                    continue
                bs = k["type"].get("bounds", [])
                d = k["type"].get("default")
                params.append(p["name"] + (": " + self.bounds(bs) if bs else "") + (" = " + self.ty(d) if d else ""))
            elif "const" in k:
                params.append("const " + p["name"] + ": " + self.ty(k["const"]["type"]))
        preds = []
        for w in g.get("where_predicates", []):
            if "bound_predicate" in w:
                bp = w["bound_predicate"]
                preds.append(self.hrtb(bp.get("generic_params", [])) + self.ty(bp["type"]) + ": " + self.bounds(bp["bounds"]))
            elif "lifetime_predicate" in w:
                lp = w["lifetime_predicate"]
                preds.append(lp["lifetime"] + ": " + " + ".join(lp["outlives"]))
            elif "eq_predicate" in w:
                ep = w["eq_predicate"]
                rhs = ep["rhs"]
                preds.append(self.ty(ep["lhs"]) + " == " + (self.ty(rhs["type"]) if "type" in rhs else str(rhs)))
        return ("<" + ", ".join(params) + ">" if params else ""), (" where " + ", ".join(preds) if preds else "")

    def function(self, name: str, f: dict) -> str:
        h = f.get("header", {})
        quals = "".join(
            q for q, on in (("const ", h.get("is_const")), ("async ", h.get("is_async")), ("unsafe ", h.get("is_unsafe"))) if on
        ) + self.abi(h.get("abi"))
        gen, where = self.generics(f.get("generics"))
        sig = f["sig"]
        ins = ", ".join(f"{n}: {self.ty(t)}" if n != "self" else self.self_param(t) for n, t in sig["inputs"])
        if sig.get("is_c_variadic"):
            ins += ", ..."
        out = f"{quals}fn {name}{gen}({ins})"
        if sig.get("output") is not None:
            out += " -> " + self.ty(sig["output"])
        return out + where

    def self_param(self, t) -> str:
        if isinstance(t, dict) and "borrowed_ref" in t:
            v = t["borrowed_ref"]
            if v["type"] == {"generic": "Self"}:
                lt = (v["lifetime"] + " ") if v.get("lifetime") else ""
                return "&" + lt + ("mut " if v.get("is_mutable") else "") + "self"
        if t == {"generic": "Self"}:
            return "self"
        return "self: " + self.ty(t)


class Walker:
    """Collects the facade's public items as snapshot lines."""

    def __init__(self, crates: Crates):
        self.c = crates
        self.p = Printer(crates)
        self.lines: set[str] = set()
        self.seen: set[tuple] = set()
        # The defining path of the item being emitted, and the experimental
        # prefixes that matched something.
        self.owner = ""
        self.matched: set[str] = set()

    def experimental(self, canon: str) -> str | None:
        for e in EXPERIMENTAL:
            if canon == e or canon.startswith(e + "::"):
                return e
        return None

    def emit(self, line: str, path: str) -> None:
        hit = self.experimental(self.owner)
        if hit is not None:
            self.matched.add(hit)
            if not INCLUDE_EXPERIMENTAL:
                return
        self.lines.add(line)

    def module(self, crate: str, key: str, path: str) -> None:
        if (crate, key, path) in self.seen:
            return
        self.seen.add((crate, key, path))
        item = self.c.item(crate, key)
        for child in item["inner"]["module"]["items"]:
            self.child(crate, str(child), path)

    def child(self, crate: str, key: str, path: str) -> None:
        item = self.c.item(crate, key)
        if item is None or item.get("visibility") != "public":
            return
        inner = item["inner"]
        if "use" in inner:
            u = inner["use"]
            if u.get("id") is None:
                return
            target = self.c.resolve(crate, u["id"])
            if target is None:
                # Outside the documented crates (std, a dependency): record the re-export itself.
                self.emit(f"use {path}::{u['name']} = {u['source']}", f"{path}::{u['name']}")
                return
            tcrate, tkey = target
            titem = self.c.item(tcrate, tkey)
            if u.get("is_glob"):
                if titem and "module" in titem["inner"]:
                    for c in titem["inner"]["module"]["items"]:
                        self.child(tcrate, str(c), path)
                return
            self.item(tcrate, tkey, f"{path}::{u['name']}", titem)
            return
        self.item(crate, key, f"{path}::{item['name']}", item)

    def item(self, crate: str, key: str, path: str, item: dict | None) -> None:
        if item is None:
            return
        name = path.rsplit("::", 1)[-1]
        if name.startswith("_"):
            return
        self.owner = self.c.canonical(crate, key) or path
        self.p.crate = crate
        inner = item["inner"]
        (kind, v), = inner.items()
        if kind == "module":
            self.emit(f"mod {path}", path)
            self.module(crate, key, path)
        elif kind == "function":
            self.emit(f"fn {path}: {self.p.function(name, v)}", path)
        elif kind == "struct":
            gen, where = self.p.generics(v.get("generics"))
            self.emit(f"struct {path}{gen}{where}", path)
            k = v.get("kind", {})
            fields = []
            if isinstance(k, dict) and "plain" in k:
                fields = k["plain"].get("fields", [])
            elif isinstance(k, dict) and "tuple" in k:
                fields = [f for f in k["tuple"] if f is not None]
            for i, f in enumerate(fields):
                fi = self.c.item(crate, f)
                if fi and fi.get("visibility") == "public":
                    self.p.crate = crate
                    self.emit(f"field {path}.{fi['name'] or i}: {self.p.ty(fi['inner']['struct_field'])}", path)
            self.impls(crate, path, v.get("impls", []))
        elif kind == "enum":
            gen, where = self.p.generics(v.get("generics"))
            self.emit(f"enum {path}{gen}{where}", path)
            for var in v.get("variants", []):
                vi = self.c.item(crate, var)
                if vi:
                    self.emit(f"variant {path}::{vi['name']}{self.variant(crate, vi['inner']['variant'])}", path)
            self.impls(crate, path, v.get("impls", []))
        elif kind == "union":
            self.emit(f"union {path}", path)
            self.impls(crate, path, v.get("impls", []))
        elif kind == "trait":
            gen, where = self.p.generics(v.get("generics"))
            sup = ": " + self.p.bounds(v["bounds"]) if v.get("bounds") else ""
            flags = ("unsafe " if v.get("is_unsafe") else "") + ("auto " if v.get("is_auto") else "")
            dyn = "" if v.get("is_dyn_compatible", True) else " (not dyn)"
            self.emit(f"trait {path}: {flags}trait {name}{gen}{sup}{where}{dyn}", path)
            for t in v.get("items", []):
                ti = self.c.item(crate, t)
                if ti:
                    self.assoc(crate, f"{path}::{ti['name']}", ti, trait=True)
        elif kind == "type_alias":
            gen, where = self.p.generics(v.get("generics"))
            self.emit(f"type {path}{gen} = {self.p.ty(v['type'])}{where}", path)
        elif kind == "constant":
            self.emit(f"const {path}: {self.p.ty(v['type'])}", path)
        elif kind == "static":
            mut = "mut " if v.get("is_mutable") else ""
            self.emit(f"static {path}: {mut}{self.p.ty(v['type'])}", path)
        elif kind in ("macro", "proc_macro"):
            self.emit(f"macro {path}", path)
        elif kind == "trait_alias":
            self.emit(f"trait_alias {path}", path)
        else:
            self.emit(f"{kind} {path}", path)

    def variant(self, crate: str, v: dict) -> str:
        k = v.get("kind")
        if k == "plain" or k is None:
            return ""
        self.p.crate = crate
        if "tuple" in k:
            tys = [self.p.ty(self.c.item(crate, f)["inner"]["struct_field"]) if f is not None else "_" for f in k["tuple"]]
            return "(" + ", ".join(tys) + ")"
        if "struct" in k:
            fs = []
            for f in k["struct"]["fields"]:
                fi = self.c.item(crate, f)
                fs.append(f"{fi['name']}: {self.p.ty(fi['inner']['struct_field'])}")
            return " { " + ", ".join(fs) + " }"
        return ""

    def assoc(self, crate: str, path: str, item: dict, trait: bool) -> None:
        self.p.crate = crate
        (kind, v), = item["inner"].items()
        name = path.rsplit("::", 1)[-1]
        if name.startswith("_"):
            return
        if kind == "function":
            req = (" (provided)" if v.get("has_body") else " (required)") if trait else ""
            self.emit(f"method {path}: {self.p.function(name, v)}{req}", path)
        elif kind == "assoc_type":
            bs = ": " + self.p.bounds(v["bounds"]) if v.get("bounds") else ""
            d = " = " + self.p.ty(v["type"]) if v.get("type") is not None else ""
            self.emit(f"assoc_type {path}{bs}{d}", path)
        elif kind == "assoc_const":
            self.emit(f"assoc_const {path}: {self.p.ty(v['type'])}", path)

    def impls(self, crate: str, path: str, impls: list) -> None:
        for i in impls:
            im = self.c.item(crate, i)
            if im is None:
                continue
            v = im["inner"]["impl"]
            if v.get("blanket_impl") is not None:
                continue
            self.p.crate = crate
            trait = v.get("trait")
            if trait is None:
                for m in v.get("items", []):
                    mi = self.c.item(crate, m)
                    if mi and mi.get("visibility") == "public":
                        self.assoc(crate, f"{path}::{mi['name']}", mi, trait=False)
                continue
            tname = self.p.path(trait)
            if v.get("is_synthetic") and tname.rsplit("::", 1)[-1].split("<")[0] not in AUTO_TRAITS:
                continue
            neg = "!" if v.get("is_negative") else ""
            gen, where = self.p.generics(v.get("generics"))
            self.emit(f"impl {path}: impl{gen} {neg}{tname} for {self.p.ty(v['for'])}{where}", path)


def snapshot(json_dir: pathlib.Path) -> str:
    docs = {}
    for krate in (FACADE, *MEMBERS):
        name = krate.replace("-", "_")
        f = json_dir / f"{name}.json"
        if f.exists():
            docs[name] = json.loads(f.read_text())
    crates = Crates(docs)
    w = Walker(crates)
    facade = "tessera_rl"
    w.module(facade, str(docs[facade]["root"]), facade)
    rustdoc = docs[facade].get("crate_version", "?")
    header = [
        "# The public API of the tessera-rl facade (ADR-0094 item 7, #642).",
        "# Generated by scripts/check_facade_api.py; do not edit. Review a diff, then `make facade-api-bless`.",
        f"# toolchain: {toolchain()}",
        f"# rustdoc JSON format_version: {FORMAT_VERSION}",
        f"# facade features: default + {FACADE_FEATURES}; crate version {rustdoc}",
        f"# INCLUDE_EXPERIMENTAL: {INCLUDE_EXPERIMENTAL}" + ("" if INCLUDE_EXPERIMENTAL else " (ADR-0094 Q1 open: host API excluded)"),
    ]
    return "\n".join(header + sorted(w.lines)) + "\n"


def toolchain() -> str:
    text = (ROOT / "rust-toolchain.toml").read_text()
    for line in text.splitlines():
        if line.strip().startswith("channel"):
            return line.split("=", 1)[1].strip().strip('"')
    return "?"


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("check", "bless", "print"):
        print(__doc__)
        return 2
    mode = argv[0]
    json_dir = None
    if "--json" in argv:
        json_dir = pathlib.Path(argv[argv.index("--json") + 1])
    else:
        json_dir = build(None)
    new = snapshot(json_dir)
    if mode == "print":
        sys.stdout.write(new)
        return 0
    if mode == "bless":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(new)
        print(f"wrote {SNAPSHOT.relative_to(ROOT)} ({new.count(chr(10))} lines)")
        return 0
    old = SNAPSHOT.read_text() if SNAPSHOT.exists() else ""
    if old == new:
        print(f"facade-api: {SNAPSHOT.relative_to(ROOT)} is current")
        return 0
    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "api/tessera-rl.api (committed)", "api/tessera-rl.api (built)"))
    print("facade-api: the public API changed: review the diff, then `make facade-api-bless`")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

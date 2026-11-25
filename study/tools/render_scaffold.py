#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Renders `cargo tessera new --kind env` templates offline, as
crates/cargo-tessera/src/render.rs does, with relative path dependencies,
so the study crates can be committed inside the repository (#639).

Usage: render_scaffold.py <repo-root> <dest> <crate-name> <agents> <turns> <rel-root>
"""
import pathlib
import re
import sys


def marker(line):
    s = line.strip()
    if s.startswith("{{") and s.endswith("}}") and len(s) > 4 and s[2] in "#^/":
        return s[2], s[3:-2]
    return None


def render(text, values, flags):
    out, stack = [], []
    for line in text.splitlines(keepends=True):
        m = marker(line)
        if m:
            kind, key = m
            if kind == "/":
                top = stack.pop()
                assert top[0] == key, key
            else:
                on = flags[key]
                stack.append((key, on if kind == "#" else not on))
            continue
        if all(k for _, k in stack):
            out.append(re.sub(r"\{\{([a-z_]+)\}\}", lambda g: values[g.group(1)], line))
    assert not stack
    return "".join(out)


def main():
    root, dest, name, agents, turns, rel = sys.argv[1:]
    root = pathlib.Path(root)
    ver = re.search(r'^version = "([^"]+)"', (root / "Cargo.toml").read_text(), re.M).group(1)
    camel = "".join(p[:1].upper() + p[1:] for p in re.split(r"[-_]", name) if p)
    dep = lambda k: "{ path = '%s/crates/%s' }" % (rel, k)
    values = {
        "crate_name": name, "py_package": name.replace("-", "_"),
        "env_id": f"{name}/{camel}-v1", "env_version": "1", "agents": agents,
        "turns": turns, "tool_version": ver, "dep_core": dep("tessera-rl-core"),
        "dep_macros": dep("tessera-rl-macros"), "dep_num": dep("tessera-rl-num"),
        "dep_engine_env": dep("tessera-rl-engine"),
    }
    flags = {"path_deps": True, "simultaneous": turns == "simultaneous",
             "sequential": turns == "sequential", "mixed": turns == "mixed"}
    tpl = root / "crates/cargo-tessera/src/templates/env"
    files = {"Cargo.toml": "Cargo.toml.tpl", "clippy.toml": "clippy.toml.tpl",
             "src/lib.rs": "lib.rs.tpl", "tests/check.rs": "check.rs.tpl",
             "benches/step.rs": "step.rs.tpl", "README.md": "README.md.tpl",
             ".gitignore": "../gitignore.tpl"}
    for d, t in files.items():
        p = pathlib.Path(dest) / d
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render((tpl / t).read_text(), values, flags))


main()

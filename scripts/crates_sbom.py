#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A CycloneDX 1.5 SBOM of the crates Tessera publishes to crates.io (#350).

    crates_sbom.py --metadata metadata.json --out tessera-rl-crates.cyclonedx.json
    crates_sbom.py --out ...            # runs `cargo metadata --locked --format-version 1`

The components are the workspace members that are published (`publish` is
not `[]`) and every package they reach through normal or build dependencies
in Cargo.lock's resolve (dev-dependencies are not shipped). The root
component is the `tessera-rl` facade. The output is deterministic: sorted,
no timestamp and no serial number, so the same lock gives the same bytes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_CRATE = "tessera-rl"


def _purl(pkg: dict) -> str:
    return f"pkg:cargo/{pkg['name']}@{pkg['version']}"


def _published(pkg: dict) -> bool:
    return pkg.get("publish") != []


def build(metadata: dict) -> dict:
    packages = {p["id"]: p for p in metadata["packages"]}
    members = [packages[i] for i in metadata["workspace_members"] if _published(packages[i])]
    roots = [p for p in members if p["name"] == ROOT_CRATE]
    if len(roots) != 1:
        raise ValueError(f"expected one published `{ROOT_CRATE}` workspace member, found {len(roots)}")
    nodes = {n["id"]: n for n in (metadata.get("resolve") or {}).get("nodes", [])}
    edges: dict[str, set[str]] = {}
    seen: set[str] = set()
    todo = [p["id"] for p in members]
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out = edges.setdefault(pid, set())
        for dep in nodes.get(pid, {}).get("deps", []):
            kinds = {k.get("kind") for k in dep.get("dep_kinds", [])}
            if kinds & {None, "build"}:
                out.add(dep["pkg"])
                todo.append(dep["pkg"])
    root = roots[0]

    def component(pkg: dict) -> dict:
        c = {"type": "library", "bom-ref": _purl(pkg), "name": pkg["name"], "version": pkg["version"], "purl": _purl(pkg)}
        if pkg.get("license"):
            c["licenses"] = [{"expression": pkg["license"]}]
        return c

    comps = sorted((packages[i] for i in seen if i != root["id"]), key=lambda p: (p["name"], p["version"]))
    deps = sorted(
        ({"ref": _purl(packages[i]), "dependsOn": sorted(_purl(packages[d]) for d in edges.get(i, ()))} for i in seen),
        key=lambda d: d["ref"],
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {"component": component(root)},
        "components": [component(p) for p in comps],
        "dependencies": deps,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, help="cargo metadata JSON (default: run cargo metadata --locked)")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.metadata:
        text = args.metadata.read_text(encoding="utf-8")
    else:
        text = subprocess.run(
            ["cargo", "metadata", "--locked", "--format-version", "1"], check=True, capture_output=True, text=True
        ).stdout
    try:
        bom = build(json.loads(text))
    except (ValueError, KeyError) as e:
        print(f"crates_sbom: {e}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(bom, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"crates_sbom: {args.out.name}: {bom['metadata']['component']['version']}, {len(bom['components'])} components")
    return 0


if __name__ == "__main__":
    sys.exit(main())

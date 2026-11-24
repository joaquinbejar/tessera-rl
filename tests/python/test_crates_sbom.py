# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/crates_sbom.py: the crates.io SBOM from cargo metadata (#350)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "crates_sbom.py"


def _pkg(name, version="1.0.0", publish=None, license="MIT"):
    return {"id": f"{name} {version}", "name": name, "version": version, "publish": publish, "license": license}


def _dep(pkg, *kinds):
    return {"pkg": pkg, "dep_kinds": [{"kind": k} for k in kinds]}


def _metadata():
    pkgs = [
        _pkg("tessera-rl", "0.4.0"),
        _pkg("tessera-rl-core", "0.4.0"),
        _pkg("tessera-rl-py", "0.4.0", publish=[]),
        _pkg("serde"),
        _pkg("cc", "1.2.0"),
        _pkg("criterion", "0.5.0"),
        _pkg("pyo3-only", "0.1.0"),
    ]
    nodes = [
        {"id": "tessera-rl 0.4.0", "deps": [_dep("tessera-rl-core 0.4.0", None), _dep("criterion 0.5.0", "dev")]},
        {"id": "tessera-rl-core 0.4.0", "deps": [_dep("serde 1.0.0", None), _dep("cc 1.2.0", "build")]},
        {"id": "tessera-rl-py 0.4.0", "deps": [_dep("pyo3-only 0.1.0", None)]},
        {"id": "serde 1.0.0", "deps": []},
        {"id": "cc 1.2.0", "deps": []},
        {"id": "criterion 0.5.0", "deps": []},
        {"id": "pyo3-only 0.1.0", "deps": []},
    ]
    return {
        "packages": pkgs,
        "workspace_members": ["tessera-rl 0.4.0", "tessera-rl-core 0.4.0", "tessera-rl-py 0.4.0"],
        "resolve": {"nodes": nodes},
    }


def _run(tmp_path, metadata):
    src = tmp_path / "metadata.json"
    src.write_text(json.dumps(metadata))
    out = tmp_path / "bom.json"
    p = subprocess.run([sys.executable, "-I", str(TOOL), "--metadata", str(src), "--out", str(out)], capture_output=True, text=True)
    return p, out


def test_crates_sbom_lists_published_crates_and_shipped_dependencies(tmp_path):
    p, out = _run(tmp_path, _metadata())
    assert p.returncode == 0, p.stderr
    bom = json.loads(out.read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.5"
    assert bom["metadata"]["component"]["name"] == "tessera-rl"
    assert bom["metadata"]["component"]["version"] == "0.4.0"
    names = [c["name"] for c in bom["components"]]
    # Normal and build dependencies; not dev-dependencies, not the unpublished
    # extension's own tree.
    assert names == ["cc", "serde", "tessera-rl-core"]
    assert bom["components"][1]["purl"] == "pkg:cargo/serde@1.0.0"
    assert bom["components"][1]["licenses"] == [{"expression": "MIT"}]
    deps = {d["ref"]: d["dependsOn"] for d in bom["dependencies"]}
    assert deps["pkg:cargo/tessera-rl@0.4.0"] == ["pkg:cargo/tessera-rl-core@0.4.0"]
    assert deps["pkg:cargo/tessera-rl-core@0.4.0"] == ["pkg:cargo/cc@1.2.0", "pkg:cargo/serde@1.0.0"]


def test_crates_sbom_is_deterministic(tmp_path):
    a, out_a = _run(tmp_path, _metadata())
    first = out_a.read_bytes()
    m = _metadata()
    m["packages"].reverse()
    m["resolve"]["nodes"].reverse()
    b, out_b = _run(tmp_path, m)
    assert a.returncode == b.returncode == 0
    assert out_b.read_bytes() == first
    assert "timestamp" not in first.decode() and "serialNumber" not in first.decode()


def test_crates_sbom_needs_the_facade(tmp_path):
    m = _metadata()
    m["workspace_members"].remove("tessera-rl 0.4.0")
    p, _ = _run(tmp_path, m)
    assert p.returncode == 1 and "expected one published `tessera-rl`" in p.stderr


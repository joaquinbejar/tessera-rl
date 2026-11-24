# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/sbom_check.py: the wheel's CycloneDX SBOM (SPEC 20.1, #350)."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "sbom_check.py"
PYPROJECT = (ROOT / "pyproject.toml").read_text()
JOB = (ROOT / "scripts" / "wheel_job.sh").read_text()
WHEEL = "tessera_rl-0.3.0-cp311-abi3-manylinux_2_28_x86_64.whl"


def _bom(version="0.3.0", components=("tessera-rl-core", "numpy-free"), fmt="CycloneDX"):
    return {
        "bomFormat": fmt,
        "specVersion": "1.5",
        "metadata": {"component": {"name": "tessera-rl-py", "version": version}},
        "components": [{"name": c, "version": "1"} for c in components],
    }


def _wheel(tmp_path, sboms=None, name=WHEEL):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tessera_rl/__init__.py", "")
        for fname, bom in (sboms if sboms is not None else {"tessera-rl-py.cyclonedx.json": _bom()}).items():
            z.writestr(f"tessera_rl-0.3.0.dist-info/sboms/{fname}", json.dumps(bom) if isinstance(bom, dict) else bom)
    return path


def _run(tmp_path, wheel, crates=("tessera-rl-py", "tessera-rl-core")):
    listing = tmp_path / "crates.txt"
    listing.write_text("\n".join(crates) + "\n")
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), "--wheel", str(wheel), "--crates", str(listing)],
        capture_output=True,
        text=True,
    )


def test_sbom_check_accepts_one_matching_sbom(tmp_path):
    p = _run(tmp_path, _wheel(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "one CycloneDX SBOM, 2 crates listed" in p.stdout


@pytest.mark.parametrize(
    "sboms, message",
    [
        ({}, "expected one CycloneDX SBOM"),
        ({"a.cyclonedx.json": _bom(), "b.cyclonedx.json": _bom()}, "expected one CycloneDX SBOM"),
        ({"a.cyclonedx.json": "{not json"}, "unreadable"),
        ({"a.cyclonedx.json": _bom(fmt="SPDX")}, "bomFormat 'SPDX'"),
        ({"a.cyclonedx.json": _bom(version="0.2.0")}, "root component version '0.2.0', expected '0.3.0'"),
        ({"a.cyclonedx.json": _bom(components=())}, "crates missing from the SBOM: ['tessera-rl-core']"),
    ],
)
def test_sbom_check_refuses_with_the_reason(tmp_path, sboms, message):
    p = _run(tmp_path, _wheel(tmp_path, sboms))
    assert p.returncode == 1 and message in p.stderr, p.stderr


def test_sbom_check_ignores_sboms_outside_dist_info(tmp_path):
    path = tmp_path / WHEEL
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tessera_rl/sboms/x.cyclonedx.json", json.dumps(_bom()))
    p = _run(tmp_path, path)
    assert p.returncode == 1 and "found []" in p.stderr, p.stderr


def test_sbom_check_refuses_an_empty_crate_list(tmp_path):
    p = _run(tmp_path, _wheel(tmp_path), crates=())
    assert p.returncode == 2 and "empty" in p.stderr


def test_sbom_enabled_in_pyproject_and_checked_by_the_wheel_job():
    assert "[tool.maturin.sbom]\nrust = true\n" in PYPROJECT
    assert 'tools/sbom_check.py --wheel "$wheel" --crates "$crates"' in JOB
    assert 'sbom="$sbom")' in JOB
    assert '[ "$sbom" -eq 0 ]' in JOB

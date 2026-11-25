# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/sbom_check.py: the wheel's CycloneDX SBOM (SPEC 20.1, #350)."""

import base64
import hashlib
import json
import subprocess
import uuid
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


def _record_line(member, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{member},sha256={digest},{len(data)}\n"


def _wheel(tmp_path, sboms=None, name=WHEEL, record=None):
    """A wheel with the SBOMs given and a RECORD that matches them, or ``record`` (a callable of the matching text)."""
    path = tmp_path / name
    lines = ""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tessera_rl/__init__.py", "")
        for fname, bom in (sboms if sboms is not None else {"tessera-rl-py.cyclonedx.json": _bom()}).items():
            member, data = f"tessera_rl-0.3.0.dist-info/sboms/{fname}", (json.dumps(bom) if isinstance(bom, dict) else bom).encode()
            z.writestr(member, data)
            lines += _record_line(member, data)
        lines += "tessera_rl-0.3.0.dist-info/RECORD,,\n"
        if record is not False:
            z.writestr("tessera_rl-0.3.0.dist-info/RECORD", record(lines) if record else lines)
    return path


def _run(tmp_path, wheel, crates=("tessera-rl-py", "tessera-rl-core"), extra=()):
    listing = tmp_path / "crates.txt"
    listing.write_text("\n".join(crates) + "\n")
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), "--wheel", str(wheel), "--crates", str(listing), *extra],
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
    assert 'tools/sbom_check.py --wheel "$wheel" --crates "$crates" \\\n        --serial-name "$main_name" --epoch "$sbom_epoch"; }' in JOB
    assert 'sbom="$sbom")' in JOB
    assert '[ "$sbom" -eq 0 ]' in JOB


# #654: the serial, the timestamp and RECORD's line.
NAME, EPOCH = "tessera-rl-wheel:" + "ab" * 20 + ":x86_64-unknown-linux-gnu:wheel:onnx", 1700000000
STAMP = "2023-11-14T22:13:20Z"
SERIAL = f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, NAME)}"


def _reproduced(**over):
    bom = _bom()
    bom["serialNumber"] = over.get("serial", SERIAL)
    bom["metadata"]["timestamp"] = over.get("timestamp", STAMP)
    return {"tessera-rl-py.cyclonedx.json": bom}


def test_sbom_check_accepts_the_reproduced_serial_and_timestamp(tmp_path):
    p = _run(tmp_path, _wheel(tmp_path, _reproduced()), extra=("--serial-name", NAME, "--epoch", str(EPOCH)))
    assert p.returncode == 0, p.stderr
    assert "RECORD matches, serialNumber and timestamp as reproduced" in p.stdout


@pytest.mark.parametrize(
    "over, extra, message",
    [
        ({"serial": "urn:uuid:" + str(uuid.uuid4())}, (), "serialNumber"),
        ({}, ("--serial-name", NAME + "x", "--epoch", str(EPOCH)), "serialNumber"),
        ({"timestamp": "2023-11-14T22:13:21Z"}, (), "metadata.timestamp '2023-11-14T22:13:21Z', expected '2023-11-14T22:13:20Z'"),
        ({}, ("--serial-name", NAME, "--epoch", str(EPOCH + 1)), "metadata.timestamp"),
    ],
    ids=["serial", "serial-name", "timestamp", "epoch"],
)
def test_sbom_check_refuses_a_wrong_serial_or_timestamp(tmp_path, over, extra, message):
    p = _run(tmp_path, _wheel(tmp_path, _reproduced(**over)), extra=extra or ("--serial-name", NAME, "--epoch", str(EPOCH)))
    assert p.returncode == 1 and message in p.stderr, p.stderr


def test_sbom_check_without_the_names_does_not_check_the_serial(tmp_path):
    p = _run(tmp_path, _wheel(tmp_path, _reproduced(serial="urn:uuid:" + str(uuid.uuid4()))))
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize(
    "record, message",
    [
        (lambda t: t.replace("sha256=", "sha256=A", 1), "does not match the SBOM's bytes"),
        (lambda t: t.replace(t.splitlines()[0], t.splitlines()[0].rsplit(",", 1)[0] + ",1", 1), "does not match the SBOM's bytes"),
        (lambda t: "\n".join(t.splitlines()[1:]) + "\n", "0 lines for"),
        (lambda t: t.splitlines()[0] + "\n" + t, "2 lines for"),
        (False, "no tessera_rl-0.3.0.dist-info/RECORD"),
    ],
    ids=["stale-hash", "stale-size", "no-line", "two-lines", "no-record"],
)
def test_sbom_check_refuses_a_record_line_that_does_not_match(tmp_path, record, message):
    p = _run(tmp_path, _wheel(tmp_path, record=record))
    assert p.returncode == 1 and message in p.stderr, p.stderr


@pytest.mark.parametrize(
    "extra",
    [("--serial-name", NAME), ("--epoch", "1"), ("--serial-name", "", "--epoch", "1"), ("--serial-name", NAME, "--epoch", "1.5")],
    ids=["name-only", "epoch-only", "empty-name", "fraction"],
)
def test_sbom_check_refuses_bad_reproduction_arguments(tmp_path, extra):
    p = _run(tmp_path, _wheel(tmp_path), extra=extra)
    assert p.returncode == 2, p.stderr


def test_sbom_check_accepts_a_wheel_the_reproducer_processed(tmp_path):
    tool = ROOT / "tools" / "sbom_reproduce.py"
    bom = _bom()
    bom["serialNumber"], bom["metadata"]["timestamp"] = "urn:uuid:" + str(uuid.uuid4()), "2026-11-25T10:11:12.345678Z"
    wheel = _wheel(tmp_path, {"tessera-rl-py.cyclonedx.json": bom})
    r = subprocess.run([sys.executable, "-I", str(tool), "--wheel", str(wheel), "--name", NAME, "--epoch", str(EPOCH)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    p = _run(tmp_path, wheel, extra=("--serial-name", NAME, "--epoch", str(EPOCH)))
    assert p.returncode == 0, p.stderr

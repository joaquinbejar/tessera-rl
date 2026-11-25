# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/sbom_reproduce.py: a wheel's SBOM made reproducible in place (#654)."""

import base64
import hashlib
import json
import subprocess
import sys
import uuid
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "sbom_reproduce.py"
WHEEL = "tessera_rl-0.3.0-cp311-abi3-manylinux_2_28_x86_64.whl"
DIST = "tessera_rl-0.3.0.dist-info"
SBOM = f"{DIST}/sboms/tessera-rl-py.cyclonedx.json"
NAME = "tessera-rl-wheel:" + "ab" * 20 + ":x86_64-unknown-linux-gnu:wheel:onnx"
EPOCH = 1700000000
STAMP = "2023-11-14T22:13:20Z"
NATIVE = bytes(range(256)) * 64

# maturin's layout: two-space indent, the serial near the top, a fractional timestamp.
SBOM_TEXT = """{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:%s",
  "version": 1,
  "metadata": {
    "timestamp": "%s",
    "component": {"name": "tessera-rl-py", "version": "0.3.0"}
  },
  "components": [{"name": "tessera-rl-core", "version": "0.3.0"}]
}
"""


def _hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _sbom(serial=None, stamp="2026-11-25T10:11:12.345678901Z") -> bytes:
    return (SBOM_TEXT % (serial or uuid.uuid4(), stamp)).encode()


def _wheel(tmp_path: Path, sbom=None, name=WHEEL, record=True, extra=()) -> Path:
    """A maturin-like wheel: a stored and a deflated member, an executable bit, an old date."""
    sbom = _sbom() if sbom is None else sbom
    members = [
        ("tessera_rl/__init__.py", b"from ._native import *\n", zipfile.ZIP_DEFLATED, 0o644, (2026, 11, 25, 10, 0, 0)),
        ("tessera_rl/_native.abi3.so", NATIVE, zipfile.ZIP_DEFLATED, 0o755, (1980, 1, 1, 0, 0, 0)),
        (f"{DIST}/WHEEL", b"Wheel-Version: 1.0\n", zipfile.ZIP_STORED, 0o644, (1980, 1, 1, 0, 0, 0)),
    ]
    if sbom is not False:
        members.append((SBOM, sbom, zipfile.ZIP_DEFLATED, 0o644, (1980, 1, 1, 0, 0, 0)))
    members += list(extra)
    lines = "".join(f"{n},{_hash(d)},{len(d)}\n" for n, d, *_ in members) + f"{DIST}/RECORD,,\n"
    if record:
        members.append((f"{DIST}/RECORD", lines.encode(), zipfile.ZIP_DEFLATED, 0o644, (1980, 1, 1, 0, 0, 0)))
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for n, d, method, mode, when in members:
            info = zipfile.ZipInfo(n, date_time=when)
            info.external_attr = (0o100000 | mode) << 16
            info.create_system = 3
            info.compress_type = method
            z.writestr(info, d, compresslevel=1)
    return path


def _run(wheel: Path, name=NAME, epoch=EPOCH, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(TOOL), "--wheel", str(wheel), "--name", name, "--epoch", str(epoch), *extra],
                          capture_output=True, text=True, check=False)


def _members(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as z:
        return {i.filename: z.read(i) for i in z.infolist()}


def test_sbom_reproduce_sets_the_serial_and_timestamp_and_logs_zlib(tmp_path):
    wheel = _wheel(tmp_path)
    p = _run(wheel)
    assert p.returncode == 0, p.stderr
    bom = json.loads(_members(wheel)[SBOM])
    assert bom["serialNumber"] == f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, NAME)}"
    assert bom["metadata"]["timestamp"] == STAMP
    assert f"zlib {zlib.ZLIB_RUNTIME_VERSION} level 9" in p.stdout and STAMP in p.stdout


def test_sbom_reproduce_changes_only_the_two_values_in_the_sbom_text(tmp_path):
    serial = uuid.uuid4()
    wheel = _wheel(tmp_path, _sbom(serial))
    assert _run(wheel).returncode == 0
    want = SBOM_TEXT % (uuid.uuid5(uuid.NAMESPACE_URL, NAME), STAMP)
    assert _members(wheel)[SBOM].decode() == want


def test_sbom_reproduce_rewrites_the_record_line(tmp_path):
    wheel = _wheel(tmp_path)
    assert _run(wheel).returncode == 0
    m = _members(wheel)
    lines = m[f"{DIST}/RECORD"].decode().splitlines()
    assert f"{SBOM},{_hash(m[SBOM])},{len(m[SBOM])}" in lines
    assert lines[-1] == f"{DIST}/RECORD,,"
    for n, d in m.items():
        if n != f"{DIST}/RECORD":
            assert f"{n},{_hash(d)},{len(d)}" in lines, n


def test_sbom_reproduce_keeps_every_other_member_byte_identical(tmp_path):
    # Recompressed, but the uncompressed bytes are maturin's, _native included.
    wheel = _wheel(tmp_path)
    before = _members(wheel)
    assert _run(wheel).returncode == 0
    after = _members(wheel)
    assert list(after) == list(before)
    for n in before:
        if n not in (SBOM, f"{DIST}/RECORD"):
            assert after[n] == before[n], n
    assert after["tessera_rl/_native.abi3.so"] == NATIVE


def test_sbom_reproduce_keeps_the_zipinfo_fields(tmp_path):
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel) as z:
        before = [(i.filename, i.date_time, i.external_attr, i.create_system, i.compress_type) for i in z.infolist()]
    assert _run(wheel).returncode == 0
    with zipfile.ZipFile(wheel) as z:
        after = [(i.filename, i.date_time, i.external_attr, i.create_system, i.compress_type) for i in z.infolist()]
        assert z.testzip() is None
    assert after == before


def test_sbom_reproduce_twice_gives_the_same_bytes(tmp_path):
    wheel = _wheel(tmp_path)
    assert _run(wheel).returncode == 0
    once = wheel.read_bytes()
    assert _run(wheel).returncode == 0
    assert wheel.read_bytes() == once


def test_sbom_reproduce_two_builds_of_one_commit_give_the_same_bytes(tmp_path):
    # The issue's case: two builds differ only in the serial and the time.
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    wa, wb = _wheel(a), _wheel(b, _sbom(stamp="2026-11-25T11:00:00.1Z"))
    assert wa.read_bytes() != wb.read_bytes()
    assert _run(wa).returncode == 0 and _run(wb).returncode == 0
    assert wa.read_bytes() == wb.read_bytes()


def test_sbom_reproduce_two_names_give_two_serials(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    wa, wb = _wheel(a), _wheel(b)
    assert _run(wa).returncode == 0 and _run(wb, NAME.replace(":wheel:", ":release:")).returncode == 0
    assert json.loads(_members(wa)[SBOM])["serialNumber"] != json.loads(_members(wb)[SBOM])["serialNumber"]


def test_sbom_reproduce_leaves_no_temporary_file(tmp_path):
    wheel = _wheel(tmp_path)
    assert _run(wheel).returncode == 0
    assert [x.name for x in tmp_path.iterdir()] == [WHEEL]


@pytest.mark.parametrize(
    "sbom, message",
    [
        (False, "expected one CycloneDX SBOM under .dist-info/sboms/, found []"),
        (_sbom().replace(b'"serialNumber"', b'"serial"'), "serialNumber matched 0 times"),
        (_sbom().replace(b'"version": 1,', b'"version": 1, "serialNumber": "urn:uuid:%s",' % str(uuid.uuid4()).encode()),
         "serialNumber matched 2 times"),
        (_sbom().replace(b'"timestamp"', b'"time"'), "timestamp matched 0 times"),
        (_sbom().replace(b'"version": 1,', b'"version": 1, "timestamp": "x",'), "timestamp matched 2 times"),
        (_sbom(serial="not-a-uuid"), "serialNumber matched 0 times"),
    ],
    ids=["no-sbom", "no-serial", "two-serials", "no-timestamp", "two-timestamps", "malformed-serial"],
)
def test_sbom_reproduce_refuses_and_leaves_the_wheel_unchanged(tmp_path, sbom, message):
    wheel = _wheel(tmp_path, sbom)
    before = wheel.read_bytes()
    p = _run(wheel)
    assert p.returncode == 1 and message in p.stderr, p.stderr
    assert wheel.read_bytes() == before
    assert [x.name for x in tmp_path.iterdir()] == [WHEEL]


def test_sbom_reproduce_refuses_two_sboms(tmp_path):
    extra = [(f"{DIST}/sboms/other.cyclonedx.json", _sbom(), zipfile.ZIP_DEFLATED, 0o644, (1980, 1, 1, 0, 0, 0))]
    p = _run(_wheel(tmp_path, extra=extra))
    assert p.returncode == 1 and "expected one CycloneDX SBOM" in p.stderr


def test_sbom_reproduce_refuses_a_wheel_without_record_or_its_line(tmp_path):
    p = _run(_wheel(tmp_path, record=False))
    assert p.returncode == 1 and f"no {DIST}/RECORD" in p.stderr
    other = tmp_path / "other"
    other.mkdir()
    wheel = _wheel(other)
    with zipfile.ZipFile(wheel) as z:
        members = {i.filename: z.read(i) for i in z.infolist()}
    record = "".join(x + "\n" for x in members[f"{DIST}/RECORD"].decode().splitlines() if not x.startswith(SBOM + ","))
    with zipfile.ZipFile(wheel, "w") as z:
        for n, d in members.items():
            z.writestr(n, record.encode() if n.endswith("/RECORD") else d)
    p = _run(wheel)
    assert p.returncode == 1 and "RECORD has 0 lines for" in p.stderr


def test_sbom_reproduce_optional_leaves_a_wheel_without_an_sbom_unchanged(tmp_path):
    wheel = _wheel(tmp_path, False)
    before = wheel.read_bytes()
    p = _run(wheel, NAME, EPOCH, "--optional")
    assert p.returncode == 0 and "no SBOM; left unchanged" in p.stdout, p.stderr
    assert wheel.read_bytes() == before


def test_sbom_reproduce_optional_still_processes_a_wheel_with_one(tmp_path):
    wheel = _wheel(tmp_path)
    p = _run(wheel, NAME, EPOCH, "--optional")
    assert p.returncode == 0, p.stderr
    assert json.loads(_members(wheel)[SBOM])["metadata"]["timestamp"] == STAMP


@pytest.mark.parametrize("name, epoch", [("", EPOCH), (NAME, "-1"), (NAME, "1.5"), (NAME, "x"), (NAME, "")],
                         ids=["empty-name", "negative", "fraction", "word", "empty-epoch"])
def test_sbom_reproduce_refuses_bad_arguments(tmp_path, name, epoch):
    wheel = _wheel(tmp_path)
    before = wheel.read_bytes()
    p = _run(wheel, name, epoch)
    assert p.returncode == 2, p.stderr
    assert wheel.read_bytes() == before


def test_sbom_reproduce_epoch_zero_is_the_unix_epoch(tmp_path):
    wheel = _wheel(tmp_path)
    assert _run(wheel, NAME, 0).returncode == 0
    assert json.loads(_members(wheel)[SBOM])["metadata"]["timestamp"] == "1970-01-01T00:00:00Z"

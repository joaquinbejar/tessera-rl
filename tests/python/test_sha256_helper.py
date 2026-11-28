# SPDX-License-Identifier: MIT OR Apache-2.0
"""The sha256 helper of install_rustup.sh and wheel_job.sh hashes any path (#795).

GNU sha256sum prefixes its line with a backslash when the file name has one
(the Windows leg's D:\\a\\... paths), so the helper reads the file from stdin.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a backslash in a file name")

ROOT = Path(__file__).resolve().parents[2]


def _helper(script: str) -> str:
    match = re.search(r"^sha256\(\) \{\n.*?\n\}\n", (ROOT / "scripts" / script).read_text(), re.M | re.S)
    assert match is not None
    return match.group(0)


@pytest.mark.parametrize("script", ["install_rustup.sh", "wheel_job.sh"])
@pytest.mark.parametrize("name", ["plain.bin", "back\\slash.bin", "D:\\a\\tessera-rl\\rustup-init.exe"])
def test_sha256_helper_any_file_name_prints_the_bare_digest(tmp_path: Path, script: str, name: str) -> None:
    f = tmp_path / name
    f.write_bytes(b"tessera\n")
    out = subprocess.run(
        ["bash", "-c", _helper(script) + 'sha256 "$1"', "_", str(f)], capture_output=True, text=True, check=True
    ).stdout
    assert out == hashlib.sha256(b"tessera\n").hexdigest() + "\n"


@pytest.mark.parametrize("script", ["install_rustup.sh", "wheel_job.sh"])
def test_sha256_helper_reads_the_file_from_stdin(script: str) -> None:
    assert '< "$1"' in _helper(script)


@pytest.mark.parametrize("script", ["install_rustup.sh", "wheel_job.sh"])
def test_sha256_helper_gnu_escaped_name_still_prints_the_bare_digest(tmp_path: Path, script: str) -> None:
    # A stand-in for GNU sha256sum on a backslash path: it escapes the line when
    # given a name, and prints "<digest>  -" for stdin, as GNU does.
    digest = hashlib.sha256(b"tessera\n").hexdigest()
    fake = tmp_path / "bin" / "sha256sum"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/bash\nif [ $# -gt 0 ]; then echo "\\\\{digest}  $1"; else cat >/dev/null; echo "{digest}  -"; fi\n')
    fake.chmod(0o755)
    f = tmp_path / "D:\\a\\rustup-init.exe"
    f.write_bytes(b"tessera\n")
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"}
    out = subprocess.run(
        ["bash", "-c", _helper(script) + 'sha256 "$1"', "_", str(f)], capture_output=True, text=True, check=True, env=env
    ).stdout
    assert out == digest + "\n"

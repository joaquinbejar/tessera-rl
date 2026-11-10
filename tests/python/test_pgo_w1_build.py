# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/pgo_w1_build.sh against fake tools: no network, no build."""

import hashlib
import io
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pgo_w1_build.sh"


def tool(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def uv_tarball(log: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in {
            "uvx": f'echo "uvx $*" >> {log}\nfor a in "$@"; do [ "$prev" = --out ] && touch "$a/tessera_rl-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"; prev="$a"; done\n',
            "uv": f'echo "uv $*" >> {log}\nif [ "$1" = venv ]; then mkdir -p "$4/bin"; printf "#!/bin/sh\\necho py \\"\\$*\\" >> {log}\\n" > "$4/bin/python"; chmod +x "$4/bin/python"; fi\n',
        }.items():
            data = ("#!/bin/sh\n" + body).encode()
            info = tarfile.TarInfo(f"uv-x86_64-unknown-linux-gnu/{name}")
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def setup(tmp_path: Path, rustup_ok: bool = True):
    bin_dir, payload, log = tmp_path / "fake", tmp_path / "payload", tmp_path / "log"
    bin_dir.mkdir()
    payload.mkdir()
    (payload / "rustup-init").write_text("#!/bin/sh\necho rustup-init >> " + str(log) + "\n")
    (payload / "uv.tar.gz").write_bytes(uv_tarball(log))
    tool(bin_dir, "curl", f'out=""; url=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; https*) url="$1";; esac; shift; done\n'
         f'case "$url" in *rustup-init) cp {payload}/rustup-init "$out";; *uv-*) cp {payload}/uv.tar.gz "$out";; esac\n')
    tool(bin_dir, "rustup", f'echo "rustup $*" >> {log}\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    rustup_sha = digest(payload / "rustup-init") if rustup_ok else "0" * 64
    return bin_dir, log, rustup_sha, digest(payload / "uv.tar.gz")


def run(tmp_path, rustup_ok=True):
    bin_dir, log, rsha, usha = setup(tmp_path, rustup_ok)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    out = tmp_path / "run"
    p = subprocess.run(["bash", str(SCRIPT), rsha, usha, str(out)], env=env, capture_output=True, text=True, check=False)
    return p, out, log.read_text() if log.exists() else ""


def test_pgo_w1_build_uses_the_wheel_profile_and_records_the_wheel(tmp_path):
    p, out, log = run(tmp_path)
    assert p.returncode == 0, p.stderr
    assert "--profile wheel" in log and "--compatibility manylinux_2_28" in log and "--locked" in log and "--release" not in log
    whl = out / "dist" / "tessera_rl-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    assert (out / "dist" / "w1.sha256").read_text().strip() == hashlib.sha256(whl.read_bytes()).hexdigest()


def test_pgo_w1_build_makes_the_venv_with_the_wheel_and_numpy_only(tmp_path):
    p, out, log = run(tmp_path)
    assert p.returncode == 0, p.stderr
    install = [x for x in log.splitlines() if x.startswith("uv pip install")]
    assert len(install) == 1 and install[0].endswith("numpy==2.4.6") and ".whl" in install[0]
    assert "py -I -c import tessera_rl._native" in log


def test_pgo_w1_build_refuses_a_tool_with_the_wrong_hash(tmp_path):
    p, out, log = run(tmp_path, rustup_ok=False)
    assert p.returncode != 0 and "sha256 mismatch" in p.stderr and "rustup-init" not in log


@pytest.mark.parametrize("name", ["w1", "dist", "venv"])
def test_pgo_w1_build_writes_only_under_the_run_dir(tmp_path, name):
    p, out, _ = run(tmp_path)
    assert p.returncode == 0 and (out / name).exists()
    assert {x.name for x in tmp_path.iterdir()} == {"fake", "payload", "log", "run"}

# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/wheel_job.sh against fake tools: which wheels it builds, with which
profile, and the W0 comparison a `wheel` build carries (pgo-wheels-design
section 7 item 2). No network, no build."""

import hashlib
import io
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash script with sh fakes")

ROOT = Path(__file__).resolve().parents[2]
AMBIENT = ("RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", "CARGO_BUILD_RUSTFLAGS",
           "CARGO_BUILD_RUSTDOCFLAGS", "CARGO_BUILD_TARGET", "CARGO_BUILD_TARGET_DIR", "RUSTC", "RUSTC_WRAPPER",
           "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC", "CARGO_BUILD_RUSTC_WRAPPER", "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER")


def _tool(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


PY = """#!/bin/sh
echo "py $0 $*" >> "$FAKE_LOG"
prev=""
for a in "$@"; do [ "$prev" = --out ] && echo "{}" > "$a"; prev="$a"; done
case " $* " in *" compare "*) exit "${FAKE_COMPARE_EXIT:-0}" ;; esac
exit 0
"""


def _uv_tarball(log: Path, payload: Path) -> bytes:
    bodies = {
        "uvx": f'echo "uvx $*" >> {log}\nprev=""; for a in "$@"; do if [ "$prev" = --out ]; then mkdir -p "$a"; '
               f'echo "$*" > "$a/tessera_rl-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"; fi; prev="$a"; done\n',
        "uv": f'echo "uv $*" >> {log}\nif [ "$1" = venv ]; then mkdir -p "$4/bin"; cp {payload}/py "$4/bin/python"; fi\n',
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in bodies.items():
            data = ("#!/bin/sh\n" + body).encode()
            info = tarfile.TarInfo(f"uv-x86_64-unknown-linux-gnu/{name}")
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def run(tmp_path: Path, *profile: str, compare_exit: int = 0):
    """The real wheel_job.sh and wheel_num_tests.sh in a scratch root, with fake tools first on PATH."""
    root, fake, payload, log = tmp_path / "root", tmp_path / "fake", tmp_path / "payload", tmp_path / "log"
    for d in (root / "scripts", fake, payload):
        d.mkdir(parents=True)
    for name in ("wheel_job.sh", "wheel_num_tests.sh"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (payload / "rustup-init").write_text(f"#!/bin/sh\necho rustup-init >> {log}\n")
    _tool(payload / "py", PY.split("\n", 1)[1])
    (payload / "uv.tar.gz").write_bytes(_uv_tarball(log, payload))
    _tool(fake / "curl", 'out=""; url=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; https*) url="$1";; esac; shift; done\n'
          f'case "$url" in *rustup-init) cp {payload}/rustup-init "$out";; *uv-*) cp {payload}/uv.tar.gz "$out";; esac\n')
    _tool(fake / "rustup", f'echo "rustup $*" >> {log}\n')
    _tool(fake / "cargo", f'echo "cargo $*" >> {log}\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT and not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))}
    env.update(PATH=f"{fake}:{os.environ['PATH']}", WHEEL_RUN=str(tmp_path / "wr"), FAKE_COMPARE_EXIT=str(compare_exit), FAKE_LOG=str(log))
    p = subprocess.run(["bash", "scripts/wheel_job.sh", "x86_64-unknown-linux-gnu", "manylinux_2_28",
                        digest(payload / "rustup-init"), digest(payload / "uv.tar.gz"), *profile],
                       env=env, cwd=root, capture_output=True, text=True, check=False)
    return p, (log.read_text() if log.exists() else "").splitlines(), root, tmp_path / "wr"


def _builds(lines):
    return [x for x in lines if x.startswith("uvx ") and " build " in x]


def test_wheel_job_wheel_profile_builds_w0_then_w1_and_compares_them(tmp_path):
    p, lines, root, wr = run(tmp_path, "wheel")
    assert p.returncode == 0, p.stderr
    builds = _builds(lines)
    assert len(builds) == 2
    assert "--release --locked --target x86_64-unknown-linux-gnu" in builds[0] and f"--out {wr}/w0-dist" in builds[0]
    assert "--profile wheel --locked --target x86_64-unknown-linux-gnu --out dist" in builds[1]
    assert all("--compatibility manylinux_2_28" in b for b in builds) and "--release" not in builds[1]
    record_w0 = [x for x in lines if "/w0-venv/bin/python" in x and " record " in x]
    assert len(record_w0) == 1 and f"--wheel {wr}/w0-dist/" in record_w0[0] and f"--out {root}/determinism-w0.json" in record_w0[0]
    compare = [x for x in lines if " compare " in x]
    assert len(compare) == 1
    assert compare[0].endswith(f"compare {root}/determinism-w0.json {root}/determinism.json --out {root}/compare.json")
    assert "/venv/bin/python" in compare[0] and "w0-venv" not in compare[0]
    num = [x for x in lines if x.startswith("cargo test -p tessera-rl-num")]
    assert num == ["cargo test -p tessera-rl-num --profile wheel --all-features --locked"]
    assert "profile=wheel anchor=0 suite=0 num=0 compare=0" in p.stdout


def test_wheel_job_wheel_profile_fails_on_a_determinism_difference_after_every_check(tmp_path):
    p, lines, root, wr = run(tmp_path, "wheel", compare_exit=1)
    assert p.returncode == 1
    assert "compare=1" in p.stdout
    # checks.py and the num tests still ran: the failure does not hide the other results.
    assert any("checks.py" in x for x in lines) and any(x.startswith("cargo test -p tessera-rl-num") for x in lines)


@pytest.mark.parametrize("profile", [(), ("release",)], ids=["absent", "release"])
def test_wheel_job_release_profile_builds_one_wheel_and_no_comparison(tmp_path, profile):
    p, lines, root, wr = run(tmp_path, *profile)
    assert p.returncode == 0, p.stderr
    builds = _builds(lines)
    assert len(builds) == 1 and "--profile release --locked --target x86_64-unknown-linux-gnu --out dist" in builds[0]
    assert not any("w0-" in x or " compare " in x for x in lines)
    assert [x for x in lines if x.startswith("cargo test -p tessera-rl-num")] == [
        "cargo test -p tessera-rl-num --release --all-features --locked"]
    assert "profile=release anchor=0 suite=0 num=0 compare=0" in p.stdout


@pytest.mark.parametrize("profile", [("release",), ("wheel",)], ids=["release", "wheel"])
def test_wheel_job_builds_every_wheel_with_the_onnx_feature(tmp_path, profile):
    p, lines, root, wr = run(tmp_path, *profile)
    assert p.returncode == 0, p.stderr
    builds = _builds(lines)
    assert builds and all(" --features onnx" in b for b in builds), builds
    text = "\n".join(lines)
    # The smoke check: the installed wheel's python, -I -c, calling onnx_load once.
    assert text.count("t._native.onnx_load(") == 1
    assert "/venv/bin/python -I -c" in text.split("t._native.onnx_load(")[0].rsplit("\npy ", 1)[-1]
    assert "onnx=0" in p.stdout


def test_wheel_job_refuses_the_onnx_feature_on_other_targets(tmp_path):
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "wheel_job.sh", root / "scripts" / "wheel_job.sh")
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT and not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))}
    env["WHEEL_RUN"] = str(tmp_path / "wr")
    fake = tmp_path / "fake"
    fake.mkdir()
    _tool(fake / "curl", "exit 1\n")
    env["PATH"] = f"{fake}:{os.environ['PATH']}"
    script = (root / "scripts" / "wheel_job.sh").read_text()
    # The target check comes before any download.
    assert script.index('case "$target" in') < script.index("fetch \"https://static.rust-lang.org")

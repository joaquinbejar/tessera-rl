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

# scripts/pgo_w1_build.sh is a bash step of a Linux container run, and its
# fakes are sh scripts; neither runs on Windows.
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash build script with sh fakes")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pgo_w1_build.sh"


# The caller's build environment the fixtures drop before spawning a build script,
# so a test sees only what the script itself sets (CI exports RUSTFLAGS=-D warnings
# workflow-wide; a developer may export any of these). An explicit list, not an
# empty environment: the scripts still need PATH, HOME and the rest.
#   RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS, CARGO_BUILD_RUSTFLAGS: compiler flags; the fake
#     maturin logs RUSTFLAGS and the W1/W2 equality test compares it.
#   CARGO_PROFILE_* (prefix), RUSTC_WRAPPER, CARGO_BUILD_TARGET: change the build
#     cargo would run; they land in the fake maturin's environment dump.
#   RUSTDOCFLAGS, CARGO_TARGET_DIR: set by other CI jobs; the fake maturin logs
#     CARGO_TARGET_DIR, and both land in the environment dump.
#   CARGO_TERM_COLOR: set workflow-wide by CI; lands in the environment dump.
AMBIENT_BUILD = ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "RUSTC_WRAPPER",
                 "CARGO_BUILD_TARGET", "RUSTDOCFLAGS", "CARGO_TARGET_DIR", "CARGO_TERM_COLOR")
AMBIENT_BUILD_PREFIX = "CARGO_PROFILE_"
HOSTILE = {"RUSTFLAGS": "-D warnings", "RUSTDOCFLAGS": "-D warnings", "CARGO_TARGET_DIR": "/nonexistent/target",
           "CARGO_TERM_COLOR": "always", "CARGO_ENCODED_RUSTFLAGS": "-Cdebuginfo=2", "CARGO_BUILD_RUSTFLAGS": "-Copt-level=0",
           "CARGO_PROFILE_WHEEL_OPT_LEVEL": "0", "RUSTC_WRAPPER": "/nonexistent/wrapper",
           "CARGO_BUILD_TARGET": "aarch64-unknown-linux-gnu"}


def clean_environ(path_prefix) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in AMBIENT_BUILD and not k.startswith(AMBIENT_BUILD_PREFIX)}
    env["PATH"] = f"{path_prefix}:{os.environ['PATH']}"
    return env


def tool(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def uv_tarball(log: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in {
            "uvx": f'[ -z "$FAKE_ENV_DUMP" ] || env | sort > "$FAKE_ENV_DUMP"\necho "uvx $*" >> {log}\nfor a in "$@"; do [ "$prev" = --out ] && touch "$a/tessera_rl-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"; prev="$a"; done\n',
            "uv": f'echo "uv $*" >> {log}\nif [ "$1" = venv ]; then mkdir -p "$4/bin"; if [ -n "$FAKE_SYSTEM_PY" ]; then ln -s "$FAKE_SYSTEM_PY" "$4/bin/python"; else printf "#!/bin/sh\\necho py \\"\\$*\\" >> {log}\\n" > "$4/bin/python"; chmod +x "$4/bin/python"; fi; fi\n',
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


def run(tmp_path, rustup_ok=True, system_py=None):
    bin_dir, log, rsha, usha = setup(tmp_path, rustup_ok)
    env = clean_environ(bin_dir)
    if system_py:
        env["FAKE_SYSTEM_PY"] = str(system_py)
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


def test_pgo_w1_build_asks_uv_for_a_managed_python_only(tmp_path):
    p, out, log = run(tmp_path)
    assert p.returncode == 0, p.stderr
    venv = [x for x in log.splitlines() if x.startswith("uv venv")]
    assert len(venv) == 1 and "--python-preference only-managed" in venv[0]


def test_pgo_w1_build_refuses_a_venv_on_an_interpreter_outside_the_run_dir(tmp_path):
    # The pgo-feas-2026-11-10-n failure: a venv on the image's /opt/python.
    system = tmp_path / "opt" / "python"
    system.parent.mkdir()
    system.write_text("#!/bin/sh\n")
    system.chmod(0o755)
    p, out, log = run(tmp_path, system_py=system)
    assert p.returncode != 0 and "resolves outside the run dir" in p.stderr
    assert "uv pip install" not in log


def test_pgo_w1_build_refuses_a_tool_with_the_wrong_hash(tmp_path):
    p, out, log = run(tmp_path, rustup_ok=False)
    assert p.returncode != 0 and "sha256 mismatch" in p.stderr and "rustup-init" not in log


@pytest.mark.parametrize("name", ["w1", "dist", "venv"])
def test_pgo_w1_build_writes_only_under_the_run_dir(tmp_path, name):
    p, out, _ = run(tmp_path)
    assert p.returncode == 0 and (out / name).exists()
    assert {x.name for x in tmp_path.iterdir()} == {"fake", "payload", "log", "run"}


def test_pgo_w1_build_pins_the_same_tools_as_the_w0_job():
    import re

    def pins(path):
        return dict(re.findall(r"^(RUSTUP_VERSION|UV_VERSION|MATURIN_VERSION)=(\S+)$", path.read_text(), re.M))

    w0 = pins(SCRIPT.parent / "wheel_job.sh")
    assert set(w0) == {"RUSTUP_VERSION", "UV_VERSION", "MATURIN_VERSION"} and pins(SCRIPT) == w0
    assert "numpy==2.4.6" in (SCRIPT.parent / "wheel_job.sh").read_text() and "NUMPY_VERSION=2.4.6" in SCRIPT.read_text()


def test_pgo_w1_build_no_wheel_fails_with_the_count_message(tmp_path):
    bin_dir, log, rsha, usha = setup(tmp_path)
    uvx = tmp_path / "payload" / "uv.tar.gz"
    import tarfile as t
    import io as i

    buf = i.BytesIO()
    with t.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in {"uvx": "exit 0\n", "uv": "exit 0\n"}.items():
            data = ("#!/bin/sh\n" + body).encode()
            info = t.TarInfo(f"uv-x86_64-unknown-linux-gnu/{name}")
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, i.BytesIO(data))
    uvx.write_bytes(buf.getvalue())
    usha = hashlib.sha256(uvx.read_bytes()).hexdigest()
    env = clean_environ(bin_dir)
    p = subprocess.run(["bash", str(SCRIPT), rsha, usha, str(tmp_path / "run")], env=env, capture_output=True, text=True, check=False)
    assert p.returncode != 0 and "expected one wheel, found 0" in p.stderr


def test_pgo_w1_build_sets_a_writable_home(tmp_path):
    p, out, _ = run(tmp_path)
    assert p.returncode == 0 and (out / "w1" / "home").is_dir()


def test_pgo_w1_build_fixture_keeps_the_callers_build_environment_out(tmp_path, monkeypatch):
    # Pins clean_environ(): with a hostile caller, the fake maturin sees none of it except
    # CARGO_TARGET_DIR, which the script itself sets under the run dir.
    for k, v in HOSTILE.items():
        monkeypatch.setenv(k, v)
    dump = tmp_path / "uvx.env"
    monkeypatch.setenv("FAKE_ENV_DUMP", str(dump))
    p, out, _ = run(tmp_path)
    assert p.returncode == 0, p.stderr
    seen = dict(line.partition("=")[::2] for line in dump.read_text().splitlines())
    leaked = {k: seen[k] for k in HOSTILE if k in seen and k != "CARGO_TARGET_DIR"}
    assert leaked == {}
    assert seen["CARGO_TARGET_DIR"] == str(out / "w1" / "target")

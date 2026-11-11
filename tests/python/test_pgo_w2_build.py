# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/pgo_w2_build.sh against fake tools (no network, no build), and
tools/pgo/coverage.py on synthetic profiles."""

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pgo_w2_build.sh"
sys.path.insert(0, str(ROOT / "tools" / "pgo"))
import coverage  # noqa: E402

posix = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash build script with sh fakes")
WHEEL = "tessera_rl-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"


def tool(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def uv_tarball(log: Path, profraw: bool) -> bytes:
    py = (f'printf "#!/bin/sh\\necho py \\"\\$*\\" >> {log}\\n'
          + ('case \\"\\$*\\" in *train.py*) d=\\$(dirname \\"\\$LLVM_PROFILE_FILE\\"); echo raw > \\"\\$d/a_1.profraw\\";; esac\\n' if profraw else "")
          + '" > "$4/bin/python"; chmod +x "$4/bin/python"')
    files = {
        # maturin: records RUSTFLAGS and the target dir, writes one wheel into --out.
        "uvx": f'echo "uvx RUSTFLAGS=$RUSTFLAGS TARGET=$CARGO_TARGET_DIR $*" >> {log}\n'
               f'for a in "$@"; do [ "$prev" = --out ] && touch "$a/{WHEEL}"; prev="$a"; done\n',
        "uv": f'echo "uv $*" >> {log}\nif [ "$1" = venv ]; then mkdir -p "$4/bin"; {py}; fi\n',
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in files.items():
            data = ("#!/bin/sh\n" + body).encode()
            info = tarfile.TarInfo(f"uv-x86_64-unknown-linux-gnu/{name}")
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def run(tmp_path, variant="full", llvm=("23", "23"), profraw=True, rel=False):
    fake, payload, log, sysroot = tmp_path / "fake", tmp_path / "payload", tmp_path / "log", tmp_path / "sysroot"
    payload.mkdir()
    (payload / "rustup-init").write_text(f"#!/bin/sh\necho rustup-init >> {log}\n")
    (payload / "uv.tar.gz").write_bytes(uv_tarball(log, profraw))
    tool(fake / "curl", 'out=""; url=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; https*) url="$1";; esac; shift; done\n'
         f'case "$url" in *rustup-init) cp {payload}/rustup-init "$out";; *uv-*) cp {payload}/uv.tar.gz "$out";; esac\n')
    tool(fake / "rustup", f'echo "rustup $*" >> {log}\n')
    tool(fake / "rustc", f'case "$*" in "--print sysroot") echo {sysroot};; "-vV") echo "LLVM version: {llvm[0]}.1.3";; esac\n')
    tool(sysroot / "lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata",
         f'echo "profdata $*" >> {log}\ncase "$1" in --version) echo "LLVM version {llvm[1]}.1.3";; '
         'merge) shift; [ "$1" = -o ] && echo data > "$2";; show) echo "  k:"; echo "    Function count: 5";; esac\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    out = (tmp_path / "run") if not rel else Path("run")
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    p = subprocess.run(["bash", str(SCRIPT), digest(payload / "rustup-init"), digest(payload / "uv.tar.gz"), str(out), variant],
                       env=env, capture_output=True, text=True, check=False, cwd=ROOT if not rel else tmp_path)
    return p, out, log.read_text() if log.exists() else ""


@posix
def test_pgo_w2_build_full_instruments_trains_merges_and_builds_with_the_profile(tmp_path):
    p, out, log = run(tmp_path)
    assert p.returncode == 0, p.stderr
    builds = [x for x in log.splitlines() if x.startswith("uvx ")]
    assert len(builds) == 2
    gen, use = builds
    assert f"-Cprofile-generate={out}/w2-full/profraw" in gen and f"TARGET={out}/w2-full/target-gen" in gen
    assert f"-Cprofile-use={out}/w2-full/merged.profdata -Cllvm-args=-pgo-warn-missing-function" in use
    assert f"TARGET={out}/w2-full/target-use" in use
    for b in builds:
        assert "--profile wheel" in b and "--target x86_64-unknown-linux-gnu" in b and "--locked" in b
    train = [x for x in log.splitlines() if "train.py" in x]
    assert len(train) == 1 and "--holdout-build" not in train[0]
    assert any(x.startswith("profdata merge -o") for x in log.splitlines())
    assert any("coverage.py" in x for x in log.splitlines()), "the coverage acceptance runs"
    lines = (out / "dist-full" / "w2-full.sha256").read_text().splitlines()
    assert lines[0].endswith(WHEEL) and lines[1].endswith("merged.profdata")


@posix
def test_pgo_w2_build_holdout_trains_without_the_holdout(tmp_path):
    p, out, log = run(tmp_path, variant="holdout")
    assert p.returncode == 0, p.stderr
    train = [x for x in log.splitlines() if "train.py" in x]
    assert len(train) == 1 and "--holdout-build" in train[0]
    assert (out / "dist-holdout" / "w2-holdout.sha256").exists()


@posix
def test_pgo_w2_build_refuses_an_llvm_version_mismatch(tmp_path):
    p, _, log = run(tmp_path, llvm=("23", "22"))
    assert p.returncode != 0 and "does not match" in p.stderr and "uvx " not in log


@posix
def test_pgo_w2_build_refuses_a_training_without_profraw(tmp_path):
    p, _, log = run(tmp_path, profraw=False)
    assert p.returncode != 0 and "no profraw" in p.stderr
    assert sum(1 for x in log.splitlines() if x.startswith("uvx ")) == 1, "no final build"


@posix
@pytest.mark.parametrize("bad", ["w2", "both"])
def test_pgo_w2_build_refuses_a_bad_variant(tmp_path, bad):
    p, _, _ = run(tmp_path, variant=bad)
    assert p.returncode == 2 and "full or holdout" in p.stderr


@posix
def test_pgo_w2_build_refuses_a_relative_run_dir(tmp_path):
    p, _, _ = run(tmp_path, rel=True)
    assert p.returncode == 2 and "absolute" in p.stderr


def wheel_with(tmp_path, blob: bytes) -> Path:
    w = tmp_path / WHEEL
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("tessera_rl/_native.abi3.so", blob)
    return w


SHOW = """Counters:
  _RINvNtCs1_17tessera_rl_engine4pool10step_chunkNtNtCs2_18tessera_rl_classic8cartpole8CartPoleE:
    Hash: 0x1
    Counters: 3
    Function count: 1200
  _RINvNtCs1_17tessera_rl_engine4pool10step_chunkNtNtCs3_16tessera_rl_games3go93Go9E:
    Hash: 0x2
    Counters: 1
    Function count: 0
Instrumentation level: IR
"""


def test_pgo_coverage_reads_counts_per_function():
    got = coverage.counts(SHOW)
    assert list(got.values()) == [1200, 0]


def test_pgo_coverage_accepts_counted_kernels_and_a_clean_wheel(tmp_path):
    r = coverage.check(["step_chunk+CartPole"], SHOW, "warning: no profile data available for function x\n",
                       wheel_with(tmp_path, b"\x7fELF clean"))
    assert r["usable"] and r["kernels"] == {"step_chunk+CartPole": 1200} and r["functions_without_profile"] == 1


@pytest.mark.parametrize("kernel", ["step_chunk+Go9", "step_chunk+Othello", "step_chunk+Cart"])
def test_pgo_coverage_refuses_a_kernel_without_counts(tmp_path, kernel):
    r = coverage.check([kernel], SHOW, "", wheel_with(tmp_path, b"clean"))
    assert not r["usable"] and r["kernels"][kernel] == 0


def test_pgo_coverage_refuses_a_hash_mismatch_in_a_tessera_crate(tmp_path):
    log = "warning: function control flow change detected (hash mismatch) _RINvNtCs1_17tessera_rl_engine4pool10step_chunk\n"
    r = coverage.check(["step_chunk+CartPole"], SHOW, log, wheel_with(tmp_path, b"clean"))
    assert not r["usable"] and r["tessera_mismatches"]
    other = "warning: function control flow change detected (hash mismatch) _ZN4core3fmt\n"
    assert coverage.check(["step_chunk+CartPole"], SHOW, other, wheel_with(tmp_path, b"clean"))["usable"]


def test_pgo_coverage_refuses_an_instrumented_wheel(tmp_path):
    r = coverage.check(["step_chunk+CartPole"], SHOW, "", wheel_with(tmp_path, b"..__llvm_profile_runtime.."))
    assert not r["usable"] and r["instrumented_objects"] == ["tessera_rl/_native.abi3.so"]


def test_pgo_coverage_main_refuses_an_empty_kernel_list(tmp_path):
    k = tmp_path / "kernels.toml"
    k.write_text("kernels = []\n")
    for name in ("show.txt", "final.log"):
        (tmp_path / name).write_text("")
    args = ["--kernels", str(k), "--profile", str(tmp_path / "show.txt"), "--build-log", str(tmp_path / "final.log"),
            "--wheel", str(wheel_with(tmp_path, b"clean")), "--out", str(tmp_path / "c.json")]
    assert coverage.main(args) == 2


def test_pgo_coverage_main_writes_the_result(tmp_path):
    k = tmp_path / "kernels.toml"
    k.write_text('kernels = ["step_chunk+CartPole"]\n')
    (tmp_path / "show.txt").write_text(SHOW)
    (tmp_path / "final.log").write_text("")
    args = ["--kernels", str(k), "--profile", str(tmp_path / "show.txt"), "--build-log", str(tmp_path / "final.log"),
            "--wheel", str(wheel_with(tmp_path, b"clean")), "--out", str(tmp_path / "c.json")]
    assert coverage.main(args) == 0 and json.loads((tmp_path / "c.json").read_text())["usable"]


def test_pgo_kernels_list_is_filled_and_matches_the_w1_symbols():
    import evidence
    kernels = evidence.load_kernels(str(ROOT / "tools" / "pgo" / "kernels.toml"))
    assert len(kernels) == 11 and all(k.startswith("step_chunk+") for k in kernels)

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


W1_SCRIPT = ROOT / "scripts" / "pgo_w1_build.sh"


# Variables of the caller's build environment (CI sets RUSTFLAGS and CARGO_TERM_COLOR
# workflow-wide, other jobs RUSTDOCFLAGS and CARGO_TARGET_DIR) that the fake tools log
# or that cargo and maturin read; the fixtures drop them so a test sees only what the
# script itself sets.
AMBIENT_BUILD = ("RUST", "CARGO", "MATURIN", "LLVM_PROFILE")
HOSTILE = {"RUSTFLAGS": "-D warnings", "RUSTDOCFLAGS": "-D warnings", "CARGO_TARGET_DIR": "/nonexistent/target",
           "CARGO_TERM_COLOR": "always", "CARGO_ENCODED_RUSTFLAGS": "-Cdebuginfo=2", "CARGO_BUILD_RUSTFLAGS": "-Copt-level=0"}


def clean_environ(path_prefix) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(AMBIENT_BUILD)}
    env["PATH"] = f"{path_prefix}:{os.environ['PATH']}"
    return env


def tool(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def uv_tarball(log: Path, profraw: bool) -> bytes:
    py = (f'printf "#!/bin/sh\\necho py \\"\\$*\\" >> {log}\\n'
          + ('case \\"\\$*\\" in *train.py*) d=\\$(dirname \\"\\$LLVM_PROFILE_FILE\\"); echo raw > \\"\\$d/a_1.profraw\\";; esac\\n' if profraw else "")
          + '" > "$4/bin/python"; chmod +x "$4/bin/python"')
    files = {
        # maturin: logs its cwd, argv and RUSTFLAGS, dumps its build environment, writes one wheel into --out.
        "uvx": f'n=$(ls {log}.env.* 2>/dev/null | wc -l | tr -d " "); env | sort > {log}.env.$n\n'
               f'echo "uvx PWD=$PWD RUSTFLAGS=$RUSTFLAGS TARGET=$CARGO_TARGET_DIR $*" >> {log}\n'
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


def fakes(tmp_path, llvm=("23", "23"), profraw=True):
    fake, payload, log, sysroot = tmp_path / "fake", tmp_path / "payload", tmp_path / "log", tmp_path / "sysroot"
    payload.mkdir()
    (payload / "rustup-init").write_text(f"#!/bin/sh\necho rustup-init >> {log}\n")
    (payload / "uv.tar.gz").write_bytes(uv_tarball(log, profraw))
    tool(fake / "curl", 'out=""; url=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; https*) url="$1";; esac; shift; done\n'
         f'case "$url" in *rustup-init) cp {payload}/rustup-init "$out";; *uv-*) cp {payload}/uv.tar.gz "$out";; esac\n')
    tool(fake / "rustup", f'echo "rustup $*" >> {log}\n')
    tool(fake / "rustc", f'case "$*" in "--print sysroot") echo {sysroot};; "-vV") echo "LLVM version: {llvm[0]}.1.1";; esac\n')
    tool(sysroot / "lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata",
         f'echo "profdata $*" >> {log}\ncase "$1" in --version) echo "  LLVM version {llvm[1]}.1.1-rust-1.99.1-stable";; '
         'merge) shift; [ "$1" = -o ] && echo data > "$2";; show) echo "  k:"; echo "    Block counts: [5]";; esac\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    env = clean_environ(fake)
    return env, log, digest(payload / "rustup-init"), digest(payload / "uv.tar.gz")


def source_tree(tmp_path) -> Path:
    """W1's commit as the script sees it: an empty kernel list, no coverage.py."""
    src = tmp_path / "src"
    (src / "tools" / "pgo").mkdir(parents=True)
    (src / "tools" / "pgo" / "kernels.toml").write_text("kernels = []\n")
    (src / "tools" / "pgo" / "manifest.toml").write_bytes((ROOT / "tools" / "pgo" / "manifest.toml").read_bytes())
    return src


def run(tmp_path, variant="full", llvm=("23", "23"), profraw=True, rel=False, swap=False, extra=None):
    env, log, rsha, usha = fakes(tmp_path, llvm, profraw)
    env.update(extra or {})
    src, tool_tree = source_tree(tmp_path), ROOT
    if swap:
        src, tool_tree = tool_tree, src
    out = (tmp_path / "run") if not rel else Path("run")
    p = subprocess.run(["bash", str(SCRIPT), rsha, usha, str(out), variant, str(src), str(tool_tree)],
                       env=env, capture_output=True, text=True, check=False, cwd=tmp_path)
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
    assert all(f"PWD={tmp_path / 'src'} " in b for b in builds), "both builds from the source tree"
    train = [x for x in log.splitlines() if "train.py" in x]
    assert len(train) == 1 and "--holdout-build" not in train[0] and str(ROOT / "tools" / "pgo" / "train.py") in train[0]
    assert any(x.startswith("profdata show --all-functions --counts") for x in log.splitlines())
    cov = [x for x in log.splitlines() if "coverage.py" in x]
    assert len(cov) == 1 and "--variant full" in cov[0] and str(ROOT / "tools" / "pgo" / "kernels.toml") in cov[0]
    lines = (out / "dist-full" / "w2-full.sha256").read_text().splitlines()
    assert lines[0].endswith(WHEEL) and lines[1].endswith("merged.profdata")


@posix
def test_pgo_w2_build_holdout_trains_without_the_holdout(tmp_path):
    p, out, log = run(tmp_path, variant="holdout")
    assert p.returncode == 0, p.stderr
    train = [x for x in log.splitlines() if "train.py" in x]
    assert len(train) == 1 and "--holdout-build" in train[0]
    assert any("--variant holdout" in x for x in log.splitlines() if "coverage.py" in x)
    assert (out / "dist-holdout" / "w2-holdout.sha256").exists()


@posix
def test_pgo_w2_build_refuses_swapped_trees(tmp_path):
    p, _, log = run(tmp_path, swap=True)
    assert p.returncode != 0 and "not the frozen one" in p.stderr and "uvx " not in log


@posix
def test_pgo_w2_build_refuses_a_tool_tree_whose_manifest_is_not_the_gates(tmp_path):
    env, log, rsha, usha = fakes(tmp_path)
    tool_tree = tmp_path / "tool"
    (tool_tree / "tools" / "pgo").mkdir(parents=True)
    (tool_tree / "tools" / "pgo" / "kernels.toml").write_bytes(KERNELS.read_bytes())
    m = bytearray((ROOT / "tools" / "pgo" / "manifest.toml").read_bytes())
    m[-1:] = b" " if m[-1:] != b" " else b"\n"
    (tool_tree / "tools" / "pgo" / "manifest.toml").write_bytes(bytes(m))
    p = subprocess.run(["bash", str(SCRIPT), rsha, usha, str(tmp_path / "run"), "full", str(source_tree(tmp_path)),
                        str(tool_tree)], env=env, capture_output=True, text=True, check=False, cwd=tmp_path)
    text = log.read_text() if log.exists() else ""
    assert p.returncode != 0 and "manifest is not the gate's" in p.stderr and "uvx " not in text


@posix
def test_pgo_w2_build_refuses_an_llvm_version_mismatch(tmp_path):
    p, _, log = run(tmp_path, llvm=("23", "22"))
    assert p.returncode != 0 and "does not match" in p.stderr and "uvx " not in log


@posix
def test_pgo_w2_build_refuses_a_training_without_profraw(tmp_path):
    p, _, log = run(tmp_path, profraw=False)
    assert p.returncode != 0 and "no profraw" in p.stderr
    assert sum(1 for x in log.splitlines() if x.startswith("uvx ")) == 1, "no final build"


REFUSED = ["RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_ENCODED_RUSTDOCFLAGS", "CARGO_BUILD_RUSTFLAGS",
           "CARGO_BUILD_RUSTDOCFLAGS", "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET_DIR", "CARGO_BUILD_TARGET",
           "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTC"]
PATTERNED = ["CARGO_PROFILE_WHEEL_OPT_LEVEL", "CARGO_PROFILE_RELEASE_LTO", "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS"]


@posix
@pytest.mark.parametrize("name", REFUSED + PATTERNED)
@pytest.mark.parametrize("value", ["x", ""], ids=["set", "empty"])
def test_pgo_w2_build_refuses_an_ambient_build_variable_before_any_fetch(tmp_path, name, value):
    p, out, log = run(tmp_path, extra={name: value})
    assert p.returncode == 2, p.stderr
    msg = (f"pgo_w2_build: {name} is set in the environment; refused" if name in REFUSED
           else "pgo_w2_build: a CARGO_PROFILE_* or per-target RUSTFLAGS variable is set; refused")
    assert msg in p.stderr
    assert log == "" and not out.exists()


@posix
def test_pgo_w1_build_today_passes_an_ambient_rustflags_to_maturin(tmp_path):
    # Documents scripts/pgo_w1_build.sh as frozen at ac92079: it does not refuse or clear an
    # ambient RUSTFLAGS; W1's protection is the run script's start check and the control probe.
    env, log, rsha, usha = fakes(tmp_path)
    env["RUSTFLAGS"] = "-D warnings"
    p = subprocess.run(["bash", str(W1_SCRIPT), rsha, usha, str(tmp_path / "run")], env=env, capture_output=True,
                       text=True, check=False, cwd=source_tree(tmp_path))
    assert p.returncode == 0, p.stderr
    assert [x for x in log.read_text().splitlines() if x.startswith("uvx ")][0].count("RUSTFLAGS=-D warnings ") == 1


@posix
@pytest.mark.parametrize("bad", ["w2", "both"])
def test_pgo_w2_build_refuses_a_bad_variant(tmp_path, bad):
    p, _, _ = run(tmp_path, variant=bad)
    assert p.returncode == 2 and "full, holdout or check" in p.stderr


@posix
def test_pgo_w2_build_refuses_a_relative_run_dir(tmp_path):
    p, _, _ = run(tmp_path, rel=True)
    assert p.returncode == 2 and "absolute" in p.stderr


def build_env(path: Path, run_dir: Path, own: str) -> dict:
    """The maturin call's environment, with this script's own tool dir and run
    dir replaced by placeholders."""
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition("=")
        out[k] = (v.replace(str(run_dir / own), "<tools>").replace(str(run_dir), "<run>")
                  .replace(str(run_dir.parent), "<test>"))
    return out


@posix
@pytest.mark.parametrize("ambient", [False, True], ids=["clean", "hostile"])
def test_pgo_w2_final_build_equals_w1_build_except_the_pgo_flags(tmp_path, monkeypatch, ambient):
    # "hostile": the caller's build environment as CI sets it (and worse) must not reach the fakes.
    if ambient:
        for k, v in HOSTILE.items():
            monkeypatch.setenv(k, v)
    # W1: scripts/pgo_w1_build.sh as it was at ac92079 (unchanged since), run from the source tree.
    w1_dir, w2_dir = tmp_path / "a", tmp_path / "b"
    w1_dir.mkdir()
    w2_dir.mkdir()
    env, log1, rsha, usha = fakes(w1_dir)
    src1 = source_tree(w1_dir)
    p = subprocess.run(["bash", str(W1_SCRIPT), rsha, usha, str(w1_dir / "run")], env=env, capture_output=True,
                       text=True, check=False, cwd=src1)
    assert p.returncode == 0, p.stderr
    p2, _, log2 = run(w2_dir)
    assert p2.returncode == 0, p2.stderr
    w1_call = [x for x in log1.read_text().splitlines() if x.startswith("uvx ")]
    w2_calls = [x for x in log2.splitlines() if x.startswith("uvx ")]
    assert len(w1_call) == 1 and len(w2_calls) == 2

    def argv(line):
        words = line.split(" TARGET=", 1)[1].split(" ", 1)[1].split()
        i = words.index("--out")
        return words[:i] + words[i + 2:]

    assert argv(w1_call[0]) == argv(w2_calls[1]), "same maturin argv, --out aside"
    assert "RUSTFLAGS= " in w1_call[0], "W1 had no RUSTFLAGS"
    e1 = build_env(Path(f"{w1_dir / 'log'}.env.0"), w1_dir / "run", "w1")
    e2 = build_env(Path(f"{w2_dir / 'log'}.env.1"), w2_dir / "run", "w2-full")
    differ = sorted(k for k in set(e1) | set(e2) if e1.get(k) != e2.get(k))
    assert set(differ) <= {"RUSTFLAGS", "CARGO_TARGET_DIR", "PWD", "OLDPWD", "_"}, differ


def wheel_with(tmp_path, blob: bytes) -> Path:
    w = tmp_path / WHEEL
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("tessera_rl/_native.abi3.so", blob)
    return w


FIX = ROOT / "tests" / "python" / "fixtures" / "pgo"
# Real output of the pinned toolchain (rustc 1.99.1, LLVM 23.1.1) on a toy crate named tessera_rl_toy:
# `llvm-profdata show --all-functions --counts`, and the warnings of a -Cprofile-use build after one
# function changed (hash mismatch) and one was added (no profile data).
SHOW = (FIX / "profdata-show-counts.txt").read_text()
WARNINGS = (FIX / "profile-use-warnings.txt").read_text()
# Every v0 defined symbol of W1's _native.abi3.so (sha256 ad64a5e1...), from llvm-nm, unfiltered.
W1_SYMBOLS = [l for l in (FIX / "w1-v0-symbols.txt").read_text().splitlines() if l.startswith("_R")]
KERNELS = ROOT / "tools" / "pgo" / "kernels.toml"


def test_pgo_coverage_reads_block_counts_of_the_real_show_format():
    got = coverage.counts(SHOW)
    assert len(W1_SYMBOLS) > 2000, "the full symbol list, not a pre-filtered one"
    hot = next(c for n, c in got.items() if n.endswith("14tessera_rl_toy3hot"))
    cold = next(c for n, c in got.items() if n.endswith("14tessera_rl_toy4cold"))
    assert hot == 1000 and cold == 0
    assert not any(";" in n for n in got), "the module prefix is stripped"


def test_pgo_coverage_a_function_is_reached_by_its_largest_block_not_its_first():
    # LLVM's counters need not include the entry block: [0, 7] is reached.
    show = SHOW + "  m.cgu.0;_RNvCs1_14tessera_rl_toy7reached:\n    Hash: 0x1\n    Counters: 2\n    Block counts: [0, 7]\n" \
                + "  m.cgu.0;_RNvCs1_14tessera_rl_toy6unseen:\n    Hash: 0x2\n    Counters: 3\n    Block counts: [0, 0, 0]\n"
    got = coverage.counts(show)
    assert got["_RNvCs1_14tessera_rl_toy7reached"] == 7 and got["_RNvCs1_14tessera_rl_toy6unseen"] == 0


def test_pgo_coverage_without_counts_the_show_output_has_no_counts():
    # The format check of the pipeline: without --counts nothing would be counted.
    plain = "\n".join(l for l in SHOW.splitlines() if "Block counts" not in l)
    assert coverage.counts(plain) == {}


def test_pgo_coverage_reads_the_real_warnings():
    mism = [l for l in WARNINGS.splitlines() if coverage.MISMATCH.search(l)]
    miss = [l for l in WARNINGS.splitlines() if coverage.MISSING.search(l)]
    assert len(mism) == 1 and "14tessera_rl_toy3hot" in mism[0]
    assert len(miss) == 2


def test_pgo_coverage_refuses_a_hash_mismatch_in_a_tessera_crate(tmp_path):
    r = coverage.check(["tessera_rl_toy+hot"], SHOW, WARNINGS, wheel_with(tmp_path, b"clean"))
    assert r["kernels"]["tessera_rl_toy+hot"] == 1000
    assert not r["usable"] and len(r["tessera_mismatches"]) == 1 and r["functions_without_profile"] == 2
    assert coverage.check(["tessera_rl_toy+hot"], SHOW, "", wheel_with(tmp_path, b"clean"))["usable"]


def test_pgo_coverage_refuses_a_kernel_without_counts(tmp_path):
    r = coverage.check(["tessera_rl_toy+cold"], SHOW, "", wheel_with(tmp_path, b"clean"))
    assert not r["usable"] and r["kernels"]["tessera_rl_toy+cold"] == 0


def test_pgo_coverage_refuses_an_instrumented_wheel(tmp_path):
    r = coverage.check(["tessera_rl_toy+hot"], SHOW, "", wheel_with(tmp_path, b"..__llvm_profile_runtime.."))
    assert not r["usable"] and r["instrumented_objects"] == ["tessera_rl/_native.abi3.so"]


MANIFEST = ROOT / "tools" / "pgo" / "manifest.toml"


def frozen():
    return evidence_mod().load_kernels(str(KERNELS)), coverage.holdout_only(str(KERNELS), str(MANIFEST))


def evidence_mod():
    import evidence
    return evidence


def test_pgo_kernels_each_match_exactly_one_real_w1_symbol():
    kernels, holdout = frozen()
    for k in kernels:
        hits = [s for s in W1_SYMBOLS if all(coverage.contains(s, p) for p in k.split("+"))]
        assert len(hits) == 1, (k, hits)
    assert set(holdout) <= set(kernels)


def test_pgo_kernels_holdout_only_set_follows_the_rows():
    _, holdout = frozen()
    assert holdout == ["tessera_rl_engine+step_chunk+tessera_rl_classic+PaddleDuel",
                       "tessera_rl_engine+step_chunk+tessera_rl_games+Go9",
                       "tessera_rl_engine+batch+run_drain+tessera_rl_classic+CartPole"]


@pytest.mark.parametrize("edit, match", [
    (lambda t: t.replace('["state/Tag-v1-2-2"]', '["state/Nope-v1"]'), "unknown"),
    (lambda t: t.replace('["search/Connect4-v1"]', '["games/Connect4-v1"]'), "reach no kernel"),
    (lambda t: t.replace('"_native+env_type+import_record" = ["state-operations/Connect4-v1"]\n', ""), "exactly the kernels"),
])
def test_pgo_kernels_rows_are_checked(tmp_path, edit, match):
    k = tmp_path / "kernels.toml"
    k.write_text(edit(KERNELS.read_text()))
    with pytest.raises(ValueError, match=match):
        coverage.holdout_only(str(k), str(MANIFEST))


def profile_of(nonzero: set[str], kernels: list[str]) -> str:
    """A profile in the real show format whose functions are W1's kernel symbols."""
    lines = ["Counters:"]
    for k in kernels:
        sym = next(s for s in W1_SYMBOLS if all(coverage.contains(s, p) for p in k.split("+")))
        lines += [f"  _native.abc-cgu.0;{sym}:", "    Hash: 0x1", "    Counters: 1",
                  f"    Block counts: [{7 if k in nonzero else 0}]"]
    return "\n".join(lines) + "\n"


def test_pgo_coverage_holdout_variant_needs_the_holdout_kernels_at_zero(tmp_path):
    kernels, holdout = frozen()
    trained = set(kernels) - set(holdout)
    w = wheel_with(tmp_path, b"clean")
    w2h_profile = profile_of(trained, kernels)
    assert coverage.check(kernels, w2h_profile, "", w, holdout)["usable"], "W2H: holdout zero, rest counted"
    assert not coverage.check(kernels, w2h_profile, "", w)["usable"], "the same profile cannot pass as W2F"
    w2f_profile = profile_of(set(kernels), kernels)
    assert coverage.check(kernels, w2f_profile, "", w)["usable"]
    assert not coverage.check(kernels, w2f_profile, "", w, holdout)["usable"], "a W2H profile that saw the holdout"


def test_pgo_coverage_anchors_kernels_on_the_identifier_boundary():
    s = "_RINvNtCs1_17tessera_rl_engine4pool10step_chunkNtNtCs2_18tessera_rl_classic8longgrid14GridExtensionsE"
    assert not coverage.contains(s, "Grid") and coverage.contains(s, "GridExtensions") and coverage.contains(s, "longgrid")
    real = next(s for s in W1_SYMBOLS if "8Connect4" in s and "10step_chunk" in s)
    assert coverage.contains(real, "Connect4"), "8connect48Connect4: the previous identifier ends in a digit"
    assert coverage.contains("_RNvNtNtCs6m83gMKcQ9H_7__native3ffi8env_type13import_record", "_native")


def test_pgo_coverage_main_refuses_an_empty_kernel_list(tmp_path):
    k = tmp_path / "kernels.toml"
    k.write_text("kernels = []\nholdout_kernels = []\n")
    (tmp_path / "show.txt").write_text(SHOW)
    (tmp_path / "final.log").write_text("")
    args = ["--kernels", str(k), "--manifest", str(MANIFEST), "--profile", str(tmp_path / "show.txt"), "--build-log",
            str(tmp_path / "final.log"), "--wheel", str(wheel_with(tmp_path, b"clean")), "--variant", "full",
            "--out", str(tmp_path / "c.json")]
    assert coverage.main(args) == 2


def test_pgo_coverage_main_writes_the_result(tmp_path):
    kernels, holdout = frozen()
    (tmp_path / "show.txt").write_text(profile_of(set(kernels) - set(holdout), kernels))
    (tmp_path / "final.log").write_text("")
    args = ["--kernels", str(KERNELS), "--manifest", str(MANIFEST), "--profile", str(tmp_path / "show.txt"), "--build-log",
            str(tmp_path / "final.log"), "--wheel", str(wheel_with(tmp_path, b"clean")), "--out", str(tmp_path / "c.json")]
    assert coverage.main([*args, "--variant", "holdout"]) == 0
    assert json.loads((tmp_path / "c.json").read_text())["variant"] == "holdout"
    assert coverage.main([*args, "--variant", "full"]) == 1


def test_pgo_w2_build_and_decide_freeze_the_same_kernel_list():
    import re
    sha = hashlib.sha256(KERNELS.read_bytes()).hexdigest()
    assert re.search(rf"^KERNELS_SHA256={sha}$", SCRIPT.read_text(), re.M)
    assert f'KERNELS_SHA256 = "{sha}"' in (ROOT / "tools" / "pgo" / "decide.py").read_text()


def test_pgo_kernels_a_kernel_with_holdout_and_trained_rows_is_not_holdout_only(tmp_path):
    k = tmp_path / "kernels.toml"
    k.write_text(KERNELS.read_text().replace(
        '"tessera_rl_engine+step_chunk+tessera_rl_games+Go9" = ["holdout/Go9-v1"]',
        '"tessera_rl_engine+step_chunk+tessera_rl_games+Go9" = ["holdout/Go9-v1", "games/Othello-v1"]'))
    assert "tessera_rl_engine+step_chunk+tessera_rl_games+Go9" not in coverage.holdout_only(str(k), str(MANIFEST))


@posix
def test_pgo_w2_build_check_variant_is_a_short_instrumented_training_and_a_parse(tmp_path):
    p, out, log = run(tmp_path, variant="check")
    assert p.returncode == 0, p.stderr
    builds = [x for x in log.splitlines() if x.startswith("uvx ")]
    assert len(builds) == 1 and "-Cprofile-generate=" in builds[0], "the instrumented build only"
    train = [x for x in log.splitlines() if "train.py" in x]
    assert len(train) == 1 and "--steps 8" in train[0] and "--holdout-build" not in train[0]
    cov = [x for x in log.splitlines() if "coverage.py" in x]
    assert len(cov) == 1 and "--variant check" in cov[0] and "--wheel" not in cov[0]
    assert not (out / "dist-check").exists() or not list((out / "dist-check").glob("*.whl"))


def test_pgo_coverage_check_variant_needs_one_kernel_counted(tmp_path):
    kernels, _ = frozen()
    base = ["--kernels", str(KERNELS), "--manifest", str(MANIFEST), "--variant", "check", "--out", str(tmp_path / "c.json")]
    (tmp_path / "one.txt").write_text(profile_of({kernels[0]}, kernels))
    assert coverage.main([*base, "--profile", str(tmp_path / "one.txt")]) == 0
    (tmp_path / "none.txt").write_text(profile_of(set(), kernels))
    assert coverage.main([*base, "--profile", str(tmp_path / "none.txt")]) == 1
    # A show without --counts parses to nothing: the check refuses it.
    plain = "\n".join(l for l in profile_of(set(kernels), kernels).splitlines() if "Block counts" not in l)
    (tmp_path / "plain.txt").write_text(plain)
    assert coverage.main([*base, "--profile", str(tmp_path / "plain.txt")]) == 1


def test_pgo_coverage_full_needs_the_wheel_and_the_build_log(tmp_path):
    (tmp_path / "s.txt").write_text(SHOW)
    args = ["--kernels", str(KERNELS), "--manifest", str(MANIFEST), "--profile", str(tmp_path / "s.txt"),
            "--variant", "full", "--out", str(tmp_path / "c.json")]
    assert coverage.main(args) == 2

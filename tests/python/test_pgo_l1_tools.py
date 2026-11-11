# SPDX-License-Identifier: MIT OR Apache-2.0
"""The tools window L1 calls (pgo-ladder-plan.md r4): the W1 control check,
the costs layout, and section 7 for an installed candidate wheel."""

import hashlib
import io
import json
import os
import pathlib
import stat
import subprocess
import tarfile

import pytest

from test_pgo_scripts import (  # noqa: F401  (shared fakes)
    ALL, MANIFEST, MANIFEST_PATH, TOOLS, FakeValgrind, all_names, callgrind, fake_environment, load, merge_now,
    run_parts,
)

control = load("control")
layout_costs = load("layout_costs")
ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pgo_candidate_checks.sh"
posix = pytest.mark.skipif(os.name == "nt", reason="POSIX only: a bash script with sh fakes")


def gate_and_expected(tmp_path):
    out, wheel, _ = run_parts(tmp_path, [all_names()])
    assert merge_now(out, wheel) == 0
    feas = json.loads((out / "feasibility.json").read_text())
    return out, wheel, control.expected(out, feas, MANIFEST)


def probe_w1(tmp_path, wheel, cost=None, scenarios=None, env=None):
    out = tmp_path / "w1-control"
    names = scenarios or [n for n in json.loads((tmp_path / "o" / "feasibility.json").read_text())["gating"]]
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", FakeValgrind(cost=cost))
        m.setattr(callgrind, "environment", env or fake_environment())
        args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--costs",
                "--wheel", str(wheel), "--probe", *sum((["--scenario", n] for n in names), [])]
        assert callgrind.main(args, platform="linux") == 0
    return out


def test_pgo_control_expected_pins_the_gates_n_step_files(tmp_path):
    out, wheel, want = gate_and_expected(tmp_path)
    assert want["wheel"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    gating = json.loads((out / "feasibility.json").read_text())["gating"]
    assert sorted(want["native_ir"]) == sorted(gating)


def test_pgo_control_holds_for_an_equal_probe(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    assert control.check(probe_w1(tmp_path, wheel), want) == []


def test_pgo_control_stops_on_one_instruction(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    off = next(iter(want["native_ir"]))
    probe = probe_w1(tmp_path, wheel, cost=lambda s, steps, out, ir: ir + (1 if s == off else 0))
    problems = control.check(probe, want)
    assert len(problems) == 1 and off in problems[0] and "(+1)" in problems[0]


def test_pgo_control_stops_on_a_missing_scenario_or_another_wheel(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    names = sorted(want["native_ir"])
    assert any("not probed" in p for p in control.check(probe_w1(tmp_path, wheel, scenarios=names[1:]), want))
    other = tmp_path / "other.whl"
    other.write_bytes(b"other")
    (tmp_path / "w1-control").rename(tmp_path / "old")
    assert any("not W1" in p for p in control.check(probe_w1(tmp_path, other), want))


def test_pgo_control_command_exits_one_on_a_difference(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(want))
    ok = probe_w1(tmp_path, wheel)
    assert control.main(["check", str(ok), str(exp)]) == 0
    want["native_ir"][next(iter(want["native_ir"]))] += 1
    exp.write_text(json.dumps(want))
    assert control.main(["check", str(ok), str(exp)]) == 1


def test_pgo_control_refuses_an_expected_file_without_the_stamp(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    want.pop("stamp")
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(want))
    assert control.main(["check", str(probe_w1(tmp_path, wheel)), str(exp)]) == 2


def test_pgo_control_stops_on_another_stamp(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    probe = probe_w1(tmp_path, wheel, env=fake_environment(toolchain="vg-2"))
    problems = control.check(probe, want)
    assert any("stamp differs" in p and "valgrind" in p for p in problems)


def test_pgo_control_ignores_the_gates_kernels_field(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    probe = probe_w1(tmp_path, wheel, env=fake_environment(kernels="k" * 64))
    assert control.check(probe, want) == []


def test_pgo_control_stops_on_a_scenario_probed_twice(tmp_path):
    _, wheel, want = gate_and_expected(tmp_path)
    probe = probe_w1(tmp_path, wheel)
    report = next(probe.glob("feasibility.probe.*.json"))
    (probe / "feasibility.probe.copy.json").write_text(report.read_text())
    assert any("probed twice" in p for p in control.check(probe, want))


def test_pgo_control_expected_refuses_a_missing_gate_file(tmp_path):
    out, _, want = gate_and_expected(tmp_path)
    name = next(iter(want["native_ir"]))
    steps = next(x.steps for x in ALL if x.name == name)
    (out / f"{name.replace('/', '_')}.{steps}.aa0.callgrind").unlink()
    args = ["expected", "--gate-out", str(out), "--feasibility", str(out / "feasibility.json"),
            "--manifest", MANIFEST_PATH, "--out", str(tmp_path / "e.json")]
    assert control.main(args) == 2 and not (tmp_path / "e.json").exists()


def probe_seconds(scale=1.0):
    return {s.name: {"seconds": 10.0 * scale * (i + 1), "ir": 1} for i, s in enumerate(ALL)}


def test_pgo_layout_costs_covers_every_aa_unit_once():
    r = layout_costs.layout(probe_seconds(), MANIFEST, 7)
    units = [u for b in r["parts"] for u in b["pairs"]]
    assert sorted(units) == sorted(callgrind.aa_units(MANIFEST)) and len(units) == len(set(units)) == 100
    assert r["unit_s"][f"{ALL[0].name}#aa0"] == 20.0 and r["unit_s"][f"{ALL[0].name}#aa3"] == 30.0


def test_pgo_layout_costs_refuses_a_missing_probe_and_stops_past_the_budget(tmp_path):
    p = probe_seconds()
    p.pop(ALL[3].name)
    with pytest.raises(ValueError, match="no probe"):
        layout_costs.layout(p, MANIFEST, 7)
    d = tmp_path / "probe"
    d.mkdir()
    stamp = {"sha256": {"wheel": "w0"}}
    (d / "feasibility.probe.x.json").write_text(json.dumps({"probe": probe_seconds(), **stamp}))
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "1e9"]) == 0
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "10"]) == 1
    (d / "feasibility.probe.x.json").write_text(json.dumps({"probe": p, **stamp}))
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "1e9"]) == 2


def test_pgo_layout_costs_refuses_mixed_wheels_duplicates_and_bad_arguments(tmp_path):
    d = tmp_path / "probe"
    d.mkdir()
    all_probe = probe_seconds()
    first = dict(list(all_probe.items())[:10])
    rest = dict(list(all_probe.items())[10:])
    (d / "feasibility.probe.a.json").write_text(json.dumps({"probe": first, "sha256": {"wheel": "w0"}}))
    (d / "feasibility.probe.b.json").write_text(json.dumps({"probe": rest, "sha256": {"wheel": "w1"}}))
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "1e9"]) == 2
    (d / "feasibility.probe.b.json").write_text(json.dumps({"probe": rest, "sha256": {"wheel": "w0"}}))
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "1e9"]) == 0
    (d / "feasibility.probe.c.json").write_text(json.dumps({"probe": first, "sha256": {"wheel": "w0"}}))
    assert layout_costs.main([str(d), MANIFEST_PATH, "7", "1e9"]) == 2
    assert layout_costs.main([str(d), MANIFEST_PATH, "0", "1e9"]) == 2
    assert layout_costs.main([str(d), MANIFEST_PATH]) == 2


def tool(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def clean_environ():
    """The caller's environment without its build settings (CI sets RUSTFLAGS
    at workflow level), so the script under test sees only what the test gives."""
    return {k: v for k, v in os.environ.items()
            if not (k.startswith("CARGO_") or k in ("RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC_WRAPPER"))}


def candidate(tmp_path, fail="", wrong_sha=False, extra=None):
    """Fakes for every tool the script calls; `fail` names a check that exits 1."""
    log, fake, payload = tmp_path / "log", tmp_path / "fake", tmp_path / "payload"
    payload.mkdir()
    (payload / "rustup-init").write_text(f"#!/bin/sh\necho rustup-init >> {log}\n")
    tool(payload / "python", f'echo "py $*" >> {log}\n'
         f'case "$*" in *wheel_golden.py*) exit {1 if fail == "anchor" else 0};; *"-m pytest"*) exit {1 if fail == "suite" else 0};; '
         f'*"determinism.py compare"*) exit {1 if fail == "compare" else 0};; esac\n')
    py = f'cp {payload}/python "$last/bin/python"'
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = (f'#!/bin/sh\necho "uv $*" >> {log}\nif [ "$1" = venv ]; then for a; do last="$a"; done; mkdir -p "$last/bin"; {py}; fi\n').encode()
        info = tarfile.TarInfo("uv-x86_64-unknown-linux-gnu/uv")
        info.size, info.mode = len(data), 0o755
        t.addfile(info, io.BytesIO(data))
    (payload / "uv.tar.gz").write_bytes(buf.getvalue())
    tool(fake / "curl", 'out=""; url=""; while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; https*) url="$1";; esac; shift; done\n'
         f'case "$url" in *rustup-init) cp {payload}/rustup-init "$out";; *uv-*) cp {payload}/uv.tar.gz "$out";; esac\n')
    tool(fake / "rustup", f'echo "rustup $*" >> {log}\n')
    tool(fake / "cargo", f'echo "cargo $*" >> {log}\n')
    src = tmp_path / "src"
    tool(src / "scripts" / "wheel_num_tests.sh", f'echo "num_tests WHEEL_CARGO_PROFILE=$WHEEL_CARGO_PROFILE" >> {log}\nexit {1 if fail == "num" else 0}\n')
    (tmp_path / "w.whl").write_bytes(b"w1")
    (tmp_path / "w0-det.json").write_text("{}")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    env = {**clean_environ(), "PATH": f"{fake}:{os.environ['PATH']}", "WHEEL_RUN": str(tmp_path / "run"), **(extra or {})}
    sha = "0" * 64 if wrong_sha else digest(tmp_path / "w.whl")
    p = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "w.whl"), sha, "wheel", digest(payload / "rustup-init"),
                        digest(payload / "uv.tar.gz"), str(tmp_path / "w0-det.json")],
                       env=env, cwd=src, capture_output=True, text=True, check=False)
    return p, log.read_text() if log.exists() else ""


@posix
def test_pgo_candidate_checks_install_the_given_wheel_and_run_every_check(tmp_path):
    p, log = candidate(tmp_path)
    assert p.returncode == 0, p.stderr
    install = [x for x in log.splitlines() if x.startswith("uv pip install")]
    assert len(install) == 1 and str(tmp_path / "w.whl") in install[0], "the given bytes, not a build"
    assert not any("maturin" in x for x in log.splitlines())
    assert "num_tests WHEEL_CARGO_PROFILE=wheel" in log
    for needle in ("wheel_golden.py", "-m pytest", "determinism.py record", "checks.py", "determinism.py compare"):
        assert needle in log, needle
    compare = next(x for x in log.splitlines() if "determinism.py compare" in x)
    assert str(tmp_path / "w0-det.json") in compare


@posix
@pytest.mark.parametrize("which", ["anchor", "suite", "num", "compare"])
def test_pgo_candidate_checks_fail_when_any_check_fails(tmp_path, which):
    p, log = candidate(tmp_path, fail=which)
    assert p.returncode != 0 and f"{which}=1" in p.stdout
    assert "checks.py" in log, "the statuses are still recorded"


@posix
def test_pgo_candidate_checks_refuse_a_wheel_other_than_the_named_one(tmp_path):
    p, log = candidate(tmp_path, wrong_sha=True)
    assert p.returncode == 2 and "not the candidate" in p.stderr and "uv pip install" not in log


@posix
@pytest.mark.parametrize("var", ["RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_TARGET_DIR", "CARGO_ENCODED_RUSTFLAGS",
                                 "CARGO_PROFILE_WHEEL_LTO"])
def test_pgo_candidate_checks_refuse_an_ambient_build_setting(tmp_path, var):
    p, log = candidate(tmp_path, extra={var: "-D warnings"})
    assert p.returncode == 2 and "refused" in p.stderr
    assert log == "", "nothing fetched, built or installed"

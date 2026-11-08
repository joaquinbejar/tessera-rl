# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo/decide.py on synthetic costs, and scripts/wheel_num_tests.sh with a
stub cargo. Nothing here has seen real Callgrind data."""

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[2]
TOOLS = ROOT / "tools" / "pgo"
sys.path.insert(0, str(TOOLS))
_spec = importlib.util.spec_from_file_location("decide", TOOLS / "decide.py")
decide = importlib.util.module_from_spec(_spec)
sys.modules["decide"] = decide
_spec.loader.exec_module(decide)
evidence = decide.evidence

MANIFEST = str(TOOLS / "manifest.toml")
ALL = evidence.scenarios(evidence.load_manifest(MANIFEST))
SHA = hashlib.sha256(pathlib.Path(MANIFEST).read_bytes()).hexdigest()
OK = {"W1": True, "W2F": True}


def costs(label, ir, mis=5000):
    events = {"Ir": ir, "Bcm": mis, "Bim": 0}
    return {"sha256": {"wheel": label, "manifest": SHA}, "costs": {s.name: events for s in ALL}}


def feas(**over):
    gating = [s.name for s in ALL if s.single_threaded]
    base = {"passed": True, "sha256": {"manifest": SHA}, "gating": gating, "mispredict": gating, "usable": True, "mispredict_available": True}
    return {**base, **over}


@pytest.fixture
def kernels(tmp_path):
    p = tmp_path / "kernels.toml"
    p.write_text('kernels = ["step"]\n')
    return str(p)


def reports(w1=980000, w2f=None, w2h=975000):
    out = {"W0": costs("w0", 1000000), "W1": costs("w1", w1)}
    if w2f is not None:
        out.update(W2F=costs("w2f", w2f), W2H=costs("w2h", w2h))
    return out


def run(r, kernels, f=None, det=OK):
    return decide.decide(r, f or feas(), det, MANIFEST, kernels)


def test_pgo_decide_w1_only_without_w2_inputs(kernels):
    out = run(reports(), kernels)
    assert out["ship"] == "W1" and "w2f_accepted" not in out and "2.0% fewer" in out["claim"]
    assert "default release wheel" in out["claim"] and "Not a time or SPS claim" in out["claim"]


@pytest.mark.parametrize(
    ("kw", "det", "ship"),
    [
        ({"w1": 985000}, OK, "W0"),
        ({"w1": 980000}, {"W1": False}, "W0"),
        ({"w1": 980000, "w2f": 950000}, OK, "W2F"),
        ({"w1": 980000, "w2f": 975000}, OK, "W1"),
        ({"w1": 980000, "w2f": 950000}, {"W1": True, "W2F": False}, "W1"),
        ({"w1": 985000, "w2f": 950000}, OK, "W0"),
    ],
)
def test_pgo_decide_follows_the_ladder(kw, det, ship, kernels):
    assert run(reports(**kw), kernels, det=det)["ship"] == ship


def test_pgo_decide_w2_needs_the_holdout_to_hold(kernels):
    assert run(reports(w2f=950000, w2h=989800), kernels)["ship"] == "W1"


def test_pgo_decide_claims_against_the_right_wheel(kernels):
    out = run(reports(w2f=931000), kernels)
    assert out["ship"] == "W2F" and "the LTO wheel" in out["claim"] and "5.0% fewer" in out["claim"]
    assert run(reports(w1=990000), kernels)["claim"].startswith("No change")


def test_pgo_decide_refuses_w2_with_an_unfilled_kernel_list(tmp_path):
    p = tmp_path / "k.toml"
    p.write_text("kernels = []\n")
    with pytest.raises(ValueError, match="nonempty"):
        run(reports(w2f=950000), str(p))
    assert run(reports(), str(p))["ship"] == "W1"


@pytest.mark.parametrize(
    ("f", "match"),
    [({"passed": False}, "feasibility"), ({"sha256": {"manifest": "x"}}, "another manifest")],
)
def test_pgo_decide_refuses_a_bad_feasibility_report(f, match, kernels):
    with pytest.raises(decide.Refused, match=match):
        run(reports(), kernels, feas(**f))


def test_pgo_decide_refuses_mismatched_costs(kernels):
    r = reports()
    r["W1"]["sha256"]["manifest"] = "x"
    with pytest.raises(decide.Refused, match="another manifest"):
        run(r, kernels)
    r = reports()
    r["W1"]["costs"].pop(ALL[0].name)
    with pytest.raises(decide.Refused, match="cover exactly"):
        run(r, kernels)
    r = reports()
    r["W1"]["sha256"]["wheel"] = "w0"
    with pytest.raises(decide.Refused, match="same wheel"):
        run(r, kernels)
    r = reports(w2f=950000)
    r.pop("W2H")
    with pytest.raises(decide.Refused, match="together"):
        run(r, kernels)


def test_pgo_decide_command_writes_the_decision_and_refuses_cleanly(tmp_path, kernels, capsys):
    paths = {}
    for name, value in (("w0", costs("w0", 1000000)), ("w1", costs("w1", 980000)), ("f", feas()), ("d", OK)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    out = tmp_path / "r.json"
    args = ["--manifest", MANIFEST, "--kernels", kernels, "--feasibility", str(paths["f"]), "--w0", str(paths["w0"]),
            "--w1", str(paths["w1"]), "--determinism", str(paths["d"]), "--out", str(out)]  # fmt: skip
    assert decide.main(args) == 0 and json.loads(out.read_text())["ship"] == "W1"
    bad = copy.deepcopy(feas(passed=False))
    paths["f"].write_text(json.dumps(bad))
    assert decide.main(args) == 2
    capsys.readouterr()


def num_tests(tmp_path, **env):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "cargo").write_text('#!/bin/sh\necho "$RUSTFLAGS|$*"\n')
    (fake / "cargo").chmod(0o755)
    environ = {k: v for k, v in os.environ.items() if not k.startswith(("WHEEL_", "RUSTFLAGS"))}
    environ.update(PATH=f"{fake}:{environ['PATH']}", **env)
    return subprocess.run([str(ROOT / "scripts" / "wheel_num_tests.sh")], capture_output=True, text=True, env=environ)


def test_pgo_num_tests_default_is_release(tmp_path):
    p = num_tests(tmp_path)
    assert p.returncode == 0 and p.stdout.strip() == "|test -p tessera-rl-num --release --all-features --locked"


def test_pgo_num_tests_use_the_wheel_profile_without_release(tmp_path):
    p = num_tests(tmp_path, WHEEL_CARGO_PROFILE="wheel")
    assert "--profile wheel" in p.stdout and "--release" not in p.stdout


def test_pgo_num_tests_apply_the_profile(tmp_path):
    p = num_tests(tmp_path, WHEEL_CARGO_PROFILE="wheel", WHEEL_PROFILE_USE="/abs/m.profdata")
    assert p.stdout.startswith("-Cprofile-use=/abs/m.profdata|")


@pytest.mark.parametrize("env", [{"WHEEL_PROFILE_USE": "/abs/m"}, {"WHEEL_CARGO_PROFILE": "wheel", "WHEEL_PROFILE_USE": "rel"}])
def test_pgo_num_tests_refuse_a_bad_profile_setup(tmp_path, env):
    assert num_tests(tmp_path, **env).returncode == 2


def test_pgo_wheel_job_runs_the_num_tests():
    assert "scripts/wheel_num_tests.sh" in (ROOT / "scripts" / "wheel_job.sh").read_text()

# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo: the workload at tiny lengths on the local wheel, and the training
and Callgrind entry points without running either. No Valgrind here."""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parents[2]
TOOLS = ROOT / "tools" / "pgo"
sys.path.insert(0, str(TOOLS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evidence = load("evidence")
workload = load("workload")
train = load("train")
callgrind = load("callgrind")
MANIFEST_PATH = str(TOOLS / "manifest.toml")
MANIFEST = evidence.load_manifest(MANIFEST_PATH)
ALL = evidence.scenarios(MANIFEST)


@pytest.mark.parametrize("scenario", ALL, ids=[s.name for s in ALL])
def test_pgo_workload_runs_every_scenario_at_a_tiny_length(scenario):
    workload.run_scenario(scenario, MANIFEST, steps=2)


def test_pgo_workload_chooses_only_legal_actions():
    rng = np.random.default_rng(1)
    masks = np.array([[[True, False, True]], [[False, True, False]]])
    for _ in range(20):
        a = workload.choose(rng, masks, (2, 1), 3)
        assert masks[0, 0, a[0, 0]] and masks[1, 0, a[1, 0]] and a.dtype == np.int32


def test_pgo_workload_actions_are_seeded():
    first = workload.choose(np.random.default_rng(1), None, (4, 2), 5)
    assert np.array_equal(first, workload.choose(np.random.default_rng(1), None, (4, 2), 5))


def test_pgo_workload_refuses_an_unknown_scenario():
    assert workload.main(["--manifest", MANIFEST_PATH, "--scenario", "nope"]) == 2


@pytest.mark.parametrize("value", ["", "relative/%p.profraw", "/abs/no-pid.profraw"])
def test_pgo_train_refuses_a_bad_profile_path(value):
    assert train.main(["--manifest", MANIFEST_PATH], environ={"LLVM_PROFILE_FILE": value}) == 2


@pytest.mark.parametrize(("flag", "count"), [([], 20), (["--holdout-build"], 17)])
def test_pgo_train_runs_the_right_set(flag, count):
    seen = []
    env = {"LLVM_PROFILE_FILE": "/abs/%m_%p.profraw"}
    assert train.main(["--manifest", MANIFEST_PATH, *flag], environ=env, run=lambda s, m, steps=None: seen.append(s)) == 0
    assert len(seen) == count and (not flag or not any(s.holdout for s in seen))


def test_pgo_callgrind_refuses_without_a_window_or_off_linux(tmp_path, capsys):
    base = ["--manifest", MANIFEST_PATH, "--out-dir", str(tmp_path / "o")]
    assert callgrind.main(base, platform="linux") == 2
    assert callgrind.main([*base, "--approved-window", "x"], platform="darwin") == 2
    assert not (tmp_path / "o").exists()
    capsys.readouterr()


def test_pgo_callgrind_command_aggregates_threads_and_simulates():
    argv = callgrind.command("py", "m", "a/b", 10, "out")
    assert argv[:2] == ["valgrind", "--tool=callgrind"]
    for flag in ("--cache-sim=yes", "--branch-sim=yes", "--separate-threads=no", "--callgrind-out-file=out"):
        assert flag in argv
    assert argv[argv.index("--steps") + 1] == "10"
    assert "--separate-threads=yes" in callgrind.command("py", "m", "a/b", 10, "out", separate_threads=True)


SAMPLE = """positions: line
events: Ir Bcm Bim I1mr D1mr D1mw ILmr DLmr DLmw
ob=(1) /x/_native.abi3.so
fn=(1) f
1 {} 1 1 0 0 0 0 0 0
"""


def test_pgo_callgrind_cost_is_the_difference_of_two_lengths(tmp_path):
    s = ALL[0]
    irs = {s.steps: 1500, 2 * s.steps: 2500}
    calls = []

    def runner(argv, check):
        assert check is True
        steps = int(argv[argv.index("--steps") + 1])
        calls.append(steps)
        out = next(a.split("=", 1)[1] for a in argv if a.startswith("--callgrind-out-file="))
        pathlib.Path(out).write_text(SAMPLE.format(irs[steps]))

    cost = callgrind.cost_of("py", "m", s, tmp_path, runner=runner)
    assert calls == [s.steps, 2 * s.steps] and cost.ir == 1000


def test_pgo_workload_control_and_worker_override_run():
    assert callable(workload.control) and callgrind.CONTROL == workload.CONTROL
    args = ["--manifest", MANIFEST_PATH, "--scenario", "threaded/CartPole-v1", "--steps", "2", "--workers", "0"]
    assert workload.main(args) == 0


def test_pgo_thread_files_sort_numerically(tmp_path):
    out = tmp_path / "x.callgrind"
    for suffix in ("1", "10", "2", "notes"):
        pathlib.Path(f"{out}-{suffix}").write_text("")
    assert [p.name.rsplit("-", 1)[1] for p in callgrind.thread_files(out)] == ["1", "2", "10"]


THREAD = "positions: line\nevents: Ir Bcm Bim I1mr D1mr D1mw ILmr DLmr DLmw\nob=(1) /x/_native.so\n1 {} 6000 0 0 0 0 0 0 0\ntotals: 1\n"


class FakeValgrind:
    """Writes Callgrind files as a function of the command line."""

    def __init__(self, control_ir=0, per_step=100000, caller_only=False, cost=None, inline_factor=1.0):
        self.calls, self.control_ir, self.per_step, self.caller_only = [], control_ir, per_step, caller_only
        self.cost, self.inline_factor = cost, inline_factor

    def __call__(self, argv, check):
        out = next(a.split("=", 1)[1] for a in argv if a.startswith("--callgrind-out-file="))
        scenario = argv[argv.index("--scenario") + 1]
        steps = int(argv[argv.index("--steps") + 1])
        self.calls.append(argv)
        if scenario == callgrind.CONTROL:
            pathlib.Path(out).write_text(THREAD.format(5000000 + self.control_ir * steps))
        elif "--separate-threads=yes" in argv:
            pathlib.Path(f"{out}-1").write_text(THREAD.format(1000))
            pathlib.Path(f"{out}-2").write_text(THREAD.format(0 if self.caller_only else 5000))
        else:
            ir = 1000 + steps * self.per_step
            if "--workers" in argv and argv[argv.index("--workers") + 1] == "0":
                ir = int(1000 + steps * self.per_step * self.inline_factor)
            if self.cost:
                ir = self.cost(scenario, steps, out, ir)
            pathlib.Path(out).write_text(THREAD.format(ir))


def gate(tmp_path, **kw):
    runner = FakeValgrind(**kw)
    return callgrind.feasibility("py", MANIFEST_PATH, MANIFEST, tmp_path, runner), runner


def test_pgo_feasibility_passes_a_clean_fake(tmp_path):
    report, runner = gate(tmp_path)
    assert report["passed"] is True and report["usable"] is True and len(report["gating"]) == 16
    entry = report["scenarios"]["split-phase/CartPole-v1"]
    assert entry["workers"]["kind"] == "per-thread" and entry["workers"]["share"] == pytest.approx(5 / 6)
    assert report["scenarios"]["threaded/CartPole-v1"]["workers"]["kind"] == "against-inline"
    assert report["scenarios"]["state/CartPole-v1"]["linear"] is True
    assert len(report["scenarios"]["state/CartPole-v1"]["aa_ir"]) == 5


def test_pgo_feasibility_single_threaded_scenario_costs_eleven_runs(tmp_path):
    _, runner = gate(tmp_path)
    runs = [a for a in runner.calls if a[a.index("--scenario") + 1] == "state/CartPole-v1"]
    assert len(runs) == 11


def test_pgo_feasibility_control_is_by_difference_so_import_cancels(tmp_path):
    report, runner = gate(tmp_path)
    assert report["control_native_ir"] == 0 and report["passed"] is True
    steps = [int(a[a.index("--steps") + 1]) for a in runner.calls if a[a.index("--scenario") + 1] == callgrind.CONTROL]
    assert sorted(steps) == [1, 2]


def test_pgo_feasibility_fails_when_the_interpreter_leaks(tmp_path):
    report, _ = gate(tmp_path, control_ir=10**12)
    assert report["passed"] is False and report["usable"] is True
    assert report["scenarios"]["state/CartPole-v1"]["interpreter_excluded"] is False


def test_pgo_feasibility_fails_without_worker_thread_cost(tmp_path):
    report, _ = gate(tmp_path, caller_only=True)
    assert report["passed"] is False and report["scenarios"]["split-phase/CartPole-v1"]["workers"]["passed"] is False


def test_pgo_feasibility_fails_on_nonlinearity_alone(tmp_path):
    name = "state/CartPole-v1"
    report, _ = gate(tmp_path, cost=lambda s, steps, out, ir: ir + (10**9 if s == name and steps == 3 * ALL[0].steps else 0))
    assert report["scenarios"][name]["linear"] is False and report["usable"] is True and report["passed"] is False


def test_pgo_feasibility_fails_on_an_unusable_set_alone(tmp_path):
    name = "holdout/Go9-v1"
    go9 = next(s for s in ALL if s.name == name)

    def cost(scenario, steps, out, ir):
        return ir + (10**8 if scenario == name and ".aa3" in out and steps == 2 * go9.steps else 0)

    report, _ = gate(tmp_path, cost=cost)
    assert name not in report["gating"] and report["usable"] is False and report["passed"] is False
    assert all(e.get("linear", True) and e["interpreter_excluded"] for e in report["scenarios"].values())
    assert all(e["workers"]["passed"] for e in report["scenarios"].values() if "workers" in e)


def test_pgo_feasibility_inline_control_really_runs_at_zero_workers(tmp_path):
    report, runner = gate(tmp_path)
    inline = [a for a in runner.calls if any(".inline." in x for x in a)]
    assert inline and all(a[a.index("--workers") + 1] == "0" for a in inline)
    assert report["scenarios"]["threaded/CartPole-v1"]["workers"]["passed"] is True


def test_pgo_feasibility_fails_when_workers_lose_cost_against_inline(tmp_path):
    report, _ = gate(tmp_path, inline_factor=1.2)
    assert report["scenarios"]["threaded/CartPole-v1"]["workers"]["passed"] is False and report["passed"] is False


def test_pgo_feasibility_resumes_from_finished_files(tmp_path):
    gate(tmp_path)
    again = FakeValgrind()
    second = callgrind.feasibility("py", MANIFEST_PATH, MANIFEST, tmp_path, again)
    assert second["passed"] is True and len(second["reused_files"]) > 100
    assert not [a for a in again.calls if "--separate-threads=no" in a]


def test_pgo_measure_reruns_an_unfinished_file(tmp_path):
    s = ALL[0]
    out = tmp_path / f"{s.name.replace('/', '_')}.{s.steps}.callgrind"
    out.write_text(THREAD.format(7).replace("totals: 1\n", ""))
    runner = FakeValgrind()
    reused = []
    callgrind.measure("py", "m", s.name, s.steps, tmp_path, runner, reused=reused)
    assert len(runner.calls) == 1 and reused == []
    callgrind.measure("py", "m", s.name, s.steps, tmp_path, runner, reused=reused)
    assert len(runner.calls) == 1 and reused == [out.name]


def test_pgo_feasibility_records_hashes_and_needs_the_wheel(tmp_path, monkeypatch):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    monkeypatch.setattr(callgrind, "feasibility", lambda *a: {"passed": True})
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    out = tmp_path / "o"
    base = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility"]
    assert callgrind.main(base, platform="linux") == 2
    assert callgrind.main([*base, "--wheel", str(wheel)], platform="linux") == 0
    saved = json.loads((out / "feasibility.json").read_text())
    assert saved["sha256"]["wheel"] == hashlib.sha256(b"abc").hexdigest()
    assert set(saved["sha256"]) == {"wheel", "manifest", "python", "valgrind", "cpu_model", "cpu_flags"}, \
        "the full stamp, without the kernel list"
    assert len(saved["kernels_sha256"]) == 64


def test_pgo_checks_writes_statuses_tied_to_the_wheel(tmp_path):
    checks = load("checks")
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    out = tmp_path / "c.json"
    assert checks.main(["--wheel", str(wheel), "--out", str(out), "python_suite=0", "num_tests=3"]) == 0
    saved = json.loads(out.read_text())
    assert saved["wheel_sha256"] == hashlib.sha256(b"abc").hexdigest() and saved["checks"]["num_tests"] == 3
    assert checks.main(["--wheel", str(wheel), "--out", str(out), "bad"]) == 2



def test_pgo_measure_only_runs_the_named_scenarios(tmp_path):
    runner = FakeValgrind()
    part = callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, tmp_path, ["state/Tag-v1-2-2", "state/CartPole-v1"], runner)
    assert [u.split("#")[0] for u in part["units"]][::6] == ["state/CartPole-v1", "state/Tag-v1-2-2"], "manifest order"
    assert len(part["units"]) == 12
    ran = {a[a.index("--scenario") + 1] for a in runner.calls}
    assert ran == {"state/CartPole-v1", "state/Tag-v1-2-2"}, "no control unless named"
    assert len(runner.calls) == 22


def test_pgo_select_refuses_unknown_and_repeated_names():
    with pytest.raises(ValueError, match="unknown"):
        callgrind.select(MANIFEST, ["state/Nope-v1"])
    with pytest.raises(ValueError, match="twice"):
        callgrind.select(MANIFEST, [callgrind.CONTROL, callgrind.CONTROL])


def fake_environment(toolchain="vg-1", **override):
    return lambda args: {**callgrind.hashes(args), "python": "py-sha", "valgrind": toolchain,
                         "cpu_model": "Some CPU 12", "cpu_flags": "flags-a", **override}


def old_stamp(out, files, wheel_path, **override):
    """Stamps written before the kernel list left the stamp: the same fields,
    plus `kernels`, as the gate on swarmer03 wrote them."""
    st = {"wheel": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
          "manifest": hashlib.sha256(pathlib.Path(MANIFEST_PATH).read_bytes()).hexdigest(),
          "kernels": "k" * 64, "python": "py-sha", "valgrind": "vg-1",
          "cpu_model": "Some CPU 12", "cpu_flags": "flags-a", **override}
    for f in files:
        callgrind.stamp_path(f).write_text(json.dumps(st, sort_keys=True))


def run_parts(tmp_path, groups, toolchain="vg-1", runner=None):
    wheel = tmp_path / "w.whl"
    if not wheel.exists():
        wheel.write_bytes(b"abc")
    out = tmp_path / "o"
    runner = runner or FakeValgrind()
    for names in groups:
        args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility",
                "--wheel", str(wheel)]
        for n in names:
            args += ["--scenario", n]
        with pytest.MonkeyPatch.context() as m:
            m.setattr(callgrind.subprocess, "run", runner)
            m.setattr(callgrind, "environment", fake_environment(toolchain))
            assert callgrind.main(args, platform="linux") == 0
    assert callgrind.STAMP is None, "main() leaves no stamp behind"
    return out, wheel, runner


def merge_now(out, wheel):
    return callgrind.main(["--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility", "--merge",
                           "--wheel", str(wheel)], platform="darwin")


def parts_then_merge(tmp_path, groups, drop=None):
    out, wheel, runner = run_parts(tmp_path, groups)
    if drop:
        for f in out.glob(drop):
            f.unlink()
    return merge_now(out, wheel), out, runner


def all_names():
    return [callgrind.CONTROL, *(s.name for s in ALL)]


def test_pgo_merge_of_disjoint_parts_equals_one_full_gate(tmp_path):
    names = all_names()
    merged, out, _ = parts_then_merge(tmp_path, [names[0::2], names[1::2]])
    assert merged == 0
    report = json.loads((out / "feasibility.json").read_text())
    (tmp_path / "full").mkdir()
    full, _ = gate(tmp_path / "full")
    assert report["passed"] is True and report["gating"] == full["gating"]
    assert {k: v["aa_ir"] for k, v in report["scenarios"].items()} == {k: v["aa_ir"] for k, v in full["scenarios"].items()}
    assert len(report["merged_parts"]) == 2


def test_pgo_merge_never_runs_valgrind_and_refuses_a_missing_file(tmp_path):
    names = all_names()
    merged, out, _ = parts_then_merge(tmp_path, [names], drop="state_Tag-v1-2-2.4000.aa3.callgrind")
    assert merged == 3 and not (out / "feasibility.json").exists()


def test_pgo_merge_refuses_parts_with_other_hashes(tmp_path):
    names = all_names()
    merged, out, _ = parts_then_merge(tmp_path, [names])
    assert merged == 0
    (tmp_path / "w.whl").write_bytes(b"other")
    again = callgrind.main(["--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility", "--merge",
                            "--wheel", str(tmp_path / "w.whl")], platform="darwin")
    assert again == 2


def test_pgo_merge_refuses_overlapping_parts(tmp_path):
    names = all_names()
    merged, _, _ = parts_then_merge(tmp_path, [names[:3], names[2:]])
    assert merged == 2


def test_pgo_part_runs_need_the_window_and_linux(tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    base = ["--manifest", MANIFEST_PATH, "--out-dir", str(tmp_path / "o"), "--feasibility", "--wheel", str(wheel),
            "--scenario", "state/CartPole-v1"]
    assert callgrind.main(base, platform="linux") == 2
    assert callgrind.main(["--approved-window", "x", *base], platform="darwin") == 2
    assert callgrind.main(["--approved-window", "x", *base[:-2], "--scenario", "nope"], platform="linux") == 2


def test_pgo_feasibility_reuses_finished_per_thread_files(tmp_path):
    gate(tmp_path)
    again = FakeValgrind()
    second = callgrind.feasibility("py", MANIFEST_PATH, MANIFEST, tmp_path, again)
    assert second["passed"] is True and again.calls == []


def test_pgo_probe_runs_one_reusable_n_step_file_per_scenario(tmp_path):
    runner = FakeValgrind()
    found = callgrind.probe("py", MANIFEST_PATH, MANIFEST, tmp_path, [callgrind.CONTROL, "state/Grid-v1"], runner)
    assert list(found["probe"]) == ["state/Grid-v1"] and len(runner.calls) == 1
    again = FakeValgrind()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, tmp_path, ["state/Grid-v1"], again)
    assert len(again.calls) == 10, "the probe's file is the first run of the first A/A pair"


def test_pgo_probe_needs_scenarios(tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(tmp_path), "--feasibility",
            "--wheel", str(wheel), "--probe"]
    assert callgrind.main(args, platform="linux") == 2


def test_pgo_merge_refuses_parts_that_do_not_cover_every_scenario(tmp_path):
    names = all_names()
    # A finished file for the uncovered scenario is in the out dir (an earlier
    # window's, say): the merge still refuses, it never adopts it silently.
    out, wheel, _ = run_parts(tmp_path, [names[:-1]])
    stale = callgrind.STAMP
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out, [names[-1]], FakeValgrind())
    assert stale is None
    assert merge_now(out, wheel) == 2 and not (out / "feasibility.json").exists()


def test_pgo_merge_refuses_a_file_without_the_parts_provenance(tmp_path):
    names = all_names()
    out, wheel, _ = run_parts(tmp_path, [names])
    side = callgrind.stamp_path(out / "state_Grid-v1.2000.aa2.callgrind")
    side.write_text(json.dumps({"other": "window"}))
    assert merge_now(out, wheel) == 3 and not (out / "feasibility.json").exists()


def test_pgo_merge_refuses_parts_from_two_toolchains(tmp_path):
    names = all_names()
    out, wheel, _ = run_parts(tmp_path, [names[:5]], toolchain="vg-1")
    run_parts(tmp_path, [names[5:]], toolchain="vg-2")
    assert merge_now(out, wheel) == 2


def test_pgo_merge_with_no_part_reports_refuses(tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    out = tmp_path / "o"
    out.mkdir()
    assert merge_now(out, wheel) == 2 and not (out / "feasibility.json").exists()


def test_pgo_a_part_reruns_a_file_measured_with_another_toolchain(tmp_path):
    out, _, _ = run_parts(tmp_path, [["state/Grid-v1"]], toolchain="vg-1")
    again = FakeValgrind()
    run_parts(tmp_path, [["state/Grid-v1"]], toolchain="vg-2", runner=again)
    assert len(again.calls) == 11


def test_pgo_unfinished_per_thread_files_are_rerun_without_leftovers(tmp_path):
    s = next(x for x in ALL if x.name == "split-phase/CartPole-v1")
    out = tmp_path / f"{s.name.replace('/', '_')}.threads.callgrind"
    pathlib.Path(f"{out}-1").write_text(THREAD.format(1000))
    pathlib.Path(f"{out}-2").write_text(THREAD.format(5000).replace("totals: 1\n", ""))
    pathlib.Path(f"{out}-3").write_text(THREAD.format(10**9))
    runner = FakeValgrind()
    result = callgrind.worker_control("py", MANIFEST_PATH, s, tmp_path, runner)
    assert len(runner.calls) == 1 and not pathlib.Path(f"{out}-3").exists()
    assert result["non_caller_ir"] == 5000


def adopt_args(out, wheel, listing):
    return ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility",
            "--wheel", str(wheel), "--adopt", str(listing)]


def test_pgo_adopt_stamps_listed_files_after_checking_them(tmp_path, monkeypatch):
    out = tmp_path / "o"
    out.mkdir()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out, ["state/CartPole-v1"], FakeValgrind())
    files = sorted(out.glob("*.callgrind"))
    listing = tmp_path / "KEPT.txt"
    listing.write_text("# header\n" + "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()} {f.name} Ir=1\n" for f in files))
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    old_stamp(out, files, wheel)
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    assert callgrind.main(adopt_args(out, wheel, listing), platform="linux") == 0
    assert all(callgrind.stamp_path(f).exists() for f in files)
    # A part with the same toolchain now reuses them: no Valgrind run.
    again = FakeValgrind()
    run_parts(tmp_path, [["state/CartPole-v1"]], runner=again)
    assert again.calls == []


def test_pgo_adopt_is_all_or_nothing(tmp_path, monkeypatch):
    out = tmp_path / "o"
    out.mkdir()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out, ["state/CartPole-v1"], FakeValgrind())
    files = sorted(out.glob("*.callgrind"))
    listing = tmp_path / "KEPT.txt"
    listing.write_text("".join(f"{'0' * 64 if i == 3 else hashlib.sha256(f.read_bytes()).hexdigest()} {f.name}\n"
                               for i, f in enumerate(files)))
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    assert callgrind.main(adopt_args(out, wheel, listing), platform="linux") == 2
    assert not any(callgrind.stamp_path(f).exists() for f in files)


def test_pgo_per_thread_reuse_needs_the_stamp(tmp_path, monkeypatch):
    s = next(x for x in ALL if x.name == "split-phase/CartPole-v1")
    out = tmp_path / f"{s.name.replace('/', '_')}.threads.callgrind"
    pathlib.Path(f"{out}-1").write_text(THREAD.format(1000))
    pathlib.Path(f"{out}-2").write_text(THREAD.format(5000))
    monkeypatch.setattr(callgrind, "STAMP", {"toolchain": "now"})
    # Unstamped: rerun under a part, missing under a merge.
    with pytest.raises(callgrind.MissingRun):
        callgrind.worker_control("py", MANIFEST_PATH, s, tmp_path, callgrind.refuse)
    callgrind.stamp_path(out).write_text(json.dumps({"toolchain": "other"}))
    runner = FakeValgrind()
    callgrind.worker_control("py", MANIFEST_PATH, s, tmp_path, runner)
    assert len(runner.calls) == 1
    again = FakeValgrind()
    callgrind.worker_control("py", MANIFEST_PATH, s, tmp_path, again)
    assert again.calls == [], "the rerun's own stamp makes it reusable"


def test_pgo_a_failed_rerun_leaves_no_stamp_on_the_old_file(tmp_path, monkeypatch):
    s = ALL[0]
    out = tmp_path / f"{s.name.replace('/', '_')}.{s.steps}.callgrind"
    out.write_text(THREAD.format(7))
    callgrind.stamp_path(out).write_text(json.dumps({"toolchain": "other"}))
    monkeypatch.setattr(callgrind, "STAMP", {"toolchain": "now"})

    def dies(argv, check):
        raise subprocess.CalledProcessError(1, argv)

    with pytest.raises(subprocess.CalledProcessError):
        callgrind.measure("py", "m", s.name, s.steps, tmp_path, dies)
    assert not callgrind.stamp_path(out).exists()
    with pytest.raises(callgrind.MissingRun):
        callgrind.measure("py", "m", s.name, s.steps, tmp_path, callgrind.refuse)


def test_pgo_select_expands_scenarios_into_units_and_refuses_repeats():
    units = callgrind.select(MANIFEST, ["threaded/CartPole-v1", callgrind.CONTROL])
    assert units[0] == callgrind.CONTROL
    assert units[1:] == [f"threaded/CartPole-v1#aa{i}" for i in range(5)] + ["threaded/CartPole-v1#workers"]
    assert callgrind.select(MANIFEST, ["state/Grid-v1#lin"]) == ["state/Grid-v1#lin"]
    for bad in (["state/Grid-v1#workers"], ["threaded/CartPole-v1#lin"], ["state/Grid-v1#aa5"]):
        with pytest.raises(ValueError, match="unknown"):
            callgrind.select(MANIFEST, bad)
    with pytest.raises(ValueError, match="twice"):
        callgrind.select(MANIFEST, ["state/Grid-v1", "state/Grid-v1#aa2"])


def test_pgo_units_cover_the_full_gate_runs_exactly(tmp_path):
    (tmp_path / "f").mkdir()
    full = FakeValgrind()
    callgrind.feasibility("py", MANIFEST_PATH, MANIFEST, tmp_path / "f", full)
    split = FakeValgrind()
    (tmp_path / "u").mkdir()
    for unit in callgrind.all_units(MANIFEST):
        callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, tmp_path / "u", [unit], split)
    def outs(r):
        # pathlib, not split("/"): the out paths use the platform's separator.
        return sorted(pathlib.Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--callgrind-out-file="))).name
                      for argv in r.calls)
    assert outs(split) == outs(full), "the units run exactly the files of the full gate"


def test_pgo_merge_of_parts_split_by_pairs_equals_one_full_gate(tmp_path):
    units = callgrind.all_units(MANIFEST)
    groups = [units[i::4] for i in range(4)]
    merged, out, _ = parts_then_merge(tmp_path, groups)
    assert merged == 0
    report = json.loads((out / "feasibility.json").read_text())
    (tmp_path / "full").mkdir()
    full, _ = gate(tmp_path / "full")
    assert report["passed"] is True and report["gating"] == full["gating"]
    assert {k: v["aa_ir"] for k, v in report["scenarios"].items()} == {k: v["aa_ir"] for k, v in full["scenarios"].items()}


def test_pgo_merge_refuses_a_missing_or_repeated_unit(tmp_path):
    units = callgrind.all_units(MANIFEST)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    missing, _, _ = parts_then_merge(tmp_path / "a", [[u for u in units if u != "games/Othello-v1#aa3"]])
    assert missing == 2
    repeated, _, _ = parts_then_merge(tmp_path / "b", [units, ["games/Othello-v1#aa3"]])
    assert repeated == 2


def test_pgo_probe_takes_scenarios_not_units(tmp_path):
    with pytest.raises(ValueError, match="not units"):
        callgrind.probe("py", MANIFEST_PATH, MANIFEST, tmp_path, ["state/Grid-v1#aa0"], FakeValgrind())


def test_pgo_merge_refuses_a_part_report_of_the_older_format(tmp_path):
    out, wheel, _ = run_parts(tmp_path, [callgrind.all_units(MANIFEST)])
    part = next(out.glob("feasibility.part.*.json"))
    old = json.loads(part.read_text())
    old["scenarios"] = old.pop("units")
    part.write_text(json.dumps(old))
    assert merge_now(out, wheel) == 2 and not (out / "feasibility.json").exists()


def test_pgo_cpu_reads_the_model_and_hashes_the_flags_order_free(tmp_path):
    a = tmp_path / "a"
    a.write_text("processor\t: 0\nmodel name\t: Some CPU 12\nflags\t\t: sse2 avx2 fma\n\n"
                 "processor\t: 1\nmodel name\t: Some CPU 12\nflags\t\t: sse2 avx2 fma\n")
    b = tmp_path / "b"
    b.write_text("model name\t: Some CPU 12\nflags\t\t: fma sse2 avx2\n")
    c = tmp_path / "c"
    c.write_text("model name\t: Some CPU 12\nflags\t\t: sse2 avx2\n")
    assert callgrind.cpu(str(a)) == callgrind.cpu(str(b))
    assert callgrind.cpu(str(a))["cpu_model"] == "Some CPU 12"
    assert callgrind.cpu(str(a))["cpu_flags"] != callgrind.cpu(str(c))["cpu_flags"], "a missing feature is another host"


def test_pgo_cpu_refuses_a_cpuinfo_without_model_or_flags(tmp_path):
    arm = tmp_path / "arm"
    arm.write_text("processor\t: 0\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\n")
    with pytest.raises(ValueError, match="cannot be stamped"):
        callgrind.cpu(str(arm))


def test_pgo_environment_stamps_the_cpu(tmp_path, monkeypatch):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    info = tmp_path / "cpuinfo"
    info.write_text("model name\t: Some CPU 12\nflags\t\t: sse2 avx2\n")

    class Done:
        stdout = "valgrind-3.24.0\n"

    monkeypatch.setattr(callgrind.subprocess, "run", lambda *a, **k: Done())
    args = argparse.Namespace(wheel=str(wheel), manifest=MANIFEST_PATH, python=sys.executable)
    env = callgrind.environment(args, cpuinfo=str(info))
    assert env["cpu_model"] == "Some CPU 12" and env["cpu_flags"] == callgrind.cpu(str(info))["cpu_flags"]
    assert env["valgrind"] == "valgrind-3.24.0"


def test_pgo_a_part_reruns_a_file_measured_on_another_cpu(tmp_path, monkeypatch):
    def env(flags):
        return lambda args: {**callgrind.hashes(args), "python": "py-sha", "valgrind": "vg-1",
                             "cpu_model": "Some CPU 12", "cpu_flags": flags}

    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    out = tmp_path / "o"
    args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility",
            "--wheel", str(wheel), "--scenario", "state/Grid-v1"]
    first = FakeValgrind()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", first)
        m.setattr(callgrind, "environment", env("flags-a"))
        assert callgrind.main(args, platform="linux") == 0
    again = FakeValgrind()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", again)
        m.setattr(callgrind, "environment", env("flags-b"))
        assert callgrind.main(args, platform="linux") == 0
    assert len(again.calls) == len(first.calls) == 11


def test_pgo_a_part_on_an_unstampable_host_is_refused(tmp_path, monkeypatch):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")

    def no_cpu(args):
        raise ValueError("no 'model name' or 'flags' line; the host cannot be stamped")

    monkeypatch.setattr(callgrind, "environment", no_cpu)
    args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(tmp_path / "o"), "--feasibility",
            "--wheel", str(wheel), "--scenario", "state/Grid-v1"]
    assert callgrind.main(args, platform="linux") == 2 and callgrind.STAMP is None


def test_pgo_a_part_times_each_unit(tmp_path, capsys):
    part = callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, tmp_path, ["state/Grid-v1"], FakeValgrind())
    assert list(part["unit_seconds"]) == part["units"]
    assert all(s >= 0 for s in part["unit_seconds"].values())
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("unit ")]
    assert [l.split()[1] for l in lines] == part["units"]


def gate_files(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out, ["state/CartPole-v1"], FakeValgrind())
    files = sorted(out.glob("*.callgrind"))
    listing = tmp_path / "KEPT.txt"
    listing.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()} {f.name}\n" for f in files))
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    return out, files, listing, wheel


def test_pgo_an_old_stamp_file_is_not_reused_without_adopt(tmp_path):
    out, files, _, wheel = gate_files(tmp_path)
    old_stamp(out, files, wheel)
    again = FakeValgrind()
    run_parts(tmp_path, [["state/CartPole-v1"]], runner=again)
    assert len(again.calls) == 11, "a stamp with the kernels key never matches silently"


@pytest.mark.parametrize("field", ["wheel", "python", "valgrind", "cpu_model", "cpu_flags"])
def test_pgo_adopt_refuses_an_old_stamp_that_differs_in_any_other_field(tmp_path, monkeypatch, field):
    out, files, listing, wheel = gate_files(tmp_path)
    old_stamp(out, files, wheel, **{field: "other"})
    before = {f: callgrind.stamp_path(f).read_text() for f in files}
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    assert callgrind.main(adopt_args(out, wheel, listing), platform="linux") == 2
    assert {f: callgrind.stamp_path(f).read_text() for f in files} == before, "nothing restamped"


def test_pgo_adopt_refuses_a_file_without_a_stamp(tmp_path, monkeypatch):
    out, files, listing, wheel = gate_files(tmp_path)
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    assert callgrind.main(adopt_args(out, wheel, listing), platform="linux") == 2


def test_pgo_merge_of_adopted_and_new_files_needs_equal_reduced_stamps(tmp_path, monkeypatch):
    names = all_names()
    adopted = [callgrind.CONTROL, "state/CartPole-v1"]
    out = tmp_path / "o"
    out.mkdir()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out, adopted, FakeValgrind())
    files = sorted(out.glob("*.callgrind"))
    listing = tmp_path / "KEPT.txt"
    listing.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()} {f.name}\n" for f in files))
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    old_stamp(out, files, wheel)
    monkeypatch.setattr(callgrind, "environment", fake_environment())
    assert callgrind.main(adopt_args(out, wheel, listing), platform="linux") == 0
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind, "environment", fake_environment())
        run_parts(tmp_path, [adopted, [n for n in names if n not in adopted]])
    assert merge_now(out, wheel) == 0
    # The same, but the new part on other CPU flags: the merge refuses.
    out2 = tmp_path / "p"
    out2.mkdir()
    callgrind.measure_only("py", MANIFEST_PATH, MANIFEST, out2, adopted, FakeValgrind())
    files2 = sorted(out2.glob("*.callgrind"))
    listing2 = tmp_path / "KEPT2.txt"
    listing2.write_text("".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()} {f.name}\n" for f in files2))
    old_stamp(out2, files2, wheel)
    assert callgrind.main(adopt_args(out2, wheel, listing2), platform="linux") == 0
    args = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out2), "--feasibility",
            "--wheel", str(wheel)]
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", FakeValgrind())
        m.setattr(callgrind, "environment", fake_environment())
        assert callgrind.main([*args, *sum((["--scenario", n] for n in adopted), [])], platform="linux") == 0
        m.setattr(callgrind, "environment", fake_environment(cpu_flags="flags-b"))
        rest = [n for n in names if n not in adopted]
        assert callgrind.main([*args, *sum((["--scenario", n] for n in rest), [])], platform="linux") == 0
    assert merge_now(out2, wheel) == 2


def plain_costs(tmp_path, wheel_bytes, env):
    """costs.json of one wheel through callgrind.py's real plain mode."""
    wheel = tmp_path / f"{wheel_bytes.decode()}.whl"
    wheel.write_bytes(wheel_bytes)
    out = tmp_path / f"c-{wheel_bytes.decode()}"
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", FakeValgrind())
        m.setattr(callgrind, "environment", env)
        assert callgrind.main(["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out),
                               "--wheel", str(wheel)], platform="linux") == 0
    return json.loads((out / "costs.json").read_text()), hashlib.sha256(wheel_bytes).hexdigest()


def decide_on_real_outputs(tmp_path, w0_env, w1_env):
    decide = load("decide")
    # The gate: parts and merge on the W1 bytes ("abc", run_parts' wheel), under fake_environment().
    out, wheel, _ = run_parts(tmp_path, [all_names()])
    assert merge_now(out, wheel) == 0
    feas = json.loads((out / "feasibility.json").read_text())
    w0, w0_sha = plain_costs(tmp_path / "w0", b"w0", w0_env)
    w1, w1_sha = plain_costs(tmp_path / "w1", b"abc", w1_env)
    seventh = {"W1": {"checks": {"wheel_sha256": w1_sha, "checks": {"python_suite": 0, "num_tests": 0}},
                      "compare": {"passed": True, "wheels": [w0_sha, w1_sha]}}}
    kernels = str(TOOLS / "kernels.toml")
    return decide.decide({"W0": w0, "W1": w1}, feas, seventh, MANIFEST_PATH, kernels), decide


def test_pgo_decide_accepts_real_outputs_under_one_stamp(tmp_path):
    for d in ("w0", "w1"):
        (tmp_path / d).mkdir()
    result, _ = decide_on_real_outputs(tmp_path, fake_environment(), fake_environment())
    assert "ship" in result


@pytest.mark.parametrize("override", [{"cpu_flags": "flags-b"}, {"cpu_model": "Other CPU"}, {"toolchain": "vg-2"}])
def test_pgo_decide_refuses_real_outputs_from_another_host_or_toolchain(tmp_path, override):
    for d in ("w0", "w1"):
        (tmp_path / d).mkdir()
    with pytest.raises(Exception, match="another stamp"):
        decide_on_real_outputs(tmp_path, fake_environment(**override), fake_environment())


def costs_args(out, wheel, *extra):
    return ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--costs",
            "--wheel", str(wheel), *extra]


def costs_merge(out, wheel):
    return callgrind.main(["--manifest", MANIFEST_PATH, "--out-dir", str(out), "--costs", "--merge",
                           "--wheel", str(wheel)], platform="darwin")


def run_costs_parts(out, wheel, groups, runner=None, env=None):
    for names in groups:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(callgrind.subprocess, "run", runner or FakeValgrind())
            m.setattr(callgrind, "environment", env or fake_environment())
            args = costs_args(out, wheel, *sum((["--pair", n] for n in names), []))
            assert callgrind.main(args, platform="linux") == 0


def test_pgo_costs_from_a_gates_own_parts(tmp_path):
    out, wheel, _ = run_parts(tmp_path, [all_names()])
    assert costs_merge(out, wheel) == 0, "W1's costs come from the gate's files, no new run"
    report = json.loads((out / "costs.json").read_text())
    assert set(report["costs"]) == {s.name for s in ALL}
    assert set(report["sha256"]) == {"wheel", "manifest", "python", "valgrind", "cpu_model", "cpu_flags"}
    assert all(len(v) == 5 for v in report["aa_ir"].values())


def test_pgo_costs_parts_merge_to_the_median_of_five_pairs(tmp_path):
    name = "state/Grid-v1"
    grid = next(s for s in ALL if s.name == name)
    bump = {".aa0": 500, ".aa1": 10, ".aa2": 30, ".aa3": 40, ".aa4": 20}

    def cost(scenario, steps, out, ir):
        tag = next((t for t in bump if t + ".callgrind" in out), None)
        return ir + (bump[tag] if scenario == name and tag and steps == 2 * grid.steps else 0)

    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    units = callgrind.aa_units(MANIFEST)
    run_costs_parts(out, wheel, [units[0::2], units[1::2]], runner=FakeValgrind(cost=cost))
    assert costs_merge(out, wheel) == 0
    report = json.loads((out / "costs.json").read_text())
    base = min(report["aa_ir"][name]) - 10
    assert report["costs"][name]["Ir"] == base + 30, "the median of 10, 20, 30, 40, 500, not their mean"


def test_pgo_costs_refuse_non_aa_units(tmp_path):
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"w")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind, "environment", fake_environment())
        assert callgrind.main(costs_args(tmp_path / "o", wheel, "--pair", "state/Grid-v1#lin"), platform="linux") == 2
        assert callgrind.main(costs_args(tmp_path / "o", wheel, "--pair", callgrind.CONTROL), platform="linux") == 2


def test_pgo_costs_merge_refuses_a_missing_unit_or_file(tmp_path):
    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    units = callgrind.aa_units(MANIFEST)
    run_costs_parts(out, wheel, [units[:-1]])
    assert costs_merge(out, wheel) == 2 and not (out / "costs.json").exists()
    run_costs_parts(out, wheel, [units[-1:]])
    next(out.glob("state_Tag-v1-2-2.4000.aa3.callgrind")).unlink()
    assert costs_merge(out, wheel) == 3 and not (out / "costs.json").exists()


def test_pgo_decide_runs_on_costs_and_gate_from_the_real_pipeline(tmp_path):
    decide = load("decide")
    out, w1_wheel, _ = run_parts(tmp_path, [all_names()])
    assert merge_now(out, w1_wheel) == 0 and costs_merge(out, w1_wheel) == 0
    w0_out = tmp_path / "w0"
    w0_wheel = tmp_path / "w0.whl"
    w0_wheel.write_bytes(b"w0")
    units = callgrind.aa_units(MANIFEST)
    run_costs_parts(w0_out, w0_wheel, [units[0::3], units[1::3], units[2::3]])
    assert costs_merge(w0_out, w0_wheel) == 0
    feas = json.loads((out / "feasibility.json").read_text())
    w1 = json.loads((out / "costs.json").read_text())
    w0 = json.loads((w0_out / "costs.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    seventh = {"W1": {"checks": {"wheel_sha256": sha(w1_wheel), "checks": {"python_suite": 0, "num_tests": 0}},
                      "compare": {"passed": True, "wheels": [sha(w0_wheel), sha(w1_wheel)]}}}
    result = decide.decide({"W0": w0, "W1": w1}, feas, seventh, MANIFEST_PATH, str(TOOLS / "kernels.toml"))
    assert result["ship"] in ("W0", "W1")
    # W0 measured on another CPU: refused.
    w0_other = tmp_path / "w0b"
    run_costs_parts(w0_other, w0_wheel, [units], env=fake_environment(cpu_flags="flags-b"))
    assert costs_merge(w0_other, w0_wheel) == 0
    with pytest.raises(decide.Refused, match="another stamp"):
        decide.decide({"W0": json.loads((w0_other / "costs.json").read_text()), "W1": w1}, feas, seventh,
                      MANIFEST_PATH, str(TOOLS / "kernels.toml"))


def test_pgo_costs_probe_on_a_wheel_is_reused_by_its_parts(tmp_path):
    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(callgrind.subprocess, "run", FakeValgrind())
        m.setattr(callgrind, "environment", fake_environment())
        assert callgrind.main(costs_args(out, wheel, "--probe", "--scenario", "state/Grid-v1"), platform="linux") == 0
    again = FakeValgrind()
    run_costs_parts(out, wheel, [[f"state/Grid-v1#aa{i}" for i in range(5)]], runner=again)
    assert len(again.calls) == 9, "the probe's N run is aa0's first file"


def test_pgo_gate_merge_ignores_costs_parts(tmp_path):
    # A costs part in a gate's out dir neither completes nor spoils the gate merge.
    out, wheel, _ = run_parts(tmp_path, [all_names()[:-1]])
    run_costs_parts(out, wheel, [[u for u in callgrind.aa_units(MANIFEST) if u.startswith(all_names()[-1] + "#")]])
    assert merge_now(out, wheel) == 2, "the gate's last scenario is not covered by a gate part"
    run_parts(tmp_path, [all_names()[-1:]])
    assert merge_now(out, wheel) == 0
    report = json.loads((out / "feasibility.json").read_text())
    assert all(p.startswith("feasibility.part.") for p in report["merged_parts"])


def test_pgo_costs_merge_refuses_parts_from_another_stamp(tmp_path):
    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    units = callgrind.aa_units(MANIFEST)
    run_costs_parts(out, wheel, [units[0::2]])
    run_costs_parts(out, wheel, [units[1::2]], env=fake_environment(cpu_flags="flags-b"))
    assert costs_merge(out, wheel) == 2 and not (out / "costs.json").exists()


def test_pgo_costs_merge_refuses_overlapping_parts(tmp_path):
    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    units = callgrind.aa_units(MANIFEST)
    run_costs_parts(out, wheel, [units[:60], units[50:]])
    assert costs_merge(out, wheel) == 2 and not (out / "costs.json").exists()


def test_pgo_train_steps_override_reaches_every_scenario():
    seen = []
    env = {"LLVM_PROFILE_FILE": "/abs/%m_%p.profraw"}
    assert train.main(["--manifest", MANIFEST_PATH, "--steps", "8"], environ=env,
                      run=lambda s, m, steps: seen.append(steps)) == 0
    assert seen and set(seen) == {8}


@pytest.mark.parametrize("edit", ["other wheel", "other cpu", "no stamp"])
def test_pgo_costs_merge_reads_every_file_under_the_parts_stamp(tmp_path, edit):
    out = tmp_path / "w0"
    wheel = tmp_path / "w0.whl"
    wheel.write_bytes(b"w0")
    run_costs_parts(out, wheel, [callgrind.aa_units(MANIFEST)])
    side = callgrind.stamp_path(out / "state_Grid-v1.4000.aa2.callgrind")
    st = json.loads(side.read_text())
    if edit == "no stamp":
        side.unlink()
    else:
        st["wheel" if edit == "other wheel" else "cpu_flags"] = "x"
        side.write_text(json.dumps(st))
    assert costs_merge(out, wheel) == 3 and not (out / "costs.json").exists()


def old_format_gate(tmp_path):
    """A gate directory as swarmer03 left it: part reports and file stamps with
    the seven-field stamp that still had `kernels`."""
    out, wheel, _ = run_parts(tmp_path, [all_names()[:10], all_names()[10:]])
    for p in out.glob("feasibility.part.*.json"):
        r = json.loads(p.read_text())
        r["sha256"] = {**r["sha256"], "kernels": "k" * 64}
        p.write_text(json.dumps(r))
    for side in out.glob("*.stamp.json"):
        st = json.loads(side.read_text())
        side.write_text(json.dumps({**st, "kernels": "k" * 64}, sort_keys=True))
    return out, wheel


def test_pgo_w1_costs_from_the_untouched_old_format_gate(tmp_path):
    out, wheel = old_format_gate(tmp_path)
    assert merge_now(out, wheel) == 0, "the gate itself still merges"
    assert costs_merge(out, wheel) == 0, "no adoption: the costs merge reads the gate as it is"
    report = json.loads((out / "costs.json").read_text())
    assert report["sha256"]["kernels"] == "k" * 64 and set(report["costs"]) == {s.name for s in ALL}


def test_pgo_decide_on_the_old_format_gate_and_a_new_w0(tmp_path):
    decide = load("decide")
    out, w1_wheel = old_format_gate(tmp_path)
    assert merge_now(out, w1_wheel) == 0 and costs_merge(out, w1_wheel) == 0
    w0_out = tmp_path / "w0"
    w0_wheel = tmp_path / "w0.whl"
    w0_wheel.write_bytes(b"w0")
    run_costs_parts(w0_out, w0_wheel, [callgrind.aa_units(MANIFEST)])
    assert costs_merge(w0_out, w0_wheel) == 0
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    seventh = {"W1": {"checks": {"wheel_sha256": sha(w1_wheel), "checks": {"python_suite": 0, "num_tests": 0}},
                      "compare": {"passed": True, "wheels": [sha(w0_wheel), sha(w1_wheel)]}}}
    result = decide.decide({"W0": json.loads((w0_out / "costs.json").read_text()),
                            "W1": json.loads((out / "costs.json").read_text())},
                           json.loads((out / "feasibility.json").read_text()), seventh, MANIFEST_PATH,
                           str(TOOLS / "kernels.toml"))
    assert result["ship"] in ("W0", "W1"), "the kernels field differs, every other field is equal"

# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo: the workload at tiny lengths on the local wheel, and the training
and Callgrind entry points without running either. No Valgrind here."""

import hashlib
import importlib.util
import json
import pathlib
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
    assert train.main(["--manifest", MANIFEST_PATH, *flag], environ=env, run=lambda s, m: seen.append(s)) == 0
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
    out = tmp_path / "o"
    base = ["--approved-window", "x", "--manifest", MANIFEST_PATH, "--out-dir", str(out), "--feasibility"]
    assert callgrind.main(base, platform="linux") == 2
    assert callgrind.main([*base, "--wheel", str(wheel)], platform="linux") == 0
    saved = json.loads((out / "feasibility.json").read_text())
    assert saved["sha256"]["wheel"] == hashlib.sha256(b"abc").hexdigest()
    assert set(saved["sha256"]) == {"wheel", "manifest", "kernels"}


def test_pgo_checks_writes_statuses_tied_to_the_wheel(tmp_path):
    checks = load("checks")
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"abc")
    out = tmp_path / "c.json"
    assert checks.main(["--wheel", str(wheel), "--out", str(out), "python_suite=0", "num_tests=3"]) == 0
    saved = json.loads(out.read_text())
    assert saved["wheel_sha256"] == hashlib.sha256(b"abc").hexdigest() and saved["checks"]["num_tests"] == 3
    assert checks.main(["--wheel", str(wheel), "--out", str(out), "bad"]) == 2


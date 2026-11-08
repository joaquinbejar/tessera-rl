# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo: the workload at tiny lengths on the local wheel, and the training
and Callgrind entry points without running either. No Valgrind here."""

import importlib.util
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


THREAD = "positions: line\nevents: Ir Bcm Bim I1mr D1mr D1mw ILmr DLmr DLmw\nob=(1) /x/_native.so\n1 {} 6000 0 0 0 0 0 0 0\n"


class FakeValgrind:
    """Writes Callgrind files as a function of scenario and steps."""

    def __init__(self, control_ir=10, per_step=100, jitter=lambda run: 0, caller_only=False):
        self.calls, self.control_ir, self.per_step, self.jitter, self.caller_only = [], control_ir, per_step, jitter, caller_only
        self.counts = {}

    def __call__(self, argv, check):
        out = next(a.split("=", 1)[1] for a in argv if a.startswith("--callgrind-out-file="))
        scenario = argv[argv.index("--scenario") + 1]
        steps = int(argv[argv.index("--steps") + 1])
        self.calls.append((scenario, steps, out))
        if scenario == callgrind.CONTROL:
            pathlib.Path(out).write_text(THREAD.format(self.control_ir))
        elif "--separate-threads=yes" in argv:
            pathlib.Path(f"{out}-1").write_text(THREAD.format(1000))
            pathlib.Path(f"{out}-2").write_text(THREAD.format(0 if self.caller_only else 5000))
        else:
            n = self.counts[scenario] = self.counts.get(scenario, 0) + 1
            pathlib.Path(out).write_text(THREAD.format(1000 + steps * self.per_step * 1000 + self.jitter(n)))


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


def test_pgo_feasibility_fails_when_the_interpreter_leaks(tmp_path):
    assert gate(tmp_path, control_ir=10**12)[0]["passed"] is False


def test_pgo_feasibility_fails_without_worker_thread_cost(tmp_path):
    report, _ = gate(tmp_path, caller_only=True)
    assert report["passed"] is False and report["scenarios"]["split-phase/CartPole-v1"]["workers"]["passed"] is False


def test_pgo_feasibility_marks_an_unstable_ir_as_not_gating(tmp_path):
    report, _ = gate(tmp_path, jitter=lambda run: 10**9 * run**2)
    assert report["gating"] == [] and report["passed"] is False


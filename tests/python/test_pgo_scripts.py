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

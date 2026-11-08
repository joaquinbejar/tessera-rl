# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo/determinism.py at a tiny scale on the local wheel. Comparing a
wheel with itself is what is possible here: no candidate wheel is built."""

import copy
import importlib.util
import json
import pathlib
import re
import sys
from typing import NamedTuple

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).parents[2] / "tools" / "pgo" / "determinism.py"
_spec = importlib.util.spec_from_file_location("determinism", _TOOL)
det = importlib.util.module_from_spec(_spec)
sys.modules["determinism"] = det
_spec.loader.exec_module(det)

import tessera_rl as t  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return det.record(t, envs=("cartpole", "connect4"), workers=(0, 2), steps=60)


def test_pgo_determinism_every_env_runs_briefly():
    out = det.record(t, workers=(0,), steps=4)
    assert out["runs"]
    assert all(v is True for v in out["checks"].values()), [k for k, v in out["checks"].items() if v is not True]


def test_pgo_determinism_is_stable_and_internally_consistent(report):
    again = det.record(t, envs=("cartpole", "connect4"), workers=(0, 2), steps=60)
    assert det.differences(report, again) == []


def test_pgo_determinism_modes_give_each_instance_the_sync_trajectory(report):
    runs = report["runs"]
    for prefix in ("cartpole/next_step", "connect4/same_step"):
        sync = runs[f"{prefix}/sync/0/owned"]["trace"]
        assert runs[f"{prefix}/pipelined/2/owned"]["trace"] == sync
        assert runs[f"{prefix}/async/2/owned"]["trace"] == sync
        assert runs[f"{prefix}/sync/0/borrowed"]["trace"] == sync


def test_pgo_determinism_detects_a_changed_instance(report):
    changed = copy.deepcopy(report)
    changed["runs"]["cartpole/next_step/sync/0/owned"]["trace"][3] = "0" * 64
    assert det.differences(report, changed) == ["differs: cartpole/next_step/sync/0/owned"]


def test_pgo_determinism_detects_a_changed_snapshot(report):
    changed = copy.deepcopy(report)
    run = changed["runs"]["cartpole/next_step/sync/0/owned"]
    run["snapshots"]["50"][0] = "0" * 64
    assert det.differences(report, changed)


def test_pgo_determinism_detects_missing_runs_and_failed_checks(report):
    changed = copy.deepcopy(report)
    changed["runs"].pop("cartpole/next_step/continuation")
    changed["checks"]["cartpole/next_step/load"] = False
    found = det.differences(report, changed)
    assert "missing run: cartpole/next_step/continuation" in found
    assert "second: check failed: cartpole/next_step/load" in found


def test_pgo_determinism_empty_reports_do_not_pass():
    assert det.differences({"runs": {}, "checks": {}}, {"runs": {}, "checks": {}}) == ["no runs"]


def test_pgo_determinism_actions_are_legal_and_fixed():
    legal = [False, True, False, True]
    assert det.action(1, 0, 5, 4, legal) in (1, 3)
    assert det.action(1, 0, 5, 4, legal) == det.action(1, 0, 5, 4, legal)
    assert det.action(1, 0, 5, 4, [False] * 4) == 0


def test_pgo_determinism_compare_command(report, tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(report))
    changed = copy.deepcopy(report)
    changed["runs"]["cartpole/next_step/sync/0/owned"]["trace"][0] = "x"
    b.write_text(json.dumps(changed))
    assert det.main(["compare", str(a), str(a)]) == 0
    assert det.main(["compare", str(a), str(b)]) == 1
    assert det.main(["compare", str(a)]) == 2
    capsys.readouterr()


def test_pgo_determinism_covers_every_registered_environment():
    source = (pathlib.Path(t.__file__).parent / "_env.py").read_text()
    block = source[source.index("makers = {") :]
    block = block[: block.index("}")]
    registered = set(re.findall(r'"(tessera/[^"]+)"', block))
    assert registered and registered == {env_id for env_id, _ in det.ENVS.values()}


class FakeStep(NamedTuple):
    observations: dict
    rewards: np.ndarray
    episode_terminated: np.ndarray
    masks: np.ndarray | None


def fake_step():
    return FakeStep(
        {"obs": np.zeros((3, 2), np.uint8)},
        np.zeros(3, np.float32),
        np.zeros(3, np.bool_),
        np.zeros((3, 2), np.bool_),
    )


def digests(step):
    trace = det.Trace(3)
    trace.add(range(3), step)
    return trace.result()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.observations["obs"].__setitem__((1, 0), 1),
        lambda s: s.rewards.__setitem__(1, 1.0),
        lambda s: s.episode_terminated.__setitem__(1, True),
        lambda s: s.masks.__setitem__((1, 1), True),
    ],
    ids=["obs", "reward", "flag", "mask"],
)
def test_pgo_determinism_trace_hashes_every_field_per_instance(change):
    base = digests(fake_step())
    step = fake_step()
    change(step)
    got = digests(step)
    assert got[1] != base[1] and got[0] == base[0] and got[2] == base[2]


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


def test_pgo_determinism_rts_is_covered_not_a_gap():
    assert "tessera/RTS-v1" not in det.GAPS and det.GAPS == det.CONTINUOUS_GAP
    assert not hasattr(det, "MULTI_HEAD_GAP")
    assert ("tessera/RTS-v1", {"size": 12, "max_units": 8}) in det.ENVS.values()


@pytest.fixture(scope="module")
def rts_report():
    return det.record(t, envs=("rts",), workers=(0, 2), steps=12)


def test_pgo_determinism_rts_every_mode_gives_the_sync_trajectory(rts_report):
    runs, checks = rts_report["runs"], rts_report["checks"]
    assert all(v is True for v in checks.values()), [k for k, v in checks.items() if v is not True]
    for auto in det.AUTO_RESET:
        sync = runs[f"rts/{auto}/sync/0/owned"]["trace"]
        for mode in ("sync/2/borrowed", "pipelined/2/owned", "pipelined/2/borrowed", "async/2/owned", "async/2/borrowed"):
            assert runs[f"rts/{auto}/{mode}"]["trace"] == sync, mode
        assert f"rts/{auto}/continuation" in runs
        assert {f"rts/{auto}/load", f"rts/{auto}/import", f"rts/{auto}/fork"} <= set(checks)


def test_pgo_determinism_rts_record_is_stable(rts_report):
    assert det.differences(rts_report, det.record(t, envs=("rts",), workers=(0, 2), steps=12)) == []


def test_pgo_determinism_multi_action_is_legal_under_conditional_masks_and_fixed():
    from tessera_rl.masks import head_views

    env = det.make_env(t, "rts", "next_step", 0, "owned")
    try:
        layout = det._layout(env)
        step = env.reset()
        for s in range(1, 6):
            acts = det.actions_for(range(env.num_envs), step, env.agents, env.action_choices, lambda _: s, layout)
            again = det.actions_for(range(env.num_envs), step, env.agents, env.action_choices, lambda _: s, layout)
            assert acts.shape == (env.num_envs, env.agents, len(env.action_nvec)) and (acts == again).all()
            ty, tg, pr = head_views(step.masks, env.mask_heads, env.entity_group_heads)
            u = acts.reshape(env.num_envs, env.agents, -1, 3)
            for i, j, k in np.ndindex(u.shape[:3]):
                a, b, c = (int(v) for v in u[i, j, k])
                for mask, choice in ((ty[i, j, k], a), (tg[i, j, k, a], b), (pr[i, j, k, a], c)):
                    assert mask[choice] or not mask.any()
            step = env.step(acts)
        assert (u[..., 0] != 0).any()
    finally:
        env.close()


def test_pgo_determinism_multi_action_depends_on_instance_slot_and_t_only():
    layout = (((0, (), 3), (3, (3,), 5)), 18, 2)
    row = [True] * 36
    base = det.multi_action(4, 1, 9, layout, row)
    assert len(base) == 4
    assert base == det.multi_action(4, 1, 9, layout, row)
    assert len({tuple(det.multi_action(i, s, k, layout, row)) for i in range(3) for s in range(2) for k in range(3)}) > 1
    only = [False] * 36
    only[2] = True  # entity 0: type 2 only
    only[3 + 2 * 5 + 4] = True  # entity 0, type 2: target 4 only
    got = det.multi_action(4, 1, 9, layout, only)
    assert got[:2] == [2, 4] and got[2:] == [0, 0]


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
    assert registered and registered - det.GAPS == {env_id for env_id, _ in det.ENVS.values()}


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


def test_pgo_determinism_compare_writes_the_wheels(report, tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({**report, "wheel_sha256": "w0"}))
    b.write_text(json.dumps({**report, "wheel_sha256": "w1"}))
    out = tmp_path / "r.json"
    assert det.main(["compare", str(a), str(b), "--out", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert saved["passed"] is True and saved["wheels"] == ["w0", "w1"]
    capsys.readouterr()


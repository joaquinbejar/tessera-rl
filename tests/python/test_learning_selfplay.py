# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3c of the learning protocols (#529, DESIGN-r1, REVIEW-design-r1):
transition bookkeeping, the acting seat's own observation as the only
policy input, the gate, the protocol entry, the fingerprint scope, the
evaluation's diversity and one tiny training smoke."""

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import learning_selfplay as ls  # noqa: E402
from learning_gate import learns_selfplay  # noqa: E402

ENV = "tessera/Lanterns-v1"
KW = {"players": 2}


def test_selfplay_one_transition_per_acting_step_with_the_team_reward():
    """Seats alternate; each acting step stores one transition with that
    step's team reward, and the episode's end marks the last one."""
    buf = ls.Transitions(1)
    for t, (reward, done) in enumerate([(0.0, False), (1.0, False), (0.0, False), (1.0, True)]):
        buf.record(0, True, reward, done, {"value": 0.0, "seat": t % 2})
    lst = buf.lists[0]
    assert len(buf) == 4
    assert [tr["seat"] for tr in lst] == [0, 1, 0, 1]
    assert [tr["reward"] for tr in lst] == [0.0, 1.0, 0.0, 1.0]
    assert [tr["done"] for tr in lst] == [False, False, False, True]


def test_selfplay_step_without_an_actor_folds_into_the_last_transition():
    buf = ls.Transitions(2)
    buf.record(0, True, 1.0, False, {"value": 0.0})
    buf.record(0, False, 2.0, True, None)
    buf.record(1, False, 5.0, False, None)  # nothing stored yet: dropped
    assert buf.lists[0] == [{"value": 0.0, "reward": 3.0, "done": True}]
    assert buf.lists[1] == []
    buf.record(0, False, 7.0, False, None)  # a new episode: nothing to fold into
    assert buf.lists[0][0]["reward"] == 3.0


def test_selfplay_advantages_stop_at_terminal_transitions():
    trs = [{"reward": 1.0, "value": 0.5, "done": True}, {"reward": 0.0, "value": 0.2, "done": False}]
    adv, ret = ls.advantages(trs, 1.0, 0.9, 1.0)
    assert adv[1] == pytest.approx(0.0 + 0.9 * 1.0 - 0.2)
    assert adv[0] == pytest.approx(1.0 - 0.5)
    assert ret == pytest.approx([a + t["value"] for a, t in zip(adv, trs, strict=True)])


def test_selfplay_policy_input_is_only_the_acting_seats_own_row():
    """rev (REVIEW-design-r1, Q1): the input at a step equals the acting
    seat's observation row; another seat's row never reaches it."""
    n, agents, d = 3, 2, 6
    obs = np.zeros((n, agents, d), dtype=np.uint8)
    obs[:, 0, :] = 10 + np.arange(n)[:, None]  # seat 0 rows
    obs[:, 1, :] = 200 + np.arange(n)[:, None]  # seat 1 rows: sentinels
    acting = np.array([[True, False], [False, True], [True, False]])
    seats = ls.acting_seats(acting)
    assert seats.tolist() == [0, 1, 0]
    x = ls.seat_obs({"obs": obs}, seats)
    assert x.dtype == np.float32 and x.shape == (n, d)
    for i in range(n):
        np.testing.assert_array_equal(x[i], obs[i, seats[i]].astype(np.float32))
        other = obs[i, 1 - seats[i]]
        assert not np.isin(other, x[i]).any(), f"instance {i}: another seat's row leaked into the input"
    masks = np.zeros((n, agents, 4), dtype=bool)
    masks[:, 0, 0] = True
    masks[:, 1, 3] = True
    m = ls.seat_mask(masks, seats, 4)
    assert m.tolist() == [[True, False, False, False], [False, False, False, True], [True, False, False, False]]
    acts = ls.full_actions(n, agents, seats, np.array([2, 3, 1]))
    assert acts.tolist() == [[2, 0], [0, 3], [1, 0]]


def test_selfplay_no_acting_seat_has_no_legal_action_and_two_actors_are_refused():
    seats = ls.acting_seats(np.array([[False, False], [False, True]]))
    assert seats.tolist() == [-1, 1]
    assert not ls.seat_mask(np.ones((2, 2, 3), dtype=bool), seats, 3)[0].any()
    assert ls.full_actions(2, 2, seats, np.array([2, 1])).tolist() == [[0, 0], [0, 1]]
    with pytest.raises(ValueError, match="sequential"):
        ls.acting_seats(np.array([[True, True]]))


def test_selfplay_policy_input_on_the_real_game_is_the_acting_seats_row():
    pytest.importorskip("tessera_rl")
    env = ls._make(ENV, 2, 7, None, KW)
    step = env.reset()
    for _ in range(6):
        seats = ls.acting_seats(step.acting)
        assert (seats >= 0).all()
        x = ls.seat_obs(step.observations, seats)
        for i in range(2):
            row = np.concatenate([np.asarray(step.observations[k][i, seats[i]], dtype=np.float32).ravel() for k in sorted(step.observations)])
            np.testing.assert_array_equal(x[i], row)
        mask = ls.seat_mask(step.masks, seats, env.action_choices)
        chosen = np.array([int(np.flatnonzero(mask[i])[0]) for i in range(2)])
        step = env.step(ls.full_actions(2, env.agents, seats, chosen))
    env.close()


def test_selfplay_gate_passes_with_fixed_numbers():
    u = [1.0, 2.0, 1.0, 2.0]
    h = [10.0, 11.0, 10.0, 11.0]
    ok = learns_selfplay([5.0, 6.0, 5.5, 6.5], u, h)
    assert ok["gate"] and ok["learns"] and ok["threshold"] and not ok["degenerate"]
    assert ok["required"] == pytest.approx(0.10 * (10.5 - 1.5))
    low = learns_selfplay([1.5, 2.0, 1.0, 2.5], u, h)
    assert low["gate"] and not low["learns"] and not low["threshold"]


def test_selfplay_no_gate_when_the_heuristic_does_not_beat_uniform():
    r = learns_selfplay([5.0, 6.0], [3.0, 4.0], [3.0, 4.0])
    assert not r["gate"] and not r["learns"] and "no gate" in r["reason"]


def test_selfplay_constant_scores_are_degenerate():
    r = learns_selfplay([9.0, 9.0, 9.0], [1.0, 2.0], [10.0, 11.0])
    assert r["gate"] and r["degenerate"] and not r["learns"]


def test_selfplay_gate_needs_two_finite_scores_each():
    assert "error" in learns_selfplay([5.0], [1.0, 2.0], [3.0, 4.0])
    assert "error" in learns_selfplay([5.0, float("nan")], [1.0, 2.0], [3.0, 4.0])


def test_selfplay_protocol_entry_and_gate():
    import learning_protocols as lp

    p = lp.protocol("lanterns")
    assert p["tier"] == "T3c" and p["env"] == ENV and p["env_kwargs"] == KW
    assert p["total_steps"] == 1_000_000 and p["hyper"] == {} and p["eval_episodes"] == 50 and p["score"] == "final"
    assert lp.gate("lanterns") == {"min_effect": 0.10}
    assert lp.selfplay_hyper("lanterns", 512).total_steps == 512
    assert not (ROOT / "compat" / "learning" / "lanterns.json").exists()


def test_selfplay_calibrate_uses_ten_seeds_and_their_minimum():
    from learning_gate import T3B_CALIBRATION_SEEDS, calibrate, check_calibration

    records = [{"seed": s, "score": float(s - 95)} for s in T3B_CALIBRATION_SEEDS]
    out = calibrate(records, T3B_CALIBRATION_SEEDS, expected=T3B_CALIBRATION_SEEDS, threshold="min")
    assert out["threshold"] == 6.0 and out["threshold_rule"] == "min"
    out.update({"records": records, "setup": {"env": ENV, "total_steps": 1_000_000, "eval_episodes": 50, "fingerprint": "f"}})
    assert check_calibration(out) == []


def test_selfplay_fingerprint_scope():
    """T3c hashes learning_selfplay and only the learning_masked functions it
    imports: other learning_masked code moves no T3c fingerprint, and
    learning_selfplay moves no other tier's."""
    pytest.importorskip("torch")
    import learning_masked as lm
    import learning_protocols as lp

    t3c = lp.fingerprint("lanterns")
    others = {n: lp.fingerprint(n) for n in ("acrobot", "snake", "connect4", "pendulum", "mountaincar")}
    real_lm, real_ls = lm.__file__, ls.__file__
    try:
        lm.__file__ = lp.__file__  # "edits" the whole learning_masked file
        assert lp.fingerprint("lanterns") == t3c
    finally:
        lm.__file__ = real_lm
    try:
        ls.__file__ = lp.__file__  # "edits" learning_selfplay
        assert lp.fingerprint("lanterns") != t3c
        assert {n: lp.fingerprint(n) for n in others} == others
    finally:
        ls.__file__ = real_ls
    real_imported = ls.IMPORTED
    try:
        ls.IMPORTED = real_imported[:-1]
        assert lp.fingerprint("lanterns") != t3c
    finally:
        ls.IMPORTED = real_imported
    assert lp.train_selfplay not in lp._training_path()


def _calibrated() -> list[str]:
    return sorted(p.stem for p in (ROOT / "compat" / "learning").glob("*.json"))


# Already stale on main before #529: #591 changed _FlatVector, which is in
# the T3a "flatten" scope, after these two were calibrated (#567). They need
# their own recalibration (reported separately); xfail until then.
STALE_ON_MAIN = {"jobshop", "vrp"}


@pytest.mark.parametrize("name", _calibrated())
def test_selfplay_every_committed_calibration_fingerprint_is_unchanged(name):
    """rev (REVIEW-design-r1): the 11 committed calibrations still match."""
    pytest.importorskip("torch")
    assert len(_calibrated()) == 11
    if name in STALE_ON_MAIN:
        pytest.xfail(f"{name}: fingerprint already stale on main (#591), needs recalibration")
    sys.path.insert(0, str(ROOT / "tools"))
    import learning_test as lt

    cal = json.loads((ROOT / "compat" / "learning" / f"{name}.json").read_text())
    lt.PROTOCOL = name
    try:
        assert lt._fingerprint() == cal["setup"]["fingerprint"], name
    finally:
        lt.PROTOCOL = "cartpole"


@pytest.mark.parametrize("kind", ["uniform", "heuristic", "trained"])
def test_selfplay_evaluation_deals_and_games_differ(kind):
    """rev (REVIEW-design-r1): distinct deals over a seed's episodes, and at
    least 8 distinct games of 10 for each player kind."""
    agent = None
    if kind == "trained":
        pytest.importorskip("torch")
        probe = ls._make(ENV, 1, 0, None, KW)
        step = probe.reset()
        obs = ls.seat_obs(step.observations, ls.acting_seats(step.acting))
        import torch

        torch.manual_seed(0)
        agent = ls._agent_class()(obs.shape[1], probe.action_choices)
        probe.close()
    trace = []
    scores = ls.eval_scores(ENV, KW, kind, 10, 10_000, agent, trace=trace)
    assert len(scores) == 10 and all(0.0 <= s <= 25.0 and float(s).is_integer() for s in scores), scores
    deals = {t[1] for t in trace}
    games = {t[2] for t in trace}
    assert len(deals) >= 8, f"{kind}: {len(deals)} distinct deals of 10"
    assert len(games) >= 8, f"{kind}: {len(games)} distinct games of 10"
    assert not any(t[3] for t in trace), f"{kind}: a game was truncated"
    again = ls.eval_scores(ENV, KW, kind, 10, 10_000, agent)
    assert again == scores


def test_selfplay_heuristic_beats_uniform_on_fixed_seeds():
    """Deterministic: fixed eval seeds 10_000..10_019, no torch, so the
    result is the same on every run."""
    h = ls.eval_scores(ENV, KW, "heuristic", 20, 10_000)
    u = ls.eval_scores(ENV, KW, "uniform", 20, 10_000)
    assert h == ls.eval_scores(ENV, KW, "heuristic", 20, 10_000)
    assert u == ls.eval_scores(ENV, KW, "uniform", 20, 10_000)
    assert sum(h) / 20 - sum(u) / 20 >= 1.0, (h, u)


def test_selfplay_stall_guard_fails_loudly():
    ls._check_stall(1000 * 64, 64, ENV)
    with pytest.raises(RuntimeError, match=r"tessera/Lanterns-v1: no seat acts any more"):
        ls._check_stall(1000 * 64 + 1, 64, ENV)


@pytest.mark.torch
def test_selfplay_lanterns_one_update_smoke():
    pytest.importorskip("torch")
    import math

    import learning_protocols as lp

    with pytest.raises(ValueError, match="not T3c"):
        lp.train_selfplay("acrobot", 1, 512, 2)
    r = lp.train_selfplay("lanterns", 1, 512, 2)
    assert len(r["eval_returns"]) == 2 and math.isfinite(r["eval_mean"])
    assert r["truncations"] == 0 and r["eval_truncations"] == 0
    assert r["distinct_deals"] == 2
    refs = lp.selfplay_references("lanterns", 2)
    assert len(refs["uniform"]) == 2 and len(refs["heuristic"]) == 2

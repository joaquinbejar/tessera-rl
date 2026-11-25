# SPDX-License-Identifier: MIT OR Apache-2.0
"""``python -m tessera_rl.eval`` (ADR-0088 section 9, #384 head E4): exactly N
episodes per opponent from fixed seeds, counted by the engine's league and
checked against a hand-stepped reference, Wilson intervals, byte-identical
reruns, the refusals and the hard error of a miscount."""

import json
import math
import os
import sys

import numpy as np
import pytest

import tessera_rl
from tessera_rl import eval as tev

sys.path.insert(0, os.path.dirname(__file__))
import eval_policies as pol  # noqa: E402

DUEL = "tessera/PaddleDuel-v1"
KW = {"agents": 2, "max_episode_steps": 40}
SEEDS = [11, 12, 13, 14, 15]
OPPONENTS = [(0, "heuristic"), (1, ("external", 7))]


def _evaluate(episodes=6, pools=3, policy=pol.chase, opponents=OPPONENTS):
    return tev.evaluate(
        DUEL, policy, opponents, seeds=SEEDS, episodes=episodes, pools=pools, env_kwargs=KW, external={7: pol.down}
    )


def _reference(kind, episodes):
    """The same episodes stepped by hand on a plain pool, with the outcome
    rule applied in Python: learner return minus opponent return."""
    group = "heuristic" if kind == "heuristic" else "external"
    env = tessera_rl.make(DUEL, num_envs=1, seed=0, workers=0, auto_reset="disabled", outputs="owned",
                          policy_map={1: group}, **KW)
    ext = pol.down if group == "external" else None
    out = {"wins": 0, "draws": 0, "losses": 0}
    for k in range(episodes):
        ret = tev._play(env, pol.chase, ext, SEEDS[k % len(SEEDS)], (0,), (1,), list(env.action_nvec))
        diff = ret[0] - ret[1]
        out["wins" if diff > 0 else "losses" if diff < 0 else "draws"] += 1
    env.close()
    return out


def test_eval_counts_equal_a_hand_stepped_reference():
    report = _evaluate(episodes=6, pools=3)
    for row, (_, kind) in zip(report["opponents"], OPPONENTS):
        assert row["wins"] + row["draws"] + row["losses"] == 6
        assert {k: row[k] for k in ("wins", "draws", "losses")} == _reference(kind, 6)


def test_eval_uneven_waves_count_exactly_n():
    """N = 13 on K = 4 pools: every pool joins every league round, the
    counts sum to 13 and nothing is dropped (rev, plan r2 nit)."""
    report = _evaluate(episodes=13, pools=4, opponents=[(0, "heuristic")])
    (row,) = report["opponents"]
    assert row["wins"] + row["draws"] + row["losses"] == 13


def test_eval_rerun_is_byte_identical():
    assert json.dumps(_evaluate()) == json.dumps(_evaluate())


def test_eval_episode_k_starts_the_same_against_every_opponent():
    starts = []
    for kind in ("heuristic", ("external", 7)):
        league = tessera_rl.League([kind], [{"matchmaking": "uniform"}])
        env = tessera_rl.make(DUEL, num_envs=1, seed=0, workers=0, auto_reset="disabled", outputs="owned",
                              policy_map={1: "opponent"}, league=league, role=0, **KW)
        row = []
        for s in SEEDS[:2]:
            obs = env.reset(seed=s).observations
            row.append(({k: np.asarray(v).tobytes() for k, v in obs.items()}, env.state_hash(0)))
        starts.append(row)
        env.close()
    # Episode k starts from the same state against every opponent...
    assert starts[0] == starts[1]
    # ...and the seed reaches the state (PaddleDuel's first frame is the
    # same for every seed: its randomness comes at the serve).
    assert starts[0][0][1] != starts[0][1][1]


def test_eval_wilson_bounds():
    z = tev.WILSON_Z

    def ref(k, n):
        p = k / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return [c - h, c + h]

    assert tev.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-15)
    assert tev.wilson(10, 10)[1] == pytest.approx(1.0)
    assert tev.wilson(22, 40) == pytest.approx(ref(22, 40))


def test_eval_refusals():
    with pytest.raises(tessera_rl.CapabilityError, match="ONNX"):
        _evaluate(opponents=[(0, ("onnx", "07" * 16))])
    with pytest.raises(tessera_rl.ArgumentError, match="needs --external 9"):
        _evaluate(opponents=[(0, ("external", 9))])
    with pytest.raises(tessera_rl.ArgumentError):
        tev._load("no_such_module:policy")
    for bad in (pol.wrong_shape, pol.float_actions, pol.out_of_range):
        with pytest.raises(tessera_rl.ArgumentError, match="the policy returned"):
            _evaluate(episodes=1, pools=1, policy=bad, opponents=[(0, "heuristic")])
    league = tessera_rl.League(["heuristic"], [{"matchmaking": "uniform"}])
    with pytest.raises(tessera_rl.ArgumentError, match="not in the export"):
        tev.from_export(league.export(), [3])


def test_eval_league_export_matches_the_population():
    league = tessera_rl.League(["heuristic", ("external", 7)], [{"matchmaking": "uniform"}])
    assert tev.from_export(league.export(), None) == tev.parse_population("heuristic,external:7")


def test_eval_miscount_is_a_hard_error(monkeypatch):
    real = tessera_rl.League.ratings

    def short(self, role=0):
        r = real(self, role)
        r["counts"][0]["wins"] = r["counts"][0]["wins"] + 1
        return r

    monkeypatch.setattr(tessera_rl.League, "ratings", short)
    with pytest.raises(RuntimeError, match="episodes counted for 2 expected"):
        _evaluate(episodes=2, pools=1, opponents=[(0, "heuristic")])


def test_eval_cli_writes_the_report(tmp_path):
    out = tmp_path / "r.json"
    sys.modules.setdefault("eval_policies", pol)
    rc = tev.main([
        "--env", DUEL, "--env-kwargs", json.dumps(KW), "--policy", "eval_policies:stay",
        "--population", "heuristic", "--seeds", "1:4", "--episodes", "3", "--pools", "2", "--output", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert [r["episodes"] for r in report["opponents"]] == [3]
    assert report["setup"]["policy"] == "eval_policies:stay"
    assert set(report["provenance"]) >= {"git", "python", "tessera_rl", "numpy"}


def test_eval_play_submits_the_external_rows():
    """The External callable's actions are what the opponent slot plays: the
    state after an episode equals a hand loop that submits them."""

    def build():
        return tessera_rl.make(DUEL, num_envs=1, seed=0, workers=0, auto_reset="disabled", outputs="owned",
                               policy_map={1: "external"}, **KW)

    a = build()
    tev._play(a, pol.stay, pol.down, SEEDS[0], (0,), (1,), list(a.action_nvec))
    b = build()
    step = b.reset(seed=SEEDS[0])
    while True:
        step = b.step(np.array([[1, 2]], dtype=np.int32))
        if bool(step.episode_terminated[0]) or bool(step.episode_truncated[0]):
            break
    assert a.state_hash(0) == b.state_hash(0)
    a.close()
    b.close()

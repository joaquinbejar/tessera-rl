# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS learning gate's evaluation tool (tools/rts_evaluation.py). No
evaluation runs: no test reaches an evaluation over the real seeds, and no
trained policy is evaluated. The play test uses a fake environment and an
untrained network."""

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import rts_evaluation as ev  # noqa: E402
import rts_policy  # noqa: E402
from rts_policy import Agent  # noqa: E402
from rts_training import Hyper  # noqa: E402

SMALL = Hyper(num_envs=2, num_steps=4, size=8, max_units=8, step_limit=40)


class Reached(Exception):
    pass


def tripwire(*_a, **_k):
    raise Reached


def test_rts_evaluation_constants_are_the_protocols():
    assert ev.EVAL_SEEDS == range(2_000_000, 2_000_400) and len(ev.EVAL_SEEDS) == 400
    assert ev.GATE == 0.60 and ev.TRAINING_SEEDS == 5


def test_rts_evaluation_wilson_lower_bound_crosses_the_gate_at_260_of_400():
    assert ev.wilson_lower(259, 400) == pytest.approx(0.5995, abs=1e-4)
    assert ev.wilson_lower(260, 400) == pytest.approx(0.6020, abs=1e-4)
    assert not ev.seed_passes(259) and ev.seed_passes(260)
    with pytest.raises(ValueError):
        ev.wilson_lower(0, 0)


def test_rts_evaluation_verdict_needs_every_seed():
    assert ev.verdict(dict.fromkeys(range(5), 260))["passes"]
    one_short = {**dict.fromkeys(range(5), 400), 3: 259}
    v = ev.verdict(one_short)
    assert not v["passes"] and [s for s, r in v["per_seed"].items() if not r["passes"]] == [3]
    with pytest.raises(ValueError):
        ev.verdict(dict.fromkeys(range(4), 400))


def test_rts_evaluation_sides_by_parity_and_outcomes():
    assert [ev.side_of(s) for s in (2_000_000, 2_000_001, 2_000_398, 2_000_399)] == [False, True, False, True]
    assert ev.outcome(True, False, 1.0) == "win"
    assert ev.outcome(True, False, -1.0) == "loss"
    assert ev.outcome(True, False, 0.0) == "draw"
    assert ev.outcome(False, True, 0.0) == "truncated"
    assert ev.outcome(True, True, 1.0) == "win"
    with pytest.raises(ValueError):
        ev.outcome(False, False, 0.0)


def test_rts_evaluation_make_env_uses_the_protocol_config_and_the_side(monkeypatch):
    seen = []
    monkeypatch.setattr(ev.tessera_rl.gym, "make", lambda env_id, **kw: seen.append((env_id, kw)))
    for seed in (2_000_000, 2_000_001):
        ev.make_env(Hyper(), seed)
    assert [s[0] for s in seen] == [ev.ENV_ID] * 2
    assert [s[1]["seat_swap"] for s in seen] == [False, True]
    assert all((s[1]["size"], s[1]["max_units"], s[1]["decision_ticks"], s[1]["step_limit"]) == (16, 16, 4, 600)
               for s in seen)  # fmt: skip


@pytest.mark.parametrize("authorization,frozen", [("", ""), ("x", "y"), ("x", ""), ("", "y")])
def test_rts_evaluation_refuses_before_any_game(monkeypatch, authorization, frozen):
    # Anything past the guard raises Reached at once: a weakened guard fails
    # this test instead of spending the evaluation seeds.
    monkeypatch.setattr(ev, "make_env", tripwire)
    monkeypatch.setattr(ev, "play", tripwire)
    with pytest.raises(PermissionError):
        ev.evaluate(Agent(11, 8, 2, 8, 6, 81, 4), SMALL, authorization, frozen)


def test_rts_evaluation_counts_wins_over_each_seed_once(monkeypatch):
    """With the guard passed, recorders for the environment and the game: each
    seed opened, played and closed once, in order; only wins count; one CPU
    thread. Nothing real runs."""
    log = []

    class Env:
        def __init__(self, seed):
            self.seed = seed

        def close(self):
            log.append(("close", self.seed))

    outcomes = {10: "win", 11: "loss", 12: "truncated", 13: "win", 14: "draw"}
    monkeypatch.setattr(ev, "make_env", lambda h, seed: (log.append(("make", seed)), Env(seed))[1])
    monkeypatch.setattr(ev, "play", lambda agent, env, seed: (log.append(("play", seed)), outcomes[seed])[1])
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        r = ev.evaluate(Agent(11, 8, 2, 8, 6, 81, 4), SMALL, "h", "h", seeds=range(10, 15))
        inside = torch.get_num_threads()
    finally:
        torch.set_num_threads(threads)
    assert log == [x for s in range(10, 15) for x in (("make", s), ("play", s), ("close", s))]
    assert (r["wins"], r["n"]) == (2, 5) and r["games"] == outcomes
    assert r["lower"] == pytest.approx(ev.wilson_lower(2, 5))
    assert inside == 1


class FakeGame:
    """A one-env vector env that ends on its third step with the learner's
    reward +1 (a win)."""

    def __init__(self, u=8, size=8):
        self.t, self.u, self.size = 0, u, size

    def _obs(self):
        return {"planes": np.zeros((1, 11, self.size, self.size), np.uint8), "globals": np.zeros((1, 2), np.uint16)}

    def _info(self):
        u = self.u
        return {"action_masks": (np.ones((1, u, 6), np.int8), np.ones((1, u, 6, 81), np.int8), np.ones((1, u, 6, 4), np.int8))}

    def reset(self, seed):
        self.seed = seed
        return self._obs(), self._info()

    def step(self, actions):
        assert actions.shape == (1, self.u * 3)
        self.t += 1
        done = self.t == 3
        return self._obs(), np.array([1.0 if done else 0.0]), np.array([done]), np.array([False]), self._info()


def test_rts_evaluation_play_is_greedy_and_reads_the_outcome(monkeypatch):
    calls = []
    real = rts_policy.Conditional.sample

    def spy(self, greedy=False):
        calls.append(greedy)
        return real(self, greedy=greedy)

    monkeypatch.setattr(rts_policy.Conditional, "sample", spy)
    torch.manual_seed(0)
    env = FakeGame()
    assert ev.play(Agent(11, 8, 2, 8, 6, 81, 4), env, 7) == "win"
    assert env.seed == 7 and env.t == 3 and calls == [True, True, True]


def test_rts_evaluation_save_and_load_round_trip(tmp_path):
    torch.manual_seed(1)
    a = Agent(11, 8, 2, 8, 6, 81, 4)
    rec = ev.record(3, SMALL, "abc")
    path = tmp_path / "policy.pt"
    ev.save(a, path, rec)
    b, got = ev.load(path, SMALL)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert got["training_seed"] == 3 and got["protocol_sha256"] == "abc"
    assert got["hyper"]["total_steps"] == SMALL.total_steps and "torch" in got["versions"]


def test_rts_evaluation_play_runs_on_the_real_environment_untrained():
    # The real tessera/RTS-vsRush-v1 at a tiny config, an untrained network,
    # a seed outside the evaluation list: play reaches an outcome.
    torch.manual_seed(0)
    h = Hyper(size=8, max_units=8, step_limit=20)
    env = ev.make_env(h, 5)
    try:
        assert ev.play(Agent(11, 8, 2, 8, 6, 81, 4), env, 5) in {"win", "loss", "draw", "truncated"}
    finally:
        env.close()

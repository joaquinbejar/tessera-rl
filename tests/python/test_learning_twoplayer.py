# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T3b of the learning protocols (#370, DESIGN-T3b-r1, REVIEW-T3b-r1):
transition bookkeeping, seats, the two thresholds, the fingerprint scope,
and one tiny training smoke."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import learning_twoplayer as l2  # noqa: E402
from learning_gate import learns_twoplayer  # noqa: E402

T3B = ("connect4", "othello", "go9", "leduc", "tag")


def test_twoplayer_opponent_turn_rewards_fold_undiscounted_into_the_last_transition():
    """Rewards on steps where the learner does not act join its last stored
    transition of the episode, undiscounted; an end on such a step marks it
    terminal, and nothing folds across episodes."""
    t = l2.Transitions(1)
    t.record(0, True, 0.0, False, {"value": 0.0})  # learner moves
    t.record(0, False, 0.5, False, None)  # opponent's turn: +0.5
    t.record(0, False, -1.0, True, None)  # opponent wins: -1, episode ends
    t.record(0, False, 2.0, False, None)  # next episode, learner not acted yet: dropped
    t.record(0, True, 1.0, True, {"value": 0.0})  # learner wins at once
    lst = t.lists[0]
    assert len(lst) == 2
    assert lst[0]["reward"] == pytest.approx(-0.5) and lst[0]["done"]
    assert lst[1]["reward"] == 1.0 and lst[1]["done"]


def test_twoplayer_reward_on_the_learner_step_stays_on_it():
    t = l2.Transitions(2)
    t.record(0, True, 1.0, False, {"value": 0.0})
    t.record(1, False, 3.0, False, None)  # instance 1 has no transition yet
    assert t.lists[0][0]["reward"] == 1.0 and t.lists[1] == [] and len(t) == 1


def test_twoplayer_advantages_stop_at_terminal_transitions():
    trs = [{"reward": 1.0, "done": True, "value": 0.5}, {"reward": 0.0, "done": False, "value": 0.2}]
    adv, ret = l2.advantages(trs, last_value=1.0, gamma=0.9, lam=1.0)
    # Last transition bootstraps from last_value; the first does not look past its end.
    assert adv[1] == pytest.approx(0.0 + 0.9 * 1.0 - 0.2)
    assert adv[0] == pytest.approx(1.0 - 0.5)
    assert ret == pytest.approx([a + t["value"] for a, t in zip(adv, trs, strict=True)])


def test_twoplayer_seats_alternate_by_seed():
    assert [l2.learner_slot(s, [0, 1]) for s in range(4)] == [0, 1, 0, 1]
    assert [l2.learner_slot(s, [0]) for s in range(3)] == [0, 0, 0]


def test_twoplayer_opponent_maps():
    assert l2.opponent_map(1, 2, "heuristic") == {0: "heuristic"}
    assert l2.opponent_map(0, 3, "self") == {0: "heuristic", 1: "heuristic", 2: "heuristic"}
    assert l2.opponent_map(0, 2, "python") is None
    with pytest.raises(ValueError):
        l2.opponent_map(0, 2, "onnx")


def test_twoplayer_episode_score_is_the_sign_or_the_return():
    assert [l2.episode_score(x, "sign") for x in (3.0, 0.0, -0.5)] == [1.0, 0.0, -1.0]
    assert l2.episode_score(-0.5, "return") == -0.5
    with pytest.raises(ValueError):
        l2.episode_score(1.0, "win")


def test_twoplayer_thresholds_with_fixed_numbers():
    u = [-1.0, -1.0, -0.8, -1.0, -0.6, -1.0] * 4  # uniform against the heuristic: about -0.9
    h = [0.0, 0.2, -0.2, 0.0, 0.1, -0.1] * 4  # heuristic against itself: about 0
    # Learns: closes far more than a tenth of the gap, and beats uniform play.
    ok = learns_twoplayer([-0.2, -0.3, -0.1, -0.25, -0.2], [0.8, 0.9, 0.7, 0.85, 0.9], u, h)
    assert ok["gate"] and ok["learns"] and ok["threshold_1"] and ok["threshold_2"]
    assert ok["required"] == pytest.approx(0.10 * (sum(h) / len(h) - sum(u) / len(u)))
    # Fails threshold 2: no better than random against a random opponent.
    t2 = learns_twoplayer([-0.2, -0.3, -0.1, -0.25, -0.2], [0.1, -0.2, 0.0, -0.1, 0.05], u, h)
    assert t2["threshold_1"] and not t2["threshold_2"] and not t2["learns"]
    # Fails threshold 1: no better than U against the heuristic.
    t1 = learns_twoplayer([-0.9, -0.95, -0.85, -0.9, -0.92], [0.8, 0.9, 0.7, 0.85, 0.9], u, h)
    assert not t1["threshold_1"] and not t1["learns"]


def test_twoplayer_no_gate_when_the_heuristic_does_not_beat_random():
    same = [-0.5, 0.5, 0.0, -0.5, 0.5, 0.0] * 4
    r = learns_twoplayer([0.9, 0.8, 0.9, 1.0, 0.9], [0.9, 0.8, 0.9, 1.0, 0.9], same, same)
    assert not r["gate"] and not r["learns"] and r["hu_lower95"] <= 0


def test_twoplayer_constant_scores_are_degenerate():
    u = [-1.0, -0.8] * 10
    h = [0.0, 0.2] * 10
    r = learns_twoplayer([0.5] * 5, [0.9, 0.8, 0.9, 1.0, 0.9], u, h)
    assert r["degenerate"] and not r["learns"]


def test_twoplayer_thresholds_need_two_finite_scores_each():
    r = learns_twoplayer([0.5], [0.5, 0.6], [0.0, 0.1], [0.2, 0.3])
    assert not r["learns"] and "error" in r


def test_twoplayer_protocol_entries_and_gates():
    import learning_protocols as lp

    for name in T3B:
        p = lp.protocol(name)
        assert p["tier"] == "T3b", name
        assert p["score"] in ("sign", "return") and p["seats"] in ([0, 1], [0]), name
        assert p["eval_episodes"] % len(p["seats"]) == 0, name
        assert lp.gate(name) == {"min_effect": 0.10}, name
    assert lp.protocol("tag")["seats"] == [0] and lp.protocol("tag")["score"] == "return"


def test_twoplayer_fingerprint_scope():
    """T3b's fingerprint hashes learning_twoplayer and only the
    learning_masked functions it imports: other learning_masked code moves
    no T3b fingerprint, and learning_twoplayer moves no T1 or T3a one."""
    pytest.importorskip("torch")
    import learning_masked as lm
    import learning_protocols as lp

    with pytest.raises(ValueError, match="not T3b"):
        lp.train_twoplayer("acrobot", 1, 512, 2)
    t3b, t3a, t1 = lp.fingerprint("connect4"), lp.fingerprint("snake"), lp.fingerprint("acrobot")
    assert t3b != lp.fingerprint("othello")
    real_lm, real_l2 = lm.__file__, l2.__file__
    try:
        lm.__file__ = lp.__file__  # "edits" the whole learning_masked file
        assert lp.fingerprint("connect4") == t3b
        assert lp.fingerprint("snake") != t3a
    finally:
        lm.__file__ = real_lm
    try:
        l2.__file__ = lp.__file__  # "edits" learning_twoplayer
        assert lp.fingerprint("connect4") != t3b
        assert lp.fingerprint("snake") == t3a and lp.fingerprint("acrobot") == t1
    finally:
        l2.__file__ = real_l2


@pytest.mark.torch
def test_twoplayer_connect4_one_update_smoke():
    pytest.importorskip("torch")
    import math

    import learning_protocols as lp

    r = lp.train_twoplayer("connect4", 1, 64, 2)
    assert r["learner_slot"] == 1
    assert len(r["vs_heuristic"]) == 2 and len(r["vs_uniform"]) == 2
    assert all(x in (-1.0, 0.0, 1.0) for x in r["vs_heuristic"] + r["vs_uniform"])
    assert math.isfinite(r["eval_mean"])
    refs = lp.twoplayer_references("connect4", 2)
    assert len(refs["uniform"]) == 2 and len(refs["heuristic"]) == 2


def test_twoplayer_stall_guard_fails_loudly():
    """REVIEW-T3b-head-r1: the guard raises, naming the environment and slot."""
    l2._check_stall(1000 * 64, 64, "tessera/Connect4-v1", 1)  # at the bound: fine
    with pytest.raises(RuntimeError, match=r"tessera/Connect4-v1: the learner in slot 1 stopped acting"):
        l2._check_stall(1000 * 64 + 1, 64, "tessera/Connect4-v1", 1)


def test_twoplayer_truncation_bootstraps_only_truncated_not_terminated():
    """REVIEW-T3b-head-r1 (a): truncated and not terminated gets gamma*V(s');
    terminated gets nothing, even when also flagged truncated."""
    term = np.array([False, True, False, True])
    trunc = np.array([True, False, False, True])
    values = np.array([2.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(l2.truncation_bootstrap(term, trunc, values, 0.5), [1.0, 0.0, 0.0, 0.0])


def test_twoplayer_truncation_target_uses_the_terminal_observation():
    """The bootstrap reads the engine's terminal-observation rows, not the
    post-reset observation, and lands on the closing transition."""

    terminal = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)

    class FakeEnv:
        terminal_observations = {"obs": terminal}

    env = FakeEnv()
    post_reset = np.zeros((2, 2, 3), dtype=np.float32)
    got = l2.terminal_obs(env, learner=1)
    np.testing.assert_array_equal(got, terminal[:, 1, :])
    assert not np.array_equal(got, post_reset[:, 1, :])
    # A value function that tells the two apart: the closing transition's
    # target gets gamma * V(terminal), and terminates there.
    value = got.sum(axis=1)
    boot = l2.truncation_bootstrap([False, False], [True, False], value, 0.9)
    buf = l2.Transitions(2)
    buf.record(0, True, 1.0 + boot[0], True, {"value": 0.0})
    (tr,) = buf.lists[0]
    assert tr["done"] and tr["reward"] == pytest.approx(1.0 + 0.9 * float(terminal[0, 1].sum()))


def test_twoplayer_terminal_obs_on_a_real_tag_truncation():
    """The native path: a Tag pool truncated by its cap exposes the terminal
    rows, and they differ from the observation after the same-step reset."""
    env = l2._make("tessera/Tag-v1", 2, 3, {s: "heuristic" for s in range(1, 8)}, {"max_episode_steps": 3})
    step = env.reset()
    for _ in range(3):
        step = env.step(np.zeros((2, env.agents), dtype=np.int32))
    assert np.asarray(step.episode_truncated).all()
    final = l2.terminal_obs(env, 0)
    after = l2.flat_obs(step.observations, 0)
    assert final.shape == after.shape
    assert not np.array_equal(final, after)
    env.close()


def test_t3b_calibrate_uses_ten_seeds_and_their_minimum():
    """rev's ruling: T3b calibrates on 101..110 with the threshold at their
    minimum; the other tiers keep 40 seeds and the 5th percentile."""
    from learning_gate import CALIBRATION_SEEDS, T3B_CALIBRATION_SEEDS, calibrate  # noqa: PLC0415

    assert T3B_CALIBRATION_SEEDS == list(range(101, 111))
    records = [{"seed": s, "score": float(s - 100)} for s in T3B_CALIBRATION_SEEDS]
    out = calibrate(records, T3B_CALIBRATION_SEEDS, expected=T3B_CALIBRATION_SEEDS, threshold="min")
    assert out["threshold"] == 1.0 and out["threshold_rule"] == "min" and "errors" not in out
    # The default is unchanged: 40 seeds, 5th percentile, and 10 seeds are refused.
    assert "errors" in calibrate(records, T3B_CALIBRATION_SEEDS)
    forty = [{"seed": s, "score": float(s)} for s in CALIBRATION_SEEDS]
    assert calibrate(forty, CALIBRATION_SEEDS)["threshold_rule"] == "p5"
    assert "errors" in calibrate(records, T3B_CALIBRATION_SEEDS, expected=T3B_CALIBRATION_SEEDS, threshold="median")


BOARD = {"tessera/Connect4-v1": 2, "tessera/Othello-v1": 4, "tessera/Go9-v1": 4}


@pytest.mark.parametrize("env_id", sorted(BOARD))
def test_twoplayer_openings_are_deterministic_and_distinct(env_id):
    """Paired comparisons need the same opening for every player kind; a
    diverse evaluation needs different openings across episodes."""
    a, skipped_a = l2.openings(env_id, {}, 10, 10_000, BOARD[env_id])
    b, skipped_b = l2.openings(env_id, {}, 10, 10_000, BOARD[env_id])
    assert a == b and skipped_a == skipped_b
    present = [r for r in a if r is not None]
    assert len(set(present)) >= 8, f"{env_id}: {len(set(present))} distinct openings of 10"
    assert l2.openings(env_id, {}, 3, 10_000, 0) == ([None, None, None], 0)


@pytest.mark.torch
@pytest.mark.parametrize("env_id", sorted(BOARD))
def test_twoplayer_trained_vs_heuristic_games_differ(env_id):
    """rev's diversity check, against the heuristic (the uniform-player check
    could not see a deterministic evaluation): a greedy agent against the
    heuristic plays different games across a seed's evaluation episodes."""
    pytest.importorskip("torch")
    probe = l2._make(env_id, 1, 0, None, {})
    obs = l2.flat_obs(probe.reset().observations, 0)
    agent = l2._agent_class()(obs.shape[1], probe.action_choices)
    probe.close()
    trace = []
    l2.eval_scores(env_id, {}, [0, 1], "trained", "heuristic", 10, 10_000, "sign", agent, opening_plies=BOARD[env_id], trace=trace)
    for seat in (0, 1):
        games = {d for s, _, d in trace if s == seat}
        assert len(games) >= 4, f"{env_id} seat {seat}: {len(games)} distinct games of 5"


def test_t3b_calibrate_refuses_too_many_skipped_openings():
    """REVIEW-T3b-opening-r1: at most 5% of a seed's openings may be skipped."""
    from learning_gate import MAX_OPENING_SKIP_SHARE, opening_skip_errors  # noqa: PLC0415

    assert MAX_OPENING_SKIP_SHARE == 0.05
    assert opening_skip_errors([{"seed": 101, "opening_draws": 25, "opening_skips": 1}]) == []
    errs = opening_skip_errors([{"seed": 102, "opening_draws": 25, "opening_skips": 2}])
    assert errs and "seed 102: 2 of 25" in errs[0]
    # No openings (Leduc, Tag): nothing to check.
    assert opening_skip_errors([{"seed": 103, "opening_draws": 0, "opening_skips": 0}]) == []


def test_t3b_missing_opening_counts_is_an_error():
    """REVIEW-T3b-skips-r1: with openings on, missing counts are an error."""
    from learning_gate import opening_skip_errors  # noqa: PLC0415

    errs = opening_skip_errors([{"seed": 104, "opening_plies": 2}])
    assert errs and "seed 104: opening_plies 2 but no opening_draws" in errs[0]
    assert opening_skip_errors([{"seed": 105, "opening_plies": 0}]) == []


def test_t3b_check_calibration_accepts_ten_seeds_with_the_min_rule():
    """The nightly's validity check (check_calibration) recomputes a T3b
    calibration with its own seed list and rule; a wrong list is refused."""
    from learning_gate import T3B_CALIBRATION_SEEDS, calibrate, check_calibration  # noqa: PLC0415

    records = [{"seed": s, "score": float(s - 100)} for s in T3B_CALIBRATION_SEEDS]
    cal = calibrate(records, T3B_CALIBRATION_SEEDS, expected=T3B_CALIBRATION_SEEDS, threshold="min")
    cal.update({"records": records, "setup": {"env": "tessera/Connect4-v1", "total_steps": 1_000_000, "eval_episodes": 50, "fingerprint": "f"}})
    assert check_calibration(cal) == []
    short = dict(cal, seeds=T3B_CALIBRATION_SEEDS[:9], records=records[:9])
    assert check_calibration(short)


def test_twoplayer_terminal_obs_errors_propagate():
    """#571 (REVIEW-design-r1 condition 3): without same_step auto-reset the
    accessor's ArgumentError stops the run; there is no fallback."""
    import tessera_rl

    env = tessera_rl.make("tessera/Tag-v1", num_envs=1, seed=3, workers=0, mode="sync", outputs="owned", auto_reset="next_step")
    env.reset()
    env.step(np.zeros((1, env.agents), dtype=np.int32))
    with pytest.raises(tessera_rl.ArgumentError, match="same_step"):
        l2.terminal_obs(env, 0)
    env.close()


class _Layout:
    def __init__(self, nvec, mask_heads, mask_len):
        self.action_nvec, self.mask_heads, self._mask_len = tuple(nvec), tuple(mask_heads), mask_len


def test_twoplayer_action_heads_of_survival_split_the_mask_row():
    env = l2._make("tessera/Survival-v1", 1, 5, None, {"agents": 16, "size": 32})
    heads = l2.action_heads(env)
    assert heads == [(0, 5), (5, 7)]
    step = env.reset()
    row = l2.slot_mask(step.masks, 0, 1, 12)[0]
    move, act = l2.head_masks(row, heads)
    np.testing.assert_array_equal(move, row[:5])
    np.testing.assert_array_equal(act, row[5:12])
    env.close()


def test_twoplayer_action_heads_refuse_parents_and_out_of_row_blocks():
    with pytest.raises(ValueError, match="conditional action heads"):
        l2.action_heads(_Layout([3, 2], [(0, (), 3), (3, (3,), 2)], 5))
    with pytest.raises(ValueError, match="outside the mask row"):
        l2.action_heads(_Layout([3, 4], [(0, (), 3), (3, (), 4)], 6))
    assert l2.action_heads(_Layout([3, 4], [], 0)) == [(0, 3), (3, 4)]


def test_twoplayer_live_heads_needs_every_head():
    heads = [(0, 2), (2, 3)]
    mask = np.array([[1, 0, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=bool)
    np.testing.assert_array_equal(l2.live_heads(mask, heads), [True, False, False])


def test_twoplayer_uniform_heads_draw_in_head_order():
    heads = [(0, 5), (5, 7)]
    mask = np.ones(12, dtype=bool)
    a = l2.uniform_legal_heads(np.random.default_rng(9), mask, heads)
    rng = np.random.default_rng(9)
    assert a == [l2.uniform_legal(rng, mask[:5]), l2.uniform_legal(rng, mask[5:])]


@pytest.mark.torch
def test_twoplayer_multihead_logprob_and_entropy_are_sums_over_heads():
    torch = pytest.importorskip("torch")
    from torch.distributions.categorical import Categorical

    torch.manual_seed(0)
    heads = [(0, 2), (2, 3)]
    agent = l2._multihead_agent_class()(4, heads)
    x = torch.randn(3, 4)
    mask = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=torch.bool)
    action = torch.tensor([[0, 2], [0, 1], [1, 1]])
    _, logp, ent, _ = agent.act(x, mask, action)
    logits = agent.actor(x)
    want_lp, want_ent = 0.0, 0.0
    for h, (lo, hi) in enumerate(((0, 2), (2, 5))):
        lg, m = logits[:, lo:hi], mask[:, lo:hi]
        want_lp = want_lp + Categorical(logits=l2.masked_logits(lg, m)).log_prob(action[:, h])
        want_ent = want_ent + l2.legal_entropy(lg, m)
    torch.testing.assert_close(logp, want_lp)
    torch.testing.assert_close(ent, want_ent)
    with pytest.raises(AssertionError, match="no legal choice"):
        agent.act(x, torch.tensor([[1, 1, 0, 0, 0]] * 3, dtype=torch.bool), action)


@pytest.mark.torch
def test_twoplayer_multihead_survival_one_update_smoke():
    pytest.importorskip("torch")
    import math

    h = l2.TwoPlayerHyper(total_steps=64, num_envs=2, num_steps=32, num_minibatches=2, update_epochs=1)
    kwargs = {"agents": 16, "size": 32, "respawn": 4, "steps": 32}
    agent, learner, rec = l2.train("tessera/Survival-v1", kwargs, 101, [0], h)
    assert learner == 0 and rec["curve"]
    scores = l2.eval_scores("tessera/Survival-v1", kwargs, [0], "trained", "heuristic", 2, 1000, "return", agent=agent)
    assert len(scores) == 2 and all(math.isfinite(s) for s in scores)

# SPDX-License-Identifier: MIT OR Apache-2.0
"""The limit order book's PPO protocol script (tools/lob_training.py,
#381): its protocol table, the noise trader's legality, the pass criterion,
and a one-update smoke run of the training loop (torch only)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import lob_training as lt  # noqa: E402


def test_lob_protocol_table_has_the_script_fields():
    p = lt.protocol()
    assert p["env"] == lt.ENV_ID
    assert p["tier"] == "T3"
    assert {"config", "seeds", "eval_episodes", "total_steps", "hyper"} <= set(p)
    assert lt.Hyper(**p["hyper"]).total_steps > 0


def test_lob_noise_trader_acts_legally_under_the_masks():
    env = lt.make_env(5)
    rng = np.random.default_rng(1)
    obs, infos = env.reset()
    acted = 0
    for _ in range(40):
        acts = {}
        for a in env.agents:
            masks = lt.masks_of(infos[a])
            act = lt.noise_action(rng, masks)
            k, o, s, c = (int(v) for v in act)
            assert masks[0][k] and masks[1][k, o] and masks[2][k, o, s] and masks[3][k, c]
            acted += int(k != 0)
            acts[a] = act
        obs, _, _, _, infos = env.step(acts)
        if not env.agents:
            break
    assert acted > 0
    env.close()


def test_lob_learner_env_shows_slot_0_and_engine_plays_1_to_7():
    """#470: the opponents are heuristic slots, hidden from the learner."""
    env = lt.make_env(3)
    _, infos = env.reset()
    assert env.agents == [lt.LEARNER]
    assert infos[lt.LEARNER]["policy_group"] == "learner"
    _, _, _, _, infos = env.step({lt.LEARNER: np.zeros(4, dtype=np.int64)})
    assert set(infos[lt.LEARNER]["heuristic_rewards"]) == {f"agent_{s}" for s in range(1, 8)}
    env.close()


def test_lob_engine_baseline_scores_are_finite():
    scores = lt.engine_baseline_scores(1)
    assert len(scores) == 1 and np.isfinite(scores[0])


def test_lob_features_are_finite_and_squashed():
    env = lt.make_env(2)
    obs, _ = env.reset()
    x = lt.features(obs[lt.LEARNER])
    assert x.dtype == np.float32 and np.isfinite(x).all()
    assert np.abs(x).max() < 20
    env.close()


def test_lob_pass_criterion_needs_the_interval_above_the_baseline():
    assert lt.passes([5.0, 6.0, 7.0, 5.5, 6.5], 1.0)["pass"]
    assert not lt.passes([0.0, 2.0, -1.0, 1.0, 0.5], 1.0)["pass"]


@pytest.mark.torch
def test_lob_training_one_update_smoke():
    pytest.importorskip("torch")
    hyper = lt.Hyper(total_steps=64, num_steps=64, num_minibatches=2, update_epochs=1)
    run = lt.train(0, hyper, eval_episodes=1)
    assert len(run["scores"]) == 1 and np.isfinite(run["scores"][0])

# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T1s: T1's training with a sampled evaluation
(tools/learning_sampled.py, #595)."""

import json
import pathlib
import sys
import threading

import pytest

# learning_sampled trains through reference_training, which imports torch.
pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

GATHER = "tessera/Gather-v1"
STEPS = 512  # one T1 rollout: 4 envs x 128 steps


def _agent(rt, seed=0):
    """An untrained agent sized from the protocol's evaluation environment."""
    import numpy as np
    import torch

    env = rt.make_single("tessera")
    obs_dim, actions = int(np.prod(env.observation_space.shape)), int(env.action_space.n)
    env.close()
    torch.manual_seed(seed)
    return rt.Agent(obs_dim, actions)


def test_learning_t1s_evaluation_samples_and_is_reproducible():
    import learning_protocols as lp
    import learning_sampled as ls

    with lp._factories(GATHER) as rt:
        agent = _agent(rt)
        greedy = rt.evaluate("tessera", agent, 6)
        a = ls.evaluate_sampled(rt, agent, 6, seed=1)
        b = ls.evaluate_sampled(rt, agent, 6, seed=1)
        c = ls.evaluate_sampled(rt, agent, 6, seed=2)
    assert a == b
    assert a != greedy or a != c, (a, greedy, c)
    # Raw returns: Gather pays +1 per item, 8 items.
    assert all(float(r).is_integer() and 0.0 <= r <= 8.0 for r in a + c)


def test_learning_t1s_episode_seeds_are_distinct():
    import learning_sampled as ls

    seeds = {ls.episode_seed(s, 10_000, i) for s in range(1, 41) for i in range(20)}
    assert len(seeds) == 40 * 20


def test_learning_t1s_restores_evaluate_after_a_run_and_after_a_raise():
    import learning_protocols as lp
    import learning_sampled as ls
    import reference_training as rt

    original = rt.evaluate
    with lp._factories(GATHER) as r, ls._sampled_evaluation(r, 1):
        assert rt.evaluate is not original
    assert rt.evaluate is original
    with pytest.raises(ZeroDivisionError):
        with lp._factories(GATHER) as r, ls._sampled_evaluation(r, 1):
            raise ZeroDivisionError
    assert rt.evaluate is original
    assert ls._SWAP.acquire(blocking=False)
    ls._SWAP.release()


def test_learning_t1s_refuses_a_nested_or_concurrent_swap():
    import learning_protocols as lp
    import learning_sampled as ls
    import reference_training as rt

    original = rt.evaluate
    with lp._factories(GATHER) as r, ls._sampled_evaluation(r, 1):
        with pytest.raises(RuntimeError, match="already in progress"):
            with ls._sampled_evaluation(r, 2):
                pass
        errors = []

        def other():
            try:
                with ls._sampled_evaluation(r, 3):
                    pass
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert len(errors) == 1
    assert rt.evaluate is original


def test_learning_t1s_training_is_t1s_own_with_the_greedy_evaluator(monkeypatch):
    """With the sampled evaluator patched to T1's greedy one, a T1s run is
    exactly T1's training and evaluation for the same seed and budget: the
    swap changes the evaluation only, and train runs the swapped one."""
    import learning_protocols as lp
    import learning_sampled as ls
    import reference_training as rt

    h = lp.hyper("gather", STEPS)
    with lp._factories(GATHER) as r:
        t1 = r.train("tessera", 3, h, 2)
    greedy = rt.evaluate
    monkeypatch.setattr(ls, "evaluate_sampled", lambda rt_, agent, episodes, seed: greedy("tessera", agent, episodes))
    t1s = ls.train(GATHER, 3, h, 2)
    for key in ("curve", "eval_returns", "episodes"):
        assert t1s[key] == t1[key], key


def test_learning_t1s_train_runs_the_sampled_evaluator(monkeypatch):
    import learning_protocols as lp
    import learning_sampled as ls

    calls = []
    monkeypatch.setattr(ls, "evaluate_sampled", lambda rt_, agent, episodes, seed: calls.append(seed) or [0.0] * episodes)
    r = lp.train_sampled("gather", 4, STEPS, 3)
    assert calls == [4] and r["eval_returns"] == [0.0, 0.0, 0.0]


def test_learning_t1s_is_deterministic_and_scores_raw_returns():
    import learning_protocols as lp

    a = lp.train_sampled("gather", 5, STEPS, 3)
    b = lp.train_sampled("gather", 5, STEPS, 3)
    assert a["eval_returns"] == b["eval_returns"] and a["curve"] == b["curve"]
    assert all(float(r).is_integer() and 0.0 <= r <= 8.0 for r in a["eval_returns"])


def test_learning_t1s_dispatch_and_gather_protocol():
    import learning_protocols as lp

    with pytest.raises(ValueError, match="not T1s"):
        lp.train_sampled("acrobot", 1, STEPS, 1)
    p = lp.protocol("gather")
    assert p["tier"] == "T1s" and p["total_steps"] == 150_000 and p["hyper"] == {}
    assert lp.gate("gather")["solved"] == 8.0


def test_learning_t1s_fingerprint_scope():
    import learning_protocols as lp
    import learning_sampled as ls
    import reference_training as rt

    before = {n: lp.fingerprint(n) for n in ("gather", "acrobot")}
    real = ls.__file__
    try:
        ls.__file__ = lp.__file__  # any other source
        assert lp.fingerprint("gather") != before["gather"]
        assert lp.fingerprint("acrobot") == before["acrobot"]
    finally:
        ls.__file__ = real
    real_rt = rt.__file__
    try:
        rt.__file__ = lp.__file__
        assert lp.fingerprint("gather") != before["gather"]
        assert lp.fingerprint("acrobot") != before["acrobot"]
    finally:
        rt.__file__ = real_rt
    assert {n: lp.fingerprint(n) for n in before} == before
    assert lp.train_sampled not in lp._training_path()


def test_learning_t1s_leaves_the_calibrated_t1_fingerprints():
    import learning_protocols as lp

    for name in ("acrobot", "snake", "paddleduel"):
        cal = json.loads((ROOT / "compat" / "learning" / f"{name}.json").read_text())
        assert lp.fingerprint(name) == cal["setup"]["fingerprint"], name

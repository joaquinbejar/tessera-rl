# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tier T1n: discrete PPO with observation and reward normalization
(tools/learning_normalized.py, #530)."""

import inspect
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def test_learning_t1n_return_norm_scales_by_the_return_std_and_resets_on_done():
    np = pytest.importorskip("numpy")
    import learning_normalized as ln

    scale = ln.ReturnNorm(2, gamma=0.5, clip=10.0)
    out = scale([1.0, 2.0], [False, True])
    # Returns are 1 and 2 after one step; the reward is divided by their std,
    # not centred.
    std = np.sqrt(scale.stats.var[0] + 1e-8)
    assert np.allclose(out, np.array([1.0, 2.0]) / std, atol=1e-6)
    # Instance 1 ended: its return accumulator restarts; instance 0's carries.
    assert scale.returns.tolist() == [1.0, 0.0]
    scale([1.0, 1.0], [False, False])
    assert scale.returns.tolist() == [1.5, 1.0]


def test_learning_t1n_return_norm_clips():
    np = pytest.importorskip("numpy")
    import learning_normalized as ln

    scale = ln.ReturnNorm(1, gamma=0.99, clip=2.0)
    for _ in range(50):
        scale([0.001], [False])
    assert np.all(np.abs(scale([1e9], [False])) <= 2.0)


def test_learning_t1n_evaluation_needs_frozen_statistics():
    import learning_continuous as lc
    import learning_normalized as ln

    with pytest.raises(ValueError, match="frozen"):
        ln.evaluate("tessera/MountainCar-v1", None, lc.RunningNorm(2), 1, 0)


def test_learning_t1n_refuses_a_non_flat_observation_space():
    pytest.importorskip("torch")
    import learning_normalized as ln

    with pytest.raises(ValueError, match="flat Box"):
        ln.train("tessera/BrickWall-v1", 1, ln.NormalizedHyper(total_steps=256), 1, 0)


def test_learning_t1n_mountaincar_protocol_is_rl_zoo_entry():
    import learning_protocols as lp

    p = lp.protocol("mountaincar")
    assert p["tier"] == "T1n" and p["total_steps"] == 1_000_000
    h = lp.normalized_hyper("mountaincar", p["total_steps"])
    assert (h.num_envs, h.num_steps, h.gae_lambda, h.update_epochs, h.ent_coef) == (16, 16, 0.98, 4, 0.0)
    # RL Zoo's batch_size 64 over a 256-step rollout.
    assert h.num_envs * h.num_steps // h.num_minibatches == 64
    assert not h.anneal_lr and h.learning_rate == 3e-4
    assert lp.gate("mountaincar") == {"solved": -110.0, "min_effect": 0.10}


def test_learning_t1n_dispatch_refuses_other_tiers():
    import learning_protocols as lp

    with pytest.raises(ValueError, match="not T1n"):
        lp.train_normalized("acrobot", 1, 256, 1)


def test_learning_t1n_fingerprint_covers_its_module_and_imports_only():
    pytest.importorskip("torch")
    import learning_normalized as ln
    import learning_protocols as lp

    before = lp.fingerprint("mountaincar")
    real = ln.__file__
    try:
        ln.__file__ = lp.__file__  # any other source
        assert lp.fingerprint("mountaincar") != before
    finally:
        ln.__file__ = real
    real_imported = ln.IMPORTED
    try:
        ln.IMPORTED = real_imported[:-1]
        assert lp.fingerprint("mountaincar") != before
    finally:
        ln.IMPORTED = real_imported
    assert lp.fingerprint("mountaincar") == before
    # T1's flattening is not in T1n's scope (#564 edits it).
    assert lp._FlatVector not in ln.IMPORTED and lp._needs_flat not in ln.IMPORTED
    assert "_FlatVector" not in inspect.getsource(ln)


def test_learning_t1n_leaves_the_t1_training_path_unchanged():
    import learning_protocols as lp

    assert lp._training_path() == (lp.hyper, lp._needs_flat, lp._flat_vector_class, lp._FlatVector, lp._factories, lp.train)
    assert lp.train_normalized not in lp._training_path()


def test_learning_t1n_calibrated_t1_fingerprints_still_match():
    pytest.importorskip("torch")
    import learning_protocols as lp

    for name in ("acrobot",):
        cal = json.loads((ROOT / "compat" / "learning" / f"{name}.json").read_text())
        assert lp.fingerprint(name) == cal["setup"]["fingerprint"], name


def test_learning_t1n_mountaincar_trains_and_is_deterministic():
    pytest.importorskip("torch")
    import learning_protocols as lp

    a = lp.train_normalized("mountaincar", 3, 512, 1)
    b = lp.train_normalized("mountaincar", 3, 512, 1)
    assert a["eval_returns"] == b["eval_returns"]
    assert -200.0 <= a["eval_mean"] <= 0.0
    assert [c["steps"] for c in a["curve"]] == [256, 512]

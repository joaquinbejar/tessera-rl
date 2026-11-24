# SPDX-License-Identifier: MIT OR Apache-2.0
"""Lanterns from Python: ``make`` accepts ``zero_on_fuse`` as a bool and
refuses any other type (#468)."""

import numpy as np
import pytest

import tessera_rl as t


def make(**kw):
    return t.make("tessera/Lanterns-v1", num_envs=2, seed=3, outputs="owned", **kw)


@pytest.mark.parametrize("value", [False, True, np.bool_(True)])
def test_lanterns_make_zero_on_fuse_accepts_bools(value):
    env = make(players=3, zero_on_fuse=value)
    s = env.reset()
    assert s.acting.shape == (2, 5)
    assert s.acting.tolist()[0][:3] == [True, False, False]
    env.close()


@pytest.mark.parametrize("value", [0, 1, "yes"])
def test_lanterns_make_zero_on_fuse_refuses_non_bools(value):
    with pytest.raises(TypeError, match="zero_on_fuse must be a bool"):
        make(zero_on_fuse=value)


def test_every_native_bool_option_is_a_strict_bool_option():
    """Every option a native maker reads as a bool (``kw.bool`` in the
    bindings) is in ``_BOOL_OPTIONS``, so ``make`` passes it through (#468)."""
    import pathlib
    import re

    from tessera_rl import _env

    root = pathlib.Path(__file__).resolve().parents[2]
    source = (root / "crates" / "tessera-rl-py" / "src" / "api.rs").read_text()
    found = set(re.findall(r'kw\.bool\(g, "([a-z_]+)"\)', source))
    assert found, "no bool option found: the pattern no longer matches the bindings"
    assert found <= _env._BOOL_OPTIONS, found - _env._BOOL_OPTIONS


def _play_first_slot(zero_on_fuse, seed):
    """Every player plays its first slot until the game ends: the episode's
    rewards and its final score (the sum of the gains)."""
    env = t.make(
        "tessera/Lanterns-v1",
        num_envs=1,
        seed=seed,
        outputs="owned",
        auto_reset="disabled",
        players=2,
        zero_on_fuse=zero_on_fuse,
    )
    env.reset()
    total, gains = 0.0, 0.0
    for _ in range(200):
        s = env.step(np.zeros((1, 5), dtype=np.int32))
        r = float(s.rewards[0, 0])
        total += r
        gains += max(r, 0.0)
        if bool(s.episode_terminated[0]):
            break
    env.close()
    return total, gains


def test_lanterns_zero_on_fuse_takes_the_score_back_on_the_third_fuse():
    # A seed where blind plays score before the third fuse.
    seed = next(s for s in range(100) if _play_first_slot(False, s)[0] > 0)
    plain, _ = _play_first_slot(False, seed)
    zeroed, gains = _play_first_slot(True, seed)
    assert gains == plain > 0
    assert zeroed == 0.0, "the third fuse takes the score back"

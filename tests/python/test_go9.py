# SPDX-License-Identifier: MIT OR Apache-2.0
"""Go 9x9 from Python: ``make`` builds it with sequential turns, 82-action
masks and a configurable komi; two passes end the game with komi deciding an
empty board; the PettingZoo AEC adapter passes the official API tests and is
never parallel."""

import numpy as np
import pytest
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import _native, pz

GO = "tessera/Go9-v1"
PASS = 81


def make(n=1, **kw):
    return t.make(GO, num_envs=n, seed=5, outputs="owned", auto_reset="disabled", **kw)


def test_go9_make_is_sequential_with_82_action_masks():
    env = make(2)
    assert env.turns == "sequential"
    assert env.agents == 2 and env.action_choices == 82
    s = env.reset()
    assert s.observations["board"].shape == (2, 2, 2, 9, 9)
    assert s.acting.tolist() == [[True, False]] * 2
    assert s.masks[0, 0].all(), "every point and the pass"
    env.close()


@pytest.mark.parametrize(("komi", "black"), [(None, -1.0), (0, 0.0), (-3, 1.0)])
def test_go9_make_two_passes_score_the_empty_board_by_komi(komi, black):
    env = make(**({} if komi is None else {"komi_half_points": komi}))
    env.reset()
    a = np.full((1, 2), PASS, dtype=np.int32)
    env.step(a)
    s = env.step(a)
    assert bool(s.episode_terminated[0])
    assert s.rewards[0].tolist() == [black, -black]
    env.close()


def test_go9_make_rejects_komi_out_of_range():
    with pytest.raises(t.ConfigError):
        make(komi_half_points=163)


def test_pz_go9_passes_the_aec_api_and_seed_tests():
    pz_test.api_test(pz.aec_env(GO, seed=1), num_cycles=300, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(GO), num_cycles=200)


def test_pz_go9_is_never_parallel():
    with pytest.raises(_native.ArgumentError):
        pz.parallel_env(GO)

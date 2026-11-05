# SPDX-License-Identifier: MIT OR Apache-2.0
"""Othello from Python: ``make`` builds it with sequential turns, 65-action
masks and the observation layout; the published nine-ply wipeout pays the
winner; the PettingZoo AEC adapter passes the official API tests and is
never parallel."""

import numpy as np
import pytest
from pettingzoo import test as pz_test

import tessera_rl as t
from tessera_rl import _native, pz

OTHELLO = "tessera/Othello-v1"
PASS = 64
OPENING = [19, 26, 37, 44]
# f5 d6 c5 f4 e3 f6 g5 e6 e7: white holds no disc at the end.
WIPEOUT = [37, 43, 34, 29, 20, 45, 38, 44, 52]


def make(n=1, **kw):
    return t.make(OTHELLO, num_envs=n, seed=5, outputs="owned", auto_reset="disabled", **kw)


def test_othello_make_is_sequential_with_65_action_masks():
    env = make(2)
    assert env.turns == "sequential"
    assert env.agents == 2 and env.action_choices == 65
    s = env.reset()
    board = s.observations["board"]
    assert board.shape == (2, 2, 2, 8, 8) and board.dtype == np.uint8
    assert s.acting.tolist() == [[True, False]] * 2
    assert np.flatnonzero(s.masks[0, 0]).tolist() == OPENING
    env.close()


def test_othello_make_wipeout_pays_the_winner_and_terminates():
    env = make()
    env.reset()
    for i, square in enumerate(WIPEOUT):
        a = np.full((1, 2), PASS, dtype=np.int32)
        a[0, i % 2] = square
        s = env.step(a)
    assert bool(s.episode_terminated[0])
    assert s.rewards[0].tolist() == [1.0, -1.0]
    env.close()


def test_pz_othello_passes_the_aec_api_and_seed_tests():
    pz_test.api_test(pz.aec_env(OTHELLO, seed=1), num_cycles=300, verbose_progress=False)
    pz_test.seed_test(lambda: pz.aec_env(OTHELLO), num_cycles=200)


def test_pz_othello_is_never_parallel():
    with pytest.raises(_native.ArgumentError):
        pz.parallel_env(OTHELLO)

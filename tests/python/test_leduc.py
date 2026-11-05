# SPDX-License-Identifier: MIT OR Apache-2.0
"""Leduc hold'em from Python: ``make`` builds it with sequential turns,
three-action masks and the observation layout; a scripted hand pays
normalized chips (times 13 is chips), zero-sum."""

import numpy as np

import tessera_rl as t

FOLD, CALL, RAISE = 0, 1, 2


def make(n=1, **kw):
    return t.make("tessera/Leduc-v1", num_envs=n, seed=5, outputs="owned", auto_reset="disabled", **kw)


def act(to_move, action):
    a = np.full((1, 2), CALL, dtype=np.int32)
    a[0, to_move] = action
    return a


def test_leduc_make_is_sequential_with_three_action_masks():
    env = make(2)
    assert env.turns == "sequential"
    assert env.agents == 2 and env.action_choices == 3
    s = env.reset()
    view = s.observations["view"]
    assert view.shape == (2, 2, 11) and view.dtype == np.uint8
    # Slot 0 to move, antes of 1 each, no public card (3).
    assert list(view[0, 0, 1:7]) == [3, 0, 1, 1, 1, 0]
    assert view[0, 1, 3] == 0, "slot 1 is not to move"
    assert view[0, 0, 0] < 3 and view[0, 1, 0] < 3
    assert s.acting.tolist() == [[True, False]] * 2
    assert s.masks.tolist()[0][0] == [False, True, True], "no bet to fold against"
    env.close()


def test_leduc_make_scripted_fold_pays_normalized_chips():
    env = make()
    env.reset()
    env.step(act(0, RAISE))
    s = env.step(act(1, FOLD))
    assert bool(s.episode_terminated[0])
    assert np.isclose(s.rewards[0, 0] * 13, 1.0) and np.isclose(s.rewards[0, 1] * 13, -1.0)
    assert s.rewards[0].sum() == 0.0
    env.close()

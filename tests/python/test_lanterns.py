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

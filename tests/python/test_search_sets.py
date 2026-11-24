# SPDX-License-Identifier: MIT OR Apache-2.0
"""Search over a ``Set`` observation from Python (ADR-0082 point 7 as
amended, #490) on ``tessera/Gather-v1``."""

import numpy as np

import tessera_rl as t
from tessera_rl.search import Search


def test_search_gather_names_the_set_and_its_leaves():
    s = Search(env_id="tessera/Gather-v1", simulations=16)
    assert s.actions == 4
    # agent, items, items_mask.
    assert s.sets == [("items", 8, [1, 2])]
    assert [name for name, _, _ in s._leaves] == ["agent", "items", "items_mask"]


def test_search_gather_evaluator_gets_set_leaves_with_absent_slots_zero():
    env = t.make("tessera/Gather-v1", num_envs=1, seed=4, outputs="owned", auto_reset="disabled")
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.step(rng.integers(0, 4, size=(1, 1)).astype(np.int32))
    s = Search(env_id="tessera/Gather-v1", simulations=48)
    s.set_root(env.snapshot(0))
    seen = []

    def evaluator(obs, legal, hashes):
        n = legal.shape[0]
        assert set(obs) == {"agent", "items", "items_mask"}
        assert obs["items"].shape == (n, 8, 2) and obs["items"].dtype == np.int32
        assert obs["items_mask"].shape == (n, 8) and obs["items_mask"].dtype == np.bool_
        absent = ~obs["items_mask"]
        assert not obs["items"][absent].any(), "absent slots must be zero"
        seen.append(int(absent.sum()))
        return np.full((n, 4), 0.25), np.zeros(n)

    s.run(evaluator)
    assert int(s.root_visits().sum()) == 48
    assert seen
    env.close()

# SPDX-License-Identifier: MIT OR Apache-2.0
"""#311: for an environment with several masked discrete heads, the
PettingZoo adapters' ``infos[agent]["action_mask"]`` is a tuple of per-head
masks that ``MultiDiscrete.sample`` accepts (a conditional head relaxed to
the union over its parents), and ``"action_masks"`` keeps the exact masks.
Over the two-head masked host environment of the test host wheel
(``test/MultiHead-v1``, ``make py-test-host``) plus synthetic masks."""

import numpy as np
import pytest

pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")

ID = "test/MultiHead-v1"


def test_relaxed_masks_flat_heads_are_the_heads_themselves():
    row = np.array([1, 1, 0, 1, 0, 1, 1], dtype=np.bool_)
    got = pz.relaxed_masks(row, ((0, (), 3), (3, (), 4)))
    assert [m.tolist() for m in got] == [[1, 1, 0], [1, 0, 1, 1]]
    assert all(m.dtype == np.int8 for m in got)


def test_relaxed_masks_conditional_head_is_the_union_and_can_admit_an_illegal_action():
    # Root head of 2 choices; child head of 3 choices under each root choice:
    # root 0 allows child 0, root 1 allows child 1.
    root = [1, 1]
    child = [[1, 0, 0], [0, 1, 0]]
    row = np.array(root + child[0] + child[1], dtype=np.bool_)
    got = pz.relaxed_masks(row, ((0, (), 2), (2, (2,), 3)))
    assert [m.tolist() for m in got] == [[1, 1], [1, 1, 0]]
    # (root 0, child 1) passes the relaxed masks but is illegal exactly.
    assert got[0][0] and got[1][1] and not child[0][1]


def test_relaxed_masks_two_parents_reduce_over_both():
    block = np.zeros((2, 2, 3), dtype=np.bool_)
    block[1, 0, 2] = True
    row = np.concatenate([[True, True], block.ravel()])
    got = pz.relaxed_masks(row, ((0, (), 2), (2, (2, 2), 3)))
    assert got[1].tolist() == [0, 0, 1]


def _host():
    return pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")


def test_multihead_infos_carry_the_relaxed_tuple_and_the_exact_masks():
    _host()
    env = pz.parallel_env(ID, seed=3)
    obs, infos = env.reset(seed=3)
    # Each agent gets its own masks: agent 1 also has head 0's choice 0 illegal.
    want = {"agent_0": [[1, 1, 0], [1, 0, 1, 1]], "agent_1": [[0, 1, 0], [1, 0, 1, 1]]}
    assert env.agents == ["agent_0", "agent_1"]
    for a in env.agents:
        relaxed, exact = infos[a]["action_mask"], infos[a]["action_masks"]
        assert isinstance(relaxed, tuple) and [m.tolist() for m in relaxed] == want[a]
        # The exact masks keep tessera_rl.masks' entity axis (one entity).
        assert [m.tolist() for m in exact] == [[row] for row in want[a]]
        assert "action_mask" not in obs[a]
    env.close()


def test_multihead_sample_with_the_relaxed_tuple_is_legal_for_flat_heads():
    _host()
    env = pz.parallel_env(ID, seed=3)
    _, infos = env.reset(seed=3)
    space = env.action_space("agent_1")
    space.seed(1)
    for _ in range(50):
        a0, a1 = space.sample(infos["agent_1"]["action_mask"])
        assert a0 == 1 and a1 != 1
    env.close()


def test_multihead_official_aec_api_test():
    _host()
    from pettingzoo.test import api_test

    env = pz.aec_env(ID, seed=3)
    api_test(env, num_cycles=30)
    env.close()


def test_multihead_official_parallel_api_test():
    _host()
    from pettingzoo.test import parallel_api_test

    env = pz.parallel_env(ID, seed=3)
    parallel_api_test(env, num_cycles=30)
    env.close()

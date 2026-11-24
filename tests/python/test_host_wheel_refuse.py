# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host protocol 2 through a compiled host wheel (ADR-0070): the test host
wheel's ``test/Refuser-v1`` (``make py-test-host``) speaks protocol 2,
exports the ``refused`` and ``refuse_code`` views, and the wrapper and
adapters report them."""

import numpy as np
import pytest

import tessera_rl

ID = "test/Refuser-v1"


def _host():
    return pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")


def test_host_wheel_refuser_speaks_protocol_2():
    host = _host()
    obj = host.make_refuser(num_envs=2, seed=0)
    assert obj.host_protocol == 2
    assert obj.layout()["refuses_agents"] is True
    out = obj.reset(None)
    assert np.asarray(out["refused"]).shape == (2, 2)
    assert np.asarray(out["refuse_code"]).dtype == np.uint32
    obj.close()
    counter = host.make(num_envs=1, seed=0)
    assert counter.host_protocol == 1
    assert counter.layout()["refuses_agents"] is False
    assert "refused" not in counter.reset(None)
    counter.close()


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_host_wheel_refuser_reports_refused_agents_and_steps_the_rest(outputs):
    _host()
    env = tessera_rl.make(ID, num_envs=3, seed=0, outputs=outputs)
    assert env.refuses_agents is True
    env.reset()
    out = env.step(np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int32))
    assert np.asarray(out.refused).tolist() == [[False, False], [True, False], [False, True]]
    assert np.asarray(out.refuse_code).tolist() == [[0, 0], [30, 0], [0, 31]]
    assert np.asarray(out.rewards).tolist() == [[1.5, 1.5], [-1.0, 0.5], [0.5, -1.0]]
    # The refusal is counted in the refused agent's observation.
    assert np.asarray(out.observations["obs"])[1, 0].tolist() == [1, 1]
    # The next call clears the rows.
    out = env.step(np.ones((3, 2), dtype=np.int32))
    assert not np.asarray(out.refused).any() and not np.asarray(out.refuse_code).any()
    env.close()


def test_host_wheel_refuser_pz_infos():
    _host()
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    env = pz.parallel_env(ID, seed=0)
    env.reset(seed=0)
    a, b = env.agents
    _, rewards, _, _, infos = env.step({a: 2, b: 1})
    assert infos[a] == {"refused": True, "refuse_code": 30}
    assert infos[b] == {"refused": False, "refuse_code": 0}
    assert (rewards[a], rewards[b]) == (-1.0, 1.5)
    env.close()

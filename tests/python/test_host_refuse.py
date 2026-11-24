# SPDX-License-Identifier: MIT OR Apache-2.0
"""Per-agent refusal through host protocol 2 (ADR-0070 item 10): the
wrapper, ``HostEnv``, the Gymnasium infos and the PettingZoo infos, on the
pure-Python fake host."""

from __future__ import annotations

import functools

import numpy as np
import pytest

import tessera_rl
import tessera_rl.gym
import tessera_rl.pz
from fake_host import FakeHost, install

FAKE = "fake/Counter-v1"


@pytest.fixture
def fake(monkeypatch):
    def setup(**overrides):
        factory = functools.partial(FakeHost, **{"protocol": 2, "refuses": True, **overrides})
        install(monkeypatch, {FAKE: [("fake-dist", factory)]})

    return setup


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_host_refuse_hostenv_reports_refused_agents(fake, outputs):
    fake()
    env = tessera_rl.make(FAKE, num_envs=3, outputs=outputs)
    assert env.refuses_agents is True
    first = env.reset(seed=1)
    assert not np.asarray(first.refused).any()
    acts = np.array([[1, 1], [2, 1], [1, 2]], dtype=np.int32)
    out = env.step(acts)
    assert np.asarray(out.refused).tolist() == [[False, False], [True, False], [False, True]]
    assert np.asarray(out.refuse_code).tolist() == [[0, 0], [30, 0], [0, 31]]
    # The other agents stepped.
    assert np.asarray(out.rewards).tolist() == [[1.5, 1.5], [-1.0, 1.5], [1.5, -1.0]]
    env.close()


def test_host_refuse_protocol_1_hostenv_has_no_refusal_fields(fake):
    fake(protocol=1, refuses=False)
    env = tessera_rl.make(FAKE, num_envs=2)
    assert env.refuses_agents is False
    env.reset()
    out = env.step(np.ones((2, 2), dtype=np.int32))
    assert out.refused is None and out.refuse_code is None
    env.close()


def test_host_refuse_gym_infos_carry_refusals(fake):
    fake(agents=1)
    env = tessera_rl.gym.make(FAKE, num_envs=2)
    assert env.metadata["refuses_agents"] is True
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([2, 1]))
    assert info["refused"].tolist() == [[True], [False]]
    assert info["refuse_code"].tolist() == [[30], [0]]
    assert info["_refused"].tolist() == [True, True]
    assert info["_refuse_code"].tolist() == [True, True]
    env.close()


def test_host_refuse_pz_parallel_infos_carry_refusals(fake):
    fake()
    env = tessera_rl.pz.parallel_env(FAKE)
    assert env.metadata["refuses_agents"] is True
    _, infos = env.reset(seed=0)
    assert all(i == {"refused": False, "refuse_code": 0} for i in infos.values())
    a, b = env.agents
    _, rewards, _, _, infos = env.step({a: 2, b: 1})
    assert infos[a] == {"refused": True, "refuse_code": 30}
    assert infos[b] == {"refused": False, "refuse_code": 0}
    assert rewards[a] == -1.0
    env.close()

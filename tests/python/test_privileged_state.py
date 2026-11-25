# SPDX-License-Identifier: MIT OR Apache-2.0
"""The privileged state buffer from Python (ADR-0079 item 7, #347 stage B3):
``make(..., privileged_state=True)``, ``Env.state`` in borrowed and owned
mode, PettingZoo ``state()`` and ``state_space``, Gymnasium
``info["state"]``, and the refusals."""

import numpy as np
import pettingzoo.test as pz_test
import pytest

import tessera_rl
from tessera_rl import _native, gym, pz

LEDUC = "tessera/Leduc-v1"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def _call_actions(env):
    return np.ones((env.num_envs, env.agents), dtype=np.int32)


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_env_state_holds_both_cards_and_follows_each_call(outputs):
    env = tessera_rl.make(LEDUC, num_envs=3, seed=4, outputs=outputs, privileged_state=True, workers=0)
    assert env.privileged_state
    assert env.state is None
    env.reset()
    state = env.state["state"]
    assert state.shape == (3, 13)
    assert set(state[:, :2].ravel()) <= {0, 1, 2}
    state = env.state["state"]
    out = env.step(_call_actions(env))
    if outputs == "borrowed":
        assert tessera_rl.is_stale(state)
    now = env.state["state"]
    for i in range(3):
        if not out.episode_terminated[i]:
            assert now[i, 0] == out.observations["view"][i, 0, 0]
            assert now[i, 1] == out.observations["view"][i, 1, 0]
    env.close()


def test_env_state_is_none_without_the_option():
    env = tessera_rl.make(LEDUC, num_envs=2, workers=0)
    env.reset()
    assert not env.privileged_state
    assert env.state is None
    assert env._native.layout()["privileged_state"] is False
    env.close()


def test_layout_marks_the_buffer_privileged():
    env = tessera_rl.make(LEDUC, num_envs=1, privileged_state=True, workers=0)
    layout = env._native.layout()
    assert layout["privileged_state"] is True
    assert [leaf[2] for leaf in layout["state_leaves"]] == [[13]]
    env.close()


def test_privileged_state_refused_without_an_implementation_or_in_split_modes():
    with pytest.raises(_native.CapabilityError):
        tessera_rl.make("tessera/CartPole-v1", privileged_state=True)
    with pytest.raises(_native.CapabilityError):
        tessera_rl.make(LEDUC, num_envs=4, chunk_size=2, mode="pipelined", privileged_state=True)


def test_pettingzoo_state_and_state_space():
    env = pz.aec_env(LEDUC, seed=3, privileged_state=True)
    assert env.state_space.shape == (13,)
    env.reset()
    state = env.state()
    assert state.shape == (13,)
    assert env.state_space.contains(state)
    own = env.observe(env.agent_selection)["view"]
    assert own[0] in state[:2]
    env.close()
    plain = pz.aec_env(LEDUC, seed=3)
    plain.reset()
    assert not hasattr(plain, "state_space") or plain.state_space is None
    with pytest.raises(_native.CapabilityError):
        plain.state()
    plain.close()


def test_gymnasium_info_state_rows():
    vec = gym.TesseraVectorEnv.__new__(gym.TesseraVectorEnv)
    vec.num_envs = 2
    vec.outputs = "owned"
    vec._state_leaves = [("state", np.dtype(np.uint8), (3,))]
    info = vec._state_info({"state": [bytearray(range(6))]})
    assert info["state"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert gym._first_info(info)["state"].tolist() == [0, 1, 2]
    vec._state_leaves = []
    assert vec._state_info({}) == {}


LOB = "tessera/LimitOrderBook-v1"
LOB_SMALL = {"agents": 4, "steps": 60, "latency": 1}


def test_pettingzoo_state_of_several_leaves_is_one_flat_box():
    """PettingZoo defines ``state()`` as one array: LimitOrderBook's several
    state leaves flatten in layout order to their lossless result_type."""
    env = pz.parallel_env(LOB, seed=3, privileged_state=True, **LOB_SMALL)
    native = tessera_rl.make(LOB, num_envs=1, seed=3, privileged_state=True, workers=0, outputs="owned", **LOB_SMALL)
    leaves = native._native.layout()["state_leaves"]
    assert len(leaves) > 1
    space = env.state_space
    assert len(space.shape) == 1
    assert space.dtype == np.result_type(*[np.dtype(leaf[1]) for leaf in leaves])
    assert space.shape[0] == sum(int(np.prod(leaf[2])) for leaf in leaves)
    env.reset(seed=3)
    native.reset()
    state = env.state()
    assert state.shape == space.shape and state.dtype == space.dtype
    assert space.contains(state)
    flat = np.concatenate([np.asarray(v[0]).ravel() for v in native.state.values()]).astype(space.dtype)
    np.testing.assert_array_equal(state, flat)
    env.close()
    native.close()


@pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
    "ignore:Environment has not defined a render:UserWarning",
)
def test_pettingzoo_parallel_api_test_passes_with_the_state_on():
    pz_test.parallel_api_test(pz.parallel_env(LOB, seed=3, privileged_state=True, **LOB_SMALL), num_cycles=60)

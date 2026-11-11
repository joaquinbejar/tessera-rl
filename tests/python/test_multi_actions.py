"""The multi-head (``MultiDiscrete``) action path, on the test-only
environment with heads ``[3, 5]``: the reward is ``10 * head0 + head1``, and
choice 4 of head 1 is masked illegal."""

import numpy as np
import pytest

from tessera_rl import _native
from tessera_rl._env import Env

if not hasattr(_native, "_make_multi_probe"):
    pytest.skip("needs a test-hooks build (make py-test-hooks)", allow_module_level=True)

gym = pytest.importorskip("tessera_rl.gym")
ID = "tessera/_MultiProbe"


def env(n=2, **kw):
    return Env(_native._make_multi_probe(num_envs=n, seed=1, outputs="owned", **kw), kw.get("mode", "sync"))


def rows(*pairs):
    return np.array([[p] for p in pairs], dtype=np.int32)


def test_env_layout_reports_the_heads_and_the_mask_row():
    with env() as e:
        assert e.action_nvec == (3, 5)
        assert e.action_choices == 0
        masks = e.reset().masks
        assert masks.shape == (2, 1, 8)
        assert masks[0, 0].tolist() == [True] * 7 + [False]


def test_env_step_carries_every_head():
    with env() as e:
        e.reset()
        assert e.step(rows((2, 3), (1, 0))).rewards[:, 0].tolist() == [23.0, 10.0]


def test_env_masked_choice_falls_back_to_the_noop():
    with env() as e:
        e.reset()
        assert e.step(rows((2, 4), (1, 1))).rewards[:, 0].tolist() == [0.0, 11.0]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 1), dtype=np.int32),  # wrong ndim
        np.zeros((2, 2), dtype=np.int32),  # wrong ndim
        np.zeros((2, 1, 3), dtype=np.int32),  # wrong head count
        np.zeros((2, 2, 2), dtype=np.int32),  # wrong agent count
        np.zeros((3, 1, 2), dtype=np.int32),  # wrong row count
        rows((3, 0), (0, 0)),  # out of range, head 0
        rows((0, 5), (0, 0)),  # out of range, head 1
        rows((-1, 0), (0, 0)),  # negative
        np.full((2, 1, 2), 1.5),  # not an integer
        np.full((2, 1, 2), "a"),  # not a number
    ],
)
def test_env_step_refuses_before_the_pool_is_touched(bad):
    with env() as e:
        e.reset()
        e.step(rows((1, 1), (1, 1)))
        before = e.snapshot(0)
        with pytest.raises(_native.ActionError):
            e.step(bad)
        assert e.snapshot(0) == before


def test_env_integral_floats_are_accepted():
    with env() as e:
        e.reset()
        assert e.step(rows((2, 3), (0, 1)).astype(np.float64)).rewards[:, 0].tolist() == [23.0, 1.0]


def test_env_async_send_carries_every_head():
    with env(2, mode="async", batch_size=2, workers=1) as e:
        e.reset()
        e.send([0, 1], rows((2, 1), (0, 3)))
        got = e.recv()
        by_id = dict(zip(got.ids.tolist(), got.step.rewards[:, 0].tolist(), strict=True))
        assert by_id == {0: 21.0, 1: 3.0}


def test_env_pipelined_send_carries_every_head():
    with env(4, mode="pipelined", chunk_size=2, workers=1) as e:
        e.reset()
        e.send(0, rows((1, 2), (2, 0)))
        assert e.recv().step.rewards[:, 0].tolist() == [12.0, 20.0]


@pytest.fixture
def vec(monkeypatch):
    monkeypatch.setitem(gym._MAKERS, ID, _native._make_multi_probe)
    v = gym.TesseraVectorEnv(ID, num_envs=2, seed=1, outputs="owned")
    v.reset()
    yield v
    v.close()


def test_gym_space_is_multidiscrete(vec):
    s = vec.single_action_space
    assert isinstance(s, gym.spaces.MultiDiscrete)
    assert s.nvec.tolist() == [3, 5]
    assert vec.action_space.shape == (2, 2)


def test_gym_step_carries_every_head(vec):
    assert vec.step(np.array([[2, 3], [1, 0]]))[1].tolist() == [23.0, 10.0]


def test_gym_sample_is_accepted(vec):
    vec.step(vec.action_space.sample())


@pytest.mark.parametrize("bad", [np.zeros(2), np.zeros((2, 3)), np.zeros((2, 1, 2)), np.array([[3, 0], [0, 0]])])
def test_gym_refuses_bad_shapes_and_ranges(vec, bad):
    with pytest.raises(_native.ActionError):
        vec.step(bad)


def test_single_env_takes_one_row_of_heads(monkeypatch):
    monkeypatch.setitem(gym._MAKERS, ID, _native._make_multi_probe)
    e = gym.TesseraEnv(ID, seed=1)
    e.reset()
    assert e.step(np.array([2, 3]))[1] == 23.0
    with pytest.raises(_native.ActionError):
        e.step(np.array([2]))
    e.close()


SEND_BAD = [
    np.zeros((2, 1), dtype=np.int32),  # wrong ndim
    rows((3, 0), (0, 0)),  # out of range
    np.zeros((3, 1, 2), dtype=np.int32),  # wrong row count
]


@pytest.mark.parametrize("bad", SEND_BAD)
def test_env_async_send_refuses_before_the_pool_is_touched(bad):
    with env(2, mode="async", batch_size=2, workers=1) as e:
        e.reset()
        e.send([0, 1], rows((1, 1), (1, 1)))
        e.recv()
        with pytest.raises(_native.ActionError):
            e.send([0, 1], bad)
        # Nothing was admitted: both ids are still idle, and the next send
        # is the one stepped.
        e.send([0, 1], rows((2, 1), (0, 3)))
        got = e.recv()
        by_id = dict(zip(got.ids.tolist(), got.step.rewards[:, 0].tolist(), strict=True))
        assert by_id == {0: 21.0, 1: 3.0}


@pytest.mark.parametrize("bad", SEND_BAD)
def test_env_pipelined_send_refuses_before_the_pool_is_touched(bad):
    with env(4, mode="pipelined", chunk_size=2, workers=1) as e:
        e.reset()
        with pytest.raises(_native.ActionError):
            e.send(0, bad)
        e.send(0, rows((1, 2), (2, 0)))
        assert e.recv().step.rewards[:, 0].tolist() == [12.0, 20.0]


def test_gym_refuses_a_ragged_input(vec):
    with pytest.raises(_native.ActionError):
        vec.step([[1, 2], [1]])


"""The float32 action path, on the test-only continuous environment."""

import importlib
import sys
import types

import numpy as np
import pytest

from tessera_rl import _native
from tessera_rl._env import Env

if not hasattr(_native, "_make_continuous_probe"):
    pytest.skip("needs a test-hooks build (make py-test-hooks)", allow_module_level=True)

gym = pytest.importorskip("tessera_rl.gym")
ID = "tessera/_ContinuousProbe"
UP = np.nextafter(np.float32(2.0), np.float32(3.0))


def env(n=2, **kw):
    return Env(_native._make_continuous_probe(num_envs=n, seed=1, outputs="owned", **kw), kw.get("mode", "sync"))


def bits(a):
    return np.asarray(a, dtype=np.float32).tobytes()


@pytest.fixture
def vec(monkeypatch):
    monkeypatch.setitem(gym._MAKERS, ID, _native._make_continuous_probe)
    v = gym.TesseraVectorEnv(ID, num_envs=2, seed=1, outputs="owned")
    v.reset()
    yield v
    v.close()


def test_env_layout_reports_bounds_and_no_choices():
    with env() as e:
        assert e.action_bounds == (-2.0, 2.0)
        assert e.action_choices == 0


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_env_step_float32_arrives_bit_exact(shape):
    with env() as e:
        e.reset()
        acts = np.array([-0.0, 2.0], dtype=np.float32)
        assert bits(e.step(acts.reshape(shape)).rewards[:, 0]) == bits(acts)


def test_env_step_float64_rounds_to_nearest_float32():
    with env() as e:
        e.reset()
        a = np.array([0.1, -1.0 / 3.0], dtype=np.float64)
        assert bits(e.step(a).rewards[:, 0]) == bits(a.astype(np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, UP, -UP, 1e300])
def test_env_step_refuses_without_clipping(bad):
    with env() as e:
        e.reset()
        with pytest.raises(_native.ActionError):
            e.step(np.array([0.0, bad], dtype=np.float64))


@pytest.mark.parametrize(
    "bad",
    [0.5, [[0.0, 1.0]], [[[0.0]], [[1.0]]], [True, False], ["a", "b"], [0.0], [0.0, 0.0, 0.0], [1j, 1j]],
)
def test_env_step_refuses_bad_shapes_and_dtypes(bad):
    with env() as e:
        e.reset()
        with pytest.raises(_native.ActionError):
            e.step(bad)


def test_env_pipelined_send_carries_floats():
    with env(4, mode="pipelined", chunk_size=2, workers=1) as e:
        e.reset()
        e.send(0, np.array([0.5, -0.5], dtype=np.float32))
        assert bits(e.recv().step.rewards[:, 0]) == bits([0.5, -0.5])


def test_gym_space_is_float32_box(vec):
    s = vec.single_action_space
    assert (s.shape, s.dtype, s.low[0], s.high[0]) == ((1,), np.float32, -2.0, 2.0)
    assert vec.action_space.shape == (2, 1)


def test_gym_sample_is_accepted(vec):
    vec.step(vec.action_space.sample())


@pytest.mark.parametrize("shape", [(2,), (2, 1)])
def test_gym_clips_in_float32(vec, shape):
    acts = np.array([np.inf, -3.0], dtype=np.float64).reshape(shape)
    assert bits(vec.step(acts)[1]) == bits([2.0, -2.0])


def test_gym_casts_before_clipping(vec):
    # Above 2.0 in f64, but rounds to exactly 2.0 in f32; below 2.0 is kept.
    acts = np.array([2.0 + 1e-12, 1.0], dtype=np.float64)
    assert bits(vec.step(acts)[1]) == bits([2.0, 1.0])


def test_gym_accepts_integers(vec):
    assert bits(vec.step(np.array([1, -5]))[1]) == bits([1.0, -2.0])


def test_gym_refuses_nan(vec):
    with pytest.raises(_native.ActionError):
        vec.step(np.array([np.nan, 0.0]))


@pytest.mark.parametrize("bad", [0.5, np.zeros((2, 2)), np.zeros(3)])
def test_gym_refuses_bad_shape(vec, bad):
    with pytest.raises(_native.ActionError):
        vec.step(bad)


@pytest.mark.parametrize("action", [1.5, np.array([1.5], dtype=np.float32)])
def test_single_env_accepts_scalar_and_one_element(monkeypatch, action):
    monkeypatch.setitem(gym._MAKERS, ID, _native._make_continuous_probe)
    e = gym.TesseraEnv(ID, seed=1)
    e.reset()
    assert e.step(action)[1] == 1.5
    with pytest.raises(_native.ActionError):
        e.step(np.zeros((1, 1)))
    e.close()


def test_pettingzoo_refuses_continuous(monkeypatch):
    pz = pytest.importorskip("tessera_rl.pz")
    monkeypatch.setitem(pz._ENVS, ID, (_native._make_continuous_probe, {"parallel"}))
    with pytest.raises(_native.ArgumentError, match="continuous"):
        pz.parallel_env(ID)


def test_jax_refuses_continuous():
    jx = pytest.importorskip("tessera_rl.jax", exc_type=ImportError)
    with env() as e, pytest.raises(_native.ArgumentError, match="discrete only"):
        jx.Env(e)


def test_puffer_refuses_continuous(monkeypatch):
    # A stand-in base class: this suite runs without PufferLib installed.
    stub = types.ModuleType("pufferlib")
    stub.PufferEnv = object
    monkeypatch.setitem(sys.modules, "pufferlib", stub)
    monkeypatch.delitem(sys.modules, "tessera_rl.puffer", raising=False)
    puffer = importlib.import_module("tessera_rl.puffer")
    monkeypatch.setattr(puffer, "make", lambda _id, **kw: env(1))
    monkeypatch.setattr(puffer, "_check_env_id", lambda _id: None)
    try:
        with pytest.raises(_native.ArgumentError, match="continuous"):
            puffer.TesseraPufferEnv(ID, num_envs=1)
    finally:
        sys.modules.pop("tessera_rl.puffer", None)

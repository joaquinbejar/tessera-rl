# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exact integer Box bounds (#570): ``layout()`` gives integer leaf bounds
as Python ints and float leaf bounds as floats; the adapters build their
spaces from them without a ``float64`` round trip, so a bound such as
``2**63 - 1`` stays exact and float bounds keep their bits."""

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _host, _native
from tessera_rl._bounds import leaf_bound

LOB = "tessera/LimitOrderBook-v1"
SMALL = {"agents": 4, "steps": 60, "latency": 1}
I32 = np.iinfo(np.int32)
I64 = np.iinfo(np.int64)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def _bits(a: np.ndarray) -> list[int]:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.uint64).tolist()


# -- leaf_bound -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "dtype"),
    [(2**63 - 1, "int64"), (-(2**63), "int64"), (2**64 - 1, "uint64"), (2**53 + 1, "int64"), (2**53 + 1, "uint64")],
)
def test_leaf_bound_integer_extremes_exact(value, dtype):
    one = leaf_bound([value], [2, 3], dtype)
    assert one.dtype == np.dtype(dtype) and one.shape == (2, 3)
    assert all(int(v) == value for v in one.ravel())
    each = leaf_bound([value, 0], [2], dtype)
    assert [int(v) for v in each] == [value, 0]


def test_leaf_bound_integer_out_of_range_raises_overflow_error():
    with pytest.raises(OverflowError):
        leaf_bound([2**63], [1], "int64")


def test_leaf_bound_bool_zero_one_exact():
    lo, hi = leaf_bound([0], [3], "bool"), leaf_bound([1], [3], "bool")
    assert lo.dtype == np.bool_ and lo.tolist() == [False] * 3 and hi.tolist() == [True] * 3


def test_leaf_bound_float_bits_unchanged():
    values = [float("-inf"), -0.0, 5e-324, 2.2250738585072014e-308 / 4, -1.5, float("inf")]
    got = leaf_bound(values, [2, 3], "float64")
    assert _bits(got.ravel()) == _bits(np.asarray(values, dtype=np.float64))
    # float32: the same values as the float64 then astype path.
    got32 = leaf_bound(values, [6], "float32")
    want32 = np.asarray(values, dtype=np.float64).astype(np.float32)
    assert got32.view(np.uint32).tolist() == want32.view(np.uint32).tolist()


def test_leaf_bound_wrong_count_raises_value_error():
    with pytest.raises(ValueError, match="reshape"):
        leaf_bound([1, 2, 3], [2, 2], "int32")


# -- layout() and the native adapters -------------------------------------------------


def test_int_bounds_layout_integer_leaves_are_ints_float_leaves_floats():
    lob = tessera_rl.make(LOB, num_envs=1, seed=3, outputs="owned", **SMALL)
    try:
        for name, dtype, _, low, high in lob._native.layout()["leaves"]:
            kind = float if np.dtype(dtype).kind == "f" else int
            assert all(type(v) is kind for v in (*low, *high)), name
        own = {name: (low, high) for name, _, _, low, high in lob._native.layout()["leaves"]}["own"]
        assert own == ([I32.min], [I32.max])
    finally:
        lob.close()
    cart = tessera_rl.make("tessera/CartPole-v1", num_envs=1, outputs="owned")
    try:
        for _, _, _, low, high in cart._native.layout()["leaves"]:
            assert all(type(v) is float for v in (*low, *high))
    finally:
        cart.close()


def test_int_bounds_pz_lob_space_exact_ordered_and_contains_an_observation():
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    env = pz.parallel_env(LOB, seed=3, **SMALL)
    try:
        a = env.possible_agents[0]
        space = env.observation_space(a)
        own = space["own"]
        assert own.dtype == np.int32
        assert (own.high == I32.max).all() and (own.low == I32.min).all()
        for leaf in space.spaces.values():
            if hasattr(leaf, "low"):
                assert (leaf.high >= leaf.low).all()
        obs, _ = env.reset(seed=3)
        assert space.contains(obs[a])
    finally:
        env.close()


def test_int_bounds_gym_cartpole_float_bounds_unchanged():
    from tessera_rl import gym

    env = gym.TesseraVectorEnv("tessera/CartPole-v1", num_envs=1)
    try:
        space = env.single_observation_space
        native = tessera_rl.make("tessera/CartPole-v1", num_envs=1, outputs="owned")
        try:
            (_, dtype, shape, low, high), *_ = native._native.layout()["leaves"]
        finally:
            native.close()
        # The pre-#570 path: float64, broadcast, then astype.
        old = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), shape).astype(dtype)  # noqa: E731
        assert space.low.view(np.uint32).tolist() == old(low).view(np.uint32).tolist()
        assert space.high.view(np.uint32).tolist() == old(high).view(np.uint32).tolist()
    finally:
        env.close()


# -- int64 extremes through the adapters' space builders ---------------------------------


def _int64_leaves():
    return [("x", "int64", [3], [I64.min, 0, -(2**53) - 1], [I64.max, 2**53 + 1, I64.max])]


def test_int_bounds_gym_int64_leaf_high_is_int64_max_exactly():
    from tessera_rl import gym

    box = gym._single_space({"leaves": _int64_leaves()})["x"]
    assert box.dtype == np.int64
    assert [int(v) for v in box.high] == [int(I64.max), 2**53 + 1, int(I64.max)]
    assert [int(v) for v in box.low] == [int(I64.min), 0, -(2**53) - 1]
    assert (box.high >= box.low).all()
    assert box.contains(np.asarray([I64.max, 2**53 + 1, 7], dtype=np.int64))


def test_int_bounds_puffer_int64_leaf_high_is_int64_max_exactly():
    pytest.importorskip("pufferlib")
    from tessera_rl import puffer

    dtype, shape, low, high = puffer._observation_space(_int64_leaves())
    assert dtype == np.int64 and shape == (3,)
    assert [int(v) for v in high] == [int(I64.max), 2**53 + 1, int(I64.max)]
    assert [int(v) for v in low] == [int(I64.min), 0, -(2**53) - 1]
    assert (high >= low).all()


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "uint32", "uint64"])
def test_int_bounds_host_dtype_bounds_are_the_exact_range(dtype):
    info = np.iinfo(dtype)
    low, high = _host._dtype_bounds(dtype)
    assert low == [int(info.min)] and high == [int(info.max)]
    assert all(type(v) is int for v in (*low, *high))
    assert int(leaf_bound(high, [1], dtype)[0]) == int(info.max)


def test_int_bounds_host_dtype_bounds_bool_ints_float_infinite():
    assert _host._dtype_bounds("bool") == ([0], [1])
    assert _host._dtype_bounds("float32") == ([float("-inf")], [float("inf")])


@pytest.mark.skipif(not hasattr(_native, "_int_i64_extremes"), reason="needs a test-hooks build (make py-test-hooks)")
def test_capi_int_i64_extremes_are_exact_python_ints():
    """The native ``capi::int_i64`` at both ends of ``i64`` (test hook)."""
    low, high = _native._int_i64_extremes()
    assert type(low) is int and type(high) is int
    assert low == -(2**63)
    assert high == 2**63 - 1


def test_pettingzoo_state_space_int64_top_is_exact():
    """LimitOrderBook's privileged state leaves declare ``i64::MAX`` tops: the
    flat PettingZoo ``state_space`` carries them exactly, with no clamp."""
    from tessera_rl import pz

    env = pz.parallel_env("tessera/LimitOrderBook-v1", seed=3, privileged_state=True, agents=4, steps=60, latency=1)
    space = env.state_space
    assert space.dtype == np.int64
    assert int(space.high.max()) == np.iinfo(np.int64).max
    assert (space.high >= space.low).all()
    env.reset(seed=3)
    assert space.contains(env.state())
    env.close()

# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native types cannot be created or copied from Python (#272 step 3):
every object holds a Rust body that only the native module builds."""

import copy

import pytest
import tessera_rl as t
import tessera_rl._native as native
from tessera_rl.search import Search, SearchGroup

TYPES = ["Env", "OutputView", "PoolSnapshot", "Search", "SearchGroup", "WatchHandle"]


@pytest.mark.parametrize("name", TYPES)
def test_native_type_call_raises_type_error(name):
    ty = getattr(native, name)
    with pytest.raises(TypeError):
        ty()
    with pytest.raises(TypeError):
        ty.__new__(ty)
    with pytest.raises(TypeError):
        object.__new__(ty)


@pytest.mark.parametrize("name", TYPES)
def test_native_type_is_immutable(name):
    ty = getattr(native, name)
    with pytest.raises(TypeError):
        ty.extra = 1


def find_view(obj):
    """The first native OutputView in a raw native result."""
    if type(obj) is native.OutputView:
        return obj
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else ()
    for item in items:
        found = find_view(item)
        if found is not None:
            return found
    return None


def test_native_objects_cannot_be_copied():
    env = t.make("tessera/CartPole-v1", num_envs=2, seed=1, outputs="borrowed")
    split = t.make("tessera/CartPole-v1", num_envs=4, seed=1, mode="pipelined", chunk_size=2)
    try:
        split.reset()
        objs = {
            "Env": env._native,
            "OutputView": find_view(env._native.reset()),
            "PoolSnapshot": split.snapshot_all(),
            "Search": Search(simulations=8)._native,
            "SearchGroup": SearchGroup(2, simulations=8)._native,
            "WatchHandle": env.watch(),
        }
        assert sorted(objs) == sorted(TYPES)
        for name, obj in objs.items():
            assert type(obj) is getattr(native, name), name
            with pytest.raises(TypeError):
                copy.copy(obj)
            with pytest.raises(TypeError):
                copy.deepcopy(obj)
    finally:
        env.close()
        split.close()

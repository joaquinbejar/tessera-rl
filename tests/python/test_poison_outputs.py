# SPDX-License-Identifier: MIT OR Apache-2.0
"""outputs() on a poisoned pool (ADR-0028), through the `test-hooks` wheel."""

import pytest
import tessera_rl
from tessera_rl import _native

if not hasattr(_native, "_make_panic_counter"):
    pytest.skip("needs the test-hooks wheel: make py-test-hooks", allow_module_level=True)


def poisoned(mode):
    env = _native._make_panic_counter(num_envs=4, seed=1, outputs=mode)
    env.reset()
    before = env.outputs()
    agents = len(bytes(before["rewards"])) // 4
    with pytest.raises(tessera_rl.PoisonedError):
        env.step(b"\0" * (4 * agents))
    assert env.is_poisoned()
    return env, before


@pytest.mark.parametrize("mode", ["borrowed", "owned"])
def test_outputs_raises_while_poisoned_and_recovers(mode):
    env, _ = poisoned(mode)
    with pytest.raises(tessera_rl.PoisonedError):
        env.outputs()
    env.reset()
    assert not env.is_poisoned()
    env.outputs()
    env.close()


def test_a_view_from_before_the_panic_is_stale_also_after_close():
    env, before = poisoned("borrowed")
    view = before["rewards"]
    assert view.is_stale()
    env.close()
    assert view.is_stale()


def test_an_owned_copy_from_before_the_panic_stays():
    env, before = poisoned("owned")
    copy = bytes(before["rewards"])
    with pytest.raises(tessera_rl.PoisonedError):
        env.outputs()
    assert bytes(before["rewards"]) == copy
    env.close()

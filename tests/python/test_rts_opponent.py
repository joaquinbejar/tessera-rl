"""RTS against a scripted opponent (stage 5c, plan r2 O2 and O5): one agent,
the learner in slot 0; nothing of the scripted seat reaches Python; the
Gymnasium ids fix the opponent and pass the env checker; their masks come in
`info` and equal the native head views."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import tessera_rl.gym  # noqa: F401  (registers the ids)
from tessera_rl import _native
from tessera_rl._env import make
from tessera_rl.masks import head_views

SMALL = {"size": 12, "max_units": 8, "step_limit": 60}
IDS = {"tessera/RTS-vsRush-v1": 1, "tessera/RTS-vsProspector-v1": 2, "tessera/RTS-vsRandom-v1": 3}


def test_opponent_pool_has_one_agent_and_never_surfaces_the_scripted_seat():
    with make("tessera/RTS-v1", num_envs=2, seed=3, outputs="owned", opponent=1, **SMALL) as e:
        assert (e.agents, e._native.layout()["initial_agents"]) == (2, 1)
        out = e.reset()
        for _ in range(60):
            assert not out.alive[:, 1].any()
            assert not out.acting[:, 1].any()
            assert (out.rewards[:, 1] == 0).all()
            assert not out.masks[:, 1].any()
            for leaf in out.observations.values():
                assert not np.asarray(leaf)[:, 1].any()
            if out.episode_terminated.any() or out.episode_truncated.any():
                break
            out = e.step(np.zeros((2, 2, 24), dtype=np.int32))


@pytest.mark.parametrize("env_id", sorted(IDS))
def test_gym_ids_fix_the_opponent_and_pass_the_env_checker(env_id):
    e = gymnasium.make(env_id, **SMALL)
    check_env(e.unwrapped, skip_render_check=True)
    e.close()
    with pytest.raises(_native.ArgumentError):
        gymnasium.make(env_id, opponent=2, **SMALL)


def test_gym_masks_in_info_equal_the_native_head_views():
    e = gymnasium.make("tessera/RTS-vsRush-v1", seed=4, **SMALL).unwrapped
    native = make("tessera/RTS-v1", num_envs=1, seed=4, outputs="owned", auto_reset="disabled", opponent=1, **SMALL)
    _, info = e.reset(seed=4)
    out = native.reset(4)
    for _ in range(10):
        views = head_views(out.masks, native.mask_heads, native.entity_group_heads)
        for got, want in zip(info["action_masks"], views, strict=True):
            assert (got == want[0, 0]).all()
        assert (info["action_mask"] == info["action_masks"][0]).all()
        act = np.zeros(24, dtype=np.int64)
        _, _, term, trunc, info = e.step(act)
        out = native.step(np.zeros((1, 2, 24), dtype=np.int32))
        if term or trunc:
            break
    e.close()
    native.close()


def test_the_registered_environment_stays_rts_v1():
    ids = {row[0] for row in _native.registered_envs()}
    assert "tessera/RTS-v1" in ids
    assert not any(i.startswith("tessera/RTS-vs") for i in ids)

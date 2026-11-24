# SPDX-License-Identifier: MIT OR Apache-2.0
"""Frames and scenes from snapshots in Python (ADR-0083 item 4)."""

import warnings

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _native, _render, gym, pz


def _fnv(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def test_env_render_cartpole_shape_dtype_and_canvas():
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=2, outputs="owned", workers=0)
    try:
        env.reset()
        frames = env.render()
        assert env.canvas == (600, 400)
        assert frames.shape == (2, 400, 600, 3) and frames.dtype == np.uint8
        assert frames.flags.writeable
        assert env.render([1]).shape == (1, 400, 600, 3)
    finally:
        env.close()


def test_env_render_cartpole_matches_rust_golden_digest():
    # The Rust golden (crates/tessera-rl-render/tests/frames.rs): one
    # instance, seed 7, after a reset.
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=1, seed=7, outputs="owned", workers=0)
    try:
        env.reset()
        assert _fnv(env.render()[0].tobytes()) == 0xC5933C0EF72D727A
    finally:
        env.close()


def test_env_render_twice_same_state_equal_frames():
    env = tessera_rl.make("tessera/Connect4-v1", outputs="owned", workers=0)
    try:
        env.reset()
        np.testing.assert_array_equal(env.render(), env.render())
    finally:
        env.close()


def test_env_scene_connect4_background_then_discs():
    env = tessera_rl.make("tessera/Connect4-v1", outputs="owned", workers=0)
    try:
        env.reset()
        scene = env.scene(0)
        w, h = env.canvas
        assert scene[0][0] == "rect" and scene[0][1:5] == (0.0, 0.0, float(w), float(h))
        assert {p[0] for p in scene[1:]} == {"circle"}
        assert all(len(p[-1]) == 4 for p in scene)
    finally:
        env.close()


def test_env_render_not_renderable_capability_error():
    env = tessera_rl.make("tessera/Acrobot-v1", outputs="owned", workers=0)
    try:
        env.reset()
        assert env.canvas is None
        with pytest.raises(_native.CapabilityError, match="Render"):
            env.render()
        with pytest.raises(_native.CapabilityError, match="Render"):
            env.scene(0)
    finally:
        env.close()


class _Overflowing:
    def canvas(self):
        return (1, 1)

    def render(self, ids):
        return bytes(3 * len(ids)), [True] * len(ids)


def test_render_frames_overflow_warns_render_warning():
    with pytest.warns(tessera_rl.TesseraRenderWarning, match="instance 0"):
        frames = _render.frames(_Overflowing(), [0])
    assert frames.shape == (1, 1, 1, 3)


def test_gym_vector_render_rgb_array_batch():
    env = gym.TesseraVectorEnv("tessera/CartPole-v1", num_envs=3, seed=1, render_mode="rgb_array")
    try:
        env.reset()
        assert env.metadata["render_modes"] == ["rgb_array"]
        assert env.render().shape == (3, 400, 600, 3)
    finally:
        env.close()


def test_gym_env_render_mode_none_returns_none():
    env = gym.TesseraEnv("tessera/CartPole-v1", seed=1)
    try:
        env.reset()
        assert env.render() is None
    finally:
        env.close()


def test_gym_env_render_rgb_array_one_frame():
    env = gym.TesseraEnv("tessera/CartPole-v1", seed=1, render_mode="rgb_array")
    try:
        env.reset()
        assert env.render().shape == (400, 600, 3)
    finally:
        env.close()


def test_gym_render_mode_human_refused():
    with pytest.raises(_native.ArgumentError, match="render_mode"):
        gym.TesseraVectorEnv("tessera/CartPole-v1", render_mode="human")


def test_gym_render_not_renderable_refused_and_listed_empty():
    with pytest.raises(_native.CapabilityError, match="does not render"):
        gym.TesseraVectorEnv("tessera/Acrobot-v1", render_mode="rgb_array")
    env = gym.TesseraVectorEnv("tessera/Acrobot-v1")
    try:
        assert env.metadata["render_modes"] == []
    finally:
        env.close()


def test_pz_parallel_render_rgb_array_frame():
    env = pz.parallel_env("tessera/Snake-v1", seed=3, render_mode="rgb_array")
    try:
        env.reset(seed=3)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
        assert env.metadata["render_modes"] == ["rgb_array"]
    finally:
        env.close()


def test_pz_aec_render_rgb_array_frame():
    env = pz.aec_env("tessera/Connect4-v1", seed=3, render_mode="rgb_array")
    try:
        env.reset(seed=3)
        assert env.render().shape == (480, 560, 3)
    finally:
        env.close()


def test_pz_render_not_renderable_tag_refused():
    with pytest.raises(_native.CapabilityError, match="does not render"):
        pz.parallel_env("tessera/Tag-v1", render_mode="rgb_array")


def test_render_warning_is_runtime_warning():
    assert issubclass(tessera_rl.TesseraRenderWarning, RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = tessera_rl.make("tessera/Go9-v1", outputs="owned", workers=0)
        try:
            env.reset()
            env.render()
        finally:
            env.close()


def test_gymnasium_make_render_mode_rgb_array_frame():
    import gymnasium

    env = gymnasium.make("tessera/CartPole-v1", render_mode="rgb_array")
    try:
        env.reset(seed=1)
        assert env.render().shape == (400, 600, 3)
    finally:
        env.close()

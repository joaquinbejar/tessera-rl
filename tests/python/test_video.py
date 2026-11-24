# SPDX-License-Identifier: MIT OR Apache-2.0
"""Video export of rendered frames (ADR-0083 item 5)."""

import shutil

import numpy as np
import pytest

import tessera_rl
from tessera_rl import _native, render


def _episode_frames(steps: int = 4) -> np.ndarray:
    env = tessera_rl.make("tessera/CartPole-v1", seed=1, outputs="owned", workers=0)
    try:
        env.reset()
        frames = [env.render()[0]]
        for _ in range(steps - 1):
            env.step(np.ones((1, 1), dtype=np.int32))
            frames.append(env.render()[0])
        return np.stack(frames)
    finally:
        env.close()


def test_export_gif_cartpole_episode_valid_file(tmp_path):
    path = tmp_path / "cartpole.gif"
    render.export(_episode_frames(), path, fps=25)
    data = path.read_bytes()
    assert data.startswith(b"GIF89a") and data.endswith(b";")


def test_export_gif_same_frames_same_bytes(tmp_path):
    frames = _episode_frames(2)
    a, b = tmp_path / "a.gif", tmp_path / "b.gif"
    render.export(frames, a)
    render.export(frames, b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize(
    "frames",
    [np.zeros((0, 2, 2, 3), np.uint8), np.zeros((1, 2, 2, 3), np.float32), np.zeros((1, 2, 2), np.uint8)],
    ids=["empty", "float", "no-channels"],
)
def test_export_bad_frames_refused(tmp_path, frames):
    with pytest.raises(_native.ArgumentError, match="frames"):
        render.export(frames, tmp_path / "x.gif")


def test_export_unknown_suffix_refused(tmp_path):
    with pytest.raises(_native.ArgumentError, match="unsupported video format"):
        render.export(np.zeros((1, 2, 2, 3), np.uint8), tmp_path / "x.avi")


def test_export_gif_fps_out_of_range_refused(tmp_path):
    with pytest.raises(_native.ArgumentError, match="fps"):
        render.export(np.zeros((1, 2, 2, 3), np.uint8), tmp_path / "x.gif", fps=0)


def test_export_mp4_missing_ffmpeg_capability_error(tmp_path):
    with pytest.raises(_native.CapabilityError, match="ffmpeg"):
        render.export(np.zeros((1, 2, 2, 3), np.uint8), tmp_path / "x.mp4", ffmpeg=tmp_path / "no-ffmpeg")


def test_export_mp4_no_ffmpeg_on_path_capability_error(tmp_path, monkeypatch):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    with pytest.raises(_native.CapabilityError, match="pass ffmpeg= or put it on PATH"):
        render.export(np.zeros((1, 2, 2, 3), np.uint8), tmp_path / "x.mp4")


def test_export_mp4_odd_size_refused(tmp_path):
    with pytest.raises(_native.ArgumentError, match="even"):
        render.export(np.zeros((1, 3, 2, 3), np.uint8), tmp_path / "x.mp4", ffmpeg="ffmpeg")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs an ffmpeg executable on PATH")
def test_export_mp4_with_ffmpeg_writes_file(tmp_path):
    path = tmp_path / "cartpole.mp4"
    render.export(_episode_frames(), path, fps=25)
    assert path.stat().st_size > 0


def test_export_mp4_dash_name_passed_as_absolute_path(tmp_path, monkeypatch):
    seen = {}

    class Done:
        returncode = 0
        stderr = b""

    def fake_run(args, **kwargs):
        seen["args"], seen["shell"] = args, kwargs.get("shell")
        return Done()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(render.subprocess, "run", fake_run)
    render.export(np.zeros((1, 2, 2, 3), np.uint8), "-y.mp4", ffmpeg="ffmpeg")
    assert seen["args"][-1] == str((tmp_path / "-y.mp4").resolve())
    assert seen["shell"] is False

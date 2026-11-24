# SPDX-License-Identifier: MIT OR Apache-2.0
import os

import numpy.typing as npt

from tessera_rl._env import Env

__all__ = ["export", "replay_video"]

def export(
    frames: npt.ArrayLike,
    path: str | os.PathLike[str],
    fps: int = 50,
    *,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> None: ...
def replay_video(
    path: str | os.PathLike[str],
    env: Env,
    out: str | os.PathLike[str],
    *,
    instance: int = 0,
    fps: int = 50,
    every: int = 1,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> int: ...

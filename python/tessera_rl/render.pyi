# SPDX-License-Identifier: MIT OR Apache-2.0
import os

import numpy.typing as npt

__all__ = ["export"]

def export(
    frames: npt.ArrayLike,
    path: str | os.PathLike[str],
    fps: int = 50,
    *,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> None: ...

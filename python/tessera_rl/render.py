# SPDX-License-Identifier: MIT OR Apache-2.0
"""Video export of rendered frames (ADR-0083 item 5).

``export(frames, path, fps=50)`` writes ``uint8`` frames of shape
``(n, height, width, 3)`` (for example from ``Env.render`` over an
episode) to ``path``:

- ``.gif``: encoded natively (the ``gif`` crate; each frame gets its own
  256-colour palette), looping. GIF delays are in hundredths of a second,
  so ``fps`` (1 to 100) plays at ``100 / round(100 / fps)``: exact for
  rates that divide 100 (1, 2, 4, 5, 10, 20, 25, 50, 100); 30 plays at
  about 33.3, 60 at 50.
- ``.mp4``: piped as raw RGB24 to an ``ffmpeg`` executable the user
  provides (``ffmpeg=`` or found on ``PATH``), started directly with an
  argument list, never through a shell; H.264 in ``yuv420p``, so width and
  height must be even. Tessera links no video codec.

Video files are not covered by any golden (frames are); their bytes may
differ across platforms and encoder versions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from tessera_rl import _native

__all__ = ["export", "replay_video"]


def _frames(frames) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(frames))
    if arr.dtype != np.uint8 or arr.ndim != 4 or arr.shape[3] != 3 or arr.shape[0] == 0:
        raise _native.ArgumentError(
            f"frames must be uint8 of shape (n >= 1, height, width, 3), not {arr.dtype} {arr.shape}"
        )
    return arr


def export(frames, path: str | os.PathLike[str], fps: int = 50, *, ffmpeg: str | os.PathLike[str] | None = None) -> None:
    """Writes ``frames`` to ``path`` as a GIF or an MP4 (by its suffix)."""
    arr = _frames(frames)
    fps = int(fps)
    n, height, width, _ = arr.shape
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".gif":
        data = _native.encode_gif(arr.tobytes(), width, height, fps)
        target.write_bytes(data)
        return
    if suffix == ".mp4":
        _mp4(arr, target, fps, ffmpeg)
        return
    raise _native.ArgumentError(f"unsupported video format {suffix!r}: use .gif or .mp4")


def replay_video(
    path: str | os.PathLike[str],
    env,
    out: str | os.PathLike[str],
    *,
    instance: int = 0,
    fps: int = 50,
    every: int = 1,
    ffmpeg: str | os.PathLike[str] | None = None,
) -> int:
    """Replays the file at ``path`` into ``env`` and writes a video of
    instance ``instance`` to ``out`` (``.gif`` or ``.mp4``, as
    :func:`export`): one frame after every ``every``-th replayed call
    (:func:`tessera_rl.replay.steps`). ``env`` must be built as for
    :func:`tessera_rl.replay.run` and render (ADR-0083). Returns the number
    of frames. Raises ``ReplayDivergence`` if the replay diverges, and
    ``ArgumentError`` for a file with no call to show.
    """
    from tessera_rl import replay

    frames = []
    with replay.steps(path, env, every=every) as it:
        for _ in it:
            frames.append(env.render([instance])[0])
    if not frames:
        raise _native.ArgumentError("the replay file holds no call to render")
    export(np.stack(frames), out, fps, ffmpeg=ffmpeg)
    return len(frames)


def _mp4(arr: np.ndarray, target: Path, fps: int, ffmpeg) -> None:
    n, height, width, _ = arr.shape
    if width % 2 or height % 2:
        raise _native.ArgumentError(f"MP4 (yuv420p) needs even width and height, not {width}x{height}")
    if not 1 <= fps <= 1000:
        raise _native.ArgumentError("fps must be in 1..=1000")
    exe = str(ffmpeg) if ffmpeg is not None else shutil.which("ffmpeg")
    if exe is None:
        raise _native.CapabilityError("MP4 export needs an ffmpeg executable: pass ffmpeg= or put it on PATH")
    args = [
        exe, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        # An absolute path, so a name starting with "-" is never read as an option.
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(target.resolve()),
    ]
    try:
        done = subprocess.run(args, input=arr.tobytes(), capture_output=True, check=False, shell=False)
    except OSError as e:
        raise _native.CapabilityError(f"could not start ffmpeg ({exe}): {e}") from e
    if done.returncode != 0:
        message = done.stderr.decode(errors="replace").strip()
        raise _native.ArgumentError(f"ffmpeg failed with status {done.returncode}: {message}")

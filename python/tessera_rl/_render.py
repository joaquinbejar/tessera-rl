# SPDX-License-Identifier: MIT OR Apache-2.0
"""Frames and scenes from snapshots.

An environment that implements ``Render`` draws a vector scene from a
snapshot's state; frames are rasterized from it, bit-exact across
platforms for scenes built without floats. Rendering runs on the caller's
thread between calls, in sync mode, and never touches live state. A scene
that does not fit its environment's ``MAX_PRIMITIVES`` is drawn truncated,
and the call warns with ``TesseraRenderWarning``.
"""

from __future__ import annotations

import struct
import warnings
from typing import Any

import numpy as np

from tessera_rl import _native

#: The binary record of one primitive (see ``render_api.rs``).
_RECORD = struct.Struct("<I4BI33i")
_KINDS = ("rect", "circle", "line", "polygon")

#: The adapters' ``metadata["render_fps"]``: frames are drawn per step, and
#: 50 is Gymnasium CartPole's value; replay video export sets its own rate.
RENDER_FPS = 50


class TesseraRenderWarning(RuntimeWarning):
    """A scene overflowed its environment's capacity and was truncated."""


def canvas(native: Any) -> tuple[int, int] | None:
    """The ``(width, height)`` of a pool that renders, else ``None``."""
    probe = getattr(native, "canvas", None)
    return None if probe is None else probe()


def _refuse() -> Exception:
    return _native.CapabilityError(
        "render and scene need a sync-mode pool of an environment that implements Render (ADR-0083)"
    )


def frames(native: Any, ids: list[int]) -> np.ndarray:
    """``uint8`` frames of shape ``(len(ids), height, width, 3)``."""
    size = canvas(native)
    if size is None:
        raise _refuse()
    width, height = size
    data, overflowed = native.render(ids)
    for i, over in zip(ids, overflowed):
        if over:
            warnings.warn(f"instance {i}'s scene overflowed and was truncated", TesseraRenderWarning, stacklevel=3)
    return np.frombuffer(data, dtype=np.uint8).reshape(len(ids), height, width, 3).copy()


def scene(native: Any, index: int) -> list[tuple]:
    """Instance ``index``'s scene: ``("rect", x0, y0, x1, y1, rgba)``,
    ``("circle", cx, cy, r, rgba)``, ``("line", x0, y0, x1, y1, width, rgba)``
    or ``("polygon", [(x, y), ...], rgba)``, coordinates in pixels (floats,
    multiples of 1/16), ``rgba`` a 4-tuple of ints."""
    if canvas(native) is None:
        raise _refuse()
    data, overflowed = native.scene(index)
    if overflowed:
        warnings.warn(f"instance {index}'s scene overflowed and was truncated", TesseraRenderWarning, stacklevel=3)
    out: list[tuple] = []
    for record in _RECORD.iter_unpack(data):
        kind, r, g, b, a, count, *values = record
        coords = [v / 16 for v in values[:count]]
        rgba = (r, g, b, a)
        name = _KINDS[kind]
        if name == "polygon":
            out.append((name, list(zip(coords[0::2], coords[1::2])), rgba))
        else:
            out.append((name, *coords, rgba))
    return out


def check_mode(render_mode, renders: bool, env_id: str) -> str | None:
    """``None``, or ``"rgb_array"`` for an environment that renders;
    anything else is refused."""
    if render_mode is None:
        return None
    if render_mode != "rgb_array":
        raise _native.ArgumentError(f"render_mode must be None or 'rgb_array', not {render_mode!r}")
    if not renders:
        raise _native.CapabilityError(f"{env_id} does not render (it does not implement Render, ADR-0083)")
    return render_mode

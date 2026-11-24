# SPDX-License-Identifier: MIT OR Apache-2.0
"""Observation transforms (ADR-0078): pass a list of these to
``tessera_rl.make(..., transforms=[...])``. They run in the engine, on the
workers, right after the environment's ``observe``, in list order per leaf;
the pool's ``layout()`` and the Gymnasium and PettingZoo spaces describe the
transformed observation.

A leaf is named as ``layout()`` names it: ``Dict`` keys and ``Tuple``
positions joined by ``.``, the empty name for an observation that is a
single leaf. Transforms on ``Set`` leaves are refused (ADR-0082).

- ``FrameStack(leaf, k)``: the last ``k`` observations, oldest first, shape
  ``(k, *shape)``; ``k`` in ``2..=64``. The stacks are pool state beside the
  records: ``restore`` refills them from the restored observation.
- ``OneHot(leaf, n)``: an integer leaf as ``bool`` rows of ``n``; a value
  outside ``0..n`` gives an all-false row.
- ``Normalize(leaf, clip)``: ``clamp((x - mean) / sqrt(var + 1e-8), -clip,
  clip)`` as ``float32``, with the pool's running statistics, which change
  only at ``Env.epoch_barrier()`` and stop with
  ``Env.freeze_normalization(True)``. Before the first barrier it is the
  identity up to the clip. Not in Async mode unless frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FrameStack", "Normalize", "OneHot"]


@dataclass(frozen=True)
class FrameStack:
    """The last ``k`` observations of ``leaf``, oldest first."""

    leaf: str
    k: int


@dataclass(frozen=True)
class OneHot:
    """Integer ``leaf`` as one-hot rows of ``n`` classes."""

    leaf: str
    n: int


@dataclass(frozen=True)
class Normalize:
    """``leaf`` normalized by the pool's running statistics, clipped to
    ``[-clip, clip]``."""

    leaf: str
    clip: float = 5.0


def native_spec(transforms) -> str:
    """The native form of a transform list: one ``kind<TAB>leaf<TAB>param``
    line per transform. ``TypeError`` for anything but these objects."""
    lines = []
    for t in transforms:
        if isinstance(t, FrameStack):
            kind, param = "frame_stack", _int(t.k, "k")
        elif isinstance(t, OneHot):
            kind, param = "one_hot", _int(t.n, "n")
        elif isinstance(t, Normalize):
            kind, param = "normalize", repr(float(t.clip))
        else:
            raise TypeError(f"transforms take FrameStack, OneHot or Normalize, not {type(t).__name__}")
        leaf = str(t.leaf)
        if "\t" in leaf or "\n" in leaf:
            raise ValueError(f"invalid leaf name {leaf!r}")
        lines.append(f"{kind}\t{leaf}\t{param}")
    return "\n".join(lines)


def _int(value, what: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{what} must be a non-negative int, not {value!r}")
    return str(value)

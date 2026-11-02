# SPDX-License-Identifier: MIT OR Apache-2.0
"""Training-sample augmentation for Connect Four.

Gravity breaks rotations and vertical reflections, so the only symmetry of
the game is the left-right mirror (Z2): column ``a`` maps to ``6 - a``.
The mirror keeps the mover-relative value target unchanged.
"""

from __future__ import annotations

import numpy as np


def mirror(planes: np.ndarray, policy: np.ndarray, value: np.ndarray):
    """The mirrored samples. ``planes`` is ``(n, 2, 6, 7)``, ``policy``
    ``(n, 7)``, ``value`` ``(n,)``."""
    return (
        np.ascontiguousarray(planes[..., ::-1]),
        np.ascontiguousarray(policy[:, ::-1]),
        value.copy(),
    )


def augment(planes: np.ndarray, policy: np.ndarray, value: np.ndarray):
    """The identity and the mirror, concatenated."""
    mp, mpol, mv = mirror(planes, policy, value)
    return (
        np.concatenate([planes, mp]),
        np.concatenate([policy, mpol]),
        np.concatenate([value, mv]),
    )

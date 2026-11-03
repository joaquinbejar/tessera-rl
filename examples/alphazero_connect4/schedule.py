# SPDX-License-Identifier: MIT OR Apache-2.0
"""The frozen training schedule of the P2-M3 loop (PR 3, design revision 2).

Generation k trains for exactly ``STEPS`` steps. The global step is
``t = k * STEPS + s`` for ``s`` in ``0..STEPS-1``, and the learning rate is

    lr(t) = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + cos(pi * t / (K_MAX * STEPS)))

in float64, with the horizon ``K_MAX * STEPS`` fixed: it is never
recomputed on resume, on an early stop or on a budget stop. A shorter run
simply ends at a lower ``t``.
"""

from __future__ import annotations

import math

K_MAX = 30
STEPS = 1000
LR_MAX = 0.02
LR_MIN = 0.002
HORIZON = K_MAX * STEPS


def lr(t: int) -> float:
    if not 0 <= t < HORIZON:
        raise ValueError(f"step {t} outside the registered horizon 0..{HORIZON - 1}")
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * t / HORIZON))


def global_step(k: int, s: int) -> int:
    return k * STEPS + s

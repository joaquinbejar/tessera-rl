# SPDX-License-Identifier: MIT OR Apache-2.0
"""The frozen training schedule of the P2-M3 loop (PR 3, design revision 2).

Generation k trains for exactly ``STEPS`` steps. The global step is
``t = k * STEPS + s`` for ``s`` in ``0..STEPS-1``, and the learning rate is

    lr(t) = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + cos(pi * t / (K_MAX * STEPS)))

in float64, with the horizon ``K_MAX * STEPS`` fixed: it is never
recomputed on resume, on an early stop or on a budget stop. A shorter run
simply ends at a lower ``t``.

The horizon is a registered constant of the run's profile, recorded in its
``config.json`` as ``horizon_k``: M3 uses ``K_MAX`` (30); the ADR-0052
profile scales it to its 10 generations (``K_MAX_ADR0052``), so the rate
reaches ``LR_MIN`` at its last generation.
"""

from __future__ import annotations

import math

K_MAX = 30
STEPS = 1000
LR_MAX = 0.02
LR_MIN = 0.002
HORIZON = K_MAX * STEPS
K_MAX_ADR0052 = 10
HORIZONS_K = (K_MAX, K_MAX_ADR0052)


def lr(t: int, horizon_k: int = K_MAX) -> float:
    """The rate at global step ``t`` under a registered horizon of
    ``horizon_k`` generations."""
    if horizon_k not in HORIZONS_K:
        raise ValueError(f"horizon of {horizon_k} generations is not a registered one {HORIZONS_K}")
    horizon = horizon_k * STEPS
    if not 0 <= t < horizon:
        raise ValueError(f"step {t} outside the registered horizon 0..{horizon - 1}")
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * t / horizon))


def global_step(k: int, s: int) -> int:
    return k * STEPS + s

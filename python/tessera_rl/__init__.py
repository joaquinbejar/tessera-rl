# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tessera: fast, deterministic, snapshot-able reinforcement learning environments.

The caller's thread must be in the default floating-point environment
(round to nearest, no flush-to-zero, exceptions masked). Calling Tessera
after, for example, ``torch.set_flush_denormal(True)`` is unsupported until
the native floating-point boundary of ADR-0041 lands; no typed rejection is
promised before then.
"""

from tessera_rl._env import Env, Step, is_stale, make
from tessera_rl._native import (
    ActionError,
    ArgumentError,
    ConcurrencyError,
    ConfigError,
    ExhaustedError,
    PendingResetError,
    PoisonedError,
    RecordError,
    TesseraError,
    version,
)

__all__ = [
    "ActionError",
    "ArgumentError",
    "ConcurrencyError",
    "ConfigError",
    "Env",
    "ExhaustedError",
    "PendingResetError",
    "PoisonedError",
    "RecordError",
    "Step",
    "TesseraError",
    "is_stale",
    "make",
    "version",
]
__version__ = version()

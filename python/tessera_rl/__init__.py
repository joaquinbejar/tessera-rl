# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tessera: fast, deterministic, snapshot-able reinforcement learning environments.

Floating-point controls (ADR-0041): every native entry saves the caller's
floating-point controls (for example after ``torch.set_flush_denormal(True)``
or a changed rounding mode), computes under the default controls, and
restores the caller's afterwards, keeping the sticky exception flags.
Results are therefore identical whatever the caller's controls. By default
such calls proceed, are counted (``fp_non_canonical_entries()``) and warn
once per process with a ``RuntimeWarning``; ``set_fp_policy("reject")``
makes them raise ``FloatingPointEnvironmentError`` instead, before any
work. Deallocation and export deleters always run. The boundary covers
x86-64 (System V) and aarch64 (Apple and Linux); Windows comes later and
until then needs the default controls.
"""

from tessera_rl._env import Env, Step, is_stale, make
from tessera_rl._native import (
    ActionError,
    ArgumentError,
    ConcurrencyError,
    ConfigError,
    ExhaustedError,
    FloatingPointEnvironmentError,
    PendingResetError,
    PoisonedError,
    RecordError,
    TesseraError,
    fp_non_canonical_entries,
    get_fp_policy,
    set_fp_policy,
    version,
)

__all__ = [
    "ActionError",
    "ArgumentError",
    "ConcurrencyError",
    "ConfigError",
    "Env",
    "ExhaustedError",
    "FloatingPointEnvironmentError",
    "PendingResetError",
    "PoisonedError",
    "RecordError",
    "Step",
    "TesseraError",
    "fp_non_canonical_entries",
    "get_fp_policy",
    "is_stale",
    "make",
    "set_fp_policy",
    "version",
]
__version__ = version()

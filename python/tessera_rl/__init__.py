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
x86-64 (System V) and aarch64 (AAPCS64) on Linux and macOS, and Win64
(x86-64) on Windows.

On x86-64 the caller's x87 control word is preserved as well (ADR-0044):
it is restored after every call and after every callback into Python
(for example a warning hook), so a change Python or C code makes to it
inside a callback never reaches the caller. It is never counted, warned
about or rejected: Tessera does not compute with x87. Only the control
word is restored; the x87 status flags are left as host code set them.
Precondition: code running in a callback must not leave an unmasked x87
exception pending (an x87 status flag set whose exception is unmasked in
the current control word or in the caller's). Restoring the control word
then faults in hardware (``SIGFPE`` on Linux and macOS; a floating-point
exception status on Windows, ``STATUS_FLOAT_DIVIDE_BY_ZERO`` for a pending
zero-divide), which Tessera cannot turn into a Python exception. Win64 x87
preservation is validated natively on one host (Windows 10.0.26300, Python
3.11.18 MSVC build; issue #163).

API changes (pre-release; the package is not published yet):

- P2-M1b: ``Step`` gained a ninth field, ``acting``, after ``masks``.
  Access by name or index is unchanged, but unpacking a ``Step`` into a
  fixed number of names is a breaking change: it now needs 9 names, not 8.
  ``Env`` gained ``turns``.
"""

from tessera_rl._env import AsyncStep, Env, HalfStep, Step, is_stale, make
from tessera_rl._native import (
    ActionError,
    ArgumentError,
    ConcurrencyError,
    ConfigError,
    EvaluatorError,
    ExhaustedError,
    FloatingPointEnvironmentError,
    ForkError,
    OnnxError,
    PersistError,
    PendingResetError,
    PoisonedError,
    RecordError,
    SearchError,
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
    "AsyncStep",
    "Env",
    "HalfStep",
    "EvaluatorError",
    "ExhaustedError",
    "FloatingPointEnvironmentError",
    "ForkError",
    "OnnxError",
    "PersistError",
    "PendingResetError",
    "PoisonedError",
    "RecordError",
    "SearchError",
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

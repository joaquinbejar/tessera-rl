# SPDX-License-Identifier: MIT OR Apache-2.0
"""The registry records and the deprecation warning.

``envs()`` lists the first-party environments registered in the native
module: the facts of ``_native.registered_envs()`` plus ``adapters``, the
PettingZoo interfaces each declares (``"parallel"``, ``"aec"``,
``"turn_masked"``, SPEC 12), and ``deprecated`` (``None``, or a dict with
``since`` and ``note``, SPEC 16.1). Host environments of installed wheels
are not listed (their entry points are read by ``tessera_rl.make``).

Building a pool of a deprecated id warns with ``TesseraDeprecationWarning``,
a ``FutureWarning``, so it is shown by default; Python's warning filters
handle repetition and silencing.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from typing import Any

from tessera_rl import _native

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


class TesseraDeprecationWarning(FutureWarning):
    """A deprecated environment id (SPEC 16.1) or argument (#538) was used."""


@functools.cache
def _records() -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in _native.env_records()}


def envs() -> list[dict[str, Any]]:
    """One dict per registered first-party id, in id order."""
    return [
        {
            **r,
            "adapters": frozenset(r["adapters"]),
            "deprecated": None if r["deprecated"] is None else dict(r["deprecated"]),
        }
        for r in _native.env_records()
    ]


def adapters_of(env_id: str) -> frozenset[str]:
    """The PettingZoo interfaces ``env_id`` declares; empty if unregistered."""
    record = _records().get(env_id)
    return frozenset(record["adapters"]) if record is not None else frozenset()


def warn_if_deprecated(env_id: str) -> None:
    """Warns with ``TesseraDeprecationWarning`` if ``env_id`` is deprecated,
    attributed to the first caller outside this package."""
    record = _records().get(env_id)
    if record is None or record["deprecated"] is None:
        return
    level = 2
    frame = sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back
        level += 1
    d = record["deprecated"]
    warnings.warn(
        f"{env_id} is deprecated since {d['since']}: {d['note']} (SPEC 16.1: removed only in a major release)",
        TesseraDeprecationWarning,
        stacklevel=level,
    )

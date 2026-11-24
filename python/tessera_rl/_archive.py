# SPDX-License-Identifier: MIT OR Apache-2.0
"""Archives of saved states for archive resets."""

from __future__ import annotations

from tessera_rl import _native


class Archive:
    """An immutable, validated archive of saved states of one environment
    and config.

    ``Archive(env, snapshots)`` builds it from ``env``'s environment and
    exact config, and ``snapshots``: instance records as ``env.snapshot(i)``
    returns them (``bytes``). Every entry is checked with the environment's
    invariants. Attach it with ``env.set_reset_source(archive, weights)`` to
    any environment of the same id and config; it is shared, never copied.
    """

    __slots__ = ("_native", "_digest", "_len")

    def __init__(self, env, snapshots) -> None:
        records = [bytes(s) for s in snapshots]
        native = getattr(env, "_native", None)
        if native is None:
            raise _native.ArgumentError(f"env must be a tessera_rl environment, not {type(env).__name__}")
        self._native, self._digest, self._len = native._make_archive(records)

    @property
    def digest(self) -> int:
        """The archive's identity: XXH3-64 over the environment id, version,
        config hash and entries. Not authentication."""
        return self._digest

    def __len__(self) -> int:
        return self._len

    def __repr__(self) -> str:
        return f"Archive(len={self._len}, digest={self._digest:016x})"

# SPDX-License-Identifier: MIT OR Apache-2.0
"""``tessera_rl.League``: a league shared
by every pool made with ``make(..., policy_map={slot: "opponent"},
league=league, role=r)``.

A league is always shared: every pool's ``epoch_barrier()`` hands in its
results for the round and gets back the table committed at the previous
round, and the round's last barrier, from any thread, commits by merging the
pools' results in the order they were made. So a league lags one round,
with one pool or many, and the result does not depend on which pool's
barrier runs first or on which thread. A pool's second barrier before the
round commits raises ``TesseraError`` naming the pools still to submit; it
never blocks.

``population`` is a list of ``"heuristic"``, ``("external", tag)`` or
``("onnx", content_hash)`` (32 hex digits) entries. ONNX entries are kept,
saved and exported; ``make`` plays them with the models of
``onnx_opponents=tessera_rl.search.OnnxOpponents(...)``, which must serve
every such hash (``ArgumentError`` otherwise). Results with ONNX opponents
are repeatable only in ADR-0057's class: the same machine, the same
library and model bytes, the same batch composition; never across
platforms.

``roles`` is a list of dicts, one per learner role:

- ``matchmaking``: ``"uniform"``, ``"pfsp"`` (with ``weighting``
  ``"hard"``, the default, ``"variance"`` or ``"linear"``) or
  ``("elo", temperature)``;
- ``self_play``: ``(share, tag)``, the role's own entry drawn with exactly
  that share; its results never move the ratings;
- ``snapshot``: ``("every", n)`` or ``("win_rate", threshold, min_games)``,
  adding ``("external", snapshot_tags + k)`` entries;
- ``snapshot_tags``: the first snapshot's tag.
"""

from __future__ import annotations

import json
from typing import Any

from tessera_rl import _native


def _num(v: Any, what: str) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{what} must be a number, got {type(v).__name__}")
    return repr(float(v)) if isinstance(v, float) else str(v)


def _policy(entry: Any) -> str:
    if entry == "heuristic":
        return "policy heuristic"
    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "external":
        return f"policy external {_num(entry[1], 'an external tag')}"
    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "onnx":
        h = entry[1].hex() if isinstance(entry[1], (bytes, bytearray)) else str(entry[1])
        return f"policy onnx {h}"
    raise _native.ArgumentError(f"a population entry is 'heuristic', ('external', tag) or ('onnx', hash), not {entry!r}")


def _role(role: dict) -> str:
    unknown = set(role) - {"matchmaking", "weighting", "self_play", "snapshot", "snapshot_tags"}
    if unknown:
        raise _native.ArgumentError(f"unknown role keys {sorted(unknown)}")
    m = role.get("matchmaking", "pfsp")
    if m == "uniform":
        words = ["uniform"]
    elif m == "pfsp":
        words = [f"pfsp:{role.get('weighting', 'hard')}"]
    elif isinstance(m, tuple) and len(m) == 2 and m[0] == "elo":
        words = [f"elo:{_num(float(m[1]), 'an elo temperature')}"]
    else:
        raise _native.ArgumentError(f"matchmaking is 'uniform', 'pfsp' or ('elo', temperature), not {m!r}")
    if role.get("self_play") is not None:
        share, tag = role["self_play"]
        words += ["self_play", _num(float(share), "a self-play share"), _num(tag, "a self-play tag")]
    snap = role.get("snapshot")
    if snap is not None:
        if snap[0] == "every" and len(snap) == 2:
            words += ["snapshot", "every", _num(snap[1], "a snapshot period")]
        elif snap[0] == "win_rate" and len(snap) == 3:
            words += ["snapshot", "win_rate", _num(float(snap[1]), "a threshold"), _num(snap[2], "min_games")]
        else:
            raise _native.ArgumentError(f"snapshot is ('every', n) or ('win_rate', threshold, min_games), not {snap!r}")
    if "snapshot_tags" in role:
        words += ["tags", _num(role["snapshot_tags"], "snapshot_tags")]
    return "role " + " ".join(words)


class League:
    """A league shared by several pools: one population, one learner per
    role, and a barrier that does not depend on thread timing."""

    def __init__(
        self,
        population: list,
        roles: list[dict],
        *,
        elo_k: float = 16.0,
        trueskill: tuple[float, float, float] | None = None,
        rate_population: bool = False,
    ) -> None:
        ratings = f"ratings {_num(float(elo_k), 'elo_k')} {int(bool(rate_population))}"
        if trueskill is not None:
            beta, tau, draw = trueskill
            ratings += " " + " ".join(_num(float(v), "a trueskill parameter") for v in (beta, tau, draw))
        lines = [ratings] + [_policy(p) for p in population] + [_role(r) for r in roles]
        self._handle: int | None = _native.league_new(spec="\n".join(lines))

    @classmethod
    def load(cls, data: bytes) -> League:
        """A league from ``save()``'s bytes. Its pools are made again with
        ``make(..., league=, role=)`` in their saved order, before any
        barrier; each pool is then restored from its own snapshot."""
        league = cls.__new__(cls)
        league._handle = _native.league_load(data=bytes(data))
        return league

    def save(self) -> bytes:
        """The league's canonical bytes, between rounds (``PersistError``
        while some pools have submitted the round)."""
        return _native.league_save(league=self._handle)

    def export(self) -> str:
        """The JSON export: ratings, counts, population."""
        return _native.league_export(league=self._handle)

    @property
    def round(self) -> int:
        """The round the pools' next barriers submit to."""
        return int(json.loads(self.export())["round"])

    def population(self) -> list[dict]:
        """The population, as in the export."""
        return json.loads(self.export())["population"]

    def ratings(self, role: int = 0) -> dict:
        """Role ``role``'s learner: ``elo``, ``trueskill``, ``counts`` and
        ``snapshots``, as in the export."""
        return json.loads(self.export())["roles"][role]

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                _native.league_drop(league=handle)
            except Exception:  # noqa: BLE001  (interpreter shutdown)
                pass
            self._handle = None

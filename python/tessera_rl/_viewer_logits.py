# SPDX-License-Identifier: MIT OR Apache-2.0
"""Policy logits for the Viewer (#683, ADR-0083 item 6): a side file next to
a replay, and the per-agent, per-head payload the Viewer serves.

The side file is ``<replay>.logits.npz`` (NumPy's zip format, read with
``allow_pickle=False``). It is an optional, Python-side sidecar outside the
1.x stability promise (ADR-0094): its ``format`` entry is
``"tessera-logits/1"``, and any other value is refused. It never changes the
replay file. It is bound to one replay by that file's sha256: a side file
whose hash does not match is ignored, and the Viewer says why in ``/meta``.

A side file is treated as untrusted input: the format, every entry's dtype
and shape, and the ordinals are checked before anything is served.
"""

from __future__ import annotations

import contextlib
import hashlib
import os

import numpy as np

from tessera_rl._native import ArgumentError

__all__ = ["FORMAT", "LogitsLog", "load", "logits_log", "payload", "side_path", "width_of"]

#: The side file's format; any other value is refused.
FORMAT = "tessera-logits/1"
#: Entries a side file holds, and nothing else.
_KEYS = frozenset({"format", "replay_sha256", "shape", "ordinals", "logits"})


def side_path(replay_path) -> str:
    """The side file of the replay at ``replay_path``."""
    return os.fspath(replay_path) + ".logits.npz"


def width_of(env) -> int:
    """The width of one agent's logits row: the heads side by side, as in
    the masks."""
    nvec = tuple(int(n) for n in getattr(env, "action_nvec", ()) or ())
    if nvec:
        return int(sum(nvec))
    return int(env.action_choices)


def heads_of(env) -> tuple[int, ...]:
    """Each head's number of choices."""
    nvec = tuple(int(n) for n in getattr(env, "action_nvec", ()) or ())
    return nvec if nvec else (int(env.action_choices),)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(os.fspath(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LogitsLog:
    """The logits of each ``env.step`` while a replay records. Use
    :func:`logits_log`; call :meth:`step` once after every ``env.step``."""

    def __init__(self, replay_path, env) -> None:
        self.replay_path = os.fspath(replay_path)
        self.shape = (int(env.num_envs), int(env.agents), width_of(env))
        self._ordinal = 0
        self._ordinals: list[int] = []
        self._rows: list[np.ndarray] = []

    def step(self, logits) -> None:
        """Records the logits that chose the last step's actions, shape
        ``(num_envs, agents, width)``, or ``None`` for a step without them.
        Raises ``ArgumentError`` on another shape or a non-numeric array."""
        ordinal = self._ordinal
        self._ordinal += 1
        if logits is None:
            return
        try:
            arr = np.asarray(logits, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ArgumentError(f"logits are not numeric: {e}") from None
        if arr.shape != self.shape:
            raise ArgumentError(f"logits have shape {arr.shape}, expected {self.shape}")
        self._ordinals.append(ordinal)
        self._rows.append(arr.copy())

    def _write(self) -> None:
        if not os.path.exists(self.replay_path):
            raise ArgumentError(
                "the replay file does not exist yet: replay.record writes it when its block exits, "
                "so open logits_log outside replay.record"
            )
        n = len(self._rows)
        logits = np.stack(self._rows) if n else np.zeros((0, *self.shape), dtype=np.float32)
        target = side_path(self.replay_path)
        tmp = target + ".tmp"
        try:
            with open(tmp, "wb") as f:
                np.savez(
                    f,
                    format=np.array(FORMAT),
                    replay_sha256=np.array(_sha256(self.replay_path)),
                    shape=np.array(self.shape, dtype=np.int64),
                    ordinals=np.array(self._ordinals, dtype=np.int64),
                    logits=logits.astype(np.float32, copy=False),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise


@contextlib.contextmanager
def logits_log(replay_path, env):
    """Logs policy logits next to the replay at ``replay_path``; written as
    ``<replay>.logits.npz`` when the block exits normally, bound to the
    finished replay by its sha256. Open it **outside** ``replay.record``, so
    the replay is complete when the side file is written::

        with viewer.logits_log("run.replay", env) as log, replay.record(env, "run.replay") as env:
            ...
            out = env.step(actions)
            log.step(logits)

    If the block raises, nothing is written."""
    log = LogitsLog(replay_path, env)
    yield log
    log._write()


class Refused(Exception):
    """A side file that cannot be served; the message says why."""


def load(side, replay_path, env, step_count: int) -> dict[int, np.ndarray]:
    """The logits of a side file by step ordinal, checked against the replay
    at ``replay_path`` (its sha256), ``env`` (num_envs, agents, width) and
    the replay's ``step_count``. Raises :class:`Refused` with the reason for
    anything else."""
    try:
        return _load(side, replay_path, env, step_count)
    except Refused:
        raise
    except Exception as e:  # untrusted bytes: any decode failure is a refusal
        raise Refused(f"not a readable .npz file ({type(e).__name__})") from None


def _load(side, replay_path, env, step_count: int) -> dict[int, np.ndarray]:
    with np.load(os.fspath(side), allow_pickle=False) as data:
        keys = set(data.files)
        if keys != _KEYS:
            raise Refused(f"entries {sorted(keys)} are not {sorted(_KEYS)}")
        fmt = data["format"]
        if fmt.dtype.kind != "U" or fmt.shape != () or str(fmt) != FORMAT:
            raise Refused(f"format is not {FORMAT!r}")
        sha = data["replay_sha256"]
        if sha.dtype.kind != "U" or sha.shape != ():
            raise Refused("replay_sha256 is not a string")
        if str(sha) != _sha256(replay_path):
            raise Refused("replay_sha256 does not match the replay file: the side file belongs to another recording")
        expected = (int(env.num_envs), int(env.agents), width_of(env))
        shape = data["shape"]
        if shape.dtype.kind not in "iu" or shape.shape != (3,) or tuple(int(x) for x in shape) != expected:
            raise Refused(f"shape is not {expected}")
        ordinals = data["ordinals"]
        logits = data["logits"]
        if ordinals.dtype.kind not in "iu" or ordinals.ndim != 1:
            raise Refused("ordinals are not a 1-D integer array")
        if logits.dtype != np.float32 or logits.shape != (ordinals.shape[0], *expected):
            raise Refused(f"logits are not float32 of shape ({ordinals.shape[0]}, {expected[0]}, {expected[1]}, {expected[2]})")
        ords = [int(o) for o in ordinals]
        if any(b <= a for a, b in zip(ords, ords[1:])):
            raise Refused("ordinals are not strictly increasing")
        if ords and (ords[0] < 0 or ords[-1] >= step_count):
            raise Refused(f"an ordinal is outside the replay's {step_count} steps")
        return {o: np.array(logits[k]) for k, o in enumerate(ords)}


def _finite(values: list) -> list:
    return [None if not np.isfinite(v) else float(v) for v in values]


def payload(row, mask_row, heads: tuple[int, ...]):
    """One agent's logits as served: per head, the raw values, the softmax
    probabilities over its legal choices (all choices without a mask) and
    the legality. ``row`` has the heads side by side; NaN and infinity are
    reported as ``null``, and a head with no legal or finite choice has
    ``null`` probabilities."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    out = []
    start = 0
    for n in heads:
        raw = row[start : start + n]
        legal = (
            np.ones(n, dtype=bool)
            if mask_row is None
            else np.asarray(mask_row[start : start + n], dtype=bool).reshape(-1)
        )
        ok = legal & np.isfinite(raw)
        probs = None
        if ok.any():
            shifted = np.where(ok, raw - raw[ok].max(), -np.inf)
            e = np.exp(shifted)
            p = e / e.sum()
            probs = [float(x) for x in p]
        out.append({"logits": _finite(raw.tolist()), "probs": probs, "legal": legal.tolist()})
        start += n
    return out

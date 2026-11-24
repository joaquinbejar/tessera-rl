# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay files (ADR-0072 section 9): record the calls on an environment and
replay them, checking that the replay reproduces the recording.

A replay file is a call log: every ``reset``, ``step`` and ``fork`` while
recording, with keyframes of the full state and checkpoints of the state hash
(ADR-0072 sections 1 to 4). It holds full state, so it is privileged like a
persisted snapshot (ADR-0035): a Leduc file holds hidden cards.

Scope and known gaps:

- First-party environments made by :func:`tessera_rl.make`, in Inline or Sync
  mode. Host environments (``tessera_rl.HostEnv``) need a host-protocol change
  and are not supported yet; neither are the Gymnasium, PettingZoo and
  PufferLib adapters (pass the environment ``make`` returned).
- While recording, ``import_record`` and ``load`` are refused: they would
  write state the file does not log.
- The recording is held in memory until it ends, and replay reads the whole
  file (ADR-0072 section 8 asks for streaming).
- :func:`run` replays into an environment the caller built with the same
  ``make`` arguments; rebuilding it from the file is issue #377.

Errors: a divergence raises :class:`ReplayDivergence`. A damaged, truncated,
foreign or newer file, or one whose environment, config, options or instance
count differ from ``env``, raises :class:`tessera_rl.PersistError`, as
``Env.load`` does for a persisted snapshot (the two kinds share the ADR-0011
header). A pool error raises its usual class.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

from tessera_rl._native import TesseraError

__all__ = ["ReplayDivergence", "ReplaySummary", "record", "run"]


@dataclass(frozen=True)
class ReplaySummary:
    """What a replay checked.

    ``events``: events replayed; ``calls``: calls re-issued; ``poisoned``: the
    file ends with a poisoning call that the replay reproduced;
    ``unverified_poison``: the file ends with a state write (``fork``) that
    poisoned the recorded pool, which replay cannot re-run, so the replayed
    pool is not poisoned.
    """

    events: int
    calls: int
    poisoned: bool = False
    unverified_poison: bool = False


class ReplayDivergence(TesseraError):
    """The replay differs from the recording (ADR-0072 section 5).

    ``event_index`` is the event whose check failed; ``kind`` names the check
    (``"checkpoint"``, ``"call_failed"``, ``"keyframe"``, ...); ``instance``
    is the instance for a per-instance check, else ``None``;
    ``last_matching_checkpoint`` is the event index of the last checkpoint
    that matched, or ``None``. A state divergence happened between that
    checkpoint and ``event_index``.
    """

    def __init__(
        self,
        event_index: int,
        kind: str,
        instance: int | None,
        last_matching_checkpoint: int | None,
    ) -> None:
        super().__init__(
            f"replay diverged at event {event_index} ({kind}, instance {instance}); "
            f"last matching checkpoint {last_matching_checkpoint}"
        )
        self.event_index = event_index
        self.kind = kind
        self.instance = instance
        self.last_matching_checkpoint = last_matching_checkpoint


def _native_of(env):
    native = getattr(env, "_native", None)
    if native is None or not hasattr(native, "_replay"):
        raise TypeError(
            "replay needs the environment tessera_rl.make returned; adapters and host "
            "environments are not supported yet"
        )
    return native


@contextlib.contextmanager
def record(env, path, *, keyframe_every: int = 0, checkpoint_every: int = 1024):
    """Records the calls on ``env`` to ``path`` while the block runs.

    Every ``env.reset``, ``env.step`` and ``env.fork`` inside the block is
    logged. A keyframe of every instance starts the file, another follows
    every ``keyframe_every`` calls (0: only the first), and a checkpoint of
    every instance's state hash every ``checkpoint_every`` calls and at the
    end. A call the pool refuses without poisoning is not logged; a call that
    poisons it is logged and ends the recording (later calls raise
    ``PoisonedError``).

    The file is written when the block exits normally: to ``path + ".tmp"``,
    flushed, then moved onto ``path`` with ``os.replace``, so ``path`` is
    either absent or complete. If the block raises, the recording is dropped
    and nothing is written.
    """
    native = _native_of(env)
    native._record_start(keyframe_every, checkpoint_every)
    try:
        yield env
    except BaseException:
        native._record_abort()
        raise
    data = native._record_finish()
    target = os.fspath(path)
    tmp = target + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def run(path, env) -> ReplaySummary:
    """Replays the file at ``path`` into ``env`` and checks it.

    ``env`` must be built like the recorded one (same environment, config,
    options and ``num_envs``; the mode and worker count may differ); its
    current state does not matter, since the file starts from a keyframe.
    Raises :class:`ReplayDivergence` at the first failing check, and
    :class:`tessera_rl.PersistError` for a file that is damaged or does not
    match ``env``. On a divergence ``env`` is left as the replay left it.
    """
    native = _native_of(env)
    with open(os.fspath(path), "rb") as f:
        data = f.read()
    result = native._replay(data)
    if result[0] == "diverged":
        raise ReplayDivergence(*result[1:])
    return ReplaySummary(*result[1:])

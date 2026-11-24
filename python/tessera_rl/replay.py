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
- Async-mode environments cannot record or replay from Python yet. In Rust,
  an async file replays only into a pool with batch size 1: batch size
  changes how ids are grouped into batches, not their results, and only
  single-id batches let replay wait for any recorded batch.
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

from tessera_rl._native import ArgumentError, TesseraError

__all__ = ["ReplayDivergence", "ReplaySteps", "ReplaySummary", "record", "run", "steps"]


@dataclass(frozen=True)
class ReplaySummary:
    """What a replay checked.

    ``events``: events replayed; ``calls``: calls re-issued; ``poisoned``: the
    file ends with a poisoning call that the replay reproduced;
    ``unverified_poison``: the file ends with a state write (``fork``) that
    poisoned the recorded pool, which replay cannot re-run, so the replayed
    pool is not poisoned; ``unclaimed``: for an async file, ids that came
    back during the replay but that no recorded ``recv`` claimed before the
    file ended (a file cut while ids were outstanding), empty otherwise;
    ``stopped_at``: for a stepped replay (:func:`steps`) closed before the
    end, the index of the next event, else ``None``; ``started_at``: the
    keyframe a stepped replay seeked to, else ``None``.
    """

    events: int
    calls: int
    poisoned: bool = False
    unverified_poison: bool = False
    unclaimed: tuple[int, ...] = ()
    stopped_at: int | None = None
    started_at: int | None = None

    @property
    def verified(self) -> bool:
        """Whether the whole file was replayed and passed every check. A
        replay stopped early, or started from a seek, is never verified."""
        return self.stopped_at is None and self.started_at is None


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
    # The native object itself (test-hooks factories return one) or an Env.
    if hasattr(env, "_replay") and hasattr(env, "_record_start"):
        return env
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


class ReplaySteps:
    """A stepped replay (issue #378), as :func:`steps` returns it.

    Iterating yields an event index after every ``every``-th replayed call
    (``reset``, ``step``, ``fork``); between two yields the environment is
    free for calls that do not change it (``snapshot``, ``state_hash``,
    ``render``, ``scene``, ``outputs``). Calls that change it (``reset``,
    ``step``, ``fork``, ``import_record``, ``load``, recording or another
    replay) raise ``ArgumentError`` until the replay ends.

    The replay ends when the file does (the end-of-file checks run and
    :attr:`summary` is verified), or on :meth:`close` (also on leaving a
    ``with`` block or on a divergence): then :attr:`summary` has
    ``stopped_at`` set and is not verified. A divergence raises
    :class:`ReplayDivergence` from the iteration.
    """

    def __init__(self, native, every: int) -> None:
        self._native = native
        self._every = every
        self._active = True
        #: The summary once the replay ended, else ``None``.
        self.summary: ReplaySummary | None = None

    def __iter__(self) -> ReplaySteps:
        return self

    def __next__(self) -> int:
        if not self._active:
            raise StopIteration
        try:
            result = self._native._replay_advance(self._every)
        except BaseException:
            self.close()
            raise
        if result[0] == "diverged":
            self.close()
            raise ReplayDivergence(*result[1:])
        if result[1] is None:
            self._finish()
            raise StopIteration
        return result[1]

    def keyframes(self) -> tuple[int, ...]:
        """Event indices of the keyframes :meth:`seek` accepts."""
        return tuple(self._native._replay_keyframes())

    def seek(self, keyframe: int) -> None:
        """Imports the keyframe at event ``keyframe`` into every instance and
        continues the replay after it. Counters of the final summary then
        start at that keyframe."""
        self._native._replay_seek(int(keyframe))

    def _finish(self) -> ReplaySummary:
        self._active = False
        result = self._native._replay_finish()
        if result[0] == "diverged":
            raise ReplayDivergence(*result[1:])
        self.summary = ReplaySummary(*result[1:])
        return self.summary

    def close(self) -> ReplaySummary | None:
        """Ends the replay where it stands; never claims verification for
        unreplayed events. Returns the summary (``None`` if it already
        ended with a divergence)."""
        if self._active:
            return self._finish()
        return self.summary

    def __del__(self) -> None:
        # Dropped without close(): release the pool, which stays locked
        # against mutation while a cursor is active.
        if getattr(self, "_active", False):
            try:
                self._finish()
            except Exception:
                pass

    def __enter__(self) -> ReplaySteps:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def steps(path, env, *, every: int = 1) -> ReplaySteps:
    """Replays the file at ``path`` into ``env`` a few calls at a time.

    Returns a :class:`ReplaySteps` iterator that yields an event index after
    every ``every``-th call, leaving ``env`` free for inspection or rendering
    between yields (see :func:`tessera_rl.render.replay_video`). Use it in a
    ``with`` block, or call ``close()``, if you may stop early. ``env`` must
    be built as for :func:`run`; the file is refused here as :func:`run`
    refuses it.
    """
    every = int(every)
    if every < 1:
        raise ArgumentError("every must be at least 1")
    native = _native_of(env)
    with open(os.fspath(path), "rb") as f:
        data = f.read()
    native._replay_begin(data)
    return ReplaySteps(native, every)

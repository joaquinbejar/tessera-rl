# SPDX-License-Identifier: MIT OR Apache-2.0

from contextlib import AbstractContextManager
from dataclasses import dataclass
from os import PathLike
from types import TracebackType
from typing import Any

from tessera_rl._env import Env
from tessera_rl._native import TesseraError

__all__ = ["ReplayDivergence", "ReplaySteps", "ReplaySummary", "record", "run", "steps"]

@dataclass(frozen=True)
class ReplaySummary:
    events: int
    calls: int
    poisoned: bool = False
    unverified_poison: bool = False
    unclaimed: tuple[int, ...] = ()
    stopped_at: int | None = None
    started_at: int | None = None
    @property
    def verified(self) -> bool: ...

class ReplayDivergence(TesseraError):
    event_index: int
    kind: str
    instance: int | None
    last_matching_checkpoint: int | None
    def __init__(
        self,
        event_index: int,
        kind: str,
        instance: int | None,
        last_matching_checkpoint: int | None,
    ) -> None: ...

def record(
    env: Env,
    path: str | PathLike[str],
    *,
    keyframe_every: int = 0,
    checkpoint_every: int = 1024,
) -> AbstractContextManager[Env]: ...
def run(path: str | PathLike[str], env: Env) -> ReplaySummary: ...

class ReplaySteps:
    summary: ReplaySummary | None
    def __init__(self, native: Any, every: int) -> None: ...
    def __del__(self) -> None: ...
    def __iter__(self) -> ReplaySteps: ...
    def __next__(self) -> int: ...
    def keyframes(self) -> tuple[int, ...]: ...
    def seek(self, keyframe: int) -> None: ...
    def close(self) -> ReplaySummary | None: ...
    def __enter__(self) -> ReplaySteps: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...

def steps(path: str | PathLike[str], env: Env, *, every: int = 1) -> ReplaySteps: ...

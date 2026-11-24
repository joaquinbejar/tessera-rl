# SPDX-License-Identifier: MIT OR Apache-2.0

from contextlib import AbstractContextManager
from dataclasses import dataclass
from os import PathLike

from tessera_rl._env import Env
from tessera_rl._native import TesseraError

__all__ = ["ReplayDivergence", "ReplaySummary", "record", "run"]

@dataclass(frozen=True)
class ReplaySummary:
    events: int
    calls: int
    poisoned: bool = False
    unverified_poison: bool = False

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

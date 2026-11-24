# SPDX-License-Identifier: MIT OR Apache-2.0
from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Any, NamedTuple, NoReturn

import numpy as np
import numpy.typing as npt

from tessera_rl._env import Env as Env
from tessera_rl._env import _Ids
from tessera_rl._env import Step as Step

PROTOCOL: int
PROTOCOLS: tuple[int, ...]
GROUP: str

def host_ids() -> list[str]: ...
def is_host_id(env_id: str) -> bool: ...
def make(
    env_id: str,
    known_first_party: Iterable[str],
    num_envs: int,
    seed: int,
    outputs: str,
    kwargs: dict[str, Any],
) -> HostEnv: ...
def native_for(
    env_id: str, num_envs: int, seed: int, outputs: str, kwargs: dict[str, Any]
) -> _HostNative: ...
def refuse(env_id: str, module: str) -> None: ...
def host_infos(
    out: dict[str, Any], owned: bool, n: int, agents: int = 1, refuses: bool = False
) -> dict[str, Any]: ...

class _HostNative:
    def __init__(self, obj: Any, outputs: str) -> None: ...
    def layout(self) -> dict[str, Any]: ...
    def reset(self, seed: int | None = None, ids: Any = None, *_: Any) -> dict[str, Any]: ...
    def step(self, actions: Any) -> dict[str, Any]: ...
    def is_poisoned(self) -> bool: ...
    def close(self) -> None: ...

def check(obj: Any) -> None: ...

class HostStep(NamedTuple):
    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    agent_terminated: np.ndarray
    agent_truncated: np.ndarray
    alive: np.ndarray
    episode_terminated: np.ndarray
    episode_truncated: np.ndarray
    masks: np.ndarray | None
    acting: np.ndarray
    slot_generations: np.ndarray
    aborted: np.ndarray
    abort_code: np.ndarray
    pending_reset: np.ndarray
    refused: np.ndarray | None = None
    refuse_code: np.ndarray | None = None

class HostEnv:
    mode: str
    num_envs: int
    agents: int
    initial_agents: int
    spawns: bool
    action_choices: int
    action_bounds: tuple[float, float] | None
    action_nvec: tuple[int, ...]
    mask_heads: tuple[tuple[int, tuple[int, ...], int], ...]
    entity_group_heads: int
    turns: str
    outputs: str
    halves: None
    env_id: str
    env_version: int
    auto_reset: str
    kind: str
    canvas: None
    refuses_agents: bool
    snapshots: bool
    def __init__(self, obj: Any, outputs: str = "borrowed") -> None: ...
    def reset(self, seed: int | None = None) -> HostStep: ...
    def step(self, actions: npt.ArrayLike) -> HostStep: ...
    def is_poisoned(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self) -> HostEnv: ...
    def __exit__(
        self,
        *exc: type[BaseException] | BaseException | TracebackType | None,
    ) -> None: ...
    def snapshot(self, index: int) -> bytes: ...
    def restore(self, ids: _Ids, snapshots: Sequence[bytes]) -> None: ...
    def restore_all(self, snapshots: Sequence[bytes]) -> None: ...
    def import_record(self, index: int, record: bytes) -> None: ...
    def fork(self, src: int, dst: list[int]) -> None: ...
    def state_hash(self, index: int) -> int: ...
    def encode(self, ids: _Ids = None) -> bytes: ...
    def load(
        self,
        data: bytes,
        ids: _Ids = None,
        max_instances: None = None,
        max_decoded_bytes: None = None,
    ) -> None: ...
    # Present for Env's shape; each raises CapabilityError.
    def snapshot_all(self, *args: Any, **kwargs: Any) -> NoReturn: ...
    def send(self, *args: Any, **kwargs: Any) -> NoReturn: ...
    def recv(self, *args: Any, **kwargs: Any) -> NoReturn: ...
    def render(self, *args: Any, **kwargs: Any) -> NoReturn: ...
    def scene(self, *args: Any, **kwargs: Any) -> NoReturn: ...

def set_fp_policy(policy: str) -> None: ...

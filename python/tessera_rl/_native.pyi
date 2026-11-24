# SPDX-License-Identifier: MIT OR Apache-2.0
"""Type stubs for the native extension module (issue #348).

Every native function and method carries a ``__text_signature__`` (issue
#376), so stubtest checks these parameters against the runtime. Test-hook
functions (built only with the ``test-hooks`` feature) are not stubbed.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias, final

import numpy as np
import numpy.typing as npt

_Ids: TypeAlias = int | Sequence[int] | npt.NDArray[np.integer[Any]] | range | None
_Out: TypeAlias = dict[str, Any]

class TesseraError(Exception): ...
class ConfigError(TesseraError): ...
class ArgumentError(TesseraError): ...
class ActionError(TesseraError): ...
class RecordError(TesseraError): ...
class PendingResetError(TesseraError): ...
class PoisonedError(TesseraError): ...
class ConcurrencyError(TesseraError): ...
class ExhaustedError(TesseraError): ...
class FloatingPointEnvironmentError(TesseraError): ...
class SearchError(TesseraError): ...
class EvaluatorError(SearchError): ...
class ForkError(TesseraError): ...
class PersistError(TesseraError): ...
class OnnxError(TesseraError): ...
class CapabilityError(TesseraError): ...

def version() -> str: ...
def set_fp_policy(name: Literal["canonicalize", "reject"], /) -> None: ...
def get_fp_policy() -> Literal["canonicalize", "reject"]: ...
def fp_non_canonical_entries() -> int: ...
def fp_boundary() -> str: ...
def physical_cores() -> int: ...
def build_profile() -> Literal["release", "debug"]: ...
def encode_gif(frames: bytes, width: int, height: int, fps: int, /) -> bytes: ...
def registered_envs() -> list[tuple[str, int, int, str, bool]]: ...
def env_records() -> list[dict[str, Any]]: ...

@final
class OutputView:
    """A borrowed, read-only view of one output buffer."""

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any: ...
    def __dlpack_device__(self, /) -> tuple[int, int]: ...
    def generation(self) -> int: ...
    def is_stale(self) -> bool: ...

@final
class PoolSnapshot:
    """An opaque whole-pool snapshot."""

@final
class WatchHandle:
    """The single consumer of a pool's live watch (use ``Env.watch``)."""

    def request(self, instance: int, /) -> None: ...
    def take(self) -> tuple[int, bytes] | None: ...
    def close(self) -> None: ...
    def record_len(self) -> int: ...

@final
class Archive:
    """A native archive of saved states (use ``tessera_rl.Archive``)."""

@final
class Env:
    """A native pool (use ``tessera_rl.make``)."""

    def layout(self) -> dict[str, Any]: ...
    def outputs(self) -> _Out: ...
    def is_poisoned(self) -> bool: ...
    def close(self) -> None: ...
    def reset(
        self, seed: int | list[int | None] | None = None, ids: _Ids = None, delivered: int | None = None, /
    ) -> Any: ...
    def step(self, actions: bytes, /) -> _Out: ...
    def snapshot(self, id: int, /) -> Any: ...
    def illegal_actions(self, id: int, /) -> list[list[int]]: ...
    def state_hash(self, id: int, /) -> int: ...
    def canvas(self) -> tuple[int, int] | None: ...
    def render(self, ids: Sequence[int], /) -> tuple[bytes, list[bool]]: ...
    def scene(self, id: int, /) -> tuple[bytes, bool]: ...
    def import_record(self, id: int, record: bytes, /) -> None: ...
    def encode(self, ids: _Ids, /) -> bytes: ...
    def load(
        self, data: bytes, ids: _Ids, max_instances: int | None, max_decoded_bytes: int | None, /
    ) -> _Out | None: ...
    def send(self, half: Any, actions: bytes, delivered: int, /) -> int: ...
    def recv(self, delivered: int, /) -> tuple[Any, ...]: ...
    def restore(self, ids: _Ids, snapshots: Sequence[Any], delivered: int, /) -> Any: ...
    def snapshot_all(self) -> PoolSnapshot: ...
    def restore_all(self, snapshot: PoolSnapshot, /) -> Any: ...
    def fork(self, src: int, dst: Sequence[int], /) -> None: ...
    def watch(self) -> WatchHandle: ...
    def _replay_begin(self, data: bytes, archive: Archive | None, /) -> None: ...
    def _make_archive(self, snapshots: list[bytes], /) -> tuple[Archive, int, int]: ...
    def set_reset_source(self, archive: Archive | None, weights: list[int] | None, /) -> None: ...
    def reset_from(
        self, ids: Sequence[int], entries: Sequence[int], delivered: int | None = None, /
    ) -> Any: ...
    def _replay_advance(self, calls: int, /) -> tuple[Any, ...]: ...
    def _replay_seek(self, keyframe: int, /) -> None: ...
    def _replay_keyframes(self) -> tuple[int, ...]: ...
    def _replay_finish(self) -> tuple[Any, ...]: ...

_Evaluator: TypeAlias = Callable[[bytes, bytes, bytes, int], tuple[bytes, ...]]

@final
class Search:
    """A native search (use ``tessera_rl.search.Search``)."""

    def layout(self) -> dict[str, Any]: ...
    def set_root(self, record: bytes, /) -> None: ...
    def run(self, evaluator: _Evaluator, /) -> None: ...
    def root_visits(self) -> list[int]: ...
    def select_action(self, temperature: float, /) -> int: ...
    def improved_policy(self) -> list[float] | None: ...
    def considered(self) -> list[int] | None: ...
    def calls(self) -> int: ...
    def tree_hash(self) -> int: ...
    def is_privileged(self) -> bool: ...
    def attach_onnx(
        self,
        *,
        model: str,
        library: str,
        accept_telemetry: bool = False,
        max_model_bytes: int | None = None,
    ) -> dict[str, Any]: ...
    def run_onnx(self) -> None: ...

@final
class SearchGroup:
    """Native grouped searches (use ``tessera_rl.search.SearchGroup``)."""

    def layout(self) -> dict[str, Any]: ...
    def last_failure(self) -> tuple[int | None, str, bool] | None: ...
    def set_root(self, i: int, record: bytes, /) -> None: ...
    def set_active(self, i: int, flag: bool, /) -> None: ...
    def active(self, i: int, /) -> bool: ...
    def run(self, evaluator: _Evaluator, /) -> None: ...
    def root_visits(self, i: int, /) -> list[int]: ...
    def select_action(self, i: int, temperature: float, /) -> int: ...
    def improved_policy(self, i: int, /) -> list[float] | None: ...
    def considered(self, i: int, /) -> list[int] | None: ...
    def calls(self, i: int, /) -> int: ...
    def tree_hash(self, i: int, /) -> int: ...

def make_cartpole(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_acrobot(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_mountaincar(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_pendulum(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_tag(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    width: int | None = None,
    height: int | None = None,
    pursuers: int | None = None,
    evaders: int | None = None,
    slip: int | None = None,
    radius: int | None = None,
) -> Env: ...
def make_rts(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    size: int | None = None,
    decision_ticks: int | None = None,
    max_units: int | None = None,
    step_limit: int | None = None,
    seat_swap: bool | None = None,
    opponent: int | None = None,
) -> Env: ...
def make_lob(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    agents: int | None = None,
    steps: int | None = None,
    depth: int | None = None,
    min_tick: int | None = None,
    max_tick: int | None = None,
    initial_price: int | None = None,
    up_permille: int | None = None,
    down_permille: int | None = None,
    latency: int | None = None,
    size_0: int | None = None,
    size_1: int | None = None,
    size_2: int | None = None,
    size_3: int | None = None,
    max_position: int | None = None,
    initial_cash: int | None = None,
    allow_short_cash: bool | None = None,
    bankrupt: int | None = None,
    noise: int | None = None,
    reward_shift: int | None = None,
) -> Env: ...
def make_connect4(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_lanterns(
    *,
    players: int | None = None,
    zero_on_fuse: bool | None = None,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_leduc(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_othello(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
) -> Env: ...
def make_go9(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    komi_half_points: int | None = None,
) -> Env: ...
def make_grid(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    size: int | None = None,
    max_steps: int | None = None,
) -> Env: ...
def make_keydoor(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    size: int | None = None,
    agents: int | None = None,
) -> Env: ...
def make_gather(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    size: int | None = None,
    items: int | None = None,
    max_steps: int | None = None,
) -> Env: ...
def make_snake(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    size: int | None = None,
    agents: int | None = None,
    obs: str | None = None,
) -> Env: ...
def make_brickwall(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    frameskip: int | None = None,
    lives: int | None = None,
    obs: str | None = None,
) -> Env: ...
def make_paddleduel(
    *,
    mode: str | None = None,
    batch_size: int | None = None,
    gather: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    max_episode_steps: int | None = None,
    auto_reset: str | None = None,
    chunk_size: int | None = None,
    workers: int | None = None,
    spin: int | None = None,
    outputs: str | None = None,
    agents: int | None = None,
    frameskip: int | None = None,
    max_score: int | None = None,
    bounce: int | None = None,
    obs: str | None = None,
) -> Env: ...
def make_search(
    *,
    env_id: str | None = None,
    capacity: int | None = None,
    batch: int | None = None,
    eviction: str | None = None,
    c_puct: float | None = None,
    fpu: float | None = None,
    virtual_loss: int | None = None,
    batch_size: int | None = None,
    simulations: int | None = None,
    noise_alpha: float | None = None,
    noise_epsilon: float | None = None,
    seed: int | None = None,
    root: str | None = None,
    gumbel_m: int | None = None,
    c_visit: float | None = None,
    c_scale: float | None = None,
) -> Search: ...
def make_search_group(
    *,
    trees: int,
    slot_options: Sequence[dict[str, Any] | None] | None = None,
    env_id: str | None = None,
    capacity: int | None = None,
    batch: int | None = None,
    eviction: str | None = None,
    c_puct: float | None = None,
    fpu: float | None = None,
    virtual_loss: int | None = None,
    batch_size: int | None = None,
    simulations: int | None = None,
    noise_alpha: float | None = None,
    noise_epsilon: float | None = None,
    seed: int | None = None,
    root: str | None = None,
    gumbel_m: int | None = None,
    c_visit: float | None = None,
    c_scale: float | None = None,
) -> SearchGroup: ...
def onnx_load(*, library: str, accept_telemetry: bool = False) -> dict[str, Any]: ...

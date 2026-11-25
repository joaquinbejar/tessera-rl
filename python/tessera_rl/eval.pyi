# SPDX-License-Identifier: MIT OR Apache-2.0
from typing import Any, Callable

WILSON_Z: float
WAVE: int
MAX_STEPS: int

def wilson(k: float, n: int, z: float = ...) -> list[float]: ...
def parse_population(text: str) -> list[tuple[int, Any]]: ...
def from_export(text: str, ids: list[int] | None) -> list[tuple[int, Any]]: ...
def provenance() -> dict[str, Any]: ...
def evaluate(
    env_id: str,
    policy: Callable[..., Any],
    opponents: list[tuple[int, Any]],
    *,
    seeds: list[int],
    episodes: int,
    pools: int = 8,
    env_kwargs: dict[str, Any] | None = None,
    opponent_slots: list[int] | None = None,
    external: dict[int, Callable[..., Any]] | None = None,
    names: dict[str, Any] | None = None,
    max_steps: int = ...,
) -> dict[str, Any]: ...
def main(argv: list[str] | None = None) -> int: ...

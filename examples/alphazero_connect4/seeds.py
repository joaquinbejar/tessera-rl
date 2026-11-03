# SPDX-License-Identifier: MIT OR Apache-2.0
"""Domain-separated seeds of the self-play baseline (P2-M3 PR 2).

A seed is the first 8 bytes, little-endian, of the SHA-256 of the
canonical tuple bytes::

    b"tessera-rl/p2-m3/seed/v1\\x00"
    || run_seed as 8 bytes little-endian
    || domain as UTF-8 || b"\\x00"
    || index as 8 bytes little-endian

The domains are ``env`` (index: the logical game id), ``search`` (index:
the game id) and ``shuffle`` (index: the epoch); PR 3 adds ``generation``
(the self-play run seed of generation k) and ``sample`` (the batch sampler
of generation k). No seed depends on the
worker count, the worker or the shard. Every derived seed is checked
against the seed ranges the project already uses (the ADR-0042 bot tie
seeds and the evaluation, held-out and dev opening seeds), and a
collision is an error.
"""

from __future__ import annotations

import hashlib

PREFIX = b"tessera-rl/p2-m3/seed/v1\x00"
DOMAINS = ("env", "search", "shuffle", "generation", "sample")

# Seed ranges in use elsewhere: (name, base); each covers base + [0, 1000).
RESERVED = (
    ("tie", 0x7E55E2A000000000),
    ("eval", 0x5032473100000000),
    ("dev", 0x5032473140000000),
    ("heldout", 0x5032473180000000),
)
RESERVED_SPAN = 1000

DERIVATION = (
    "sha256(b'tessera-rl/p2-m3/seed/v1\\x00' || u64le(run_seed) || utf8(domain) || b'\\x00' || u64le(index))[0:8] "
    "as u64 little-endian; domains env(g), search(g), shuffle(epoch), generation(k), sample(k)"
)


class SeedCollision(ValueError):
    """A derived seed falls in a reserved range."""


def canonical_bytes(run_seed: int, domain: str, index: int) -> bytes:
    if domain not in DOMAINS:
        raise ValueError(f"unknown seed domain {domain!r}")
    if not (0 <= run_seed < 1 << 64 and 0 <= index < 1 << 64):
        raise ValueError("run_seed and index must be unsigned 64-bit")
    return PREFIX + run_seed.to_bytes(8, "little") + domain.encode() + b"\x00" + index.to_bytes(8, "little")


def derive(run_seed: int, domain: str, index: int) -> int:
    """The seed of ``domain`` and ``index``; raises ``SeedCollision``."""
    seed = int.from_bytes(hashlib.sha256(canonical_bytes(run_seed, domain, index)).digest()[:8], "little")
    for name, base in RESERVED:
        if base <= seed < base + RESERVED_SPAN:
            raise SeedCollision(f"{domain}({index}) = {seed:#x} is in the {name} range")
    return seed


def env(run_seed: int, g: int) -> int:
    return derive(run_seed, "env", g)


def search(run_seed: int, g: int) -> int:
    return derive(run_seed, "search", g)


def shuffle(run_seed: int, epoch: int) -> int:
    return derive(run_seed, "shuffle", epoch)


def generation(base: int, k: int) -> int:
    return derive(base, "generation", k)


def sample(base: int, k: int) -> int:
    return derive(base, "sample", k)

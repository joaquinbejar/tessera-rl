# SPDX-License-Identifier: MIT OR Apache-2.0
"""Canonical, atomic artifacts of the self-play baseline (P2-M3 PR 2).

A shard is one file: a magic line, one line of sorted-key JSON (the
arrays' names, dtypes and shapes, and the metadata), then the arrays'
bytes in name order. Its bytes depend on its content only, never on time
or paths, so its SHA-256 is its identity. Every file is written to
``<name>.tmp``, flushed, fsync'd and renamed. A manifest is written last
and says ``"complete": true``. A directory without it, or with any listed
file missing or changed, is incomplete, and training refuses it.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

MAGIC = b"TESSERA-SHARD 1\n"


class Incomplete(RuntimeError):
    """A run directory without a complete, consistent manifest."""


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def encode_shard(arrays: dict[str, np.ndarray], meta: dict) -> bytes:
    names = sorted(arrays)
    header = {
        "arrays": [
            {"name": n, "dtype": arrays[n].dtype.str, "shape": list(arrays[n].shape)} for n in names
        ],
        "meta": meta,
    }
    parts = [MAGIC, canonical_json(header), b"\n"]
    parts += [np.ascontiguousarray(arrays[n]).tobytes() for n in names]
    return b"".join(parts)


def decode_shard(data: bytes) -> tuple[dict[str, np.ndarray], dict]:
    if not data.startswith(MAGIC):
        raise Incomplete("not a shard")
    end = data.index(b"\n", len(MAGIC))
    header = json.loads(data[len(MAGIC) : end])
    at = end + 1
    arrays = {}
    for a in header["arrays"]:
        dtype = np.dtype(a["dtype"])
        n = int(np.prod(a["shape"], dtype=np.int64)) * dtype.itemsize
        arrays[a["name"]] = np.frombuffer(data[at : at + n], dtype=dtype).reshape(a["shape"]).copy()
        at += n
    if at != len(data):
        raise Incomplete("trailing bytes in shard")
    return arrays, header["meta"]


def fresh_dir(directory: str, allowed: tuple[str, ...] = ()) -> None:
    """Creates ``directory`` or checks that it is empty (apart from
    ``allowed`` prefixes): a run never reuses an earlier run's output, so
    a failed retry cannot leave an old complete manifest behind."""
    os.makedirs(directory, exist_ok=True)
    stale = [n for n in os.listdir(directory) if not n.startswith(allowed)]
    if stale:
        raise Incomplete(f"{directory} is not empty ({sorted(stale)[:3]}...): use a new output directory")


def write_manifest(directory: str, manifest: dict) -> str:
    """Writes ``manifest.json`` (canonical JSON, ``complete`` set) last;
    returns its SHA-256, the run's identity."""
    body = canonical_json({**manifest, "complete": True})
    write_atomic(os.path.join(directory, "manifest.json"), body)
    return sha256(body)


def load_manifest(directory: str) -> tuple[dict, str]:
    """The manifest and its digest, after checking every listed file;
    raises ``Incomplete``."""
    path = os.path.join(directory, "manifest.json")
    if not os.path.exists(path):
        raise Incomplete(f"{directory}: no manifest (the run did not finish)")
    with open(path, "rb") as f:
        body = f.read()
    manifest = json.loads(body)
    if manifest.get("complete") is not True:
        raise Incomplete(f"{directory}: manifest not complete")
    for entry in manifest.get("files", []):
        p = os.path.join(directory, entry["name"])
        if not os.path.exists(p):
            raise Incomplete(f"{directory}: {entry['name']} is missing")
        if sha256_file(p) != entry["sha256"]:
            raise Incomplete(f"{directory}: {entry['name']} does not match its digest")
    return manifest, sha256(body)

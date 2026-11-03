# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checkpoints with provenance (P2-M3 PR 2).

A checkpoint is two files: ``<name>.pt``, the network's ``state_dict``
(weights and BatchNorm running statistics) serialized by ``torch.save``,
and ``<name>.json``, its canonical manifest. The manifest holds the file's
SHA-256 (the checkpoint's identity), the net size, and whatever the
producer records (parent digest, data digest, steps, hyperparameters).
Both are written atomically, the manifest last. Loading checks the digest.
"""

from __future__ import annotations

import io
import json
import os

import artifacts
import net as netmod
import torch


class NonFinite(ValueError):
    """A checkpoint with a non-finite weight or statistic."""


def check_finite(state: dict, what: str) -> None:
    for name, t in state.items():
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            raise NonFinite(f"{what}: {name} is not finite")


def save(model: torch.nn.Module, path: str, size: str, meta: dict) -> str:
    """Writes the checkpoint at ``path`` (``.pt``) and its manifest;
    returns its digest. Refuses non-finite state, and an existing file:
    a checkpoint is never overwritten."""
    check_finite(model.state_dict(), path)
    if os.path.exists(path) or os.path.exists(os.path.splitext(path)[0] + ".json"):
        raise FileExistsError(f"{path} exists: checkpoints are never overwritten")
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    data = buf.getvalue()
    digest = artifacts.sha256(data)
    artifacts.write_atomic(path, data)
    manifest = {"sha256": digest, "size": size, **meta, "complete": True}
    artifacts.write_atomic(os.path.splitext(path)[0] + ".json", artifacts.canonical_json(manifest))
    return digest


def manifest(path: str) -> dict:
    with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
        return json.load(f)


def load(path: str) -> tuple[torch.nn.Module, dict]:
    """The network in eval mode and its manifest; raises
    ``artifacts.Incomplete`` if the file does not match its digest."""
    m = manifest(path)
    with open(path, "rb") as f:
        data = f.read()
    if artifacts.sha256(data) != m.get("sha256") or m.get("complete") is not True:
        raise artifacts.Incomplete(f"{path}: checkpoint does not match its manifest")
    model = netmod.Net(*netmod.SIZES[m["size"]])
    state = torch.load(io.BytesIO(data), weights_only=True)
    check_finite(state, path)
    model.load_state_dict(state)
    return model.eval(), m

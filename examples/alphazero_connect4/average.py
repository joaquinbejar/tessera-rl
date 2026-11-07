# SPDX-License-Identifier: MIT OR Apache-2.0
"""A predeclared weight average of P2-M3 checkpoints, with BatchNorm
calibration (``docs/reviews/p2g1-checkpoint-averaging.md`` r3).

Candidate A is the uniform mean of C21 to C30, the low learning-rate tail
of the M3 run. Its inputs and calibration data are pinned below by full
SHA-256, and ``build`` refuses anything else.

The averaging step:
- every input is loaded through ``ckpt.load``, so its digest is checked;
- every input must have the same keys, in the same order, with the same
  shape and dtype for each key;
- floating tensors are averaged uniformly in float64, in input order,
  and cast to float32 once. BatchNorm running statistics are not
  averaged, because the calibration replaces them;
- the only integer tensors allowed are the BatchNorm
  ``num_batches_tracked`` counters. They are not averaged.

The calibration step is model calibration on training data, without
gradients. It changes no weight:
- the data is C30's training window, read with ``train_step.load_samples``
  in its manifest order;
- 65,536 indices are drawn without replacement from the mirror-augmented
  pool of size ``2N`` by ``PCG64(20261107)``, and kept unsorted;
- they form 64 batches of 1,024, built with the mirror rule of
  ``train_step.train_window``;
- every BatchNorm layer is reset with ``momentum = None`` (a cumulative
  average), and a forward pass runs in train mode under ``no_grad``.

The ``preflight`` and ``verify-look`` commands guard the single look:
it may run only in the registered M3 look runtime, and its report must
show that runtime.

Usage::

    python examples/alphazero_connect4/average.py build --root ROOT --out A.pt --source-sha SHA
    python examples/alphazero_connect4/average.py preflight
    python examples/alphazero_connect4/average.py verify-look --report DIR/look.json --checkpoint-sha SHA
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import ckpt  # noqa: E402
import botmatch  # noqa: E402
import h2h  # noqa: E402
import look  # noqa: E402
import provenance  # noqa: E402
import train_step  # noqa: E402

# Candidate A: C21 to C30, written in gen-020 to gen-029 (selection.jsonl numbering).
INPUTS: tuple[tuple[str, str], ...] = (
    ("gen-020", "f9d5c128f76a61497466ce132cc2187c6b8cc54e5cb1883cb16fddaa41fbd4f6"),
    ("gen-021", "3ca8a42f8094870fe9c4d5de0414208317753e79003606f512e9a0e2c0bd2f26"),
    ("gen-022", "d496491c7daa23898482c3395064f491f67f104fd33e51dd029878c73fbbfe36"),
    ("gen-023", "4c60d98e50563452fa817496c4188356149dae5c66d055bbe5199408d60f1912"),
    ("gen-024", "d563475aec899b1b56133f3bbf78c804888ece68dfdd321788b0b93d69388899"),
    ("gen-025", "c1948bbdbb213d8180a3a6fec0ea65c4acf8c886df492a0a1b1860f259e2d4b5"),
    ("gen-026", "f19c5120bae4fbb7b01df8dd101e57285b06116d596c04dccfdc9a176fb08931"),
    ("gen-027", "6662861d59a1b6c95e2690e4f9d4503aa0521a9d03addb4e0bc0d00300c2b6b8"),
    ("gen-028", "ad5feb68ea769da7db5b069b364d22c019e8671907607da8f1c9e3bfe8e64960"),
    ("gen-029", "30eaf149ab25a4093c16ae4be44d06b031b2132f42d25402790042548d1326f8"),
)
# C30's training window, in its manifest's data_manifests order.
DATA: tuple[tuple[str, str], ...] = (
    ("gen-026", "ca88c5a7e4ccbd2745e15bc235c8d616683889cbfa1f44b9547a0934b820895b"),
    ("gen-027", "568daca458f191876fc1e975dc27584cd09b60697a82f9c633251226817d19ee"),
    ("gen-028", "88d32426c1fb03df5b5148f0db599c30164b55156e53a56fcf1fca3e4be7718a"),
    ("gen-029", "1f7b227854728e8637acea0f0baad61f9e1ee4027d51c5e4607226c48c7a53c9"),
)
SEED = 20261107
SAMPLE = 65536
BATCH = 1024
TAIL_RULE = "uniform mean of C21..C30 (gen-020..gen-029), the low learning-rate tail"
# The runtime block every M3 look recorded (for example gen-005/look-0/look.json).
LOOK_RUNTIME = {
    "host": h2h.REGISTERED_HOST,
    "image_digest": h2h.REGISTERED_IMAGE,
    "cpuset": "2",
    "affinity_cores": 1,
    "torch_threads": 1,
    "torch_interop_threads": 1,
    "deterministic": True,
    "registered": True,
}


class Refused(RuntimeError):
    """An input, schema, sample or runtime that the predeclared design does not allow."""


def is_counter(name: str) -> bool:
    return name.endswith("num_batches_tracked")


def is_running_stat(name: str) -> bool:
    return name.endswith("running_mean") or name.endswith("running_var")


def raw_state(path: str, want: str) -> tuple[dict, str]:
    """The serialized state dict exactly as stored, before any ``Net`` is
    built, and the net size from the manifest. The file's digest must match
    its manifest and ``want``."""
    m = ckpt.manifest(path)
    with open(path, "rb") as f:
        data = f.read()
    digest = artifacts.sha256(data)
    if digest != m.get("sha256") or m.get("complete") is not True:
        raise artifacts.Incomplete(f"{path}: checkpoint does not match its manifest")
    if digest != want:
        raise Refused(f"{path}: digest {digest}, expected {want}")
    state = torch.load(io.BytesIO(data), weights_only=True)
    if not isinstance(state, dict):
        raise Refused(f"{path}: not a state dict")
    return state, m["size"]


def check_schema(path: str, state: dict, reference: dict) -> None:
    """Refuses any difference from ``reference`` in keys, key order, shape
    or dtype, and any non-floating entry other than a counter."""
    if list(state) != list(reference):
        raise Refused(f"{path}: keys or key order differ from the expected schema")
    for k, t in state.items():
        r = reference[k]
        if not isinstance(t, torch.Tensor) or t.shape != r.shape or t.dtype != r.dtype:
            got = f"{t.dtype} {tuple(t.shape)}" if isinstance(t, torch.Tensor) else type(t).__name__
            raise Refused(f"{path}: {k} is {got}, expected {r.dtype} {tuple(r.shape)}")
        if not t.is_floating_point() and not is_counter(k):
            raise Refused(f"{path}: {k} is a non-floating entry other than num_batches_tracked")


def load_inputs(paths: list[str], expected: list[str]) -> tuple[list[dict], str]:
    """The inputs' serialized state dicts and their common net size.
    Refuses a digest other than ``expected`` (in order). Every raw state
    must match, key for key and in order, the schema of a freshly built net
    of the manifest's size, so a dtype or order difference in a file is
    refused rather than cast away by ``load_state_dict``."""
    import net as netmod

    if len(paths) != len(expected) or not paths:
        raise Refused(f"{len(paths)} inputs for {len(expected)} expected digests")
    states, size, reference = [], None, None
    for path, want in zip(paths, expected):
        state, s = raw_state(path, want)
        if size is None:
            size = s
            reference = netmod.Net(*netmod.SIZES[size]).state_dict()
        elif s != size:
            raise Refused(f"{path}: size {s}, expected {size}")
        check_schema(path, state, reference)
        ckpt.check_finite(state, path)
        states.append(state)
    return states, size


def average(states: list[dict]) -> dict:
    """The uniform mean of every floating tensor except the BatchNorm
    running statistics, summed in float64 in input order and cast once.
    Running statistics and counters are copied from the first input, to be
    replaced by ``calibrate``."""
    out = {}
    w = 1.0 / len(states)
    for k, t in states[0].items():
        if t.is_floating_point() and not is_running_stat(k):
            acc = torch.zeros(t.shape, dtype=torch.float64)
            for s in states:
                acc += s[k].to(torch.float64)
            out[k] = (acc * w).to(t.dtype)
        else:
            out[k] = t.clone()
    return out


def sample_indices(n: int, seed: int = SEED, sample: int = SAMPLE) -> np.ndarray:
    """``sample`` indices of the mirror-augmented pool of size ``2n``, drawn
    without replacement and kept in the order drawn."""
    if 2 * n < sample:
        raise Refused(f"the augmented pool has {2 * n} samples, fewer than {sample}")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.choice(2 * n, size=sample, replace=False)


def indices_sha256(idx: np.ndarray) -> str:
    """The digest of the indices as contiguous little-endian uint64."""
    return artifacts.sha256(np.ascontiguousarray(idx, dtype="<u8").tobytes())


def batch_planes(planes_base: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """The input planes of one batch, built as ``train_window`` builds them:
    index ``i >= N`` is the mirror of ``i - N``."""
    half = planes_base.shape[0]
    src = torch.from_numpy(idx % half)
    mirrored = torch.from_numpy(idx >= half)
    x = planes_base[src].clone()
    x[mirrored] = x[mirrored].flip(-1)
    return x


def bn_layers(model: torch.nn.Module) -> list[torch.nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def calibrate(model: torch.nn.Module, planes: np.ndarray, idx: np.ndarray, batch: int = BATCH) -> None:
    """Recomputes every BatchNorm layer's running statistics over the
    batches of ``idx``, in order, without gradients. Leaves the net in eval
    mode; each counter ends at the number of batches."""
    if len(idx) % batch:
        raise Refused(f"{len(idx)} indices are not a whole number of batches of {batch}")
    layers = bn_layers(model)
    if not layers:
        raise Refused("the net has no BatchNorm layer to calibrate")
    saved = [m.momentum for m in layers]
    for m in layers:
        m.reset_running_stats()
        m.momentum = None
    base = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32))
    model.train()
    try:
        with torch.no_grad():
            for b in range(len(idx) // batch):
                model(batch_planes(base, idx[b * batch : (b + 1) * batch]))
    finally:
        for m, mom in zip(layers, saved):
            m.momentum = mom
        model.eval()


def check_output(state: dict, batches: int) -> None:
    ckpt.check_finite(state, "average")
    for k, t in state.items():
        if k.endswith("running_var") and not bool((t > 0).all()):
            raise Refused(f"{k} has a running variance that is not positive")
        if is_counter(k) and int(t) != batches:
            raise Refused(f"{k} is {int(t)}, expected {batches}")


def build_model(paths, expected, data_dirs, expected_data, seed=SEED, sample=SAMPLE, batch=BATCH):
    """The averaged and calibrated net, and its manifest fields."""
    import net as netmod

    states, size = load_inputs(paths, expected)
    model = netmod.Net(*netmod.SIZES[size])
    model.load_state_dict(average(states))
    planes, _policy, _z, digests = train_step.load_samples(data_dirs)
    if digests != list(expected_data):
        raise Refused(f"data manifests {digests}, expected {list(expected_data)}")
    n = len(planes)
    idx = sample_indices(n, seed, sample)
    calibrate(model, planes, idx, batch)
    check_output(model.state_dict(), sample // batch)
    meta = {
        "kind": "p2-m3-average",
        "parents": list(expected),
        "weights": [1.0 / len(expected)] * len(expected),
        "tail_rule": TAIL_RULE,
        "calibration": {
            "policy": "BatchNorm statistics recomputed on training data, without gradients",
            "data_manifests": list(expected_data),
            "n": n,
            "seed": seed,
            "sample": sample,
            "batch": batch,
            "mirrored": int((idx >= n).sum()),
            "indices_sha256": indices_sha256(idx),
            "indices_encoding": "contiguous little-endian uint64",
        },
    }
    return model, size, meta


def preflight(observed: dict | None = None) -> None:
    """Refuses to start the look outside the registered M3 look runtime."""
    obs = h2h.runtime() if observed is None else observed
    bad = {k: obs.get(k) for k, v in LOOK_RUNTIME.items() if obs.get(k) != v}
    if bad:
        raise Refused(f"not the registered look runtime: {bad}")


def is_count(value) -> bool:
    """A nonnegative integer; ``bool`` is not a count."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def frozen_protocol(checkpoint_sha: str) -> dict:
    """The protocol block a look of ``checkpoint_sha`` must report, field
    for field: ADR-0048's frozen budget on the pinned held-out set, with the
    registered bot for the registered image and the M3 seeds."""
    return {
        "checkpoint": checkpoint_sha,
        "sims": look.FROZEN[0],
        "batch_size": look.FROZEN[1],
        "capacity": look.FROZEN[2],
        "openings": look.FROZEN[3],
        "openings_sha256": look.HELDOUT_SHA256,
        "bot_sha256": look.REGISTERED_BOTS[h2h.REGISTERED_IMAGE],
        "search_seed": botmatch.SEARCH_SEED,
        "tie_base": botmatch.TIE_BASE,
        "size": "medium",
    }


def verify_look(report: dict, checkpoint_sha: str) -> dict:
    """Checks a look report on its own evidence, not only ``eligible``: the
    averaged checkpoint, the registered runtime, every field of the frozen
    protocol, and counts that agree with each other."""
    preflight(report.get("runtime", {}))
    if report.get("checkpoint") != checkpoint_sha:
        raise Refused(f"the report is for {report.get('checkpoint')}, not {checkpoint_sha}")
    want = frozen_protocol(checkpoint_sha)
    proto = report.get("protocol", {})
    bad = {k: proto.get(k) for k, v in want.items() if proto.get(k) != v}
    if bad or set(proto) != set(want):
        raise Refused(f"not the frozen protocol: {bad or sorted(set(proto) ^ set(want))}")
    games = 2 * want["openings"]
    wdl = report.get("wdl")
    colours = report.get("by_colour", {})
    if not isinstance(colours, dict):
        raise Refused("the look's colour counts are not a mapping")
    counts = [report.get("games"), report.get("wins")]
    for c in [wdl, *colours.values()]:
        counts.extend(c if isinstance(c, list) else [c])
    if not all(is_count(v) for v in counts):
        raise Refused("the look's counts are not all nonnegative integers")
    if (
        report.get("games") != games
        or not isinstance(wdl, list)
        or len(wdl) != 3
        or sum(wdl) != games
        or report.get("wins") != wdl[0]
        or sorted(colours) != ["bot_first", "candidate_first"]
        or any(len(c) != 3 or sum(c) != want["openings"] for c in colours.values())
        or [sum(c[i] for c in colours.values()) for i in range(3)] != wdl
    ):
        raise Refused("the look's game counts are not consistent")
    if report.get("eligible") is not True or report.get("complete") is not True:
        raise Refused("the look is not complete and eligible")
    return {"wins": report["wins"], "eligible": True, "candidate": report["wins"] >= 336}


def work_build(args) -> int:
    train_step.configure_torch()
    paths = [os.path.join(args.root, d, "train-0", "C.pt") for d, _ in INPUTS]
    data_dirs = [os.path.join(args.root, d, "selfplay-0") for d, _ in DATA]
    model, size, meta = build_model(paths, [s for _, s in INPUTS], data_dirs, [s for _, s in DATA])
    meta["provenance"] = provenance.block(args.source_sha, {"inputs": INPUTS, "data": DATA, "seed": SEED})
    digest = ckpt.save(model, args.out, size, meta)
    print(json.dumps({"sha256": digest, "manifest": os.path.splitext(args.out)[0] + ".json"}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--source-sha", required=True)
    sub.add_parser("preflight")
    v = sub.add_parser("verify-look")
    v.add_argument("--report", required=True)
    v.add_argument("--checkpoint-sha", required=True)
    args = p.parse_args(argv)
    try:
        if args.cmd == "build":
            return work_build(args)
        if args.cmd == "preflight":
            train_step.configure_torch()
            preflight()
            print("registered look runtime")
            return 0
        with open(args.report, encoding="utf-8") as f:
            print(json.dumps(verify_look(json.load(f), args.checkpoint_sha)))
        return 0
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

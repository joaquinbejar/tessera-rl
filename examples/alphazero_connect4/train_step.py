# SPDX-License-Identifier: MIT OR Apache-2.0
"""The minimal training step of the P2-M3 baseline (PR 2), supervised.

``init`` writes a random-initialized checkpoint, C0. ``train`` reads
complete self-play runs (refusing incomplete ones), trains a checkpoint
for whole epoch-equivalents, and writes the child checkpoint with its
provenance.

The training contract is frozen (PR 2 design, revision 3):
- SGD with lr 0.02, momentum 0.9, no Nesterov, weight decay 1e-4;
- loss = cross-entropy(policy) + 1.0 x MSE(value);
- batch 1024 from the left-right mirror-augmented samples;
- one epoch-equivalent is floor(samples / batch) steps over a shuffle
  seeded by ``seeds.shuffle(run_seed, epoch)``, and the last partial
  batch is dropped and counted;
- BatchNorm is in train mode in the step and in eval mode everywhere
  else.

The targets are the root visit distribution (zero on illegal moves by
construction, checked) and the mover-relative outcome.

Usage::

    python examples/alphazero_connect4/train_step.py init --size medium --seed 7 --out C0.pt
    python examples/alphazero_connect4/train_step.py train --checkpoint C0.pt --data DIR \\
        --out C1.pt --run-seed 1 --ledger LEDGER.json --cores 1 \\
        --budget-core-hours 12 --wall-cap-s 1800
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import augment  # noqa: E402
import ckpt  # noqa: E402
import net as netmod  # noqa: E402
import provenance  # noqa: E402
import schedule  # noqa: E402
import seeds  # noqa: E402
import supervisor  # noqa: E402

HYPER = {
    "optimizer": "sgd",
    "lr": 0.02,
    "momentum": 0.9,
    "nesterov": False,
    "weight_decay": 1e-4,
    "value_loss_weight": 1.0,
    "batch": 1024,
}


def configure_torch() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


SCHEMA = {
    "planes": (np.dtype(np.uint8), (2, 6, 7)),
    "visits": (np.dtype(np.uint32), (7,)),
    "z": (np.dtype(np.int8), ()),
    "g": (np.dtype(np.uint32), ()),
    "ply": (np.dtype(np.uint16), ()),
}


def check_shard(name: str, arrays: dict) -> int:
    """Checks a shard's schema and targets; returns its sample count or
    raises ``artifacts.Incomplete``."""
    if set(arrays) != set(SCHEMA):
        raise artifacts.Incomplete(f"{name}: arrays {sorted(arrays)}, expected {sorted(SCHEMA)}")
    n = len(arrays["z"])
    for key, (dtype, shape) in SCHEMA.items():
        a = arrays[key]
        if a.dtype != dtype or a.shape != (n, *shape):
            raise artifacts.Incomplete(f"{name}: {key} is {a.dtype} {a.shape}, expected {dtype} {(n, *shape)}")
    if n == 0:
        raise artifacts.Incomplete(f"{name}: no samples")
    if not np.isin(arrays["z"], (-1, 0, 1)).all():
        raise artifacts.Incomplete(f"{name}: an outcome outside {{-1, 0, 1}}")
    if (arrays["planes"] > 1).any():
        raise artifacts.Incomplete(f"{name}: a plane value other than 0 or 1")
    total = arrays["visits"].astype(np.uint64).sum(axis=1)
    if (total == 0).any():
        raise artifacts.Incomplete(f"{name}: a sample without visits")
    # Illegal moves have zero visits: the mover's top cell is filled
    # (either plane) exactly where the visits must be 0.
    full = arrays["planes"][:, :, 0, :].sum(axis=1) > 0
    if (arrays["visits"][full] != 0).any():
        raise artifacts.Incomplete(f"{name}: visits on an illegal move")
    return n


def load_samples(directories: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Planes, visit policies and outcomes of complete runs, and their
    manifest digests; raises ``artifacts.Incomplete``."""
    planes, policy, z, digests = [], [], [], []
    for d in directories:
        manifest, digest = artifacts.load_manifest(d)
        digests.append(digest)
        for entry in manifest["files"]:
            with open(os.path.join(d, entry["name"]), "rb") as f:
                arrays, _ = artifacts.decode_shard(f.read())
            check_shard(f"{d}/{entry['name']}", arrays)
            v = arrays["visits"].astype(np.float64)
            p = v / v.sum(axis=1, keepdims=True)
            if not (np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1.0)):
                raise artifacts.Incomplete(f"{d}/{entry['name']}: a policy is not a distribution")
            planes.append(arrays["planes"])
            policy.append(p)
            z.append(arrays["z"].astype(np.float64))
    return np.concatenate(planes), np.concatenate(policy), np.concatenate(z), digests


class Diverged(RuntimeError):
    """A non-finite loss or parameter during training."""


def train(model, planes, policy, z, run_seed: int, epochs: int, batch: int, clock=None):
    """Trains in place; returns the per-step losses and the step count."""
    p, pol, v = augment.augment(planes.astype(np.float32), policy, z)
    x_all = torch.from_numpy(np.ascontiguousarray(p))
    pol_all = torch.from_numpy(np.ascontiguousarray(pol, dtype=np.float32))
    v_all = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    opt = torch.optim.SGD(
        model.parameters(),
        lr=HYPER["lr"],
        momentum=HYPER["momentum"],
        nesterov=HYPER["nesterov"],
        weight_decay=HYPER["weight_decay"],
    )
    n = len(v)
    steps_per_epoch = n // batch
    losses = []
    model.train()
    try:
        for epoch in range(epochs):
            order = np.random.Generator(np.random.PCG64(seeds.shuffle(run_seed, epoch))).permutation(n)
            for s in range(steps_per_epoch):
                idx = torch.from_numpy(order[s * batch : (s + 1) * batch])
                logits, values = model(x_all[idx])
                lp = -(pol_all[idx] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
                lv = ((values - v_all[idx]) ** 2).mean()
                loss = lp + HYPER["value_loss_weight"] * lv
                if not bool(torch.isfinite(loss)):
                    raise Diverged(f"epoch {epoch} step {s}: the loss is not finite")
                opt.zero_grad()
                loss.backward()
                opt.step()
                if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()):
                    raise Diverged(f"epoch {epoch} step {s}: a parameter is not finite")
                losses.append([lp.item(), lv.item()])
    finally:
        model.eval()
    return losses, {
        "augmented_samples": n,
        "steps_per_epoch": steps_per_epoch,
        "dropped_per_epoch": n - steps_per_epoch * batch,
        "epochs": epochs,
        "steps": steps_per_epoch * epochs,
    }


EXIT_DIVERGED = 5


def train_window(model, planes, policy, z, generation: int, base: int, steps: int, batch: int):
    """PR 3: ``steps`` steps of generation ``generation`` (at most
    ``schedule.STEPS``), each a batch drawn uniformly with replacement from
    the mirror-augmented window pool. The rate follows the registered
    schedule at ``t = generation * schedule.STEPS + s``. A fresh optimizer
    is built each generation: momentum is reset by design. Returns the
    per-step losses and counts."""
    if not 1 <= steps <= schedule.STEPS:
        raise ValueError(f"steps must be in 1..{schedule.STEPS}")
    p, pol, v = augment.augment(planes.astype(np.float32), policy, z)
    x_all = torch.from_numpy(np.ascontiguousarray(p))
    pol_all = torch.from_numpy(np.ascontiguousarray(pol, dtype=np.float32))
    v_all = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    n = len(v)
    rng = np.random.Generator(np.random.PCG64(seeds.sample(base, generation)))
    indices = rng.integers(0, n, size=(steps, batch))
    opt = torch.optim.SGD(
        model.parameters(),
        lr=schedule.lr(schedule.global_step(generation, 0)),
        momentum=HYPER["momentum"],
        nesterov=HYPER["nesterov"],
        weight_decay=HYPER["weight_decay"],
    )
    losses, rates = [], []
    model.train()
    try:
        for s_ in range(steps):
            rate = schedule.lr(schedule.global_step(generation, s_))
            for group in opt.param_groups:
                group["lr"] = rate
            idx = torch.from_numpy(indices[s_])
            logits, values = model(x_all[idx])
            lp = -(pol_all[idx] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            lv = ((values - v_all[idx]) ** 2).mean()
            loss = lp + HYPER["value_loss_weight"] * lv
            if not bool(torch.isfinite(loss)):
                raise Diverged(f"generation {generation} step {s_}: the loss is not finite")
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values() if t.is_floating_point()):
                raise Diverged(f"generation {generation} step {s_}: a parameter is not finite")
            losses.append([lp.item(), lv.item()])
            rates.append(rate)
    finally:
        model.eval()
    return losses, {
        "augmented_samples": n,
        "steps": steps,
        "t_first": schedule.global_step(generation, 0),
        "t_last": schedule.global_step(generation, steps - 1),
        "lr_first_last": [rates[0], rates[-1]],
        "indices_sha256": artifacts.sha256(np.ascontiguousarray(indices).tobytes()),
    }


def work_window(args, clock) -> int:
    configure_torch()
    if os.path.abspath(args.out) == os.path.abspath(args.checkpoint):
        raise SystemExit("--out must not be the parent checkpoint")
    model, parent = ckpt.load(args.checkpoint)
    planes, policy, z, digests = load_samples(args.data)
    try:
        losses, counts = train_window(model, planes, policy, z, args.generation, args.base, args.steps, args.batch)
    except Diverged as e:
        print(f"diverged, no checkpoint written: {e}", file=sys.stderr)
        return EXIT_DIVERGED
    if counts["steps"] < 1 or len(losses) != counts["steps"]:
        raise SystemExit("no checkpoint without an update")
    config = {
        "parent_sha256": parent["sha256"],
        "data_manifests": digests,
        "generation": args.generation,
        "base": args.base,
        "steps": args.steps,
        "batch": args.batch,
        "schedule": {"k_max": schedule.K_MAX, "steps": schedule.STEPS, "lr_max": schedule.LR_MAX, "lr_min": schedule.LR_MIN},
        "hyper": {k: v for k, v in HYPER.items() if k != "lr"},
    }
    meta = {
        "kind": "p2-m3-checkpoint",
        "provenance": provenance.block(args.source_sha, config),
        **config,
        "sample_seed": seeds.sample(args.base, args.generation),
        "samples": int(len(z)),
        **counts,
        "loss_first_last": [losses[0], losses[-1]],
    }
    digest = ckpt.save(model, args.out, parent["size"], meta)
    artifacts.write_atomic(os.path.splitext(args.out)[0] + ".losses.json", json.dumps(losses).encode())
    print(f"checkpoint {digest} steps {counts['steps']}")
    return 0


def work_train(args, clock) -> int:
    configure_torch()
    if os.path.abspath(args.out) == os.path.abspath(args.checkpoint):
        raise SystemExit("--out must not be the parent checkpoint")
    if args.epochs < 1 or args.batch < 1:
        raise SystemExit("--epochs and --batch must be at least 1")
    model, parent = ckpt.load(args.checkpoint)
    planes, policy, z, digests = load_samples(args.data)
    if (2 * len(z)) // args.batch < 1:
        raise SystemExit(f"{2 * len(z)} augmented samples make no full batch of {args.batch}: no update")
    batch = args.batch
    try:
        losses, counts = train(model, planes, policy, z, args.run_seed, args.epochs, batch, clock)
    except Diverged as e:
        print(f"diverged, no checkpoint written: {e}", file=sys.stderr)
        return EXIT_DIVERGED
    if counts["steps"] < 1 or len(losses) != counts["steps"]:
        raise SystemExit(f"{counts['steps']} steps realized: no checkpoint without an update")
    config = {
        "parent_sha256": parent["sha256"],
        "data_manifests": digests,
        "hyper": {**HYPER, "batch": batch},
        "run_seed": args.run_seed,
        "epochs": args.epochs,
    }
    meta = {
        "provenance": provenance.block(args.source_sha, config),
        "kind": "p2-m3-checkpoint",
        "parent_sha256": parent["sha256"],
        "data_manifests": digests,
        "hyper": {**HYPER, "batch": batch},
        "seed_derivation": seeds.DERIVATION,
        "run_seed": args.run_seed,
        "samples": int(len(z)),
        **counts,
        "loss_first_last": [losses[0], losses[-1]] if losses else None,
    }
    digest = ckpt.save(model, args.out, parent["size"], meta)
    # The full loss trend, for the report (reported, not a criterion).
    artifacts.write_atomic(os.path.splitext(args.out)[0] + ".losses.json", json.dumps(losses).encode())
    print(f"checkpoint {digest} steps {counts['steps']}")
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--size", required=True, choices=list(netmod.SIZES))
    i.add_argument("--seed", type=int, required=True)
    i.add_argument("--out", required=True)
    i.add_argument("--source-sha", required=True)
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--run-seed", type=int, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=HYPER["batch"])
    t.add_argument("--source-sha", required=True)
    t.add_argument("--ledger", required=True)
    t.add_argument("--cores", type=int, default=1)
    t.add_argument("--budget-core-hours", type=float, required=True)
    t.add_argument("--wall-cap-s", type=float, required=True)
    w = sub.add_parser("train-window", help="PR 3: one generation of the loop")
    w.add_argument("--checkpoint", required=True)
    w.add_argument("--data", nargs="+", required=True, help="the window's runs, oldest generation first")
    w.add_argument("--out", required=True)
    w.add_argument("--generation", type=int, required=True)
    w.add_argument("--base", type=int, required=True)
    w.add_argument("--steps", type=int, default=schedule.STEPS)
    w.add_argument("--batch", type=int, default=HYPER["batch"])
    w.add_argument("--source-sha", required=True)
    w.add_argument("--attempt", default="train")
    w.add_argument("--ledger", required=True)
    w.add_argument("--cores", type=int, default=1)
    w.add_argument("--budget-core-hours", type=float, required=True)
    w.add_argument("--wall-cap-s", type=float, required=True)
    return p.parse_args(supervisor.strip_child_flag(argv))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    if args.command in ("train", "train-window") and supervisor.child_flag(argv) is None and os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists: checkpoints are never overwritten")
    if args.command == "init":
        configure_torch()
        model = netmod.build(args.size, seed=args.seed)
        config = {"size": args.size, "init_seed": args.seed}
        meta = {"kind": "p2-m3-checkpoint", "init_seed": args.seed, "provenance": provenance.block(args.source_sha, config)}
        digest = ckpt.save(model, args.out, args.size, meta)
        print(f"checkpoint {digest}")
        return 0
    work = work_window if args.command == "train-window" else work_train
    return supervisor.run(
        __file__,
        argv,
        lambda clock: work(args, clock),
        ledger=args.ledger,
        budget_core_hours=args.budget_core_hours,
        cores=args.cores,
        wall_cap_s=args.wall_cap_s,
        out=os.path.dirname(os.path.abspath(args.out)),
        name=getattr(args, "attempt", "train"),
        exit_names={EXIT_DIVERGED: "training diverged"},
        # The checkpoint directory also holds the parent; the checkpoint
        # itself is never overwritten (ckpt.save).
        fresh_out=False,
    )


if __name__ == "__main__":
    sys.exit(main())

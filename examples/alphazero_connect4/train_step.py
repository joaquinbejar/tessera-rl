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
            v = arrays["visits"].astype(np.float64)
            total = v.sum(axis=1, keepdims=True)
            if (total <= 0).any():
                raise artifacts.Incomplete(f"{d}/{entry['name']}: a sample without visits")
            # Illegal moves have zero visits: the mover's top cell is
            # filled (either plane) exactly where the visits must be 0.
            full = arrays["planes"][:, :, 0, :].sum(axis=1) > 0
            if (arrays["visits"][full] != 0).any():
                raise artifacts.Incomplete(f"{d}/{entry['name']}: visits on an illegal move")
            planes.append(arrays["planes"])
            policy.append(v / total)
            z.append(arrays["z"].astype(np.float64))
    return np.concatenate(planes), np.concatenate(policy), np.concatenate(z), digests


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
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append([float(lp), float(lv)])
    finally:
        model.eval()
    return losses, {
        "augmented_samples": n,
        "steps_per_epoch": steps_per_epoch,
        "dropped_per_epoch": n - steps_per_epoch * batch,
        "epochs": epochs,
        "steps": steps_per_epoch * epochs,
    }


def work_train(args, clock) -> int:
    configure_torch()
    model, parent = ckpt.load(args.checkpoint)
    planes, policy, z, digests = load_samples(args.data)
    batch = args.batch
    losses, counts = train(model, planes, policy, z, args.run_seed, args.epochs, batch, clock)
    meta = {
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
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--run-seed", type=int, required=True)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=HYPER["batch"])
    t.add_argument("--ledger", required=True)
    t.add_argument("--cores", type=int, default=1)
    t.add_argument("--budget-core-hours", type=float, required=True)
    t.add_argument("--wall-cap-s", type=float, required=True)
    return p.parse_args(supervisor.strip_child_flag(argv))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse(argv)
    if args.command == "init":
        configure_torch()
        model = netmod.build(args.size, seed=args.seed)
        digest = ckpt.save(model, args.out, args.size, {"kind": "p2-m3-checkpoint", "init_seed": args.seed})
        print(f"checkpoint {digest}")
        return 0
    return supervisor.run(
        __file__,
        argv,
        lambda clock: work_train(args, clock),
        ledger=args.ledger,
        budget_core_hours=args.budget_core_hours,
        cores=args.cores,
        wall_cap_s=args.wall_cap_s,
        out=os.path.dirname(os.path.abspath(args.out)),
        name="train",
    )


if __name__ == "__main__":
    sys.exit(main())

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# P2-M3 training loop (PR 3)

This file pre-registers the training loop before any run. The contract is design revision 2, accepted by Codex and TR, with their precision notes. This example is not a shipped trainer (ADR-0018). The tools are `loop.py`, `train_step.py train-window`, `look.py` and `assets.py`. Results go below the contract after the run.

## Trajectory

- **C_0:** the PR 2 C1, SHA-256 `89855dce4aa429d657af96c5effc42917cb18edbedb5f614e458832a6cb7ff7d`, checked by `loop.py init --c0-sha256`.
- **Generations:** `K_max = 30`, a fixed maximum. Generation `k`:
  1. plays 10,000 self-play games with `C_k` (run seed `seeds.generation(20261103, k)`; PR 2 settings: 100 simulations, noise alpha 1.0 and eps 0.25, temperature 1 for 8 plies);
  2. trains `C_{k+1}` from `C_k`'s weights with a fresh optimizer (momentum is reset by design);
  3. runs a held-out look at `C_{k+1}`.
- **Training:**
  - exactly `S = 1000` steps per generation;
  - each step a batch of 1024 drawn uniformly with replacement from the mirror-augmented window (the samples, then their mirrors), mirrored lazily per batch;
  - a formal run asserts exactly 1000 steps;
  - the window is the self-play data of generations `max(0, k-3)..k`, in generation, file and sample order;
  - the indices come from `PCG64(seeds.sample(20261103, k))`, drawn in one call.
- **Learning rate:** `lr(t) = 0.002 + 0.5 * 0.018 * (1 + cos(pi * t / 30000))` with `t = k * 1000 + s`. The horizon is fixed and never recomputed on resume or stop.

## Exactly-once recovery

- **Attempts:** every phase attempt `(k, phase, n)` has a fresh directory `gen-<k>/<phase>-<n>/` and its ledger run name `g<k>-<phase>-<n>`. Its command is recorded next to it.
- **Done:** an attempt is done when its output is complete and digest-verified, or evicted after a verified upload, **and** its ledger entry is `ok`. That settlement is rechecked on every resolution: a descriptor whose entry is failed or missing (a reset ledger included) is broken history and stops the loop with `incomplete input`, and so does a look reference whose origin is not settled.
- **Descriptor:** the loop then writes `attempt.json`, the attempt's immutable identity: output digests, inputs and command.
- **Recomputed from evidence:** every decision comes from the descriptors, the ledger and the pinned lineage:
  - `C_0` matches the configured digest;
  - self-play names `C_k`;
  - training names `C_k` and the window's self-play manifests;
  - the look names `C_{k+1}`.
- **No regeneration:** a recorded attempt whose output is gone or changed, or any broken link, stops the loop with `incomplete input`. The loop never regenerates a different history.
- **Audit log only:** `selection.jsonl` holds one look entry per generation plus every stop decision with its inputs, and `loop.json` is a cache. Selection and the early stop never trust either of them.
- **Look reuse:** a look at a checkpoint that already has a verified look is referenced (`look-ref`), not run again, and every generation keeps its own reference.
- **One owner:** one process at a time holds `loop.lock`, for `run` and for `select`.

## Budget and stops

The ledger has a STOP at 200 core-hours, over self-play, training, looks and retries. The gate of PR 4 has its own allowance.

Each phase has a hard reservation through the supervisor:

| Phase | Reservation |
| --- | --- |
| Self-play | 32 x 900 core-seconds |
| Training | 1,200 core-seconds |
| Look | 900 core-seconds |

That is exactly 30,900 core-seconds per generation.

A phase starts only if the remaining budget (settled spend plus open reservations) covers the reservations of its generation's remaining phases. A generation that cannot finish never yields an eligible checkpoint.

The stop reasons are `K_max`, `budget`, `early`, `failed` (3 attempts), `disk` (the root over 600 MB) and `incomplete input`. The projection is about 4.4 core-hours per generation, 132 for 30.

## Selection (held-out only)

- **Protocol** (frozen, identical to the PR 4 gate candidate):
  - single-tree search at 100 simulations, batch 8, capacity 8192;
  - greedy moves, no noise, search seed 20261102;
  - the ADR-0042 bot with tie seeds `TIE_BASE + j`;
  - the 200 held-out openings x 2 colours;
  - the registered runtime.
- **Looks start at C_1:** `C_0` is never selectable.
- **Eligible:** only a verified look with an ok ledger entry, on a **formal** run, and only with all of:
  - the registered runtime;
  - the registered bot binary for that image (`look.REGISTERED_BOTS`, filled by a reviewed pre-run commit after the build);
  - a medium net;
  - the frozen protocol.
- **Formal run:** `K_max` 30, 1000 steps, early stop at 380 wins, budget 200, window 4, 10,000 games, base 20261103, 100 sims, batch 8, capacity 8192, train batch 1024, 200 look openings, the registered `C_0`, and test settings off. The flag is in `config.json`, `loop.json` and the selection manifest, and PR 4 refuses a non-formal selection.
- **Rule:** the most held-out wins (draws are non-wins), ties going to the later generation. `loop.py select` writes `selection-manifest.json`. It names every eligible look, the selected checkpoint, the final (last trained) checkpoint and the stop decisions, and it is committed before PR 4 opens the gate set. With no eligible look there is no selection and no gate.
- **Early stop:** two consecutive generations with at least 380/400 held-out wins.

**Bias.** Up to 30 looks at one held-out set make the selected checkpoint's held-out score optimistically biased, and the early stop uses the same looks. That score is never reported as an estimate of gate performance; only PR 4's blind 1,000-game run is.

## ADR-0052 profile (P2-G1, 400-simulation self-play)

`loop.py init --profile adr-0052` registers a second formal profile beside
M3's, whose values above are unchanged. It follows the ADR-0052 text frozen
at sha256 `d3a091482e70f53caee6fe23b7b905c38a65d204022408727151d1deeaa1b0ee`.

- M3's recipe with self-play at 400 simulations, `K_max` 10, and the
  learning-rate horizon scaled to 10 generations (`horizon_k` in
  `config.json`, `--horizon-k 10` to `train_step.py`).
- `--budget-core-hours` (at most 250), `--selfplay-cores` and
  `--selfplay-wall-s` are required: they are the owner's figure and the
  reviewed design's reservation, never defaults. The budget is a hard stop
  through the usual admission against the ledger.
- Looks run only at generations 6, 8 and 10. The other generations train
  without a look; a look attempt at any other generation stops the loop
  with `incomplete input`, and the loop refuses to start one.
- Looks run at 100 simulations, batch 8, capacity 8192 (`look_sims`, 100
  also for a root without the key); self-play alone runs at 400.
- The first look on that frozen budget reaching 336 stops training (`early`);
  a look at any other budget neither stops nor closes the root; only that look is
  eligible for selection. Without one, the run stops `not eligible`: after
  generation 10 (`cause: K_max`) or on a budget stop at any point
  (`cause: budget`), with no extra look.
- `early` and `not eligible` close the root: a later `run` is refused.
- The gate is unchanged (336 of 400 at a look, 815 of 1,000 formal). A
  run's host and its bot binary are registered by a reviewed pre-run commit
  (`h2h.REGISTERED_HOSTS`, `h2h.REGISTERED_IMAGES`, `look.REGISTERED_BOTS`).

## Artifacts

`assets.py pack` makes one content-addressed tarball per done phase attempt and an `assets-manifest.json` with the tag, name, size, SHA-256, generation, phase, attempt, source SHA and config hash. The coordinator uploads the tarballs, after their ledger entries are settled, to the private draft release `p2-m3-train-<date>`.

`assets.py verify` checks every download and refuses substitutes. The manifest only grows; a changed entry is refused.

`assets.py evict` removes an attempt's output only after the read-back of its asset verifies. It keeps the descriptor, marked evicted with its asset entry. An evicted input that a later phase needs (the active window, the current checkpoint) stops the loop with `incomplete input` until `assets.py restore` brings it back from a verified asset. `pack`, `evict` and `restore` hold the loop's owner lock and refuse a broken history. Eviction writes its mark before deleting and restore clears it last, so an interrupted operation resumes by running it again.

## Results

Not run yet.

## Remote phases (remote.py)

Self-play needs 32 cores (swarmer02), and the looks need the registered runtime (xaskar, 1 core). The loop driver therefore runs on the Mac, owns the root and holds `loop.lock`. Each phase reaches its host through `remote.py run`, set as that phase's `runners` command at `init`, for example:

```
{python} remote.py run --host swarmer02 --cpuset 0-31 --root ROOT -- {tool} {args}
```

The phase runs as one detached, named container (`docker run -d --name tessera-m3-<attempt>`), with the pinned image and bind mounts by absolute path only. Its round trip has five steps:

1. A durable dispatch record next to the attempt. It holds the host, cpuset, container, image, prior-ledger digest and the input hashes.
2. The inputs are pushed to `~/tessera-rl-m3loop/root` and their hashes are checked on the host.
3. The run, then a wait.
4. The import. The host's ledger is accepted only if it keeps the ledger that was sent as an exact prefix and appends only this attempt's entry. Outputs are staged, verified and published atomically, then acknowledged.
5. Cleanup, only after the acknowledgement. The host keeps only the ledger and the cached image. The image is removed with `docker rmi` and the directory with `rm -rf` when the loop ends.

A dead driver loses no work and never doubles a charge. `remote.py reconcile --root ROOT` closes any open dispatch:
- It finds a still-running container by name and waits for it.
- It imports a successful attempt once.
- It settles an entry whose container is gone at the full reservation. It does so only after a by-name check on that host, recorded in the dispatch.
- It replays a missing acknowledgement.

The loop runs `remote.py reconcile --root ROOT --owner-held`, set with `init --reconcile`, before every step. Every open dispatch is therefore closed before the next attempt is allocated, and while a dispatch is open no new one is accepted.

Each step leaves a record in the dispatch:
- Publication is journaled with the staged output and ledger digests, then replayed against them. The output rename, the ledger replace and the acknowledgement are not jointly atomic, so a replay must reach the same result.
- Before replacing the local ledger, the import checks that it is still the one sent: a compare and replace. The import also checks that the budgets match.
- "Nothing ran" needs positive evidence: a by-name check, and the remote ledger read with no entry for the attempt.
- A container that outlives its wall cap plus 900 s of setup and a 300 s margin is killed by the driver, and the kill is recorded.

Fail-closed rules:
- Provisioning is explicit. `remote.py provision --host H --archive TAR --uv UV` deploys the reviewed source archive and uv once and prints the evidence: archive, uv and deployed-tree digests.
- Every dispatch checks the host's source tree against `--source-tree-sha256` before anything is pushed or reserved. It also pushes and checks the root's `config.json`.
- An ssh failure (exit 255), a docker daemon error or a malformed answer is never read as "gone". Only docker's own "No such object" counts as absent.
- An ok attempt is imported only if its staged output equals the remote one and its identity loads; the loop's lineage check runs before the acknowledgement. The payload is fsynced before the acknowledgement.
- The settled entry is checked against the dispatch's frozen cores and cap, against the status rules and against the charge arithmetic.
- The container carries its own deadline (`timeout` around uv setup and the phase, at the wall cap plus 900 s), so a dead driver cannot leave it running.
- A killed or interrupted attempt is marked `published_partial` and is never packed.

Container names are `tessera-m3-<config digest>-<attempt>`, unique per experiment. Run alone, `remote.py reconcile --root ROOT` takes the owner lock itself. The driver is:

```
caffeinate -dims python loop.py run --root ROOT
```

Disk: the PR(2) sizes give about 33 MB per generation, about 1.0 GB over 30, so the formal run sets `--max-disk-mb 1536` at `init`. That needs at least 5 GB free on the Mac. On each host, the current phase's inputs and outputs take at most about 130 MB. On top of that sit the cached source, its `.venv` and build (estimated at 1.5 to 2 GB, to be measured on the first phase) and the cached image. The Mac holds the root, up to 1.5 GB, plus one phase's staging (about 35 MB).

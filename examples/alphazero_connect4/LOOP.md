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
  - each step a batch of 1024 drawn uniformly with replacement from the mirror-augmented window;
  - the window is the self-play data of generations `max(0, k-3)..k`, in generation, file and sample order;
  - the indices come from `PCG64(seeds.sample(20261103, k))`, drawn in one call.
- **Learning rate:** `lr(t) = 0.002 + 0.5 * 0.018 * (1 + cos(pi * t / 30000))` with `t = k * 1000 + s`. The horizon is fixed and never recomputed on resume or stop.

## Exactly-once recovery

- **Attempts:** every phase attempt `(k, phase, n)` has a fresh directory `gen-<k>/<phase>-<n>/` and its ledger run name `g<k>-<phase>-<n>`.
- **Done:** a phase is done if and only if an attempt has a complete, digest-verified output **and** an `ok` ledger entry; the lowest such `n` counts.
- **The ledger** is authoritative for spend. `loop.json` is a cache.
- **The selection log** `selection.jsonl` is append-only and deduplicated by checkpoint SHA-256. It also records every stop decision with its inputs.
- **One owner:** one process at a time holds `loop.lock`.

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
- **Eligible:** only a complete look with an ok ledger entry, on the registered runtime and protocol.
- **Rule:** the most held-out wins (draws are non-wins), ties going to the later generation. `loop.py select` writes `selection-manifest.json`. It names every eligible look, the selected checkpoint, the final (last trained) checkpoint and the stop decisions, and it is committed before PR 4 opens the gate set. With no eligible look there is no selection and no gate.
- **Early stop:** two consecutive generations with at least 380/400 held-out wins.

**Bias.** Up to 30 looks at one held-out set make the selected checkpoint's held-out score optimistically biased, and the early stop uses the same looks. That score is never reported as an estimate of gate performance; only PR 4's blind 1,000-game run is.

## Artifacts

`assets.py pack` makes one content-addressed tarball per done phase attempt and an `assets-manifest.json` with the tag, name, size, SHA-256, generation, phase, attempt, source SHA and config hash. The coordinator uploads the tarballs, after their ledger entries are settled, to the private draft release `p2-m3-train-<date>`.

`assets.py verify` checks every download and refuses substitutes. A sole local copy is deleted only after the read-back verifies.

## Results

Not run yet.

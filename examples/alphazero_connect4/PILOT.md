<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# P2-M3 pilot: CPU throughput and budget

This file pre-registers the plan and the decision criteria of the pilot
**before its first run**. The results are added below the plan afterwards.
This example is not a shipped trainer (ADR-0018).

## What is measured

Each cost is measured separately. Every cell discards its warmup
repetitions (`--warmup`, default 2) and keeps a fixed number of samples
(`--repeats`, default 7). The report gives the median, p10 and p90 of
each cell.

- **I, inference:** the forward pass plus the masked softmax per batch
  `{1, 8, 64, 256}`, for nets `small` (4 blocks x 32 channels), `medium`
  (6 x 64) and `large` (10 x 128). Each runs at 1 thread and at the
  allocated core count.
- **S, self-play:** full games at simulations `{50, 100, 200}`, with
  leaf batches of 8 (`batch_size`) and `W` in `{1, 8, allocated}` worker
  processes. Each worker owns one `Search` (capacity 8192 nodes, bounded),
  one net and 1 torch thread. Two evaluators are used:
  - `uniform`, a canned Python evaluator. Its timing is the **end-to-end
    canned search**: the engine plus the Python boundary. It is not the
    pure engine cost, which only a Rust bench measures.
  - `net`: the network.

  The net section (tensor conversion, forward, masked softmax) is timed
  inside the evaluator. It is disjoint from the rest of the search call,
  so `search - net` is the engine, the boundary and the call overhead
  together. Worker startup and model replication are timed separately
  (`pool_startup_s`). The report also gives the measured leaf-batch
  size histogram: batches can be smaller than 8.
- **T, training:** an SGD step (policy cross-entropy plus value MSE) on
  CPU at 1 thread, with batches of 256 and 1024 samples. The samples are
  synthetic and augmented with the left-right mirror only (the one
  symmetry of Connect Four: column `a` to `6 - a`, value unchanged). GPU
  is not measured.
- **E, evaluation:** a frozen random-init checkpoint is serialized once
  and its SHA-256 recorded. The candidate (greedy, no root noise,
  temperature 0, fixed search seed, a fresh search per game) plays the
  ADR-0042 bot on the held-out openings and on the evaluation openings,
  both colours, with tie seeds `TIE_BASE + j`. This is **timing and
  repeatability evidence only**: it is not a P2-G1 result, and it is
  never used to select a model or a search budget.

## Repeatability

Each evaluation set is played twice from the same serialized checkpoint,
with fresh nets, fresh searches and a fresh bot process. The canonical
log is hashed: for every game, the opening, the colour, the result and
every move with the candidate's root visits, excluding all timing. Equal
hashes show repeatability on the same host and the same image only.
Torch CPU kernels are not bit-identical across ISAs or thread counts, so
no cross-machine claim is made. A mismatch fails the check and stops any
reproducibility claim.

The gate evaluation (PR 4) is pinned to xaskar and the image digest,
both recorded in the report. Torch runs with
`use_deterministic_algorithms(True)`, 1 intra-op and 1 inter-op thread,
and `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set
to 1 before torch is imported. Search seeds go through Tessera (Philox)
only. Net initialization is seeded explicitly and leaves the global
torch RNG untouched.

## Starvation criterion (pre-registered)

This decides whether a multi-tree evaluator scheduler is designed inside
M3 (its own PR between PR 1 and PR 2, with an ADR-0043 note reviewed by
the coordinator, Codex and TR) or left as a follow-up. A binding of the
current `SearchPool` alone would not help: it is a single-root tree.

The CPU counts as **starved** if either condition holds:

1. The per-sample inference cost of the `small` net at batch 8 is
   more than 2x its cost at batch 64, at 1 thread (medians).
2. Everything outside the net section is more than 30% of the self-play
   search wall time, for the `small` net at 100 simulations and `W = 1`.
   This counts the engine, the Python boundary and the call overhead, so
   it overstates the boundary share alone; the canned `uniform` cells
   are reported next to it.

If either cell is missing because of the budget, the verdict is
"incomplete", not a pass.

## Budget and stop rules

- One core-hour is one allocated core for one hour of wall time, whatever
  the cores did.
- swarmer: one `docker run --rm` with the image pinned by digest,
  `--cpuset-cpus` set to all cores (the owner's decision), and a 90 minute
  wall cap. That is at most `1.5 x N` core-hours for `N` allocated cores.
- xaskar: E, the repeatability check, and a short S cross-check, capped
  at 1 hour of wall time on its allocated cores.
- The harness keeps a cumulative ledger across runs, failed and
  interrupted runs included. A cell starts only if its projection (the
  previous repetition, or the declared cell allowance) fits both the
  remaining core-hours and the remaining wall time. Cells that do not fit
  are reported as missing, never extrapolated.
- The full M3 budget is proposed from the pilot as a p10 to p90 range,
  separately for self-play, training, held-out selection and evaluation.
  Crossing it is reported, never spent. A GPU fallback is an owner
  decision.

## Results

Not run yet.

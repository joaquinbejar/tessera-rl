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
  processes. Each worker owns one net, 1 torch thread and, per round, one
  `Search` with a bounded capacity of 8192 nodes. Two evaluators are used:
  - `uniform`, a canned Python evaluator. Its timing is the **end-to-end
    canned search**: the engine plus the Python boundary. It is not the
    pure engine cost, which only a Rust bench measures.
  - `net`: the network.

  **Startup** runs from pool creation until a barrier releases every
  worker, after each one has imported torch and built its net. It is
  reported separately, with each worker's initialization time. A cell
  then runs `--warmup` discarded rounds and `--repeats` kept rounds; in
  each round every worker plays `--games-per-worker` games. Rates come in
  two forms:
  - steady state, from the kept rounds, per round;
  - end to end, over the whole cell including startup and warmup.

  Per-core rates divide by the **allocated** cores, not by the worker
  count. Per-worker rates are reported next to them.

  The net section (tensor conversion, forward, masked softmax) is timed
  inside the evaluator. It is disjoint from the rest of the search call,
  so `search - net` is the engine, the boundary and the call overhead
  together. The report also gives the measured leaf-batch size histogram:
  batches can be smaller than 8.
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
log covers, for every game, the opening, the colour, the result and every
move with the candidate's root visits, excluding all timing. It is written
to its own file per pass and hashed. Equal hashes show repeatability on
the same host and the same image only. Torch CPU kernels are not
bit-identical across ISAs or thread counts, so no cross-machine claim is
made.

A mismatch between two completed passes **fails the run**: exit code 3,
status `failed: evaluation not repeatable`, and the index of the first
differing game in the report. An incomplete pair is reported as
`incomplete`, never as repeatable.

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
- The command is a supervisor. Before any work it reserves the run's
  whole allowance in the ledger, under a file lock: the deadline times the
  allocated cores. The deadline is the smaller of the wall cap and what is
  left of the core-hour budget. The supervisor runs the measurements in a
  child process group and, on its own monotonic clock, kills the whole
  group (pool workers and the bot included) 2 s before the end of the
  reservation, exit code 124; the 2 s are for the kill and cleanup. It
  then settles the ledger to the wall time actually used, never clamped:
  an overrun is charged in full and marked. The child's watchdog is only
  a second guard, since native code holding the GIL can starve it.
- One run at a time per ledger file: a reservation is refused while
  another entry is open. A dead supervisor's entry is settled
  automatically only on its own host (its container's hostname); an open
  entry of another host blocks until it is settled by hand with
  `pilot.py ledger-settle --ledger PATH --index N --interrupted`, at its
  full reservation.
- The child also kills its own group when the supervisor disappears, even
  while the child is still starting: the supervisor passes its pid, and the
  child compares it with its current parent before its heavy imports, again
  when it arms its watchdog, and on every watchdog poll. A
  supervisor that died without settling (SIGKILL) leaves a `running`
  entry, which the next run settles at its full reservation.
- Inside the deadline, a cell's first unit starts only if
  `--cell-allowance-s` (or `--eval-allowance-s`) is left, and each later
  unit only if the previous one's duration is left. A cell or section
  that does not run is reported as missing, never extrapolated.
- The full M3 budget is proposed from the pilot as a p10 to p90 range,
  separately for self-play, training, held-out selection and evaluation.
  Crossing it is reported, never spent. A GPU fallback is an owner
  decision.

## Results (2026-11-02)

Source `4e6efb8ceb23ac079dfabaabfcb64ce5e5d0db5a` (content-identical to main `775ffd9`, the squash of #48), image `rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137`, torch 2.15.0+cpu, Python 3.11.18. The reports and ledgers are in [`results/pilot-20261102/`](results/pilot-20261102/).

- **swarmer02:** AMD Ryzen 9 9955HX 16-Core Processor, 16 physical / 32 logical cores (SMT), governor `performance`, cpuset `0-31`: 32 allocated cores. Sections `infer`, `selfplay` and `train`.
- **xaskar:** cpuset `2`, 1 allocated core. Section `eval`.

### Starvation criterion

**Criterion 1 FIRED**, so the verdict is starved.

1. Per-sample inference of the `small` net at 1 thread (7 kept samples per cell): batch 8 median 66.192 us, batch 64 median 31.024 us, ratio **2.1336** (threshold 2.0): **FIRED**. The 2.1355 quoted earlier on the bus is the ratio of the rounded medians (66.2 / 31.0); 2.1336 is the exact one. Spread (p10 to p90): batch 8 66.135 to 66.631 us, batch 64 30.990 to 31.044 us. The ratio range is 66.135 / 31.044 = 2.1303 to 66.631 / 30.990 = 2.1501, which does not straddle 2.0. This envelope is descriptive, the spread of the kept samples; it is not a confidence interval.
2. Non-net share of the self-play search wall, `small` net, 100 simulations, `W = 1`: 3.2% (threshold 30%): not fired.

The two criteria were measured in separate runs (`infer` and `selfplay`), so each report alone shows the verdict as `incomplete`; together they are complete. The consequence was decided by consensus (coordinator, Codex, TR) and recorded in the team note `tessera-rl:m3-starvation-ruling`.

### I, inference

Median per-sample cost in microseconds (forward plus masked softmax).

| Net | Params | Threads | b1 | b8 | b64 | b256 |
| --- | --- | --- | --- | --- | --- | --- |
| small | 78,394 | 1 | 262.6 | 66.2 | 31.0 | 31.5 |
| small | 78,394 | 32 | 473.1 | 123.7 | 23.2 | 16.4 |
| medium | 448,794 | 1 | 753.9 | 198.5 | 139.7 | 153.7 |
| medium | 448,794 | 32 | 927.4 | 213.3 | 54.3 | 51.1 |
| large | 2,960,602 | 1 | 3575.6 | 974.6 | 869.7 | 905.1 |
| large | 2,960,602 | 32 | 2612.2 | 485.7 | 177.9 | 152.7 |

### S, self-play

Steady state, median over the kept rounds (2 warmup, 7 kept, 2 games per worker per round). Per-core rates divide by the 32 allocated cores; divide by 16 instead for physical cores (multiply the per-allocated-core value by 2).

| Evaluator | Sims | W | Search ms/move | Non-net share | Games/h | Games/h per allocated core | Startup s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 50 | 1 | 0.07 | 100.0% | 1,697,418 | 53,044 | 1.00 |
| uniform | 50 | 8 | 0.07 | 100.0% | 8,743,703 | 273,241 | 1.29 |
| uniform | 50 | 32 | 0.14 | 100.0% | 13,743,491 | 429,484 | 2.33 |
| uniform | 100 | 1 | 0.13 | 100.0% | 725,785 | 22,681 | 0.99 |
| uniform | 100 | 8 | 0.13 | 100.0% | 4,457,718 | 139,304 | 1.32 |
| uniform | 100 | 32 | 0.23 | 100.0% | 7,731,197 | 241,600 | 2.37 |
| uniform | 200 | 1 | 0.24 | 100.0% | 561,910 | 17,560 | 1.00 |
| uniform | 200 | 8 | 0.24 | 100.0% | 2,728,255 | 85,258 | 1.31 |
| uniform | 200 | 32 | 0.45 | 100.0% | 5,767,401 | 180,231 | 2.40 |
| small | 50 | 1 | 4.13 | 3.4% | 46,696 | 1,459 | 1.01 |
| small | 50 | 8 | 4.36 | 3.4% | 227,566 | 7,111 | 1.31 |
| small | 50 | 32 | 9.27 | 4.3% | 468,124 | 14,629 | 2.38 |
| small | 100 | 1 | 7.04 | 3.2% | 19,483 | 609 | 1.01 |
| small | 100 | 8 | 7.79 | 3.2% | 107,705 | 3,366 | 1.30 |
| small | 100 | 32 | 16.62 | 4.3% | 225,332 | 7,042 | 2.40 |
| small | 200 | 1 | 13.35 | 3.2% | 11,183 | 349 | 1.01 |
| small | 200 | 8 | 14.50 | 3.1% | 58,013 | 1,813 | 1.30 |
| small | 200 | 32 | 32.18 | 4.3% | 116,119 | 3,629 | 2.39 |
| medium | 50 | 1 | 10.65 | 1.3% | 14,990 | 468 | 1.01 |
| medium | 50 | 8 | 12.06 | 1.5% | 78,068 | 2,440 | 1.31 |
| medium | 50 | 32 | 25.60 | 2.3% | 173,402 | 5,419 | 2.45 |
| medium | 100 | 1 | 19.42 | 1.2% | 6,652 | 208 | 0.99 |
| medium | 100 | 8 | 21.91 | 1.4% | 38,769 | 1,212 | 1.31 |
| medium | 100 | 32 | 48.23 | 2.2% | 76,818 | 2,401 | 2.30 |
| medium | 200 | 1 | 37.34 | 1.2% | 5,317 | 166 | 1.01 |
| medium | 200 | 8 | 42.12 | 1.4% | 20,974 | 655 | 1.29 |
| medium | 200 | 32 | 94.81 | 2.1% | 40,301 | 1,259 | 2.30 |
| large | 50 | 1 | 51.11 | 0.3% | 3,622 | 113 | 1.02 |
| large | 50 | 8 | 60.73 | 0.6% | 18,473 | 577 | 1.30 |
| large | 50 | 32 | 131.00 | 0.7% | 35,625 | 1,113 | 2.33 |
| large | 100 | 1 | 93.15 | 0.3% | 1,392 | 44 | 1.01 |
| large | 100 | 8 | 109.98 | 0.6% | 7,270 | 227 | 1.29 |
| large | 100 | 32 | 264.45 | 0.7% | 14,728 | 460 | 2.38 |
| large | 200 | 1 | 175.95 | 0.3% | 681 | 21 | 1.05 |
| large | 200 | 8 | 214.96 | 0.5% | 3,693 | 115 | 1.29 |
| large | 200 | 32 | 511.48 | 0.6% | 7,254 | 227 | 2.39 |

The `uniform` rows are the end-to-end canned search (engine plus Python boundary), not the pure engine cost. The leaf-batch histograms are in the report.

### T, training

Median SGD step time on CPU, 1 thread, synthetic mirrored samples.

| Net | Batch 256 ms | Batch 1024 ms |
| --- | --- | --- |
| small | 31.3 | 143.7 |
| medium | 134.1 | 590.5 |
| large | 752.5 | 3418.3 |

GPU: not measured.

### E, evaluation

Frozen random-init `small` checkpoint `80998924a6423e7ed515d64e1be849a439dbb9d53b8555618534b1397e5ed5c6`, 100 simulations, bot binary `910c1075c59994eee9ac796ff01a31b399dc8372cd44095c2274d73afc034afc`. Timing and repeatability only; the results against the bot are those of an untrained net and are not a P2-G1 result.

| Set | Openings | Games per pass | Pass walls s | Canonical log SHA-256 (both passes) | Repeatable |
| --- | --- | --- | --- | --- | --- |
| heldout | 200 of 200 | 400 | 60.5, 36.6 | `e47a3491c76ab9115c3131e3f6d6555b8bf0e32093bd1eb2cc9737d1bbf6c90e` | True |
| eval | 500 of 500 | 1000 | 113.8, 113.0 | `b26b9be63f7e9c2b043690bf01fcfec91a768e40d6fd8ad2de26ba0090b585df` | True |

The held-out passes differ (60.5 s against 36.6 s). The slower one is the first pass of the run, which also pays the process's one-time costs, such as the first torch calls (every pass starts its own bot process, so that is not the difference); this is the likely cause, not a measured split. The evaluation set, played afterwards, shows no such gap (113.8 s against 113.0 s).

The bot takes 0.5% of the evaluation wall time.

### Spend

| Run | Host | Allocated cores | Wall s | Core-hours | Status |
| --- | --- | --- | --- | --- | --- |
| pilot infer 20261102T175108Z | swarmer02 | 32 | 203 | 1.81 | ok |
| pilot selfplay 20261102T175432Z | swarmer02 | 32 | 1141 | 10.14 | ok |
| pilot train 20261102T181334Z | swarmer02 | 32 | 47 | 0.42 | ok |
| pilot eval 20261102T175120Z | xaskar | 1 | 334 | 0.09 | ok |

Outside the ledger, the swarmer02 container also spent about a minute on the image pull and the build, on the same 32 cores. In all, the container ran about 24 minutes, under the 90 minute cap (48 core-hours).

### Projection for the full run (self-play only)

From the steady 32-worker self-play rates (p10 to p90), on swarmer02 at 32 allocated cores. **This is a self-play-only projection.** Training, checkpoint selection on the held-out set, and the gate evaluation are not included; the full-run budget must cover all of them.

| Net | Sims | Wall hours per 100k games | Core-hours per 100k games |
| --- | --- | --- | --- |
| small | 100 | 0.43 to 0.46 | 13.6 to 14.7 |
| small | 200 | 0.86 to 0.88 | 27.4 to 28.2 |
| medium | 100 | 1.19 to 1.34 | 38.1 to 43.0 |
| medium | 200 | 2.32 to 2.61 | 74.2 to 83.4 |
| large | 100 | 6.54 to 7.16 | 209.2 to 229.0 |
| large | 200 | 13.00 to 14.45 | 415.9 to 462.5 |

### Training cost per epoch-equivalent

One epoch-equivalent is one pass over the mirrored samples of 100k self-play games: 100k games times the median moves per game (100 simulations, `W = 32` cells above) times 2 for the mirror, in SGD steps of batch 1024 at the T table's median step time, on 1 core. These are 1-core core-hours: to be added to the self-play projection when sizing the 200 core-hour STOP budget over self-play, training, held-out selection and the gate.

| Net | Moves per game | Samples | Steps (batch 1024) | Step ms | Core-hours per epoch-equivalent |
| --- | --- | --- | --- | --- | --- |
| small | 27 | 5.40 M | 5,273 | 143.7 | 0.21 |
| medium | 26 | 5.20 M | 5,078 | 590.5 | 0.83 |

### Provenance, commands and cleanup

- Source: `4e6efb8ceb23ac079dfabaabfcb64ce5e5d0db5a` (git archive), content-identical to main `775ffd9` (the squash of #48).
- Image: `rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137`.
- Evaluation checkpoint (random init, `small`, seed 20261102): SHA-256 `80998924a6423e7ed515d64e1be849a439dbb9d53b8555618534b1397e5ed5c6`.
- Bot binary (`tessera-rl-connect4-ref`, built in the container): SHA-256 `910c1075c59994eee9ac796ff01a31b399dc8372cd44095c2274d73afc034afc`.
- Canonical evaluation logs, one file per set and pass, committed gzip-compressed as `xaskar-eval-<set>-pass<n>.log.json.gz`. The uncompressed files are equal across both passes: held-out `f9e34abd380b06f4f9145c2923845071008cf4fc4ef671948e550efaaec6d42c`, eval `e40c5dc8f358132ffaf3fe5fbc4e56d28511fa8edb193c0662a1a951126b3493` (file digests; the report's `log_sha256` is the digest of the canonical JSON).
- Raw reports, with every kept sample, and the ledgers: [`results/pilot-20261102/`](results/pilot-20261102/). The self-play report is gzip-compressed (`swarmer02-selfplay.json.gz`); the SHA-256 of the uncompressed file is `2c0880c5ea4fea3ea9e9a17229ff14b7d92d6a0d6b1edc398b7bdc6027d6eb72`.

In the commands, `D=sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137`.

swarmer02, from `~/tessera-rl-pilot` holding the source, a copy of the `uv` binary and `inner.sh`:

```sh
timeout 5400 docker run --rm --name tessera-m3-pilot --hostname swarmer02 --cpuset-cpus 0-31 \
  -e PILOT_IMAGE_DIGEST=rust:1.99.1@$D -e PILOT_CPUSET=0-31 \
  -v $PWD/src:/work/tessera -v $PWD/out:/out -v $PWD/uv:/usr/local/bin/uv:ro -v $PWD/inner.sh:/inner.sh:ro \
  -w /work/tessera -e UV_CACHE_DIR=/work/tessera/.uvcache -e UV_LINK_MODE=copy \
  rust:1.99.1@$D sh -c "sh /inner.sh; chown -R $(id -u):$(id -g) /work/tessera /out"
```

with `inner.sh`:

```sh
set -e
uv sync -q --locked --python 3.11 --group test --group torch
for s in infer:400 selfplay:2000 train:300; do
  name=${s%%:*}; cap=${s##*:}
  uv run -q --locked --python 3.11 --group test --group torch python examples/alphazero_connect4/pilot.py $name \
    --out /out/$name --ledger /out/ledger.json --cores 32 --budget-core-hours 24 --wall-cap-s $cap \
    --source-sha 4e6efb8ceb23ac079dfabaabfcb64ce5e5d0db5a || echo "section $name exit $?"
done
```

xaskar, from `~/tessera-rl-bench/m3`:

```sh
timeout 3700 docker run --rm --name tessera-m3-eval --hostname xaskar --cpuset-cpus=2 \
  -e PILOT_IMAGE_DIGEST=rust:1.99.1@$D -e PILOT_CPUSET=2 \
  -v $PWD/src:/work/tessera -v $PWD/out:/out -v $HOME/.local/bin/uv:/usr/local/bin/uv:ro \
  -w /work/tessera -e UV_CACHE_DIR=/work/tessera/.uvcache -e UV_LINK_MODE=copy rust:1.99.1@$D \
  sh -c "cargo build -q --release -p tessera-rl-connect4-ref && uv sync -q --locked --python 3.11 --group test --group torch && \
    uv run -q --locked --python 3.11 --group test --group torch python examples/alphazero_connect4/pilot.py eval \
    --out /out --ledger /out/ledger.json --cores 1 --budget-core-hours 1 --wall-cap-s 3300 \
    --source-sha 4e6efb8ceb23ac079dfabaabfcb64ce5e5d0db5a --bot target/release/tessera-rl-connect4-ref; \
    chown -R 1000:1000 /work/tessera /out"
```

Both runs exited 0; every section has status `ok` and no missing cells.

Cleanup, checked after copying the results out:

- swarmer02: `rm -rf ~/tessera-rl-pilot`; `docker rmi` of the image (it was not on the host before); `docker ps -a --filter name=tessera -q` returned no container, `docker image ls rust -q` returned no image, and the home directory held only what was there before.
- xaskar: `rm -rf ~/tessera-rl-bench/m3`; `docker ps -a --filter name=tessera-m3 -q` returned no container.

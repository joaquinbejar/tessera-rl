<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# P2-M3 self-play baseline (PR 2)

This file pre-registers the contract (design revision 3, accepted by Codex and TR) **before any training run**. The pre-registration record is commit `ab6bbf92de3b40507a649d3da7ffc800e7075b51` (the dev set was frozen in `02eb3e535bbcefe58100db34b6e6c829a9801843`). The results go below it afterwards. This example is not a shipped trainer (ADR-0018).

## Experiment

The experiment is one iteration:
1. a random-initialized `medium` checkpoint C0;
2. 10,000 self-play games with C0 (`selfplay.py`);
3. one epoch-equivalent of training, giving C1 (`train_step.py`);
4. C1 against C0 on the dev openings (`h2h.py`).

**Why medium.** The pilot measured it at 38 to 43 core-hours of self-play per 100k games and 0.83 core-hours of training per epoch-equivalent. Its 1-thread b8/b64 ratio is 1.42, so it is not starved. The small net fired the starvation criterion (ADR-0045), and the large one is over budget.

**Self-play.**
- Layout: `W` = allocated cores, independent workers, each with one single-tree `Search` and one torch thread, and the checkpoint frozen by SHA-256 in eval mode.
- Search: 100 simulations, batch 8, root Dirichlet noise with alpha 1.0 and eps 0.25 (self-play only).
- Moves: temperature 1 for plies 0 to 7, then greedy.
- Each game starts from a fresh environment and a fresh `Search`.

**Seeds** (`seeds.py`). Each seed is the first 8 bytes, little-endian, of the SHA-256 of a canonical tuple. The domains are `env(g)`, `search(g)` and `shuffle(epoch)`. Seeds are independent of `W`, the worker and the shard. A seed that falls in the bot tie range or any of the evaluation, dev and held-out ranges (each `+ [0, 1000)`) is an error. The game's root-noise and temperature draws come from the search stream of its root record (ADR-0043), which `env(g)` determines.

**Targets.**
- `z` is mover-relative: the winner's moves get +1, the loser's -1, and a draw 0.
- The policy is root visits over their total, which is zero on illegal moves; this is checked when loading.

**Training** (`train_step.py`, frozen):
- SGD with lr 0.02, momentum 0.9, no Nesterov, weight decay 1e-4;
- loss = cross-entropy + 1.0 x MSE;
- batch 1024 over the left-right mirror-augmented samples;
- floor(samples / 1024) steps per epoch-equivalent, the last partial batch dropped and counted;
- the shuffle seeded by `shuffle(epoch)`;
- BatchNorm in train mode in the step and in eval mode elsewhere.

With 10k games of about 26 plies, an epoch-equivalent is about 500 steps.

**Artifacts.**
- Shards and manifests have canonical bytes and are written atomically, the manifest last.
- A run without a complete manifest, or with a changed file, is refused.
- Checkpoints carry their SHA-256, parent, data manifests, steps and hyperparameters.

## Criterion (pre-registered)

- **Games:** C1 against C0 on the 200 frozen dev openings (`compat/connect4/openings-dev.txt`, sha256 `6208bf75...ebd2`, see `DEV-PROVENANCE.md`), both colours: 400 games.
- **Settings:** single-tree `Search`, 100 simulations, greedy, no noise, a fixed search seed. Run on xaskar with the pinned image. Both checkpoints are `medium`.
- **Score:** `p = (W + 0.5 D) / 400`.
- **PASS** if and only if the one-sided Wilson lower bound of `p` at `z = 1.645` is above 0.5, which needs about `p >= 0.541`.

This is a **nominal, descriptive** threshold, not calibrated 95% evidence: draws are fractional and the two games of an opening are paired.

The criterion applies only on the registered runtime: xaskar, the pinned image, one allocated core and one torch thread. The head-to-head checks the observed affinity and thread counts, but the host name and the image digest are self-declared by the container, so `registered: true` guards against accidents and is not an attestation.

The report gives W/D/L, the colour splits, the per-opening pair scores, the distinct move sequences and the log hash. The C1 loss trend is reported too, but it is not a criterion.

A failure is reported as a failure; nothing is rerun until it passes, and the schedule is not changed afterwards.

The held-out openings stay untouched until PR 3's checkpoint selection, which logs every look. The ADR-0042 gate is unchanged: z = 1.96, draws count as non-wins, and the 1,000 gate games stay blind.

## Budget

There is one ledger for the whole experiment, with a 12 core-hour ceiling across every phase and retry:
- self-play on swarmer02: at most 10.5 core-hours (32 cores);
- training: at most 0.5 core-hours on 1 core;
- the head-to-head on xaskar: at most 1.0 core-hours on 1 core.

Every phase runs under the supervisor, and the run is announced on #tessera-rl first. No GPU.

## Results (2026-11-02)

**PASS.** This is the nominal, descriptive criterion above; the result is reported as measured, with no rerun.

All three tools ran from a git archive of `ba8bb5245dfbef38e5b80321a186e94d90566c8f` (PR #51 before its rebase). After the rebase onto main `8ca4792`, its counterpart `d474fcc` is content-identical to it for `examples/alphazero_connect4`, `compat` and `tools/connect4-ref`, and under `tests` it differs only by `tests/python/test_search_group.py`, which #50 added, with image `rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137`. The artifacts are in [`results/baseline-20261102/`](results/baseline-20261102/).

### Head-to-head (the criterion)

C1 played C0 on the 200 dev openings, both colours, on xaskar. The run is formal: `criterion_applicable: true`, and the runtime matched the registered one (host `xaskar`, cpuset `2`, affinity 1 core, torch threads 1 and 1, deterministic algorithms).

| Quantity | Value |
| --- | --- |
| Games | 400 |
| C1 W / D / L | 349 / 16 / 35 |
| C1 moving first (W / D / L) | 167 / 13 / 20 |
| C0 moving first (C1's W / D / L) | 182 / 3 / 15 |
| Score p = (W + 0.5 D) / 400 | 0.8925 |
| One-sided Wilson lower bound, z = 1.645 | 0.8643 (threshold 0.5) |
| Pair scores (2 / 1.5 / 1 / 0.5 / 0) | 152 / 14 / 31 / 2 / 1 |
| Distinct move sequences | 399 |
| Log SHA-256 | `f93869fd53cae95822df12ca1cdee02e7b9283914ecfe15585220cf3408457ee` |

### Pipeline

- **C0:** `medium`, init seed 20261102, SHA-256 `9219438ab348905cd31ac6af1eb4a8ada053e9071056f85cc049cf6ab469d9ca`.
- **Self-play** on swarmer02, 32 workers, run seed 1:
  - 10000 games and 247688 samples;
  - 7 to 42 plies per game, 24.77 on average;
  - 9953 distinct 8-ply prefixes;
  - manifest SHA-256 `ba643a179924c124892f5a0fbb8ef0bbf7d2e1106be4c9caccd25d6d129b0d86`, games digest `90c765ea29177348de475fb6eeb8af59dd28115ef833a31fc7ae8a68f2a52e07`, and 320 shards.
- **Training:** one epoch-equivalent of 483 steps, over 495376 augmented samples with 784 dropped. C1 SHA-256 `89855dce4aa429d657af96c5effc42917cb18edbedb5f614e458832a6cb7ff7d`, parent `9219438ab348905cd31ac6af1eb4a8ada053e9071056f85cc049cf6ab469d9ca`.
- **Loss** (policy, value), first step to last, reported only and not a criterion: (2.010, 0.931) to (1.791, 0.804). The per-step trend is in `C1.losses.json.gz`.

### Spend

One ledger for the whole experiment, against a 12 core-hour ceiling.

| Run | Host | Allocated cores | Wall s | Core-hours | Status |
| --- | --- | --- | --- | --- | --- |
| selfplay 20261102T200803Z | swarmer02 | 32 | 461.9 | 4.106 | ok |
| train 20261102T201552Z | swarmer02 | 1 | 281.0 | 0.078 | ok |
| h2h 20261102T202136Z | xaskar | 1 | 329.4 | 0.091 | ok |

Total: 4.275 core-hours.

### Artifacts and cleanup

- **Committed** (gzipped where large):
  - both checkpoints with their manifests;
  - C1's losses;
  - the self-play manifest, which holds every shard's SHA-256 and every game's digest;
  - the head-to-head report and its canonical log;
  - the ledger and the three run logs.
- **Shards:** the 320 shards (30 MB raw) are committed as `selfplay-shards.tar.gz`, SHA-256 `49654ae461fb651a6ceeede87950f87e7c82409309f0c412ca50ed6e7b6b88da`. Each shard can also be checked against the manifest's per-shard digests.
- **Cleanup:**
  - swarmer02: the run directory and the image were removed, and no container was left.
  - xaskar: the run directory was removed, and no container was left.

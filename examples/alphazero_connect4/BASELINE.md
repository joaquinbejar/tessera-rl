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

The report gives W/D/L, the colour splits, the per-opening pair scores, the distinct move sequences and the log hash. The C1 loss trend is reported too, but it is not a criterion.

A failure is reported as a failure; nothing is rerun until it passes, and the schedule is not changed afterwards.

The held-out openings stay untouched until PR 3's checkpoint selection, which logs every look. The ADR-0042 gate is unchanged: z = 1.96, draws count as non-wins, and the 1,000 gate games stay blind.

## Budget

There is one ledger for the whole experiment, with a 12 core-hour ceiling across every phase and retry:
- self-play on swarmer02: at most 10.5 core-hours (32 cores);
- training: at most 0.5 core-hours on 1 core;
- the head-to-head on xaskar: at most 1.0 core-hours on 1 core.

Every phase runs under the supervisor, and the run is announced on #tessera-rl first. No GPU.

## Results

Not run yet.

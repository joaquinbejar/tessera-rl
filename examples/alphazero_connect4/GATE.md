<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# The P2-G1 gate (P2-M3 PR 4)

`gate.py` runs the phase 2 exit gate of SPEC 22 under ADR-0042, once, on the checkpoint that the loop's committed selection names (see [LOOP.md](LOOP.md)).

## Contract

- **Games:** the 500 pinned evaluation openings (`compat/connect4/openings-eval.txt`, sha256 `54a1cea7…0475`), each played twice, with the candidate moving first and then second. That makes 1,000 games against the ADR-0042 bot.
- **Search:** the look's protocol, through the same loop (`botmatch.py`):
  - 100 simulations, batch 8, capacity 8192;
  - greedy, no noise;
  - search seed 20261102;
  - tie seeds `TIE_BASE + j`.
- **Pass:** both rules must hold:
  - at least 900 wins out of 1,000;
  - the lower bound of the two-sided Wilson interval at z = 1.96 is at least 0.90, which takes 919 wins.

  Draws are non-wins. The interval is nominal: the games come in pairs and may repeat, so it is a stricter descriptive criterion, not calibrated coverage.
- **Report:** `gate.json` and `gate-log.json` hold:
  - W/D/L, overall and per colour;
  - per-pair results, distinct games and openings covered;
  - the rule and its numbers;
  - the canonical log and its digest;
  - the runtime;
  - provenance: source, config hash, checkpoint, selection manifest and bot digests, and the command.

  The exit code and the spend are in the gate's ledger entry.

## Refusals before anything is played

The formal mode reads the evaluation openings only after every check below passes. It refuses when:
- the selection manifest's digest is not the committed one;
- the manifest is not formal, or names no checkpoint;
- a recomputation from the loop root's verified, settled evidence (under the owner lock) disagrees with it;
- the checkpoint is not the selected one, or is not a medium net;
- the bot binary is not the expected one, or not the registered one for the image;
- the runtime is not the registered one;
- the search settings are not the frozen ones;
- the ledger already holds a gate entry, whatever its status.

## Blindness

- **Who reads the set:** the evaluation set is played only by this tool. The P2-M3 pilot timed a randomly initialized net on it before any training, and never selected or tuned with it.
- **Diagnostic mode:** `--diagnostic-openings` is for tests and dry runs. It refuses the evaluation file, whether given by path or as a copy, before reading it. It never names a selection and has no verdict.
- **One shot:** a gate runs once. A failed or interrupted gate is reported as it is, and a retry needs the owner's decision.

This score is the only gate estimate. The selected checkpoint's held-out score is biased by repeated looks and is never reported as one.

## Run

Not authorized yet (note `tessera-rl:m3-compute-authorization`). The plan is one supervised run on xaskar's registered runtime: 1 core, its own ledger allowance, and a wall cap near 2,400 s, which is about 2.5 times a look. The plan goes to the coordinator and the owner once the formal selection exists.

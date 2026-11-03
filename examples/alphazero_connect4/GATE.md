<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# The P2-G1 gate (P2-M3 PR 4)

`gate.py` runs the phase 2 exit gate of SPEC 22 under ADR-0042, once, on the checkpoint that the loop's committed selection names (see [LOOP.md](LOOP.md)). Only `selected.checkpoint` is ever gated, never the final (last trained) checkpoint.

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

  Draws are non-wins. A complete, valid, formal gate records `pass` as true or false; null is only for an incomplete, inapplicable or diagnostic run. The interval is nominal: the games come in pairs and may repeat, so it is a stricter descriptive criterion, not calibrated coverage.
- **Report:** `gate.json` and `gate-log.json` hold:
  - W/D/L, overall and per colour;
  - per-pair results, distinct games and openings covered;
  - the rule and its numbers;
  - the canonical log and its digest;
  - the runtime;
  - provenance: source, config hash, checkpoint, selection manifest and bot digests, and the command.

  The exit code and the spend are in the gate's ledger entry.

## The formal flow

All of it runs under the loop's owner lock. The gate's paths are fixed under the root: `<root>/gate/` holds `claim.json`, `ledger.json` and `run/`.

1. **Preflight.** Nothing of the evaluation set is touched yet. It refuses when:
   - the selection manifest's digest is not the committed one;
   - the manifest is not formal, or names no checkpoint;
   - a recomputation from the loop root's verified, settled evidence disagrees with it;
   - the selected checkpoint is evicted, is not the selected one, or is not a medium net;
   - the bot binary is not the expected one, or not the registered one for the image;
   - the runtime is not the registered one;
   - the search settings are not the frozen ones;
   - the resources are not one core, a finite positive budget that funds the whole wall cap, and a cap above the cleanup grace.
2. **Claim.** An atomic `mkdir <root>/gate` and a durable `claim.json`, plus a `gate-claim` entry in the loop's audit log (`selection.jsonl`). A second gate is refused, whatever paths it names. The selection is never rewritten.
3. **Run.** The supervisor reserves the budget. The work then revalidates the claim at use (checkpoint, bot, runtime), and only then hashes, compares and parses the evaluation file. The file's digest is recomputed after the games.

4. **Finalize.** The child's report is only provisional (`gate.provisional.json`). The parent writes `gate.json` with a verdict only after exit 0 and an ok ledger settlement. A failed or interrupted attempt stays consumed, with `pass` null and the reason, and a retry needs the owner's decision. If the parent dies after the child settles but before `gate.json` is written, `gate.py finalize --root ROOT` rebuilds it, read-only and idempotently, from the provisional report and the ledger entry under the same rules.

The claim syncs the root directory after the `mkdir`, then the claim file and the gate directory, then the audit log, all before the evaluation set is read. `distinct_games` counts distinct complete move sequences (the opening's columns, then the played ones).

## Blindness

- **Who reads the set:** within the example, only this tool plays the evaluation set. The P2-M3 pilot (`pilot.py`, #48, results in #49) timed an untrained random init on it before any training, and never selected or tuned with it. It stays as it is, allowlisted in the blindness test, because changing it would break #49's reproducibility. The pin test only hashes the file.
- **Diagnostic mode:** `--any-openings --openings FIXTURE` is for tests and dry runs. It has no default fixture. It refuses the evaluation file and its aliases (symlinks, hard links, by device and inode) before opening it, and a byte-identical copy by digest. It never names a selection and has no verdict.

This score is the only gate estimate. The selected checkpoint's held-out score is biased by repeated looks and is never reported as one.

## Run

Not authorized yet (note `tessera-rl:m3-compute-authorization`). The plan is one supervised run on xaskar's registered runtime: 1 core, its own ledger allowance, and a wall cap near 2,400 s, which is about 2.5 times a look. The plan goes to the coordinator and the owner once the formal selection exists.

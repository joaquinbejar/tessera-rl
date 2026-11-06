<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# The P2-G1 gate (P2-M3 PR 4)

`gate.py` runs the phase 2 exit gate of SPEC 22 under ADR-0042's protocol and ADR-0048's rule, once, on the checkpoint that the loop's committed selection names (see [LOOP.md](LOOP.md)). Only `selected.checkpoint` is ever gated, never the final (last trained) checkpoint.

## Contract

- **Games:** the 500 pinned evaluation openings (`compat/connect4/openings-eval.txt`, sha256 `54a1cea7…0475`), each played twice, with the candidate moving first and then second. That makes 1,000 games against the ADR-0042 bot.
- **Search:** the look's protocol, through the same loop (`botmatch.py`):
  - 100 simulations, batch 8, capacity 8192;
  - greedy, no noise;
  - search seed 20261102;
  - tie seeds `TIE_BASE + j`.
- **Pass** (ADR-0048, `rule_id` `adr-0048-wins-815`): at least 815 wins of 1,000, the frozen solver's published score. Draws are non-wins. No interval decides it, and a pass makes no statistical claim.

  A complete, valid, formal attempt records `pass` as true or false. A development run has `pass` null at both levels, with `meets_815` and a reduced `claim`, as ADR-0048 leaves closing the gate on it to the owner. A short or inapplicable result has both null, and a diagnostic run has no verdict. The nominal Wilson bound is in `descriptive`: the games come in pairs and may repeat, and it does not decide the gate. A report with no `rule_id` is under ADR-0042, and keeps that meaning.
- **Eligibility:** the selected look won at least 336 of 400 held-out games, from its verified report, carried into the claim and reloaded at use, with its digest, count and frozen protocol.
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
   - the selected look is under 336 of 400;
   - for a formal attempt, no solver binary is pinned, as the report needs the comparison;
   - the resources are not one core, a finite positive budget that funds the whole wall cap, and a cap above the cleanup grace.
2. **Claim.** The journal's admission, then an atomic `mkdir <root>/gate` (`gate-dev-<first 16 of the digest>` for a development run) and a durable `claim.json`, plus a `gate-claim` entry in the loop's audit log (`selection.jsonl`). A second gate is refused, whatever paths it names. The selection is never rewritten.
3. **Run.** The supervisor reserves the budget. The work then revalidates the claim at use (checkpoint, bot, runtime), and only then hashes, compares and parses the evaluation file. The file's digest is recomputed after the games.

4. **Finalize.** The child's report is only provisional (`gate.provisional.json`). The parent writes `gate.json` with a verdict only after exit 0 and an ok ledger settlement. A failed or interrupted attempt stays consumed, with `pass` null and the reason, and a retry needs the owner's decision. If the parent dies before `gate.json` is written, `gate.py finalize --root ROOT` rebuilds it from the provisional report and the ledger entry under the same rules, and settles the journal.

The claim syncs the root directory after the `mkdir`, then the claim file and the gate directory, then the audit log, all before the evaluation set is read. `distinct_games` counts distinct complete move sequences (the opening's columns, then the played ones).

## The attempts journal

`attempts.py` records attempts across loop roots, as ADR-0048's limit crosses them. The standing path, designated by the owner when a formal run is proposed: `<ATTEMPTS-JOURNAL-PATH: not designated yet>`.

- **Format:** an append-only JSONL file with a header naming the evaluation digest and rule family, each line holding the previous line's digest. A claim and one terminal `settled` event. Malformed, truncated or unknown content fails closed. A missing file needs `--create-journal`, recorded in the attempt's report.
- **Admission:** one lock across validating, checking and appending. A formal attempt is refused after any claim. A development run is refused until a formal attempt has settled, while any attempt lacks a settlement, and for a checkpoint with any entry.
- **States:** active (claimed, owner lock held), interrupted (claimed, no lock, no settlement), settled.
- **Recovery:** `gate.py finalize` runs nothing and settles an interrupted attempt only once the run has certainly stopped: the recorded process group answering ESRCH, whatever the ledger says. With no record no child was started. Otherwise it stays interrupted and development stays refused. It never creates a formal claim.
- **A crash after the claim and before the first evaluation read still spends the attempt.** Nothing could prove the file stayed unread. Anything else is the owner's decision.
- **No claim file:** after a crash or a failed `mkdir` following the journal entry, `gate.py finalize --root ROOT --attempts-journal JOURNAL` settles the root's entry as `consumed-no-verdict`, under the same quiescence rule.
- **Dispositions:** `met-815`, `below-815` and `consumed-no-verdict`. They state the score: a development run that meets 815 closes nothing.
- **Development is narrower than the ADR:** a run evaluates only the checkpoint its root's selection names, while ADR-0048 allows any checkpoint. Another one needs a root whose selection names it.
- **Hard links** to the journal are refused, as they would split its lock.
- **Limit:** a record against mistakes, not a security boundary or a complete enforcement of the global rule: another file escapes it. So the owner or coordinator also records each attempt in a shared note.

## The registered bot

A formal run uses the existing binary registered in `look.REGISTERED_BOTS`, digest `ef274131…f857`, built from source 46462481 in the registered image, kept read-only and not rebuilt. Its source archive `2dabd40a…74e74b` is not located.

## The solver comparison

ADR-0048's report needs the discordant slots against the solver, and the repo holds only its totals: a committed per-slot file would put evaluation data in the repo before the blind attempt. `gate.py compare` runs `solver_slots` on the evaluation openings after a valid settled attempt, supervised and bounded, and writes `run/comparison.json`.

- **Refused** until `SOLVER_SLOTS_SHA256` is pinned from an approved build in the registered image, so a formal attempt cannot start before then. It is also refused on any invalid input: an invalid attempt, a log not matching its digest or counts, unexpected slots, or sources other than the pinned ones. Running it needs its own authorization.
- **Publication:** the child writes a provisional comparison. The parent, holding the root's owner lock, publishes `run/comparison.json` only after exit 0 and an ok ledger entry. A rerun with a published file checks that entry, and the inputs include the registered build environment. The attempt needs an evaluated settlement in the journal.
- **Outcomes:** `complete`; `invalid`, for unexpected rows or a solver total other than 815, with no McNemar value and the bar still 815; `failed`, in `compare/outcome.json`, publishing nothing. A rerun with the same inputs changes nothing, and other inputs are refused.
- **It never touches `gate.json`.** A failed or missing comparison leaves the score and verdict intact, but ADR-0048's report is delivered only with a complete comparison.

## Before the first real use

The tests run `compare` through a real child from a copied script. The formal `main` has not run through one. Before any formal run, rehearse both through a real child, on synthetic openings with the pin set by a scratch copy of the tool, and report the result.

## Blindness

- **Who reads the set:** within the example, only this tool plays the evaluation set, and only the solver replays it after an attempt, playing no candidate. The P2-M3 pilot (`pilot.py`, #48, results in #49) timed an untrained random init on it before any training, and never selected or tuned with it. It stays as it is, allowlisted in the blindness test, because changing it would break #49's reproducibility. The pin test only hashes the file.
- **Diagnostic mode:** `--any-openings --openings FIXTURE` is for tests and dry runs. It has no default fixture. It refuses the evaluation file and its aliases (symlinks, hard links, by device and inode) before opening it, and a byte-identical copy by digest. It never names a selection and has no verdict.

This score is the only gate estimate. The selected checkpoint's held-out score is biased by repeated looks and is never reported as one.

## Run

Not authorized: ADR-0048 section 3 needs the owner's approval for any evaluation run. The plan is one supervised run on xaskar's registered runtime: 1 core, its own ledger allowance, and a wall cap near 2,400 s, which is about 2.5 times a look. The plan goes to the coordinator and the owner once the formal selection exists.

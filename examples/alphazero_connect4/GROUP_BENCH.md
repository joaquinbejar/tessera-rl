<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# The ADR-0045 section 5 SearchGroup measurement

`group_bench.py` measures SearchGroup against independent single trees for self-play. Its module docstring holds the frozen settings, the cells and the report fields.

**Protocol.** ADR-0045 section 5, with the local addendum `docs/adr/0045-addendum-1-measurement-quota.md` (ADR-0045 addendum 1), agreed by the coordinator, Codex and TR on 2026-11-03 before any data. The addendum gives every cell an equal quota of 32 logical games per worker per round. At the same W, every cell plays the same game ids and seeds. It leaves the ADR's decision text unchanged.

**Criterion.** The criterion is unchanged: the median end-to-end games per hour per allocated core of the group at T = 8 and W = allocated must be at least 1.2x the single-tree median at W = allocated. The verdict is formal only under the conditions of `applicability()`; otherwise `adopted` is null.

**Companion evidence, never a criterion:**
- per-worker busy wall seconds, with CPU seconds reported separately;
- games per busy wall second, from the kept rounds only (startup and warmup excluded);
- the fixed-root matched-work comparison;
- moves, simulations and leaves per hour.

W = 1 cells keep the allocated-core denominator and record `unused_allocated_cores`.

**Provenance.** The report records:
- the layout of every round: worker, batch, slot, logical game, its length and seeds, and each group's active schedule;
- the runtime observations: host name, boot id, CPU model and MemTotal.

The host name is self-declared provenance, not an attestation.

No run is authorized by this document; the measurement needs the owner's explicit approval.

<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

# Classic test suites

| File | Covers |
| --- | --- |
| `cartpole.rs`, `tag.rs` | Spaces, dynamics, rules, masks, observations, baselines |
| `cartpole_reference.rs` | One-step and short-trajectory reference vectors (SPEC 18) |
| `golden.rs` | Golden fixtures (ADR-0009, ADR-0024 layer 2), Inline and 1, 2, 7, 64 workers (SPEC 8.5) |
| `continuation.rs` | Continuation (SPEC 9.1): same slot, other slot, other pool by `restore` and `import`; `fork`, `fork_branched` |
| `reseed.rs` | Reseeding (SPEC 8.5) |
| `edge.rs` | Classic edge fixtures |
| `support/` | Action generator, digests, fixture format and blessing |

## Golden fixtures

The fixture format is documented in `support/mod.rs`.

Fixtures are written only with `TESSERA_BLESS=1 cargo test -p tessera-rl-classic --test golden`. Blessing follows a policy (ADR-0008, ADR-0023):

- If the output did not change, the file is left untouched.
- A new fixture is written.
- A changed fixture needs an `env_version` bump. A `MATH_VERSION` bump alone does **not** authorize it: new math is a new environment version that records it, and the old version stays reproducible.
- A harness migration needs an explicit reason: `TESSERA_BLESS_REASON=format` with a changed format version, or `TESSERA_BLESS_REASON=generator` with a changed action generator.
- A refused change leaves the fixture intact.

A plain run is read-only. It fails on any difference, reporting the first diverging call and field and the instances that end with a different record. The negative tests of the policy use in-memory text and a temporary file, never `tests/golden/`.

## ADR-0024 layer 3: edge fixtures

| Item | Test |
| --- | --- |
| Signed zeros, subnormals, enums without a zero discriminant | engine `snapshot.rs::test_edge_fixtures_survive_snapshot_and_encoding`, `masks.rs::test_negative_zero_reward_survives_without_illegal_actions` |
| Agent death at t, respawn of the same slot at t+1, snapshots on both sides | engine `snapshot.rs::test_continuation_across_death_and_respawn`; Tag death with snapshots at t-1 and t in `continuation.rs` |
| Same-step slot reuse rejected by an invariant check | engine `inline.rs::test_invalid_agent_events_are_ignored_and_counted` (ignored and counted), `inline.rs::test_invalid_spawn_fails_and_poisons_under_debug_checks` (`debug-checks`) |
| Death reported for an agent that is already dead (an invalid agent event; first found by the `record` fuzz target) | engine `inline.rs::test_death_of_a_dead_agent_is_ignored_and_counted`, `inline.rs::test_death_of_a_dead_agent_fails_and_poisons_under_debug_checks` |
| Ignored actions (dead, pending reset, empty slots) | engine `inline.rs::test_agent_death_clears_alive_and_skips_its_validation`, `inline.rs::test_pool_initial_agents_starts_the_rest_empty_until_spawned`; `edge.rs::test_edge_tag_actions_of_empty_and_caught_slots_are_ignored` |
| Illegal actions | engine `masks.rs` (`noop`, `penalize`, `debug-checks`); `edge.rs::test_edge_tag_illegal_move_is_penalized_or_rejected` |
| Duplicate ids | engine `inline.rs::test_reset_rejects_duplicate_and_unknown_ids_and_bad_seed_lists` |
| Reseeding and duplicate seeds | engine `inline.rs::test_pool_*seed*`; `reseed.rs` |
| Duplicate fork labels, branch counter exhaustion | engine `snapshot.rs::test_fork_branched_labels_and_counter_errors` |
| Poisoning and a failed recovery that stays poisoned | engine `modes.rs::test_panic_*` and `modes.rs::test_panic_during_recovery_stays_poisoned`, `snapshot.rs::test_import_all_failure_keeps_the_poison`, `sync.rs::test_worker_panic_poisons_and_full_reset_recovers` |

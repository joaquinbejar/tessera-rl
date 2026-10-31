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

Fixtures are rewritten only with `TESSERA_BLESS=1 cargo test -p tessera-rl-classic --test golden`. Blessing refuses to change an existing fixture unless `env_version`, `MATH_VERSION`, the action generator or the format version changed (ADR-0023). A plain run fails on any difference: it reports the first diverging call and field, and which instances end with a different record.

## ADR-0024 layer 3: edge fixtures

| Item | Test |
| --- | --- |
| Signed zeros, subnormals, enums without a zero discriminant | engine `snapshot.rs::test_edge_fixtures_survive_snapshot_and_encoding`, `masks.rs::test_negative_zero_reward_survives_without_illegal_actions` |
| Agent death at t, respawn of the same slot at t+1, snapshots on both sides | engine `snapshot.rs::test_continuation_across_death_and_respawn`; Tag death with snapshots at t-1 and t in `continuation.rs` |
| Same-step slot reuse rejected by an invariant check | **gap**: the engine ignores a spawn on a dying slot and only debug-asserts it; there is no test yet |
| Ignored actions (dead, pending reset, empty slots) | engine `inline.rs::test_agent_death_clears_alive_and_skips_its_validation`, `inline.rs::test_pool_initial_agents_starts_the_rest_empty_until_spawned`; `edge.rs::test_edge_tag_actions_of_empty_and_caught_slots_are_ignored` |
| Illegal actions | engine `masks.rs` (`noop`, `penalize`, `debug-checks`); `edge.rs::test_edge_tag_illegal_move_is_penalized_or_rejected` |
| Duplicate ids | engine `inline.rs::test_reset_rejects_duplicate_and_unknown_ids_and_bad_seed_lists` |
| Reseeding and duplicate seeds | engine `inline.rs::test_pool_*seed*`; `reseed.rs` |
| Duplicate fork labels, branch counter exhaustion | engine `snapshot.rs::test_fork_branched_labels_and_counter_errors` |
| Poisoning and a failed recovery that stays poisoned | engine `modes.rs::test_panic_*` and `modes.rs::test_panic_during_recovery_stays_poisoned`, `snapshot.rs::test_import_all_failure_keeps_the_poison`, `sync.rs::test_worker_panic_poisons_and_full_reset_recovers` |

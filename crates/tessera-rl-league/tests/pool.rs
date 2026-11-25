// SPDX-License-Identifier: MIT OR Apache-2.0

//! A league on an engine pool (ADR-0088 sections 2, 3, 4.1 and 7, erratum
//! 1; #384 head B), on `PaddleDuel` with slot 1 an `Opponent` slot and a
//! population of one heuristic and one external policy.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use proptest::prelude::*;
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_core::PolicyGroup;
use tessera_rl_engine::league::{LeagueSnapshot, LeagueView, OpponentKind};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed, Schedule,
    SyncOptions,
};
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Matchmaking, PolicyKind, PoolLeague, Population, Table, elo,
};
use tessera_rl_num::identity::{RngIdentity, Seed};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const N: u32 = 8;

fn cfg() -> PaddleDuelConfig {
    PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    }
}

fn population(kinds: &[PolicyKind]) -> Population {
    let mut p = Population::new();
    for &k in kinds {
        p.add(k, elo::INITIAL, Rating::initial(), 0).unwrap();
    }
    p
}

fn league(kinds: &[PolicyKind]) -> PoolLeague {
    let config = LeagueConfig {
        matchmaking: Matchmaking::Uniform,
        ..LeagueConfig::default()
    };
    PoolLeague::new(League::new(config, population(kinds)).unwrap()).unwrap()
}

const MIXED: [PolicyKind; 2] = [PolicyKind::Heuristic, PolicyKind::External { tag: 7 }];

fn options(schedule: Schedule, steps: u32) -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::NextStep,
        max_episode_steps: NonZeroU32::new(steps),
        schedule,
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    }
}

fn pool_with(
    kinds: &[PolicyKind],
    workers: Option<usize>,
    schedule: Schedule,
) -> EnvPool<PaddleDuel, NoMasks> {
    let l = league(kinds);
    let view = l.view().unwrap();
    let p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(schedule, 6))
        .unwrap()
        .with_league(&[1], view, Box::new(l))
        .unwrap();
    match workers {
        Some(w) => p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
        None => p,
    }
}

fn pool() -> EnvPool<PaddleDuel, NoMasks> {
    pool_with(&MIXED, None, Schedule::Static)
}

/// The draw ADR-0088 item 3.1 defines, recomputed outside the engine.
fn expected_draw(p: &EnvPool<PaddleDuel, NoMasks>, i: u32) -> u32 {
    let id = p.header(i).unwrap().engine.identity;
    let identity = RngIdentity::from_parts(
        id.seed(),
        id.branch_salt(),
        id.logical_id(),
        id.episode_index(),
        0,
    );
    let mut rng = StepRng::league(&RngAuthority::engine_only(), &identity);
    p.league_view().unwrap().draw(rng.next_u64())
}

/// Actions for both slots: the learner plays `a`, the opponent row `opp`.
fn act(a: i32, opp: i32) -> Vec<i32> {
    (0..N).flat_map(|_| [a, opp]).collect()
}

fn step(p: &mut EnvPool<PaddleDuel, NoMasks>, a: i32) {
    // Opponent rows: a legal external action; ignored for heuristic ones.
    p.step(ActionInput {
        discrete: &act(a, 0),
        continuous: &[],
    })
    .unwrap();
}

proptest! {
    /// Erratum 1, condition 1: the engine view's draw equals the league
    /// table's for the same `u64`, over random tables.
    #[test]
    fn prop_view_draw_equals_the_table_draw(
        weights in prop::collection::vec(prop::option::of(0.0f64..10.0), 1..12),
        us in prop::collection::vec(any::<u64>(), 1..64),
    ) {
        let Ok(table) = Table::from_weights(&weights) else {
            return Ok(());
        };
        let kinds = vec![OpponentKind::Heuristic; weights.len()];
        let view = LeagueView::new(kinds, table.thresholds().to_vec(), table.last(), 1).unwrap();
        for u in us.into_iter().chain([0, u64::MAX]) {
            prop_assert_eq!(Some(view.draw(u)), table.draw(u).map(|id| id.0));
        }
    }
}

#[test]
fn test_league_pool_groups_and_assignment_output() {
    let mut p = pool();
    assert_eq!(
        p.policy_groups(),
        &[PolicyGroup::Learner, PolicyGroup::Opponent]
    );
    p.reset_all(Reseed::Keep).unwrap();
    let a = p.assignment().unwrap().to_vec();
    assert_eq!(a.len(), N as usize);
    for i in 0..N {
        assert_eq!(a[i as usize], expected_draw(&p, i), "instance {i}");
    }
    // Both policies appear among 8 instances (uniform table).
    assert!(a.contains(&0) && a.contains(&1), "{a:?}");
}

#[test]
fn test_league_draw_is_the_same_at_every_worker_count_and_schedule() {
    let run = |workers: Option<usize>, schedule: Schedule| {
        let mut p = pool_with(&MIXED, workers, schedule);
        p.reset_all(Reseed::Keep).unwrap();
        let mut seen = Vec::new();
        for t in 0..30 {
            step(&mut p, t % 3);
            seen.extend_from_slice(p.assignment().unwrap());
            seen.extend(p.state_hash(0).map(|h| (h.get() & 0xFFFF_FFFF) as u32));
        }
        seen
    };
    let inline = run(None, Schedule::Static);
    for w in [1, 2, 7] {
        assert_eq!(run(Some(w), Schedule::Static), inline, "{w} workers");
        assert_eq!(
            run(Some(w), Schedule::Dynamic),
            inline,
            "{w} workers, dynamic"
        );
    }
}

#[test]
fn test_league_assignment_changes_only_at_episode_boundaries() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..40 {
        let before: Vec<(u32, u32)> = (0..N)
            .map(|i| {
                (
                    p.header(i).unwrap().engine.identity.episode_index(),
                    p.assignment().unwrap()[i as usize],
                )
            })
            .collect();
        step(&mut p, 1);
        for i in 0..N {
            let episode = p.header(i).unwrap().engine.identity.episode_index();
            let now = p.assignment().unwrap()[i as usize];
            if episode == before[i as usize].0 {
                assert_eq!(
                    now, before[i as usize].1,
                    "instance {i} changed mid-episode"
                );
            } else {
                assert_eq!(now, expected_draw(&p, i), "instance {i} at a new episode");
            }
        }
    }
}

#[test]
fn test_league_external_rows_are_validated_heuristic_rows_ignored() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    let a = p.assignment().unwrap().to_vec();
    let external = a.iter().position(|&x| x == 1).unwrap();
    let heuristic = a.iter().position(|&x| x == 0).unwrap();
    // Garbage in a heuristic opponent's row only: accepted.
    let mut rows = act(0, 0);
    rows[heuristic * 2 + 1] = 99;
    p.step(ActionInput {
        discrete: &rows,
        continuous: &[],
    })
    .unwrap();
    // Garbage in an external opponent's row: refused at the boundary.
    let mut rows = act(0, 0);
    rows[external * 2 + 1] = 99;
    assert!(matches!(
        p.step(ActionInput {
            discrete: &rows,
            continuous: &[],
        }),
        Err(EngineError::DiscreteOutOfRange { .. })
    ));
}

#[test]
fn test_league_fork_copies_the_assignment() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    let a = p.assignment().unwrap()[0];
    p.fork(0, &[1, 2, 3]).unwrap();
    for i in 1..4 {
        assert_eq!(p.assignment().unwrap()[i], a);
    }
    p.fork_branched(0, &[4, 5], None).unwrap();
    for i in 4..6 {
        assert_eq!(
            p.assignment().unwrap()[i],
            a,
            "branched forks keep the opponent"
        );
    }
}

#[test]
fn test_league_plain_restore_redraws_from_the_record_identity_equal_and_different_tables() {
    // Equal tables: a mid-episode record of instance 3 restored into
    // instance 5 of a pool with the same table draws the source's policy.
    let mut a = pool();
    a.reset_all(Reseed::Keep).unwrap();
    step(&mut a, 0);
    let record = a.snapshot(3).unwrap();
    let source = a.assignment().unwrap()[3];
    let mut b = pool();
    b.reset_all(Reseed::Keep).unwrap();
    b.restore(5, &record).unwrap();
    assert_eq!(
        b.assignment().unwrap()[5],
        source,
        "same table: the source's assignment"
    );

    // Different tables: a pool whose table assigns everything to policy 1
    // redraws from its own table, never keeping its previous assignment.
    let only_external = league(&[
        PolicyKind::External { tag: 1 },
        PolicyKind::External { tag: 2 },
    ]);
    let skewed = LeagueView::new(
        vec![
            OpponentKind::External { tag: 1 },
            OpponentKind::External { tag: 2 },
        ],
        vec![0, u64::MAX],
        Some(1),
        only_external.view().unwrap().population_digest(),
    )
    .unwrap();
    let mut c = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 6))
        .unwrap()
        .with_league(&[1], skewed, Box::new(only_external))
        .unwrap();
    c.reset_all(Reseed::Keep).unwrap();
    assert!(c.assignment().unwrap().iter().all(|&x| x == 1));
    c.restore(5, &record).unwrap();
    assert_eq!(
        c.assignment().unwrap()[5],
        1,
        "different table: the destination's table decides"
    );
    assert_eq!(c.assignment().unwrap()[5], expected_draw(&c, 5));
}

/// A never-reset pool's snapshot restored into another pool draws at its
/// first reset exactly as the source does (#384 head E3): the restored
/// assignments of fresh or pending-reset instances are not marked drawn.
#[test]
fn test_league_restore_of_a_never_reset_pool_draws_as_the_source() {
    let mut a = pool();
    let state = a.snapshot_pool().unwrap();
    let mut b = pool();
    b.restore_pool(&state, None).unwrap();
    a.reset_all(Reseed::Keep).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    assert_eq!(b.assignment().unwrap(), a.assignment().unwrap());
    assert!(
        a.assignment().unwrap().iter().any(|&p| p != 0),
        "the draw is not all the default entry"
    );
}

#[test]
fn test_league_snapshot_and_restore_with_league_round_trip_and_refuse_another_population() {
    let mut a = pool();
    a.reset_all(Reseed::Keep).unwrap();
    step(&mut a, 2);
    let records = [a.snapshot(0).unwrap(), a.snapshot(1).unwrap()];
    let league = a.snapshot_league(&[0, 1]).unwrap();
    let mut b = pool();
    b.reset_all(Reseed::Keep).unwrap();
    b.restore_with_league(&[6, 7], &records, &league).unwrap();
    assert_eq!(&b.assignment().unwrap()[6..8], &league.assignments[..]);
    assert_eq!(b.state_hash(6).unwrap(), a.state_hash(0).unwrap());

    let before = b.assignment().unwrap().to_vec();
    let other = LeagueSnapshot {
        assignments: league.assignments.clone(),
        population_digest: league.population_digest ^ 1,
    };
    assert!(matches!(
        b.restore_with_league(&[6, 7], &records, &other),
        Err(EngineError::LeagueMismatch { .. })
    ));
    assert_eq!(
        b.assignment().unwrap(),
        &before[..],
        "a refused restore mutates nothing"
    );
}

#[test]
fn test_league_barrier_applies_episode_results() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    // Episodes end every 6 steps (time limit).
    for t in 0..30 {
        step(&mut p, t % 3);
    }
    let report = p.epoch_barrier().unwrap();
    let league = report.league.unwrap();
    assert!(league.applied > 0, "{league:?}");
    // A second barrier right away has nothing to apply.
    assert_eq!(p.epoch_barrier().unwrap().league.unwrap().applied, 0);
    // Frozen: nothing kept, the barrier reports nothing.
    p.freeze_league(true);
    for t in 0..12 {
        step(&mut p, t % 3);
    }
    assert_eq!(p.epoch_barrier().unwrap().league, None);
}

#[test]
fn test_league_onnx_policies_need_onnx_opponents_and_bad_setups_are_refused() {
    let onnx = PoolLeague::new(
        League::new(
            LeagueConfig {
                matchmaking: Matchmaking::Uniform,
                ..LeagueConfig::default()
            },
            population(&[PolicyKind::Onnx {
                content_hash: [7; 16],
            }]),
        )
        .unwrap(),
    )
    .unwrap();
    let view = onnx.view().unwrap();
    // ADR-0088 erratum 2: an ONNX opponent without `with_onnx_opponents`
    // is refused at the step, before any state changes.
    let mut p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 2))
        .unwrap()
        .with_league(&[1], view, Box::new(onnx))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let actions = vec![0i32; N as usize * 2];
    let refused = p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    });
    assert!(matches!(refused, Err(EngineError::League { .. })));
    assert!(!p.is_poisoned());
    let l = league(&MIXED);
    let view = l.view().unwrap();
    let none = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 6))
        .unwrap()
        .with_league(&[], view, Box::new(l));
    assert!(matches!(none, Err(EngineError::League { .. })));
}

/// ADR-0088 item 4.2: results are applied in instance order, then oldest
/// first, so the league after a barrier is the same bits at any worker
/// count and schedule.
#[test]
fn test_league_barrier_is_bit_identical_at_every_worker_count_and_schedule() {
    let run = |workers: Option<usize>, schedule: Schedule| {
        let config = LeagueConfig {
            matchmaking: Matchmaking::Elo { temperature: 50.0 },
            rate_population: true,
            ..LeagueConfig::default()
        };
        let l = PoolLeague::new(League::new(config, population(&MIXED)).unwrap()).unwrap();
        let view = l.view().unwrap();
        let p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(schedule, 4))
            .unwrap()
            .with_league(&[1], view, Box::new(l))
            .unwrap();
        let mut p = match workers {
            Some(w) => p
                .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
                .unwrap(),
            None => p,
        };
        p.reset_all(Reseed::Keep).unwrap();
        let mut out = Vec::new();
        for round in 0..3 {
            for t in 0..16 {
                step(&mut p, (t + round) % 3);
            }
            let report = p.epoch_barrier().unwrap().league.unwrap();
            out.push((report, p.league_view().unwrap().clone()));
        }
        out
    };
    let inline = run(None, Schedule::Static);
    assert!(inline.iter().any(|(r, _)| r.applied > 0));
    for w in [2, 7] {
        assert_eq!(run(Some(w), Schedule::Static), inline, "{w} workers");
        assert_eq!(
            run(Some(w), Schedule::Dynamic),
            inline,
            "{w} workers, dynamic"
        );
    }
}

/// ADR-0088 item 4.1: a full ring (capacity 8) drops its oldest entry and
/// counts the drop; the barrier reports it.
#[test]
fn test_league_ring_overflow_counts_drops() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    // Episodes of 6 steps with NextStep resets: 7 steps per episode, so 70
    // steps end 10 episodes per instance, 2 over the capacity.
    for t in 0..70 {
        step(&mut p, t % 3);
    }
    let report = p.epoch_barrier().unwrap().league.unwrap();
    assert_eq!(report.applied, u64::from(N) * 8, "{report:?}");
    assert_eq!(report.dropped, u64::from(N) * 2, "{report:?}");
}

/// A subsystem that records what the barrier hands it, then keeps the view.
struct Recording {
    view: LeagueView,
    seen: std::sync::Arc<std::sync::Mutex<Vec<(u32, u32)>>>,
}

impl tessera_rl_engine::league::sealed::Sealed for Recording {}

impl tessera_rl_engine::league::LeagueSubsystem for Recording {
    fn barrier(
        &mut self,
        results: tessera_rl_engine::league::LeagueResults<'_>,
    ) -> Result<
        (LeagueView, tessera_rl_engine::league::LeagueReport),
        tessera_rl_engine::league::LeagueBarrierError,
    > {
        let mut seen = self.seen.lock().unwrap();
        seen.extend(results.results.iter().map(|r| (r.instance, r.policy)));
        Ok((
            self.view.clone(),
            tessera_rl_engine::league::LeagueReport::new(
                results.results.len() as u64,
                results.dropped,
                false,
            ),
        ))
    }
}

/// ADR-0088 item 4.2: the barrier hands the results in instance order, and
/// oldest first within an instance (each instance's policies in the order
/// its episodes were assigned).
#[test]
fn test_league_barrier_hands_results_in_instance_then_oldest_first_order() {
    let view = league(&MIXED).view().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sub = Recording {
        view: view.clone(),
        seen: seen.clone(),
    };
    let mut p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 6))
        .unwrap()
        .with_league(&[1], view, Box::new(sub))
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut assigned: Vec<Vec<u32>> = vec![Vec::new(); N as usize];
    let mut episode = vec![u32::MAX; N as usize];
    for t in 0..40 {
        for i in 0..N {
            let e = p.header(i).unwrap().engine.identity.episode_index();
            if e != episode[i as usize] {
                episode[i as usize] = e;
                assigned[i as usize].push(p.assignment().unwrap()[i as usize]);
            }
        }
        step(&mut p, t % 3);
    }
    p.epoch_barrier().unwrap();
    let seen = seen.lock().unwrap().clone();
    let instances: Vec<u32> = seen.iter().map(|&(i, _)| i).collect();
    assert!(
        instances.windows(2).all(|w| w[0] <= w[1]),
        "instance order: {instances:?}"
    );
    for i in 0..N {
        let policies: Vec<u32> = seen
            .iter()
            .filter(|&&(k, _)| k == i)
            .map(|&(_, p)| p)
            .collect();
        let ended = &assigned[i as usize][..policies.len()];
        assert_eq!(policies, ended, "instance {i}: oldest first");
    }
}

/// A subsystem whose barrier switches to `next` (a table change).
struct Switch {
    next: LeagueView,
}

impl tessera_rl_engine::league::sealed::Sealed for Switch {}

impl tessera_rl_engine::league::LeagueSubsystem for Switch {
    fn barrier(
        &mut self,
        results: tessera_rl_engine::league::LeagueResults<'_>,
    ) -> Result<
        (LeagueView, tessera_rl_engine::league::LeagueReport),
        tessera_rl_engine::league::LeagueBarrierError,
    > {
        Ok((
            self.next.clone(),
            tessera_rl_engine::league::LeagueReport::new(
                results.results.len() as u64,
                results.dropped,
                false,
            ),
        ))
    }
}

/// ADR-0088 item 3.2: a plain restore redraws even a record of the same
/// instance and episode, from the destination's current table, never
/// keeping the previous assignment.
#[test]
fn test_league_plain_restore_of_its_own_record_redraws_from_the_current_table() {
    let uniform = league(&MIXED).view().unwrap();
    let all_external = LeagueView::new(
        uniform.kinds().to_vec(),
        vec![0, u64::MAX],
        Some(1),
        uniform.population_digest(),
    )
    .unwrap();
    let mut p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 6))
        .unwrap()
        .with_league(&[1], uniform, Box::new(Switch { next: all_external }))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let i = p
        .assignment()
        .unwrap()
        .iter()
        .position(|&a| a == 0)
        .unwrap();
    let record = p.snapshot(u32::try_from(i).unwrap()).unwrap();
    p.epoch_barrier().unwrap();
    // Mid-episode, the assignment stays (the table changed at a barrier).
    assert_eq!(p.assignment().unwrap()[i], 0);
    p.restore(u32::try_from(i).unwrap(), &record).unwrap();
    assert_eq!(
        p.assignment().unwrap()[i],
        1,
        "redrawn from the current table"
    );
}

/// ADR-0088 item 3.1: the draw is pinned (the `LEAGUE` stream, the league
/// salt, step 0): the assignments of seed 11 after the first reset.
#[test]
fn test_league_draw_golden() {
    let mut p = pool();
    p.reset_all(Reseed::Keep).unwrap();
    let got = p.assignment().unwrap().to_vec();
    assert_eq!(got, GOLDEN, "the league draw changed: a determinism break");
}

const GOLDEN: [u32; 8] = [1, 0, 1, 1, 1, 0, 1, 0];

/// ADR-0088 section 8 (#384 head D): a league pool records, with
/// `has_league` set; `tests/replay.rs` replays such files.
#[test]
fn test_league_pool_recording_writes_a_league_file() {
    let mut p = pool();
    let rec = tessera_rl_engine::replay::Recorder::new(
        &mut p,
        Vec::new(),
        tessera_rl_engine::replay::ReplayWriterOptions::default(),
    )
    .unwrap();
    let bytes = rec.finish().unwrap();
    let limits = tessera_rl_engine::DecodeLimits {
        max_instances: 1 << 16,
        max_decoded_bytes: 1 << 30,
        max_host_record_bytes: 1 << 20,
    };
    let replay = tessera_rl_engine::replay::Replay::open(&bytes, limits).unwrap();
    assert!(replay.header().has_league);
}

/// A subsystem that fails its first barrier, then records what it gets.
struct FailOnce {
    view: LeagueView,
    calls: u32,
    seen: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
}

impl tessera_rl_engine::league::sealed::Sealed for FailOnce {}

impl tessera_rl_engine::league::LeagueSubsystem for FailOnce {
    fn barrier(
        &mut self,
        results: tessera_rl_engine::league::LeagueResults<'_>,
    ) -> Result<
        (LeagueView, tessera_rl_engine::league::LeagueReport),
        tessera_rl_engine::league::LeagueBarrierError,
    > {
        self.calls += 1;
        self.seen.lock().unwrap().push(results.results.len());
        if self.calls == 1 {
            return Err(tessera_rl_engine::league::LeagueBarrierError {
                reason: "first barrier fails".to_owned(),
            });
        }
        Ok((
            self.view.clone(),
            tessera_rl_engine::league::LeagueReport::new(
                results.results.len() as u64,
                results.dropped,
                false,
            ),
        ))
    }
}

/// Rev's ruling 1 on head B: after a failed league barrier the results are
/// kept, the pool is not poisoned, and the next barrier applies them
/// exactly once.
#[test]
fn test_league_results_kept_after_a_failed_barrier_apply_exactly_once() {
    let view = league(&MIXED).view().unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sub = FailOnce {
        view: view.clone(),
        calls: 0,
        seen: seen.clone(),
    };
    let mut p = EnvPool::<PaddleDuel, NoMasks>::new(cfg(), options(Schedule::Static, 6))
        .unwrap()
        .with_league(&[1], view, Box::new(sub))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..14 {
        step(&mut p, t % 3);
    }
    assert!(matches!(
        p.epoch_barrier(),
        Err(EngineError::LeagueBarrier(_))
    ));
    let applied = p.epoch_barrier().unwrap().league.unwrap().applied;
    assert_eq!(p.epoch_barrier().unwrap().league.unwrap().applied, 0);
    let seen = seen.lock().unwrap().clone();
    assert!(seen[0] > 0, "{seen:?}");
    // The second barrier got exactly the kept results, the third none.
    assert_eq!(usize::try_from(applied).unwrap(), seen[0], "{seen:?}");
    assert_eq!(seen, vec![seen[0], seen[0], 0]);
    // Not poisoned: the pool still steps.
    step(&mut p, 0);
}

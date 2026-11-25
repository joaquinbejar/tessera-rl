// SPDX-License-Identifier: MIT OR Apache-2.0

//! A league shared by several pools (ADR-0088 section 6; #384 head E1,
//! plan r2): the result does not depend on the order or the threads the
//! pools' barriers run on; an early barrier is refused, never blocked; a
//! shared league lags one round, one pool included; self-play's share is
//! exact and its results stay out of the ratings; snapshot rules add
//! entries the reports name.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::league::LeagueView;
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_league::trueskill::{Params, Rating};
use tessera_rl_league::{
    Matchmaking, PfspWeighting, PolicyKind, Population, RoleConfig, SelfPlay, SharedConfig,
    SharedLeague, SnapshotRule, elo,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 8;
const A: usize = 2;

type Pool = EnvPool<PaddleDuel, NoMasks>;

fn population() -> Population {
    let mut p = Population::new();
    for k in [
        PolicyKind::Heuristic,
        PolicyKind::External { tag: 7 },
        PolicyKind::External { tag: 9 },
    ] {
        p.add(k, elo::INITIAL, Rating::initial(), 0).unwrap();
    }
    p
}

fn config(roles: Vec<RoleConfig>) -> SharedConfig {
    SharedConfig {
        elo_k: 16.0,
        trueskill: Params::default(),
        rate_population: true,
        roles,
    }
}

/// PFSP tables move with the draw counts that short `PaddleDuel` episodes
/// give, so every commit changes the views.
fn pfsp() -> RoleConfig {
    RoleConfig::new(Matchmaking::Pfsp(PfspWeighting::Hard))
}

fn pool_for(league: &SharedLeague, role: usize, seed: u64) -> Pool {
    let handle = league.member(role).unwrap();
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        max_episode_steps: NonZeroU32::new(60),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(seed))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    let mut p = Pool::new(cfg, options)
        .unwrap()
        .with_league(&[1], league.view(role).unwrap(), Box::new(handle))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn steps(p: &mut Pool, t: usize) {
    for s in 0..60 {
        let a: Vec<i32> = (0..N as usize * A)
            .map(|k| i32::try_from((t * 5 + s + k) % 3).unwrap())
            .collect();
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
}

/// Everything the league and the pools hold after `rounds` rounds, with
/// each round's barriers run in the order `order(round)` returns, each on
/// its own thread when `threads` is set.
fn run(rounds: usize, order: impl Fn(usize) -> Vec<usize>, threads: bool) -> Vec<u8> {
    let snapshots = RoleConfig {
        snapshot: Some(SnapshotRule::EveryRounds(3)),
        snapshot_tags: 500,
        ..pfsp()
    };
    let league = SharedLeague::new(config(vec![snapshots, pfsp()]), population()).unwrap();
    let initial = league.view(0).unwrap();
    let mut pools = vec![
        pool_for(&league, 0, 1),
        pool_for(&league, 1, 2),
        pool_for(&league, 0, 3),
    ];
    let mut out = Vec::new();
    for round in 0..rounds {
        for (i, p) in pools.iter_mut().enumerate() {
            steps(p, round * 3 + i);
        }
        let order = order(round);
        if threads {
            // All barriers of the round concurrently: the commit happens on
            // whichever thread submits last.
            std::thread::scope(|s| {
                for p in &mut pools {
                    s.spawn(move || p.epoch_barrier().unwrap());
                }
            });
        } else {
            for &i in &order {
                pools.get_mut(i).unwrap().epoch_barrier().unwrap();
            }
        }
        for p in &pools {
            out.extend(p.assignment().unwrap().iter().flat_map(|x| x.to_le_bytes()));
            for t in p.league_view().unwrap().thresholds() {
                out.extend(t.to_le_bytes());
            }
        }
    }
    // The views moved: the test compares more than initial tables.
    assert_ne!(
        league.view(0).unwrap().thresholds(),
        initial.thresholds(),
        "the tables never changed"
    );
    for role in 0..2 {
        let (e, ts, counts, _) = league.role(role).unwrap();
        out.extend(e.to_bits().to_le_bytes());
        out.extend(ts.mu.to_bits().to_le_bytes());
        out.extend(ts.sigma.to_bits().to_le_bytes());
        for c in counts {
            out.extend(c.wins.to_le_bytes());
            out.extend(c.draws.to_le_bytes());
            out.extend(c.losses.to_le_bytes());
        }
    }
    for p in league.population().unwrap().entries() {
        out.extend(p.elo.to_bits().to_le_bytes());
    }
    out
}

#[test]
fn test_shared_league_is_independent_of_barrier_order_and_threads() {
    let reference = run(12, |_| vec![0, 1, 2], false);
    assert_eq!(
        run(12, |_| vec![2, 0, 1], false),
        reference,
        "reversed order"
    );
    assert_eq!(
        run(
            12,
            |r| if r % 2 == 0 {
                vec![1, 2, 0]
            } else {
                vec![2, 1, 0]
            },
            false
        ),
        reference,
        "alternating orders"
    );
    for attempt in 0..4 {
        assert_eq!(
            run(12, |_| vec![], true),
            reference,
            "threads, attempt {attempt}"
        );
    }
}

/// A barrier of a pool that already submitted the round is refused with a
/// typed error naming the pools still to submit; the pool is unchanged and
/// unpoisoned, and a retry after the others succeeds. Nothing blocks.
#[test]
fn test_shared_league_early_barrier_is_refused_never_blocked() {
    let league = SharedLeague::new(config(vec![pfsp()]), population()).unwrap();
    let mut a = pool_for(&league, 0, 1);
    let mut b = pool_for(&league, 0, 2);
    steps(&mut a, 0);
    steps(&mut b, 1);
    a.epoch_barrier().unwrap();
    steps(&mut a, 2);
    let view = a.league_view().unwrap().clone();
    let refused = a.epoch_barrier();
    assert!(
        matches!(&refused, Err(EngineError::LeagueBarrier(e)) if e.reason.contains("waiting for pools [1]")),
        "{refused:?}"
    );
    assert!(!a.is_poisoned());
    assert_eq!(a.league_view(), Some(&view));
    assert_eq!(league.round().unwrap(), 0);
    b.epoch_barrier().unwrap();
    assert_eq!(league.round().unwrap(), 1);
    a.epoch_barrier().unwrap();
}

/// A shared league lags one round whatever the pool count, one included:
/// a pool's barrier returns the view committed before it, and installs the
/// new view at its next barrier.
#[test]
fn test_shared_league_lags_one_round_with_one_pool() {
    // A snapshot every commit, so each commit changes the view.
    let role = RoleConfig {
        snapshot: Some(SnapshotRule::EveryRounds(1)),
        ..pfsp()
    };
    let league = SharedLeague::new(config(vec![role]), population()).unwrap();
    let initial = league.view(0).unwrap();
    let mut p = pool_for(&league, 0, 1);
    steps(&mut p, 0);
    p.epoch_barrier().unwrap();
    let committed = league.view(0).unwrap();
    assert_ne!(committed, initial, "the commit changed the table");
    assert_eq!(p.league_view(), Some(&initial), "installed one round later");
    steps(&mut p, 1);
    p.epoch_barrier().unwrap();
    assert_eq!(p.league_view(), Some(&committed));
}

/// The self-play entry gets exactly `floor(share x 2^64)` of the range,
/// and its draw frequency is within the binomial bound.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "exact scaling of shares and counts in the test's own arithmetic"
)]
fn test_shared_league_self_play_share_is_exact() {
    for share in [0.25, 0.5, 0.1] {
        let role = RoleConfig {
            self_play: Some(SelfPlay { share, tag: 77 }),
            ..pfsp()
        };
        let league = SharedLeague::new(config(vec![role]), population()).unwrap();
        let view: LeagueView = league.view(0).unwrap();
        // The self-play entry is added after the three given entries.
        let id = 3usize;
        let t = view.thresholds();
        // The entry is the last one: its `u64::MAX` threshold stands for
        // 2^64, the end of the draw range.
        assert_eq!(view.last(), u32::try_from(id).unwrap());
        let width = (1u128 << 64) - u128::from(t[id - 1]);
        let expected = (share * 18_446_744_073_709_551_616.0).floor() as u64;
        assert_eq!(width, u128::from(expected), "share {share}");
        // Draws over 2^16 evenly spaced u64 values: exact proportion.
        let n = 1u64 << 16;
        let hits = (0..n)
            .filter(|k| view.draw(k.wrapping_mul(u64::MAX / n)) == u32::try_from(id).unwrap())
            .count() as f64;
        let p = share;
        let sd = (n as f64 * p * (1.0 - p)).sqrt();
        assert!(
            (hits - n as f64 * p).abs() <= 4.0 * sd + 1.0,
            "share {share}: {hits}"
        );
    }
}

/// `EveryRounds(2)` adds a snapshot at commits 2 and 4, tagged from the
/// role's base, drawable by every role, and the reports name them.
#[test]
fn test_shared_league_snapshot_rule_adds_tagged_entries() {
    let role = RoleConfig {
        snapshot: Some(SnapshotRule::EveryRounds(2)),
        snapshot_tags: 1000,
        ..pfsp()
    };
    let league = SharedLeague::new(config(vec![role, pfsp()]), population()).unwrap();
    let mut a = pool_for(&league, 0, 1);
    let mut b = pool_for(&league, 1, 2);
    let mut tags = Vec::new();
    for t in 0..4 {
        steps(&mut a, t);
        steps(&mut b, t + 10);
        a.epoch_barrier().unwrap();
        b.epoch_barrier().unwrap();
        tags.extend(
            league
                .last_commit()
                .unwrap()
                .unwrap()
                .snapshots
                .iter()
                .map(|s| s.tag),
        );
    }
    assert_eq!(tags, [1000, 1001]);
    let population = league.population().unwrap();
    assert_eq!(population.len(), 5);
    let kinds: Vec<_> = population.entries().iter().map(|p| p.kind).collect();
    assert_eq!(kinds[3], PolicyKind::External { tag: 1000 });
    assert_eq!(kinds[4], PolicyKind::External { tag: 1001 });
    // Role 1 may draw role 0's snapshots (no role bits).
    let (_, _, _, view) = league.role(1).unwrap();
    assert!(view.thresholds()[4] > view.thresholds()[2]);
}

#[test]
fn test_shared_league_refuses_bad_configs_and_late_members() {
    let bad = |roles| SharedLeague::new(config(roles), population()).is_err();
    assert!(bad(vec![]));
    assert!(bad(vec![RoleConfig {
        self_play: Some(SelfPlay { share: 1.0, tag: 1 }),
        ..pfsp()
    }]));
    assert!(bad(vec![RoleConfig {
        snapshot: Some(SnapshotRule::EveryRounds(0)),
        ..pfsp()
    }]));
    let league = SharedLeague::new(config(vec![pfsp()]), population()).unwrap();
    assert!(league.member(1).is_err());
    let mut p = pool_for(&league, 0, 1);
    steps(&mut p, 0);
    p.epoch_barrier().unwrap();
    assert!(
        league.member(0).is_err(),
        "registration after the first barrier"
    );
}

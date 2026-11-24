// SPDX-License-Identifier: MIT OR Apache-2.0

//! The barrier: order, determinism, freeze, validation (ADR-0088 item 4).

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    reason = "test helpers unwrap, index and cast: a failure is the assertion"
)]

use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, LeagueError, Matchmaking, Outcome, PfspWeighting, PolicyId, PolicyKind,
    Population, Results, elo,
};

fn population(n: u32) -> Population {
    let mut p = Population::new();
    for tag in 0..n {
        p.add(
            PolicyKind::External {
                tag: u64::from(tag),
            },
            elo::INITIAL,
            Rating::initial(),
            0,
        )
        .unwrap();
    }
    p
}

fn filled(instances: usize, k: usize, seed: u64) -> Results {
    let mut r = Results::new(instances, k).unwrap();
    let mut s = seed | 1;
    for i in 0..instances {
        for _ in 0..k {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let o = [Outcome::Win, Outcome::Draw, Outcome::Loss][(s % 3) as usize];
            r.record(i, PolicyId(((s >> 8) % 4) as u32), o).unwrap();
        }
    }
    r
}

fn bits(l: &League) -> Vec<u64> {
    let mut out = vec![
        l.learner_elo().to_bits(),
        l.learner_trueskill().mu.to_bits(),
        l.learner_trueskill().sigma.to_bits(),
    ];
    out.extend(l.table().thresholds());
    for c in l.counts() {
        out.extend([c.wins, c.draws, c.losses]);
    }
    out
}

#[test]
fn test_league_barrier_is_bit_identical_for_the_same_inputs() {
    let cfg = LeagueConfig {
        rate_population: true,
        ..LeagueConfig::default()
    };
    let mut a = League::new(cfg, population(4)).unwrap();
    let mut b = League::new(cfg, population(4)).unwrap();
    for seed in 1..4 {
        a.barrier(&mut filled(16, 8, seed)).unwrap();
        b.barrier(&mut filled(16, 8, seed)).unwrap();
    }
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn test_league_barrier_applies_instance_order_then_oldest_first() {
    // Elo updates are sequential: the same results in another order give
    // other bits, so the order is observable and must be the contract one.
    let cfg = LeagueConfig::default();
    let mut forward = Results::new(2, 2).unwrap();
    forward.record(0, PolicyId(0), Outcome::Win).unwrap();
    forward.record(0, PolicyId(1), Outcome::Loss).unwrap();
    forward.record(1, PolicyId(2), Outcome::Win).unwrap();
    let mut a = League::new(cfg, population(3)).unwrap();
    a.barrier(&mut forward).unwrap();
    // Reference: the same updates applied by hand in contract order.
    let mut r = elo::INITIAL;
    for (o, opp) in [
        (Outcome::Win, elo::INITIAL),
        (Outcome::Loss, elo::INITIAL),
        (Outcome::Win, elo::INITIAL),
    ] {
        r = elo::update(r, opp, o, cfg.elo_k).0;
    }
    assert_eq!(a.learner_elo().to_bits(), r.to_bits());
    assert_eq!(a.counts()[0].wins, 1);
    assert_eq!(a.counts()[1].losses, 1);
    assert_eq!(a.counts()[2].wins, 1);
}

#[test]
fn test_league_full_ring_drops_the_oldest_and_counts_it() {
    let mut r = Results::new(1, 2).unwrap();
    r.record(0, PolicyId(0), Outcome::Win).unwrap();
    r.record(0, PolicyId(1), Outcome::Draw).unwrap();
    r.record(0, PolicyId(2), Outcome::Loss).unwrap();
    assert_eq!(r.dropped(), 1);
    assert_eq!(
        r.of(0).collect::<Vec<_>>(),
        [(PolicyId(1), Outcome::Draw), (PolicyId(2), Outcome::Loss)]
    );
    let mut l = League::new(LeagueConfig::default(), population(3)).unwrap();
    let report = l.barrier(&mut r).unwrap();
    assert_eq!((report.applied, report.dropped), (2, 1));
    assert_eq!(r.dropped(), 0);
    assert_eq!(r.of(0).count(), 0);
    assert_eq!(
        Results::new(1, 2)
            .unwrap()
            .record(1, PolicyId(0), Outcome::Win),
        Err(LeagueError::UnknownInstance {
            instance: 1,
            instances: 1
        })
    );
}

#[test]
fn test_league_frozen_barrier_changes_nothing() {
    let mut l = League::new(LeagueConfig::default(), population(4)).unwrap();
    let before = bits(&l);
    l.set_frozen(true);
    let report = l.barrier(&mut filled(8, 4, 9)).unwrap();
    assert_eq!(report.applied, 0);
    assert_eq!(bits(&l), before);
}

#[test]
fn test_league_unknown_policy_mutates_nothing() {
    let mut l = League::new(LeagueConfig::default(), population(2)).unwrap();
    let before = bits(&l);
    let mut r = Results::new(2, 2).unwrap();
    r.record(0, PolicyId(0), Outcome::Win).unwrap();
    r.record(1, PolicyId(7), Outcome::Win).unwrap();
    assert_eq!(l.barrier(&mut r), Err(LeagueError::UnknownPolicy(7)));
    assert_eq!(bits(&l), before);
    assert_eq!(r.of(0).count(), 1);
}

#[test]
fn test_league_pfsp_prefers_the_opponents_the_learner_loses_to() {
    let cfg = LeagueConfig {
        matchmaking: Matchmaking::Pfsp(PfspWeighting::Hard),
        ..LeagueConfig::default()
    };
    let mut l = League::new(cfg, population(2)).unwrap();
    let mut r = Results::new(1, 8).unwrap();
    for _ in 0..4 {
        r.record(0, PolicyId(0), Outcome::Win).unwrap();
        r.record(0, PolicyId(1), Outcome::Loss).unwrap();
    }
    l.barrier(&mut r).unwrap();
    // p0 = 5/6, p1 = 1/6: weights 1/36 and 25/36.
    let t = l.table().thresholds();
    let share0 = t[0] as f64 / 18_446_744_073_709_551_616.0;
    assert!((share0 - 1.0 / 26.0).abs() < 1e-12, "{share0}");
}

#[test]
fn test_league_retire_and_add_keep_ids_and_rebuild_the_table() {
    let mut l = League::new(LeagueConfig::default(), population(2)).unwrap();
    let id = l.add_policy(PolicyKind::Heuristic, 1).unwrap();
    assert_eq!(id, PolicyId(2));
    let digest = l.population().digest();
    l.retire(PolicyId(0)).unwrap();
    assert_ne!(l.population().digest(), digest);
    assert!(l.population().get(PolicyId(0)).unwrap().retired);
    for u in [0, u64::MAX / 2, u64::MAX] {
        assert_ne!(l.table().draw(u), Some(PolicyId(0)));
    }
    l.retire(PolicyId(1)).unwrap();
    assert_eq!(l.retire(PolicyId(2)), Err(LeagueError::NoActivePolicy));
    assert!(!l.population().get(PolicyId(2)).unwrap().retired);
}

#[test]
fn test_league_config_rejects_invalid_values() {
    let bad = [
        LeagueConfig {
            elo_k: 0.0,
            ..LeagueConfig::default()
        },
        LeagueConfig {
            elo_k: f64::NAN,
            ..LeagueConfig::default()
        },
        LeagueConfig {
            matchmaking: Matchmaking::Elo { temperature: 0.0 },
            ..LeagueConfig::default()
        },
    ];
    for cfg in bad {
        assert!(matches!(
            League::new(cfg, population(1)),
            Err(LeagueError::InvalidConfig(_))
        ));
    }
    assert!(matches!(
        Results::new(4, 0),
        Err(LeagueError::InvalidConfig(_))
    ));
}

#[test]
fn test_league_win_rate_counts_draws_as_half() {
    let c = tessera_rl_league::PairCounts {
        wins: 0,
        draws: 2,
        losses: 0,
    };
    assert_eq!(c.win_rate(), 0.5);
    let c = tessera_rl_league::PairCounts {
        wins: 1,
        draws: 2,
        losses: 3,
    };
    assert_eq!(c.win_rate(), 3.0 / 8.0);
}

#[test]
fn test_league_population_ratings_move_only_with_rate_population() {
    for rate in [false, true] {
        let cfg = LeagueConfig {
            rate_population: rate,
            ..LeagueConfig::default()
        };
        let mut l = League::new(cfg, population(1)).unwrap();
        let mut r = Results::new(1, 1).unwrap();
        r.record(0, PolicyId(0), Outcome::Win).unwrap();
        l.barrier(&mut r).unwrap();
        let opp = *l.population().get(PolicyId(0)).unwrap();
        assert_eq!(opp.elo != elo::INITIAL, rate, "elo, rate_population {rate}");
        assert_eq!(
            opp.trueskill != Rating::initial(),
            rate,
            "trueskill, rate_population {rate}"
        );
        assert!(l.learner_elo() > elo::INITIAL);
    }
}

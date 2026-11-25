// SPDX-License-Identifier: MIT OR Apache-2.0

//! League persistence and export (ADR-0088 item 7.5, REVIEW-headB-r1
//! ruling 4; #384 head E2, plan r1 as ruled): bit-exact round trips, a
//! restart mid-training that equals no restart, the refusals, corrupt
//! files as typed errors, and the pinned export text.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use proptest::prelude::*;
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_league::trueskill::{Params, Rating};
use tessera_rl_league::{
    League, LeagueConfig, LeagueError, Matchmaking, Outcome, PfspWeighting, PolicyId, PolicyKind,
    Population, Results, RoleConfig, SelfPlay, SharedConfig, SharedLeague, SnapshotRule, elo,
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
        PolicyKind::Onnx {
            content_hash: [7; 16],
        },
    ] {
        p.add(k, elo::INITIAL, Rating::initial(), 0).unwrap();
    }
    p
}

fn pfsp() -> RoleConfig {
    RoleConfig::new(Matchmaking::Pfsp(PfspWeighting::Hard))
}

fn shared_config() -> SharedConfig {
    SharedConfig {
        elo_k: 16.0,
        trueskill: Params::default(),
        rate_population: true,
        roles: vec![
            RoleConfig {
                self_play: Some(SelfPlay {
                    share: 0.25,
                    tag: 99,
                }),
                snapshot: Some(SnapshotRule::EveryRounds(2)),
                snapshot_tags: 500,
                ..pfsp()
            },
            pfsp(),
        ],
    }
}

fn heuristic_population() -> Population {
    let mut p = Population::new();
    for k in [PolicyKind::Heuristic, PolicyKind::External { tag: 7 }] {
        p.add(k, elo::INITIAL, Rating::initial(), 0).unwrap();
    }
    p
}

fn pool_for(league: &SharedLeague, role: usize, seed: u64) -> Pool {
    pool_with(league, role, seed, league.view(role).unwrap())
}

/// A pool of `role` with `view`: after a restart, the view its snapshot's
/// pool context holds (one round behind the league's current one).
fn pool_with(
    league: &SharedLeague,
    role: usize,
    seed: u64,
    view: tessera_rl_engine::league::LeagueView,
) -> Pool {
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
        .with_league(&[1], view, Box::new(handle))
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

const ROLES: [usize; 3] = [0, 1, 0];

/// 12 rounds of three pools (the heuristic population: the External
/// self-play and snapshot entries are served by the caller's zero rows);
/// with `restart`, the league is saved and every pool snapshotted after
/// round 6, and the run goes on in a loaded league and new pools.
fn run(restart: bool) -> (Vec<u8>, String) {
    let mut league = SharedLeague::new(shared_config(), heuristic_population()).unwrap();
    let mut pools: Vec<Pool> = ROLES
        .iter()
        .enumerate()
        .map(|(i, &r)| pool_for(&league, r, 1 + i as u64))
        .collect();
    for round in 0..12 {
        if restart && round == 6 {
            let bytes = league.save().unwrap();
            let states: Vec<_> = pools.iter().map(|p| p.snapshot_pool().unwrap()).collect();
            drop(pools);
            league = SharedLeague::load(&bytes).unwrap();
            pools = ROLES
                .iter()
                .zip(&states)
                .enumerate()
                .map(|(i, (&r, s))| {
                    let view = s.context().adaptive.league.as_ref().unwrap().view.clone();
                    pool_with(&league, r, 100 + i as u64, view)
                })
                .collect();
            for ((p, s), &r) in pools.iter_mut().zip(&states).zip(&ROLES) {
                p.restore_pool(s, None).unwrap();
                league.check_view(r, p.league_view().unwrap()).unwrap();
            }
        }
        for (i, p) in pools.iter_mut().enumerate() {
            steps(p, round * 3 + i);
        }
        for p in &mut pools {
            p.epoch_barrier().unwrap();
        }
    }
    let mut out = Vec::new();
    for p in &pools {
        out.extend(p.assignment().unwrap().iter().flat_map(|x| x.to_le_bytes()));
        for i in 0..N {
            out.extend(p.state_hash(i).unwrap().get().to_le_bytes());
        }
    }
    out.extend(league.save().unwrap());
    (out, league.export().unwrap())
}

#[test]
fn test_persist_restart_mid_training_equals_no_restart() {
    let (a, ea) = run(false);
    let (b, eb) = run(true);
    assert_eq!(a, b);
    assert_eq!(ea, eb);
}

#[test]
fn test_persist_shared_league_round_trips_bit_for_bit() {
    let league = SharedLeague::new(shared_config(), heuristic_population()).unwrap();
    let mut pools: Vec<Pool> = ROLES
        .iter()
        .enumerate()
        .map(|(i, &r)| pool_for(&league, r, 1 + i as u64))
        .collect();
    for round in 0..5 {
        for (i, p) in pools.iter_mut().enumerate() {
            steps(p, round + i);
            p.epoch_barrier().unwrap();
        }
    }
    let bytes = league.save().unwrap();
    let loaded = SharedLeague::load(&bytes).unwrap();
    for (i, &r) in ROLES.iter().enumerate() {
        drop(loaded.member(r).unwrap());
        let _ = i;
    }
    assert_eq!(loaded.save().unwrap(), bytes);
    for role in 0..2 {
        assert_eq!(loaded.view(role).unwrap(), league.view(role).unwrap());
    }
    assert_eq!(loaded.export().unwrap(), league.export().unwrap());
}

#[test]
fn test_persist_league_round_trips_bit_for_bit() {
    let mut league = League::new(
        LeagueConfig {
            matchmaking: Matchmaking::Elo { temperature: 40.0 },
            rate_population: true,
            ..LeagueConfig::default()
        },
        population(),
    )
    .unwrap();
    let mut results = Results::new(2, 4).unwrap();
    for (i, id, o) in [
        (0, 0, Outcome::Win),
        (0, 1, Outcome::Loss),
        (1, 2, Outcome::Draw),
        (1, 0, Outcome::Win),
    ] {
        results.record(i, PolicyId(id), o).unwrap();
    }
    league.barrier(&mut results).unwrap();
    let bytes = league.save().unwrap();
    let loaded = League::load(&bytes).unwrap();
    assert_eq!(loaded, league);
    assert_eq!(loaded.save().unwrap(), bytes);
}

/// The pinned export text: key order, float form and escaping fixed.
#[test]
fn test_persist_league_export_text_is_pinned() {
    let league = League::new(
        LeagueConfig {
            matchmaking: Matchmaking::Uniform,
            ..LeagueConfig::default()
        },
        population(),
    )
    .unwrap();
    let ts = r#"{"mu":25,"sigma":8.333333333333334}"#;
    let entry = |id: u32, kind: &str| {
        format!(
            r#"{{"id":{id},"kind":{kind},"elo":1200,"trueskill":{ts},"roles":0,"retired":false}}"#
        )
    };
    let zero = r#"{"wins":0,"draws":0,"losses":0}"#;
    let expected = format!(
        r#"{{"round":0,"frozen":false,"roles":[{{"elo":1200,"trueskill":{ts},"counts":[{zero},{zero},{zero}],"snapshots":0}}],"population":[{},{},{}]}}"#,
        entry(0, r#""heuristic""#),
        entry(1, r#"{"external":7}"#),
        entry(2, r#"{"onnx":"07070707070707070707070707070707"}"#),
    );
    assert_eq!(league.export().unwrap(), expected);
}

#[test]
fn test_persist_refusals_are_typed() {
    let league = SharedLeague::new(shared_config(), heuristic_population()).unwrap();
    let mut a = pool_for(&league, 0, 1);
    let mut b = pool_for(&league, 1, 2);
    steps(&mut a, 0);
    a.epoch_barrier().unwrap();
    // Mid-round: pool a has submitted.
    assert!(matches!(league.save(), Err(LeagueError::Persist(_))));
    steps(&mut b, 1);
    b.epoch_barrier().unwrap();
    let bytes = league.save().unwrap();
    // Re-registration: another role at index 0 is refused.
    let loaded = SharedLeague::load(&bytes).unwrap();
    assert!(matches!(
        loaded.member(1),
        Err(LeagueError::InvalidConfig(_))
    ));
    // A barrier before every saved pool is back is refused.
    let state = a.snapshot_pool().unwrap();
    let view = state
        .context()
        .adaptive
        .league
        .as_ref()
        .unwrap()
        .view
        .clone();
    let mut a2 = pool_with(&loaded, 0, 3, view);
    a2.restore_pool(&state, None).unwrap();
    steps(&mut a2, 2);
    assert!(matches!(
        a2.epoch_barrier(),
        Err(EngineError::LeagueBarrier(e)) if e.reason.contains("register again")
    ));
    // A view that is not the one the role's pools hold.
    assert!(matches!(
        loaded.check_view(0, &loaded.view(1).unwrap()),
        Err(LeagueError::ViewMismatch { role: 0 })
    ));
    // Corruption: truncation, a flipped byte, the version, the kind.
    let bad = |b: &[u8]| matches!(SharedLeague::load(b), Err(LeagueError::Persist(_)));
    assert!(bad(&bytes[..bytes.len() - 1]));
    let mut flipped = bytes.clone();
    flipped[40] ^= 1;
    assert!(bad(&flipped));
    assert!(matches!(
        League::load(&bytes),
        Err(LeagueError::Persist("another league kind"))
    ));
    // A saved view digest the recomputed table does not reproduce, the
    // checksum resealed: refused (TR, E2 plan ruling 3). The last role's
    // view digest sits before its installed digest and the state block
    // (frozen, round, two members) and the checksum.
    let mut digest = bytes.clone();
    let at = digest.len() - 8 - (1 + 8 + 4 + 4 * 2) - 16;
    digest[at] ^= 1;
    let body = digest.len() - 8;
    let sum = tessera_rl_core::encoding::checksum(&digest[..body]);
    digest[body..].copy_from_slice(&sum.to_le_bytes());
    assert!(matches!(
        SharedLeague::load(&digest),
        Err(LeagueError::Persist(
            "a role's recomputed table is not the saved one"
        ))
    ));
}

proptest! {
    /// Any bytes, and any single-byte change of a valid file, load to a
    /// league or a typed error, never a panic.
    #[test]
    fn prop_persist_load_never_panics(
        bytes in proptest::collection::vec(any::<u8>(), 0..512),
        at in 0usize..4096,
        flip in 1u8..=255,
    ) {
        let _ = SharedLeague::load(&bytes);
        let _ = League::load(&bytes);
        let league = SharedLeague::new(shared_config(), population()).unwrap();
        let mut valid = league.save().unwrap();
        let i = at % valid.len();
        valid[i] ^= flip;
        let _ = SharedLeague::load(&valid);
        // The checksum is checked first: a changed byte is refused.
        prop_assert!(SharedLeague::load(&valid).is_err());
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! League replay (ADR-0088 section 8, erratum 3; #384 head D): a recorded
//! league pool replays bit-identically, barriers and freezes included, and
//! so does a recording that starts after barriers changed the view; the
//! pre-event checks refuse a mismatched pool before any write; seeks
//! restore the view and the assignments of their keyframe.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::league::{
    LeagueBarrierError, LeagueReport, LeagueResults, LeagueSubsystem, LeagueView, OpponentKind,
    sealed,
};
use tessera_rl_engine::replay::{Event, Recorder, Replay, ReplayError, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodeLimits, EngineError, EnvPool, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Matchmaking, PolicyKind, PoolLeague, Population, elo,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 8;
const A: usize = 2;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 16,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 20,
};

type Pool = EnvPool<PaddleDuel, NoMasks>;

/// A stand-in league subsystem whose table moves at every barrier (a real
/// league's Elo table does not move on draws, and short `PaddleDuel`
/// episodes end in draws): barrier `k` gives entry `k mod 3` half the
/// mass. Replay never calls it.
struct Rotating {
    kinds: Vec<OpponentKind>,
    digest: u64,
    barriers: u64,
}

impl Rotating {
    fn view(&self) -> LeagueView {
        let n = u64::try_from(self.kinds.len()).unwrap();
        let heavy = self.barriers % n;
        let unit = u64::MAX / (n + 1);
        let mut thresholds = Vec::new();
        let mut acc = 0u64;
        for i in 0..n {
            acc += if i == heavy { 2 * unit } else { unit };
            thresholds.push(acc);
        }
        *thresholds.last_mut().unwrap() = u64::MAX;
        LeagueView::new(
            self.kinds.clone(),
            thresholds,
            Some(u32::try_from(n - 1).unwrap()),
            self.digest,
        )
        .unwrap()
    }
}

impl sealed::Sealed for Rotating {}

impl LeagueSubsystem for Rotating {
    fn barrier(
        &mut self,
        _: LeagueResults<'_>,
    ) -> Result<(LeagueView, LeagueReport), LeagueBarrierError> {
        self.barriers += 1;
        Ok((self.view(), LeagueReport::new(0, 0, false)))
    }
}

fn league(kinds: &[PolicyKind]) -> Rotating {
    let mut population = Population::new();
    for &k in kinds {
        population
            .add(k, elo::INITIAL, Rating::initial(), 0)
            .unwrap();
    }
    let real = PoolLeague::new(
        League::new(
            LeagueConfig {
                matchmaking: Matchmaking::Uniform,
                ..LeagueConfig::default()
            },
            population,
        )
        .unwrap(),
    )
    .unwrap()
    .view()
    .unwrap();
    Rotating {
        kinds: real.kinds().to_vec(),
        digest: real.population_digest(),
        barriers: 0,
    }
}

const MIXED: [PolicyKind; 3] = [
    PolicyKind::Heuristic,
    PolicyKind::External { tag: 7 },
    PolicyKind::External { tag: 9 },
];

fn pool_of(kinds: &[PolicyKind]) -> Pool {
    let l = league(kinds);
    let view = l.view();
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        max_episode_steps: NonZeroU32::new(5),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(13))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    Pool::new(cfg, options)
        .unwrap()
        .with_league(&[1], view, Box::new(l))
        .unwrap()
}

fn pool() -> Pool {
    pool_of(&MIXED)
}

/// Step `t`'s rows: the learner and the external opponents.
fn rows(t: usize) -> Vec<i32> {
    (0..N as usize * A)
        .map(|k| i32::try_from((t * 7 + k * 3) % 3).unwrap())
        .collect()
}

fn hashes(p: &Pool) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

fn options() -> ReplayWriterOptions {
    ReplayWriterOptions {
        keyframe_every: 7,
        checkpoint_every: 3,
        outputs_hash: false,
    }
}

/// Runs `steps` steps on `p` with a barrier every 4, unrecorded.
fn run(p: &mut Pool, from: usize, steps: usize) {
    for t in from..from + steps {
        p.step(ActionInput {
            discrete: &rows(t),
            continuous: &[],
        })
        .unwrap();
        if t % 4 == 3 {
            p.epoch_barrier().unwrap();
        }
    }
}

/// Records `steps` steps from step `from` with a barrier every 4 and the
/// league frozen for the third; returns the file, the final hashes and
/// assignments.
fn record(p: &mut Pool, from: usize, steps: usize) -> (Vec<u8>, Vec<u64>, Vec<u32>) {
    let mut rec = Recorder::new(p, Vec::new(), options()).unwrap();
    for t in from..from + steps {
        rec.step(ActionInput {
            discrete: &rows(t),
            continuous: &[],
        })
        .unwrap();
        if t % 4 == 3 {
            rec.freeze_league((t / 4) % 3 == 2);
            rec.epoch_barrier().unwrap();
        }
    }
    rec.freeze_league(false);
    let bytes = rec.finish().unwrap();
    (bytes, hashes(p), p.assignment().unwrap().to_vec())
}

fn league_events(bytes: &[u8]) -> Vec<u64> {
    Replay::open(bytes, LIMITS)
        .unwrap()
        .events()
        .filter_map(|e| match e.unwrap() {
            (i, Event::League { .. }) => Some(i),
            _ => None,
        })
        .collect()
}

#[test]
fn test_league_replay_is_bit_identical_with_barriers_and_freezes() {
    let mut live = pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (bytes, live_hashes, live_assignment) = record(&mut live, 0, 40);
    // The first event, then at least one barrier that changed the view.
    let leagues = league_events(&bytes);
    assert_eq!(leagues.first(), Some(&0));
    assert!(leagues.len() > 1, "{leagues:?}");
    let mut replayed = pool();
    let summary = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut replayed)
        .unwrap();
    assert!(summary.verified());
    assert_eq!(hashes(&replayed), live_hashes);
    assert_eq!(replayed.assignment().unwrap(), live_assignment.as_slice());
    assert_eq!(replayed.league_view(), live.league_view());
}

/// A recording that starts after barriers changed the view, with episodes
/// in flight: the first keyframe's assignments, drawn from older views,
/// are replayed as they were, not redrawn.
#[test]
fn test_league_replay_started_after_barriers_keeps_old_assignments() {
    let prefix = |p: &mut Pool| {
        p.reset_all(Reseed::Keep).unwrap();
        run(p, 0, 22);
    };
    let mut live = pool();
    prefix(&mut live);
    let (bytes, live_hashes, live_assignment) = record(&mut live, 22, 30);
    // The replay pool has run the same prefix, so its view is the file's
    // first `League` event; its records are then overwritten.
    let mut replayed = pool();
    prefix(&mut replayed);
    replayed.reset_all(Reseed::Base(Seed::new(99))).unwrap();
    let summary = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut replayed)
        .unwrap();
    assert!(summary.verified());
    assert_eq!(hashes(&replayed), live_hashes);
    assert_eq!(replayed.assignment().unwrap(), live_assignment.as_slice());
}

#[test]
fn test_league_replay_refuses_a_mismatched_pool_before_any_write() {
    let mut live = pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (bytes, _, _) = record(&mut live, 0, 12);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    // Another population: `LeagueMismatch`, pool unchanged.
    let mut other = pool_of(&[PolicyKind::Heuristic, PolicyKind::External { tag: 7 }]);
    other.reset_all(Reseed::Keep).unwrap();
    let before = hashes(&other);
    assert!(matches!(
        replay.run(&mut other),
        Err(ReplayError::Engine(EngineError::LeagueMismatch { .. }))
    ));
    assert_eq!(hashes(&other), before);
    // The same population, another table: `League`, pool unchanged.
    let mut later = pool();
    later.reset_all(Reseed::Keep).unwrap();
    run(&mut later, 0, 20);
    assert_ne!(later.league_view(), pool().league_view());
    let before = hashes(&later);
    assert!(matches!(
        replay.run(&mut later),
        Err(ReplayError::Engine(EngineError::League { .. }))
    ));
    assert_eq!(hashes(&later), before);
}

/// A league file into a pool without a league, and a plain file into a
/// league pool, are refused before any write.
#[test]
fn test_league_replay_refuses_a_league_on_one_side_only() {
    let mut live = pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (bytes, _, _) = record(&mut live, 0, 6);
    let pool_options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        max_episode_steps: NonZeroU32::new(5),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(13))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    let mut plain = Pool::new(cfg, pool_options).unwrap();
    assert!(
        Replay::open(&bytes, LIMITS)
            .unwrap()
            .run(&mut plain)
            .is_err()
    );
    let mut rec = Recorder::new(&mut plain, Vec::new(), options()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let plain_bytes = rec.finish().unwrap();
    assert_eq!(league_events(&plain_bytes), Vec::<u64>::new());
    let mut with_league = pool();
    assert!(matches!(
        Replay::open(&plain_bytes, LIMITS)
            .unwrap()
            .run(&mut with_league),
        // The opponent slot is first caught by the policy map check.
        Err(ReplayError::Engine(
            EngineError::PolicyMapMismatch { .. } | EngineError::League { .. }
        ))
    ));
}

/// A seek to a periodic keyframe after a barrier restores that keyframe's
/// view and assignments; replaying on from it ends where the live pool did.
#[test]
fn test_league_replay_seek_restores_the_view_and_assignments() {
    let mut live = pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (bytes, live_hashes, live_assignment) = record(&mut live, 0, 40);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let leagues = league_events(&bytes);
    let target = *replay
        .keyframes()
        .unwrap()
        .iter()
        // A keyframe whose view is not the initial one: `Rotating` repeats
        // every three barriers.
        .rfind(|&&k| (leagues.iter().filter(|&&l| l < k).count() - 1) % 3 != 0)
        .unwrap();
    let mut replayed = pool();
    let mut cur = replay.cursor(&replayed).unwrap();
    replay.seek(&mut replayed, &mut cur, target).unwrap();
    // The view at the keyframe: one `Rotating` barrier per `League` event
    // before it, after the first.
    let before = leagues.iter().filter(|&&l| l < target).count();
    let mut expected = league(&MIXED);
    expected.barriers = u64::try_from(before - 1).unwrap();
    assert!(before > 1);
    assert_eq!(replayed.league_view(), Some(&expected.view()));
    while replay
        .advance(&mut replayed, &mut cur, |_| false)
        .unwrap()
        .is_some()
    {}
    let summary = replay.finish(cur).unwrap();
    assert_eq!(summary.started_at, Some(target));
    assert_eq!(hashes(&replayed), live_hashes);
    assert_eq!(replayed.assignment().unwrap(), live_assignment.as_slice());
}

const HASH: [u8; 16] = [7; 16];

/// A Rust stand-in ONNX policy: a byte of the row, mod 3.
struct FirstByte;

impl tessera_rl_engine::league::sealed::Sealed for FirstByte {}

impl tessera_rl_engine::league::OpponentPolicies for FirstByte {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        *content_hash == HASH
    }

    fn act(
        &mut self,
        _: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), tessera_rl_engine::league::OpponentPolicyError> {
        let row = rows.len() / n;
        for (j, a) in actions.iter_mut().enumerate() {
            *a = i64::from(rows[j * row + 3] % 3);
        }
        Ok(())
    }
}

const WITH_ONNX: [PolicyKind; 2] = [
    PolicyKind::External { tag: 7 },
    PolicyKind::Onnx { content_hash: HASH },
];

/// An ONNX league replays with the same policies (ADR-0088 section 8, in
/// ADR-0057's class; here a Rust stand-in), and a pool whose ONNX
/// opponents do not serve a recorded policy is refused before any write.
#[test]
fn test_league_replay_with_onnx_opponents() {
    let onnx_pool = || {
        pool_of(&WITH_ONNX)
            .with_onnx_opponents(Box::new(FirstByte))
            .unwrap()
    };
    let mut live = onnx_pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (bytes, live_hashes, _) = record(&mut live, 0, 24);
    let mut replayed = onnx_pool();
    let summary = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut replayed)
        .unwrap();
    assert!(summary.verified());
    assert_eq!(hashes(&replayed), live_hashes);
    // Without the ONNX opponents: refused, pool unchanged.
    let mut bare = pool_of(&WITH_ONNX);
    let before = hashes(&bare);
    assert!(matches!(
        Replay::open(&bytes, LIMITS).unwrap().run(&mut bare),
        Err(ReplayError::Engine(EngineError::OpponentPolicy(_)))
    ));
    assert_eq!(hashes(&bare), before);
}

/// The header's length: the shared fields, the extension (policy map, the
/// two flags, a `Fresh` source, the chunk size) and the checksum.
fn header_len(b: &[u8]) -> usize {
    let u16_at = |at: usize| usize::from(u16::from_le_bytes([b[at], b[at + 1]]));
    let mut at = 48;
    for _ in 0..2 {
        at += 2 + u16_at(at);
    }
    at += 2; // asset_count
    at += 2 + u16_at(at); // descriptor
    let config = usize::try_from(u32::from_le_bytes(b[at..at + 4].try_into().unwrap())).unwrap();
    let mut q = at + 4 + config + 12 + 4;
    q += 2 + u16_at(q) + 2;
    assert_eq!(b[q], 0, "a Fresh reset source");
    q + 1 + 4 + 8
}

/// The (offset, tag, body length) of every frame, the end event included.
fn frames(b: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut at = header_len(b);
    let mut out = Vec::new();
    while at < b.len() {
        let len =
            usize::try_from(u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap())).unwrap();
        out.push((at, b[at], len));
        at += 5 + len;
    }
    out
}

/// A keyframe's recorded assignment that differs from the replayed one is
/// a `Keyframe` divergence naming the instance (ADR-0088 erratum 3):
/// assignments are not in the state hash, so only this check sees them.
#[test]
fn test_league_replay_tampered_keyframe_assignment_diverges() {
    let mut live = pool();
    live.reset_all(Reseed::Keep).unwrap();
    let (mut bytes, _, _) = record(&mut live, 0, 20);
    let fr = frames(&bytes);
    // A periodic keyframe (tag 5) after the first; its body ends with the
    // last instance's assignment.
    let &(at, _, len) = fr.iter().skip(2).find(|&&(_, tag, _)| tag == 5).unwrap();
    let end = at + 5 + len;
    let old = u32::from_le_bytes(bytes[end - 4..end].try_into().unwrap());
    bytes[end - 4..end].copy_from_slice(&((old + 1) % 3).to_le_bytes());
    // Reseal the trailer's events checksum.
    let (trailer, _, _) = *fr.last().unwrap();
    let sum = tessera_rl_core::encoding::checksum(&bytes[header_len(&bytes)..trailer]);
    bytes[trailer + 13..trailer + 21].copy_from_slice(&sum.to_le_bytes());
    let mut replayed = pool();
    assert!(matches!(
        Replay::open(&bytes, LIMITS).unwrap().run(&mut replayed),
        Err(ReplayError::AssignmentDiverged {
            instance: 7,
            recorded,
            replayed: Some(replayed),
            ..
        }) if recorded == (old + 1) % 3 && replayed == old
    ));
    let err = Replay::open(&bytes, LIMITS)
        .unwrap()
        .run(&mut pool())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(&format!(
            "league assignment of instance 7: recorded policy {}, replayed Some({old})",
            (old + 1) % 3
        )),
        "{err}"
    );
}

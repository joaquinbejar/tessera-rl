// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stepped replay (issue #378): `run_with` sees every event in order and
//! ends where `run` ends, a break stops it unverified, a cursor resumes
//! where it stopped, seeking to a periodic keyframe then replaying to the
//! end reaches the recorded state, and seeking anywhere else is refused.

#![cfg(feature = "replay")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;
use core::ops::ControlFlow;

use common::{CountConfig, Counter};
use tessera_rl_engine::replay::{
    Event, Ids, PointKind, Recorder, Replay, ReplayError, ReplayPoint, ReplayWriterOptions,
};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

const N: u32 = 4;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

fn pool() -> Pool {
    let cfg = CountConfig {
        length: 6,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11));
    EnvPool::with_masks(cfg, options).unwrap()
}

/// Two agents per instance; agent 0 never takes its masked action 2.
fn actions(t: usize) -> Vec<i32> {
    (0..N as usize * 2)
        .map(|k| {
            let c = (t + k) % 3;
            let c = if k % 2 == 0 && c == 2 { 1 } else { c };
            i32::try_from(c).unwrap()
        })
        .collect()
}

fn step(rec: &mut Recorder<'_, Counter, Masked, Vec<u8>>, t: usize) {
    rec.step(ActionInput {
        discrete: &actions(t),
        continuous: &[],
    })
    .unwrap();
}

/// Resets, steps, a fork and a branched fork (state writes), with periodic
/// keyframes and checkpoints.
fn record() -> (Vec<u8>, Pool) {
    let mut p = pool();
    let options = ReplayWriterOptions {
        keyframe_every: 5,
        checkpoint_every: 3,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(7))).unwrap();
    for t in 0..4 {
        step(&mut rec, t);
    }
    rec.fork_branched(0, &[1, 2], None).unwrap();
    for t in 4..9 {
        step(&mut rec, t);
    }
    rec.fork(2, &[0]).unwrap();
    let saved: Vec<_> = (0..N).map(|i| rec.pool().snapshot(i).unwrap()).collect();
    for t in 9..12 {
        step(&mut rec, t);
    }
    // An all-ids state write: its keyframe has every id, so only the
    // state-write rule refuses a seek to it.
    rec.restore_all(&saved).unwrap();
    for t in 12..20 {
        step(&mut rec, t);
    }
    let bytes = rec.finish().unwrap();
    (bytes, p)
}

fn hashes(p: &Pool) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

/// The event index of the keyframe that completes the all-ids state write
/// (`restore_all`).
fn all_ids_write_keyframe(replay: &Replay<'_>) -> u64 {
    let events: Vec<(u64, Event<'_>)> = replay.events().map(Result::unwrap).collect();
    events
        .windows(2)
        .find_map(|w| match (&w[0].1, &w[1].1) {
            (Event::StateWrite { ids: Ids::All, .. }, Event::Keyframe { ids: Ids::All, .. }) => {
                Some(w[1].0)
            }
            _ => None,
        })
        .unwrap()
}

/// The event index of a keyframe that completes a state write.
fn write_keyframe(replay: &Replay<'_>) -> u64 {
    let events: Vec<(u64, Event<'_>)> = replay.events().map(Result::unwrap).collect();
    events
        .windows(2)
        .find_map(|w| match (&w[0].1, &w[1].1) {
            (Event::StateWrite { .. }, Event::Keyframe { .. }) => Some(w[1].0),
            _ => None,
        })
        .unwrap()
}

#[test]
fn test_replay_run_with_every_event_in_order_ends_as_run() {
    let (bytes, recorded) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut plain = pool();
    let expected = replay.run(&mut plain).unwrap();

    let mut p = pool();
    let mut seen: Vec<ReplayPoint> = Vec::new();
    let summary = replay
        .run_with(&mut p, |point, _| {
            seen.push(point);
            ControlFlow::Continue(())
        })
        .unwrap();
    let indices: Vec<u64> = seen.iter().map(|p| p.event_index).collect();
    assert_eq!(indices, (0..replay.event_count()).collect::<Vec<_>>());
    assert_eq!(seen[0].kind, PointKind::Keyframe);
    let calls = seen.iter().filter(|p| p.kind == PointKind::Call).count() as u64;
    assert_eq!(calls, summary.calls);
    assert_eq!(summary, expected);
    assert!(summary.verified());
    assert_eq!(summary.stopped_at, None);
    assert_eq!(hashes(&p), hashes(&plain));
    assert_eq!(hashes(&p), hashes(&recorded));
}

#[test]
fn test_replay_run_with_break_stops_early_unverified() {
    let (bytes, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut p = pool();
    let summary = replay
        .run_with(&mut p, |point, pool| {
            // The pool is readable between events.
            pool.state_hash(0).unwrap();
            if point.event_index == 4 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
    assert_eq!(summary.stopped_at, Some(5));
    assert_eq!(summary.events, 5);
    assert!(!summary.verified());
}

#[test]
fn test_replay_cursor_advance_resumes_and_finishes_verified() {
    let (bytes, recorded) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut p = pool();
    let mut cur = replay.cursor(&p).unwrap();
    let mut stops = Vec::new();
    while let Some(point) = replay
        .advance(&mut p, &mut cur, |pt| pt.kind == PointKind::Call)
        .unwrap()
    {
        assert_eq!(point.kind, PointKind::Call);
        stops.push(point.event_index);
    }
    assert!(cur.is_done());
    // Strictly increasing: nothing is applied twice.
    assert!(stops.windows(2).all(|w| w[0] < w[1]));
    // Advancing a done cursor applies nothing.
    assert_eq!(replay.advance(&mut p, &mut cur, |_| true).unwrap(), None);
    let summary = replay.finish(cur).unwrap();
    assert!(summary.verified());
    assert_eq!(stops.len() as u64, summary.calls);
    assert_eq!(hashes(&p), hashes(&recorded));
}

#[test]
fn test_replay_cursor_finished_early_is_stopped() {
    let (bytes, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut p = pool();
    let mut cur = replay.cursor(&p).unwrap();
    let point = replay
        .advance(&mut p, &mut cur, |pt| pt.kind == PointKind::Call)
        .unwrap()
        .unwrap();
    let summary = replay.finish(cur).unwrap();
    assert_eq!(summary.stopped_at, Some(point.event_index + 1));
    assert!(!summary.verified());
}

#[test]
fn test_replay_seek_periodic_keyframe_then_end_matches_full_run() {
    let (bytes, recorded) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let keyframes = replay.keyframes().unwrap();
    assert_eq!(keyframes[0], 0);
    let target = *keyframes.iter().rfind(|&&k| k > 0).unwrap();
    assert!(!keyframes.contains(&write_keyframe(&replay)));

    let mut p = pool();
    let mut cur = replay.cursor(&p).unwrap();
    replay.seek(&mut p, &mut cur, target).unwrap();
    assert_eq!(cur.next_event(), target + 1);
    assert_eq!(replay.advance(&mut p, &mut cur, |_| false).unwrap(), None);
    let summary = replay.finish(cur).unwrap();
    // Reached the end, but started from a seek: not verified.
    assert_eq!(summary.stopped_at, None);
    assert_eq!(summary.started_at, Some(target));
    assert!(!summary.verified());
    assert_eq!(hashes(&p), hashes(&recorded));
}

#[test]
fn test_replay_seek_backwards_after_the_end_replays_again() {
    let (bytes, recorded) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let keyframes = replay.keyframes().unwrap();
    let mut p = pool();
    let mut cur = replay.cursor(&p).unwrap();
    assert_eq!(replay.advance(&mut p, &mut cur, |_| false).unwrap(), None);
    replay.seek(&mut p, &mut cur, keyframes[1]).unwrap();
    assert!(!cur.is_done());
    assert_eq!(replay.advance(&mut p, &mut cur, |_| false).unwrap(), None);
    let summary = replay.finish(cur).unwrap();
    assert_eq!(summary.started_at, Some(keyframes[1]));
    assert!(!summary.verified());
    assert_eq!(hashes(&p), hashes(&recorded));
}

#[test]
fn test_replay_seek_state_write_keyframe_and_other_events_refused() {
    let (bytes, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut p = pool();
    let mut cur = replay.cursor(&p).unwrap();
    let refused = |r: Result<(), ReplayError>| {
        assert!(matches!(r, Err(ReplayError::Unsupported { .. })), "{r:?}");
    };
    refused(replay.seek(&mut p, &mut cur, write_keyframe(&replay)));
    refused(replay.seek(&mut p, &mut cur, all_ids_write_keyframe(&replay)));
    assert!(
        !replay
            .keyframes()
            .unwrap()
            .contains(&all_ids_write_keyframe(&replay))
    );
    // Event 1 is the first reset, not a keyframe.
    refused(replay.seek(&mut p, &mut cur, 1));
    refused(replay.seek(&mut p, &mut cur, replay.event_count()));
    refused(replay.seek(&mut p, &mut cur, u64::MAX));
    // The cursor did not move.
    assert_eq!(cur.next_event(), 0);
}

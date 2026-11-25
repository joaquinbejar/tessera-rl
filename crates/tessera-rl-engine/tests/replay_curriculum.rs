// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replays of curriculum runs (ADR-0081 item 7, #359 head C): tag 7 events
//! set the targets at their call positions, a replay never runs the
//! curriculum, seeks rebuild the targets, and the pool must match the
//! header's `has_curriculum`.

#![cfg(all(feature = "replay", not(loom)))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    Adaptive, AgentSlot, Environment, EpisodeEnd, InvariantViolation, ObsLeaf, Observation,
    Persist, RecordHeader, SpaceError, StateHash, StepOut, Tier, Validate,
};
use tessera_rl_engine::replay::{Recorder, Replay, ReplayError, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, AdaptiveTargets, Curriculum, DecodeLimits, EnvPool, EpochReport, NoMasks,
    PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 1 << 24,
    max_decoded_bytes: 1 << 30,
    max_host_record_bytes: 1 << 26,
};

#[derive(EnvConfig, Debug, Clone)]
struct LevelConfig {
    #[config(default = 1, adaptive)]
    level: u32,
    #[config(default = 3)]
    length: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct LevelState {
    level: u32,
    steps: u32,
}

#[derive(Observation)]
struct LevelObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    level: ObsLeaf<u32>,
}

/// The reward of every step is the episode's level, read at reset.
struct Level;

impl Environment for Level {
    type State = LevelState;
    type Config = LevelConfig;
    type Obs = LevelObs;
    type Action = ();
    type Header = RecordHeader<1, LevelConfigAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;

    fn observation_space(cfg: &LevelConfig) -> Result<Space, SpaceError> {
        LevelObs::space(cfg)
    }

    fn action_space(_: &LevelConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(cfg: &LevelConfig, _: &mut StepRng, state: &mut LevelState) {
        *state = LevelState {
            level: cfg.level,
            steps: 0,
        };
    }

    fn reset_with(
        _: &LevelConfig,
        adaptive: &Adaptive<Self>,
        _: &mut StepRng,
        state: &mut LevelState,
    ) {
        *state = LevelState {
            level: adaptive.level,
            steps: 0,
        };
    }

    fn step(
        cfg: &LevelConfig,
        state: &mut LevelState,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        #[expect(clippy::cast_precision_loss, reason = "small test values")]
        out.set_reward(AgentSlot::FIRST, state.level as f32);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &LevelConfig,
        state: &LevelState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LevelObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(LevelObs::LEVEL) {
            leaf.set(0, state.level);
        }
    }
}

impl Validate for Level {
    fn validate(_: &LevelConfig, _: &LevelState) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

impl Persist for Level {
    const ENV_ID: &'static str = "tests/Level-v1";
    const ENV_VERSION: u32 = 1;
}

/// Raises each slot's level by one when episodes ended.
struct Raise;

impl Curriculum<Level> for Raise {
    const ID: &'static str = "tests/raise-v1";
    type State = u32;

    fn update(
        &mut self,
        report: &EpochReport,
        state: &mut u32,
        targets: &mut AdaptiveTargets<'_, Level>,
    ) {
        *state += 1;
        if report.episodes.count == 0 {
            return;
        }
        // Each slot from its own target, so slot 2's explicit one survives.
        for id in 0..N {
            let mut next = *targets.get(id).unwrap();
            next.level += 1;
            assert!(targets.set(id, next));
        }
    }
}

type Pool = EnvPool<Level, NoMasks>;

const N: u32 = 4;

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(3))
}

fn hashes(p: &Pool) -> Vec<StateHash> {
    (0..N).map(|i| p.state_hash(i).unwrap()).collect()
}

fn targets_only() -> Pool {
    Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_targets()
        .unwrap()
}

/// A curriculum run: barriers that raise the levels, and one explicit
/// target for slot 2. Returns the file, the final hashes and targets.
fn record() -> (Vec<u8>, Vec<StateHash>, Vec<u32>) {
    let mut pool = Pool::new(LevelConfig::default(), options())
        .unwrap()
        .with_curriculum(Raise)
        .unwrap();
    let writer = ReplayWriterOptions {
        keyframe_every: 6,
        checkpoint_every: 4,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut pool, Vec::new(), writer).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let actions = [0; N as usize];
    for k in 0..30 {
        rec.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
        if k % 5 == 4 {
            rec.epoch_barrier().unwrap();
        }
        if k == 12 {
            let mut v = *rec.pool().targets().unwrap().get(2).unwrap();
            v.level = 40;
            rec.set_targets(&[(2, v)]).unwrap();
        }
    }
    let bytes = rec.finish().unwrap();
    let levels = pool.targets().unwrap().iter().map(|a| a.level).collect();
    (bytes, hashes(&pool), levels)
}

#[test]
fn test_replay_curriculum_reproduces_the_run_through_tag_7_without_the_curriculum() {
    let (bytes, want, levels) = record();
    assert!(
        levels.iter().all(|&l| l > 2),
        "the curriculum moved: {levels:?}"
    );
    assert!(levels[2] > 40, "slot 2's explicit target, then raised");
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut pool = targets_only();
    let summary = replay.run(&mut pool).unwrap();
    assert!(summary.verified(), "{summary:?}");
    assert_eq!(hashes(&pool), want);
    let got: Vec<u32> = pool.targets().unwrap().iter().map(|a| a.level).collect();
    assert_eq!(got, levels);
}

#[test]
fn test_replay_curriculum_build_pool_checks_the_callers_targets() {
    let (bytes, want, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    // The caller builds the pool; without targets it is refused.
    assert!(matches!(
        replay.build_pool(Pool::new),
        Err(ReplayError::Persist(_))
    ));
    let mut pool = replay
        .build_pool(|cfg, options| Pool::new(cfg, options)?.with_targets())
        .unwrap();
    assert!(replay.run(&mut pool).unwrap().verified());
    assert_eq!(hashes(&pool), want);
}

#[test]
fn test_replay_curriculum_seek_rebuilds_the_targets_of_the_keyframe() {
    let (bytes, want, levels) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let keyframes = replay.keyframes().unwrap();
    assert!(keyframes.len() > 2, "{keyframes:?}");
    for &k in &keyframes[1..] {
        // A fresh pool: its targets are the config's, whatever the file.
        let mut pool = targets_only();
        let mut cur = replay.cursor(&pool).unwrap();
        replay.seek(&mut pool, &mut cur, k).unwrap();
        assert_eq!(
            replay.advance(&mut pool, &mut cur, |_| false).unwrap(),
            None
        );
        assert_eq!(hashes(&pool), want, "seek to {k}");
        let got: Vec<u32> = pool.targets().unwrap().iter().map(|a| a.level).collect();
        assert_eq!(got, levels, "seek to {k}");
    }
}

#[test]
fn test_replay_curriculum_seek_targets_equal_a_straight_run_at_the_keyframe() {
    let (bytes, _, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    for &k in &replay.keyframes().unwrap() {
        // Straight: every event up to and including keyframe `k`.
        let mut straight = targets_only();
        let mut cur = replay.cursor(&straight).unwrap();
        let stopped = replay
            .advance(&mut straight, &mut cur, |pt| pt.event_index >= k)
            .unwrap();
        assert_eq!(stopped.map(|p| p.event_index), Some(k));
        // Seek: straight to `k`.
        let mut seeked = targets_only();
        let mut cur = replay.cursor(&seeked).unwrap();
        replay.seek(&mut seeked, &mut cur, k).unwrap();
        let levels = |p: &Pool| {
            p.targets()
                .unwrap()
                .iter()
                .map(|a| a.level)
                .collect::<Vec<_>>()
        };
        assert_eq!(levels(&seeked), levels(&straight), "keyframe {k}");
        assert_eq!(hashes(&seeked), hashes(&straight), "keyframe {k}");
    }
}

#[test]
fn test_replay_curriculum_needs_a_pool_with_targets_and_writes_the_header_flag() {
    let (bytes, _, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut plain = Pool::new(LevelConfig::default(), options()).unwrap();
    assert!(matches!(
        replay.run(&mut plain),
        Err(ReplayError::Persist(_))
    ));
    // A pool of the same adaptive environment without targets records a
    // file without a curriculum, which a pool with targets refuses.
    let mut fixed = Pool::new(LevelConfig::default(), options()).unwrap();
    let mut rec = Recorder::new(&mut fixed, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let plain_file = rec.finish().unwrap();
    let replay = Replay::open(&plain_file, LIMITS).unwrap();
    assert!(matches!(
        replay.run(&mut targets_only()),
        Err(ReplayError::Persist(_))
    ));
    let mut again = Pool::new(LevelConfig::default(), options()).unwrap();
    assert!(replay.run(&mut again).unwrap().verified());
}

#[test]
fn test_replay_curriculum_initial_tag_7_carries_the_targets_at_recording_start() {
    // Targets changed before the recorder: only the initial tag 7 tells the
    // replay, whose pool starts from the config's.
    let mut pool = targets_only();
    let mut v = *pool.targets().unwrap().get(0).unwrap();
    v.level = 9;
    pool.set_targets(&(0..N).map(|i| (i, v)).collect::<Vec<_>>())
        .unwrap();
    let mut rec = Recorder::new(&mut pool, Vec::new(), ReplayWriterOptions::default()).unwrap();
    rec.reset_all(Reseed::Keep).unwrap();
    let actions = [0; N as usize];
    for _ in 0..4 {
        rec.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    }
    let bytes = rec.finish().unwrap();
    let want = hashes(&pool);
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let mut again = targets_only();
    assert!(replay.run(&mut again).unwrap().verified());
    assert_eq!(hashes(&again), want);
    assert!(again.targets().unwrap().iter().all(|a| a.level == 9));
}

#[test]
fn test_replay_curriculum_tag_7_of_another_value_length_is_malformed() {
    let (bytes, _, _) = record();
    let replay = Replay::open(&bytes, LIMITS).unwrap();
    let offsets = replay.event_offsets().unwrap();
    // The event bytes end where the 21-byte trailer starts.
    let trailer = bytes.len() - 21;
    let frame_len = |b: &[u8], at: usize| {
        usize::try_from(u32::from_le_bytes(b[at + 1..at + 5].try_into().unwrap())).unwrap()
    };
    let last = *offsets.last().unwrap();
    // The event bytes' start: where the last frame ends at the trailer.
    let start = (0..trailer)
        .find(|&s| s + last + 5 <= trailer && s + last + 5 + frame_len(&bytes, s + last) == trailer)
        .unwrap();
    let mut events = Vec::new();
    let mut forged = false;
    for (i, &off) in offsets.iter().enumerate() {
        let at = start + off;
        let tag = bytes[at];
        let mut body = bytes[at + 5..at + 5 + frame_len(&bytes, at)].to_vec();
        if tag == 7 && !forged && i > 0 {
            // One extra byte after every value: value_len becomes ENCODED_LEN + 1.
            let count = usize::try_from(u32::from_le_bytes(body[..4].try_into().unwrap())).unwrap();
            let len = (body.len() - 4) / count - 4;
            let mut out = body[..4].to_vec();
            for e in body[4..].chunks(4 + len) {
                out.extend_from_slice(e);
                out.push(0);
            }
            body = out;
            forged = true;
        }
        events.push((tag, body));
    }
    assert!(forged);
    let mut out = bytes[..start].to_vec();
    for (tag, body) in &events {
        out.push(*tag);
        out.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
        out.extend_from_slice(body);
    }
    let sum = tessera_rl_core::encoding::checksum(&out[start..]);
    out.push(0);
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&u64::try_from(events.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&sum.to_le_bytes());
    let replay = Replay::open(&out, LIMITS).unwrap();
    assert!(matches!(
        replay.run(&mut targets_only()),
        Err(ReplayError::Malformed {
            field: "adaptive value length",
            ..
        })
    ));
}

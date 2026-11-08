// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mountain car through the engine: spaces, the reset distribution, the
//! dynamics against a transcription of Gymnasium's step written for these
//! tests, rewards, termination, truncation, the scripted baseline and
//! import validation by phase and end flags.

#![expect(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::needless_range_loop,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::mountaincar::MAX_EPISODE_STEPS;
use tessera_rl_classic::mountaincar::physics::{MAX_POSITION, MAX_SPEED, MIN_POSITION};
use tessera_rl_classic::{MountainCar, MountainCarConfig, MountainCarState};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{Bool8, Environment};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::math;

type Pool = EnvPool<MountainCar>;

fn pool_with(n: u32, seed: u64, auto_reset: AutoReset, limit: u32) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::new(MountainCarConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(seed: u64) -> Pool {
    pool_with(1, seed, AutoReset::Disabled, 1_000_000)
}

fn step(p: &mut Pool, actions: &[i32]) {
    p.step(ActionInput {
        discrete: actions,
        continuous: &[],
    })
    .unwrap();
}

fn state(p: &Pool, i: usize) -> [f64; 2] {
    p.state(u32::try_from(i).unwrap()).unwrap().s
}

/// Gymnasium 1.4.0 `MountainCarEnv.step`, line by line, in plain f64.
fn reference(s: [f64; 2], action: i32) -> [f64; 2] {
    let (mut position, mut velocity) = (s[0], s[1]);
    velocity += f64::from(action - 1) * 0.001 + math::cos(3.0 * position) * (-0.0025);
    velocity = velocity.clamp(-0.07, 0.07); // np.clip
    position += velocity;
    position = position.clamp(-1.2, 0.6); // np.clip
    if position == -1.2 && velocity < 0.0 {
        velocity = 0.0;
    }
    [position, velocity]
}

#[test]
fn test_mountaincar_spaces_match_gymnasium() {
    let cfg = MountainCarConfig::default();
    assert_eq!(
        MountainCar::action_space(&cfg).unwrap(),
        Space::Discrete { n: 3 }
    );
    let Space::Box {
        shape,
        dtype,
        bounds: BoxBounds::Float { low, high },
    } = MountainCar::observation_space(&cfg).unwrap()
    else {
        panic!("not a float box")
    };
    assert_eq!((shape, dtype), (vec![2], DType::F32));
    assert_eq!(low, vec![f64::from(-1.2f32), f64::from(-0.07f32)]);
    assert_eq!(high, vec![f64::from(0.6f32), f64::from(0.07f32)]);
}

#[test]
fn test_mountaincar_reset_is_uniform_on_the_valley_floor_at_rest() {
    let p = pool_with(4096, 7, AutoReset::Disabled, 200);
    let mut low = f64::MAX;
    let mut high = f64::MIN;
    for i in 0..4096 {
        let [position, velocity] = state(&p, i);
        assert!((-0.6..-0.4).contains(&position), "{position}");
        assert_eq!(velocity.to_bits(), 0);
        low = low.min(position);
        high = high.max(position);
    }
    assert!(low < -0.599 && high > -0.401, "{low} {high}");
}

#[test]
fn test_mountaincar_step_matches_the_reference_transcription() {
    let n = 64;
    let mut p = pool_with(n, 11, AutoReset::Disabled, 1_000_000);
    let mut seed = 0x9e37_79b9_u32;
    for _ in 0..400 {
        let actions: Vec<i32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                i32::try_from(seed >> 30).unwrap() % 3
            })
            .collect();
        let before: Vec<_> = (0..n as usize).map(|i| state(&p, i)).collect();
        let done: Vec<bool> = (0..n as usize)
            .map(|i| p.outputs().unwrap().episode_terminated()[i].get())
            .collect();
        if done.iter().any(|d| *d) {
            break;
        }
        step(&mut p, &actions);
        for i in 0..n as usize {
            let expect = reference(before[i], actions[i]);
            assert_eq!(state(&p, i).map(f64::to_bits), expect.map(f64::to_bits));
            assert_eq!(p.outputs().unwrap().rewards()[i], -1.0);
        }
    }
}

#[test]
fn test_mountaincar_right_edge_clips_the_position() {
    // Valid and not terminal (moving left past the goal), the step turns
    // it right and past 0.6: the clip holds it there, at the goal.
    let mut p = pool(1);
    let s = [0.5999, -0.0001];
    p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
    step(&mut p, &[2]);
    assert_eq!(state(&p, 0)[0], MAX_POSITION);
    assert_eq!(
        state(&p, 0).map(f64::to_bits),
        reference(s, 2).map(f64::to_bits)
    );
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
}

#[test]
fn test_mountaincar_terminates_at_the_goal_and_truncates_at_the_limit() {
    let mut p = pool(1);
    let mut s = *p.state(0).unwrap();
    s.s = [0.49, 0.03];
    let rec = DecodedRecord::from_parts(*DecodedRecord::from(p.snapshot(0).unwrap()).header(), s)
        .unwrap();
    p.import(&[0], &[rec]).unwrap();
    step(&mut p, &[2]);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
    let mut t = pool_with(1, 1, AutoReset::Disabled, MAX_EPISODE_STEPS);
    for _ in 0..MAX_EPISODE_STEPS {
        step(&mut t, &[1]);
    }
    let out = t.outputs().unwrap();
    assert!(out.episode_truncated()[0].get() && !out.episode_terminated()[0].get());
}

#[test]
fn test_mountaincar_baseline_reaches_the_goal_within_the_limit() {
    let n = 256;
    let mut p = pool_with(n, 3, AutoReset::NextStep, MAX_EPISODE_STEPS);
    let mut ended = vec![0u32; n as usize];
    for t in 1..=MAX_EPISODE_STEPS {
        let actions: Vec<i32> = (0..n as usize)
            .map(|i| if state(&p, i)[1] >= 0.0 { 2 } else { 0 })
            .collect();
        step(&mut p, &actions);
        let out = p.outputs().unwrap();
        for i in 0..n as usize {
            if ended[i] == 0 && out.episode_terminated()[i].get() {
                ended[i] = t;
            }
        }
        if ended.iter().all(|e| *e > 0) {
            break;
        }
    }
    assert!(
        ended.iter().all(|e| *e > 0),
        "every first episode reaches the goal"
    );
    let mean = f64::from(ended.iter().sum::<u32>()) / f64::from(n);
    assert!(mean < 130.0, "mean episode length {mean}");
}

// -- import validation (design note 4.2) -----------------------------------

fn record(p: &Pool, s: [f64; 2], pending: bool, end_kind: u8) -> DecodedRecord<MountainCar> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.engine.pending_reset = Bool8::new(pending);
    header.engine.end_kind = end_kind;
    DecodedRecord::from_parts(header, MountainCarState { s }).unwrap()
}

fn code(r: Result<(), EngineError>) -> &'static str {
    match r {
        Err(EngineError::Invariant { violation, .. }) => violation.code,
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_mountaincar_import_refuses_states_out_of_the_domain() {
    let mut p = pool(5);
    let before = p.state_hash(0).unwrap();
    for (s, expect) in [
        ([-0.5, f64::INFINITY], "a non-finite component"),
        ([MAX_POSITION.next_up(), 0.0], "a position out of range"),
        ([MIN_POSITION.next_down(), 0.0], "a position out of range"),
        ([-0.5, MAX_SPEED.next_up()], "a velocity out of range"),
        ([-0.5, (-MAX_SPEED).next_down()], "a velocity out of range"),
        ([MIN_POSITION, -1e-9], "a leftward velocity at the wall"),
    ] {
        assert_eq!(
            code(p.import(&[0], &[record(&p, s, false, 0)])),
            expect,
            "{s:?}"
        );
    }
    assert_eq!(
        p.state_hash(0).unwrap(),
        before,
        "a refused import changes nothing"
    );
    for s in [
        [MIN_POSITION, 0.0],
        [MIN_POSITION, -0.0],
        [MAX_POSITION, -MAX_SPEED],
        [0.0, MAX_SPEED],
    ] {
        p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
    }
}

#[test]
fn test_mountaincar_import_checks_the_terminal_predicate_by_end_flags() {
    let mut p = pool(5);
    let goal = [0.5, 0.0];
    let road = [-0.5, 0.01];
    // Running: never at the goal.
    assert_eq!(
        code(p.import(&[0], &[record(&p, goal, false, 0)])),
        "a terminal state not reported as terminated"
    );
    p.import(&[0], &[record(&p, road, false, 0)]).unwrap();
    // Terminated, alone or at the limit: at the goal only.
    for kind in [1, 3] {
        p.import(&[0], &[record(&p, goal, true, kind)]).unwrap();
        assert_eq!(
            code(p.import(&[0], &[record(&p, road, true, kind)])),
            "a terminated record in a non-terminal state"
        );
    }
    // Truncated only: any valid state that is not at the goal.
    p.import(&[0], &[record(&p, road, true, 2)]).unwrap();
    assert_eq!(
        code(p.import(&[0], &[record(&p, goal, true, 2)])),
        "a terminal state not reported as terminated"
    );
}

#[test]
fn test_mountaincar_import_before_the_first_reset_takes_the_exact_default() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let fresh = Pool::new(MountainCarConfig::default(), options).unwrap();
    let rec = DecodedRecord::from(fresh.snapshot(0).unwrap());
    let mut q = Pool::new(MountainCarConfig::default(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    for s in [[-0.0, 0.0], [0.0, -0.0], [-0.5, 0.0]] {
        let bad = DecodedRecord::from_parts(*rec.header(), MountainCarState { s }).unwrap();
        assert_eq!(
            code(q.import(&[0], &[bad])),
            "a state other than the initial value before the first reset"
        );
    }
}

#[test]
fn test_mountaincar_import_requires_the_agent_alive_and_acting() {
    let mut p = pool(5);
    for (alive, acting) in [(true, false), (false, false)] {
        let good = DecodedRecord::from(p.snapshot(0).unwrap());
        let mut header = *good.header();
        header.agents[0].alive = Bool8::new(alive);
        header.agents[0].acting = Bool8::new(acting);
        let rec = DecodedRecord::from_parts(header, *good.state()).unwrap();
        assert!(p.import(&[0], &[rec]).is_err());
    }
}

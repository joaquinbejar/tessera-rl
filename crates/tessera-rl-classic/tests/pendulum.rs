// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pendulum through the engine: spaces, reset, the dynamics against a
//! transcription, action refusal, truncation, the baseline and import
//! validation by phase and end flags.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::pendulum::MAX_EPISODE_STEPS;
use tessera_rl_classic::pendulum::physics::{self, MAX_SPEED};
use tessera_rl_classic::{Pendulum, PendulumConfig, PendulumState};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::math;
use tessera_rl_num::rng::{RngAuthority, StepRng};

type Pool = EnvPool<Pendulum>;

fn pool_with(n: u32, seed: u64, auto_reset: AutoReset, limit: u32) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::new(PendulumConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(seed: u64) -> Pool {
    pool_with(1, seed, AutoReset::Disabled, 1_000_000)
}

fn step(p: &mut Pool, actions: &[f32]) -> Result<(), EngineError> {
    p.step(ActionInput {
        discrete: &[],
        continuous: actions,
    })
    .map(|_| ())
}

fn state(p: &Pool, i: u32) -> [f64; 2] {
    p.state(i).unwrap().s
}

/// Gymnasium 1.4.0 `PendulumEnv.step` in plain arithmetic, from the source.
fn reference(s: [f64; 2], u: f32) -> [f64; 2] {
    let (th, thdot) = (s[0], s[1]);
    let newthdot =
        thdot + (3.0 * 10.0 / (2.0 * 1.0) * math::sin(th) + f64::from(3.0f32 / 1.0 * u)) * 0.05;
    let newthdot = newthdot.clamp(-8.0, 8.0);
    [th + newthdot * 0.05, newthdot]
}

#[test]
fn test_pendulum_spaces_match_gymnasium() {
    let cfg = PendulumConfig::default();
    let Space::Box {
        shape,
        dtype,
        bounds: BoxBounds::Float { low, high },
    } = Pendulum::action_space(&cfg).unwrap()
    else {
        panic!("not a float box")
    };
    assert_eq!(
        (shape, dtype, low, high),
        (vec![1], DType::F32, vec![-2.0], vec![2.0])
    );
    let Space::Box {
        shape,
        bounds: BoxBounds::Float { low, high },
        ..
    } = Pendulum::observation_space(&cfg).unwrap()
    else {
        panic!("not a float box")
    };
    assert_eq!(
        (shape, low, high),
        (vec![3], vec![-1.0, -1.0, -8.0], vec![1.0, 1.0, 8.0])
    );
}

#[test]
fn test_pendulum_reset_is_within_the_gymnasium_ranges() {
    let p = pool_with(256, 3, AutoReset::Disabled, 1_000_000);
    for i in 0..256 {
        let [theta, speed] = state(&p, i);
        assert!((-core::f64::consts::PI..=core::f64::consts::PI).contains(&theta));
        assert!((-1.0..=1.0).contains(&speed));
    }
}

#[test]
fn test_pendulum_step_matches_the_reference_transcription() {
    let mut p = pool(7);
    let mut s = state(&p, 0);
    for k in 0..400u16 {
        let u = -2.0 + f32::from(k % 41) * 0.1;
        step(&mut p, &[u]).unwrap();
        s = reference(s, u);
        assert_eq!(state(&p, 0).map(f64::to_bits), s.map(f64::to_bits));
    }
}

#[test]
fn test_pendulum_step_reports_the_reward_of_the_state_before_it() {
    let mut p = pool(7);
    let s = state(&p, 0);
    step(&mut p, &[1.5]).unwrap();
    #[expect(clippy::cast_possible_truncation, reason = "the public reward")]
    let want = physics::reward(s, 1.5) as f32;
    assert_eq!(p.outputs().unwrap().rewards()[0].to_bits(), want.to_bits());
}

#[test]
fn test_pendulum_refuses_invalid_actions_and_changes_nothing() {
    let mut p = pool(7);
    let before = p.state_hash(0).unwrap();
    for bad in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0f32.next_up(),
        (-2.0f32).next_down(),
    ] {
        assert!(step(&mut p, &[bad]).is_err(), "{bad}");
        assert_eq!(p.state_hash(0).unwrap(), before);
    }
    step(&mut p, &[2.0]).unwrap();
    step(&mut p, &[-2.0]).unwrap();
}

#[test]
fn test_pendulum_truncates_at_the_limit_and_never_terminates() {
    let mut p = pool_with(1, 5, AutoReset::Disabled, MAX_EPISODE_STEPS);
    for t in 1..=MAX_EPISODE_STEPS {
        step(&mut p, &[0.0]).unwrap();
        let out = p.outputs().unwrap();
        assert!(!out.episode_terminated()[0].get());
        assert_eq!(out.episode_truncated()[0].get(), t == MAX_EPISODE_STEPS);
    }
}

#[test]
fn test_pendulum_huge_angles_stay_closed_and_deterministic() {
    for theta in [f64::MAX, -f64::MAX, 1e300, -1e300, -0.0] {
        let mut p = pool(5);
        p.import(&[0], &[record(&p, [theta, 8.0], false, 0)])
            .unwrap();
        step(&mut p, &[2.0]).unwrap();
        assert!(state(&p, 0).iter().all(|v| v.is_finite()));
        assert!(p.outputs().unwrap().rewards()[0].is_finite());
    }
}

fn heuristic_rng() -> StepRng {
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

fn mean_return(n: u32, baseline: bool) -> f64 {
    let mut p = pool_with(n, 11, AutoReset::Disabled, MAX_EPISODE_STEPS);
    let cfg = PendulumConfig::default();
    let mut total = 0.0;
    let mut draw = 77u64;
    for _ in 0..MAX_EPISODE_STEPS {
        let actions: Vec<f32> = (0..n)
            .map(|i| {
                if baseline {
                    let mut c = [0.0f32];
                    let mut w = tessera_rl_core::io::ActionWriter::new(&mut [], &mut c);
                    Pendulum::act(
                        &cfg,
                        p.state(i).unwrap(),
                        AgentSlot::FIRST,
                        &mut heuristic_rng(),
                        &mut w,
                    );
                    c[0]
                } else {
                    draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                    f32::from(u16::try_from((draw >> 40) & 0x3ff).unwrap()) / 256.0 - 2.0
                }
            })
            .collect();
        step(&mut p, &actions).unwrap();
        total += p
            .outputs()
            .unwrap()
            .rewards()
            .iter()
            .map(|r| f64::from(*r))
            .sum::<f64>();
    }
    total / f64::from(n)
}

#[test]
fn test_pendulum_baseline_beats_random_actions() {
    let (base, random) = (mean_return(128, true), mean_return(128, false));
    assert!(base > random + 200.0, "baseline {base}, random {random}");
}

// -- import validation (design note 4.2) -----------------------------------

fn record(p: &Pool, s: [f64; 2], pending: bool, end_kind: u8) -> DecodedRecord<Pendulum> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.engine.pending_reset = Bool8::new(pending);
    header.engine.end_kind = end_kind;
    DecodedRecord::from_parts(header, PendulumState { s }).unwrap()
}

fn code(r: Result<(), EngineError>) -> &'static str {
    match r {
        Err(EngineError::Invariant { violation, .. }) => violation.code,
        other => panic!("{other:?}"),
    }
}

#[test]
fn test_pendulum_import_refuses_states_out_of_the_domain() {
    let mut p = pool(5);
    let before = p.state_hash(0).unwrap();
    for (s, expect) in [
        ([f64::INFINITY, 0.0], "a non-finite component"),
        ([0.0, MAX_SPEED.next_up()], "an angular speed out of range"),
        (
            [0.0, (-MAX_SPEED).next_down()],
            "an angular speed out of range",
        ),
    ] {
        assert_eq!(
            code(p.import(&[0], &[record(&p, s, false, 0)])),
            expect,
            "{s:?}"
        );
    }
    assert_eq!(p.state_hash(0).unwrap(), before);
    for s in [[f64::MAX, MAX_SPEED], [-f64::MAX, -MAX_SPEED], [-0.0, -0.0]] {
        p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
    }
}

#[test]
fn test_pendulum_record_refuses_nan_at_decoding() {
    let p = pool(5);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    for s in [[f64::NAN, 0.0], [0.0, f64::NAN]] {
        assert!(
            DecodedRecord::<Pendulum>::from_parts(*good.header(), PendulumState { s }).is_err()
        );
    }
}

#[test]
fn test_pendulum_import_refuses_a_terminated_record() {
    let mut p = pool(5);
    let any = [0.5, 0.1];
    p.import(&[0], &[record(&p, any, true, 2)]).unwrap();
    for kind in [1, 3] {
        assert_eq!(
            code(p.import(&[0], &[record(&p, any, true, kind)])),
            "a terminated record: pendulum never terminates"
        );
    }
}

#[test]
fn test_pendulum_import_before_the_first_reset_takes_the_exact_default() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let fresh = Pool::new(PendulumConfig::default(), options).unwrap();
    let rec = DecodedRecord::from(fresh.snapshot(0).unwrap());
    let mut q = Pool::new(PendulumConfig::default(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    for s in [[-0.0, 0.0], [0.0, -0.0], [0.5, 0.0]] {
        let bad = DecodedRecord::from_parts(*rec.header(), PendulumState { s }).unwrap();
        assert_eq!(
            code(q.import(&[0], &[bad])),
            "a state other than the initial value before the first reset"
        );
    }
}

#[test]
fn test_pendulum_import_requires_the_agent_alive_and_acting() {
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

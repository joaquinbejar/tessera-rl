// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acrobot through the engine: spaces, the reset distribution, the dynamics
//! against a transcription of Gymnasium's step written for these tests,
//! rewards, termination, truncation, the scripted baseline and import
//! validation by phase and end flags.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    reason = "test helpers: a failure is the assertion"
)]

use core::f64::consts::PI;
use core::num::NonZeroU32;

use tessera_rl_classic::acrobot::MAX_EPISODE_STEPS;
use tessera_rl_classic::acrobot::physics::{MAX_VEL_1, MAX_VEL_2, is_terminal};
use tessera_rl_classic::{Acrobot, AcrobotConfig, AcrobotState};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{Bool8, Environment};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::math::{cos, sin};

type Pool = EnvPool<Acrobot>;

fn pool_with(n: u32, seed: u64, auto_reset: AutoReset, limit: u32) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::new(AcrobotConfig::default(), options).unwrap();
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

fn state(p: &Pool, i: usize) -> [f64; 4] {
    p.state(u32::try_from(i).unwrap()).unwrap().s
}

fn record(p: &Pool, s: [f64; 4], pending: bool, end_kind: u8) -> DecodedRecord<Acrobot> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.engine.pending_reset = Bool8::new(pending);
    header.engine.end_kind = end_kind;
    DecodedRecord::from_parts(header, AcrobotState { s }).unwrap()
}

/// Gymnasium 1.4.0 `AcrobotEnv._dsdt` ("book"), with its names, in plain
/// f64 and with `x * x` for its `x ** 2`.
fn dsdt(y: [f64; 4], a: f64) -> [f64; 4] {
    let (m1, m2, l1, lc1, lc2, i1, i2, g) = (1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 9.8);
    let (t1, t2, v1, v2): (f64, f64, f64, f64) = (y[0], y[1], y[2], y[3]);
    let d1 = m1 * (lc1 * lc1) + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * cos(t2)) + i1 + i2;
    let d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos(t2)) + i2;
    let phi2 = m2 * lc2 * g * cos(t1 + t2 - PI / 2.0);
    let phi1 = -m2 * l1 * lc2 * (v2 * v2) * sin(t2) - 2.0 * m2 * l1 * lc2 * v2 * v1 * sin(t2)
        + (m1 * lc1 + m2 * l1) * g * cos(t1 - PI / 2.0)
        + phi2;
    let dd2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * (v1 * v1) * sin(t2) - phi2)
        / (m2 * (lc2 * lc2) + i2 - d2 * d2 / d1);
    [v1, v2, -(d2 * dd2 + phi1) / d1, dd2]
}

fn reference(y: [f64; 4], action: i32) -> [f64; 4] {
    let a = [-1.0, 0.0, 1.0][usize::try_from(action).unwrap()];
    let add = |y: [f64; 4], h: f64, k: [f64; 4]| core::array::from_fn(|i| y[i] + h * k[i]);
    let k1 = dsdt(y, a);
    let k2 = dsdt(add(y, 0.1, k1), a);
    let k3 = dsdt(add(y, 0.1, k2), a);
    let k4 = dsdt(add(y, 0.2, k3), a);
    let n: [f64; 4] =
        core::array::from_fn(|i| y[i] + 0.2 / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    let wrap = |mut x: f64| {
        while x > PI {
            x -= 2.0 * PI;
        }
        while x < -PI {
            x += 2.0 * PI;
        }
        x
    };
    let bound = |x: f64, m: f64| x.max(-m).min(m);
    [
        wrap(n[0]),
        wrap(n[1]),
        bound(n[2], 4.0 * PI),
        bound(n[3], 9.0 * PI),
    ]
}

#[test]
fn test_acrobot_spaces_match_gymnasium() {
    let cfg = AcrobotConfig::default();
    assert_eq!(
        Acrobot::action_space(&cfg).unwrap(),
        Space::Discrete { n: 3 }
    );
    let Space::Box {
        shape,
        dtype,
        bounds: BoxBounds::Float { low, high },
    } = Acrobot::observation_space(&cfg).unwrap()
    else {
        panic!("not a float box")
    };
    assert_eq!((shape, dtype), (vec![6], DType::F32));
    let top = [
        1.0,
        1.0,
        1.0,
        1.0,
        f64::from((4.0 * PI) as f32),
        f64::from((9.0 * PI) as f32),
    ];
    assert_eq!(high, top.to_vec());
    assert_eq!(low, top.map(|v| -v).to_vec());
}

#[test]
fn test_acrobot_reset_is_small_and_rounded_to_float32() {
    let p = pool_with(2048, 7, AutoReset::Disabled, 500);
    let (mut low, mut high) = (f64::MAX, f64::MIN);
    for i in 0..2048 {
        for v in state(&p, i) {
            assert!((-0.1..=0.100_000_1).contains(&(v + 1e-9)), "{v}");
            assert_eq!(f64::from(v as f32).to_bits(), v.to_bits(), "float32 values");
            low = low.min(v);
            high = high.max(v);
        }
    }
    assert!(low < -0.099 && high > 0.099, "{low} {high}");
}

#[test]
fn test_acrobot_step_matches_the_reference_transcription() {
    // Uniform over the valid domain, so wraps and velocity bounds occur.
    let mut p = pool(11);
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let mut unit = move || {
        x = x.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
        f64::from(u32::try_from(x >> 33).unwrap()) / f64::from(1u32 << 31)
    };
    let (mut wraps, mut bounds, mut checked) = (0, 0, 0);
    for k in 0..20_000 {
        let s = [
            -PI + 2.0 * PI * unit(),
            -PI + 2.0 * PI * unit(),
            -MAX_VEL_1 + 2.0 * MAX_VEL_1 * unit(),
            -MAX_VEL_2 + 2.0 * MAX_VEL_2 * unit(),
        ];
        if is_terminal(s) {
            continue;
        }
        let action = k % 3;
        let expect = reference(s, action);
        p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
        step(&mut p, &[action]);
        if p.outputs().unwrap().episode_terminated()[0].get() {
            assert_eq!(p.outputs().unwrap().rewards()[0], 0.0);
            assert!(is_terminal(expect));
        } else {
            assert_eq!(p.outputs().unwrap().rewards()[0], -1.0);
            assert!(!is_terminal(expect));
        }
        assert_eq!(
            state(&p, 0).map(f64::to_bits),
            expect.map(f64::to_bits),
            "{s:?} {action}"
        );
        wraps += i32::from((expect[0] - s[0]).abs() > PI || (expect[1] - s[1]).abs() > PI);
        bounds += i32::from(expect[2].abs() == MAX_VEL_1 || expect[3].abs() == MAX_VEL_2);
        checked += 1;
        p.reset_all(Reseed::Keep).unwrap();
    }
    assert!(
        checked > 10_000 && wraps > 100 && bounds > 100,
        "{checked} {wraps} {bounds}"
    );
}

#[test]
fn test_acrobot_terminates_above_the_bar_and_truncates_at_the_limit() {
    let mut p = pool(1);
    let s = [2.0, 0.0, 5.0, 0.0];
    assert!(!is_terminal(s));
    p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
    step(&mut p, &[1]);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
    assert_eq!(p.outputs().unwrap().rewards()[0], 0.0);
    let mut t = pool_with(1, 1, AutoReset::Disabled, MAX_EPISODE_STEPS);
    for _ in 0..MAX_EPISODE_STEPS {
        step(&mut t, &[1]);
    }
    let out = t.outputs().unwrap();
    assert!(out.episode_truncated()[0].get() && !out.episode_terminated()[0].get());
    assert_eq!(out.rewards()[0], -1.0);
}

#[test]
fn test_acrobot_baseline_reaches_the_goal_within_the_limit() {
    let n = 64;
    let mut p = pool_with(n, 3, AutoReset::NextStep, MAX_EPISODE_STEPS);
    let mut ended = vec![0u32; n as usize];
    for t in 1..=MAX_EPISODE_STEPS {
        let actions: Vec<i32> = (0..n as usize)
            .map(|i| if state(&p, i)[3] >= 0.0 { 2 } else { 0 })
            .collect();
        step(&mut p, &actions);
        let out = p.outputs().unwrap();
        for i in 0..n as usize {
            if ended[i] == 0 && out.episode_terminated()[i].get() {
                ended[i] = t;
            }
        }
    }
    assert!(
        ended.iter().all(|e| *e > 0),
        "every first episode reaches the goal"
    );
}

fn code(r: Result<(), EngineError>) -> &'static str {
    match r {
        Err(EngineError::Invariant { violation, .. }) => violation.code,
        other => panic!("{other:?}"),
    }
}

const ROAD: [f64; 4] = [0.1, -0.1, 0.5, -0.5];
const TOP: [f64; 4] = [PI, 0.0, 0.0, 0.0];

#[test]
fn test_acrobot_import_refuses_states_out_of_the_domain() {
    let mut p = pool(5);
    let before = p.state_hash(0).unwrap();
    for (s, expect) in [
        ([f64::INFINITY, 0.0, 0.0, 0.0], "a non-finite component"),
        ([0.0, 0.0, 0.0, f64::NEG_INFINITY], "a non-finite component"),
        ([PI.next_up(), 0.0, 0.0, 0.0], "an angle out of range"),
        ([0.0, (-PI).next_down(), 0.0, 0.0], "an angle out of range"),
        ([1e300, 0.0, 0.0, 0.0], "an angle out of range"),
        (
            [0.0, 0.0, MAX_VEL_1.next_up(), 0.0],
            "a velocity out of range",
        ),
        (
            [0.0, 0.0, 0.0, (-MAX_VEL_2).next_down()],
            "a velocity out of range",
        ),
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
    for s in [[0.0, 0.0, MAX_VEL_1, -MAX_VEL_2], [-PI, PI, 0.0, 0.0], ROAD] {
        p.import(&[0], &[record(&p, s, false, 0)]).unwrap();
    }
}

#[test]
fn test_acrobot_import_checks_the_terminal_predicate_by_end_flags() {
    let mut p = pool(5);
    assert_eq!(
        code(p.import(&[0], &[record(&p, TOP, false, 0)])),
        "a terminal state not reported as terminated"
    );
    for kind in [1, 3] {
        p.import(&[0], &[record(&p, TOP, true, kind)]).unwrap();
        assert_eq!(
            code(p.import(&[0], &[record(&p, ROAD, true, kind)])),
            "a terminated record in a non-terminal state"
        );
    }
    p.import(&[0], &[record(&p, ROAD, true, 2)]).unwrap();
    assert_eq!(
        code(p.import(&[0], &[record(&p, TOP, true, 2)])),
        "a terminal state not reported as terminated"
    );
}

#[test]
fn test_acrobot_import_before_the_first_reset_takes_the_exact_default() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let fresh = Pool::new(AcrobotConfig::default(), options).unwrap();
    let rec = DecodedRecord::from(fresh.snapshot(0).unwrap());
    let mut q = Pool::new(AcrobotConfig::default(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    for s in [[-0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.0], ROAD] {
        let bad = DecodedRecord::from_parts(*rec.header(), AcrobotState { s }).unwrap();
        assert_eq!(
            code(q.import(&[0], &[bad])),
            "a state other than the initial value before the first reset"
        );
    }
}

#[test]
fn test_acrobot_import_requires_the_agent_alive_and_acting() {
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

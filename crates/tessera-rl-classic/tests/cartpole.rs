// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole through the engine: spaces, reset distribution, rewards,
//! termination, truncation and the scripted baseline.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::cartpole::physics::{THETA_THRESHOLD, X_THRESHOLD};
use tessera_rl_classic::{CartPole, CartPoleConfig, CartPoleState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{Bool8, Environment, Heuristic, StateHash};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

/// A `CartPole-v1` pool: the step limit is 500 unless `limit` says otherwise.
fn pool_limit(n: u32, seed: u64, limit: u32) -> EnvPool<CartPole> {
    let options = PoolOptions {
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    EnvPool::new(CartPoleConfig::default(), options).unwrap()
}

fn pool(n: u32, seed: u64, _: CartPoleConfig) -> EnvPool<CartPole> {
    pool_limit(n, seed, MAX_EPISODE_STEPS)
}

fn act(values: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: values,
        continuous: &[],
    }
}

fn obs(p: &EnvPool<CartPole>, i: usize) -> [f32; 4] {
    let bytes = &p.outputs().unwrap().observation(0).unwrap()[i * 16..(i + 1) * 16];
    core::array::from_fn(|k| f32::from_le_bytes(bytes[k * 4..k * 4 + 4].try_into().unwrap()))
}

fn heuristic(state: &CartPoleState) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    CartPole::act(
        &CartPoleConfig::default(),
        state,
        tessera_rl_core::AgentSlot::FIRST,
        &mut rng,
        &mut writer,
    );
    discrete[0]
}

#[test]
fn test_cartpole_spaces_match_gymnasium_float32_bounds() {
    let cfg = CartPoleConfig::default();
    let high = [4.8_f32, f32::INFINITY, 0.418_879_03_f32, f32::INFINITY].map(f64::from);
    let expected = Space::Box {
        shape: vec![4],
        dtype: DType::F32,
        bounds: BoxBounds::Float {
            low: high.iter().map(|v| -v).collect(),
            high: high.to_vec(),
        },
    };
    let space = CartPole::observation_space(&cfg).unwrap();
    assert_eq!(space, expected);
    assert_eq!(space.validate(), Ok(()));
    assert_eq!(
        CartPole::action_space(&cfg).unwrap(),
        Space::Discrete { n: 2 }
    );
    assert_eq!(MAX_EPISODE_STEPS, 500);
}

#[test]
fn test_cartpole_reset_draws_each_component_in_the_reset_range() {
    let mut p = pool(64, 3, CartPoleConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    for i in 0..64 {
        let s = p.state(i).unwrap().s;
        assert!(s.iter().all(|v| (-0.05..0.05).contains(v)), "{s:?}");
        #[expect(clippy::cast_possible_truncation, reason = "float32 observations")]
        let expected = s.map(|v| v as f32);
        assert_eq!(obs(&p, usize::try_from(i).unwrap()), expected);
    }
    // Distinct instances draw distinct states.
    assert_ne!(p.state(0).unwrap().s, p.state(1).unwrap().s);
}

#[test]
fn test_cartpole_same_seed_is_bit_identical() {
    let run = || {
        let mut p = pool(4, 11, CartPoleConfig::default());
        p.reset_all(Reseed::Keep).unwrap();
        for t in 0..300 {
            p.step(act(&[t % 2, 1, 0, (t / 3) % 2])).unwrap();
        }
        (0..4)
            .map(|i| StateHash::of(p.state(i).unwrap()).get())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn test_cartpole_constant_push_terminates_with_reward_one() {
    let mut p = pool(1, 5, CartPoleConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    let mut steps = 0_u16;
    loop {
        let out = p.step(act(&[1])).unwrap();
        steps += 1;
        assert_eq!(out.rewards(), [1.0]);
        if out.episode_terminated()[0].get() {
            assert_eq!(out.episode_truncated(), [Bool8::FALSE]);
            assert_eq!(out.episode_length(), [f32::from(steps)]);
            break;
        }
        assert!(steps < 100, "a constant push must topple the pole");
    }
    let s = p.state(0).unwrap().s;
    assert!(s[0].abs() > X_THRESHOLD || s[2].abs() > THETA_THRESHOLD);
}

#[test]
fn test_cartpole_truncates_at_the_engine_step_limit() {
    let mut p = pool_limit(1, 5, 3);
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..3 {
        let out = p.step(act(&[t % 2])).unwrap();
        assert_eq!(out.episode_truncated()[0].get(), t == 2);
        assert_eq!(out.episode_terminated(), [Bool8::FALSE]);
    }
}

#[test]
fn test_cartpole_heuristic_balances_to_truncation() {
    let n = 16;
    let mut p = pool(n, 9, CartPoleConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    let mut actions = vec![0; usize::try_from(n).unwrap()];
    for _ in 0..500 {
        for i in 0..n {
            actions[usize::try_from(i).unwrap()] = heuristic(p.state(i).unwrap());
        }
        let out = p.step(act(&actions)).unwrap();
        assert!(out.episode_terminated().iter().all(|f| !f.get()));
    }
    assert!(
        p.outputs()
            .unwrap()
            .episode_truncated()
            .iter()
            .all(|f| f.get())
    );
}

/// Gymnasium's `TimeLimit`: a terminal step that reaches the limit reports
/// terminated and truncated. The pre-step state and horizon are the ones
/// codex reproduced against Gymnasium 1.0.0 (horizon 9, 8 steps elapsed,
/// action 1).
#[test]
fn test_cartpole_terminal_step_at_the_horizon_is_terminated_and_truncated() {
    let mut p = pool_limit(1, 5, 9);
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..8 {
        let out = p.step(act(&[t % 2])).unwrap();
        assert!(!out.episode_terminated()[0].get() && !out.episode_truncated()[0].get());
    }
    let record = p.snapshot(0).unwrap();
    let state = CartPoleState {
        s: [
            0.131_335_108_872_431_9,
            1.601_798_890_794_725_8,
            -0.207_035_334_068_830_26,
            -2.519_182_399_894_15,
        ],
    };
    let injected = DecodedRecord::from_parts(*record.header(), state).unwrap();
    p.import(&[0], &[injected]).unwrap();
    let out = p.step(act(&[1])).unwrap();
    assert_eq!(out.rewards(), [1.0]);
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
    assert_eq!(out.episode_truncated(), [Bool8::TRUE]);
    assert_eq!(out.agent_terminated(), [Bool8::TRUE]);
    assert_eq!(out.agent_truncated(), [Bool8::TRUE]);
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole as a host environment (ADR-0062), for the host pool's
//! instruction counts. Self-contained: it does not depend on
//! `tessera-rl-classic`, which dev-depends on the engine (a dependency the
//! other way would be a cycle that `cargo package` refuses).
//!
//! The physics is the first-party `CartPole`'s (Gymnasium `CartPole-v1`,
//! explicit Euler, sine and cosine from the frozen math library, ADR-0008),
//! with the same spaces, reward and termination. The state is held by the
//! instance. Its reset draws the four components from the seed and episode
//! index with splitmix64, not from `StepRng` (host environments have none);
//! the draw is fixed by its inputs, so every run reaches the same states.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::leaf::LeafId;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EnvConfig, EpisodeEnd, HostEnvironment, HostError, HostSeed, SpaceError, StepOut,
};
use tessera_rl_num::math;

/// Gravity (m/s^2).
const GRAVITY: f64 = 9.8;
/// Cart mass (kg).
const MASS_CART: f64 = 1.0;
/// Pole mass (kg).
const MASS_POLE: f64 = 0.1;
/// Half the pole length (m).
const LENGTH: f64 = 0.5;
/// Magnitude of the push (N).
const FORCE_MAG: f64 = 10.0;
/// Integration step (s).
const TAU: f64 = 0.02;
/// The episode terminates when `|x|` exceeds this (m).
const X_THRESHOLD: f64 = 2.4;
/// The episode terminates when `|theta|` exceeds this: 12 degrees.
const THETA_THRESHOLD: f64 = 12.0 * 2.0 * core::f64::consts::PI / 360.0;
const TOTAL_MASS: f64 = MASS_POLE + MASS_CART;
const POLEMASS_LENGTH: f64 = MASS_POLE * LENGTH;

/// The reset range of every component, as the first-party `CartPole`.
const RESET_BOUND: f64 = 0.05;

/// The observation's single leaf: the four components as `f32`.
const STATE: LeafId<f32> = LeafId::new(0);

/// One Euler step of `[x, x_dot, theta, theta_dot]` under `push_right`.
#[must_use]
pub fn physics_step(s: [f64; 4], push_right: bool) -> [f64; 4] {
    let [x, x_dot, theta, theta_dot] = s;
    let force = if push_right { FORCE_MAG } else { -FORCE_MAG };
    let costheta = math::cos(theta);
    let sintheta = math::sin(theta);
    let temp = (force + POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) / TOTAL_MASS;
    let thetaacc = (GRAVITY * sintheta - costheta * temp)
        / (LENGTH * (4.0 / 3.0 - MASS_POLE * (costheta * costheta) / TOTAL_MASS));
    let xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;
    [
        x + TAU * x_dot,
        x_dot + TAU * xacc,
        theta + TAU * theta_dot,
        theta_dot + TAU * thetaacc,
    ]
}

/// Whether `s` is outside the allowed cart position or pole angle.
fn is_terminal(s: [f64; 4]) -> bool {
    let [x, _, theta, _] = s;
    x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD || theta > THETA_THRESHOLD
}

/// splitmix64: one step of the reset draw.
const fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The port's configuration: none.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostCartPoleConfig;

impl EnvConfig for HostCartPoleConfig {}

/// CartPole, host-driven.
#[derive(Debug, Default)]
pub struct HostCartPole {
    s: [f64; 4],
}

impl HostEnvironment for HostCartPole {
    const ENV_ID: &'static str = "bench/HostCartPole-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;

    type Config = HostCartPoleConfig;

    fn new(_: &HostCartPoleConfig) -> Result<Self, HostError> {
        Ok(Self::default())
    }

    /// Gymnasium's float32 bounds: the thresholds doubled, rounded to `f32`,
    /// and infinity for the velocities.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the bounds are defined as float32 values"
    )]
    fn observation_space(_: &HostCartPoleConfig) -> Result<Space, SpaceError> {
        let high: Vec<f64> = [
            X_THRESHOLD * 2.0,
            f64::INFINITY,
            THETA_THRESHOLD * 2.0,
            f64::INFINITY,
        ]
        .map(|v| f64::from(v as f32))
        .to_vec();
        Ok(Space::Box {
            shape: vec![4],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: high.iter().map(|v| -v).collect(),
                high,
            },
        })
    }

    fn action_space(_: &HostCartPoleConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits to a unit f64, exactly"
    )]
    fn reset(&mut self, seed: HostSeed, _: &mut StepOut<'_>) {
        let mut s = seed.seed() ^ (u64::from(seed.episode_index()) << 32);
        for v in &mut self.s {
            let unit = (splitmix(&mut s) >> 11) as f64 / (1_u64 << 53) as f64;
            *v = -RESET_BOUND + 2.0 * RESET_BOUND * unit;
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut StepOut<'_>) {
        let push_right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
        self.s = physics_step(self.s, push_right);
        out.set_reward(AgentSlot::FIRST, 1.0);
        if is_terminal(self.s) {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        if let Some(mut leaf) = obs.leaf_mut(STATE) {
            for (i, v) in self.s.iter().enumerate() {
                leaf.set(i, *v as f32);
            }
        }
    }
}

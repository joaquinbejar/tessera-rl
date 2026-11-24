// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SoaCartPole`: CartPole's kernels with a lane-wise state (ADR-0090). Its
//! state holds the same four `f64` fields as `CartPoleState`, in the same
//! order, so its canonical encoding, and with it every record, persisted
//! file and hash, is byte-identical; only the in-memory store differs
//! (Struct-of-Arrays, one column per field). Shared by the store tests and
//! benches.

use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, Heuristic, InvariantViolation, NoAdaptive, Persist, RecordHeader,
    SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::EnvState;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// CartPole's state as four lane fields.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
#[state(lanes)]
pub struct SoaCartPoleState {
    /// Cart position.
    pub x: f64,
    /// Cart velocity.
    pub x_dot: f64,
    /// Pole angle.
    pub theta: f64,
    /// Pole angular velocity.
    pub theta_dot: f64,
}

impl SoaCartPoleState {
    /// The same state as `CartPoleState`.
    pub const fn rows(self) -> CartPoleState {
        CartPoleState {
            s: [self.x, self.x_dot, self.theta, self.theta_dot],
        }
    }

    /// The state of a `CartPoleState`.
    pub const fn of(s: CartPoleState) -> Self {
        let [x, x_dot, theta, theta_dot] = s.s;
        Self {
            x,
            x_dot,
            theta,
            theta_dot,
        }
    }
}

/// CartPole on the Struct-of-Arrays store.
pub struct SoaCartPole;

impl Environment for SoaCartPole {
    type State = SoaCartPoleState;
    type Config = CartPoleConfig;
    type Obs = CartPoleObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::observation_space(cfg)
    }

    fn action_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::action_space(cfg)
    }

    fn reset(cfg: &CartPoleConfig, rng: &mut StepRng, state: &mut SoaCartPoleState) {
        let mut s = state.rows();
        CartPole::reset(cfg, rng, &mut s);
        *state = SoaCartPoleState::of(s);
    }

    fn step(
        cfg: &CartPoleConfig,
        state: &mut SoaCartPoleState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let mut s = state.rows();
        CartPole::step(cfg, &mut s, actions, rng, out);
        *state = SoaCartPoleState::of(s);
    }

    fn observe(
        cfg: &CartPoleConfig,
        state: &SoaCartPoleState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CartPoleObs>,
    ) {
        CartPole::observe(cfg, &state.rows(), agent, rng, out);
    }
}

impl Validate for SoaCartPole {
    fn validate(cfg: &CartPoleConfig, state: &SoaCartPoleState) -> Result<(), InvariantViolation> {
        CartPole::validate(cfg, &state.rows())
    }
}

impl Persist for SoaCartPole {
    const ENV_ID: &'static str = CartPole::ENV_ID;
    const ENV_VERSION: u32 = CartPole::ENV_VERSION;
}

impl Heuristic for SoaCartPole {
    fn act(
        cfg: &CartPoleConfig,
        state: &SoaCartPoleState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        CartPole::act(cfg, &state.rows(), agent, rng, out);
    }
}

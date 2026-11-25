// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SoaCartPole`: CartPole's kernels with a lane-wise state (ADR-0090). Its
//! state holds the same four `f64` fields as `CartPoleState`, in the same
//! order, so its canonical encoding, and with it every record, persisted
//! file and hash, is byte-identical; only the in-memory store differs
//! (Struct-of-Arrays, one column per field). Its lane kernels (ADR-0090
//! decision 4) are CartPole's `lane::step` on the four columns and the
//! `f32` cast of `CartPole::observe`, lane by lane. Shared by the store
//! tests and benches.

use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState, lane};
use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EnvState, Environment, EpisodeEnd, Heuristic, InvariantViolation, LANES,
    LaneEnvHooks, LaneEnvironment, LaneGroup, LaneOut, NoAdaptive, Persist, RecordHeader,
    SpaceError, StepOut, Tier, Validate, lane_env_hooks,
};
use tessera_rl_macros::EnvState;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::Resolved;

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
    const LANE_ENV: Option<LaneEnvHooks<Self>> = Some(lane_env_hooks::<Self>());

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

/// Groups stepped through `step_lanes`, counted in test builds only, so a
/// test can tell the lane path ran (the differential alone cannot).
#[cfg(test)]
pub static LANE_STEPS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// CartPole's lane kernel on the columns: the four column groups are the
/// kernel's `Columns` (four `[f64; 8]`), copied in and out as whole arrays.
impl LaneEnvironment for SoaCartPole {
    const STEP_DRAWS_RNG: bool = false;
    const OBSERVE_DRAWS_RNG: bool = false;

    fn step_lanes(
        _: &CartPoleConfig,
        level: Resolved,
        lanes: &mut LaneGroup<'_, Self>,
        actions: &[i32; LANES],
        out: &mut LaneOut,
    ) {
        #[cfg(test)]
        LANE_STEPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let (x, x_dot, theta, theta_dot) = lanes;
        let mut columns: lane::Columns = [**x, **x_dot, **theta, **theta_dot];
        let outcome = lane::step(level, &mut columns, actions.map(|a| a == 1));
        let [cx, cxd, ct, ctd] = columns;
        (**x, **x_dot, **theta, **theta_dot) = (cx, cxd, ct, ctd);
        out.reward = [1.0; LANES];
        for (l, end) in out.end.iter_mut().enumerate() {
            *end = (outcome.terminal >> l & 1 == 1).then_some(EpisodeEnd::Terminated);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as CartPole::observe"
    )]
    fn observe_lanes(_: &CartPoleConfig, lanes: &LaneGroup<'_, Self>, leaves: &mut [&mut [u8]]) {
        let (x, x_dot, theta, theta_dot) = lanes;
        let Some(rows) = leaves.first_mut() else {
            return;
        };
        // Row `l` holds the four f32 of lane `l`, as CartPole::observe
        // writes them.
        let lanes = x
            .iter()
            .zip(x_dot.iter())
            .zip(theta.iter())
            .zip(theta_dot.iter());
        for (row, (((&a, &b), &c), &d)) in rows.as_chunks_mut::<16>().0.iter_mut().zip(lanes) {
            for (dst, v) in row.as_chunks_mut::<4>().0.iter_mut().zip([a, b, c, d]) {
                dst.copy_from_slice(&(v as f32).to_le_bytes());
            }
        }
    }
}

const _: () = assert!(<SoaCartPoleState as EnvState>::LANES);

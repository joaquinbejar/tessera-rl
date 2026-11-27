// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent-lane step hook (ADR-0100 decision 1.5).
//!
//! A many-agent environment whose step has order-free per-agent phases may
//! set [`Environment::AGENT_LANE`] to a hook that runs those phases as lane
//! kernels over groups of 8 agents of one instance. The engine resolves the
//! SIMD level once, when a pool is built, and calls the hook in place of
//! [`Environment::step_with`] when the pool's
//! `agent_lanes` option allows it. The hook must leave the instance exactly
//! as the reference step would, bit for bit (ADR-0100 decision 5). The state
//! keeps its Array-of-Structs store: no gather, scatter or new layout.

use tessera_rl_num::rng::StepRng;
use tessera_rl_simd::Resolved;

use crate::env::{Adaptive, Environment};
use crate::io::Actions;
use crate::step_out::StepOut;

/// The engine's entry point for an agent-lane step: the parameters of
/// [`Environment::step_with`] plus the pool's resolved SIMD level, last
/// (ADR-0100 decision 1.5). An environment that reads its adaptive values
/// never takes the hook, so a hook may ignore `adaptive`.
pub type AgentLaneHook<E> = fn(
    &<E as Environment>::Config,
    &Adaptive<E>,
    &mut <E as Environment>::State,
    &Actions<'_, <E as Environment>::Action>,
    &mut StepRng,
    &mut StepOut<'_>,
    Resolved,
);

/// A many-agent environment whose step runs its order-free per-agent phases
/// as lane kernels over groups of 8 agents of one instance (ADR-0100
/// decision 1). The phases and kernels are the environment's own; the
/// engine only calls the whole agent-lane step through
/// [`Environment::AGENT_LANE`], set to `Some(agent_lane_hook::<Self>())`.
pub trait AgentLaneEnvironment: Environment {
    /// Steps `state` as [`Environment::step_with`] does, bit for bit, with
    /// the order-free per-agent phases at `level` (ADR-0100 decision 5).
    fn step_agent_lanes(
        cfg: &Self::Config,
        adaptive: &Adaptive<Self>,
        state: &mut Self::State,
        actions: &Actions<'_, Self::Action>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
        level: Resolved,
    );
}

/// The value of [`Environment::AGENT_LANE`] for an
/// [`AgentLaneEnvironment`]. Fails to compile when the environment reads
/// its adaptive values, since a pool never takes the hook for it (ADR-0100
/// decision 1.5).
#[must_use]
pub const fn agent_lane_hook<E: AgentLaneEnvironment>() -> AgentLaneHook<E> {
    const {
        assert!(
            !E::READS_ADAPTIVE,
            "AgentLaneEnvironment with READS_ADAPTIVE = true is not supported (ADR-0100)"
        );
    }
    E::step_agent_lanes
}

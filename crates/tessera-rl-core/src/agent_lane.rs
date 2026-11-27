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

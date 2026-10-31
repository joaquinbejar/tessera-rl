// SPDX-License-Identifier: MIT OR Apache-2.0

//! Action masks and the illegal-action policy (ADR-0014, ADR-0028).
//!
//! Masks are chosen at compile time: `EnvPool<E, NoMasks>` (the default)
//! compiles the mask code away, and `EnvPool<E, Masked>` calls
//! `ActionMask` for environments that implement it. No dynamic dispatch on
//! the hot path.

use tessera_rl_core::io::{ActionWriter, MaskWriter};
use tessera_rl_core::{ActionMask, AgentSlot, Environment};

/// Provides masks and no-op actions for an environment.
pub trait MaskProvider<E: Environment>: 'static {
    /// Whether masks exist; when `false` the mask code is never run.
    const ENABLED: bool;

    /// Writes the mask of `agent`.
    fn write_mask(cfg: &E::Config, state: &E::State, agent: AgentSlot, out: &mut MaskWriter<'_>);

    /// Writes the no-op action of `agent`.
    fn noop(cfg: &E::Config, state: &E::State, agent: AgentSlot, out: &mut ActionWriter<'_>);
}

/// No masks: every valid action is legal.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoMasks;

impl<E: Environment> MaskProvider<E> for NoMasks {
    const ENABLED: bool = false;

    #[inline]
    fn write_mask(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut MaskWriter<'_>) {}

    #[inline]
    fn noop(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut ActionWriter<'_>) {}
}

/// Masks from the environment's `ActionMask` implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Masked;

impl<E: ActionMask> MaskProvider<E> for Masked {
    const ENABLED: bool = true;

    #[inline]
    fn write_mask(cfg: &E::Config, state: &E::State, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        E::action_mask(cfg, state, agent, out);
    }

    #[inline]
    fn noop(cfg: &E::Config, state: &E::State, agent: AgentSlot, out: &mut ActionWriter<'_>) {
        E::noop(cfg, state, agent, out);
    }
}

/// What happens to an action that is valid but illegal under the mask.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IllegalActionPolicy {
    /// Replace the agent's action with the environment's no-op.
    #[default]
    Noop,
    /// Replace it with the no-op and add this (finite) reward to the agent.
    Penalize(f32),
}

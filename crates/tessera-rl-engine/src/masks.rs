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

    /// Parents of discrete head `head`.
    fn head_parents(_cfg: &E::Config, _head: usize) -> [Option<u16>; 2] {
        [None, None]
    }

    /// Discrete heads per entity group, if declared (ADR-0051).
    fn entity_group_heads(_cfg: &E::Config) -> Option<core::num::NonZeroU16> {
        None
    }
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

    fn head_parents(cfg: &E::Config, head: usize) -> [Option<u16>; 2] {
        E::head_parents(cfg, head)
    }

    fn entity_group_heads(cfg: &E::Config) -> Option<core::num::NonZeroU16> {
        E::entity_group_heads(cfg)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::NoMasks {}
    impl Sealed for super::Masked {}
}

/// The mask providers a persisted file can name (ADR-0011): the built-in
/// two, sealed. An enabled flag alone would conflate providers with
/// different masks or no-ops, so a custom provider cannot persist in format 1.
/// [`MaskProvider`] itself stays open.
///
/// A custom provider compiles:
///
/// ```
/// use tessera_rl_core::io::{ActionWriter, MaskWriter};
/// use tessera_rl_core::{AgentSlot, Environment};
/// use tessera_rl_engine::MaskProvider;
///
/// struct Mine;
/// impl<E: Environment> MaskProvider<E> for Mine {
///     const ENABLED: bool = true;
///     fn write_mask(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut MaskWriter<'_>) {}
///     fn noop(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut ActionWriter<'_>) {}
/// }
/// ```
///
/// The same provider with this one added impl does not, because `Mine` is not
/// `Sealed`. `compile_fail` cannot pin that reason on stable, which does not check
/// the error code; the twin above is what rules out the other causes.
///
/// ```compile_fail,E0277
/// use tessera_rl_core::io::{ActionWriter, MaskWriter};
/// use tessera_rl_core::{AgentSlot, Environment};
/// use tessera_rl_engine::{MaskProvider, PersistMask};
///
/// struct Mine;
/// impl<E: Environment> MaskProvider<E> for Mine {
///     const ENABLED: bool = true;
///     fn write_mask(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut MaskWriter<'_>) {}
///     fn noop(_: &E::Config, _: &E::State, _: AgentSlot, _: &mut ActionWriter<'_>) {}
/// }
/// impl<E: Environment> PersistMask<E> for Mine {
///     const PROVIDER_ID: u8 = 2;
/// }
/// ```
pub trait PersistMask<E: Environment>: MaskProvider<E> + sealed::Sealed {
    /// The provider's byte in the file.
    const PROVIDER_ID: u8;
}

impl<E: Environment> PersistMask<E> for NoMasks {
    const PROVIDER_ID: u8 = 0;
}

impl<E: ActionMask> PersistMask<E> for Masked {
    const PROVIDER_ID: u8 = 1;
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

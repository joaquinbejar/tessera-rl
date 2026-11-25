// SPDX-License-Identifier: MIT OR Apache-2.0

//! Curricula through per-slot targets (ADR-0081 item 5): a curriculum sees
//! the epoch report and writes target adaptive values; each instance's next
//! reset copies its slot's target into its header, so an episode never
//! sees its configuration change. The environment reads the values through
//! `Environment::reset_with` and `step_with` (ADR-0091).

use tessera_rl_core::encoding::{Canonical, StateHasher, decode_from_slice, encode_to_vec};
use tessera_rl_core::{Adaptive, Environment, StateInit};

use super::EpochReport;
use crate::error::EngineError;

/// A deterministic curriculum (ADR-0081 item 5.2), run on the calling
/// thread at each epoch barrier.
pub trait Curriculum<E: Environment>: Send + Sync + 'static {
    /// A stable name, with a version: `"my-crate/levels-v1"`. Part of the
    /// curriculum's identity in the pool context; change it when `State`
    /// or the meaning of `update` changes.
    const ID: &'static str;
    /// The curriculum's own plain-data state, in the pool context.
    type State: Canonical + StateInit + Copy + Send + Sync + 'static;

    /// Reads the epoch report and its state, and writes targets. It runs
    /// inside the barrier, in the canonical FP environment (ADR-0037), so
    /// everything it derives from the report (`EpisodeStats::var_return`
    /// included) is computed there, whatever the caller's controls; derive
    /// what the targets depend on here, not later on the caller.
    fn update(
        &mut self,
        report: &EpochReport,
        state: &mut Self::State,
        targets: &mut AdaptiveTargets<'_, E>,
    );
}

/// The per-slot targets a curriculum writes (ADR-0081 item 5.2). A write
/// changes nothing an episode sees until that slot's next reset.
pub struct AdaptiveTargets<'a, E: Environment> {
    values: &'a mut [Adaptive<E>],
    changed: &'a mut [bool],
}

impl<E: Environment> AdaptiveTargets<'_, E> {
    /// The number of instance slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the pool has no slots (never: a pool has at least one).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Slot `id`'s target, `None` out of range.
    #[must_use]
    pub fn get(&self, id: u32) -> Option<&Adaptive<E>> {
        self.values.get(usize::try_from(id).ok()?)
    }

    /// Sets slot `id`'s target; `false` (nothing changed) out of range.
    pub fn set(&mut self, id: u32, value: Adaptive<E>) -> bool {
        let Some(i) = usize::try_from(id).ok() else {
            return false;
        };
        match (self.values.get_mut(i), self.changed.get_mut(i)) {
            (Some(v), Some(c)) => {
                *v = value;
                *c = true;
                true
            }
            _ => false,
        }
    }

    /// Sets every slot's target.
    pub fn set_all(&mut self, value: Adaptive<E>) {
        self.values.fill(value);
        self.changed.fill(true);
    }
}

/// The curriculum part of the pool context (ADR-0081 item 6): identity,
/// state and targets in canonical bytes, and the frozen flag.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CurriculumContext {
    /// `0` for a pool without targets; else the hash of the curriculum's
    /// `ID` and sizes, or of the targets-only marker.
    pub identity: u64,
    /// The curriculum's canonical state; empty for targets only.
    pub state: Vec<u8>,
    /// One canonical adaptive value per slot, in slot order.
    pub targets: Vec<u8>,
    /// Whether the curriculum's barrier is frozen.
    pub frozen: bool,
}

/// Which slots a barrier's curriculum changed (ADR-0081 items 1.3 and 7).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CurriculumChanges {
    /// The changed slots, ascending.
    pub ids: Vec<u32>,
}

/// The type-erased curriculum held by a pool: called only at the barrier
/// and by context operations, never on the step path.
pub(crate) trait HeldCurriculum<E: Environment>: Send + Sync {
    fn update(&mut self, report: &EpochReport, targets: &mut AdaptiveTargets<'_, E>);
    fn state_bytes(&self) -> Vec<u8>;
    /// Checks `bytes` decode as the state, writing nothing.
    fn check_state(&self, bytes: &[u8]) -> Result<(), EngineError>;
    /// Restores checked `bytes`.
    fn restore_state(&mut self, bytes: &[u8]);
}

struct Held<C: Curriculum<E>, E: Environment> {
    curriculum: C,
    state: C::State,
    _env: core::marker::PhantomData<fn() -> E>,
}

impl<C: Curriculum<E>, E: Environment> HeldCurriculum<E> for Held<C, E> {
    fn update(&mut self, report: &EpochReport, targets: &mut AdaptiveTargets<'_, E>) {
        self.curriculum.update(report, &mut self.state, targets);
    }

    fn state_bytes(&self) -> Vec<u8> {
        encode_to_vec(&self.state)
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), EngineError> {
        decode_from_slice::<C::State>(bytes)
            .map(|_| ())
            .map_err(|_| EngineError::AdaptiveContext {
                reason: "the curriculum state does not decode",
            })
    }

    fn restore_state(&mut self, bytes: &[u8]) {
        if let Ok(state) = decode_from_slice::<C::State>(bytes) {
            self.state = state;
        }
    }
}

/// The targets-only identity (replays, Python curricula: ADR-0081 items 5.5
/// and 7.5). A Python curriculum's own identity is not verified.
fn targets_identity<E: Environment>() -> u64 {
    let mut h = StateHasher::new();
    h.add_bytes(b"tessera.targets.v1");
    h.add(&u64::try_from(<Adaptive<E> as Canonical>::ENCODED_LEN).unwrap_or(u64::MAX));
    h.finish().get()
}

fn curriculum_identity<C: Curriculum<E>, E: Environment>() -> u64 {
    let mut h = StateHasher::new();
    h.add_bytes(b"tessera.curriculum.v1");
    h.add_bytes(C::ID.as_bytes());
    h.add(&u64::try_from(<C::State as Canonical>::ENCODED_LEN).unwrap_or(u64::MAX));
    h.add(&u64::try_from(<Adaptive<E> as Canonical>::ENCODED_LEN).unwrap_or(u64::MAX));
    h.finish().get()
}

/// A pool's per-slot targets (ADR-0081 item 5.3), read by the kernels.
pub(crate) struct Targets<E: Environment>(pub(crate) Vec<Adaptive<E>>);

impl<E: Environment> core::ops::Deref for Targets<E> {
    type Target = [Adaptive<E>];

    fn deref(&self) -> &[Adaptive<E>] {
        &self.0
    }
}

impl<E: Environment> core::ops::DerefMut for Targets<E> {
    fn deref_mut(&mut self) -> &mut [Adaptive<E>] {
        &mut self.0
    }
}

impl<E: Environment> core::fmt::Debug for Targets<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Targets")
            .field("slots", &self.0.len())
            .finish()
    }
}

impl<E: Environment> core::fmt::Debug for CurriculumSlot<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CurriculumSlot")
            .field("identity", &self.identity)
            .field("curriculum", &self.held.is_some())
            .field("frozen", &self.frozen)
            .finish_non_exhaustive()
    }
}

/// A pool's curriculum subsystem: identity, optional curriculum, the
/// changed-slot scratch and the frozen flag. The targets themselves live
/// in `EnvPool::targets`, read by the kernels.
pub(crate) struct CurriculumSlot<E: Environment> {
    pub(crate) identity: u64,
    pub(crate) held: Option<Box<dyn HeldCurriculum<E>>>,
    changed: Vec<bool>,
    pub(crate) frozen: bool,
}

impl<E: Environment> CurriculumSlot<E> {
    pub(crate) fn targets_only(slots: usize) -> Self {
        Self {
            identity: targets_identity::<E>(),
            held: None,
            changed: vec![false; slots],
            frozen: false,
        }
    }

    pub(crate) fn with<C: Curriculum<E>>(slots: usize, curriculum: C) -> Self {
        Self {
            identity: curriculum_identity::<C, E>(),
            held: Some(Box::new(Held::<C, E> {
                curriculum,
                state: C::State::state_default(),
                _env: core::marker::PhantomData,
            })),
            changed: vec![false; slots],
            frozen: false,
        }
    }

    /// The barrier (ADR-0081 item 1.2, after normalization): runs the
    /// curriculum on `report` and returns the changed slots, `None` when
    /// frozen or without a curriculum.
    pub(crate) fn barrier(
        &mut self,
        report: &EpochReport,
        values: &mut [Adaptive<E>],
    ) -> Option<CurriculumChanges> {
        if self.frozen {
            return None;
        }
        let held = self.held.as_mut()?;
        self.changed.fill(false);
        let mut targets = AdaptiveTargets {
            values,
            changed: &mut self.changed,
        };
        held.update(report, &mut targets);
        let ids = self
            .changed
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .filter_map(|(i, _)| u32::try_from(i).ok())
            .collect();
        Some(CurriculumChanges { ids })
    }

    /// The context of this subsystem and `values`.
    pub(crate) fn context(&self, values: &[Adaptive<E>]) -> CurriculumContext {
        let mut targets = Vec::with_capacity(
            values
                .len()
                .saturating_mul(<Adaptive<E> as Canonical>::ENCODED_LEN),
        );
        for v in values {
            targets.extend_from_slice(&encode_to_vec(v));
        }
        CurriculumContext {
            identity: self.identity,
            state: self
                .held
                .as_ref()
                .map(|h| h.state_bytes())
                .unwrap_or_default(),
            targets,
            frozen: self.frozen,
        }
    }
}

/// Checks a curriculum context against a pool's subsystem (`None`: the
/// pool has no targets) and decodes its targets, writing nothing.
pub(crate) fn check_curriculum<E: Environment>(
    slot: Option<&CurriculumSlot<E>>,
    slots: usize,
    context: &CurriculumContext,
) -> Result<Option<Vec<Adaptive<E>>>, EngineError> {
    let Some(slot) = slot else {
        return if context.identity == 0 {
            Ok(None)
        } else {
            Err(EngineError::AdaptiveContext {
                reason: "the context has curriculum targets and the pool has none",
            })
        };
    };
    if context.identity != slot.identity {
        return Err(EngineError::AdaptiveContext {
            reason: "the context's curriculum identity differs from the pool's",
        });
    }
    match slot.held.as_ref() {
        Some(held) => held.check_state(&context.state)?,
        None if !context.state.is_empty() => {
            return Err(EngineError::AdaptiveContext {
                reason: "a targets-only context holds no curriculum state",
            });
        }
        None => {}
    }
    let len = <Adaptive<E> as Canonical>::ENCODED_LEN;
    if context.targets.len() != slots.saturating_mul(len) {
        return Err(EngineError::AdaptiveContext {
            reason: "the context's targets do not match the pool's instance count",
        });
    }
    let mut values = Vec::with_capacity(slots);
    for i in 0..slots {
        let bytes = context.targets.get(i * len..(i + 1) * len).unwrap_or(&[]);
        values.push(decode_from_slice::<Adaptive<E>>(bytes).map_err(|_| {
            EngineError::AdaptiveContext {
                reason: "a target does not decode",
            }
        })?);
    }
    Ok(Some(values))
}

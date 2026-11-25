// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attaching a curriculum or targets to a pool, and setting targets
//! (ADR-0081 items 5 and 7.5, ADR-0091 decision 3).

use tessera_rl_core::encoding::Canonical;
use tessera_rl_core::{Adaptive, AdaptiveValues, Environment, RecordLayout};

use super::{Curriculum, CurriculumSlot};
use crate::error::EngineError;
use crate::masks::MaskProvider;
use crate::pool::EnvPool;

impl<E, M> EnvPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    /// Attaches a curriculum (ADR-0081 item 5.2): targets start at the
    /// config's adaptive values, and the curriculum updates them at each
    /// epoch barrier. Before the first reset, once.
    ///
    /// # Errors
    ///
    /// [`EngineError::AdaptiveNotRead`] when the environment has no
    /// adaptive fields or does not read them (ADR-0091 decision 3), and
    /// [`EngineError::Options`] after the first reset or with targets
    /// already attached.
    pub fn with_curriculum<C: Curriculum<E>>(mut self, curriculum: C) -> Result<Self, EngineError> {
        self.check_targets_allowed()?;
        let n = self.headers.len();
        self.attach_targets(CurriculumSlot::with(n, curriculum));
        Ok(self)
    }

    /// Enables targets without a curriculum: replays of curriculum runs
    /// (ADR-0081 item 7.5) and curricula driven from outside the engine
    /// (Python) set them with [`Self::set_targets`]. Refused as
    /// [`Self::with_curriculum`].
    ///
    /// # Errors
    ///
    /// As [`Self::with_curriculum`].
    pub fn with_targets(mut self) -> Result<Self, EngineError> {
        self.check_targets_allowed()?;
        let n = self.headers.len();
        self.attach_targets(CurriculumSlot::targets_only(n));
        Ok(self)
    }

    /// The current targets, one per slot; `None` without targets.
    #[must_use]
    pub fn targets(&self) -> Option<&[Adaptive<E>]> {
        self.targets.as_deref()
    }

    /// Sets the targets of the listed slots at a call boundary; each
    /// applies at that slot's next reset. Every id is checked before any
    /// write.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], [`EngineError::AdaptiveNotRead`] without
    /// targets, or [`EngineError::InstanceOutOfRange`].
    pub fn set_targets(&mut self, changes: &[(u32, Adaptive<E>)]) -> Result<(), EngineError> {
        self.guarded(crate::pool::Poison::Reject, |pool| {
            let n = pool.headers.len();
            let targets = pool.targets.as_mut().ok_or(EngineError::AdaptiveNotRead {
                reason: "the pool has no targets (with_curriculum or with_targets)",
            })?;
            for &(id, _) in changes {
                if usize::try_from(id).map_or(true, |i| i >= n) {
                    return Err(EngineError::InstanceOutOfRange {
                        id,
                        num_envs: pool.options.num_envs.get(),
                    });
                }
            }
            for &(id, value) in changes {
                if let Some(t) = usize::try_from(id).ok().and_then(|i| targets.get_mut(i)) {
                    *t = value;
                }
            }
            Ok(())
        })
    }

    /// Freezes (or unfreezes) the curriculum: its barrier changes nothing.
    pub fn freeze_curriculum(&mut self, frozen: bool) {
        if let Some(c) = self.curriculum.as_mut() {
            c.frozen = frozen;
        }
    }

    fn check_targets_allowed(&self) -> Result<(), EngineError> {
        if <Adaptive<E> as Canonical>::ENCODED_LEN == 0 {
            return Err(EngineError::AdaptiveNotRead {
                reason: "the environment has no adaptive fields",
            });
        }
        if !E::READS_ADAPTIVE {
            return Err(EngineError::AdaptiveNotRead {
                reason: "the environment does not read its adaptive values (READS_ADAPTIVE)",
            });
        }
        if self.targets.is_some() || self.headers.iter().any(|h| !h.engine().fresh.get()) {
            return Err(EngineError::Options {
                option: "curriculum",
                reason: "a curriculum or targets are attached before the first reset, once",
            });
        }
        Ok(())
    }

    fn attach_targets(&mut self, slot: CurriculumSlot<E>) {
        let n = self.headers.len();
        let initial = Adaptive::<E>::from_config(&self.cfg);
        self.targets = Some(super::Targets(vec![initial; n]));
        self.curriculum = Some(slot);
    }
}

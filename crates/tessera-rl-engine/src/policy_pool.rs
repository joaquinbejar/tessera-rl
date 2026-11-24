// SPDX-License-Identifier: MIT OR Apache-2.0

//! Setting a pool's policy map (ADR-0079 items 2 and 4): learner and
//! external groups for any environment, heuristic groups only under the
//! bound that keeps a heuristic inside the information boundary. Every
//! setter runs before the first reset.

use tessera_rl_core::{
    Bool8, Environment, Heuristic, ImperfectInformation, InfoHeuristic, PolicyGroup, RecordLayout,
};

use crate::error::EngineError;
use crate::heuristic::{HeuristicFn, HeuristicRoute, run_act, run_info};
use crate::masks::MaskProvider;
use crate::policy::PolicyMap;
use crate::pool::EnvPool;

impl<E: Environment, M: MaskProvider<E>> EnvPool<E, M> {
    /// The pool's policy map: every slot a learner unless one was set.
    #[must_use]
    pub const fn policy_map(&self) -> &PolicyMap {
        &self.policy
    }

    /// The policy group of each agent slot, in slot order (`layout()`'s
    /// `policy_groups`, ADR-0079 item 5.1).
    #[must_use]
    pub fn policy_groups(&self) -> &[PolicyGroup] {
        self.policy.groups()
    }

    /// The heuristic actions of the last step, `[N, A, discrete heads]`:
    /// each heuristic slot's computed action, zero for every other slot and
    /// for a heuristic slot that did not act (ADR-0079 item 3.5). `None`
    /// without a heuristic slot. Only `step` writes it: resets, restores
    /// and forks leave the last step's rows. Valid until the next call that
    /// takes `&mut self` (the borrow window of ADR-0003).
    #[must_use]
    pub fn heuristic_actions(&self) -> Option<&[i32]> {
        self.heuristic
            .as_ref()
            .map(|_| self.heuristic_discrete.as_slice())
    }

    /// As [`Self::heuristic_actions`], for the continuous heads,
    /// `[N, A, continuous heads]`.
    #[must_use]
    pub fn heuristic_continuous(&self) -> Option<&[f32]> {
        self.heuristic
            .as_ref()
            .map(|_| self.heuristic_continuous.as_slice())
    }

    /// Sets the learner and external groups (ADR-0079 item 2.2), before the
    /// first reset and before any heuristic slot. A map without a
    /// heuristic group, one group per agent slot. External slots are only
    /// marked: the caller still submits their actions.
    ///
    /// # Errors
    ///
    /// [`EngineError::PolicyMap`] for a map of another slot count, a
    /// heuristic group (set those with the heuristic setters), a call
    /// after the first reset, or after heuristic slots were set.
    pub fn with_policy_map(mut self, map: PolicyMap) -> Result<Self, EngineError> {
        self.check_before_reset()?;
        if map.groups().len() != E::MAX_AGENTS {
            return Err(EngineError::PolicyMap {
                reason: "the map needs one group per agent slot",
            });
        }
        if map.has_heuristic() {
            return Err(EngineError::PolicyMap {
                reason: "heuristic groups are set with with_heuristic_slots, \
                         with_info_heuristic_slots or with_privileged_heuristic_slots",
            });
        }
        if self.heuristic.is_some() {
            return Err(EngineError::PolicyMap {
                reason: "set the policy map before the heuristic slots",
            });
        }
        self.policy = map;
        Ok(self)
    }

    /// Refuses a policy setter after the first reset: every record must
    /// still be fresh.
    fn check_before_reset(&self) -> Result<(), EngineError> {
        if self.headers.iter().any(|h| !h.engine().fresh.get()) {
            return Err(EngineError::PolicyMap {
                reason: "the policy map is set before the first reset",
            });
        }
        Ok(())
    }

    /// Puts `slots` in `group`, run by `run`: the common part of the
    /// heuristic setters. Once per pool; an empty list changes nothing.
    fn set_heuristic(
        mut self,
        slots: &[usize],
        group: PolicyGroup,
        run: HeuristicFn<E>,
    ) -> Result<Self, EngineError> {
        self.check_before_reset()?;
        if <E::State as tessera_rl_core::EnvState>::LANES {
            // The heuristic pass reads the chunk's state rows, which a
            // lane-wise state does not have (ADR-0090, head 1).
            return Err(EngineError::PolicyMap {
                reason: "heuristic slots need an Array-of-Structs state, not #[state(lanes)]",
            });
        }
        if self.heuristic.is_some() {
            return Err(EngineError::PolicyMap {
                reason: "heuristic slots are set once",
            });
        }
        if slots.is_empty() {
            return Ok(self);
        }
        let mut map = self.policy.clone();
        for (k, &slot) in slots.iter().enumerate() {
            if slots.get(..k).is_some_and(|seen| seen.contains(&slot)) {
                return Err(EngineError::PolicyMap {
                    reason: "a slot appears more than once",
                });
            }
            map = map.assign(slot, group)?;
        }
        let rows = self
            .headers
            .len()
            .checked_mul(E::MAX_AGENTS)
            .ok_or_else(crate::pool::too_large)?;
        let (dh, ch) = (self.actions.discrete.len(), self.actions.continuous.len());
        let discrete = vec![0; rows.checked_mul(dh).ok_or_else(crate::pool::too_large)?];
        let continuous = vec![0.0; rows.checked_mul(ch).ok_or_else(crate::pool::too_large)?];
        self.actions.heuristic = map
            .groups()
            .iter()
            .map(|g| Bool8::from(g.is_heuristic()))
            .collect();
        self.heuristic = Some(HeuristicRoute {
            run,
            slots: map.heuristic_slots(),
        });
        self.heuristic_discrete = discrete;
        self.heuristic_continuous = continuous;
        self.policy = map;
        // The boundary's facts change with the skipped rows.
        self.forget_fast_facts();
        Ok(self)
    }
}

impl<E: Heuristic, M: MaskProvider<E>> EnvPool<E, M> {
    /// Makes `slots` heuristic slots of a perfect-information environment
    /// (ADR-0079 item 4.1): each step, before the kernel, the engine calls
    /// `Heuristic::act` on the state for every alive, acting one, and the
    /// caller's rows for them are never read. Before the first reset, once.
    ///
    /// # Errors
    ///
    /// [`EngineError::PolicyMap`] for an imperfect-information environment
    /// (use [`Self::with_info_heuristic_slots`], or
    /// [`Self::with_privileged_heuristic_slots`] for a cheating opponent),
    /// a slot out of range or repeated, a call after the first reset, or a
    /// second call.
    pub fn with_heuristic_slots(self, slots: &[usize]) -> Result<Self, EngineError> {
        if !E::PERFECT_INFORMATION {
            return Err(EngineError::PolicyMap {
                reason: "an imperfect-information environment needs InfoHeuristic \
                         (with_info_heuristic_slots) or with_privileged_heuristic_slots",
            });
        }
        let run: HeuristicFn<E> = run_act::<E>;
        self.set_heuristic(slots, PolicyGroup::Heuristic, run)
    }
}

impl<E: InfoHeuristic, M: MaskProvider<E>> EnvPool<E, M> {
    /// Makes `slots` heuristic slots of an imperfect-information
    /// environment (ADR-0079 item 4.2): the engine passes only
    /// `info_state(cfg, state, agent)` to `InfoHeuristic::act_info`, so the
    /// heuristic stays inside the information boundary (ADR-0035). Before
    /// the first reset, once.
    ///
    /// # Errors
    ///
    /// As [`Self::with_heuristic_slots`], without the information check.
    pub fn with_info_heuristic_slots(self, slots: &[usize]) -> Result<Self, EngineError> {
        let run: HeuristicFn<E> = run_info::<E>;
        self.set_heuristic(slots, PolicyGroup::Heuristic, run)
    }
}

impl<E: Heuristic + ImperfectInformation, M: MaskProvider<E>> EnvPool<E, M> {
    /// Makes `slots` privileged heuristic slots (ADR-0079 item 4.3):
    /// `Heuristic::act` on the full state of an imperfect-information
    /// environment. These are cheating opponents, for testing only;
    /// `layout()` marks them `"privileged_heuristic"`. Before the first
    /// reset, once.
    ///
    /// # Errors
    ///
    /// As [`Self::with_heuristic_slots`], without the information check.
    pub fn with_privileged_heuristic_slots(self, slots: &[usize]) -> Result<Self, EngineError> {
        let run: HeuristicFn<E> = run_act::<E>;
        self.set_heuristic(slots, PolicyGroup::PrivilegedHeuristic, run)
    }
}

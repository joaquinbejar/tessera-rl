// SPDX-License-Identifier: MIT OR Apache-2.0

//! The policy map (ADR-0079 item 2): the policy group of each agent slot,
//! fixed for the pool's life, and its digest, part of the pool context.

use tessera_rl_core::PolicyGroup;

use crate::error::EngineError;

/// Domain of the map digest, ASCII `TSRPMAP1`.
const DIGEST_DOMAIN: u64 = 0x5453_5250_4D41_5031;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

/// The policy group of every agent slot of a pool (ADR-0079 item 2). The
/// default is every slot [`PolicyGroup::Learner`]: a pool without a map
/// behaves exactly as before.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMap {
    groups: Vec<PolicyGroup>,
}

impl PolicyMap {
    /// A map of `slots` agent slots, every one a learner.
    #[must_use]
    pub fn new(slots: usize) -> Self {
        Self {
            groups: vec![PolicyGroup::Learner; slots],
        }
    }

    /// A map with one group per slot, in slot order.
    #[must_use]
    pub const fn from_groups(groups: Vec<PolicyGroup>) -> Self {
        Self { groups }
    }

    /// The map with `slot` in `group`.
    ///
    /// # Errors
    ///
    /// [`EngineError::PolicyMap`] for a slot past the map's slots.
    pub fn assign(mut self, slot: usize, group: PolicyGroup) -> Result<Self, EngineError> {
        let entry = self.groups.get_mut(slot).ok_or(EngineError::PolicyMap {
            reason: "slot out of range",
        })?;
        *entry = group;
        Ok(self)
    }

    /// The group of each slot, in slot order.
    #[must_use]
    pub fn groups(&self) -> &[PolicyGroup] {
        &self.groups
    }

    /// The group of `slot`; `None` past the last slot.
    #[must_use]
    pub fn group(&self, slot: usize) -> Option<PolicyGroup> {
        self.groups.get(slot).copied()
    }

    /// Whether some slot is in a heuristic group.
    #[must_use]
    pub fn has_heuristic(&self) -> bool {
        self.groups.iter().any(|g| g.is_heuristic())
    }

    /// Whether every slot is a learner (the default map).
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.groups.iter().all(|&g| g == PolicyGroup::Learner)
    }

    /// The slots in a heuristic group, in slot order.
    #[must_use]
    pub fn heuristic_slots(&self) -> Vec<u16> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.is_heuristic())
            .filter_map(|(slot, _)| u16::try_from(slot).ok())
            .collect()
    }

    /// The map's digest (ADR-0079 item 2.3): 0 for the default map,
    /// whatever its slot count, so contexts built before policy maps
    /// existed match a pool without one; otherwise FNV-1a over a domain,
    /// the slot count and each slot's group code, never 0.
    #[must_use]
    pub fn digest(&self) -> u64 {
        if self.is_default() {
            return 0;
        }
        let mut h = DIGEST_DOMAIN;
        let count = u64::try_from(self.groups.len()).unwrap_or(u64::MAX);
        for byte in count
            .to_le_bytes()
            .into_iter()
            .chain(self.groups.iter().map(|g| g.code()))
        {
            h ^= u64::from(byte);
            // FNV-1a multiplies modulo 2^64 by definition.
            h = h.wrapping_mul(FNV_PRIME);
        }
        h.max(1)
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::PolicyGroup;

    use super::PolicyMap;
    use crate::error::EngineError;

    #[test]
    fn test_policy_map_new_every_slot_learner_digest_zero() {
        for slots in [1, 2, 8] {
            let map = PolicyMap::new(slots);
            assert!(map.is_default());
            assert!(!map.has_heuristic());
            assert_eq!(map.digest(), 0);
        }
    }

    #[test]
    fn test_policy_map_assign_out_of_range_refused() {
        assert_eq!(
            PolicyMap::new(2).assign(2, PolicyGroup::External),
            Err(EngineError::PolicyMap {
                reason: "slot out of range"
            })
        );
    }

    #[test]
    fn test_policy_map_digest_distinguishes_groups_and_slots() {
        let a = PolicyMap::new(2)
            .assign(1, PolicyGroup::Heuristic)
            .unwrap_or_else(|_| PolicyMap::new(0));
        let b = PolicyMap::new(2)
            .assign(0, PolicyGroup::Heuristic)
            .unwrap_or_else(|_| PolicyMap::new(0));
        let c = PolicyMap::new(2)
            .assign(1, PolicyGroup::External)
            .unwrap_or_else(|_| PolicyMap::new(0));
        let d = PolicyMap::new(3)
            .assign(1, PolicyGroup::Heuristic)
            .unwrap_or_else(|_| PolicyMap::new(0));
        let digests = [a.digest(), b.digest(), c.digest(), d.digest()];
        for (i, x) in digests.iter().enumerate() {
            assert_ne!(*x, 0);
            for y in digests.iter().skip(i + 1) {
                assert_ne!(x, y);
            }
        }
        assert_eq!(a.heuristic_slots(), vec![1]);
        assert!(a.has_heuristic() && !c.has_heuristic());
    }
}

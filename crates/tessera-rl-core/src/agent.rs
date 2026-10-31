// SPDX-License-Identifier: MIT OR Apache-2.0

//! Agent slots and determinism tiers (SPEC 8.4, ADR-0013).

/// An agent slot, `0..MAX_AGENTS`.
///
/// Slots are validated when they are created, so kernels can index
/// per-agent buffers with them without failing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct AgentSlot(u16);

impl AgentSlot {
    /// The only slot of a single-agent environment.
    pub const FIRST: Self = Self(0);

    /// Returns the slot `index` if it is below `max_agents`.
    #[must_use]
    #[inline]
    pub const fn new(index: u16, max_agents: usize) -> Option<Self> {
        if (index as usize) < max_agents {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Returns the slot index.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The determinism tier an environment declares (SPEC 8.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Bit-identical trajectories on every supported platform.
    CrossPlatform,
    /// Bit-identical trajectories on the same OS, ISA path and build.
    SamePlatform,
}

#[cfg(test)]
mod tests {
    use super::AgentSlot;

    #[test]
    fn test_agent_slot_new_rejects_out_of_range() {
        assert_eq!(AgentSlot::new(0, 1), Some(AgentSlot::FIRST));
        assert_eq!(AgentSlot::new(1, 1), None);
        assert_eq!(AgentSlot::new(3, 4).map(AgentSlot::index), Some(3));
        assert_eq!(AgentSlot::new(0, 0), None);
    }
}

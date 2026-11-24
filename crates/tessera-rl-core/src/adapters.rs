// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declared adapter capabilities (SPEC 12, ADR-0013, ADR-0080 item 1).

use crate::agent::TurnStructure;

/// The PettingZoo interfaces an environment declares (ADR-0080 item 1).
///
/// A bit set. Only what SPEC 12 calls adapter capabilities is declared:
/// Gymnasium, PufferLib and JAX acceptance follows from the layout and is
/// not part of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Adapters(u8);

impl Adapters {
    /// No PettingZoo interface.
    pub const NONE: Self = Self(0);
    /// `tessera_rl.pz.parallel_env`.
    pub const PZ_PARALLEL: Self = Self(1);
    /// `tessera_rl.pz.aec_env`: native for `Sequential`, through cycle
    /// buffering for `Simultaneous`.
    pub const PZ_AEC: Self = Self(1 << 1);
    /// SPEC 12's optional explicit turn-masked interface,
    /// `tessera_rl.pz.turn_masked_env`; `Sequential` only.
    pub const PZ_TURN_MASKED: Self = Self(1 << 2);

    const KNOWN: u8 = 0b111;

    /// The set of both `self` and `other`.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every interface of `other` is in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The raw bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The set of `bits`, or `None` when a reserved bit is set.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::KNOWN == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// The default of a host environment (ADR-0080 item 3), equal to the
    /// Python adapters' table for host environments: `Simultaneous` gives
    /// Parallel and AEC, `Sequential` AEC, `Mixed` Parallel (ADR-0065 item 6).
    #[must_use]
    pub const fn for_turns(turns: TurnStructure) -> Self {
        match turns {
            TurnStructure::Simultaneous => Self::PZ_PARALLEL.union(Self::PZ_AEC),
            TurnStructure::Sequential => Self::PZ_AEC,
            TurnStructure::Mixed => Self::PZ_PARALLEL,
        }
    }

    /// Whether the set is allowed for `turns` (ADR-0080 item 2): turn-masked
    /// only with `Sequential`, AEC only with `Simultaneous` or `Sequential`.
    #[must_use]
    pub const fn valid_for(self, turns: TurnStructure) -> bool {
        let turn_masked_ok =
            !self.contains(Self::PZ_TURN_MASKED) || matches!(turns, TurnStructure::Sequential);
        let aec_ok = !self.contains(Self::PZ_AEC) || !matches!(turns, TurnStructure::Mixed);
        turn_masked_ok && aec_ok
    }
}

#[cfg(test)]
mod tests {
    use super::{Adapters, TurnStructure};

    const BOTH: Adapters = Adapters::PZ_PARALLEL.union(Adapters::PZ_AEC);

    #[test]
    fn test_for_turns_each_structure_matches_host_table() {
        assert_eq!(Adapters::for_turns(TurnStructure::Simultaneous), BOTH);
        assert_eq!(
            Adapters::for_turns(TurnStructure::Sequential),
            Adapters::PZ_AEC
        );
        assert_eq!(
            Adapters::for_turns(TurnStructure::Mixed),
            Adapters::PZ_PARALLEL
        );
    }

    #[test]
    fn test_from_bits_reserved_bit_none() {
        assert_eq!(
            Adapters::from_bits(0b111),
            Some(BOTH.union(Adapters::PZ_TURN_MASKED))
        );
        assert_eq!(Adapters::from_bits(0), Some(Adapters::NONE));
        assert_eq!(Adapters::from_bits(0b1000), None);
        assert_eq!(Adapters::from_bits(0xff), None);
    }

    #[test]
    fn test_contains_subset_and_superset_as_expected() {
        assert!(BOTH.contains(Adapters::PZ_AEC));
        assert!(BOTH.contains(Adapters::NONE));
        assert!(!Adapters::PZ_AEC.contains(BOTH));
        assert!(Adapters::NONE.is_empty());
        assert!(!BOTH.is_empty());
    }

    #[test]
    fn test_valid_for_turn_masked_only_sequential() {
        let tm = Adapters::PZ_TURN_MASKED;
        assert!(tm.valid_for(TurnStructure::Sequential));
        assert!(!tm.valid_for(TurnStructure::Simultaneous));
        assert!(!tm.valid_for(TurnStructure::Mixed));
    }

    #[test]
    fn test_valid_for_aec_with_mixed_false() {
        assert!(!Adapters::PZ_AEC.valid_for(TurnStructure::Mixed));
        assert!(Adapters::PZ_PARALLEL.valid_for(TurnStructure::Mixed));
        assert!(BOTH.valid_for(TurnStructure::Simultaneous));
        assert!(Adapters::PZ_AEC.valid_for(TurnStructure::Sequential));
        assert!(Adapters::NONE.valid_for(TurnStructure::Mixed));
    }
}

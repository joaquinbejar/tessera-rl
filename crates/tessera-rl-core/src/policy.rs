// SPDX-License-Identifier: MIT OR Apache-2.0

//! Policy groups of a pool's agent slots (ADR-0079 item 1): who computes a
//! slot's action.

/// Who computes the action of an agent slot (ADR-0079). A slot's group is
/// fixed for the pool's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PolicyGroup {
    /// The policy being trained: the caller submits the action.
    #[default]
    Learner,
    /// A frozen policy serviced by the trainer: the caller submits the
    /// action; the engine only marks the slot.
    External,
    /// An in-engine scripted opponent. On an imperfect-information
    /// environment it sees only the information state (`InfoHeuristic`).
    Heuristic,
    /// An in-engine scripted opponent that reads the full state of an
    /// imperfect-information environment: a cheating opponent, for
    /// testing only (ADR-0079 item 4.3).
    PrivilegedHeuristic,
    /// A league opponent (ADR-0088 section 2): every `Opponent` slot of an
    /// instance plays the policy the league assigned that instance for the
    /// episode, computed by the engine (`Heuristic`) or submitted by the
    /// caller (`External`). Set only through `EnvPool::with_league`.
    Opponent,
}

impl PolicyGroup {
    /// Every group, in code order.
    pub const ALL: [Self; 5] = [
        Self::Learner,
        Self::External,
        Self::Heuristic,
        Self::PrivilegedHeuristic,
        Self::Opponent,
    ];

    /// The group's code in the replay header (ADR-0079 item 6.2): 0
    /// learner, 1 external, 2 heuristic, 3 privileged heuristic, 4
    /// opponent (ADR-0088 section 8).
    #[must_use]
    #[inline]
    pub const fn code(self) -> u8 {
        match self {
            Self::Learner => 0,
            Self::External => 1,
            Self::Heuristic => 2,
            Self::PrivilegedHeuristic => 3,
            Self::Opponent => 4,
        }
    }

    /// The group of `code`; `None` for an unknown code.
    #[must_use]
    #[inline]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Learner),
            1 => Some(Self::External),
            2 => Some(Self::Heuristic),
            3 => Some(Self::PrivilegedHeuristic),
            4 => Some(Self::Opponent),
            _ => None,
        }
    }

    /// The group's name in `layout()` (ADR-0079 item 5.1).
    #[must_use]
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Learner => "learner",
            Self::External => "external",
            Self::Heuristic => "heuristic",
            Self::PrivilegedHeuristic => "privileged_heuristic",
            Self::Opponent => "opponent",
        }
    }

    /// The group named `name`; `None` for an unknown name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.name() == name)
    }

    /// Whether the engine always computes the slot's action (a heuristic
    /// group). An `Opponent` slot is computed only when its instance's
    /// assignment is a heuristic policy, so it is not one.
    #[must_use]
    #[inline]
    pub const fn is_heuristic(self) -> bool {
        matches!(self, Self::Heuristic | Self::PrivilegedHeuristic)
    }
}

#[cfg(test)]
mod tests {
    use super::PolicyGroup;

    #[test]
    fn test_policy_group_codes_and_names_round_trip() {
        for g in PolicyGroup::ALL {
            assert_eq!(PolicyGroup::from_code(g.code()), Some(g));
            assert_eq!(PolicyGroup::from_name(g.name()), Some(g));
        }
        assert_eq!(PolicyGroup::from_code(4), Some(PolicyGroup::Opponent));
        assert_eq!(PolicyGroup::from_code(5), None);
        assert_eq!(PolicyGroup::from_name("onnx"), None);
        assert_eq!(PolicyGroup::default(), PolicyGroup::Learner);
    }

    #[test]
    fn test_policy_group_is_heuristic_only_for_engine_groups() {
        assert!(!PolicyGroup::Learner.is_heuristic());
        assert!(!PolicyGroup::External.is_heuristic());
        assert!(PolicyGroup::Heuristic.is_heuristic());
        assert!(PolicyGroup::PrivilegedHeuristic.is_heuristic());
    }
}

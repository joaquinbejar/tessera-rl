// SPDX-License-Identifier: MIT OR Apache-2.0

//! The registration record (ADR-0054 item 3): only facts that are consts
//! of the environment's traits.

use crate::{Environment, Persist, Tier};

/// What [`register!`](crate::register) records about an environment.
///
/// `#[non_exhaustive]` and built only by [`EnvRegistration::of`], so the
/// fields ADR-0054 defers (capabilities, constructor) can be added later
/// without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct EnvRegistration {
    id: &'static str,
    version: u32,
    max_agents: usize,
    tier: Tier,
    perfect_information: bool,
}

impl EnvRegistration {
    /// The record of `E`, read from its `Persist` and `Environment` consts.
    #[must_use]
    pub const fn of<E: Persist>() -> Self {
        Self {
            id: E::ENV_ID,
            version: E::ENV_VERSION,
            max_agents: <E as Environment>::MAX_AGENTS,
            tier: <E as Environment>::TIER,
            perfect_information: <E as Environment>::PERFECT_INFORMATION,
        }
    }

    /// A record with `id` and fixed other fields, for the lookup's tests.
    #[cfg(test)]
    pub(crate) const fn test_record(id: &'static str) -> Self {
        Self {
            id,
            version: 1,
            max_agents: 1,
            tier: Tier::CrossPlatform,
            perfect_information: true,
        }
    }

    /// The versioned id, `tessera/<Name>-v<N>` (`Persist::ENV_ID`).
    #[must_use]
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// The outputs version (`Persist::ENV_VERSION`), bumped with any change
    /// to the environment's outputs. It is not the `-vN` suffix of the id:
    /// `tessera/RTS-v1` has version 2.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// The number of agent slots (`Environment::MAX_AGENTS`).
    #[must_use]
    pub const fn max_agents(&self) -> usize {
        self.max_agents
    }

    /// The determinism tier (`Environment::TIER`, SPEC 8.4).
    #[must_use]
    pub const fn tier(&self) -> Tier {
        self.tier
    }

    /// Whether every agent sees the whole state
    /// (`Environment::PERFECT_INFORMATION`).
    #[must_use]
    pub const fn perfect_information(&self) -> bool {
        self.perfect_information
    }
}

inventory::collect!(EnvRegistration);

/// Byte equality of two ids, usable in a `const` assertion (the
/// [`register!`](crate::register) id check).
#[must_use]
pub const fn same_id(a: &str, b: &str) -> bool {
    let (mut a, mut b) = (a.as_bytes(), b.as_bytes());
    loop {
        match (a.split_first(), b.split_first()) {
            (None, None) => return true,
            (Some((x, rest_a)), Some((y, rest_b))) => {
                if *x != *y {
                    return false;
                }
                a = rest_a;
                b = rest_b;
            }
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::same_id;

    #[test]
    fn test_same_id_equal_ids_true() {
        assert!(same_id("tessera/Grid-v1", "tessera/Grid-v1"));
        assert!(same_id("", ""));
    }

    #[test]
    fn test_same_id_one_byte_differs_false() {
        assert!(!same_id("tessera/Grid-v1", "tessera/Grid-v2"));
        assert!(!same_id("a", "b"));
    }

    #[test]
    fn test_same_id_prefix_false() {
        assert!(!same_id("tessera/Grid-v1", "tessera/Grid-v"));
        assert!(!same_id("tessera/Grid-v", "tessera/Grid-v1"));
        assert!(!same_id("", "a"));
    }
}

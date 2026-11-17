// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host registration record (ADR-0062 item 5): only facts that are
//! consts of [`HostEnvironment`]. It has no tier, because a host
//! environment makes no determinism claim.

use crate::{HostEnvironment, TurnStructure};

/// What [`register_host!`](crate::register_host) records about a host
/// environment.
///
/// `#[non_exhaustive]` and built only by [`HostRegistration::of`], so fields
/// can be added later without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct HostRegistration {
    id: &'static str,
    version: u32,
    max_agents: usize,
    turns: TurnStructure,
    perfect_information: bool,
}

impl HostRegistration {
    /// The record of `E`, read from its `HostEnvironment` consts.
    #[must_use]
    pub const fn of<E: HostEnvironment>() -> Self {
        Self {
            id: E::ENV_ID,
            version: E::ENV_VERSION,
            max_agents: E::MAX_AGENTS,
            turns: E::TURNS,
            perfect_information: E::PERFECT_INFORMATION,
        }
    }

    /// A record with `id` and fixed other fields, for the lookup's tests.
    #[cfg(test)]
    pub(crate) const fn test_record(id: &'static str) -> Self {
        Self {
            id,
            version: 1,
            max_agents: 1,
            turns: TurnStructure::Simultaneous,
            perfect_information: true,
        }
    }

    /// The versioned id, `<namespace>/<Name>-v<N>` (`HostEnvironment::ENV_ID`).
    #[must_use]
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// The outputs version (`HostEnvironment::ENV_VERSION`).
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// The number of agent slots (`HostEnvironment::MAX_AGENTS`).
    #[must_use]
    pub const fn max_agents(&self) -> usize {
        self.max_agents
    }

    /// The turn structure (`HostEnvironment::TURNS`).
    #[must_use]
    pub const fn turns(&self) -> TurnStructure {
        self.turns
    }

    /// Whether every agent sees the whole state
    /// (`HostEnvironment::PERFECT_INFORMATION`).
    #[must_use]
    pub const fn perfect_information(&self) -> bool {
        self.perfect_information
    }
}

inventory::collect!(HostRegistration);

/// The namespace reserved for first-party environments.
pub const RESERVED_NAMESPACE: &str = "tessera/";

/// Whether `id` is in the reserved first-party namespace, usable in a
/// `const` assertion (the [`register_host!`](crate::register_host) check).
#[must_use]
pub const fn in_reserved_namespace(id: &str) -> bool {
    let (mut id, mut prefix) = (id.as_bytes(), RESERVED_NAMESPACE.as_bytes());
    loop {
        match (prefix.split_first(), id.split_first()) {
            (None, _) => return true,
            (Some((p, rest_p)), Some((c, rest_id))) => {
                if *p != *c {
                    return false;
                }
                prefix = rest_p;
                id = rest_id;
            }
            (Some(_), None) => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::in_reserved_namespace;

    #[test]
    fn test_in_reserved_namespace_tessera_prefix_true() {
        assert!(in_reserved_namespace("tessera/Grid-v1"));
        assert!(in_reserved_namespace("tessera/"));
    }

    #[test]
    fn test_in_reserved_namespace_other_ids_false() {
        assert!(!in_reserved_namespace("example/OrderBook-v1"));
        assert!(!in_reserved_namespace("tessera"));
        assert!(!in_reserved_namespace("tesseract/X-v1"));
        assert!(!in_reserved_namespace(""));
    }
}

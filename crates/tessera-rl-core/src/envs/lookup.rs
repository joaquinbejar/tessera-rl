// SPDX-License-Identifier: MIT OR Apache-2.0

//! The first-use lookup (ADR-0054 item 4, as amended by ADR-0062 item 5):
//! `register!` and `register_host!` share one id space.

use std::sync::OnceLock;

use thiserror::Error;

use super::host_record::HostRegistration;
use super::record::EnvRegistration;

/// A registry lookup failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RegistryError {
    /// Two linked registrations use the same id: two `register!` calls, two
    /// `register_host!` calls, or one of each (ADR-0032, ADR-0054,
    /// ADR-0062).
    #[error("environment id {id:?} is registered more than once")]
    DuplicateId {
        /// The duplicated id.
        id: &'static str,
    },
}

/// Both record lists, each sorted by id.
#[derive(Debug, PartialEq)]
struct Lists {
    env: Vec<EnvRegistration>,
    host: Vec<HostRegistration>,
}

/// The collected records: written once, at the first lookup of either list.
static REGISTERED: OnceLock<Result<Lists, RegistryError>> = OnceLock::new();

fn lists() -> Result<&'static Lists, RegistryError> {
    match REGISTERED.get_or_init(|| {
        collect(
            inventory::iter::<EnvRegistration>.into_iter().copied(),
            inventory::iter::<HostRegistration>.into_iter().copied(),
        )
    }) {
        Ok(lists) => Ok(lists),
        Err(e) => Err(*e),
    }
}

/// Every environment registered with `register!` in this binary, sorted by
/// id.
///
/// The first call of this function or of [`registered_host`] collects the
/// submissions; later calls return the same result. The order does not
/// depend on link order.
///
/// # Errors
///
/// [`RegistryError::DuplicateId`] when two registrations share an id,
/// including a `register!` id that equals a `register_host!` id, on the
/// first call and every later one.
pub fn registered() -> Result<&'static [EnvRegistration], RegistryError> {
    lists().map(|l| l.env.as_slice())
}

/// Every host environment registered with `register_host!` in this binary,
/// sorted by id (ADR-0062 item 5).
///
/// # Errors
///
/// As [`registered`]: the two lookups fail together.
pub fn registered_host() -> Result<&'static [HostRegistration], RegistryError> {
    lists().map(|l| l.host.as_slice())
}

/// Sorts both lists by id and refuses an id that appears twice across
/// them; the reported id is the first duplicate in id order.
fn collect(
    env: impl IntoIterator<Item = EnvRegistration>,
    host: impl IntoIterator<Item = HostRegistration>,
) -> Result<Lists, RegistryError> {
    let mut env: Vec<EnvRegistration> = env.into_iter().collect();
    env.sort_unstable_by(|a, b| a.id().cmp(b.id()));
    let mut host: Vec<HostRegistration> = host.into_iter().collect();
    host.sort_unstable_by(|a, b| a.id().cmp(b.id()));
    let mut ids: Vec<&'static str> = env
        .iter()
        .map(EnvRegistration::id)
        .chain(host.iter().map(HostRegistration::id))
        .collect();
    ids.sort_unstable();
    for pair in ids.windows(2) {
        if let [a, b] = pair
            && a == b
        {
            return Err(RegistryError::DuplicateId { id: a });
        }
    }
    Ok(Lists { env, host })
}

#[cfg(test)]
mod tests {
    use super::{RegistryError, collect};
    use crate::envs::{EnvRegistration, HostRegistration};

    fn env(id: &'static str) -> EnvRegistration {
        EnvRegistration::test_record(id)
    }

    fn host(id: &'static str) -> HostRegistration {
        HostRegistration::test_record(id)
    }

    fn env_ids(env: &[EnvRegistration]) -> Vec<&'static str> {
        env.iter().map(EnvRegistration::id).collect()
    }

    fn host_ids(host: &[HostRegistration]) -> Vec<&'static str> {
        host.iter().map(HostRegistration::id).collect()
    }

    #[test]
    fn test_collect_unordered_input_sorted_by_id() {
        let out = collect(
            [env("t/C-v1"), env("t/A-v1"), env("t/B-v1")],
            [host("h/Z-v1"), host("h/Y-v1")],
        );
        let out = out.map(|l| (env_ids(&l.env), host_ids(&l.host)));
        assert_eq!(
            out,
            Ok((vec!["t/A-v1", "t/B-v1", "t/C-v1"], vec!["h/Y-v1", "h/Z-v1"]))
        );
    }

    #[test]
    fn test_collect_duplicate_env_id_error() {
        let out = collect([env("t/B-v1"), env("t/A-v1"), env("t/B-v1")], []);
        assert_eq!(out, Err(RegistryError::DuplicateId { id: "t/B-v1" }));
    }

    #[test]
    fn test_collect_duplicate_host_id_error() {
        let out = collect([env("t/A-v1")], [host("h/X-v1"), host("h/X-v1")]);
        assert_eq!(out, Err(RegistryError::DuplicateId { id: "h/X-v1" }));
    }

    #[test]
    fn test_collect_id_in_both_lists_error() {
        let out = collect([env("x/Same-v1")], [host("x/Same-v1")]);
        assert_eq!(out, Err(RegistryError::DuplicateId { id: "x/Same-v1" }));
    }

    #[test]
    fn test_collect_reports_first_duplicate_in_id_order() {
        let out = collect(
            [env("t/B-v1"), env("t/B-v1")],
            [host("a/A-v1"), host("a/A-v1")],
        );
        assert_eq!(out, Err(RegistryError::DuplicateId { id: "a/A-v1" }));
    }

    #[test]
    fn test_collect_empty_ok() {
        let out = collect([], []).map(|l| (l.env.len(), l.host.len()));
        assert_eq!(out, Ok((0, 0)));
    }
}

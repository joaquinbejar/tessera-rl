// SPDX-License-Identifier: MIT OR Apache-2.0

//! The first-use lookup (ADR-0054 item 4).

use std::sync::OnceLock;

use thiserror::Error;

use super::record::EnvRegistration;

/// A registry lookup failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RegistryError {
    /// Two linked `register!` calls use the same id (ADR-0032, ADR-0054).
    #[error("environment id {id:?} is registered more than once")]
    DuplicateId {
        /// The duplicated id.
        id: &'static str,
    },
}

/// The collected records, sorted by id: written once, at the first lookup.
static REGISTERED: OnceLock<Result<Vec<EnvRegistration>, RegistryError>> = OnceLock::new();

/// Every environment registered in this binary, sorted by id.
///
/// The first call collects the `register!` submissions; later calls return
/// the same result. The order does not depend on link order.
///
/// # Errors
///
/// [`RegistryError::DuplicateId`] when two registrations share an id, on
/// the first call and every later one.
pub fn registered() -> Result<&'static [EnvRegistration], RegistryError> {
    match REGISTERED
        .get_or_init(|| sorted_unique(inventory::iter::<EnvRegistration>.into_iter().copied()))
    {
        Ok(records) => Ok(records.as_slice()),
        Err(e) => Err(*e),
    }
}

/// Sorts `records` by id and refuses a duplicate id.
fn sorted_unique(
    records: impl IntoIterator<Item = EnvRegistration>,
) -> Result<Vec<EnvRegistration>, RegistryError> {
    let mut records: Vec<EnvRegistration> = records.into_iter().collect();
    records.sort_unstable_by(|a, b| a.id().cmp(b.id()));
    for pair in records.windows(2) {
        if let [a, b] = pair
            && a.id() == b.id()
        {
            return Err(RegistryError::DuplicateId { id: a.id() });
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::{RegistryError, sorted_unique};
    use crate::envs::EnvRegistration;

    fn ids(records: &[EnvRegistration]) -> Vec<&'static str> {
        records.iter().map(EnvRegistration::id).collect()
    }

    #[test]
    fn test_sorted_unique_unordered_input_sorted_by_id() {
        let out = sorted_unique([
            EnvRegistration::test_record("t/C-v1"),
            EnvRegistration::test_record("t/A-v1"),
            EnvRegistration::test_record("t/B-v1"),
        ]);
        assert_eq!(out.map(|r| ids(&r)), Ok(vec!["t/A-v1", "t/B-v1", "t/C-v1"]));
    }

    #[test]
    fn test_sorted_unique_duplicate_id_error() {
        let out = sorted_unique([
            EnvRegistration::test_record("t/B-v1"),
            EnvRegistration::test_record("t/A-v1"),
            EnvRegistration::test_record("t/B-v1"),
        ]);
        assert_eq!(out, Err(RegistryError::DuplicateId { id: "t/B-v1" }));
    }

    #[test]
    fn test_sorted_unique_empty_ok() {
        assert_eq!(sorted_unique([]), Ok(vec![]));
    }
}

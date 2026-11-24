// SPDX-License-Identifier: MIT OR Apache-2.0

//! `register!`'s `deprecated(...)` form records the deprecation, and
//! `deprecation_of` finds it from the type (ADR-0080 items 4 and 6). A
//! separate test binary, so its registry holds only this submission.

mod registry_support;

use registry_support::{FIXTURE_ID, Fixture};
use tessera_rl_core::Adapters;
use tessera_rl_core::envs::{Deprecation, EnvRegistration, deprecation_of, registered};

tessera_rl_core::register!(
    Fixture<1>,
    id = "test/Fixture-v1",
    deprecated(since = "0.4.0", note = "use test/Fixture-v2"),
);

#[test]
#[cfg_attr(miri, ignore = "Miri does not run inventory's constructors")]
fn test_registered_deprecated_fixture_carries_deprecation() {
    let got = registered().map(|r| r.iter().map(EnvRegistration::deprecation).collect::<Vec<_>>());
    assert_eq!(got, Ok(vec![Some(Deprecation::new("0.4.0", "use test/Fixture-v2"))]));
}

#[test]
#[cfg_attr(miri, ignore = "Miri does not run inventory's constructors")]
fn test_deprecation_of_registered_type_found_other_type_none() {
    assert_eq!(
        deprecation_of::<Fixture<1>>(),
        Some((FIXTURE_ID, Deprecation::new("0.4.0", "use test/Fixture-v2")))
    );
    assert_eq!(deprecation_of::<Fixture<2>>(), None);
}

#[test]
fn test_env_registration_of_default_adapters_none_not_deprecated() {
    let r = EnvRegistration::of::<Fixture<3>>();
    assert_eq!(r.adapters(), Adapters::NONE);
    assert_eq!(r.deprecation(), None);
}

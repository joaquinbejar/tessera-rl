// SPDX-License-Identifier: MIT OR Apache-2.0

//! `EnvRegistration::of` copies every const of the environment (ADR-0054
//! item 3). Reads no `inventory` submission, so it runs under Miri.

mod registry_support;

use registry_support::{FIXTURE_ID, Fixture};
use tessera_rl_core::Tier;
use tessera_rl_core::envs::EnvRegistration;

#[test]
fn test_env_registration_of_fixture_copies_every_const() {
    let r = EnvRegistration::of::<Fixture<7>>();
    assert_eq!(r.id(), FIXTURE_ID);
    assert_eq!(r.version(), 7);
    assert_eq!(r.max_agents(), 3);
    assert_eq!(r.tier(), Tier::SamePlatform);
    assert!(!r.perfect_information());
}

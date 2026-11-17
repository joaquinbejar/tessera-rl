// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HostRegistration::of` copies every const of the host environment
//! (ADR-0062 item 5). Reads no `inventory` submission, so it runs under
//! Miri.

mod host_support;

use host_support::{HOST_ID, HostFixture};
use tessera_rl_core::TurnStructure;
use tessera_rl_core::envs::HostRegistration;

#[test]
fn test_host_registration_of_fixture_copies_every_const() {
    let r = HostRegistration::of::<HostFixture>();
    assert_eq!(r.id(), HOST_ID);
    assert_eq!(r.version(), 5);
    assert_eq!(r.max_agents(), 2);
    assert_eq!(r.turns(), TurnStructure::Sequential);
    assert!(!r.perfect_information());
}

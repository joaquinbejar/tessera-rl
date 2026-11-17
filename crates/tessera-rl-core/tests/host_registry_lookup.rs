// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `register_host!` submission is listed by `registered_host` and not by
//! `registered` (ADR-0062 item 5). A separate test binary, so its registry
//! holds only this test's submissions.

mod host_support;

use host_support::{HOST_ID, HostFixture};
use tessera_rl_core::envs::{HostRegistration, registered, registered_host};

tessera_rl_core::register_host!(HostFixture, id = "test/Host-v1");

#[test]
#[cfg_attr(miri, ignore = "Miri does not run inventory's constructors")]
fn test_registered_host_lists_host_submission_only_there() {
    let host = registered_host().map(|r| r.iter().map(HostRegistration::id).collect::<Vec<_>>());
    assert_eq!(host, Ok(vec![HOST_ID]));
    assert_eq!(registered().map(<[_]>::len), Ok(0));
}

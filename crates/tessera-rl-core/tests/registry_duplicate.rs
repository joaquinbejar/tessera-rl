// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two registrations under one id make the real `inventory` lookup fail
//! (ADR-0054 item 4). A separate test binary, so no other binary's
//! registry holds the duplicate.

mod registry_support;

use registry_support::{FIXTURE_ID, Fixture};
use tessera_rl_core::envs::{RegistryError, registered};

tessera_rl_core::register!(Fixture<1>, id = "test/Fixture-v1");
tessera_rl_core::register!(Fixture<2>, id = "test/Fixture-v1");

#[test]
#[cfg_attr(miri, ignore = "Miri does not run inventory's constructors")]
fn test_registered_duplicate_id_error_on_every_call() {
    let want = Err(RegistryError::DuplicateId { id: FIXTURE_ID });
    assert_eq!(registered(), want);
    assert_eq!(registered(), want);
}

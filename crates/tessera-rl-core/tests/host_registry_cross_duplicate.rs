// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `register!` id equal to a `register_host!` id makes both lookups fail
//! (ADR-0054 item 4 as amended by ADR-0062 item 5). A separate test
//! binary, so no other binary's registry holds the duplicate.

mod host_support;
mod registry_support;

use host_support::HostSameId;
use registry_support::{FIXTURE_ID, Fixture};
use tessera_rl_core::envs::{RegistryError, registered, registered_host};

tessera_rl_core::register!(Fixture<1>, id = "test/Fixture-v1");
tessera_rl_core::register_host!(HostSameId, id = "test/Fixture-v1");

#[test]
#[cfg_attr(miri, ignore = "Miri does not run inventory's constructors")]
fn test_registered_id_in_both_lists_error_on_both_lookups() {
    let want = RegistryError::DuplicateId { id: FIXTURE_ID };
    assert_eq!(registered().map(<[_]>::len), Err(want));
    assert_eq!(registered_host().map(<[_]>::len), Err(want));
    assert_eq!(registered().map(<[_]>::len), Err(want));
}

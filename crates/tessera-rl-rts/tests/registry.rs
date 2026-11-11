// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS is registered under its id (ADR-0054).

use tessera_rl_core::Tier;
use tessera_rl_core::envs::registered;

#[test]
fn test_registered_rts_listed_with_its_consts() {
    let records = registered().unwrap_or_default();
    let rts = records
        .iter()
        .find(|r| r.id() == tessera_rl_rts::env::ENV_ID)
        .copied();
    assert_eq!(
        rts.map(|r| (
            r.version(),
            r.max_agents(),
            r.tier(),
            r.perfect_information()
        )),
        Some((
            tessera_rl_rts::env::ENV_VERSION,
            2,
            Tier::CrossPlatform,
            false
        ))
    );
}

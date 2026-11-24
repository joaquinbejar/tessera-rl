// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots of JobShop-v1 and Vrp-v1 (ADR-0011): a pool continues from its
//! file, and the file matches its committed fixture.

#[path = "../../tessera-rl-classic/tests/persist_support/mod.rs"]
mod support;

use tessera_rl_core::Persist;
use tessera_rl_engine::{EnvPool, Masked};
use tessera_rl_ops::{JobShop, JobShopConfig, Vrp, VrpConfig};

#[test]
fn test_persist_jobshop_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<JobShop, Masked>::with_masks(JobShopConfig::default(), o).unwrap();
    support::check_continuation(&build, 6);
    support::check_fixture("persist-jobshop", env!("CARGO_MANIFEST_DIR"), &build, 6);
    assert_eq!(
        <JobShop as Persist>::ENV_ID,
        tessera_rl_ops::jobshop::ENV_ID
    );
    assert_eq!(
        <JobShop as Persist>::ENV_VERSION,
        tessera_rl_ops::jobshop::ENV_VERSION
    );
}

#[test]
fn test_persist_vrp_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Vrp, Masked>::with_masks(VrpConfig::default(), o).unwrap();
    support::check_continuation(&build, 21);
    support::check_fixture("persist-vrp", env!("CARGO_MANIFEST_DIR"), &build, 21);
    assert_eq!(<Vrp as Persist>::ENV_ID, tessera_rl_ops::vrp::ENV_ID);
    assert_eq!(
        <Vrp as Persist>::ENV_VERSION,
        tessera_rl_ops::vrp::ENV_VERSION
    );
}

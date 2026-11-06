// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots of each environment here.

#[path = "persist_support/mod.rs"]
mod support;

use tessera_rl_classic::{
    CartPole, CartPoleConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig, Tag, TagConfig,
};
use tessera_rl_core::Persist;
use tessera_rl_engine::{EnvPool, Masked, NoMasks};

#[test]
fn test_persist_tag_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), o).unwrap();
    support::check_continuation(&build, 5);
    support::check_fixture("persist-tag", env!("CARGO_MANIFEST_DIR"), &build, 5);
    assert_eq!(<Tag as Persist>::ENV_ID, tessera_rl_classic::tag::ENV_ID);
    assert_eq!(
        <Tag as Persist>::ENV_VERSION,
        tessera_rl_classic::tag::ENV_VERSION
    );
}

#[test]
fn test_persist_cartpole_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), o).unwrap();
    support::check_continuation(&build, 2);
    support::check_fixture("persist-cartpole", env!("CARGO_MANIFEST_DIR"), &build, 2);
    assert_eq!(
        <CartPole as Persist>::ENV_ID,
        tessera_rl_classic::cartpole::ENV_ID
    );
    assert_eq!(
        <CartPole as Persist>::ENV_VERSION,
        tessera_rl_classic::cartpole::ENV_VERSION
    );
}

#[test]
fn test_persist_grid_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Grid, NoMasks>::new(GridConfig::default(), o).unwrap();
    support::check_continuation(&build, 4);
    support::check_fixture("persist-grid", env!("CARGO_MANIFEST_DIR"), &build, 4);
    assert_eq!(<Grid as Persist>::ENV_ID, tessera_rl_classic::grid::ENV_ID);
    assert_eq!(
        <Grid as Persist>::ENV_VERSION,
        tessera_rl_classic::grid::ENV_VERSION
    );
}

#[test]
fn test_persist_keydoor_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<KeyDoor, Masked>::with_masks(KeyDoorConfig::default(), o).unwrap();
    support::check_continuation(&build, 5);
    support::check_fixture("persist-keydoor", env!("CARGO_MANIFEST_DIR"), &build, 5);
    assert_eq!(
        <KeyDoor as Persist>::ENV_ID,
        tessera_rl_classic::keydoor::ENV_ID
    );
    assert_eq!(
        <KeyDoor as Persist>::ENV_VERSION,
        tessera_rl_classic::keydoor::ENV_VERSION
    );
}

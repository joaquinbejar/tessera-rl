// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots of each environment here.

#[path = "persist_support/mod.rs"]
mod support;

use tessera_rl_classic::{
    BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig,
    PaddleDuel, PaddleDuelConfig, Snake, SnakeConfig, Tag, TagConfig,
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

#[test]
fn test_persist_snake_continues_and_matches_its_fixture() {
    let cfg = SnakeConfig {
        agents: 2,
        ..SnakeConfig::default()
    };
    let build = |o| EnvPool::<Snake, Masked>::with_masks(cfg.clone(), o).unwrap();
    support::check_continuation(&build, 4);
    support::check_fixture("persist-snake", env!("CARGO_MANIFEST_DIR"), &build, 4);
    assert_eq!(
        <Snake as Persist>::ENV_ID,
        tessera_rl_classic::snake::ENV_ID
    );
    assert_eq!(
        <Snake as Persist>::ENV_VERSION,
        tessera_rl_classic::snake::ENV_VERSION
    );
}

#[test]
fn test_persist_brickwall_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<BrickWall, NoMasks>::new(BrickWallConfig::default(), o).unwrap();
    support::check_continuation(&build, 3);
    support::check_fixture("persist-brickwall", env!("CARGO_MANIFEST_DIR"), &build, 3);
    assert_eq!(
        <BrickWall as Persist>::ENV_ID,
        tessera_rl_classic::brickwall::ENV_ID
    );
    assert_eq!(
        <BrickWall as Persist>::ENV_VERSION,
        tessera_rl_classic::brickwall::ENV_VERSION
    );
}

#[test]
fn test_persist_paddleduel_continues_and_matches_its_fixture() {
    let cfg = PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    };
    let build = |o| EnvPool::<PaddleDuel, NoMasks>::new(cfg.clone(), o).unwrap();
    support::check_continuation(&build, 3);
    support::check_fixture("persist-paddleduel", env!("CARGO_MANIFEST_DIR"), &build, 3);
    assert_eq!(
        <PaddleDuel as Persist>::ENV_ID,
        tessera_rl_classic::paddleduel::ENV_ID
    );
    assert_eq!(
        <PaddleDuel as Persist>::ENV_VERSION,
        tessera_rl_classic::paddleduel::ENV_VERSION
    );
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persisted snapshots of each environment here.

#[path = "../../tessera-rl-classic/tests/persist_support/mod.rs"]
mod support;

use tessera_rl_core::Persist;
use tessera_rl_engine::{EnvPool, Masked};
use tessera_rl_games::{
    Connect4, Connect4Config, Go9, Go9Config, Leduc, LeducConfig, Othello, OthelloConfig,
};

#[test]
fn test_persist_connect4_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, o).unwrap();
    support::check_continuation(&build, 7);
    support::check_fixture("persist-connect4", env!("CARGO_MANIFEST_DIR"), &build, 7);
    assert_eq!(
        <Connect4 as Persist>::ENV_ID,
        tessera_rl_games::connect4::ENV_ID
    );
    assert_eq!(
        <Connect4 as Persist>::ENV_VERSION,
        tessera_rl_games::connect4::ENV_VERSION
    );
}

#[test]
fn test_persist_othello_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Othello, Masked>::with_masks(OthelloConfig {}, o).unwrap();
    support::check_continuation(&build, 65);
    support::check_fixture("persist-othello", env!("CARGO_MANIFEST_DIR"), &build, 65);
    assert_eq!(
        <Othello as Persist>::ENV_ID,
        tessera_rl_games::othello::ENV_ID
    );
    assert_eq!(
        <Othello as Persist>::ENV_VERSION,
        tessera_rl_games::othello::ENV_VERSION
    );
}

#[test]
fn test_persist_go9_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Go9, Masked>::with_masks(Go9Config::default(), o).unwrap();
    support::check_continuation(&build, 82);
    support::check_fixture("persist-go9", env!("CARGO_MANIFEST_DIR"), &build, 82);
    assert_eq!(<Go9 as Persist>::ENV_ID, tessera_rl_games::go9::ENV_ID);
    assert_eq!(
        <Go9 as Persist>::ENV_VERSION,
        tessera_rl_games::go9::ENV_VERSION
    );
}

#[test]
fn test_persist_leduc_continues_and_matches_its_fixture() {
    let build = |o| EnvPool::<Leduc, Masked>::with_masks(LeducConfig {}, o).unwrap();
    support::check_continuation(&build, 3);
    support::check_fixture("persist-leduc", env!("CARGO_MANIFEST_DIR"), &build, 3);
    assert_eq!(<Leduc as Persist>::ENV_ID, tessera_rl_games::leduc::ENV_ID);
    assert_eq!(
        <Leduc as Persist>::ENV_VERSION,
        tessera_rl_games::leduc::ENV_VERSION
    );
}

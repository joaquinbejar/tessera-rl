// SPDX-License-Identifier: MIT OR Apache-2.0

//! The facade lists every shipped environment (ADR-0054 acceptance): a
//! crate whose submissions the linker dropped would be missing here.

use tessera_rl::classic::{
    Acrobot, BrickWall, CartPole, Grid, KeyDoor, MountainCar, PaddleDuel, Pendulum, Snake, Tag,
};
use tessera_rl::core::{Environment, Persist};
use tessera_rl::envs::{EnvRegistration, registered};
use tessera_rl::games::{Connect4, Go9, Leduc, Othello};

/// The record `E` must have, read straight from its consts.
fn expected<E: Persist>() -> (&'static str, u32, usize, tessera_rl::core::Tier, bool) {
    (
        <E as Persist>::ENV_ID,
        <E as Persist>::ENV_VERSION,
        <E as Environment>::MAX_AGENTS,
        <E as Environment>::TIER,
        <E as Environment>::PERFECT_INFORMATION,
    )
}

fn fields(r: &EnvRegistration) -> (&'static str, u32, usize, tessera_rl::core::Tier, bool) {
    (
        r.id(),
        r.version(),
        r.max_agents(),
        r.tier(),
        r.perfect_information(),
    )
}

#[test]
fn test_registered_facade_every_shipped_environment_listed() {
    let mut want = vec![
        expected::<Acrobot>(),
        expected::<BrickWall>(),
        expected::<CartPole>(),
        expected::<Grid>(),
        expected::<KeyDoor>(),
        expected::<MountainCar>(),
        expected::<PaddleDuel>(),
        expected::<Pendulum>(),
        expected::<Snake>(),
        expected::<Tag>(),
        expected::<Connect4>(),
        expected::<Go9>(),
        expected::<Leduc>(),
        expected::<Othello>(),
    ];
    want.sort_by_key(|w| w.0);
    let got: Vec<_> = registered()
        .map(|records| records.iter().map(fields).collect())
        .unwrap_or_default();
    assert_eq!(got, want);
}

#[test]
fn test_registered_record_fields_match_consts() {
    // Spot values, so a record field wired to the wrong const fails.
    let records = registered().unwrap_or_default();
    let find = |id: &str| records.iter().find(|r| r.id() == id).copied();
    let tag = find("tessera/Tag-v1");
    assert_eq!(tag.map(|r| r.max_agents()), Some(8));
    assert_eq!(tag.map(|r| r.perfect_information()), Some(false));
    let grid = find("tessera/Grid-v1");
    assert_eq!(grid.map(|r| r.max_agents()), Some(1));
    assert_eq!(grid.map(|r| r.perfect_information()), Some(true));
    assert_eq!(
        grid.map(|r| r.version()),
        Some(tessera_rl::classic::grid::ENV_VERSION)
    );
}

#[test]
fn test_registered_second_call_same_slice() {
    let first = registered().map(<[EnvRegistration]>::as_ptr);
    let second = registered().map(<[EnvRegistration]>::as_ptr);
    assert!(first.is_ok());
    assert_eq!(first, second);
}

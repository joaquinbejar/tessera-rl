// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every shipped environment of the facade keeps its action heads
//! (ADR-0062 item 6): `Space::heads` of its action space and the built
//! pool's `action_heads` against a pinned table. The table is the action
//! column of the README's environment table, in heads. RTS is pinned in
//! `tessera-rl-rts/tests/action_heads.rs`, which this crate does not see.

use core::num::NonZeroU32;

use tessera_rl::classic::{
    Acrobot, AcrobotConfig, BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Grid, GridConfig,
    KeyDoor, KeyDoorConfig, MountainCar, MountainCarConfig, PaddleDuel, PaddleDuelConfig, Pendulum,
    PendulumConfig, Snake, SnakeConfig, Tag, TagConfig,
};
use tessera_rl::core::space::ActionHeads;
use tessera_rl::core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl::engine::{EnvPool, NoMasks, PoolOptions};
use tessera_rl::games::{
    Connect4, Connect4Config, Go9, Go9Config, Leduc, LeducConfig, Othello, OthelloConfig,
};
use tessera_rl::num::identity::Seed;

/// `(discrete heads' choices, continuous heads' inclusive bounds)`.
type Heads = (Vec<u32>, Vec<(f64, f64)>);

/// What `E` reports with `cfg`: its action space's heads, and the built
/// pool's `(discrete, continuous)` head counts.
fn observed<E>(cfg: E::Config) -> (Result<Heads, String>, Result<(usize, usize), String>)
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let heads = E::action_space(&cfg)
        .map_err(|e| e.to_string())
        .and_then(|space| space.heads().map_err(|e| e.to_string()))
        .map(ActionHeads::into_parts);
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(0));
    let counts = EnvPool::<E, NoMasks>::new(cfg, options)
        .map(|pool| pool.action_heads())
        .map_err(|e| e.to_string());
    (heads, counts)
}

/// What `observed` must return for `want`.
fn expected(want: Heads) -> (Result<Heads, String>, Result<(usize, usize), String>) {
    let counts = (want.0.len(), want.1.len());
    (Ok(want), Ok(counts))
}

fn discrete(n: u32) -> Heads {
    (vec![n], vec![])
}

#[test]
fn test_action_heads_classic_envs_match_pinned_table() {
    assert_eq!(
        observed::<CartPole>(CartPoleConfig::default()),
        expected(discrete(2)),
        "CartPole"
    );
    assert_eq!(
        observed::<MountainCar>(MountainCarConfig::default()),
        expected(discrete(3)),
        "MountainCar"
    );
    assert_eq!(
        observed::<Acrobot>(AcrobotConfig::default()),
        expected(discrete(3)),
        "Acrobot"
    );
    assert_eq!(
        observed::<Pendulum>(PendulumConfig::default()),
        expected((vec![], vec![(-2.0, 2.0)])),
        "Pendulum"
    );
    assert_eq!(
        observed::<Tag>(TagConfig::default()),
        expected(discrete(5)),
        "Tag"
    );
    assert_eq!(
        observed::<Grid>(GridConfig::default()),
        expected(discrete(4)),
        "Grid"
    );
    assert_eq!(
        observed::<KeyDoor>(KeyDoorConfig::default()),
        expected(discrete(5)),
        "KeyDoor"
    );
    assert_eq!(
        observed::<Snake>(SnakeConfig::default()),
        expected(discrete(4)),
        "Snake"
    );
    assert_eq!(
        observed::<BrickWall>(BrickWallConfig::default()),
        expected(discrete(3)),
        "BrickWall"
    );
    assert_eq!(
        observed::<PaddleDuel>(PaddleDuelConfig::default()),
        expected(discrete(3)),
        "PaddleDuel"
    );
}

#[test]
fn test_action_heads_games_match_pinned_table() {
    assert_eq!(
        observed::<Connect4>(Connect4Config::default()),
        expected(discrete(7)),
        "Connect4"
    );
    assert_eq!(
        observed::<Leduc>(LeducConfig::default()),
        expected(discrete(3)),
        "Leduc"
    );
    assert_eq!(
        observed::<Othello>(OthelloConfig::default()),
        expected(discrete(65)),
        "Othello"
    );
    assert_eq!(
        observed::<Go9>(Go9Config::default()),
        expected(discrete(82)),
        "Go9"
    );
}

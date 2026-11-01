// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect Four through the engine: spaces, the turn order and the acting
//! mask, observations from each player's view, wins, draws, the mask and
//! the no-op, import validation, snapshots and the scripted baseline.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_games::connect4::board::cell;
use tessera_rl_games::{Connect4, Connect4Config, Connect4State};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

/// A full board without four in a row (found by a seeded random search).
const DRAW: &str = "545062455041104565311226266362030334314210";

fn pool_with(auto_reset: AutoReset) -> EnvPool<Connect4, Masked> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(11))
    };
    let mut p = EnvPool::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool() -> EnvPool<Connect4, Masked> {
    pool_with(AutoReset::Disabled)
}

/// Plays `col` for whoever is to move; the other slot sends column 6.
fn play(p: &mut EnvPool<Connect4, Masked>, col: i32) {
    let to_move = p.state(0).unwrap().to_move();
    let mut act = [6, 6];
    act[to_move] = col;
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn play_all(p: &mut EnvPool<Connect4, Masked>, seq: &str) {
    for ch in seq.chars() {
        play(p, i32::try_from(ch.to_digit(10).unwrap()).unwrap());
    }
}

fn acting(p: &EnvPool<Connect4, Masked>) -> Vec<bool> {
    p.outputs().acting().iter().map(|f| f.get()).collect()
}

fn mask(p: &EnvPool<Connect4, Masked>, agent: usize) -> Vec<bool> {
    p.outputs().masks()[agent * 7..agent * 7 + 7]
        .iter()
        .map(|f| f.get())
        .collect()
}

fn terminated(p: &EnvPool<Connect4, Masked>) -> bool {
    p.outputs().episode_terminated()[0].get()
}

#[test]
fn test_connect4_spaces_are_two_u8_planes_and_seven_columns() {
    let cfg = Connect4Config {};
    let expected = Space::Dict(vec![(
        "board".to_owned(),
        Space::Box {
            shape: vec![2, 6, 7],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![1],
            },
        },
    )]);
    assert_eq!(Connect4::observation_space(&cfg).unwrap(), expected);
    assert_eq!(
        Connect4::action_space(&cfg).unwrap(),
        Space::Discrete { n: 7 }
    );
}

#[test]
fn test_connect4_acting_alternates_starting_with_slot_zero() {
    let mut p = pool();
    assert_eq!(acting(&p), [true, false]);
    play(&mut p, 3);
    assert_eq!(acting(&p), [false, true]);
    play(&mut p, 3);
    assert_eq!(acting(&p), [true, false]);
    assert_eq!(p.state(0).unwrap().moves, 2);
}

#[test]
fn test_connect4_observation_is_from_each_players_view() {
    let mut p = pool();
    play(&mut p, 3);
    play(&mut p, 0);
    let obs = p.outputs().observation(0).unwrap().to_vec();
    assert_eq!(obs.len(), 2 * 84);
    // Row 5 is the bottom row: slot 0's stone in column 3, slot 1's in 0.
    let at = |agent: usize, channel: usize, row: usize, col: usize| {
        obs[agent * 84 + channel * 42 + row * 7 + col]
    };
    assert_eq!(at(0, 0, 5, 3), 1);
    assert_eq!(at(0, 1, 5, 0), 1);
    assert_eq!(at(1, 0, 5, 0), 1);
    assert_eq!(at(1, 1, 5, 3), 1);
    assert_eq!(obs.iter().map(|&v| u32::from(v)).sum::<u32>(), 4);
}

#[test]
fn test_connect4_vertical_win_rewards_the_first_player() {
    let mut p = pool();
    play_all(&mut p, "010101");
    assert!(!terminated(&p));
    assert_eq!(p.outputs().rewards(), [0.0, 0.0]);
    play(&mut p, 0);
    assert_eq!(p.outputs().rewards(), [1.0, -1.0]);
    assert!(terminated(&p));
    assert!(p.outputs().agent_terminated().iter().all(|f| f.get()));
}

#[test]
fn test_connect4_horizontal_win_rewards_the_second_player() {
    let mut p = pool();
    // Slot 1 builds the bottom row in columns 1 to 4 while slot 0 plays
    // columns 6 and 5.
    play_all(&mut p, "61625364");
    assert_eq!(p.outputs().rewards(), [-1.0, 1.0]);
    assert!(terminated(&p));
}

#[test]
fn test_connect4_diagonal_win_ends_the_game() {
    let mut p = pool();
    // Slot 0 fills (0,0) (1,1) (2,2) (3,3).
    play_all(&mut p, "0112232336");
    assert!(!terminated(&p));
    play(&mut p, 3);
    assert_eq!(p.outputs().rewards(), [1.0, -1.0]);
    assert!(terminated(&p));
}

#[test]
fn test_connect4_full_board_without_four_is_a_draw() {
    let mut p = pool();
    play_all(&mut p, &DRAW[..41]);
    assert!(!terminated(&p));
    play_all(&mut p, &DRAW[41..]);
    assert!(terminated(&p));
    assert_eq!(p.outputs().rewards(), [0.0, 0.0]);
    assert_eq!(p.state(0).unwrap().moves, 42);
}

#[test]
fn test_connect4_full_column_is_masked_for_the_acting_player() {
    let mut p = pool();
    play_all(&mut p, "333333");
    let expected = [true, true, true, false, true, true, true];
    assert_eq!(mask(&p, 0), expected);
}

#[test]
fn test_connect4_illegal_column_is_replaced_by_the_noop_or_rejected() {
    let mut p = pool();
    play_all(&mut p, "333333");
    let before = *p.state(0).unwrap();
    let result = p.step(ActionInput {
        discrete: &[3, 6],
        continuous: &[],
    });
    if let Err(e) = result {
        // `debug-checks`: rejected atomically.
        assert!(matches!(e, EngineError::IllegalAction { agent: 0, .. }));
        assert_eq!(*p.state(0).unwrap(), before);
    } else {
        // Default: column 3 is full, so the no-op plays column 2, the
        // next in centre-first order, and the illegal action is counted.
        assert_eq!(p.state(0).unwrap().stones[0] & cell(2, 0), cell(2, 0));
        assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 1);
    }
}

#[test]
fn test_connect4_snapshot_and_restore_continue_the_game() {
    let mut p = pool();
    play_all(&mut p, "3324");
    let rec = p.snapshot(0).unwrap();
    play_all(&mut p, "4455");
    let after = p.state_hash(0).unwrap();
    p.restore(0, &rec).unwrap();
    assert_eq!(acting(&p), [true, false]);
    play_all(&mut p, "4455");
    assert_eq!(p.state_hash(0).unwrap(), after);
}

#[test]
fn test_connect4_auto_reset_starts_a_fresh_board() {
    let mut p = pool_with(AutoReset::SameStep);
    play_all(&mut p, "0101010");
    assert!(terminated(&p));
    assert_eq!(*p.state(0).unwrap(), Connect4State::default());
    assert_eq!(acting(&p), [true, false]);
}

fn import(p: &mut EnvPool<Connect4, Masked>, state: Connect4State) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let rec = DecodedRecord::from_parts(*good.header(), state).unwrap();
    p.import(&[0], &[rec])
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

#[test]
fn test_connect4_import_rejects_floating_and_overlapping_stones() {
    let mut p = pool();
    play(&mut p, 3);
    let base = *p.state(0).unwrap();
    let floating = Connect4State {
        stones: [cell(3, 1), 0],
        ..base
    };
    assert!(invalid(&import(&mut p, floating)));
    let overlap = Connect4State {
        stones: [cell(3, 0), cell(3, 0)],
        moves: 2,
        ..base
    };
    assert!(invalid(&import(&mut p, overlap)));
    assert!(import(&mut p, base).is_ok());
}

#[test]
fn test_connect4_import_rejects_counts_that_do_not_match_the_turn() {
    let mut p = pool();
    play(&mut p, 3);
    let base = *p.state(0).unwrap();
    let wrong_plies = Connect4State { moves: 2, ..base };
    assert!(invalid(&import(&mut p, wrong_plies)));
    let second_ahead = Connect4State {
        stones: [0, cell(3, 0)],
        ..base
    };
    assert!(invalid(&import(&mut p, second_ahead)));
}

#[test]
fn test_connect4_import_rejects_a_running_episode_on_a_won_board() {
    let mut p = pool();
    play(&mut p, 3);
    // Slot 0 holds four in column 0 and slot 1 three stones: slot 1 to
    // move, yet the game is over while the header says it is running.
    let won = Connect4State {
        stones: [
            cell(0, 0) | cell(0, 1) | cell(0, 2) | cell(0, 3),
            cell(1, 0) | cell(1, 1) | cell(1, 2),
        ],
        moves: 7,
        ..*p.state(0).unwrap()
    };
    let before = p.state_hash(0).unwrap();
    assert!(invalid(&import(&mut p, won)));
    assert!(!p.is_poisoned());
    assert_eq!(p.state_hash(0).unwrap(), before);
}

#[test]
fn test_connect4_validate_rejects_play_after_a_four() {
    use tessera_rl_core::Validate;
    let cfg = Connect4Config {};
    // Slot 0 made four in column 0, then slot 1 covered it and slot 0
    // played again: slot 1 to move, slot 0's four is not its last stone.
    let after = Connect4State {
        stones: [
            cell(0, 0) | cell(0, 1) | cell(0, 2) | cell(0, 3) | cell(4, 0),
            cell(0, 4) | cell(1, 0) | cell(1, 1) | cell(2, 0),
        ],
        moves: 9,
        ..Connect4State::default()
    };
    assert_eq!(
        Connect4::validate(&cfg, &after).unwrap_err().code,
        "play went on after a four"
    );
    // The same four made by the last stone (on top of column 0) is a
    // finished game, and valid.
    let finished = Connect4State {
        stones: [
            cell(0, 0) | cell(0, 1) | cell(0, 2) | cell(0, 3),
            cell(1, 0) | cell(1, 1) | cell(2, 0),
        ],
        moves: 7,
        ..Connect4State::default()
    };
    assert!(Connect4::validate(&cfg, &finished).is_ok());
    // Two fours sharing the last stone are valid too: slot 0 completes
    // column 3 and row 3 with the stone at (3, 3), on top of column 3.
    let line = |cells: &[(u32, u32)]| cells.iter().fold(0, |b, &(c, r)| b | cell(c, r));
    let double = Connect4State {
        stones: [
            line(&[
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (0, 3),
                (1, 3),
                (2, 3),
                (0, 0),
                (2, 1),
            ]),
            line(&[
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (6, 0),
            ]),
        ],
        moves: 17,
        ..Connect4State::default()
    };
    assert!(Connect4::validate(&cfg, &double).is_ok());
}

#[test]
fn test_connect4_terminal_record_imports_as_pending_reset() {
    let mut p = pool();
    play_all(&mut p, "0101010");
    assert!(p.header(0).unwrap().engine.pending_reset.get());
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut q = pool();
    assert!(q.import(&[0], &[rec]).is_ok());
    assert_eq!(q.state(0), p.state(0));
}

#[test]
fn test_connect4_import_rejects_a_dead_player() {
    let mut p = pool();
    play(&mut p, 3);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(invalid(&p.import(&[0], &[bad])));
}

fn baseline(state: &Connect4State, agent: u16) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(agent, 2).unwrap();
    Connect4::act(&Connect4Config {}, state, slot, &mut rng, &mut writer);
    discrete[0]
}

#[test]
fn test_connect4_baseline_wins_then_blocks_then_takes_the_centre() {
    let mut p = pool();
    assert_eq!(baseline(p.state(0).unwrap(), 0), 3);
    play_all(&mut p, "010101");
    // Slot 0 to move wins in column 0.
    assert_eq!(baseline(p.state(0).unwrap(), 0), 0);
    let mut q = pool();
    play_all(&mut q, "01010");
    // Slot 1 to move cannot win and blocks column 0.
    assert_eq!(baseline(q.state(0).unwrap(), 1), 0);
    let mut r = pool();
    play_all(&mut r, "3");
    assert_eq!(baseline(r.state(0).unwrap(), 1), 3);
}

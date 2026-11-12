// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect Four observation bytes against the reference loop: the
//! per-element writer `observe` used before #232, kept here as a test-only
//! reference. Every state reached by deterministic random legal play from
//! the empty board (both seats, empty, mid-game, won and full boards) and
//! every leaf length around the 84-byte board, including leaves shorter
//! than the board (the fallback path) and longer ones (bytes past the board
//! untouched), must give identical bytes.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::{AgentSlot, Environment};
use tessera_rl_games::connect4::board::{CELLS, HEIGHT, WIDTH, can_play, cell, has_four, landing};
use tessera_rl_games::{Connect4, Connect4Config, Connect4State};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed};
use tessera_rl_num::rng::{ObsRng, RngAuthority};

/// Bytes the observer's leaf starts with, so untouched bytes show.
const SENTINEL: u8 = 0xA5;

/// Leaf lengths: empty, short (fallback path), the board, longer.
const LENGTHS: [usize; 12] = [0, 1, 6, 7, 41, 42, 43, 83, 84, 85, 90, 168];

/// A full board without four in a row (the move string of `tests/connect4.rs`).
const DRAW: &str = "545062455041104565311226266362030334314210";

/// The observation writer before #232: element `k` of the u8 leaf, written
/// only when `k` is in range (the semantics of `LeafMut::set`).
fn reference(state: &Connect4State, seat: usize, leaf: &mut [u8]) {
    let me = seat.min(1);
    let planes = [state.stones[me], state.stones[1 - me]];
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    for (channel, stones) in planes.iter().enumerate() {
        for top_row in 0..HEIGHT {
            for col in 0..WIDTH {
                let row = HEIGHT - 1 - top_row;
                let k = channel * h * w + top_row as usize * w + col as usize;
                if let Some(byte) = leaf.get_mut(k) {
                    *byte = u8::from(stones & cell(col, row) != 0);
                }
            }
        }
    }
}

/// The bytes `Connect4::observe` writes into a leaf of `len` bytes.
fn observed(state: &Connect4State, seat: u16, len: usize) -> Vec<u8> {
    let mut board = vec![SENTINEL; len];
    {
        let mut leaves = [&mut board[..]];
        let mut out = ObsWriter::new(&mut leaves);
        let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
        let mut rng = ObsRng::new(&RngAuthority::engine_only(), &identity);
        let agent = AgentSlot::new(seat, 2).unwrap();
        Connect4::observe(&Connect4Config {}, state, agent, &mut rng, &mut out);
    }
    board
}

/// The bytes the reference writes into a leaf of `len` bytes.
fn expected(state: &Connect4State, seat: u16, len: usize) -> Vec<u8> {
    let mut board = vec![SENTINEL; len];
    reference(state, usize::from(seat), &mut board);
    board
}

/// Drops a stone for the side to move into `col`.
fn play(state: &mut Connect4State, col: u32) {
    let side = (state.moves % 2) as usize;
    state.stones[side] |= landing(state.occupied(), col);
    state.moves += 1;
}

/// Every state of `games` random legal games (splitmix64, no engine RNG),
/// each played until four in a row or a full board, plus the drawn board.
fn states(games: u64) -> Vec<Connect4State> {
    let mut out = vec![Connect4State::default()];
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for _ in 0..games {
        let mut s = Connect4State::default();
        while s.moves < CELLS && !has_four(s.stones[0]) && !has_four(s.stones[1]) {
            let legal: Vec<u32> = (0..WIDTH).filter(|&c| can_play(s.occupied(), c)).collect();
            let pick = usize::try_from(next() % legal.len() as u64).unwrap();
            play(&mut s, legal[pick]);
            out.push(s);
        }
    }
    let mut draw = Connect4State::default();
    for c in DRAW.bytes() {
        play(&mut draw, u32::from(c - b'0'));
    }
    assert_eq!(draw.moves, CELLS);
    out.push(draw);
    out
}

#[test]
fn test_connect4_observe_board_leaf_matches_reference_on_every_state() {
    let states = states(500);
    assert!(states.len() > 5_000, "{} states", states.len());
    for s in &states {
        for seat in [0, 1] {
            assert_eq!(
                observed(s, seat, 84),
                expected(s, seat, 84),
                "{s:?} seat {seat}"
            );
        }
    }
}

#[test]
fn test_connect4_observe_other_leaf_lengths_match_reference_on_every_state() {
    for s in &states(50) {
        for seat in [0, 1] {
            for len in LENGTHS {
                assert_eq!(
                    observed(s, seat, len),
                    expected(s, seat, len),
                    "{s:?} seat {seat} len {len}"
                );
            }
        }
    }
}

#[test]
fn test_connect4_observe_planes_differ_by_seat_on_a_mid_game_board() {
    let mut s = Connect4State::default();
    for c in [3, 3, 4] {
        play(&mut s, c);
    }
    let (zero, one) = (observed(&s, 0, 84), observed(&s, 1, 84));
    assert_ne!(zero, one);
    assert_eq!(zero[..42], one[42..]);
    assert_eq!(zero[42..], one[..42]);
}

/// Leaf lengths shorter than the board: the local-array fallback path.
/// The engine never builds such a leaf (`Connect4Obs` declares a fixed
/// `[2, 6, 7]` u8 box and `Connect4Config` has no field); only a hand-built
/// `ObsWriter`, as here, reaches this path.
#[test]
fn test_connect4_observe_short_leaf_writes_the_reference_prefix() {
    for s in &states(50) {
        for seat in [0, 1] {
            for len in [0, 1, 42, 83] {
                let full = expected(s, seat, 84);
                let got = observed(s, seat, len);
                assert_eq!(got, expected(s, seat, len), "{s:?} seat {seat} len {len}");
                assert_eq!(got[..], full[..len], "{s:?} seat {seat} len {len}");
            }
        }
    }
}

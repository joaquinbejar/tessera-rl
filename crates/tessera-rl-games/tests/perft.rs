// SPDX-License-Identifier: MIT OR Apache-2.0

//! Perft: the number of move sequences of each length from the empty
//! board, counting a finished game as a leaf only at its own depth. No
//! column fills before ply 6 and nobody wins before ply 7, so `perft(d) =
//! 7^d` for `d <= 6`; at depth 7 the 7 sequences that fill one column
//! have 6 last moves, `7^7 - 7`. Deeper counts are pinned and agree
//! between the environment and the reference board.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_connect4_ref::board::Position;
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::connect4::board::{WIDTH, can_play};
use tessera_rl_games::{Connect4, Connect4Config, Connect4State};
use tessera_rl_num::identity::Seed;

/// Perft counts from the empty board, depths 0 to 8 (the published
/// Connect Four perft sequence).
const PERFT: [u64; 9] = [1, 7, 49, 343, 2_401, 16_807, 117_649, 823_536, 5_673_234];

/// Perft over the environment's state transitions.
fn perft_state(s: &Connect4State, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if s.is_over() {
        return 0;
    }
    (0..WIDTH)
        .filter(|&c| can_play(s.occupied(), c))
        .map(|c| {
            let mut t = *s;
            t.play(c);
            perft_state(&t, depth - 1)
        })
        .sum()
}

/// Perft over the reference board.
fn perft_ref(p: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    (0..7)
        .filter(|&c| p.can_play(c))
        .map(|c| {
            if p.is_winning_col(c) {
                return u64::from(depth == 1);
            }
            let mut q = *p;
            q.play_col(c);
            if q.moves() == 42 {
                return u64::from(depth == 1);
            }
            perft_ref(&q, depth - 1)
        })
        .sum()
}

/// Perft through the engine: legal moves from the mask, branches by
/// snapshot and restore, a leaf where the engine reports the end.
fn perft_engine(p: &mut EnvPool<Connect4, Masked>, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if p.outputs().episode_terminated()[0].get() {
        return 0;
    }
    let me = p.state(0).unwrap().to_move();
    let legal: Vec<i32> = (0..7)
        .filter(|&c| p.outputs().masks()[me * 7 + c].get())
        .map(|c| i32::try_from(c).unwrap())
        .collect();
    let rec = p.snapshot(0).unwrap();
    let mut n = 0;
    for col in legal {
        let mut act = [0, 0];
        act[me] = col;
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
        n += perft_engine(p, depth - 1);
        p.restore(0, &rec).unwrap();
    }
    n
}

#[test]
fn test_perft_state_transitions_match_the_pinned_counts() {
    let empty = Connect4State::default();
    for (d, &expected) in PERFT.iter().enumerate() {
        assert_eq!(
            perft_state(&empty, u32::try_from(d).unwrap()),
            expected,
            "depth {d}"
        );
    }
}

#[test]
fn test_perft_reference_board_matches_the_pinned_counts() {
    for (d, &expected) in PERFT.iter().enumerate() {
        assert_eq!(
            perft_ref(&Position::new(), u32::try_from(d).unwrap()),
            expected,
            "depth {d}"
        );
    }
}

#[test]
fn test_perft_through_the_engine_matches_the_pinned_counts() {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(1))
    };
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for (d, &expected) in PERFT.iter().enumerate().take(6) {
        assert_eq!(
            perft_engine(&mut p, u32::try_from(d).unwrap()),
            expected,
            "depth {d}"
        );
    }
}

/// From a mid-game position the counts include games that end early; the
/// environment and the reference agree there too.
#[test]
fn test_perft_mid_game_counts_agree_with_the_reference() {
    for seq in ["4444", "121314", "3443554", "12121"] {
        let r = Position::from_moves(seq).unwrap();
        let mut s = Connect4State::default();
        for ch in seq.chars() {
            s.play(ch.to_digit(10).unwrap() - 1);
        }
        for d in 0..6 {
            assert_eq!(perft_state(&s, d), perft_ref(&r, d), "{seq} depth {d}");
        }
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Equivalence with the M0 reference game (`tools/connect4-ref`, the
//! board of ADR-0042): the environment, stepped through the engine, agrees
//! with the reference position on every ply of seeded random games and of
//! the 700 committed openings: the stones, the legal moves, the side to
//! move, wins, draws and rewards.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_connect4_ref::board::Position;
use tessera_rl_connect4_ref::rng::SplitMix64;
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;

fn pool() -> EnvPool<Connect4, Masked> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(2))
    };
    let mut p = EnvPool::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Checks that the pool's instance 0 is the reference position `r`.
fn agree(p: &EnvPool<Connect4, Masked>, r: &Position, ctx: &str) {
    let s = p.state(0).unwrap();
    let me = s.to_move();
    assert_eq!(s.moves, r.moves(), "{ctx}: plies");
    assert_eq!(s.stones[me], r.current(), "{ctx}: side to move");
    assert_eq!(s.stones[1 - me], r.opponent(), "{ctx}: opponent");
    let acting: Vec<bool> = p
        .outputs()
        .unwrap()
        .acting()
        .iter()
        .map(|f| f.get())
        .collect();
    let over = p.outputs().unwrap().episode_terminated()[0].get();
    if over {
        assert_eq!(acting, [false, false], "{ctx}: nobody acts after the end");
        return;
    }
    assert_eq!(acting, [me == 0, me == 1], "{ctx}: acting");
    let mask = &p.outputs().unwrap().masks()[me * 7..me * 7 + 7];
    for col in 0..7u32 {
        assert_eq!(
            mask[col as usize].get(),
            r.can_play(col),
            "{ctx}: mask of column {col}"
        );
    }
}

/// Plays `col` in both and checks the outcome; returns whether the game
/// is over.
fn play(p: &mut EnvPool<Connect4, Masked>, r: &mut Position, col: u32, ctx: &str) -> bool {
    let me = p.state(0).unwrap().to_move();
    let wins = r.is_winning_col(col);
    let mut act = [0, 0];
    act[me] = i32::try_from(col).unwrap();
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
    r.play_col(col);
    let draw = !wins && r.moves() == 42;
    let over = p.outputs().unwrap().episode_terminated()[0].get();
    assert_eq!(over, wins || draw, "{ctx}: end of game");
    let rewards = p.outputs().unwrap().rewards();
    let expected = if wins {
        let mut e = [-1.0, -1.0];
        e[me] = 1.0;
        e
    } else {
        [0.0, 0.0]
    };
    assert_eq!(rewards, expected, "{ctx}: rewards");
    agree(p, r, ctx);
    over
}

#[test]
fn test_equivalence_random_games_match_the_reference_on_every_ply() {
    let mut rng = SplitMix64::new(0x4d31_6200_0000_0000);
    let (mut wins, mut draws) = (0, 0);
    for game in 0..2_000 {
        let mut p = pool();
        let mut r = Position::new();
        agree(&p, &r, "start");
        loop {
            let legal: Vec<u32> = (0..7).filter(|&c| r.can_play(c)).collect();
            let col = legal[rng.below(legal.len())];
            let ctx = format!("game {game} ply {}", r.moves());
            let won = r.is_winning_col(col);
            if play(&mut p, &mut r, col, &ctx) {
                if won {
                    wins += 1;
                } else {
                    draws += 1;
                }
                break;
            }
        }
    }
    assert_eq!(wins + draws, 2_000);
    assert!(wins > 0);
}

#[test]
fn test_equivalence_committed_openings_replay_to_the_reference_positions() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../compat/connect4/");
    let mut count = 0;
    for file in ["openings-eval.txt", "openings-heldout.txt"] {
        let text = std::fs::read_to_string(format!("{root}{file}")).unwrap();
        for line in text.lines().filter(|l| !l.starts_with('#')) {
            let moves = line.split(' ').nth(2).unwrap();
            let expected = Position::from_moves(moves).unwrap();
            let mut p = pool();
            let mut r = Position::new();
            for ch in moves.chars() {
                let col = ch.to_digit(10).unwrap() - 1;
                assert!(!play(&mut p, &mut r, col, line), "{line}: game over");
            }
            assert_eq!(r, expected);
            count += 1;
        }
    }
    assert_eq!(count, 700);
}

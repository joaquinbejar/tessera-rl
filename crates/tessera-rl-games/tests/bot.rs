// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ported P2-G1 bot (ADR-0042) against the pinned reference
//! (`tools/connect4-ref/src/bot.rs`): the same column, root scores and
//! static evaluation on every position it is asked about, and the
//! ADR-0042 calibration reproduced number for number. The calibration
//! against the solver is long (about a minute with 14 threads in release)
//! and runs with `--ignored`.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use std::sync::Mutex;

use tessera_rl_connect4_ref::board::Position;
use tessera_rl_connect4_ref::bot::Bot as RefBot;
use tessera_rl_connect4_ref::calibrate::position_set;
use tessera_rl_connect4_ref::rng::SplitMix64;
use tessera_rl_connect4_ref::solver::Solver;
use tessera_rl_games::Connect4State;
use tessera_rl_games::connect4::bot::{Bot, DEPTH, TIE_BASE, W2, W3, WC, WIN};

/// The environment state of a reference position.
fn state(p: &Position) -> Connect4State {
    let mut s = Connect4State {
        moves: p.moves(),
        ..Connect4State::default()
    };
    let me = s.to_move();
    s.stones[me] = p.current();
    s.stones[1 - me] = p.opponent();
    s
}

/// The port and the reference agree at `p` for one tie seed and depth;
/// returns the column.
fn agree(p: &Position, depth: u32, tie_seed: u64) -> Option<u32> {
    let (ours, theirs) = (
        Bot::with_depth(depth, tie_seed),
        RefBot::with_depth(depth, tie_seed),
    );
    let s = state(p);
    assert_eq!(ours.eval(&s), theirs.eval(p), "eval at {p:?}");
    assert_eq!(
        ours.root_scores(&s),
        theirs.root_scores(p),
        "scores at {p:?}"
    );
    let col = ours.choose(&s);
    assert_eq!(col, theirs.choose(p), "choice at {p:?}");
    col
}

#[test]
fn test_bot_parameters_are_the_pinned_ones() {
    assert_eq!((DEPTH, W2, W3, WC, WIN), (4, 2, 5, 3, 1_000_000));
    assert_eq!(TIE_BASE, tessera_rl_connect4_ref::calibrate::TIE_BASE);
}

#[test]
fn test_bot_agrees_with_the_reference_on_the_calibration_positions() {
    for (i, p) in position_set(1_000).iter().enumerate() {
        let seed = TIE_BASE + u64::try_from(i).unwrap();
        agree(p, DEPTH, seed);
        agree(p, 1, seed ^ 1);
    }
}

#[test]
fn test_bot_agrees_with_the_reference_on_random_games() {
    let mut rng = SplitMix64::new(0x4d31_6202_0000_0000);
    for game in 0..300u64 {
        let mut p = Position::new();
        while p.moves() < 42 {
            agree(&p, DEPTH, TIE_BASE + game);
            let legal: Vec<u32> = (0..7).filter(|&c| p.can_play(c)).collect();
            let col = legal[rng.below(legal.len())];
            if p.is_winning_col(col) {
                break;
            }
            p.play_col(col);
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
struct Tally {
    win: u32,
    draw: u32,
    loss: u32,
}

enum Player<'a> {
    Bot(Bot),
    Random(SplitMix64),
    Solver(&'a mut Solver),
}

impl Player<'_> {
    /// The move at `p`; a bot move is checked against the reference.
    fn choose(&mut self, p: &Position) -> Option<u32> {
        match self {
            Player::Bot(b) => {
                let col = b.choose(&state(p));
                assert_eq!(col, ref_bot(b).choose(p), "choice at {p:?}");
                col
            }
            Player::Random(rng) => {
                let legal: Vec<u32> = (0..7).filter(|&c| p.can_play(c)).collect();
                legal.get(rng.below(legal.len())).copied()
            }
            Player::Solver(s) => s.best_col(p),
        }
    }
}

/// The reference bot with the same depth and tie seed.
fn ref_bot(b: &Bot) -> RefBot {
    RefBot::with_depth(b.depth(), b.tie_seed())
}

/// One game from `start`; `a` moves first when `a_first`. The result is
/// for `a`, scored like the reference `calibrate::play`.
fn play<'p>(start: Position, a: &mut Player<'p>, b: &mut Player<'p>, a_first: bool) -> Tally {
    let mut p = start;
    let mut a_to_move = a_first;
    while p.moves() < 42 {
        let mover = if a_to_move { &mut *a } else { &mut *b };
        let Some(col) = mover.choose(&p) else {
            break;
        };
        if p.is_winning_col(col) {
            return if a_to_move {
                Tally {
                    win: 1,
                    ..Tally::default()
                }
            } else {
                Tally {
                    loss: 1,
                    ..Tally::default()
                }
            };
        }
        p.play_col(col);
        a_to_move = !a_to_move;
    }
    Tally {
        draw: 1,
        ..Tally::default()
    }
}

fn add(t: &mut Tally, o: Tally) {
    t.win += o.win;
    t.draw += o.draw;
    t.loss += o.loss;
}

#[test]
fn test_bot_calibration_against_random_and_depth_one_is_reproduced() {
    let (mut random, mut depth1) = (Tally::default(), Tally::default());
    for i in 0..500u64 {
        for first in [true, false] {
            let mut bot = Player::Bot(Bot::new(TIE_BASE + i));
            let mut rnd = Player::Random(SplitMix64::new(0x5eed_0000 + i));
            add(
                &mut random,
                play(Position::new(), &mut bot, &mut rnd, first),
            );
            let mut weak = Player::Bot(Bot::with_depth(1, 0x0de9_0001 + i));
            add(
                &mut depth1,
                play(Position::new(), &mut bot, &mut weak, first),
            );
        }
    }
    assert_eq!(
        random,
        Tally {
            win: 1000,
            draw: 0,
            loss: 0
        }
    );
    assert_eq!(
        depth1,
        Tally {
            win: 984,
            draw: 0,
            loss: 16
        }
    );
}

fn openings() -> Vec<Position> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../compat/connect4/openings-eval.txt"
    );
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| Position::from_moves(l.split(' ').nth(2).unwrap()).unwrap())
        .collect()
}

/// Agreement with perfect play on the fixed position set, and the games
/// against the perfect solver from the 500 evaluation openings (ADR-0042).
#[test]
#[ignore = "long: about a minute with 14 threads in release"]
fn test_bot_calibration_against_the_solver_is_reproduced() {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let set = position_set(1_000);
    let next = Mutex::new(0usize);
    let agreement = Mutex::new((0u32, 0u32, 0u32));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut solver = Solver::new();
                let bot = Bot::new(TIE_BASE);
                loop {
                    let i = {
                        let mut g = next.lock().unwrap();
                        *g += 1;
                        *g - 1
                    };
                    let Some(p) = set.get(i) else { return };
                    let scores = solver.column_scores(p);
                    let best = scores.iter().flatten().copied().max().unwrap();
                    let col = agree(p, DEPTH, TIE_BASE).unwrap();
                    assert_eq!(Some(col), bot.choose(&state(p)));
                    let mine = scores[col as usize].unwrap();
                    let mut t = agreement.lock().unwrap();
                    t.0 += 1;
                    t.1 += u32::from(mine == best);
                    t.2 += u32::from(mine.signum() == best.signum());
                }
            });
        }
    });
    assert_eq!(agreement.into_inner().unwrap(), (1000, 844, 944));

    let starts = openings();
    let next = Mutex::new(0usize);
    let result = Mutex::new((Tally::default(), Tally::default()));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut solver = Solver::new();
                loop {
                    let i = {
                        let mut g = next.lock().unwrap();
                        *g += 1;
                        *g - 1
                    };
                    let Some(&start) = starts.get(i) else { return };
                    let seed = TIE_BASE + u64::try_from(i).unwrap();
                    for first in [true, false] {
                        let mut bot = Player::Bot(Bot::new(seed));
                        let mut sv = Player::Solver(&mut solver);
                        let o = play(start, &mut bot, &mut sv, first);
                        let mut r = result.lock().unwrap();
                        add(if first { &mut r.0 } else { &mut r.1 }, o);
                    }
                }
            });
        }
    });
    let (first, second) = result.into_inner().unwrap();
    assert_eq!(
        first,
        Tally {
            win: 82,
            draw: 30,
            loss: 388
        }
    );
    assert_eq!(
        second,
        Tally {
            win: 44,
            draw: 29,
            loss: 427
        }
    );
}

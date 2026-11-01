// SPDX-License-Identifier: MIT OR Apache-2.0

//! Calibration of the pinned bot (ADR-0042): against random play, against
//! a depth-1 bot, agreement with perfect play on a fixed position set, and
//! games against the perfect solver from the evaluation openings.

use crate::board::{CELLS, Position, WIDTH};
use crate::bot::Bot;
use crate::rng::SplitMix64;
use crate::solver::Solver;
use std::sync::Mutex;
use std::thread;

/// Seed base of the bot's tie-break stream (game `i` uses `TIE_BASE + i`),
/// disjoint from the opening seeds.
pub const TIE_BASE: u64 = 0x7e55_e2a0_0000_0000;

/// A player.
pub enum Player<'a> {
    /// The pinned bot (or a baseline depth).
    Bot(&'a Bot),
    /// Uniform random legal moves.
    Random(SplitMix64),
    /// The perfect solver.
    Solver(Box<Solver>),
}

impl Player<'_> {
    fn choose(&mut self, p: &Position) -> Option<u32> {
        match self {
            Player::Bot(b) => b.choose(p),
            Player::Random(rng) => {
                let legal: Vec<u32> = (0..WIDTH).filter(|&c| p.can_play(c)).collect();
                legal.get(rng.below(legal.len())).copied()
            }
            Player::Solver(s) => s.best_col(p),
        }
    }
}

/// Result of a game for the first-listed player.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It won.
    Win,
    /// Draw.
    Draw,
    /// It lost.
    Loss,
}

/// Plays from `start`; `a` moves first from `start` when `a_first`.
pub fn play<'p>(start: Position, a: &mut Player<'p>, b: &mut Player<'p>, a_first: bool) -> Outcome {
    let mut p = start;
    let mut a_to_move = a_first;
    while p.moves() < CELLS {
        let mover: &mut Player<'p> = if a_to_move { &mut *a } else { &mut *b };
        let Some(col) = mover.choose(&p) else {
            return Outcome::Draw;
        };
        if p.is_winning_col(col) {
            return if a_to_move {
                Outcome::Win
            } else {
                Outcome::Loss
            };
        }
        p.play_col(col);
        a_to_move = !a_to_move;
    }
    Outcome::Draw
}

/// Win, draw and loss counts.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    /// Wins.
    pub win: u32,
    /// Draws.
    pub draw: u32,
    /// Losses.
    pub loss: u32,
}

impl Tally {
    fn add(&mut self, o: Outcome) {
        match o {
            Outcome::Win => self.win += 1,
            Outcome::Draw => self.draw += 1,
            Outcome::Loss => self.loss += 1,
        }
    }

    /// As JSON.
    #[must_use]
    pub fn json(&self) -> String {
        format!(
            "{{\"win\": {}, \"draw\": {}, \"loss\": {}}}",
            self.win, self.draw, self.loss
        )
    }
}

/// The bot against random play from the empty board, `n` games per colour.
#[must_use]
pub fn vs_random(n: u32) -> Tally {
    let mut t = Tally::default();
    for i in 0..n {
        for first in [true, false] {
            let bot = Bot::new(TIE_BASE + u64::from(i));
            let mut a = Player::Bot(&bot);
            let mut b = Player::Random(SplitMix64::new(0x5eed_0000 + u64::from(i)));
            t.add(play(Position::new(), &mut a, &mut b, first));
        }
    }
    t
}

/// The bot against a depth-1 bot from the empty board, `n` games per colour
/// (tie seeds make the games differ).
#[must_use]
pub fn vs_depth1(n: u32) -> Tally {
    let mut t = Tally::default();
    for i in 0..n {
        for first in [true, false] {
            let bot = Bot::new(TIE_BASE + u64::from(i));
            let weak = Bot::with_depth(1, 0x0de9_0001 + u64::from(i));
            t.add(play(
                Position::new(),
                &mut Player::Bot(&bot),
                &mut Player::Bot(&weak),
                first,
            ));
        }
    }
    t
}

/// The fixed position set: random play from seed `0xa9_0000 + i` stopped at
/// ply `12 + (i % 17)`, skipping finished games and positions where the side
/// to move wins at once.
#[must_use]
pub fn position_set(n: u32) -> Vec<Position> {
    let mut out = Vec::new();
    let mut i = 0u64;
    while out.len() < n as usize {
        let mut rng = SplitMix64::new(0x00a9_0000 + i);
        let target = 12 + u32::try_from(i % 17).unwrap_or(0);
        let mut p = Position::new();
        let mut ok = true;
        while p.moves() < target {
            let legal: Vec<u32> = (0..WIDTH).filter(|&c| p.can_play(c)).collect();
            let Some(&col) = legal.get(rng.below(legal.len())) else {
                ok = false;
                break;
            };
            if p.is_winning_col(col) {
                ok = false;
                break;
            }
            p.play_col(col);
        }
        if ok && !p.can_win_next() {
            out.push(p);
        }
        i += 1;
    }
    out
}

/// Agreement of the bot with perfect play on `set`: the fraction of
/// positions where its move is optimal, and where it keeps the
/// game-theoretic outcome (win, draw or loss).
#[must_use]
pub fn agreement(set: &[Position], threads: usize) -> (u32, u32, u32) {
    let next = Mutex::new(0usize);
    let totals = Mutex::new((0u32, 0u32, 0u32));
    thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut solver = Solver::new();
                let bot = Bot::new(TIE_BASE);
                loop {
                    let i = {
                        let Ok(mut g) = next.lock() else { return };
                        let i = *g;
                        *g += 1;
                        i
                    };
                    let Some(p) = set.get(i) else { return };
                    let scores = solver.column_scores(p);
                    let best = scores.iter().flatten().copied().max().unwrap_or(0);
                    let Some(col) = bot.choose(p) else { continue };
                    let mine = scores
                        .get(col as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(i32::MIN);
                    let optimal = u32::from(mine == best);
                    let same_class = u32::from(mine.signum() == best.signum());
                    if let Ok(mut t) = totals.lock() {
                        t.0 += 1;
                        t.1 += optimal;
                        t.2 += same_class;
                    }
                }
            });
        }
    });
    totals.into_inner().unwrap_or_default()
}

/// The bot against the perfect solver from each opening, both colours.
#[must_use]
pub fn vs_solver(openings: &[Position], threads: usize) -> (Tally, Tally) {
    let next = Mutex::new(0usize);
    let result = Mutex::new((Tally::default(), Tally::default()));
    thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                let mut solver = Some(Box::new(Solver::new()));
                loop {
                    let i = {
                        let Ok(mut g) = next.lock() else { return };
                        let i = *g;
                        *g += 1;
                        i
                    };
                    let Some(&start) = openings.get(i) else {
                        return;
                    };
                    let bot = Bot::new(TIE_BASE + i as u64);
                    for first in [true, false] {
                        let Some(sv) = solver.take() else { return };
                        let mut b = Player::Solver(sv);
                        let o = play(start, &mut Player::Bot(&bot), &mut b, first);
                        if let Player::Solver(sv) = b {
                            solver = Some(sv);
                        }
                        if let Ok(mut r) = result.lock() {
                            if first { r.0.add(o) } else { r.1.add(o) }
                        }
                    }
                }
            });
        }
    });
    result.into_inner().unwrap_or_default()
}

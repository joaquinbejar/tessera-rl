// SPDX-License-Identifier: MIT OR Apache-2.0

//! The P2-G1 best-response measurement (`docs/reviews/best-response-design.md`,
//! revision 4): single-agent search over the candidate's moves against a fixed
//! reply function. Held-out openings only: the file is compiled in, and the tool
//! takes no other.
//!
//! A candidate node is charged one on every visit, a memo hit included. A cap
//! ends the slot's search at once, so an unresolved result reaches the root and
//! no node above it is ever stored as a non-win.

use crate::board::{CELLS, CENTRE_ORDER, Position};
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread;

/// The held-out openings, SHA-256 `3276d2d0...cc49` (asserted by `scripts/best_response_report.py`; this crate has no SHA-256).
pub const HELDOUT: &str = include_str!("../../../compat/connect4/openings-heldout.txt");
/// Pass 1 cap, candidate nodes per slot.
pub const PASS1_CAP: u64 = 1_000_000;
/// Pass 2 cap, candidate nodes per slot.
pub const PASS2_CAP: u64 = 20_000_000;
/// Budget over both passes, candidate nodes.
pub const TOTAL_BUDGET: u64 = 2_000_000_000;
/// Stored non-wins per slot; reaching it is a cap.
pub const TABLE_LIMIT: usize = 8_000_000;

/// The held-out openings in file order, as `(index, moves)`.
#[must_use]
pub fn openings() -> Vec<(u64, String)> {
    HELDOUT
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let mut f = l.split_whitespace();
            let j = f.next()?.parse().ok()?;
            let moves = f.nth(1)?.to_owned();
            Some((j, moves))
        })
        .collect()
}

/// The class of one slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Class {
    /// A line: every move from the opening, 0-based columns, both sides.
    Win(Vec<u32>),
    /// The whole tree was searched with no win.
    Exhausted,
    /// A node or table cap ended the search.
    Unresolved,
    /// The abort flag ended the search: the run is void.
    Aborted,
}

/// A search result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// Its class.
    pub class: Class,
    /// Candidate nodes charged.
    pub nodes: u64,
    /// Stored non-wins at the end.
    pub table: usize,
}

enum Node {
    Win,
    NoWin,
    Capped,
    Aborted,
}

struct Search<'a, R> {
    reply: &'a R,
    cap: u64,
    table_limit: usize,
    memo: Option<HashSet<u64>>,
    nodes: u64,
    line: Vec<u32>,
    abort: &'a AtomicBool,
}

impl<R: Fn(&Position) -> Option<u32>> Search<'_, R> {
    fn node(&mut self, p: &Position) -> Node {
        if self.nodes >= self.cap {
            return Node::Capped;
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(4096) && self.abort.load(Ordering::Relaxed) {
            return Node::Aborted;
        }
        if self.memo.as_ref().is_some_and(|m| m.contains(&p.key())) {
            return Node::NoWin;
        }
        for &col in &CENTRE_ORDER {
            if !p.can_play(col) {
                continue;
            }
            if p.is_winning_col(col) {
                self.line.push(col);
                return Node::Win;
            }
            let mut c = *p;
            c.play_col(col);
            if c.moves() >= CELLS {
                continue;
            }
            let Some(r) = (self.reply)(&c) else { continue };
            if !c.can_play(r) || c.is_winning_col(r) {
                continue;
            }
            c.play_col(r);
            if c.moves() >= CELLS {
                continue;
            }
            self.line.push(col);
            self.line.push(r);
            match self.node(&c) {
                Node::NoWin => {
                    self.line.truncate(self.line.len() - 2);
                }
                other => return other,
            }
        }
        if let Some(m) = self.memo.as_mut() {
            if m.len() >= self.table_limit {
                return Node::Capped;
            }
            m.insert(p.key());
        }
        Node::NoWin
    }
}

/// Searches one slot: the candidate moves first from `start`, or the reply
/// function does. `memo` off stores nothing, for the recheck.
pub fn search_slot<R: Fn(&Position) -> Option<u32>>(
    start: Position,
    candidate_first: bool,
    reply: &R,
    (cap, table_limit): (u64, usize),
    memo: bool,
    abort: &AtomicBool,
) -> Outcome {
    let mut p = start;
    let mut line = Vec::new();
    if !candidate_first {
        let r = reply(&p);
        match r {
            Some(r) if p.can_play(r) && !p.is_winning_col(r) => {
                p.play_col(r);
                line.push(r);
            }
            _ => {
                return Outcome {
                    class: Class::Exhausted,
                    nodes: 0,
                    table: 0,
                };
            }
        }
    }
    let mut s = Search {
        reply,
        cap,
        table_limit,
        memo: memo.then(HashSet::new),
        nodes: 0,
        line,
        abort,
    };
    let class = match s.node(&p) {
        Node::Win => Class::Win(core::mem::take(&mut s.line)),
        Node::NoWin => Class::Exhausted,
        Node::Capped => Class::Unresolved,
        Node::Aborted => Class::Aborted,
    };
    Outcome {
        class,
        nodes: s.nodes,
        table: s.memo.map_or(0, |m| m.len()),
    }
}

struct Sched {
    next: usize,
    admit: usize,
    stopped: bool,
    spent: u64,
    running: BTreeMap<usize, u64>,
    results: Vec<Option<Outcome>>,
}

/// Runs `search(i, cap)` for `0..n` on `threads` threads under the budget. A slot
/// starts only if the charges of every lower slot plus its own cap fit: the
/// decision a serial run would take, so results do not depend on scheduling and
/// executed work never exceeds the budget. A slot that does not fit is
/// `None`: not started. Slots are admitted strictly in index order, so a later
/// slot cannot pass a lower one that waits. Returns the results and the budget left.
pub fn run_budgeted<F: Fn(usize, u64) -> Outcome + Sync>(
    n: usize,
    cap: u64,
    budget: u64,
    threads: usize,
    search: F,
) -> (Vec<Option<Outcome>>, u64) {
    let state = Mutex::new(Sched {
        next: 0,
        admit: 0,
        stopped: false,
        spent: 0,
        running: BTreeMap::new(),
        results: vec![None; n],
    });
    let cv = Condvar::new();
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let Ok(mut g) = state.lock() else { return };
                    let i = g.next;
                    if i >= n {
                        return;
                    }
                    g.next += 1;
                    loop {
                        if g.stopped {
                            return;
                        }
                        if g.admit == i {
                            let reserved: u64 = g.spent + g.running.values().sum::<u64>();
                            if reserved + cap <= budget {
                                break;
                            }
                            if g.running.is_empty() {
                                // Over budget; so is every later slot, by the same sum.
                                g.stopped = true;
                                cv.notify_all();
                                return;
                            }
                        }
                        let Ok(w) = cv.wait(g) else { return };
                        g = w;
                    }
                    g.admit += 1;
                    cv.notify_all();
                    g.running.insert(i, cap);
                    drop(g);
                    let o = search(i, cap);
                    let Ok(mut g) = state.lock() else { return };
                    g.running.remove(&i);
                    g.spent += o.nodes;
                    if let Some(slot) = g.results.get_mut(i) {
                        *slot = Some(o);
                    }
                    cv.notify_all();
                }
            });
        }
    });
    let (results, spent) = state
        .into_inner()
        .map_or((Vec::new(), budget), |g| (g.results, g.spent));
    (results, budget.saturating_sub(spent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::Bot;

    fn pos(seq: &str) -> Position {
        Position::from_moves(seq).unwrap_or_default()
    }

    const CAPS: (u64, usize) = (1_000_000, 1_000_000);

    fn run<R: Fn(&Position) -> Option<u32>>(
        p: Position,
        first: bool,
        r: &R,
        memo: bool,
    ) -> Outcome {
        search_slot(p, first, r, CAPS, memo, &AtomicBool::new(false))
    }

    #[test]
    fn test_best_response_finds_a_win_in_one() {
        let o = run(pos("121212"), true, &|_: &Position| Some(6), true);
        // Depth first, centre first: any line, not the shortest; it must end in a four.
        let Class::Win(line) = o.class else {
            panic!("expected a win");
        };
        let mut p = pos("121212");
        let (last, head) = line.split_last().expect("a move");
        for &c in head {
            p.play_col(c);
        }
        assert!(p.is_winning_col(*last));
    }

    #[test]
    fn test_best_response_bot_win_first_is_an_exhausted_non_win() {
        // The reply function moves first from a position it wins at once.
        let o = run(pos("121212"), false, &|_: &Position| Some(0), true);
        assert_eq!(o.class, Class::Exhausted);
    }

    /// The bot holds an open three on the bottom row, with two ends: a forced
    /// loss for the candidate, whatever it plays. Checkable by hand.
    #[test]
    fn test_best_response_forced_loss_is_an_exhausted_non_win() {
        let bot = Bot::new(0);
        let r = |q: &Position| bot.choose(q);
        let o = run(pos("33445"), true, &r, true);
        assert_eq!(o.class, Class::Exhausted);
        assert_eq!(o.nodes, 1);
    }

    /// A fourless filling to `plies`, depth first.
    fn fill(p: Position, plies: u32) -> Option<Position> {
        if p.moves() == plies {
            return Some(p);
        }
        (0..7)
            .filter(|&c| p.can_play(c) && !p.is_winning_col(c))
            .find_map(|c| {
                let mut q = p;
                q.play_col(c);
                fill(q, plies)
            })
    }

    /// A small full-board case: no line wins, against a reply that takes the
    /// first legal column, with and without memo.
    #[test]
    fn test_best_response_full_board_is_an_exhausted_non_win() {
        let first_legal = |q: &Position| (0..7).find(|&c| q.can_play(c));
        let p = fill(Position::new(), 41).expect("a fourless filling");
        let a = run(p, true, &first_legal, true);
        assert_eq!(a, run(p, true, &first_legal, true));
        assert_eq!(a.class, Class::Exhausted, "the last cell makes no four");
        let b = run(p, true, &first_legal, false);
        assert_eq!(a.class, b.class, "memo agrees with no memo");
    }

    #[test]
    fn test_best_response_cap_is_unresolved_and_never_stored() {
        let r = |q: &Position| (0..7).find(|&c| q.can_play(c));
        let o = search_slot(pos("1"), false, &r, (3, 100), true, &AtomicBool::new(false));
        assert_eq!(o.class, Class::Unresolved);
        assert_eq!(o.nodes, 3);
        assert_eq!(o.table, 0, "no node above a cap is a proven non-win");
    }

    #[test]
    fn test_best_response_table_limit_is_a_cap() {
        // The forced loss above: its one node finishes with no win, and may not store.
        let bot = Bot::new(0);
        let r = |q: &Position| bot.choose(q);
        let o = search_slot(
            pos("33445"),
            true,
            &r,
            (1_000, 0),
            true,
            &AtomicBool::new(false),
        );
        assert_eq!(o.class, Class::Unresolved);
        assert_eq!(o.table, 0);
    }

    #[test]
    fn test_best_response_abort_is_not_a_result() {
        // Held-out opening 135, candidate second: an exhausted slot of over 4096
        // nodes, so the flag is seen at its first check, before any result.
        let (j, moves) = openings()
            .into_iter()
            .find(|o| o.0 == 135)
            .expect("opening");
        let bot = Bot::new(crate::calibrate::TIE_BASE + j);
        let r = |q: &Position| bot.choose(q);
        let stop = AtomicBool::new(true);
        let o = search_slot(pos(&moves), false, &r, CAPS, false, &stop);
        assert_eq!(o.class, Class::Aborted);
        assert_eq!(o.nodes, 4096);
    }

    #[test]
    fn test_best_response_line_replays_to_a_win() {
        let bot = Bot::new(7);
        let r = |q: &Position| bot.choose(q);
        let start = pos("44");
        for first in [true, false] {
            let o = run(start, first, &r, true);
            if let Class::Win(line) = o.class {
                let mut p = start;
                let last = line.len() - 1;
                let bot_offset = usize::from(first);
                for (i, &c) in line.iter().enumerate() {
                    if i % 2 == bot_offset {
                        assert_eq!(bot.choose(&p), Some(c), "ply {i}");
                    }
                    if i == last {
                        assert!(p.is_winning_col(c));
                    }
                    p.play_col(c);
                }
            }
        }
    }

    #[test]
    fn test_best_response_is_deterministic_with_and_without_memo() {
        let bot = Bot::new(11);
        let r = |q: &Position| bot.choose(q);
        let start = pos("11662757");
        let a = run(start, false, &r, true);
        assert_eq!(a, run(start, false, &r, true));
        let c = run(start, false, &r, false);
        assert_eq!(
            matches!(a.class, Class::Win(_)),
            matches!(c.class, Class::Win(_))
        );
    }

    #[test]
    fn test_best_response_choose_is_the_same_across_threads() {
        let bot = Bot::new(3);
        let p = pos("11662757");
        let here = bot.choose(&p);
        thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| assert_eq!(bot.choose(&p), here));
            }
        });
    }

    #[test]
    fn test_best_response_schedule_matches_serial_under_any_thread_count() {
        let charge = |i: usize, _cap: u64| Outcome {
            class: Class::Exhausted,
            nodes: (i as u64 * 37) % 90 + 10,
            table: 0,
        };
        let (serial, left) = run_budgeted(50, 100, 1000, 1, charge);
        for threads in [2, 5, 16] {
            let (par, l) = run_budgeted(50, 100, 1000, threads, charge);
            assert_eq!(par, serial);
            assert_eq!(l, left);
        }
        assert!(serial.iter().any(Option::is_none), "the budget ran out");
        let spent: u64 = serial.iter().flatten().map(|o| o.nodes).sum();
        assert!(spent <= 1000);
    }

    /// Later slots finish first and charge near the cap, so slots wait near the
    /// budget in a different order on each run.
    #[test]
    fn test_best_response_schedule_admits_in_index_order_near_the_budget() {
        let charge = |i: usize, _cap: u64| {
            thread::sleep(std::time::Duration::from_millis(((40 - i) % 4) as u64));
            Outcome {
                class: Class::Exhausted,
                nodes: (i as u64 * 53) % 60 + 40,
                table: 0,
            }
        };
        let (serial, left) = run_budgeted(40, 100, 1500, 1, charge);
        assert!(serial.iter().any(Option::is_none), "the budget ran out");
        for _ in 0..10 {
            let (par, l) = run_budgeted(40, 100, 1500, 8, charge);
            assert_eq!(par, serial);
            assert_eq!(l, left);
        }
    }

    #[test]
    fn test_best_response_heldout_has_200_openings() {
        let o = openings();
        assert_eq!(o.len(), 200);
        assert_eq!(o.first().map(|x| x.0), Some(0));
    }
}

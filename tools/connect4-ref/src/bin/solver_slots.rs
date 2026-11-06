// SPDX-License-Identifier: MIT OR Apache-2.0

//! The perfect solver's result on every slot of an opening file, for the
//! ADR-0048 comparison: `solver_slots OPENINGS OUT THREADS`.
//!
//! One row per slot, in file order: the opening's index, the candidate's seat
//! (`candidate_first` or `candidate_second`), and `win`, `draw` or `loss` from
//! the solver's side, with the solver in the candidate's seat. The seat is
//! named because `calibrate::play` takes whether the bot moves first.

use std::fmt::Write as _;
use std::process::ExitCode;
use std::sync::Mutex;
use std::thread;
use tessera_rl_connect4_ref::board::Position;
use tessera_rl_connect4_ref::bot::Bot;
use tessera_rl_connect4_ref::calibrate::{Outcome, Player, TIE_BASE, play};
use tessera_rl_connect4_ref::solver::Solver;

const SEATS: [(&str, bool); 2] = [("candidate_first", true), ("candidate_second", false)];

fn read_openings(path: &str) -> Result<Vec<Position>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    lines
        .enumerate()
        .map(|(i, l)| {
            let mut f = l.split_whitespace();
            if f.next().and_then(|j| j.parse::<usize>().ok()) != Some(i) {
                return Err(format!("opening {i} is out of order: {l}"));
            }
            f.nth(1)
                .and_then(Position::from_moves)
                .ok_or_else(|| format!("bad opening {l}"))
        })
        .collect()
}

/// One slot's result for the solver, in the candidate's seat.
fn slot(
    start: Position,
    j: usize,
    candidate_first: bool,
    solver: Box<Solver>,
) -> (&'static str, Box<Solver>) {
    let bot = Bot::new(TIE_BASE + j as u64);
    let mut candidate = Player::Solver(solver);
    // `play` takes whether its first player, the bot, moves first, and reports for the bot.
    let outcome = play(
        start,
        &mut Player::Bot(&bot),
        &mut candidate,
        !candidate_first,
    );
    let name = match outcome {
        Outcome::Loss => "win",
        Outcome::Draw => "draw",
        Outcome::Win => "loss",
    };
    let Player::Solver(back) = candidate else {
        return (name, Box::new(Solver::new()));
    };
    (name, back)
}

fn run(input: &str, out: &str, threads: usize) -> Result<(), String> {
    let openings = read_openings(input)?;
    let next = Mutex::new(0usize);
    let rows = Mutex::new(vec![["", ""]; openings.len()]);
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut solver = Box::new(Solver::new());
                loop {
                    let j = {
                        let Ok(mut g) = next.lock() else { return };
                        *g += 1;
                        *g - 1
                    };
                    let Some(&start) = openings.get(j) else {
                        return;
                    };
                    let mut row = ["", ""];
                    for (cell, (_, first)) in row.iter_mut().zip(SEATS) {
                        let (name, back) = slot(start, j, first, solver);
                        *cell = name;
                        solver = back;
                    }
                    if let Ok(mut r) = rows.lock()
                        && let Some(c) = r.get_mut(j)
                    {
                        *c = row;
                    }
                }
            });
        }
    });
    let rows = rows.into_inner().map_err(|e| e.to_string())?;
    let mut text = String::new();
    for (j, row) in rows.iter().enumerate() {
        for ((seat, _), name) in SEATS.iter().zip(row) {
            if name.is_empty() {
                return Err(format!("opening {j} did not finish"));
            }
            let _ = writeln!(text, "{j}\t{seat}\t{name}");
        }
    }
    std::fs::write(out, text).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    let threads = a.get(3).and_then(|t| t.parse::<usize>().ok()).unwrap_or(0);
    let r = match (a.get(1), a.get(2)) {
        (Some(input), Some(out)) if threads > 0 => run(input, out, threads),
        _ => Err("usage: solver_slots OPENINGS OUT THREADS".to_owned()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = std::io::Write::write_all(&mut std::io::stderr(), format!("{e}\n").as_bytes());
            ExitCode::FAILURE
        }
    }
}

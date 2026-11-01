// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 2 M0 reference tools for Connect Four (ADR-0042).
//!
//! - `openings EVAL_OUT HELDOUT_OUT`: writes the 500 evaluation openings
//!   (seeds from `EVAL_BASE`) and the 200 held-out openings for checkpoint
//!   selection (seeds from `HELDOUT_BASE`, disjoint positions).
//! - `calibrate OPENINGS THREADS GAMES`: prints the bot's calibration as
//!   JSON (random play, depth-1, solver agreement, games against the
//!   solver from the first `GAMES` openings).
//! - `solve MOVES`: prints the exact score of a position.

#![expect(
    clippy::cast_possible_wrap,
    reason = "board quantities (plies, cells, scores) are at most 42 and fit in i32"
)]

mod board;
mod bot;
mod calibrate;
mod openings;
mod rng;
mod solver;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::{self, Write};
use std::process::ExitCode;

use board::Position;
use openings::Opening;
use solver::Solver;

/// Seed base of the 500 evaluation openings.
const EVAL_BASE: u64 = 0x5032_4731_0000_0000;
/// Seed base of the 200 held-out openings.
const HELDOUT_BASE: u64 = 0x5032_4731_8000_0000;
const EVAL_COUNT: usize = 500;
const HELDOUT_COUNT: usize = 200;

fn write_openings(path: &str, kind: &str, base: u64, list: &[Opening]) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("# SPDX-License-Identifier: MIT OR Apache-2.0\n");
    let _ = writeln!(
        out,
        "# P2-G1 {kind} openings (ADR-0042): K={} random plies, forced-win filter L={}, seed base {base:#018x}.",
        openings::K,
        openings::L
    );
    out.push_str("# index seed moves(1-based columns)\n");
    for (i, o) in list.iter().enumerate() {
        let _ = writeln!(out, "{i} {:#018x} {}", o.seed, o.moves);
    }
    std::fs::write(path, out)
}

fn read_openings(path: &str) -> io::Result<Vec<Position>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let moves = line.split_whitespace().nth(2).unwrap_or("");
        let p = Position::from_moves(moves).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad opening {line}"))
        })?;
        out.push(p);
    }
    Ok(out)
}

fn run(args: &[String]) -> io::Result<String> {
    let bad = || io::Error::new(io::ErrorKind::InvalidInput, "usage: see the module docs");
    match args.first().map(String::as_str) {
        Some("openings") => {
            let (eval_out, held_out) = (args.get(1).ok_or_else(bad)?, args.get(2).ok_or_else(bad)?);
            let mut solver = Solver::new();
            let eval = openings::generate(EVAL_BASE, EVAL_COUNT, &BTreeSet::new(), &mut solver);
            let taken: BTreeSet<u64> = eval
                .iter()
                .filter_map(|o| Position::from_moves(&o.moves))
                .map(|p| p.canonical_key())
                .collect();
            let held = openings::generate(HELDOUT_BASE, HELDOUT_COUNT, &taken, &mut solver);
            write_openings(eval_out, "evaluation", EVAL_BASE, &eval)?;
            write_openings(held_out, "held-out", HELDOUT_BASE, &held)?;
            Ok(format!(
                "{{\"evaluation\": {}, \"held_out\": {}}}",
                eval.len(),
                held.len()
            ))
        }
        Some("calibrate") => {
            let path = args.get(1).ok_or_else(bad)?;
            let threads: usize = args.get(2).and_then(|s| s.parse().ok()).ok_or_else(bad)?;
            let games: usize = args.get(3).and_then(|s| s.parse().ok()).ok_or_else(bad)?;
            let openings = read_openings(path)?;
            let random = calibrate::vs_random(500);
            let depth1 = calibrate::vs_depth1(500);
            let set = calibrate::position_set(1000);
            let (n, optimal, class) = calibrate::agreement(&set, threads);
            let (first, second) = calibrate::vs_solver(
                openings.get(..games.min(openings.len())).unwrap_or(&[]),
                threads,
            );
            Ok(format!(
                "{{\n  \"bot\": {{\"depth\": {}, \"w2\": {}, \"w3\": {}, \"wc\": {}, \"tie_base\": \"{:#018x}\"}},\n  \"vs_random_1000\": {},\n  \"vs_depth1_1000\": {},\n  \"solver_agreement\": {{\"positions\": {n}, \"optimal\": {optimal}, \"same_outcome_class\": {class}}},\n  \"vs_solver\": {{\"openings\": {}, \"bot_moves_first\": {}, \"bot_moves_second\": {}}}\n}}",
                bot::DEPTH,
                bot::W2,
                bot::W3,
                bot::WC,
                calibrate::TIE_BASE,
                random.json(),
                depth1.json(),
                games.min(openings.len()),
                first.json(),
                second.json()
            ))
        }
        Some("solve") => {
            let p = Position::from_moves(args.get(1).map_or("", String::as_str)).ok_or_else(bad)?;
            Ok(Solver::new().solve(&p).to_string())
        }
        _ => Err(bad()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(out) => {
            let _ = writeln!(io::stdout(), "{out}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = writeln!(io::stderr(), "connect4-ref: {e}");
            ExitCode::FAILURE
        }
    }
}

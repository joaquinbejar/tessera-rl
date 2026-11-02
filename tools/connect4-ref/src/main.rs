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
//! - `serve`: the pinned bot as a line protocol, for the P2-M3 example.
//!   Each input line is `TIE_SEED MOVES` (the tie seed in hex with a `0x`
//!   prefix, the moves as 1-based column digits, `-` for the empty board);
//!   each output line is the bot's 0-based column, or `error` for a line
//!   that does not parse or names a finished or illegal position. It runs
//!   until the end of its input.

use tessera_rl_connect4_ref::{board, bot, calibrate, openings, solver};

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

/// The bot's reply to one `serve` line.
fn serve_line(line: &str) -> String {
    let mut parts = line.split_whitespace();
    let reply = (|| {
        let seed = u64::from_str_radix(parts.next()?.strip_prefix("0x")?, 16).ok()?;
        let moves = parts.next()?;
        if parts.next().is_some() {
            return None;
        }
        let p = Position::from_moves(if moves == "-" { "" } else { moves })?;
        bot::Bot::new(seed).choose(&p)
    })();
    reply.map_or_else(|| "error".to_owned(), |c| c.to_string())
}

/// Serves `serve_line` replies until the end of the input.
fn serve() -> io::Result<String> {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut line = String::new();
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            return Ok(String::new());
        }
        writeln!(out, "{}", serve_line(&line))?;
        out.flush()?;
    }
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
        Some("serve") => serve(),
        _ => Err(bad()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(out) if out.is_empty() => ExitCode::SUCCESS,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_line_matches_the_bot_and_rejects_bad_lines() {
        let seed = calibrate::TIE_BASE;
        let want = Position::from_moves("4453")
            .and_then(|p| bot::Bot::new(seed).choose(&p))
            .map(|c| c.to_string());
        assert_eq!(Some(serve_line(&format!("{seed:#x} 4453\n"))), want);
        let empty = bot::Bot::new(7)
            .choose(&Position::new())
            .map(|c| c.to_string());
        assert_eq!(Some(serve_line("0x7 -")), empty);
        for bad in ["", "7 44", "0xzz 44", "0x7 48", "0x7 44 1", "0x7 1212121"] {
            assert_eq!(serve_line(bad), "error", "{bad:?}");
        }
    }
}

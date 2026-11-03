// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 2 M0 reference tools for Connect Four (ADR-0042).
//!
//! - `openings EVAL_OUT HELDOUT_OUT`: writes the 500 evaluation openings
//!   (seeds from `EVAL_BASE`) and the 200 held-out openings for checkpoint
//!   selection (seeds from `HELDOUT_BASE`, disjoint positions).
//! - `calibrate OPENINGS THREADS GAMES`: prints the bot's calibration as
//!   JSON (random play, depth-1, solver agreement, games against the
//!   solver from the first `GAMES` openings).
//! - `dev-openings EVAL HELDOUT DEV_OUT`: writes the 200 P2-M3 dev
//!   openings (seeds from `DEV_BASE`), disjoint up to mirror from both
//!   opening files, and prints the disjointness check as JSON. Pipeline
//!   checks only; never P2-G1 evaluation or checkpoint selection.
//! - `solve MOVES`: prints the exact score of a position.
//! - `serve`: the pinned bot as a line protocol, for the P2-M3 example.
//!   Each input line is `TIE_SEED MOVES` (the tie seed in hex with a `0x`
//!   prefix, the moves as 1-based column digits, `-` for the empty board);
//!   each output line is the bot's 0-based column. A line that does not
//!   parse, an illegal move, a move after a win or a full board is
//!   answered `error malformed|illegal|finished|full` and ends the process
//!   with a failure. Otherwise it runs until the end of its input.

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
/// Seed base of the 200 P2-M3 dev openings (pipeline checks only).
const DEV_BASE: u64 = 0x5032_4731_4000_0000;
const DEV_COUNT: usize = 200;
const EVAL_COUNT: usize = 500;
const HELDOUT_COUNT: usize = 200;

fn write_openings(path: &str, kind: &str, base: u64, list: &[Opening]) -> io::Result<()> {
    write_titled(path, &format!("P2-G1 {kind}"), base, list)
}

/// Writes an opening file whose header line starts with `title`.
fn write_titled(path: &str, title: &str, base: u64, list: &[Opening]) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("# SPDX-License-Identifier: MIT OR Apache-2.0\n");
    let _ = writeln!(
        out,
        "# {title} openings (ADR-0042): K={} random plies, forced-win filter L={}, seed base {base:#018x}.",
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

/// Why a `serve` line has no move. Printed as `error <kind>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServeError {
    /// Not `0xSEED MOVES`.
    Malformed,
    /// A move into a full column, or a column outside 1..=7.
    Illegal,
    /// A move after the game was already won.
    Finished,
    /// A drawn, full board: the bot has no move.
    Full,
}

impl ServeError {
    const fn kind(self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::Illegal => "illegal",
            Self::Finished => "finished",
            Self::Full => "full",
        }
    }
}

/// The bot's move for one `serve` line, validated: never a guessed move.
fn serve_line(line: &str) -> Result<u32, ServeError> {
    let mut parts = line.split_whitespace();
    let seed = parts
        .next()
        .and_then(|s| s.strip_prefix("0x"))
        .and_then(|s| u64::from_str_radix(s, 16).ok())
        .ok_or(ServeError::Malformed)?;
    let moves = parts.next().ok_or(ServeError::Malformed)?;
    if parts.next().is_some() {
        return Err(ServeError::Malformed);
    }
    let mut p = Position::new();
    if moves != "-" {
        for ch in moves.chars() {
            let col = ch.to_digit(10).ok_or(ServeError::Malformed)?;
            let col = col.checked_sub(1).ok_or(ServeError::Illegal)?;
            if col >= board::WIDTH || !p.can_play(col) {
                return Err(ServeError::Illegal);
            }
            if p.is_winning_col(col) {
                // This move wins: a later move, or the bot's, is after the end.
                p.play_col(col);
                return Err(ServeError::Finished);
            }
            p.play_col(col);
        }
    }
    bot::Bot::new(seed).choose(&p).ok_or(ServeError::Full)
}

/// Serves `serve_line` replies until the end of the input. The first bad
/// line is answered `error <kind>` and ends the process with a failure.
fn serve() -> io::Result<String> {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut line = String::new();
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            return Ok(String::new());
        }
        match serve_line(&line) {
            Ok(col) => writeln!(out, "{col}")?,
            Err(e) => {
                writeln!(out, "error {}", e.kind())?;
                out.flush()?;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("serve: {} line {:?}", e.kind(), line.trim_end()),
                ));
            }
        }
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
        Some("dev-openings") => {
            let (eval_in, held_in, dev_out) = (
                args.get(1).ok_or_else(bad)?,
                args.get(2).ok_or_else(bad)?,
                args.get(3).ok_or_else(bad)?,
            );
            let keys = |path: &str| -> io::Result<BTreeSet<u64>> {
                Ok(read_openings(path)?
                    .iter()
                    .map(Position::canonical_key)
                    .collect())
            };
            let (eval, held) = (keys(eval_in)?, keys(held_in)?);
            let taken: BTreeSet<u64> = eval.union(&held).copied().collect();
            let mut solver = Solver::new();
            let dev = openings::generate(DEV_BASE, DEV_COUNT, &taken, &mut solver);
            let dev_keys: BTreeSet<u64> = dev
                .iter()
                .filter_map(|o| Position::from_moves(&o.moves))
                .map(|p| p.canonical_key())
                .collect();
            write_titled(
                dev_out,
                "P2-M3 dev (pipeline checks only, not P2-G1)",
                DEV_BASE,
                &dev,
            )?;
            Ok(format!(
                "{{\"dev\": {}, \"distinct_up_to_mirror\": {}, \"shared_with_eval\": {}, \"shared_with_heldout\": {}}}",
                dev.len(),
                dev_keys.len(),
                dev_keys.intersection(&eval).count(),
                dev_keys.intersection(&held).count()
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
        let want = Position::from_moves("4453").and_then(|p| bot::Bot::new(seed).choose(&p));
        assert_eq!(serve_line(&format!("{seed:#x} 4453\n")).ok(), want);
        let empty = bot::Bot::new(7).choose(&Position::new());
        assert_eq!(serve_line("0x7 -").ok(), empty);
        for (bad, why) in [
            ("", ServeError::Malformed),
            ("7 44", ServeError::Malformed),
            ("0xzz 44", ServeError::Malformed),
            ("0x7", ServeError::Malformed),
            ("0x7 44 1", ServeError::Malformed),
            ("0x7 4a", ServeError::Malformed),
            ("0x7 48", ServeError::Illegal),
            ("0x7 40", ServeError::Illegal),
            ("0x7 4444444", ServeError::Illegal),
            ("0x7 1212121", ServeError::Finished),
            ("0x7 12121212", ServeError::Finished),
        ] {
            assert_eq!(serve_line(bad), Err(why), "{bad:?}");
        }
        // A drawn full board: 42 moves, no four.
        let draw = "547125662261271266215743771576315353334444";
        assert_eq!(serve_line(&format!("0x7 {draw}")), Err(ServeError::Full));
    }
}

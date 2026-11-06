// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replays every line of a best-response line file through the reference bot
//! and the engine's port, independently of the search: it reads the opening
//! file itself, recomputes each tie seed, plays on the port's own board and
//! asserts every bot reply and the final win. Each slot appears once, and the
//! slots are exactly the win class of the results.
//! `best_response_replay LINES RESULTS`.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::process::ExitCode;
use tessera_rl_connect4_ref::board::Position;
use tessera_rl_connect4_ref::bot::Bot as RefBot;
use tessera_rl_games::Connect4State;
use tessera_rl_games::connect4::board::{can_play, has_four, landing};
use tessera_rl_games::connect4::bot::{Bot as PortBot, TIE_BASE};

const HELDOUT: &str = include_str!("../../../../compat/connect4/openings-heldout.txt");

fn opening(j: u64) -> Option<&'static str> {
    HELDOUT
        .lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| {
            let mut f = l.split_whitespace();
            (f.next()?.parse::<u64>().ok()? == j).then(|| f.nth(1))?
        })
}

fn place(s: &mut Connect4State, col: u32) -> Result<(), String> {
    if !can_play(s.occupied(), col) {
        return Err(format!("illegal column {col}"));
    }
    let bit = landing(s.occupied(), col);
    let side = (s.moves % 2) as usize;
    let stones = s.stones.get_mut(side).ok_or("side")?;
    *stones |= bit;
    s.moves += 1;
    Ok(())
}

fn replay(line: &str) -> Result<(), String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    let [j, colour, seed, moves] = f.as_slice() else {
        return Err("malformed".to_owned());
    };
    let j: u64 = j.parse().map_err(|_| "bad j")?;
    let tie = TIE_BASE + j;
    if *seed != format!("0x{tie:x}") {
        return Err(format!("tie seed {seed}, expected 0x{tie:x}"));
    }
    let open = opening(j).ok_or("no such opening")?;
    let mut refp = Position::new();
    let mut port = Connect4State::default();
    for c in open.bytes() {
        let col = u32::from(c - b'1');
        refp.play_col(col);
        place(&mut port, col)?;
    }
    let (rb, pb) = (RefBot::new(tie), PortBot::new(tie));
    let first = match *colour {
        "first" => true,
        "second" => false,
        other => return Err(format!("colour {other}")),
    };
    let cols: Vec<u32> = moves
        .split(',')
        .map(|m| m.parse::<u32>().map_err(|_| "bad move".to_owned()))
        .collect::<Result<_, _>>()?;
    let bot_turn = usize::from(first);
    for (i, &c) in cols.iter().enumerate() {
        let side = (port.moves % 2) as usize;
        if i % 2 == bot_turn {
            if rb.choose(&refp) != Some(c) {
                return Err(format!("ply {i}: reference bot differs from {c}"));
            }
            if pb.choose(&port) != Some(c) {
                return Err(format!("ply {i}: port bot differs from {c}"));
            }
        }
        refp.play_col(c);
        place(&mut port, c)?;
        let four = port.stones.get(side).is_some_and(|&s| has_four(s));
        let last = i + 1 == cols.len();
        if four != last {
            return Err(format!("ply {i}: four {four}, last {last}"));
        }
        if last && i % 2 == bot_turn {
            return Err("the final four is the bot's".to_owned());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let path = std::env::args().nth(1).unwrap_or_default();
    let results = std::env::args().nth(2).unwrap_or_default();
    let (Ok(text), Ok(classes)) = (
        std::fs::read_to_string(&path),
        std::fs::read_to_string(&results),
    ) else {
        return ExitCode::FAILURE;
    };
    let wins: BTreeSet<(&str, &str)> = classes
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            let slot = (f.next()?, f.next()?);
            (f.next()? == "win").then_some(slot)
        })
        .collect();
    let mut seen = BTreeSet::new();
    let (mut ok, mut report) = (0u32, String::new());
    let mut bad = 0u32;
    for l in text.lines() {
        let mut f = l.split_whitespace();
        let slot = (f.next().unwrap_or(""), f.next().unwrap_or(""));
        if !seen.insert(slot) {
            bad += 1;
            let _ = writeln!(report, "FAIL {l}: duplicate slot");
            continue;
        }
        match replay(l) {
            Ok(()) => ok += 1,
            Err(e) => {
                bad += 1;
                let _ = writeln!(report, "FAIL {l}: {e}");
            }
        }
    }
    if seen != wins {
        bad += 1;
        let _ = writeln!(
            report,
            "FAIL the slots differ from the win class of the results"
        );
    }
    let _ = writeln!(report, "replayed ok {ok} failed {bad}");
    let _ = std::io::Write::write_all(&mut std::io::stdout(), report.as_bytes());
    if bad == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

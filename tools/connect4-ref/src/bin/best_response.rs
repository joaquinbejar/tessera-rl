// SPDX-License-Identifier: MIT OR Apache-2.0

//! The P2-G1 best-response measurement, design revision 4.
//!
//! - `run OUT THREADS OPENINGS`: the search over the first `OPENINGS` held-out
//!   openings; writes `results.tsv`, `lines.txt` and `summary.txt` in `OUT`. An
//!   8 core-hour wall abort writes `VOID` and fails.
//! - `recheck OUT LIMIT`: reruns up to `LIMIT` exhausted non-wins with no memo.
//! - `solver OUT THREADS`: the perfect solver on the 400 slots, `solver.tsv`.

#![expect(
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "a one-off measurement tool; thread counts are small"
)]

use std::fmt::Write as _;
use std::fs;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tessera_rl_connect4_ref::best_response::{
    Class, Outcome, PASS1_CAP, PASS2_CAP, TABLE_LIMIT, TOTAL_BUDGET, openings, run_budgeted,
    search_slot,
};
use tessera_rl_connect4_ref::board::Position;
use tessera_rl_connect4_ref::bot::Bot;
use tessera_rl_connect4_ref::calibrate::{Outcome as Game, Player, TIE_BASE, play};
use tessera_rl_connect4_ref::solver::Solver;

const ABORT_CORE_HOURS: f64 = 8.0;

type Slot = (u64, Position, bool);

fn slots(limit: usize) -> Vec<Slot> {
    let mut out = Vec::new();
    for (j, m) in openings().into_iter().take(limit) {
        if let Some(p) = Position::from_moves(&m) {
            out.push((j, p, true));
            out.push((j, p, false));
        }
    }
    out
}

fn colour(first: bool) -> &'static str {
    if first { "first" } else { "second" }
}

fn class_name(c: Option<&Class>) -> &'static str {
    match c {
        Some(Class::Win(_)) => "win",
        Some(Class::Exhausted) => "exhausted",
        Some(Class::Unresolved) => "unresolved",
        Some(Class::Aborted) => "aborted",
        None => "not_started",
    }
}

fn search(slot: Option<&Slot>, cap: u64, abort: &AtomicBool) -> Outcome {
    let Some(&(j, p, first)) = slot else {
        return Outcome {
            class: Class::Aborted,
            nodes: 0,
            table: 0,
        };
    };
    let bot = Bot::new(TIE_BASE + j);
    search_slot(
        p,
        first,
        &|q: &Position| bot.choose(q),
        (cap, TABLE_LIMIT),
        true,
        abort,
    )
}

fn run(out: &str, threads: usize, limit: usize) -> Result<(), String> {
    let slots = slots(limit);
    let abort = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let start = Instant::now();
    let limit_s = ABORT_CORE_HOURS * 3600.0 / threads as f64;
    let mut pass1 = Vec::new();
    let mut pass2 = Vec::new();
    let mut left = TOTAL_BUDGET;
    thread::scope(|s| {
        s.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                if start.elapsed().as_secs_f64() >= limit_s {
                    abort.store(true, Ordering::Relaxed);
                    return;
                }
                thread::sleep(Duration::from_millis(200));
            }
        });
        (pass1, left) = run_budgeted(slots.len(), PASS1_CAP, left, threads, |i, cap| {
            search(slots.get(i), cap, &abort)
        });
        let unresolved: Vec<usize> = pass1
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                !matches!(
                    o.as_ref().map(|x| &x.class),
                    Some(Class::Win(_) | Class::Exhausted)
                )
            })
            .map(|(i, _)| i)
            .collect();
        let (results, l) = run_budgeted(unresolved.len(), PASS2_CAP, left, threads, |k, cap| {
            search(unresolved.get(k).and_then(|&i| slots.get(i)), cap, &abort)
        });
        left = l;
        pass2 = unresolved.into_iter().zip(results).collect::<Vec<_>>();
        done.store(true, Ordering::Relaxed);
    });
    if abort.load(Ordering::Relaxed) {
        fs::write(
            format!("{out}/VOID"),
            "8 core-hour abort: this run is void\n",
        )
        .map_err(|e| e.to_string())?;
        return Err("aborted: void".to_owned());
    }
    let mut results = String::new();
    let mut lines = String::new();
    let (mut w, mut x, mut u, mut peak) = (0u32, 0u32, 0u32, 0usize);
    let mut nodes = 0u64;
    let finals: Vec<(Option<&Outcome>, Option<&Outcome>)> = pass1
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let second = pass2
                .iter()
                .find(|(k, _)| *k == i)
                .and_then(|(_, r)| r.as_ref());
            (o.as_ref(), second)
        })
        .collect();
    for (i, (a, b)) in finals.iter().enumerate() {
        let Some(&(j, _, first)) = slots.get(i) else {
            continue;
        };
        let last = b.or(*a);
        let class = last.map(|o| &o.class);
        let charged = a.map_or(0, |o| o.nodes) + b.map_or(0, |o| o.nodes);
        nodes += charged;
        peak = peak
            .max(a.map_or(0, |o| o.table))
            .max(b.map_or(0, |o| o.table));
        match class {
            Some(Class::Win(line)) => {
                w += 1;
                let moves: Vec<String> = line.iter().map(u32::to_string).collect();
                let _ = writeln!(
                    lines,
                    "{j} {} 0x{:x} {}",
                    colour(first),
                    TIE_BASE + j,
                    moves.join(",")
                );
            }
            Some(Class::Exhausted) => x += 1,
            _ => u += 1,
        }
        let _ = writeln!(
            results,
            "{j}\t{}\t{}\t{charged}\t{}\t{}",
            colour(first),
            class_name(class),
            a.map_or(0, |o| o.table),
            b.map_or(0, |o| o.table)
        );
    }
    let summary = format!(
        "slots {}\nW {w}\nX {x}\nU {u}\nW+U {}\nnodes_charged {nodes}\nbudget_left {left}\npeak_table_entries {peak}\nthreads {threads}\nwall_s {:.1}\n",
        slots.len(),
        w + u,
        start.elapsed().as_secs_f64()
    );
    for (name, text) in [
        ("results.tsv", &results),
        ("lines.txt", &lines),
        ("summary.txt", &summary),
    ] {
        fs::write(format!("{out}/{name}"), text).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn recheck(out: &str, limit: usize) -> Result<(), String> {
    let text = fs::read_to_string(format!("{out}/results.tsv")).map_err(|e| e.to_string())?;
    let opening = openings();
    let abort = AtomicBool::new(false);
    let mut report = String::new();
    let exhausted = text
        .lines()
        .filter(|l| l.split('\t').nth(2) == Some("exhausted"));
    for l in exhausted.take(limit) {
        let mut f = l.split('\t');
        let j: u64 = f.next().and_then(|v| v.parse().ok()).ok_or("bad j")?;
        let first = f.next() == Some("first");
        let moves = opening.iter().find(|o| o.0 == j).ok_or("bad opening")?;
        let p = Position::from_moves(&moves.1).ok_or("bad moves")?;
        let bot = Bot::new(TIE_BASE + j);
        let o = search_slot(
            p,
            first,
            &|q: &Position| bot.choose(q),
            (PASS2_CAP, 0),
            false,
            &abort,
        );
        let _ = writeln!(
            report,
            "{j}\t{}\t{}\t{}",
            colour(first),
            class_name(Some(&o.class)),
            o.nodes
        );
    }
    fs::write(format!("{out}/recheck.tsv"), report).map_err(|e| e.to_string())
}

fn solver(out: &str, threads: usize) -> Result<(), String> {
    let slots = slots(usize::MAX);
    let next = Mutex::new(0usize);
    let results = Mutex::new(vec![None; slots.len()]);
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut sv = Some(Box::new(Solver::new()));
                loop {
                    let i = {
                        let Ok(mut g) = next.lock() else { return };
                        let i = *g;
                        *g += 1;
                        i
                    };
                    let Some(&(j, p, first)) = slots.get(i) else {
                        return;
                    };
                    let bot = Bot::new(TIE_BASE + j);
                    let Some(solver) = sv.take() else { return };
                    let mut b = Player::Solver(solver);
                    // The solver is the candidate: the bot moves first when it does not.
                    let o = play(p, &mut Player::Bot(&bot), &mut b, !first);
                    if let Player::Solver(back) = b {
                        sv = Some(back);
                    }
                    if let Ok(mut r) = results.lock()
                        && let Some(slot) = r.get_mut(i)
                    {
                        *slot = Some(o);
                    }
                }
            });
        }
    });
    let results = results.into_inner().map_err(|e| e.to_string())?;
    let mut text = String::new();
    let mut wins = 0;
    for (slot, r) in slots.iter().zip(&results) {
        // `play` reports for the bot.
        let name = match r {
            Some(Game::Loss) => {
                wins += 1;
                "win"
            }
            Some(Game::Draw) => "draw",
            Some(Game::Win) => "loss",
            None => "none",
        };
        let _ = writeln!(text, "{}\t{}\t{name}", slot.0, colour(slot.2));
    }
    let _ = writeln!(text, "# solver wins {wins} of {}", slots.len());
    fs::write(format!("{out}/solver.tsv"), text).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize| a.get(i).map_or("", String::as_str);
    let num = |i: usize| arg(i).parse::<usize>().unwrap_or(0);
    let r = match arg(1) {
        "run" if num(3) > 0 && num(4) > 0 => run(arg(2), num(3), num(4)),
        "recheck" => recheck(arg(2), num(3)),
        "solver" if num(3) > 0 => solver(arg(2), num(3)),
        _ => Err(
            "usage: run OUT THREADS OPENINGS | recheck OUT LIMIT | solver OUT THREADS".to_owned(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let _ = std::io::Write::write_all(&mut std::io::stderr(), format!("{e}\n").as_bytes());
            ExitCode::FAILURE
        }
    }
}

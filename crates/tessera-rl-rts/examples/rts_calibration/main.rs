// SPDX-License-Identifier: MIT OR Apache-2.0

//! RTS stage 5a calibration among scripted bots
//! (`docs/reviews/rts-stage5-plan.md` r2, section 2). Not in any step path.
//!
//!     cargo run --release -p tessera-rl-rts --example rts_calibration -- \
//!         --seeds 0..200 --size 16 --units 16 --ticks 4 --limit 600 --out tuning.json
//!
//! Plays the three pre-registered pairings (Rush against Prospector, Rush
//! against Random-legal, Prospector against Random-legal), each game of the
//! seed list once per seat, and writes one JSON with counts, decided rates,
//! Wilson 95% intervals and length figures. Deterministic: the same
//! arguments give the same bytes.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::float_arithmetic,
    clippy::panic,
    reason = "an offline calibration tool, not library code"
)]

mod games;

use std::fmt::Write as _;

use tessera_rl_rts::RtsConfig;
use tessera_rl_rts::bots::Bot;

use crate::games::{Summary, pairing, wilson};

/// The pre-registered pairings, in order (plan r2, C1).
const PAIRINGS: [(&str, Bot, Bot); 3] = [
    ("rush-prospector", Bot::Rush, Bot::Prospector),
    ("rush-random", Bot::Rush, Bot::RandomLegal),
    ("prospector-random", Bot::Prospector, Bot::RandomLegal),
];

/// The arguments the tool takes; anything else is refused, so a typo can
/// never silently fall back to a default in a pre-registered run.
const KEYS: [&str; 6] = [
    "--seeds", "--size", "--units", "--ticks", "--limit", "--out",
];

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn num<T: core::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    arg(args, key).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{key} takes a number, got {v:?}"))
    })
}

/// The `percent` quantile of `sorted`, nearest rank, in integers.
fn quantile(sorted: &[u32], percent: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) * percent + 50) / 100;
    sorted[i.min(sorted.len() - 1)]
}

fn lengths(out: &mut String, key: &str, v: &[u32]) {
    let mut s = v.to_vec();
    s.sort_unstable();
    let _ = write!(
        out,
        "\"{key}\": {{\"n\": {}, \"min\": {}, \"median\": {}, \"p90\": {}, \"max\": {}}}",
        s.len(),
        s.first().copied().unwrap_or(0),
        quantile(&s, 50),
        quantile(&s, 90),
        s.last().copied().unwrap_or(0)
    );
}

fn interval(k: u32, n: u32) -> String {
    let (lo, hi) = wilson(k, n);
    format!("[{lo:.4}, {hi:.4}]")
}

/// One pairing as JSON: counts per seat and in total, the decided rate, A's
/// share of the decided games and B's, with Wilson intervals, the
/// truncation rate and the length figures.
fn summary_json(name: &str, s: &Summary) -> String {
    let t: [u32; 4] = core::array::from_fn(|k| s.by_seat[0][k] + s.by_seat[1][k]);
    let games = t.iter().sum::<u32>();
    let decided = t[0] + t[1] + t[2];
    let mut out = String::new();
    let _ = write!(out, "\"{name}\": {{\"games\": {games}, ");
    let _ = write!(
        out,
        "\"a_wins\": {}, \"b_wins\": {}, \"draws\": {}, \"truncated\": {}, ",
        t[0], t[1], t[2], t[3]
    );
    for (seat, row) in s.by_seat.iter().enumerate() {
        let _ = write!(out, "\"a_in_seat_{seat}\": {row:?}, ");
    }
    let _ = write!(
        out,
        "\"decided_rate\": {:.4}, \"decided_interval\": {}, ",
        f64::from(decided) / f64::from(games.max(1)),
        interval(decided, games)
    );
    let _ = write!(
        out,
        "\"a_share_of_decided\": {:.4}, \"a_interval\": {}, \"b_share_of_decided\": {:.4}, \"b_interval\": {}, ",
        f64::from(t[0]) / f64::from(decided.max(1)),
        interval(t[0], decided),
        f64::from(t[1]) / f64::from(decided.max(1)),
        interval(t[1], decided)
    );
    let _ = write!(
        out,
        "\"truncation_rate\": {:.4}, ",
        f64::from(t[3]) / f64::from(games.max(1))
    );
    lengths(&mut out, "lengths", &s.lengths);
    out.push_str(", ");
    lengths(&mut out, "decided_lengths", &s.decided_lengths);
    out.push('}');
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut rest = args.iter().skip(1);
    while let Some(key) = rest.next() {
        if !KEYS.contains(&key.as_str()) || rest.next().is_none() {
            eprintln!("refused: unknown argument or missing value: {key:?}; known: {KEYS:?}");
            std::process::exit(2);
        }
    }
    let seeds = arg(&args, "--seeds").unwrap_or_else(|| "0..200".to_owned());
    let (lo, hi) = seeds
        .split_once("..")
        .and_then(|(a, b)| Some((a.parse::<u64>().ok()?, b.parse::<u64>().ok()?)))
        .unwrap_or_else(|| panic!("--seeds takes A..B, got {seeds:?}"));
    let cfg = RtsConfig {
        size: num(&args, "--size", 16),
        decision_ticks: num(&args, "--ticks", 4),
        max_units: num(&args, "--units", 16),
        step_limit: num(&args, "--limit", 600),
        seat_swap: false,
    };
    eprintln!(
        "config: size {} decision_ticks {} max_units {} step_limit {}; seeds {lo}..{hi}, {} games per seat per pairing",
        cfg.size,
        cfg.decision_ticks,
        cfg.max_units,
        cfg.step_limit,
        hi.saturating_sub(lo)
    );
    // What the record freezes travels with the measurement: the environment
    // version and the bots' constants (knob 1).
    let mut json = format!(
        "{{\"env_version\": {}, \"bots\": {{\"RUSH_DRONES\": {}, \"PROSPECTOR_DRONES\": {}, \"PROSPECTOR_ARMY\": {}}}, \"config\": {{\"size\": {}, \"decision_ticks\": {}, \"max_units\": {}, \"step_limit\": {}}}, \"seeds\": [{lo}, {hi}], \"games_per_seat\": {}, \"pairings\": {{",
        tessera_rl_rts::env::ENV_VERSION,
        tessera_rl_rts::bots::RUSH_DRONES,
        tessera_rl_rts::bots::PROSPECTOR_DRONES,
        tessera_rl_rts::bots::PROSPECTOR_ARMY,
        cfg.size,
        cfg.decision_ticks,
        cfg.max_units,
        cfg.step_limit,
        hi.saturating_sub(lo)
    );
    for (i, (name, a, b)) in PAIRINGS.iter().enumerate() {
        eprintln!("{name} over seeds {lo}..{hi}");
        let s = pairing(&cfg, *a, *b, lo..hi);
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&summary_json(name, &s));
    }
    json.push_str("}}\n");
    match arg(&args, "--out") {
        Some(path) => std::fs::write(&path, &json).expect("write the report"),
        None => print!("{json}"),
    }
}

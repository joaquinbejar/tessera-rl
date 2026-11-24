//! The checks `cargo tessera check` runs (SPEC 10.2), from
//! `tessera_rl_engine::host_testing`: determinism (two Inline pools and a
//! pool with workers, every output hashed per call), continuation after a
//! reseed, space conformance and the host invariants;
{{#snapshots}}
//! continuation after a restored snapshot and the snapshot contract;
{{/snapshots}}
//! and the golden hashes in
//! tests/golden.json, recorded by `cargo tessera check --bless`
//! (`TESSERA_BLESS=1`) and compared on every other run.

use core::num::{NonZeroU32, NonZeroUsize};
use std::path::Path;

use tessera_rl_engine::host_testing::{
    CheckRun, check_continuation, check_determinism, check_invariants, check_spaces,
};
use {{py_package}}::{AGENTS, Counter, CounterConfig, ENV_ID};
{{#snapshots}}
use tessera_rl_engine::PoolOptions;
use tessera_rl_engine::host_testing::{
    ContractStep, check_snapshot_contract, check_snapshot_continuation,
};
use tessera_rl_num::identity::Seed;
{{/snapshots}}

/// Instances per pool.
const INSTANCES: u32 = 8;
/// The pool seed.
const SEED: u64 = 351;
/// Steps after the first reset: enough for several episodes.
const CALLS: usize = 200;
/// The golden hashes, one per call of `check_determinism`.
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json");

fn cfg() -> CounterConfig {
    CounterConfig::default()
}

fn run() -> CheckRun {
    CheckRun::new(NonZeroU32::new(INSTANCES).unwrap(), SEED, CALLS)
        .with_workers(NonZeroUsize::new(4).unwrap())
        .with_max_episode_steps(NonZeroU32::new(48))
}

/// The golden file: the id and one hex hash per call.
fn golden_json(hashes: &[u64]) -> String {
    let rows: Vec<String> = hashes.iter().map(|h| format!("    \"{h:016x}\"")).collect();
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"env_id\": \"{ENV_ID}\",\n  \"calls\": [\n"));
    out.push_str(&rows.join(",\n"));
    out.push_str("\n  ]\n}\n");
    out
}

/// The hashes of a golden file: every 16-digit hex string.
fn read_golden(text: &str) -> Vec<u64> {
    text.split('"')
        .filter(|t| t.len() == 16 && t.bytes().all(|b| b.is_ascii_hexdigit()))
        .filter_map(|t| u64::from_str_radix(t, 16).ok())
        .collect()
}

#[test]
fn test_determinism_inline_and_workers_agree_and_match_golden() {
    let report = check_determinism::<Counter, AGENTS>(&cfg, &run()).unwrap();
    let path = Path::new(GOLDEN);
    if std::env::var_os("TESSERA_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, golden_json(&report.call_hashes)).unwrap();
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        panic!("no tests/golden.json: record it with `cargo tessera check --bless`, then commit it");
    };
    assert!(
        text.contains(&format!("\"{ENV_ID}\"")),
        "tests/golden.json was recorded for another id: re-bless it"
    );
    let want = read_golden(&text);
    assert_eq!(
        want.len(),
        report.call_hashes.len(),
        "tests/golden.json has another number of calls: re-bless it"
    );
    if let Some(call) = want.iter().zip(&report.call_hashes).position(|(a, b)| a != b) {
        panic!(
            "call {call} differs from tests/golden.json: the outputs changed. If that is \
             intended, bump the id's version and ENV_VERSION, then `cargo tessera check --bless`"
        );
    }
}

#[test]
fn test_continuation_reseed_matches_a_fresh_pool() {
    let report = check_continuation::<Counter, AGENTS>(&cfg, &run(), 37).unwrap();
    assert_eq!(report.call_hashes.len(), CALLS + 1);
}
{{#snapshots}}

#[test]
fn test_snapshot_contract_holds() {
    let rows = usize::try_from(INSTANCES).unwrap() * AGENTS;
    let steps: Vec<ContractStep> = (0..64)
        .map(|t| {
            let discrete = (0..rows).map(|i| i32::try_from((t + i) % 3).unwrap()).collect();
            (discrete, Vec::new())
        })
        .collect();
    let options = PoolOptions::new(NonZeroU32::new(INSTANCES).unwrap(), Seed::new(SEED));
    check_snapshot_contract::<Counter, AGENTS>(&cfg, options, &steps).unwrap();
}

#[test]
fn test_snapshot_continuation_replays_equal() {
    let report = check_snapshot_continuation::<Counter, AGENTS>(&cfg, &run(), 23).unwrap();
    assert_eq!(report.call_hashes.len(), CALLS);
}
{{/snapshots}}

#[test]
fn test_spaces_outputs_conform() {
    let report = check_spaces::<Counter, AGENTS>(&cfg, &run()).unwrap();
    assert!(report.values > 0);
}

#[test]
fn test_invariants_hold() {
    let report = check_invariants::<Counter, AGENTS>(&cfg, &run()).unwrap();
    assert!(report.episodes_ended > 0, "the run must end episodes");
    assert_eq!(report.aborts, 0);
}

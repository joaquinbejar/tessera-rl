//! The checks `cargo tessera check` runs (SPEC 10.2), on
//! `tessera_rl_engine::EnvPool`:
//! - determinism: two Inline pools, and pools with 4 workers at chunk size
//!   1 and at the default, give the same per-call hashes, and those hashes
//!   match the goldens in tests/golden.json (recorded by
//!   `cargo tessera check --bless`, `TESSERA_BLESS=1`);
//! - continuation (SPEC 9.1): every instance exported mid-run and imported
//!   into a fresh pool with another seed continues exactly as the
//!   uninterrupted run;
//! - snapshot round trip: restoring every instance in place replays equal;
//! - spaces: every observation value lies within its bounds;
//! - validation: an import of a state that breaks `Validate` is refused.
//!
//! A call's hash covers every instance's record (header and state), the
//! observations, rewards, masks and episode flags.

use core::num::{NonZeroU32, NonZeroUsize};
use std::fmt::Write as _;
use std::path::Path;

use tessera_rl_core::encoding::StateHasher;
use tessera_rl_engine::{
    ActionInput, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Record, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;
use {{py_package}}::{AGENTS, BOUND, CHOICES, Counter, CounterConfig, CounterState, ENV_ID};

/// A pool of the environment, with its action masks.
type Pool = EnvPool<Counter, Masked>;

/// Instances per pool: several chunks at the default chunk size.
const INSTANCES: u32 = 256;
/// The pool seed.
const SEED: u64 = 628;
/// Steps after the first reset: enough for several episodes.
const CALLS: u32 = 200;
/// The engine's step limit.
const LIMIT: u32 = 48;
/// The golden hashes, one per call.
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json");

/// The engine's default chunk size.
fn default_chunk() -> u32 {
    PoolOptions::new(NonZeroU32::MIN, Seed::new(0)).chunk_size.get()
}

/// A reset pool: Inline for 0 workers.
fn pool(seed: u64, chunk: u32, workers: usize) -> Result<Pool, EngineError> {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(chunk).unwrap_or(NonZeroU32::MIN),
        max_episode_steps: NonZeroU32::new(LIMIT),
        ..PoolOptions::new(
            NonZeroU32::new(INSTANCES).unwrap_or(NonZeroU32::MIN),
            Seed::new(seed),
        )
    };
    let mut pool = Pool::with_masks(CounterConfig::default(), options)?;
    if let Some(w) = NonZeroUsize::new(workers) {
        pool = pool.with_workers(SyncOptions::new(w))?;
    }
    pool.reset_all(Reseed::Keep)?;
    Ok(pool)
}

/// splitmix64: the action generator's mixer, independent of the engine.
const fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The actions of call `t`: a hashed choice per row, or the no-op (0) when
/// the mask forbids it.
fn actions(p: &Pool, t: u32) -> Result<Vec<i32>, EngineError> {
    let masks = p.outputs()?.masks();
    let width = usize::try_from(CHOICES).unwrap_or(1);
    let rows = usize::try_from(INSTANCES).unwrap_or(0).saturating_mul(AGENTS);
    Ok((0..rows)
        .map(|row| {
            let key = u64::try_from(row).unwrap_or(0) ^ (u64::from(t) << 32) ^ SEED;
            let choice = usize::try_from(mix(key) % u64::from(CHOICES)).unwrap_or(0);
            let legal = masks
                .get(row.saturating_mul(width).saturating_add(choice))
                .is_some_and(|m| m.get());
            if legal {
                i32::try_from(choice).unwrap_or(0)
            } else {
                0
            }
        })
        .collect())
}

/// The hash of the pool's records and outputs after a call.
fn digest(p: &Pool) -> Result<u64, EngineError> {
    let out = p.outputs()?;
    let mut h = StateHasher::new();
    for id in 0..INSTANCES {
        h.add_bytes(&p.state_hash(id)?.get().to_le_bytes());
    }
    if let Some(obs) = out.observation(0) {
        h.add_bytes(obs);
    }
    for r in out.rewards() {
        h.add_bytes(&r.to_bits().to_le_bytes());
    }
    let flags = out.masks().iter().chain(out.episode_terminated()).chain(out.episode_truncated());
    for f in flags {
        h.add_bytes(&[f.to_byte()]);
    }
    Ok(h.finish().get())
}

/// Steps `p` with `acts`.
fn step(p: &mut Pool, acts: &[i32]) -> Result<(), EngineError> {
    p.step(ActionInput {
        discrete: acts,
        continuous: &[],
    })
    .map(|_| ())
}

/// The hashes of the first reset and of `calls` generated steps.
fn run(p: &mut Pool, calls: u32) -> Result<Vec<u64>, EngineError> {
    let mut hashes = vec![digest(p)?];
    for t in 1..=calls {
        let acts = actions(p, t)?;
        step(p, &acts)?;
        hashes.push(digest(p)?);
    }
    Ok(hashes)
}

/// Every instance's record.
fn snapshot(p: &Pool) -> Result<Vec<Record<Counter>>, EngineError> {
    (0..INSTANCES).map(|id| p.snapshot(id)).collect()
}

/// `calls` more generated steps: the actions of each, and the hash after.
fn record(p: &mut Pool, from: u32, calls: u32) -> Result<(Vec<Vec<i32>>, Vec<u64>), EngineError> {
    let (mut rows, mut hashes) = (Vec::new(), Vec::new());
    for t in from..from.saturating_add(calls) {
        let acts = actions(p, t)?;
        step(p, &acts)?;
        rows.push(acts);
        hashes.push(digest(p)?);
    }
    Ok((rows, hashes))
}

/// The hashes after replaying `rows`.
fn replay(p: &mut Pool, rows: &[Vec<i32>]) -> Result<Vec<u64>, EngineError> {
    rows.iter()
        .map(|acts| {
            step(p, acts)?;
            digest(p)
        })
        .collect()
}

/// The golden file: the id and one hex hash per call.
fn golden_json(hashes: &[u64]) -> String {
    let rows: Vec<String> = hashes.iter().map(|h| format!("    \"{h:016x}\"")).collect();
    let mut out = String::from("{\n");
    // Writing to a String cannot fail.
    let _ = writeln!(out, "  \"env_id\": \"{ENV_ID}\",\n  \"calls\": [");
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
    let want = run(&mut pool(SEED, default_chunk(), 0).unwrap(), CALLS).unwrap();
    assert_eq!(
        run(&mut pool(SEED, default_chunk(), 0).unwrap(), CALLS).unwrap(),
        want,
        "two Inline pools disagree"
    );
    for chunk in [1, default_chunk()] {
        let got = run(&mut pool(SEED, chunk, 4).unwrap(), CALLS).unwrap();
        assert_eq!(got, want, "4 workers at chunk size {chunk} disagree with Inline");
    }
    let path = Path::new(GOLDEN);
    if std::env::var_os("TESSERA_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, golden_json(&want)).unwrap();
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        panic!("no tests/golden.json: record it with `cargo tessera check --bless`, then commit it");
    };
    assert!(
        text.contains(&format!("\"{ENV_ID}\"")),
        "tests/golden.json was recorded for another id: re-bless it"
    );
    let golden = read_golden(&text);
    assert_eq!(
        golden.len(),
        want.len(),
        "tests/golden.json has another number of calls: re-bless it"
    );
    if let Some(call) = golden.iter().zip(&want).position(|(a, b)| a != b) {
        panic!(
            "call {call} differs from tests/golden.json: the outputs changed. If that is \
             intended, bump the id's version and ENV_VERSION, then `cargo tessera check --bless`"
        );
    }
}

#[test]
fn test_continuation_import_into_a_fresh_pool_matches_the_uninterrupted_run() {
    for at in [0, 37, 101] {
        let mut a = pool(SEED, default_chunk(), 0).unwrap();
        for t in 1..=at {
            let acts = actions(&a, t).unwrap();
            step(&mut a, &acts).unwrap();
        }
        let records = snapshot(&a).unwrap();
        let (rows, want) = record(&mut a, at + 1, 60).unwrap();
        // A fresh pool with another seed, by import: the untrusted path.
        let mut b = pool(SEED ^ 0xf00d, 1, 4).unwrap();
        let decoded: Vec<DecodedRecord<Counter>> = records
            .iter()
            .map(|r| DecodedRecord::from_parts(*r.header(), *r.state()).unwrap())
            .collect();
        let ids: Vec<u32> = (0..INSTANCES).collect();
        b.import(&ids, &decoded).unwrap();
        assert_eq!(replay(&mut b, &rows).unwrap(), want, "import at call {at}");
    }
}

#[test]
fn test_snapshot_restore_in_place_replays_equal() {
    let mut p = pool(SEED, default_chunk(), 0).unwrap();
    for t in 1..=23 {
        let acts = actions(&p, t).unwrap();
        step(&mut p, &acts).unwrap();
    }
    let records = snapshot(&p).unwrap();
    let (rows, want) = record(&mut p, 24, 80).unwrap();
    p.restore_all(&records).unwrap();
    assert_eq!(replay(&mut p, &rows).unwrap(), want);
}

#[test]
fn test_spaces_observations_within_bounds() {
    let mut p = pool(SEED, default_chunk(), 0).unwrap();
    let mut episodes = 0_u32;
    for t in 1..=CALLS {
        let acts = actions(&p, t).unwrap();
        step(&mut p, &acts).unwrap();
        let out = p.outputs().unwrap();
        let obs = out.observation(0).unwrap();
        let rows = usize::try_from(INSTANCES).unwrap() * AGENTS;
        assert_eq!(obs.len(), rows * 4 * 4, "4 i32 values per agent");
        for v in obs.as_chunks::<4>().0 {
            let v = i32::from_le_bytes(*v);
            assert!((0..=BOUND).contains(&v), "{v} outside 0..={BOUND}");
        }
        assert_eq!(out.masks().len(), rows * usize::try_from(CHOICES).unwrap());
        let ended = out.episode_terminated().iter().chain(out.episode_truncated());
        episodes += u32::try_from(ended.filter(|f| f.get()).count()).unwrap();
    }
    assert!(episodes > 0, "the run must end episodes");
}

#[test]
fn test_import_of_an_invalid_state_refused() {
    let mut p = pool(SEED, default_chunk(), 0).unwrap();
    let record = p.snapshot(0).unwrap();
    let cfg = CounterConfig::default();
    let bad = CounterState {
        counter: cfg.target + 1,
        ..*record.state()
    };
    let decoded = DecodedRecord::from_parts(*record.header(), bad).unwrap();
    assert!(p.import(&[0], &[decoded]).is_err());
    assert_eq!(p.snapshot(0).unwrap().hash(), record.hash(), "nothing written");
}

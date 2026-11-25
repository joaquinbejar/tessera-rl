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
//!
//! A failed comparison names its kind, the first diverging call and
//! instance, and a hint at the usual cause (`first_divergence`).

use core::num::{NonZeroU32, NonZeroUsize};
use std::fmt::Write as _;
use std::path::Path;

use study_ref_b::{AGENTS, BOUND, CHOICES, ENV_ID, Nim, NimConfig, NimState};
use tessera_rl_core::encoding::StateHasher;
use tessera_rl_engine::{
    ActionInput, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Record, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

/// A pool of the environment, with its action masks.
type Pool = EnvPool<Nim, Masked>;

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
    PoolOptions::new(NonZeroU32::MIN, Seed::new(0))
        .chunk_size
        .get()
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
    let mut pool = Pool::with_masks(NimConfig::default(), options)?;
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
    let rows = usize::try_from(INSTANCES)
        .unwrap_or(0)
        .saturating_mul(AGENTS);
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

/// The call the snapshot test takes its records after; it replays from the
/// next call.
const SNAPSHOT_AT: u32 = 23;

/// What one call leaves: its hash and every instance's record hash.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Call {
    /// The call's hash: records and outputs (the golden value).
    hash: u64,
    /// Each instance's record hash, by instance id.
    states: Vec<u64>,
}

/// The kind of a comparison, with the usual cause of its failure.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Two Inline pools in the same process.
    InlineInline,
    /// 4 workers at a chunk size against Inline.
    Workers(u32),
    /// An import into a fresh pool against the uninterrupted run.
    Resume(u32),
    /// A restore in place against the run it was taken from.
    Snapshot,
}

impl Kind {
    /// The absolute call (as numbered in tests/golden.json) of the first
    /// entry of the runs this kind compares.
    fn first_call(self) -> usize {
        let from = match self {
            Self::InlineInline | Self::Workers(_) => 0,
            Self::Resume(at) => at.saturating_add(1),
            Self::Snapshot => SNAPSHOT_AT.saturating_add(1),
        };
        usize::try_from(from).unwrap_or(usize::MAX)
    }

    /// The kind's name and its hint.
    fn describe(self) -> String {
        match self {
            Self::InlineInline => "Inline vs Inline (same process): state that outlives a pool, \
                such as a `static`, a `thread_local!` or `HashMap` iteration order"
                .to_owned(),
            Self::Workers(chunk) => format!(
                "Inline vs 4 workers at chunk size {chunk}: state shared across instances, or \
                 keyed by storage slot or thread"
            ),
            Self::Resume(at) => format!(
                "import at call {at} vs the uninterrupted run: state the step reads that the \
                 record does not hold"
            ),
            Self::Snapshot => "restore in place vs the uninterrupted run: state the step reads \
                that the record does not hold"
                .to_owned(),
        }
    }
}

/// The first call where `got` and `want` differ, with the first instance
/// whose record differs there (`None`: only the outputs differ) and the two
/// call hashes. The call is an index into `got` and `want`.
fn first_divergence(got: &[Call], want: &[Call]) -> Option<(usize, Option<usize>, u64, u64)> {
    let call = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .or_else(|| (got.len() != want.len()).then(|| got.len().min(want.len())))?;
    let (a, b) = (got.get(call), want.get(call));
    let instance = a
        .zip(b)
        .and_then(|(a, b)| a.states.iter().zip(&b.states).position(|(x, y)| x != y));
    Some((
        call,
        instance,
        a.map_or(0, |c| c.hash),
        b.map_or(0, |c| c.hash),
    ))
}

/// `kind`'s report when `got` and `want` differ, with the call numbered as
/// in tests/golden.json.
fn report(kind: Kind, got: &[Call], want: &[Call]) -> Option<String> {
    let (index, instance, a, b) = first_divergence(got, want)?;
    let call = kind.first_call().saturating_add(index);
    let at = instance.map_or_else(
        || "the outputs only".to_owned(),
        |i| format!("instance {i}"),
    );
    Some(format!(
        "{}\nfirst divergence at call {call}, {at} (call hash {a:016x} != {b:016x})",
        kind.describe()
    ))
}

/// Fails the calling test with `kind`'s report when the runs differ.
macro_rules! compare {
    ($kind:expr, $got:expr, $want:expr) => {
        if let Some(report) = report($kind, $got, $want) {
            panic!("{report}");
        }
    };
}

/// The hash of the pool's records and outputs after a call.
fn digest(p: &Pool) -> Result<Call, EngineError> {
    let out = p.outputs()?;
    let mut h = StateHasher::new();
    let mut states = Vec::new();
    for id in 0..INSTANCES {
        let state = p.state_hash(id)?.get();
        states.push(state);
        h.add_bytes(&state.to_le_bytes());
    }
    if let Some(obs) = out.observation(0) {
        h.add_bytes(obs);
    }
    for r in out.rewards() {
        h.add_bytes(&r.to_bits().to_le_bytes());
    }
    let flags = out
        .masks()
        .iter()
        .chain(out.episode_terminated())
        .chain(out.episode_truncated());
    for f in flags {
        h.add_bytes(&[f.to_byte()]);
    }
    Ok(Call {
        hash: h.finish().get(),
        states,
    })
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
fn run(p: &mut Pool, calls: u32) -> Result<Vec<Call>, EngineError> {
    let mut hashes = vec![digest(p)?];
    for t in 1..=calls {
        let acts = actions(p, t)?;
        step(p, &acts)?;
        hashes.push(digest(p)?);
    }
    Ok(hashes)
}

/// Every instance's record.
fn snapshot(p: &Pool) -> Result<Vec<Record<Nim>>, EngineError> {
    (0..INSTANCES).map(|id| p.snapshot(id)).collect()
}

/// `calls` more generated steps: the actions of each, and the hash after.
fn record(p: &mut Pool, from: u32, calls: u32) -> Result<(Vec<Vec<i32>>, Vec<Call>), EngineError> {
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
fn replay(p: &mut Pool, rows: &[Vec<i32>]) -> Result<Vec<Call>, EngineError> {
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
    let calls = run(&mut pool(SEED, default_chunk(), 0).unwrap(), CALLS).unwrap();
    let again = run(&mut pool(SEED, default_chunk(), 0).unwrap(), CALLS).unwrap();
    compare!(Kind::InlineInline, &again, &calls);
    for chunk in [1, default_chunk()] {
        let got = run(&mut pool(SEED, chunk, 4).unwrap(), CALLS).unwrap();
        compare!(Kind::Workers(chunk), &got, &calls);
    }
    let want: Vec<u64> = calls.iter().map(|c| c.hash).collect();
    let path = Path::new(GOLDEN);
    if std::env::var_os("TESSERA_BLESS").is_some_and(|v| v == "1") {
        std::fs::write(path, golden_json(&want)).unwrap();
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        panic!(
            "no tests/golden.json: record it with `cargo tessera check --bless`, then commit it"
        );
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
        let decoded: Vec<DecodedRecord<Nim>> = records
            .iter()
            .map(|r| DecodedRecord::from_parts(*r.header(), *r.state()).unwrap())
            .collect();
        let ids: Vec<u32> = (0..INSTANCES).collect();
        b.import(&ids, &decoded).unwrap();
        compare!(Kind::Resume(at), &replay(&mut b, &rows).unwrap(), &want);
    }
}

#[test]
fn test_snapshot_restore_in_place_replays_equal() {
    let mut p = pool(SEED, default_chunk(), 0).unwrap();
    for t in 1..=SNAPSHOT_AT {
        let acts = actions(&p, t).unwrap();
        step(&mut p, &acts).unwrap();
    }
    let records = snapshot(&p).unwrap();
    let (rows, want) = record(&mut p, SNAPSHOT_AT + 1, 80).unwrap();
    p.restore_all(&records).unwrap();
    compare!(Kind::Snapshot, &replay(&mut p, &rows).unwrap(), &want);
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
        let ended = out
            .episode_terminated()
            .iter()
            .chain(out.episode_truncated());
        episodes += u32::try_from(ended.filter(|f| f.get()).count()).unwrap();
    }
    assert!(episodes > 0, "the run must end episodes");
}

#[test]
fn test_import_of_an_invalid_state_refused() {
    let mut p = pool(SEED, default_chunk(), 0).unwrap();
    let record = p.snapshot(0).unwrap();
    let cfg = NimConfig::default();
    let bad = NimState {
        step: cfg.episode + 1,
        ..*record.state()
    };
    let decoded = DecodedRecord::from_parts(*record.header(), bad).unwrap();
    assert!(p.import(&[0], &[decoded]).is_err());
    assert_eq!(
        p.snapshot(0).unwrap().hash(),
        record.hash(),
        "nothing written"
    );
}

#[test]
fn test_report_numbers_calls_as_in_golden_for_every_kind() {
    let call = |hash: u64| Call {
        hash,
        states: vec![hash, 7],
    };
    let want: Vec<Call> = (0..5).map(call).collect();
    let mut got = want.clone();
    got[2] = Call {
        hash: 99,
        states: vec![2, 8],
    };
    for (kind, absolute) in [
        (Kind::InlineInline, 2),
        (Kind::Workers(1), 2),
        (Kind::Resume(0), 3),
        (Kind::Resume(37), 40),
        (Kind::Snapshot, 26),
    ] {
        let text = report(kind, &got, &want).unwrap();
        assert!(
            text.contains(&format!("first divergence at call {absolute}, instance 1 ")),
            "{kind:?}: {text}"
        );
    }
    assert!(report(Kind::Snapshot, &want, &want).is_none());
}

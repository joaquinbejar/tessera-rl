// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared harness for the environment suites (the classic crate, and the
//! games crate by `#[path]`): a deterministic action
//! generator, per-step output digests and golden fixtures (ADR-0009,
//! ADR-0024 layer 2).
//!
//! Fixture format 3 (`tests/golden/<name>.golden`, plain text; format 1
//! had 15 fields; format 2 named the 16th `episode_invalid_spawns`):
//!
//! - `#` comment lines;
//! - `key=value` header lines: `format`, `env_id`, `env_version`,
//!   `config_hash`, `math_version`, `actions`, `seed`, `num_envs`, `chunk`,
//!   `auto_reset`, `max_episode_steps`, `steps`, `reseed`, `fields`;
//! - one line per call, `<t> <digest>...`, one hex `u64` per field, where
//!   `t = 0` is the initial `reset_all`;
//! - `final <hash>...`: each instance's record hash after the last step.
//!
//! Every digest is XXH3-64 over the canonical little-endian bytes of one
//! output buffer for the whole batch. Fixtures are written only with
//! `TESSERA_BLESS=1`, under the policy of [`bless_decision`] (see
//! `tests/README.md`).

#![allow(
    clippy::unwrap_used,
    reason = "a read after a successful call cannot be poisoned"
)]
#![allow(
    dead_code,
    reason = "each test binary uses a different part of the harness"
)]
#![expect(
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test harness: a failure is the assertion"
)]

pub mod continuation;

use core::num::{NonZeroU32, NonZeroUsize};
use std::io::Write as _;
use std::path::PathBuf;

use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::encoding::{Encoder, HashSink};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AdaptiveValues, AgentSlot, Bool8, Environment, Heuristic, RecordLayout, StateHash,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, MAX_LEAVES, MaskProvider, Outputs, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::math::MATH_VERSION;
use tessera_rl_num::rng::{RngAuthority, StepRng};

/// Fixture format version.
pub const FORMAT: u32 = 3;
/// Identifier of the action generator below; bump it with any change.
pub const ACTIONS_GEN: &str = "mix-heuristic-v1";
/// Digest fields, in fixture order.
pub const FIELDS: [&str; 16] = [
    "records",
    "obs",
    "terminal_obs",
    "rewards",
    "masks",
    "agent_terminated",
    "agent_truncated",
    "episode_terminated",
    "episode_truncated",
    "alive",
    "acting",
    "generations",
    "episode_return",
    "episode_length",
    "episode_illegal",
    "episode_invalid_agent_events",
];

/// One golden scenario.
#[derive(Debug, Clone)]
pub struct Scenario<C> {
    /// Fixture name.
    pub name: &'static str,
    /// Environment config.
    pub cfg: C,
    /// Pool seed.
    pub seed: u64,
    /// Instances.
    pub num_envs: u32,
    /// Chunk size.
    pub chunk: u32,
    /// Auto-reset mode.
    pub auto_reset: AutoReset,
    /// Engine step limit.
    pub max_episode_steps: u32,
    /// Steps after the initial reset.
    pub steps: u32,
    /// At this step, instances `reseed_ids` are reset with `Reseed::Base`.
    pub reseed_at: u32,
    /// Instances reseeded mid-run.
    pub reseed_ids: &'static [u32],
    /// Base seed of the mid-run reseed.
    pub reseed_seed: u64,
}

impl<C> Scenario<C> {
    /// Pool options of this scenario.
    pub fn options(&self) -> PoolOptions {
        PoolOptions {
            chunk_size: NonZeroU32::new(self.chunk).unwrap_or(NonZeroU32::MIN),
            auto_reset: self.auto_reset,
            max_episode_steps: NonZeroU32::new(self.max_episode_steps),
            ..PoolOptions::new(
                NonZeroU32::new(self.num_envs).unwrap_or(NonZeroU32::MIN),
                Seed::new(self.seed),
            )
        }
    }
}

/// An environment an `EnvPool` can run (the pool's bounds, implied).
pub trait Pooled:
    Environment<
        Config: Sync,
        State: Send,
        Header: Send + RecordLayout<Adaptive: AdaptiveValues<Self::Config>>,
    >
{
}

impl<E> Pooled for E where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >
{
}

/// The environments under golden test.
pub trait GoldenEnv: Pooled + Heuristic
where
    Self::Config: CanonicalConfig,
{
    /// Versioned id.
    const ENV_ID: &'static str;
    /// Version.
    const ENV_VERSION: u32;
}

/// splitmix64 (the generator's mixer; independent of the engine RNG).
#[must_use]
pub const fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The engine-independent RNG handed to `Heuristic::act` (the baselines of
/// these environments draw nothing).
fn heuristic_rng() -> StepRng {
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

/// Choices of the single discrete head.
pub fn choices<E: Environment>(cfg: &E::Config) -> u32 {
    match E::action_space(cfg) {
        Ok(Space::Discrete { n }) => n,
        _ => 1,
    }
}

/// Actions `mix-heuristic-v1` for step `t`: per instance and slot, one in
/// four is the scripted baseline, the rest uniform; a choice the mask
/// forbids becomes the baseline.
#[expect(clippy::many_single_char_names, reason = "index arithmetic")]
pub fn actions<E, M>(p: &EnvPool<E, M>, cfg: &E::Config, seed: u64, t: u32) -> Vec<i32>
where
    E: Pooled + Heuristic,
    M: MaskProvider<E>,
{
    let a = E::MAX_AGENTS;
    let n = choices::<E>(cfg);
    let masks = p.outputs().unwrap().masks();
    let instances = p.outputs().unwrap().rewards().len() / a;
    let mut out = vec![0; instances * a];
    let mut rng = heuristic_rng();
    for i in 0..instances {
        let Some(state) = u32::try_from(i).ok().and_then(|id| p.state(id)) else {
            continue;
        };
        for slot in 0..a {
            let row = i * a + slot;
            let key = seed ^ (u64::from(t) << 32) ^ ((row as u64) << 4);
            let x = mix(key);
            let baseline = || {
                let mut discrete = [0];
                let mut writer = ActionWriter::new(&mut discrete, &mut []);
                if let Some(agent) = u16::try_from(slot).ok().and_then(|s| AgentSlot::new(s, a)) {
                    E::act(cfg, state, agent, &mut heuristic_rng(), &mut writer);
                }
                discrete[0]
            };
            let mut choice = if x.is_multiple_of(4) {
                baseline()
            } else {
                i32::try_from((x >> 8) % u64::from(n)).unwrap_or(0)
            };
            let width = usize::try_from(n).unwrap_or(1);
            let legal = masks.is_empty()
                || usize::try_from(choice)
                    .ok()
                    .and_then(|c| masks.get(row * width + c))
                    .is_some_and(|m| m.get());
            if !legal {
                choice = baseline();
            }
            if let Some(slot) = out.get_mut(row) {
                *slot = choice;
            }
        }
    }
    let _ = &mut rng;
    out
}

/// XXH3-64 of `bytes`.
#[must_use]
pub fn digest(bytes: &[u8]) -> u64 {
    let mut out = Encoder::new(HashSink::new());
    out.write_bytes(bytes);
    out.sink().digest()
}

fn flags(v: &[Bool8]) -> Vec<u8> {
    v.iter().map(|f| f.to_byte()).collect()
}

fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_bits().to_le_bytes()).collect()
}

/// The per-field digests of the current outputs and records.
pub fn record_digests<E, M>(p: &EnvPool<E, M>, same_step: bool) -> Vec<u64>
where
    E: Pooled,
    M: MaskProvider<E>,
{
    let out: &Outputs = p.outputs().unwrap();
    let a = E::MAX_AGENTS;
    let instances = out.rewards().len() / a;
    let mut records = Vec::new();
    for id in 0..u32::try_from(instances).unwrap_or(0) {
        if let (Some(h), Some(s)) = (p.header(id), p.state(id)) {
            let mut hasher = tessera_rl_core::encoding::StateHasher::new();
            hasher.add(h);
            hasher.add(s);
            records.extend(hasher.finish().get().to_le_bytes());
        }
    }
    let mut obs = Vec::new();
    let mut terminal = Vec::new();
    for leaf in 0..MAX_LEAVES {
        let Some(bytes) = out.observation(leaf) else {
            break;
        };
        obs.extend_from_slice(bytes);
        // Terminal rows are stale unless the episode ended this call.
        if same_step && let Some(t) = out.terminal_observation(leaf) {
            let per = t.len() / instances.max(1);
            for (i, ended) in out.episode_terminated().iter().enumerate() {
                let truncated = out.episode_truncated().get(i).is_some_and(|f| f.get());
                if (ended.get() || truncated)
                    && let Some(rows) = t.get(i * per..(i + 1) * per)
                {
                    terminal.extend_from_slice(rows);
                }
            }
        }
    }
    let generations: Vec<u8> = out
        .generations()
        .iter()
        .flat_map(|g| g.to_le_bytes())
        .collect();
    vec![
        digest(&records),
        digest(&obs),
        digest(&terminal),
        digest(&floats(out.rewards())),
        digest(&flags(out.masks())),
        digest(&flags(out.agent_terminated())),
        digest(&flags(out.agent_truncated())),
        digest(&flags(out.episode_terminated())),
        digest(&flags(out.episode_truncated())),
        digest(&flags(out.alive())),
        digest(&flags(out.acting())),
        digest(&generations),
        digest(&floats(out.episode_return())),
        digest(&floats(out.episode_length())),
        digest(&floats(out.episode_illegal_actions())),
        digest(&floats(out.episode_invalid_agent_events())),
    ]
}

/// Each instance's record hash.
pub fn record_hashes<E, M>(p: &EnvPool<E, M>) -> Vec<u64>
where
    E: Pooled,
    M: MaskProvider<E>,
{
    let instances = p.outputs().unwrap().rewards().len() / E::MAX_AGENTS;
    (0..u32::try_from(instances).unwrap_or(0))
        .filter_map(|id| {
            let (h, s) = (p.header(id)?, p.state(id)?);
            let mut hasher = tessera_rl_core::encoding::StateHasher::new();
            hasher.add(h);
            hasher.add(s);
            Some(hasher.finish().get())
        })
        .collect()
}

/// Runs `scenario` on a pool from `build` (optionally with `workers`) and
/// returns the fixture lines.
pub fn run<E, M>(
    scenario: &Scenario<E::Config>,
    build: impl Fn(E::Config, PoolOptions) -> EnvPool<E, M>,
    workers: Option<usize>,
) -> Vec<String>
where
    E: GoldenEnv,
    E::Config: CanonicalConfig + Clone,
    M: MaskProvider<E>,
{
    let mut p = build(scenario.cfg.clone(), scenario.options());
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(
                NonZeroUsize::new(w).unwrap_or(NonZeroUsize::MIN),
            ))
            .unwrap_or_else(|e| panic!("workers: {e}"));
    }
    let same_step = scenario.auto_reset == AutoReset::SameStep;
    let mut dump = action_dump(scenario.name, workers);
    let mut lines = header::<E>(scenario);
    p.reset_all(Reseed::Keep)
        .unwrap_or_else(|e| panic!("reset: {e}"));
    lines.push(step_line(0, &record_digests(&p, same_step)));
    for t in 1..=scenario.steps {
        if t == scenario.reseed_at {
            p.reset(
                scenario.reseed_ids,
                Reseed::Base(Seed::new(scenario.reseed_seed)),
            )
            .unwrap_or_else(|e| panic!("reseed: {e}"));
        }
        let acts = actions(&p, &scenario.cfg, scenario.seed, t);
        if let Some(file) = dump.as_mut() {
            let bytes: Vec<u8> = acts.iter().flat_map(|a| a.to_le_bytes()).collect();
            file.write_all(&bytes)
                .unwrap_or_else(|e| panic!("action dump: {e}"));
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap_or_else(|e| panic!("step {t}: {e}"));
        lines.push(step_line(t, &record_digests(&p, same_step)));
    }
    let finals: Vec<String> = record_hashes(&p)
        .iter()
        .map(|h| format!("{h:016x}"))
        .collect();
    lines.push(format!("final {}", finals.join(" ")));
    lines
}

/// With `TESSERA_GOLDEN_ACTIONS=<dir>`, the Inline run of a scenario writes
/// its actions (little-endian `i32`, every step in order) to
/// `<dir>/<name>.actions`, for `tools/wheel_golden.py` to replay through an
/// installed wheel. The run still checks the committed fixture, so the file
/// holds the actions that produce it. The actions depend on environment
/// state through the scripted baseline, which Python cannot reach.
fn action_dump(name: &str, workers: Option<usize>) -> Option<std::fs::File> {
    let dir = PathBuf::from(std::env::var_os("TESSERA_GOLDEN_ACTIONS")?);
    if workers.is_some() {
        return None;
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("action dump: {e}"));
    Some(
        std::fs::File::create(dir.join(format!("{name}.actions")))
            .unwrap_or_else(|e| panic!("action dump: {e}")),
    )
}

fn step_line(t: u32, digests: &[u64]) -> String {
    let hex: Vec<String> = digests.iter().map(|d| format!("{d:016x}")).collect();
    format!("{t} {}", hex.join(" "))
}

fn header<E>(s: &Scenario<E::Config>) -> Vec<String>
where
    E: GoldenEnv,
    E::Config: CanonicalConfig,
{
    let hash = config_hash(E::ENV_ID, E::ENV_VERSION, &s.cfg).unwrap_or(0);
    vec![
        "# SPDX-License-Identifier: MIT OR Apache-2.0".to_owned(),
        format!("# golden fixture {}; see tests/support/mod.rs", s.name),
        format!("format={FORMAT}"),
        format!("env_id={}", E::ENV_ID),
        format!("env_version={}", E::ENV_VERSION),
        format!("config_hash={hash:016x}"),
        format!("math_version={MATH_VERSION}"),
        format!("actions={ACTIONS_GEN}"),
        format!("seed={}", s.seed),
        format!("num_envs={}", s.num_envs),
        format!("chunk={}", s.chunk),
        format!("auto_reset={:?}", s.auto_reset),
        format!("max_episode_steps={}", s.max_episode_steps),
        format!("steps={}", s.steps),
        format!(
            "reseed={}:{:?}:{}",
            s.reseed_at, s.reseed_ids, s.reseed_seed
        ),
        format!("fields={}", FIELDS.join(",")),
    ]
}

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(format!("{name}.golden"))
}

fn value<'a>(lines: &'a [&'a str], key: &str) -> Option<&'a str> {
    lines
        .iter()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
}

/// Why a harness migration may re-bless a fixture (`TESSERA_BLESS_REASON`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlessReason {
    /// The fixture format version changed.
    Format,
    /// The action generator changed.
    Generator,
}

impl BlessReason {
    /// Parses `TESSERA_BLESS_REASON`.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        match std::env::var("TESSERA_BLESS_REASON").ok()?.as_str() {
            "format" => Some(Self::Format),
            "generator" => Some(Self::Generator),
            _ => None,
        }
    }
}

/// What blessing does with a fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bless {
    /// The output did not change: the file is left untouched.
    Unchanged,
    /// The fixture is new, or the change is authorized: write it.
    Write,
}

/// The bless policy (ADR-0008, ADR-0023), pure. A changed fixture needs
/// an `env_version` bump; a `MATH_VERSION` bump alone does not authorize it
/// (new math is a new environment version that records it). A format or
/// generator change is a harness migration and needs the matching
/// `reason`.
///
/// # Errors
///
/// Returns why the change is refused.
pub fn bless_decision(
    old: Option<&str>,
    new: &str,
    reason: Option<BlessReason>,
) -> Result<Bless, String> {
    let Some(old) = old else {
        return Ok(Bless::Write);
    };
    if old == new {
        return Ok(Bless::Unchanged);
    }
    let (o, n): (Vec<&str>, Vec<&str>) = (old.lines().collect(), new.lines().collect());
    let changed = |key| value(&o, key) != value(&n, key);
    if changed("env_version") {
        return Ok(Bless::Write);
    }
    match reason {
        Some(BlessReason::Format) if changed("format") => Ok(Bless::Write),
        Some(BlessReason::Generator) if changed("actions") => Ok(Bless::Write),
        _ => Err(
            "a changed fixture needs an env_version bump (ADR-0023); MATH_VERSION alone \
                  does not authorize it, and a format or generator migration needs \
                  TESSERA_BLESS_REASON=format|generator"
                .to_owned(),
        ),
    }
}

/// Blesses `lines` into `file` under the policy; returns what it did.
///
/// # Errors
///
/// Returns why the change is refused; the file is then left intact.
pub fn bless_into(
    file: &std::path::Path,
    lines: &[String],
    reason: Option<BlessReason>,
) -> Result<Bless, String> {
    let fresh = lines.join("\n") + "\n";
    let old = std::fs::read_to_string(file).ok();
    let decision = bless_decision(old.as_deref(), &fresh, reason)?;
    if decision == Bless::Write {
        std::fs::write(file, &fresh).map_err(|e| format!("{}: {e}", file.display()))?;
    }
    Ok(decision)
}

/// Compares `lines` with fixture text `committed`, pure.
///
/// # Errors
///
/// The first diverging call and field, and the instances whose final
/// record differs; or a header value that differs from the code.
pub fn compare_text(name: &str, committed: &str, lines: &[String]) -> Result<(), String> {
    let old: Vec<&str> = committed.lines().collect();
    let new: Vec<&str> = lines.iter().map(String::as_str).collect();
    for key in [
        "format",
        "env_id",
        "env_version",
        "math_version",
        "actions",
        "fields",
    ] {
        if value(&old, key) != value(&new, key) {
            return Err(format!(
                "{name}: fixture {key} differs from the code; a behaviour change needs a version bump"
            ));
        }
    }
    let Some(k) = old.iter().zip(&new).position(|(a, b)| a != b) else {
        if old.len() != new.len() {
            return Err(format!("{name}: fixture length differs"));
        }
        return Ok(());
    };
    let (a, b) = (old[k], new[k]);
    let fields: Vec<&str> = a
        .split(' ')
        .zip(b.split(' '))
        .enumerate()
        .skip(1)
        .filter(|(_, (x, y))| x != y)
        .filter_map(|(i, _)| FIELDS.get(i - 1).copied())
        .collect();
    let finals = |l: &[&str]| -> Vec<String> {
        l.iter()
            .find_map(|x| x.strip_prefix("final "))
            .map(|x| x.split(' ').map(str::to_owned).collect())
            .unwrap_or_default()
    };
    let instances: Vec<usize> = finals(&old)
        .iter()
        .zip(finals(&new))
        .enumerate()
        .filter(|(_, (x, y))| *x != y)
        .map(|(i, _)| i)
        .collect();
    Err(format!(
        "{name}: first divergence at line {k} (call {}), fields {fields:?}; \
         instances with a different final record: {instances:?}\n  fixture: {a}\n  code:    {b}",
        a.split(' ').next().unwrap_or("?")
    ))
}

/// Compares `lines` with the committed fixture `name` (read-only), or
/// blesses it under `TESSERA_BLESS=1` and the bless policy.
///
/// # Panics
///
/// On any difference, or a refused bless.
pub fn check(name: &str, lines: &[String]) {
    let file = path(name);
    if std::env::var_os("TESSERA_BLESS").is_some_and(|v| v == "1") {
        if let Err(e) = bless_into(&file, lines, BlessReason::from_env()) {
            panic!("{name}: {e}");
        }
        return;
    }
    let Ok(committed) = std::fs::read_to_string(&file) else {
        panic!(
            "{name}: missing fixture {}; run with TESSERA_BLESS=1",
            file.display()
        );
    };
    if let Err(e) = compare_text(name, &committed, lines) {
        panic!("{e}");
    }
}

/// The committed fixture `name`.
#[must_use]
pub fn committed(name: &str) -> String {
    std::fs::read_to_string(path(name)).unwrap_or_default()
}

/// `StateHash` of every instance (for tests that compare pools).
pub fn state_hashes<E, M>(p: &EnvPool<E, M>, n: u32) -> Vec<u64>
where
    E: Pooled,
    M: MaskProvider<E>,
{
    (0..n)
        .filter_map(|i| p.state(i).map(|s| StateHash::of(s).get()))
        .collect()
}

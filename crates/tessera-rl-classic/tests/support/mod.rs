// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared harness for the classic suites: a deterministic action
//! generator, per-step output digests and golden fixtures (ADR-0009,
//! ADR-0024 layer 2).
//!
//! Fixture format 1 (`tests/golden/<name>.golden`, plain text):
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
//! `TESSERA_BLESS=1`, and blessing refuses to change an existing fixture
//! unless `env_version` or `math_version` changed (ADR-0023).

#![allow(
    dead_code,
    reason = "each test binary uses a different part of the harness"
)]
#![expect(
    clippy::panic,
    clippy::indexing_slicing,
    reason = "test harness: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};
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
pub const FORMAT: u32 = 1;
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
    "episode_invalid_spawns",
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

impl GoldenEnv for tessera_rl_classic::CartPole {
    const ENV_ID: &'static str = tessera_rl_classic::cartpole::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_classic::cartpole::ENV_VERSION;
}

impl GoldenEnv for tessera_rl_classic::Tag {
    const ENV_ID: &'static str = tessera_rl_classic::tag::ENV_ID;
    const ENV_VERSION: u32 = tessera_rl_classic::tag::ENV_VERSION;
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
    let masks = p.outputs().masks();
    let instances = p.outputs().rewards().len() / a;
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
    let out: &Outputs = p.outputs();
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
        digest(&floats(out.episode_invalid_spawns())),
    ]
}

/// Each instance's record hash.
pub fn record_hashes<E, M>(p: &EnvPool<E, M>) -> Vec<u64>
where
    E: Pooled,
    M: MaskProvider<E>,
{
    let instances = p.outputs().rewards().len() / E::MAX_AGENTS;
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

/// Compares `lines` with the committed fixture `name`, or writes it under
/// `TESSERA_BLESS=1`.
///
/// # Panics
///
/// On any difference: the first diverging call and field, and the
/// instances whose final record differs.
pub fn check(name: &str, lines: &[String]) {
    let bless = std::env::var_os("TESSERA_BLESS").is_some_and(|v| v == "1");
    check_mode(name, lines, bless);
}

/// [`check`] that never writes, whatever `TESSERA_BLESS` says.
pub fn compare(name: &str, lines: &[String]) {
    check_mode(name, lines, false);
}

fn check_mode(name: &str, lines: &[String], bless: bool) {
    let file = path(name);
    let fresh = lines.join("\n") + "\n";
    let committed = std::fs::read_to_string(&file).ok();
    if bless {
        if let Some(old) = &committed
            && *old != fresh
        {
            let old_lines: Vec<&str> = old.lines().collect();
            let new_lines: Vec<&str> = fresh.lines().collect();
            let same_versions = ["env_version", "math_version", "actions", "format"]
                .iter()
                .all(|k| value(&old_lines, k) == value(&new_lines, k));
            assert!(
                !same_versions,
                "{name}: refusing to bless a changed fixture without bumping env_version \
                 (ADR-0023), MATH_VERSION, the action generator or the format"
            );
        }
        std::fs::write(&file, &fresh).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        return;
    }
    let Some(committed) = committed else {
        panic!(
            "{name}: missing fixture {}; run with TESSERA_BLESS=1",
            file.display()
        );
    };
    let old: Vec<&str> = committed.lines().collect();
    let new: Vec<&str> = lines.iter().map(String::as_str).collect();
    for key in ["format", "env_id", "env_version", "math_version", "actions"] {
        assert_eq!(
            value(&old, key),
            value(&new, key),
            "{name}: fixture {key} differs from the code; a behaviour change needs a version bump"
        );
    }
    let first = old.iter().zip(&new).position(|(a, b)| a != b);
    let Some(k) = first else {
        assert_eq!(old.len(), new.len(), "{name}: fixture length differs");
        return;
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
    panic!(
        "{name}: first divergence at line {k} (call {}), fields {fields:?}; \
         instances with a different final record: {instances:?}\n  fixture: {a}\n  code:    {b}",
        a.split(' ').next().unwrap_or("?")
    );
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

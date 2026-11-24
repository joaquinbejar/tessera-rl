// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-sample benchmark runner for the ADR-0025 harness (`tools/bench`).
//! It measures; the harness computes statistics and adds the metadata.
//! Every command writes one JSON object to stdout, and any invalid input
//! or engine error fails the run (a benchmark never falls back silently).
//!
//! - `throughput <env> <num_envs> <workers> <warmup_s> <reps> <rep_s>`:
//!   steps/s per repetition, pregenerated seeded random actions, auto-reset
//!   and observation writing included (SPEC 19.1).
//! - `latency <env> <num_envs> <workers> <warmup_s> <reps> <batches>`: p50
//!   and p99 of the batch step time per repetition, in nanoseconds.
//! - `snapshot <env> <num_envs> <warmup_s> <reps>`: per-instance
//!   nanoseconds, after warming the measured operations up. Copy
//!   only: `snapshot` (one record at a time) and `snapshot_all` (into a
//!   preallocated buffer). Including re-observation (ADR-0039):
//!   `restore`, `import` (plus validation), `fork` and `fork_branched`.
//! - `memory <env> <num_envs>`: continuation bytes per instance, mask
//!   bytes written per step and output storage bytes.
//!
//! `env` is one of the catalog environments named in [`run`] (every one
//! has a benchmark entry, SPEC 16.1); `workers` 0 is Inline mode. The plan
//! draws each discrete head in `0..choices` and each continuous head
//! uniformly within its bounds. Options:
//! `--legal-actions` replaces each planned discrete action by the first
//! legal choice at or after it (cyclically) under the current mask, inside
//! the timed step, so masked games step real moves rather than the no-op
//! of the illegal-action policy; it applies to masked pools whose mask is
//! one flat entry per choice of each head (not RTS, whose heads are
//! conditioned on a parent head: RTS steps under the no-op policy).
//! `--cpus a,b,...` pins the Sync-mode workers to those logical CPUs (pin
//! the process itself with `taskset` on Linux); `--max-episode-steps N`
//! sets the truncation limit (0, the default, is none). The output records
//! the effective method and pool options, the FNV-1a fingerprint of the
//! action plan (the same plan the Python harness generates), and the
//! process CPU affinity where the OS exposes it.

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tessera_rl_classic::{
    Acrobot, AcrobotConfig, BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Grid, GridConfig,
    KeyDoor, KeyDoorConfig, MountainCar, MountainCarConfig, PaddleDuel, PaddleDuelConfig, Pendulum,
    PendulumConfig, Snake, SnakeConfig, Tag, TagConfig,
};
use tessera_rl_core::{Bool8, Environment};
use tessera_rl_engine::{
    ActionInput, DecodedRecord, EngineError, EnvPool, MaskProvider, Masked, NoMasks, PoolOptions,
    Record, Reseed, SyncOptions,
};
use tessera_rl_games::{
    Connect4, Connect4Config, Go9, Go9Config, Leduc, LeducConfig, Othello, OthelloConfig,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::{Rts, RtsConfig};

/// Batches of pregenerated actions cycled through a measurement.
const ACTION_BATCHES: usize = 64;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("usage: {0}")]
    Usage(String),
    #[error("engine: {0}")]
    Engine(#[from] EngineError),
}

fn usage(msg: &str) -> Error {
    Error::Usage(msg.to_owned())
}

fn arg<T: core::str::FromStr>(args: &[String], i: usize, name: &str) -> Result<T, Error> {
    args.get(i)
        .and_then(|a| a.parse().ok())
        .ok_or_else(|| usage(&format!("missing or invalid <{name}>")))
}

/// The value after `flag`, if given.
fn option<T: core::str::FromStr>(args: &[String], flag: &str) -> Result<Option<T>, Error> {
    let Some(pos) = args.iter().position(|a| a == flag) else {
        return Ok(None);
    };
    args.get(pos + 1)
        .and_then(|v| v.parse().ok())
        .map(Some)
        .ok_or_else(|| usage(&format!("{flag} needs a value")))
}

/// FNV-1a over the little-endian bytes of the plan, batch by batch: the
/// discrete actions, then the continuous ones (none for discrete-only
/// spaces, so their fingerprint is the Python harness's).
fn fingerprint(batches: &[Vec<i32>], continuous: &[Vec<f32>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let discrete = batches.iter().flatten().flat_map(|v| v.to_le_bytes());
    let continuous = continuous.iter().flatten().flat_map(|v| v.to_le_bytes());
    for byte in discrete.chain(continuous) {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The `--cpus a,b,...` list, if given.
fn cpus(args: &[String]) -> Result<Option<Vec<usize>>, Error> {
    let Some(pos) = args.iter().position(|a| a == "--cpus") else {
        return Ok(None);
    };
    let list = args
        .get(pos + 1)
        .ok_or_else(|| usage("--cpus needs a list"))?;
    list.split(',')
        .map(|c| c.parse().map_err(|_| usage("--cpus takes CPU numbers")))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// One step of the plan's 64-bit LCG: the new state and its 31-bit draw.
fn lcg(state: u64) -> (u64, u32) {
    let next = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    // The top 31 bits always fit.
    (next, u32::try_from(next >> 33).unwrap_or(0))
}

/// Seeded discrete actions, `rows` rows of one value per head, head `h` in
/// `0..heads[h]` (a 64-bit LCG; the values only need to be reproducible and
/// spread). With one head this is the plan of the Python harness.
fn actions(rows: usize, heads: &[u32], seed: u64) -> Result<Vec<i32>, Error> {
    if heads.contains(&0) {
        return Err(usage("an action head without choices"));
    }
    let len = rows
        .checked_mul(heads.len())
        .ok_or_else(|| usage("action plan size"))?;
    let mut state = seed;
    (0..len)
        .map(|i| {
            let (next, draw) = lcg(state);
            state = next;
            let choices = heads
                .get(i % heads.len())
                .ok_or_else(|| usage("action head"))?;
            i32::try_from(draw % choices).map_err(|_| usage("action out of i32"))
        })
        .collect()
}

/// Seeded continuous actions, `rows` rows of one value per head, uniform
/// within each head's inclusive bounds (same LCG, its own stream).
#[expect(clippy::cast_possible_truncation, reason = "f32 is the action dtype")]
fn continuous(rows: usize, bounds: &[(f64, f64)], seed: u64) -> Result<Vec<f32>, Error> {
    let len = rows
        .checked_mul(bounds.len())
        .ok_or_else(|| usage("action plan size"))?;
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..len)
        .map(|i| {
            let (next, draw) = lcg(state);
            state = next;
            let &(low, high) = bounds
                .get(i % bounds.len())
                .ok_or_else(|| usage("action head"))?;
            let unit = f64::from(draw) / f64::from(1u32 << 31);
            Ok((low + unit * (high - low)) as f32)
        })
        .collect()
}

/// Writes into `out` each planned action of `plan`, moved to the first
/// legal choice at or after it (cyclically) under `masks`, laid out
/// `[rows, sum of heads]`. A head with no legal choice keeps its planned
/// action (the illegal-action policy applies).
fn legalize(out: &mut [i32], plan: &[i32], masks: &[Bool8], heads: &[u32]) {
    let width: usize = heads.iter().map(|&n| usize::try_from(n).unwrap_or(0)).sum();
    let rows = out
        .chunks_exact_mut(heads.len())
        .zip(plan.chunks_exact(heads.len()))
        .zip(masks.chunks_exact(width));
    for ((row_out, row_plan), row_mask) in rows {
        let mut offset = 0usize;
        for ((slot, &planned), &n) in row_out.iter_mut().zip(row_plan).zip(heads) {
            let n = usize::try_from(n).unwrap_or(0);
            let head = row_mask.get(offset..offset + n).unwrap_or(&[]);
            let start = usize::try_from(planned).unwrap_or(0);
            *slot = (0..n)
                .map(|d| (start + d) % n)
                .find(|&c| head.get(c).is_some_and(|b| b.get()))
                .and_then(|c| i32::try_from(c).ok())
                .unwrap_or(planned);
            offset += n;
        }
    }
}

fn pool<E, M>(
    make: impl FnOnce(PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    num_envs: u32,
    workers: usize,
    cpu_set: Option<Vec<usize>>,
    max_episode_steps: u32,
) -> Result<EnvPool<E, M>, Error>
where
    E: Environment + tessera_rl_core::Validate,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    let n = NonZeroU32::new(num_envs).ok_or_else(|| usage("num_envs must be at least 1"))?;
    let mut options = PoolOptions::new(n, Seed::new(1));
    options.max_episode_steps = NonZeroU32::new(max_episode_steps);
    let mut p = make(options)?;
    if let Some(w) = NonZeroUsize::new(workers) {
        let mut sync = SyncOptions::new(w);
        if let Some(set) = cpu_set {
            if set.len() < workers {
                return Err(usage("--cpus needs one CPU per worker"));
            }
            sync.cpu_set = Some(set);
        }
        p = p.with_workers(sync)?;
    } else if cpu_set.is_some() {
        return Err(usage(
            "--cpus applies to Sync mode; pin Inline runs with taskset",
        ));
    }
    p.reset_all(Reseed::Keep)?;
    Ok(p)
}

/// One environment's pool with its action batches.
struct Bench<E: Environment, M: MaskProvider<E>> {
    pool: EnvPool<E, M>,
    batches: Vec<Vec<i32>>,
    continuous: Vec<Vec<f32>>,
    heads: Vec<u32>,
    /// The legalized discrete actions of the current step (`--legal-actions`).
    legal: Option<Vec<i32>>,
    num_envs: u32,
}

impl<E, M> Bench<E, M>
where
    E: Environment + tessera_rl_core::Validate,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    fn new(
        pool: EnvPool<E, M>,
        cfg: &E::Config,
        num_envs: u32,
        legal: bool,
    ) -> Result<Self, Error> {
        let space = E::action_space(cfg).map_err(|_| usage("action space"))?;
        let (heads, bounds) = space.heads().map_err(|e| usage(e.what()))?.into_parts();
        if pool.action_heads() != (heads.len(), bounds.len()) {
            return Err(usage("the pool's action heads differ from its space"));
        }
        let out = pool.outputs().map_err(Error::from)?;
        let rows = out.rewards().len();
        let mask_len = out.masks().len();
        let seed = |b: usize| u64::try_from(b).map_err(|_| usage("batch"));
        let batches = (0..ACTION_BATCHES)
            .map(|b| actions(rows, &heads, seed(b)?))
            .collect::<Result<_, _>>()?;
        let continuous = (0..ACTION_BATCHES)
            .map(|b| continuous(rows, &bounds, seed(b)?))
            .collect::<Result<_, _>>()?;
        let legal = if legal {
            let width: u64 = heads.iter().map(|&n| u64::from(n)).sum();
            let expected = u64::try_from(rows).ok().and_then(|r| r.checked_mul(width));
            if mask_len == 0 || expected != u64::try_from(mask_len).ok() {
                return Err(usage(
                    "--legal-actions needs a flat mask, one entry per choice of each head",
                ));
            }
            Some(vec![0; rows * heads.len()])
        } else {
            None
        };
        Ok(Self {
            pool,
            batches,
            continuous,
            heads,
            legal,
            num_envs,
        })
    }

    fn step(&mut self, k: usize) -> Result<(), Error> {
        let planned = self
            .batches
            .get(k % ACTION_BATCHES)
            .ok_or_else(|| usage("action batch"))?;
        let continuous = self
            .continuous
            .get(k % ACTION_BATCHES)
            .ok_or_else(|| usage("action batch"))?;
        let discrete = match self.legal.as_mut() {
            Some(legal) => {
                legalize(legal, planned, self.pool.outputs()?.masks(), &self.heads);
                legal
            }
            None => planned,
        };
        let out = self.pool.step(ActionInput {
            discrete,
            continuous,
        })?;
        black_box(out.rewards());
        Ok(())
    }

    fn run_for(&mut self, time: Duration) -> Result<(u64, Duration), Error> {
        let start = Instant::now();
        let mut k = 0usize;
        while start.elapsed() < time {
            self.step(k)?;
            k += 1;
        }
        let batches = u64::try_from(k).map_err(|_| usage("batch count"))?;
        Ok((batches, start.elapsed()))
    }

    fn throughput(&mut self, warmup: f64, reps: usize, rep: f64) -> Result<Vec<f64>, Error> {
        self.run_for(Duration::from_secs_f64(warmup))?;
        (0..reps)
            .map(|_| {
                let (batches, took) = self.run_for(Duration::from_secs_f64(rep))?;
                let steps = batches
                    .checked_mul(u64::from(self.num_envs))
                    .ok_or_else(|| usage("step count overflow"))?;
                Ok(rate(steps) / took.as_secs_f64())
            })
            .collect()
    }

    fn latency(
        &mut self,
        warmup: f64,
        reps: usize,
        batches: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
        if batches == 0 {
            return Err(usage("latency needs at least one batch"));
        }
        self.run_for(Duration::from_secs_f64(warmup))?;
        let mut p50 = Vec::new();
        let mut p99 = Vec::new();
        let mut times = Vec::with_capacity(batches);
        for _ in 0..reps {
            times.clear();
            for k in 0..batches {
                let start = Instant::now();
                self.step(k)?;
                times.push(start.elapsed().as_secs_f64() * 1e9);
            }
            times.sort_by(f64::total_cmp);
            p50.push(quantile(&times, 0.50)?);
            p99.push(quantile(&times, 0.99)?);
        }
        Ok((p50, p99))
    }

    /// Per-instance costs, after `warmup` seconds of the same operations
    /// on the same preallocated buffers; returns the samples and the
    /// number of warm-up iterations.
    fn snapshot(&mut self, warmup: f64, reps: usize) -> Result<([Vec<f64>; 6], u64), Error> {
        let n = self.num_envs;
        if n < 2 {
            return Err(usage("snapshot costs need at least two instances"));
        }
        let per = f64::from(n);
        let dst_count = f64::from(n - 1);
        let ids: Vec<u32> = (0..n).collect();
        let others: Vec<u32> = (1..n).collect();
        let mut out: [Vec<f64>; 6] = Default::default();
        let mut records: Vec<Record<E>> = Vec::with_capacity(ids.len());
        let mut buffer: Vec<Record<E>> = Vec::with_capacity(ids.len());
        let warm = Duration::from_secs_f64(warmup);
        let began = Instant::now();
        let mut warmups = 0u64;
        let mut scratch: [Vec<f64>; 6] = Default::default();
        let total = reps;
        let mut done = 0usize;
        loop {
            let warming = began.elapsed() < warm;
            if !warming && done == total {
                break;
            }
            let out = if warming { &mut scratch } else { &mut out };
            self.step(0)?;
            records.clear();
            let start = Instant::now();
            for &id in &ids {
                records.push(self.pool.snapshot(id)?);
            }
            push(&mut out[0], start.elapsed(), per);
            buffer.clear();
            buffer.extend_from_slice(&records);
            let start = Instant::now();
            self.pool.snapshot_all(&mut buffer)?;
            push(&mut out[1], start.elapsed(), per);
            let start = Instant::now();
            for (&id, record) in ids.iter().zip(&records) {
                self.pool.restore(id, record)?;
            }
            push(&mut out[2], start.elapsed(), per);
            let decoded: Vec<DecodedRecord<E>> =
                records.iter().map(|r| DecodedRecord::from(*r)).collect();
            let start = Instant::now();
            self.pool.import(&ids, &decoded)?;
            push(&mut out[3], start.elapsed(), per);
            let start = Instant::now();
            self.pool.fork(0, &others)?;
            push(&mut out[4], start.elapsed(), dst_count);
            let start = Instant::now();
            self.pool.fork_branched(0, &others, None)?;
            push(&mut out[5], start.elapsed(), dst_count);
            if warming {
                warmups += 1;
                scratch.iter_mut().for_each(Vec::clear);
            } else {
                done += 1;
            }
        }
        Ok((out, warmups))
    }

    fn memory(&self) -> Result<(usize, usize, usize), Error> {
        let out = self.pool.outputs().map_err(Error::from)?;
        Ok((
            Record::<E>::record_size(),
            out.masks().len(),
            out.storage().bytes(),
        ))
    }
}

#[expect(clippy::cast_precision_loss, reason = "a step count as a rate")]
const fn rate(steps: u64) -> f64 {
    steps as f64
}

fn push(samples: &mut Vec<f64>, took: Duration, per: f64) {
    samples.push(took.as_secs_f64() * 1e9 / per);
}

/// Nearest-rank quantile of sorted data.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a rank from a fraction of the sample size"
)]
fn quantile(sorted: &[f64], q: f64) -> Result<f64, Error> {
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted
        .get(rank.saturating_sub(1))
        .copied()
        .ok_or_else(|| usage("quantile of no samples"))
}

fn list(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.3}")).collect();
    format!("[{}]", items.join(", "))
}

/// The process CPU affinity as the OS reports it.
fn affinity() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
                .map(|v| v.trim().to_owned())
        })
        .unwrap_or_else(|| "not exposed by this OS".to_owned())
}

fn run_env<E, M>(args: &[String], mut bench: Bench<E, M>) -> Result<String, Error>
where
    E: Environment + tessera_rl_core::Validate,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    let cmd = args.get(1).map_or("", String::as_str);
    let body = match cmd {
        "throughput" => {
            let (warmup, reps, rep): (f64, usize, f64) = (
                arg(args, 5, "warmup_s")?,
                arg(args, 6, "reps")?,
                arg(args, 7, "rep_s")?,
            );
            let s = bench.throughput(warmup, reps, rep)?;
            format!(
                r#""metric": "throughput", "unit": "steps/s", "method": {{"warmup_s": {warmup}, "reps": {reps}, "rep_s": {rep}}}, "samples": {}"#,
                list(&s)
            )
        }
        "latency" => {
            let (warmup, reps, batches): (f64, usize, usize) = (
                arg(args, 5, "warmup_s")?,
                arg(args, 6, "reps")?,
                arg(args, 7, "batches")?,
            );
            let (p50, p99) = bench.latency(warmup, reps, batches)?;
            format!(
                r#""metric": "latency", "unit": "ns/batch", "method": {{"warmup_s": {warmup}, "reps": {reps}, "batches": {batches}}}, "p50": {}, "p99": {}"#,
                list(&p50),
                list(&p99)
            )
        }
        "snapshot" => {
            let warmup: f64 = arg(args, 4, "warmup_s")?;
            let reps: usize = arg(args, 5, "reps")?;
            let ([snap, all, restore, import, fork, branched], warmups) =
                bench.snapshot(warmup, reps)?;
            format!(
                r#""metric": "snapshot", "unit": "ns/instance", "method": {{"warmup_s": {warmup}, "warmup_iterations": {warmups}, "reps": {reps}}}, "record_bytes": {}, "snapshot_copy": {}, "snapshot_all_copy": {}, "restore_with_reobserve": {}, "import_with_validation_and_reobserve": {}, "fork_with_reobserve": {}, "fork_branched_with_reobserve": {}"#,
                Record::<E>::record_size(),
                list(&snap),
                list(&all),
                list(&restore),
                list(&import),
                list(&fork),
                list(&branched)
            )
        }
        "memory" => {
            let (record, mask, storage) = bench.memory()?;
            format!(
                r#""metric": "memory", "record_bytes_per_instance": {record}, "mask_bytes_per_step": {mask}, "output_storage_bytes": {storage}"#
            )
        }
        other => return Err(usage(&format!("unknown command {other:?}"))),
    };
    let requested = cpus(args)?.map_or_else(|| "engine default".to_owned(), |c| format!("{c:?}"));
    let limit: u32 = option(args, "--max-episode-steps")?.unwrap_or(0);
    Ok(format!(
        r#"{{{body}, "options": {{"seed": 1, "auto_reset": "next_step", "max_episode_steps": {limit}, "action_batches": {ACTION_BATCHES}{}}}, "plan_fingerprint": "{:016x}", "profile": "{}", "pinning": {{"worker_cpu_set": "{requested}", "process_affinity": "{}"}}}}"#,
        // Only when set, so plain runs keep the Python harness's options.
        if bench.legal.is_some() {
            r#", "legal_actions": true"#
        } else {
            ""
        },
        fingerprint(&bench.batches, &bench.continuous),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        affinity()
    ))
}

fn run(args: &[String]) -> Result<String, Error> {
    let cmd = args.get(1).map_or("", String::as_str);
    let env = args.get(2).map_or("", String::as_str);
    let num_envs: u32 = arg(args, 3, "num_envs")?;
    let workers: usize = if matches!(cmd, "throughput" | "latency") {
        arg(args, 4, "workers")?
    } else {
        0
    };
    let cpu_set = cpus(args)?;
    let limit: u32 = option(args, "--max-episode-steps")?.unwrap_or(0);
    let legal = args.iter().any(|a| a == "--legal-actions");
    macro_rules! bench {
        ($env:ty, $cfg:ty, $mask:ty, $make:ident) => {
            run_env(
                args,
                Bench::new(
                    pool::<$env, $mask>(
                        |o| EnvPool::$make(<$cfg>::default(), o),
                        num_envs,
                        workers,
                        cpu_set,
                        limit,
                    )?,
                    &<$cfg>::default(),
                    num_envs,
                    legal,
                )?,
            )
        };
    }
    match env {
        "acrobot" => bench!(Acrobot, AcrobotConfig, NoMasks, new),
        "brickwall" => bench!(BrickWall, BrickWallConfig, NoMasks, new),
        "cartpole" => bench!(CartPole, CartPoleConfig, NoMasks, new),
        "grid" => bench!(Grid, GridConfig, NoMasks, new),
        "keydoor" => bench!(KeyDoor, KeyDoorConfig, Masked, with_masks),
        "mountaincar" => bench!(MountainCar, MountainCarConfig, NoMasks, new),
        "paddleduel" => bench!(PaddleDuel, PaddleDuelConfig, NoMasks, new),
        "pendulum" => bench!(Pendulum, PendulumConfig, NoMasks, new),
        "snake" => bench!(Snake, SnakeConfig, Masked, with_masks),
        "tag" => bench!(Tag, TagConfig, Masked, with_masks),
        "connect4" => bench!(Connect4, Connect4Config, Masked, with_masks),
        "go9" => bench!(Go9, Go9Config, Masked, with_masks),
        "leduc" => bench!(Leduc, LeducConfig, Masked, with_masks),
        "othello" => bench!(Othello, OthelloConfig, Masked, with_masks),
        "rts" => bench!(Rts, RtsConfig, Masked, with_masks),
        other => Err(usage(&format!("unknown env {other:?}"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(json) => {
            if writeln!(io::stdout().lock(), "{json}").is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            let _ = writeln!(io::stderr().lock(), "tessera-rl-bench-runner: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actions_one_head_matches_python_plan() {
        // The first draws of the Python harness's lcg_plan(rows, 2), seed 0.
        let mut state = 0u64;
        let expected: Vec<i32> = (0..8)
            .map(|_| {
                let (next, draw) = lcg(state);
                state = next;
                i32::try_from(draw % 2).unwrap_or(-1)
            })
            .collect();
        assert_eq!(actions(8, &[2], 0).ok(), Some(expected));
    }

    #[test]
    fn test_actions_multi_head_each_head_in_range() {
        let heads = [3, 5, 2];
        let plan = actions(100, &heads, 7).unwrap_or_default();
        assert_eq!(plan.len(), 300);
        for row in plan.as_chunks::<3>().0 {
            for (&v, &n) in row.iter().zip(&heads) {
                assert!((0..i32::try_from(n).unwrap_or(0)).contains(&v));
            }
        }
    }

    #[test]
    fn test_actions_zero_choices_rejected() {
        assert!(actions(4, &[2, 0], 0).is_err());
    }

    #[test]
    fn test_continuous_within_bounds() {
        let bounds = [(-2.0, 2.0), (0.0, 1.0)];
        let plan = continuous(500, &bounds, 3).unwrap_or_default();
        assert_eq!(plan.len(), 1000);
        for row in plan.as_chunks::<2>().0 {
            assert!((-2.0..=2.0).contains(&row[0]));
            assert!((0.0..=1.0).contains(&row[1]));
        }
    }

    #[test]
    fn test_legalize_moves_to_next_legal_cyclically() {
        let t = Bool8::TRUE;
        let f = Bool8::FALSE;
        // Two rows, heads [3, 2]: mask width 5.
        let masks = [f, f, t, t, f, /* row 2 */ t, f, f, f, f];
        let plan = [1, 1, 2, 0];
        let mut out = [9; 4];
        legalize(&mut out, &plan, &masks, &[3, 2]);
        // Row 1: head 0 from 1 to 2; head 1 from 1 wraps to 0.
        // Row 2: head 0 from 2 wraps to 0; head 1 has no legal choice: kept.
        assert_eq!(out, [2, 0, 0, 0]);
    }

    #[test]
    fn test_legalize_keeps_legal_plan() {
        let t = Bool8::TRUE;
        let plan = [1, 0];
        let mut out = [9; 2];
        legalize(&mut out, &plan, &[t, t, t, t], &[2]);
        assert_eq!(out, [1, 0]);
    }
}

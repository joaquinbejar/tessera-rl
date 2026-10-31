// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-sample benchmark runner for the ADR-0025 harness (`tools/bench`).
//! It measures; the harness computes statistics and adds the metadata.
//! Every command writes one JSON object to stdout.
//!
//! - `throughput <env> <num_envs> <workers> <warmup_s> <reps> <rep_s>`:
//!   steps/s per repetition, pregenerated seeded random actions, auto-reset
//!   and observation writing included (SPEC 19.1).
//! - `latency <env> <num_envs> <workers> <warmup_s> <reps> <batches>`: p50
//!   and p99 of the batch step time per repetition, in nanoseconds.
//! - `snapshot <env> <num_envs> <reps>`: per-instance nanoseconds of
//!   snapshot, restore, fork and `fork_branched`, per repetition.
//! - `memory <env> <num_envs>`: continuation bytes per instance, mask
//!   bytes written per step and output storage bytes.
//!
//! `env` is `cartpole` or `tag`; `workers` 0 is Inline mode.

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::Environment;
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, MaskProvider, Masked, NoMasks, PoolOptions, Record, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

/// Batches of pregenerated actions cycled through a measurement.
const ACTION_BATCHES: usize = 64;

#[derive(Debug)]
enum Error {
    Usage(String),
    Engine(EngineError),
    Io(io::Error),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Usage(msg) => write!(f, "usage: {msg}"),
            Self::Engine(e) => write!(f, "engine: {e}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl From<EngineError> for Error {
    fn from(e: EngineError) -> Self {
        Self::Engine(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

fn usage(msg: &str) -> Error {
    Error::Usage(msg.to_owned())
}

fn arg<T: core::str::FromStr>(args: &[String], i: usize, name: &str) -> Result<T, Error> {
    args.get(i)
        .and_then(|a| a.parse().ok())
        .ok_or_else(|| usage(&format!("missing or invalid <{name}>")))
}

/// Seeded actions in `0..choices` (a 64-bit LCG; the values only need to
/// be reproducible and spread).
fn actions(len: usize, choices: u32, seed: u64) -> Vec<i32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = u32::try_from(state >> 33).unwrap_or(0) % choices.max(1);
            i32::try_from(v).unwrap_or(0)
        })
        .collect()
}

fn pool<E, M>(
    make: impl FnOnce(PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    num_envs: u32,
    workers: usize,
) -> Result<EnvPool<E, M>, Error>
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    let n = NonZeroU32::new(num_envs).ok_or_else(|| usage("num_envs must be at least 1"))?;
    let mut p = make(PoolOptions::new(n, Seed::new(1)))?;
    if let Some(w) = NonZeroUsize::new(workers) {
        p = p.with_workers(SyncOptions::new(w))?;
    }
    p.reset_all(Reseed::Keep)?;
    Ok(p)
}

/// One environment's pool with its action batches.
struct Bench<E: Environment, M: MaskProvider<E>> {
    pool: EnvPool<E, M>,
    batches: Vec<Vec<i32>>,
    steps_per_batch: u64,
}

impl<E, M> Bench<E, M>
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    fn new(pool: EnvPool<E, M>) -> Self {
        let (heads, choices) = pool.action_heads();
        let rows = pool.outputs().rewards().len();
        let num_envs = u64::try_from(rows / E::MAX_AGENTS.max(1)).unwrap_or(0);
        let batches = (0..ACTION_BATCHES)
            .map(|b| {
                actions(
                    rows * heads.max(1),
                    u32::try_from(choices).unwrap_or(1),
                    u64::try_from(b).unwrap_or(0),
                )
            })
            .collect();
        Self {
            pool,
            batches,
            steps_per_batch: num_envs,
        }
    }

    fn step(&mut self, k: usize) -> Result<(), Error> {
        let actions = self
            .batches
            .get(k % ACTION_BATCHES)
            .map_or(&[][..], Vec::as_slice);
        let out = self.pool.step(ActionInput {
            discrete: actions,
            continuous: &[],
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
        Ok((u64::try_from(k).unwrap_or(u64::MAX), start.elapsed()))
    }

    fn throughput(&mut self, warmup: f64, reps: usize, rep: f64) -> Result<Vec<f64>, Error> {
        self.run_for(Duration::from_secs_f64(warmup))?;
        (0..reps)
            .map(|_| {
                let (batches, took) = self.run_for(Duration::from_secs_f64(rep))?;
                Ok(steps(batches, self.steps_per_batch) / took.as_secs_f64())
            })
            .collect()
    }

    fn latency(
        &mut self,
        warmup: f64,
        reps: usize,
        batches: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
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
            p50.push(quantile(&times, 0.50));
            p99.push(quantile(&times, 0.99));
        }
        Ok((p50, p99))
    }

    fn snapshot(&mut self, reps: usize) -> Result<[Vec<f64>; 4], Error> {
        let n = u32::try_from(self.steps_per_batch).unwrap_or(0);
        let per = f64::from(n.max(1));
        let others: Vec<u32> = (1..n).collect();
        let mut out: [Vec<f64>; 4] = Default::default();
        let mut records: Vec<Record<E>> = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        for _ in 0..reps {
            self.step(0)?;
            records.clear();
            let start = Instant::now();
            for id in 0..n {
                records.push(self.pool.snapshot(id)?);
            }
            push(&mut out[0], start.elapsed(), per);
            let start = Instant::now();
            for (id, record) in (0..n).zip(&records) {
                self.pool.restore(id, record)?;
            }
            push(&mut out[1], start.elapsed(), per);
            let dst = f64::from(n.saturating_sub(1).max(1));
            let start = Instant::now();
            self.pool.fork(0, &others)?;
            push(&mut out[2], start.elapsed(), dst);
            let start = Instant::now();
            self.pool.fork_branched(0, &others, None)?;
            push(&mut out[3], start.elapsed(), dst);
        }
        Ok(out)
    }

    fn memory(&self) -> (usize, usize, usize) {
        let mask = self.pool.outputs().masks().len();
        (
            Record::<E>::record_size(),
            mask,
            self.pool.outputs().storage().bytes(),
        )
    }
}

#[expect(clippy::cast_precision_loss, reason = "a step count as a rate")]
fn steps(batches: u64, per_batch: u64) -> f64 {
    batches.saturating_mul(per_batch) as f64
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
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted
        .get(rank.saturating_sub(1).min(sorted.len() - 1))
        .copied()
        .unwrap_or(0.0)
}

fn list(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.3}")).collect();
    format!("[{}]", items.join(", "))
}

fn run_env<E, M>(args: &[String], mut bench: Bench<E, M>) -> Result<String, Error>
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    let cmd = args.get(1).map_or("", String::as_str);
    Ok(match cmd {
        "throughput" => {
            let s = bench.throughput(
                arg(args, 5, "warmup_s")?,
                arg(args, 6, "reps")?,
                arg(args, 7, "rep_s")?,
            )?;
            format!(
                r#"{{"metric": "throughput", "unit": "steps/s", "samples": {}}}"#,
                list(&s)
            )
        }
        "latency" => {
            let (p50, p99) = bench.latency(
                arg(args, 5, "warmup_s")?,
                arg(args, 6, "reps")?,
                arg(args, 7, "batches")?,
            )?;
            format!(
                r#"{{"metric": "latency", "unit": "ns/batch", "p50": {}, "p99": {}}}"#,
                list(&p50),
                list(&p99)
            )
        }
        "snapshot" => {
            let [snap, restore, fork, branched] = bench.snapshot(arg(args, 4, "reps")?)?;
            format!(
                r#"{{"metric": "snapshot", "unit": "ns/instance", "snapshot": {}, "restore": {}, "fork": {}, "fork_branched": {}}}"#,
                list(&snap),
                list(&restore),
                list(&fork),
                list(&branched)
            )
        }
        "memory" => {
            let (record, mask, storage) = bench.memory();
            format!(
                r#"{{"metric": "memory", "record_bytes_per_instance": {record}, "mask_bytes_per_step": {mask}, "output_storage_bytes": {storage}}}"#
            )
        }
        other => return Err(usage(&format!("unknown command {other:?}"))),
    })
}

fn run(args: &[String]) -> Result<String, Error> {
    let env = args.get(2).map_or("", String::as_str);
    let num_envs: u32 = arg(args, 3, "num_envs")?;
    let cmd = args.get(1).map_or("", String::as_str);
    let workers: usize = if matches!(cmd, "throughput" | "latency") {
        arg(args, 4, "workers")?
    } else {
        0
    };
    match env {
        "cartpole" => run_env(
            args,
            Bench::new(pool::<CartPole, NoMasks>(
                |o| EnvPool::new(CartPoleConfig::default(), o),
                num_envs,
                workers,
            )?),
        ),
        "tag" => run_env(
            args,
            Bench::new(pool::<Tag, Masked>(
                |o| EnvPool::with_masks(TagConfig::default(), o),
                num_envs,
                workers,
            )?),
        ),
        other => Err(usage(&format!("unknown env {other:?}"))),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut out = io::stdout().lock();
    let mut err = io::stderr().lock();
    match run(&args) {
        Ok(json) => {
            if writeln!(out, "{json}").is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            let _ = writeln!(err, "tessera-rl-bench-runner: {e}");
            ExitCode::FAILURE
        }
    }
}

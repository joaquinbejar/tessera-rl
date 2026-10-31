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
//! - `snapshot <env> <num_envs> <reps>`: per-instance nanoseconds. Copy
//!   only: `snapshot` (one record at a time) and `snapshot_all` (into a
//!   preallocated buffer). Including re-observation (ADR-0039):
//!   `restore`, `import` (plus validation), `fork` and `fork_branched`.
//! - `memory <env> <num_envs>`: continuation bytes per instance, mask
//!   bytes written per step and output storage bytes.
//!
//! `env` is `cartpole` or `tag`; `workers` 0 is Inline mode. An optional
//! trailing `--cpus a,b,...` pins the Sync-mode workers to those logical
//! CPUs (pin the process itself with `taskset` on Linux). The report
//! includes the process CPU affinity where the OS exposes it.

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::Environment;
use tessera_rl_engine::{
    ActionInput, DecodedRecord, EngineError, EnvPool, MaskProvider, Masked, NoMasks, PoolOptions,
    Record, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

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

/// Seeded actions in `0..choices` (a 64-bit LCG; the values only need to
/// be reproducible and spread).
fn actions(len: usize, choices: u32, seed: u64) -> Result<Vec<i32>, Error> {
    if choices == 0 {
        return Err(usage("an action head without choices"));
    }
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = u32::try_from(state >> 33).map_err(|_| usage("action draw"))? % choices;
            i32::try_from(v).map_err(|_| usage("action out of i32"))
        })
        .collect()
}

fn pool<E, M>(
    make: impl FnOnce(PoolOptions) -> Result<EnvPool<E, M>, EngineError>,
    num_envs: u32,
    workers: usize,
    cpu_set: Option<Vec<usize>>,
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
    let mut p = make(PoolOptions::new(n, Seed::new(1)))?;
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
    fn new(pool: EnvPool<E, M>, cfg: &E::Config, num_envs: u32) -> Result<Self, Error> {
        if pool.action_heads() != (1, 0) {
            return Err(usage("the runner drives one discrete head"));
        }
        let Ok(tessera_rl_core::space::Space::Discrete { n: choices }) = E::action_space(cfg)
        else {
            return Err(usage("the runner drives a Discrete action space"));
        };
        let rows = pool.outputs().rewards().len();
        let batches = (0..ACTION_BATCHES)
            .map(|b| actions(rows, choices, u64::try_from(b).map_err(|_| usage("batch"))?))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            pool,
            batches,
            num_envs,
        })
    }

    fn step(&mut self, k: usize) -> Result<(), Error> {
        let actions = self
            .batches
            .get(k % ACTION_BATCHES)
            .ok_or_else(|| usage("action batch"))?;
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

    fn snapshot(&mut self, reps: usize) -> Result<[Vec<f64>; 6], Error> {
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
        for _ in 0..reps {
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
        }
        Ok(out)
    }

    fn memory(&self) -> (usize, usize, usize) {
        (
            Record::<E>::record_size(),
            self.pool.outputs().masks().len(),
            self.pool.outputs().storage().bytes(),
        )
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
            let s = bench.throughput(
                arg(args, 5, "warmup_s")?,
                arg(args, 6, "reps")?,
                arg(args, 7, "rep_s")?,
            )?;
            format!(
                r#""metric": "throughput", "unit": "steps/s", "samples": {}"#,
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
                r#""metric": "latency", "unit": "ns/batch", "p50": {}, "p99": {}"#,
                list(&p50),
                list(&p99)
            )
        }
        "snapshot" => {
            let [snap, all, restore, import, fork, branched] =
                bench.snapshot(arg(args, 4, "reps")?)?;
            format!(
                r#""metric": "snapshot", "unit": "ns/instance", "record_bytes": {}, "snapshot_copy": {}, "snapshot_all_copy": {}, "restore_with_reobserve": {}, "import_with_validation_and_reobserve": {}, "fork_with_reobserve": {}, "fork_branched_with_reobserve": {}"#,
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
            let (record, mask, storage) = bench.memory();
            format!(
                r#""metric": "memory", "record_bytes_per_instance": {record}, "mask_bytes_per_step": {mask}, "output_storage_bytes": {storage}"#
            )
        }
        other => return Err(usage(&format!("unknown command {other:?}"))),
    };
    let requested = cpus(args)?.map_or_else(|| "engine default".to_owned(), |c| format!("{c:?}"));
    Ok(format!(
        r#"{{{body}, "pinning": {{"worker_cpu_set": "{requested}", "process_affinity": "{}"}}}}"#,
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
    match env {
        "cartpole" => run_env(
            args,
            Bench::new(
                pool::<CartPole, NoMasks>(
                    |o| EnvPool::new(CartPoleConfig::default(), o),
                    num_envs,
                    workers,
                    cpu_set,
                )?,
                &CartPoleConfig::default(),
                num_envs,
            )?,
        ),
        "tag" => run_env(
            args,
            Bench::new(
                pool::<Tag, Masked>(
                    |o| EnvPool::with_masks(TagConfig::default(), o),
                    num_envs,
                    workers,
                    cpu_set,
                )?,
                &TagConfig::default(),
                num_envs,
            )?,
        ),
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

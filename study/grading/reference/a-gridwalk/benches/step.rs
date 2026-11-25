//! The bench `cargo tessera bench` runs (SPEC 10.2): steps per second at 1,
//! 1024 and 8192 instances, Inline and with 4 workers (scaling), and the
//! snapshot and restore cost per instance at 1024 instances. The numbers
//! hold for the machine that runs it.

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use study_ref_a::{AGENTS, GridConfig, GridWalk};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Record, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;

/// Pool sizes.
const SIZES: [u32; 3] = [1, 1024, 8192];
/// Workers of the Sync-mode pools.
const WORKERS: usize = 4;

/// A reset pool of `n` instances: Inline for 0 workers.
fn pool(n: u32, workers: usize) -> Option<EnvPool<GridWalk, Masked>> {
    let options = PoolOptions::new(NonZeroU32::new(n)?, Seed::new(1));
    let mut pool = EnvPool::<GridWalk, Masked>::with_masks(GridConfig::default(), options).ok()?;
    if let Some(w) = NonZeroUsize::new(workers) {
        pool = pool.with_workers(SyncOptions::new(w)).ok()?;
    }
    pool.reset_all(Reseed::Keep).ok()?;
    Some(pool)
}

/// One action per agent: add 1 (the no-op replaces it once illegal).
fn actions(n: u32) -> Vec<i32> {
    vec![1; usize::try_from(n).unwrap_or(0).saturating_mul(AGENTS)]
}

fn step(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    for n in SIZES {
        group.throughput(Throughput::Elements(u64::from(n)));
        let discrete = actions(n);
        for (name, workers) in [("inline", 0), ("workers4", WORKERS)] {
            let Some(mut pool) = pool(n, workers) else {
                continue;
            };
            group.bench_with_input(BenchmarkId::new(name, n), &discrete, |b, discrete| {
                b.iter(|| {
                    let input = ActionInput {
                        discrete,
                        continuous: &[],
                    };
                    black_box(pool.step(input).map(|o| o.rewards().len()))
                });
            });
        }
    }
    group.finish();
}

fn snapshot(c: &mut Criterion) {
    let n = 1024u32;
    let Some(mut pool) = pool(n, 0) else {
        return;
    };
    let discrete = actions(n);
    for _ in 0..8 {
        let input = ActionInput {
            discrete: &discrete,
            continuous: &[],
        };
        if pool.step(input).is_err() {
            return;
        }
    }
    let Ok(mut records) = (0..n)
        .map(|id| pool.snapshot(id))
        .collect::<Result<Vec<Record<GridWalk>>, _>>()
    else {
        return;
    };
    let mut group = c.benchmark_group("snapshot");
    group.throughput(Throughput::Elements(u64::from(n)));
    group.bench_function(BenchmarkId::new("save", n), |b| {
        b.iter(|| black_box(pool.snapshot_all(&mut records).is_ok()));
    });
    group.bench_function(BenchmarkId::new("restore", n), |b| {
        b.iter(|| black_box(pool.restore_all(&records).is_ok()));
    });
    group.finish();
}

criterion_group!(benches, step, snapshot);
criterion_main!(benches);

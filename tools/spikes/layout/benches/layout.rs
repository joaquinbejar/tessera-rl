// SPDX-License-Identifier: MIT OR Apache-2.0

//! The layout spike's timings (ADR-0025 method: warm-up 3 s, 20 samples,
//! 5 s measurement; median and 95% bootstrap CI computed by
//! `scripts/spike_stats.py` from Criterion's samples).
//!
//! - W1: CartPole step, N = 1024 and 16384.
//! - W2: Tag step, default and crowded (8 agents, slip), N = 1024.
//! - W3: fork-heavy (snapshot, fork into k = 1, 8, 64, 10 steps,
//!   restore), and snapshot/restore of every instance, N = 16384.
//! - Physics only, AoS against SoA lanes (CartPole).

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::doc_markdown,
    reason = "bench setup; layout names are prose"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tessera_rl_classic::{CartPole, CartPoleConfig, CartPoleState, Tag, TagConfig};
use tessera_rl_spike_layout::columns::Columnar;
use tessera_rl_spike_layout::kernel::{SpikeEnv, Work};
use tessera_rl_spike_layout::lanes::{physics_aos, physics_soa};
use tessera_rl_spike_layout::layouts::{Blocks, Columns, Combined, Layout, Parallel};
use tessera_rl_spike_layout::workload::{actions, fork_heavy, legalize};

const SEED: u64 = 20_261_031;

fn step_bench<E: SpikeEnv, L: Layout<E>>(
    c: &mut Criterion,
    group: &str,
    cfg: &E::Config,
    n: usize,
    choices: u32,
    limit: u32,
) {
    let mut g = c.benchmark_group(group);
    let mut layout = L::new(cfg, n, SEED);
    let mut work = Work::new::<E>(cfg, n);
    let rows = n * E::MAX_AGENTS;
    let width = usize::try_from(choices).unwrap();
    // Pre-generated actions, legalized per step: the cost is the step.
    let base: Vec<Vec<i32>> = (0..64).map(|t| actions(rows, choices, t)).collect();
    let mut t = 0usize;
    g.bench_function(BenchmarkId::new(L::NAME, n), |b| {
        b.iter(|| {
            let mut acts = base[t % base.len()].clone();
            legalize(&mut acts, &work, width);
            layout.step_all(cfg, &acts, &mut work, limit);
            t += 1;
            black_box(&work.rewards);
        });
    });
    g.finish();
}

fn fork_bench<E: SpikeEnv, L: Layout<E>>(c: &mut Criterion, cfg: &E::Config, choices: u32) {
    let mut g = c.benchmark_group("w3_fork_heavy_cartpole");
    for k in [1usize, 8, 64] {
        let mut layout = L::new(cfg, k + 1, SEED);
        let mut work = Work::new::<E>(cfg, k + 1);
        g.bench_function(BenchmarkId::new(L::NAME, k), |b| {
            b.iter(|| {
                fork_heavy::<E, L>(&mut layout, cfg, &mut work, choices, k, 10, 500);
                black_box(&work.rewards);
            });
        });
    }
    g.finish();
}

/// Whole-pool snapshot and restore in each layout's own storage, and the
/// per-instance path (one `get` or `set` per instance) for comparison.
fn snapshot_bench<E: SpikeEnv, L: Layout<E>>(c: &mut Criterion, group: &str, cfg: &E::Config) {
    const N: usize = 16_384;
    let mut g = c.benchmark_group(group);
    let mut layout = L::new(cfg, N, SEED);
    let mut copy = L::new(cfg, N, SEED + 1);
    g.bench_function(BenchmarkId::new(L::NAME, "snapshot_all"), |b| {
        b.iter(|| {
            layout.copy_all(&mut copy);
            black_box(&copy);
        });
    });
    g.bench_function(BenchmarkId::new(L::NAME, "restore_all"), |b| {
        b.iter(|| {
            copy.copy_all(&mut layout);
            black_box(&layout);
        });
    });
    let mut saved: Vec<(E::Header, E::State)> = (0..N).map(|i| layout.get(i).unwrap()).collect();
    g.bench_function(BenchmarkId::new(L::NAME, "get_each"), |b| {
        b.iter(|| {
            for (i, slot) in saved.iter_mut().enumerate() {
                *slot = layout.get(i).unwrap();
            }
            black_box(&saved);
        });
    });
    g.bench_function(BenchmarkId::new(L::NAME, "set_each"), |b| {
        b.iter(|| {
            for (i, (h, s)) in saved.iter().enumerate() {
                layout.set(i, h, s);
            }
            black_box(&layout);
        });
    });
    g.finish();
}

fn all_layouts_cartpole(c: &mut Criterion) {
    let cfg = CartPoleConfig::default();
    for n in [1024, 16_384] {
        step_bench::<CartPole, Parallel<CartPole>>(c, "w1_cartpole_step", &cfg, n, 2, 500);
        step_bench::<CartPole, Combined<CartPole>>(c, "w1_cartpole_step", &cfg, n, 2, 500);
        step_bench::<CartPole, Columns<CartPole>>(c, "w1_cartpole_step", &cfg, n, 2, 500);
        step_bench::<CartPole, Blocks<CartPole>>(c, "w1_cartpole_step", &cfg, n, 2, 500);
    }
    fork_bench::<CartPole, Parallel<CartPole>>(c, &cfg, 2);
    fork_bench::<CartPole, Combined<CartPole>>(c, &cfg, 2);
    fork_bench::<CartPole, Columns<CartPole>>(c, &cfg, 2);
    fork_bench::<CartPole, Blocks<CartPole>>(c, &cfg, 2);
    snapshot_bench::<CartPole, Parallel<CartPole>>(c, "w3_snapshot_cartpole", &cfg);
    snapshot_bench::<CartPole, Combined<CartPole>>(c, "w3_snapshot_cartpole", &cfg);
    snapshot_bench::<CartPole, Columns<CartPole>>(c, "w3_snapshot_cartpole", &cfg);
    snapshot_bench::<CartPole, Blocks<CartPole>>(c, "w3_snapshot_cartpole", &cfg);
}

fn all_layouts_tag(c: &mut Criterion) {
    let crowded = TagConfig {
        pursuers: 4,
        evaders: 4,
        slip: 300,
        ..TagConfig::default()
    };
    for (name, cfg) in [
        ("w2_tag_default_step", TagConfig::default()),
        ("w2_tag_crowded_step", crowded),
    ] {
        step_bench::<Tag, Parallel<Tag>>(c, name, &cfg, 1024, 5, 200);
        step_bench::<Tag, Combined<Tag>>(c, name, &cfg, 1024, 5, 200);
        step_bench::<Tag, Columns<Tag>>(c, name, &cfg, 1024, 5, 200);
        step_bench::<Tag, Blocks<Tag>>(c, name, &cfg, 1024, 5, 200);
    }
    let cfg = TagConfig::default();
    snapshot_bench::<Tag, Parallel<Tag>>(c, "w3_snapshot_tag", &cfg);
    snapshot_bench::<Tag, Combined<Tag>>(c, "w3_snapshot_tag", &cfg);
    snapshot_bench::<Tag, Columns<Tag>>(c, "w3_snapshot_tag", &cfg);
    snapshot_bench::<Tag, Blocks<Tag>>(c, "w3_snapshot_tag", &cfg);
}

fn physics(c: &mut Criterion) {
    const N: usize = 16_384;
    let mut g = c.benchmark_group("physics_only_cartpole");
    let push: Vec<i32> = (0..N).map(|i| i32::from(i % 3 == 0)).collect();
    let mut aos: Vec<CartPoleState> = (0..N)
        .map(|_| CartPoleState {
            s: [0.01, 0.0, 0.02, 0.0],
        })
        .collect();
    let mut soa = CartPoleState::columns(N);
    for (i, s) in aos.iter().enumerate() {
        s.scatter(&mut soa, i);
    }
    g.bench_function(BenchmarkId::new("aos", N), |b| {
        b.iter(|| {
            physics_aos(&mut aos, &push);
            // Keep the states bounded.
            for s in &mut aos {
                if s.s[0].abs() > 1.0 {
                    s.s = [0.01, 0.0, 0.02, 0.0];
                }
            }
            black_box(&aos);
        });
    });
    g.bench_function(BenchmarkId::new("soa", N), |b| {
        b.iter(|| {
            physics_soa(&mut soa, &push);
            for i in 0..N {
                if soa.s[0][i].abs() > 1.0 {
                    CartPoleState {
                        s: [0.01, 0.0, 0.02, 0.0],
                    }
                    .scatter(&mut soa, i);
                }
            }
            black_box(&soa);
        });
    });
    g.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(5))
        .sample_size(20)
}

criterion_group! {
    name = benches;
    config = config();
    targets = all_layouts_cartpole, all_layouts_tag, physics
}
criterion_main!(benches);

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search costs on Connect Four: one batched expansion of the root's 7
//! children, and one PUCT call of 256 simulations (selection, batched
//! expansion, evaluation by a uniform evaluator, and backup).

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, Mcts, PuctConfig, SearchOptions, SearchPool,
};

struct Uniform;

impl Evaluator<Connect4> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

fn root() -> Record<Connect4> {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.snapshot(0).unwrap()
}

fn pool(capacity: u32, batch: u32) -> SearchPool<Connect4, Masked> {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(batch).unwrap(),
    );
    o.eviction = Eviction::Oldest;
    SearchPool::new(Connect4Config {}, o, &root(), EnvPool::with_masks).unwrap()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    let mut s = pool(64, 7);
    let r = s.root();
    let requests: Vec<_> = (0..7).map(|a| (r, a)).collect();
    let mut out = vec![r; 7];
    group.bench_function("expand_7", |b| {
        b.iter(|| {
            s.clear().unwrap();
            s.expand(black_box(&requests), &mut out).unwrap();
            black_box(out.first().copied());
        });
    });
    let mut cfg = PuctConfig::default();
    cfg.simulations = 256;
    let mut m = Mcts::new(pool(1_024, 8), cfg).unwrap();
    let start = root();
    group.bench_function("puct_256", |b| {
        b.iter(|| {
            m.pool_mut().set_root(&start).unwrap();
            m.run(&mut Uniform).unwrap();
            black_box(m.select_action(0.0).unwrap());
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

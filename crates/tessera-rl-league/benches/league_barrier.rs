// SPDX-License-Identifier: MIT OR Apache-2.0

//! The league barrier at N = 8,192 instances, K = 8 results each (ADR-0088
//! item 11), and one result record.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Outcome, PolicyId, PolicyKind, Population, Results, elo,
};

const N: usize = 8192;
const K: usize = 8;
const POPULATION: u32 = 64;

fn league() -> League {
    let mut p = Population::new();
    for tag in 0..POPULATION {
        p.add(
            PolicyKind::External {
                tag: u64::from(tag),
            },
            elo::INITIAL,
            Rating::initial(),
            0,
        )
        .unwrap_or(PolicyId(0));
    }
    League::new(
        LeagueConfig {
            rate_population: true,
            ..LeagueConfig::default()
        },
        p,
    )
    .unwrap_or_else(|_| std::process::abort())
}

fn fill(r: &mut Results) {
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    for i in 0..N {
        for _ in 0..K {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let o = match s % 3 {
                0 => Outcome::Win,
                1 => Outcome::Draw,
                _ => Outcome::Loss,
            };
            let id = PolicyId(u32::try_from((s >> 8) % u64::from(POPULATION)).unwrap_or(0));
            let _ = r.record(i, id, o);
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("league");
    g.bench_function("barrier_n8192_k8_pop64", |b| {
        let mut l = league();
        let mut template = Results::new(N, K).unwrap_or_else(|_| std::process::abort());
        fill(&mut template);
        b.iter_batched_ref(
            || template.clone(),
            |r| black_box(l.barrier(r)),
            criterion::BatchSize::LargeInput,
        );
    });
    g.bench_function("record", |b| {
        let mut r = Results::new(N, K).unwrap_or_else(|_| std::process::abort());
        let mut i = 0usize;
        b.iter(|| {
            i = (i + 1) % N;
            black_box(r.record(black_box(i), PolicyId(3), Outcome::Win))
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

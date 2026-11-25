// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine part of the ONNX opponent pass (ADR-0088 item 2.3, erratum
//! 2): a `PaddleDuel` step at 1,024 instances, slot 1 an opponent, with one
//! ONNX policy (a Rust stand-in that only reads its rows, so the time is
//! the engine's gather, range check and write) against the same pool with
//! an External opponent. The runtime's own cost is reported separately.

#![expect(
    clippy::unwrap_used,
    reason = "bench setup: a failure aborts the bench"
)]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::paddleduel::OBS_STATE;
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::league::{OpponentPolicies, OpponentPolicyError, sealed};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_league::trueskill::Rating;
use tessera_rl_league::{
    League, LeagueConfig, Matchmaking, PolicyKind, PoolLeague, Population, elo,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 1024;
const HASH: [u8; 16] = [9; 16];

struct FirstByte;

impl sealed::Sealed for FirstByte {}

impl OpponentPolicies for FirstByte {
    fn serves(&self, content_hash: &[u8; 16]) -> bool {
        *content_hash == HASH
    }

    fn act(
        &mut self,
        _: &[u8; 16],
        rows: &[u8],
        n: usize,
        actions: &mut [i64],
    ) -> Result<(), OpponentPolicyError> {
        let row = rows.len() / n;
        for (j, a) in actions.iter_mut().enumerate() {
            *a = i64::from(rows.get(j * row).copied().unwrap_or(0) % 3);
        }
        Ok(())
    }
}

fn pool(kind: PolicyKind) -> EnvPool<PaddleDuel, NoMasks> {
    let mut population = Population::new();
    population
        .add(kind, elo::INITIAL, Rating::initial(), 0)
        .unwrap();
    let config = LeagueConfig {
        matchmaking: Matchmaking::Uniform,
        ..LeagueConfig::default()
    };
    let league = PoolLeague::new(League::new(config, population).unwrap()).unwrap();
    let view = league.view().unwrap();
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
    };
    let cfg = PaddleDuelConfig {
        agents: 2,
        obs: OBS_STATE,
        ..PaddleDuelConfig::default()
    };
    let p = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options)
        .unwrap()
        .with_league(&[1], view, Box::new(league))
        .unwrap();
    let mut p = if matches!(kind, PolicyKind::Onnx { .. }) {
        p.with_onnx_opponents(Box::new(FirstByte)).unwrap()
    } else {
        p
    };
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn bench(c: &mut Criterion) {
    let actions = vec![1i32; N as usize * 2];
    let mut g = c.benchmark_group("league_onnx_pass");
    for (name, kind) in [
        ("external", PolicyKind::External { tag: 1 }),
        ("onnx_stand_in", PolicyKind::Onnx { content_hash: HASH }),
    ] {
        let mut p = pool(kind);
        g.bench_function(name, |b| {
            b.iter(|| {
                black_box(p.step(ActionInput {
                    discrete: &actions,
                    continuous: &[],
                }))
                .unwrap();
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

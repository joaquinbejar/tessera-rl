// SPDX-License-Identifier: MIT OR Apache-2.0

//! One scripted bot's act for one side on a played state at sizes 16 and
//! 32 (16 unit slots): the side view, five flow fields and the orders.
//! Local only, no claim (SPEC 16.1).

#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "bench setup")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_core::AgentSlot;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig};

fn bench(c: &mut Criterion) {
    for size in [16u16, 32] {
        let cfg = RtsConfig {
            size,
            max_units: 16,
            ..RtsConfig::default()
        };
        let n = usize::from(cfg.max_units) * HEADS_PER_UNIT;
        let options = PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(9));
        let mut pool = EnvPool::<Rts, Masked>::with_masks(cfg.clone(), options).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        let identity = RngIdentity::new(Seed::new(1), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut actions = vec![0; 2 * n];
        // 100 decisions of Rush against Prospector, so the state is grown.
        for _ in 0..100 {
            let s = *pool.state(0).unwrap();
            for (agent, bot) in [(0u16, Bot::Rush), (1, Bot::Prospector)] {
                let lo = usize::from(agent) * n;
                let mut w = ActionWriter::new(&mut actions[lo..lo + n], &mut []);
                bot.act(
                    &cfg,
                    &s,
                    AgentSlot::new(agent, 2).unwrap(),
                    &mut rng,
                    &mut w,
                );
            }
            pool.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
        }
        let s = *pool.state(0).unwrap();
        let mut heads = vec![0; n];
        for bot in [Bot::Rush, Bot::Prospector] {
            c.bench_function(&format!("rts/bots/{bot:?}/size{size}"), |b| {
                b.iter(|| {
                    let mut w = ActionWriter::new(&mut heads, &mut []);
                    bot.act(
                        &cfg,
                        black_box(&s),
                        AgentSlot::new(0, 2).unwrap(),
                        &mut rng,
                        &mut w,
                    );
                });
            });
        }
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);

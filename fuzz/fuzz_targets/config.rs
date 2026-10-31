// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configs at construction: arbitrary Tag configs and pool options are
//! validated or rejected with a typed error; a pool that builds must reset
//! and step.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_core::EnvConfig;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

fn u32_at(data: &[u8], i: usize) -> u32 {
    let mut b = [0u8; 4];
    for (k, slot) in b.iter_mut().enumerate() {
        *slot = data.get(i * 4 + k).copied().unwrap_or(0);
    }
    u32::from_le_bytes(b)
}

fuzz_target!(|data: &[u8]| {
    let cfg = TagConfig {
        width: u32_at(data, 0) % 80,
        height: u32_at(data, 1) % 80,
        pursuers: u32_at(data, 2) % 10,
        evaders: u32_at(data, 3) % 10,
        slip: u32_at(data, 4) % 1_200,
        radius: u32_at(data, 5) % 10,
    };
    let valid = cfg.validate().is_ok();
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(u32_at(data, 6) % 4_096 + 1).unwrap(),
        max_episode_steps: NonZeroU32::new(u32_at(data, 7) % 1_000),
        ..PoolOptions::new(NonZeroU32::new(u32_at(data, 8) % 8 + 1).unwrap(), Seed::new(1))
    };
    match EnvPool::<Tag, Masked>::with_masks(cfg, options) {
        Ok(mut p) => {
            assert!(valid, "an invalid config built a pool");
            p.reset_all(Reseed::Keep).unwrap();
            let acts = vec![0; p.outputs().rewards().len()];
            p.step(ActionInput {
                discrete: &acts,
                continuous: &[],
            })
            .unwrap();
        }
        Err(_) => {}
    }
});

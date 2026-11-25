// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gather v1 against v2 (#613 head A, REVIEW-design-r1 condition 1): at
//! the default `items` and with no targets, making `items` adaptive changes
//! only the record and header encoding. The per-step state bytes (which
//! follow the RNG draw order), observations, rewards and episode flags
//! hash to the value pinned from v1 (`ENV_VERSION` 1) on the golden
//! scenarios' configuration, seed, pool size and both auto-reset modes.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{Gather, GatherConfig};
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// FNV-1a, folded over every byte the trajectory produces.
struct Fnv(u64);

impl Fnv {
    fn add(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// 16 instances of the golden configuration (6x6, 5 items, 30 steps),
/// 300 steps of seeded pseudo-random actions.
fn trajectory(auto_reset: AutoReset) -> u64 {
    let cfg = GatherConfig {
        size: 6,
        items: 5,
        max_steps: 30,
    };
    let n = 16;
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(24_112_026))
    };
    let mut p = EnvPool::<Gather, NoMasks>::new(cfg, options).unwrap();
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let out = p.reset_all(Reseed::Keep).unwrap();
    for leaf in 0..3 {
        h.add(out.observation(leaf).unwrap());
    }
    for _ in 0..300 {
        let actions: Vec<i32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                i32::try_from(x % 4).unwrap()
            })
            .collect();
        let out = p
            .step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
        for leaf in 0..3 {
            h.add(out.observation(leaf).unwrap());
        }
        for r in out.rewards() {
            h.add(&r.to_bits().to_le_bytes());
        }
        for f in out
            .episode_terminated()
            .iter()
            .chain(out.episode_truncated())
        {
            h.add(&[u8::from(f.get())]);
        }
        for i in 0..n {
            h.add(&encode_to_vec(&p.state(i).unwrap()));
        }
    }
    h.0
}

#[test]
fn test_gather_v2_default_trajectory_equals_v1() {
    let got = (
        trajectory(AutoReset::NextStep),
        trajectory(AutoReset::SameStep),
    );
    // Pinned on ENV_VERSION 1 (`e6f72bfe`, Linux x86-64), before `items`
    // became adaptive.
    assert_eq!(got, (12_935_342_411_642_232_344, 14_503_861_675_341_482_048));
}

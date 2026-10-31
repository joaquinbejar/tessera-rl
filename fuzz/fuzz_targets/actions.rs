// SPDX-License-Identifier: MIT OR Apache-2.0

//! The validation boundary (SPEC 5.5): arbitrary actions for a masked Tag
//! pool and a CartPole pool. `check_actions` and `step` must return a typed
//! result, never panic, and agree: `step` only fails on actions that
//! `check_actions` (or the shape check) rejects.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

fn words(data: &[u8]) -> Vec<i32> {
    data.chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            i32::from_le_bytes(b)
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let Some((&seed, rest)) = data.split_first() else {
        return;
    };
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(u64::from(seed)));
    let acts = words(rest);
    let input = ActionInput {
        discrete: &acts,
        continuous: &[],
    };
    if let Ok(mut tag) = EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), options) {
        tag.reset_all(Reseed::Keep).unwrap();
        let checked = tag.check_actions(input);
        let stepped = tag.step(input).map(|_| ());
        if checked.is_ok() {
            assert!(stepped.is_ok(), "check_actions accepted what step rejected: {stepped:?}");
        }
    }
    if let Ok(mut cp) = EnvPool::<CartPole>::new(CartPoleConfig::default(), options) {
        cp.reset_all(Reseed::Keep).unwrap();
        let _ = cp.step(input);
    }
});

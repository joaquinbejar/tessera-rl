// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay files (ADR-0072): `Replay::open`, the event iterator and
//! `Replay::run` on arbitrary bytes, and on a valid CartPole recording mutated
//! by the input with both checksums resealed, so the input reaches the checks
//! behind them. Every refusal must be a typed error, never a panic.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_classic::{CartPole, CartPoleConfig};
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::replay::{Recorder, Replay, ReplayWriterOptions};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 8,
    max_decoded_bytes: 1 << 20,
    max_host_record_bytes: 1 << 16,
};

fn pool() -> EnvPool<CartPole, NoMasks> {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(3));
    EnvPool::new(CartPoleConfig::default(), options).unwrap()
}

fn recording() -> Vec<u8> {
    let mut p = pool();
    let options = ReplayWriterOptions {
        keyframe_every: 2,
        checkpoint_every: 2,
        outputs_hash: false,
    };
    let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(5))).unwrap();
    for t in 0..3 {
        let a = [t % 2, (t + 1) % 2];
        rec.step(ActionInput { discrete: &a, continuous: &[] }).unwrap();
    }
    rec.reset(&[1], Reseed::Each(&[Some(Seed::new(8))])).unwrap();
    rec.fork_branched(0, &[1], None).unwrap();
    rec.step(ActionInput { discrete: &[1, 0], continuous: &[] }).unwrap();
    rec.finish().unwrap()
}

fn u16_at(b: &[u8], at: usize) -> Option<usize> {
    Some(usize::from(u16::from_le_bytes([*b.get(at)?, *b.get(at + 1)?])))
}

fn u32_at(b: &[u8], at: usize) -> Option<usize> {
    let bytes: [u8; 4] = b.get(at..at + 4)?.try_into().ok()?;
    usize::try_from(u32::from_le_bytes(bytes)).ok()
}

/// Rewrites the header checksum and the end event's checksum, where the
/// mutated layout still lets them be found.
fn reseal(b: &mut [u8]) -> Option<()> {
    let mut at: usize = 48;
    for _ in 0..2 {
        at = at.checked_add(2)?.checked_add(u16_at(b, at)?)?;
    }
    at = at.checked_add(2)?;
    at = at.checked_add(2)?.checked_add(u16_at(b, at)?)?;
    at = at.checked_add(4)?.checked_add(u32_at(b, at)?)?.checked_add(16)?;
    let sum = checksum(b.get(..at)?);
    b.get_mut(at..at.checked_add(8)?)?.copy_from_slice(&sum.to_le_bytes());
    let start = at + 8;
    let mut at = start;
    loop {
        let tag = *b.get(at)?;
        let len = u32_at(b, at + 1)?;
        if tag == 0 {
            let sum = checksum(b.get(start..at)?);
            let end = at.checked_add(13)?;
            b.get_mut(end..end.checked_add(8)?)?.copy_from_slice(&sum.to_le_bytes());
            return Some(());
        }
        at = at.checked_add(5)?.checked_add(len)?;
    }
}

fn exercise(bytes: &[u8]) {
    let Ok(replay) = Replay::open(bytes, LIMITS) else {
        return;
    };
    for event in replay.events() {
        if event.is_err() {
            break;
        }
    }
    let _ = replay.run(&mut pool());
}

fuzz_target!(|data: &[u8]| {
    exercise(data);
    let mut bytes = recording();
    let len = bytes.len();
    for edit in data.chunks_exact(3) {
        let at = usize::from(u16::from_le_bytes([edit[0], edit[1]])) % len;
        bytes[at] ^= edit[2];
    }
    let _ = reseal(&mut bytes);
    exercise(&bytes);
});

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host replay files (ADR-0072, ADR-0073): `Replay::open`, the event
//! iterator and `run_host_snapshots` on arbitrary bytes, and on a valid
//! recording of a snapshot host mutated by the input with both checksums
//! resealed, so the input reaches the checks behind them. Every refusal must
//! be a typed error, never a panic.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_core::config::{CanonicalConfig, write_entry};
use tessera_rl_core::encoding::{Encoder, Sink, checksum};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot,
    SnapshotReject, SpaceError,
};
use tessera_rl_engine::replay::{HostRecorder, Replay, ReplayWriterOptions};
use tessera_rl_engine::{ActionInput, DecodeLimits, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const LIMITS: DecodeLimits = DecodeLimits {
    max_instances: 8,
    max_decoded_bytes: 1 << 20,
    max_host_record_bytes: 64,
};

struct Cfg;

impl EnvConfig for Cfg {}

impl CanonicalConfig for Cfg {
    const ADAPTIVE_FIELDS: &'static [&'static str] = &[];
    const STATIC_ENTRIES: u16 = 1;

    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
        write_entry(out, "cap", &16u32);
    }
}

/// Heap state: the actions taken, at most 16.
struct Heap(Vec<u8>);

impl HostEnvironment for Heap {
    const ENV_ID: &'static str = "fuzz/ReplayHeap-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = Cfg;

    fn new(_: &Cfg) -> Result<Self, HostError> {
        Ok(Self(Vec::new()))
    }
    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.0.clear();
    }
    fn step(&mut self, actions: &Actions<'_, ()>, _: &mut HostOut<'_>) {
        let a = actions
            .discrete(AgentSlot::FIRST, 0)
            .and_then(|a| u8::try_from(a).ok())
            .unwrap_or(0);
        if self.0.len() < 16 {
            self.0.push(a);
        }
    }
    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

impl HostSnapshot for Heap {
    const SNAPSHOT_VERSION: u32 = 1;
    fn max_snapshot_len(_: &Cfg) -> usize {
        16
    }
    fn save(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn check_snapshot(_: &Cfg, bytes: &[u8]) -> Result<(), SnapshotReject> {
        if bytes.len() <= 16 {
            Ok(())
        } else {
            Err(SnapshotReject {
                code: 1,
                reason: "too long",
            })
        }
    }
    fn load(&mut self, bytes: &[u8]) {
        self.0.clear();
        self.0.extend_from_slice(bytes);
    }
}

fn pool() -> HostPool<Heap, 1> {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(1));
    HostPool::new(Cfg, options).unwrap()
}

fn recording() -> Vec<u8> {
    let mut p = pool();
    let options = ReplayWriterOptions {
        keyframe_every: 2,
        checkpoint_every: 2,
        outputs_hash: true,
    };
    let mut rec = HostRecorder::with_snapshots(&mut p, Vec::new(), options).unwrap();
    rec.reset_all(Reseed::Base(Seed::new(5))).unwrap();
    for t in 0..3 {
        let a = [t % 2, (t + 1) % 2];
        rec.step(ActionInput { discrete: &a, continuous: &[] }).unwrap();
    }
    rec.fork(0, &[1]).unwrap();
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
    at = at.checked_add(4)?.checked_add(u32_at(b, at)?)?.checked_add(12)?;
    let identity = if *b.get(at)? & 8 == 0 { 0 } else { 8 };
    at = at.checked_add(4 + identity)?;
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
    let _ = replay.run_host_snapshots(&mut pool());
    let _ = replay.run_host(&mut pool());
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

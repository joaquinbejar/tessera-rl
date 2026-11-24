// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host snapshot files and records (ADR-0073 item 5): arbitrary bytes are
//! a typed error, or records that a pool checks before importing; a host
//! record that decodes re-encodes to exactly its bytes.

#![no_main]

use core::num::NonZeroU32;

use libfuzzer_sys::fuzz_target;
use tessera_rl_core::config::{CanonicalConfig, write_entry};
use tessera_rl_core::encoding::{Encoder, Sink};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot,
    SnapshotReject, SpaceError,
};
use tessera_rl_engine::{DecodeLimits, HostPool, HostRecord, PoolOptions};
use tessera_rl_num::identity::Seed;

struct Cfg;

impl EnvConfig for Cfg {}

impl CanonicalConfig for Cfg {
    const ADAPTIVE_FIELDS: &'static [&'static str] = &[];
    const STATIC_ENTRIES: u16 = 1;

    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
        write_entry(out, "cap", &16u32);
    }
}

/// Heap state: a list of bytes, at most 16.
struct Heap(Vec<u8>);

impl HostEnvironment for Heap {
    const ENV_ID: &'static str = "fuzz/Heap-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
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
    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
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

fuzz_target!(|data: &[u8]| {
    let Some((&which, rest)) = data.split_first() else {
        return;
    };
    if which % 2 == 0 {
        if let Ok(record) = HostRecord::<2>::from_bytes(rest, 64) {
            assert_eq!(record.to_bytes().as_deref(), Some(rest));
        }
        return;
    }
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap_or(NonZeroU32::MIN), Seed::new(1));
    let Ok(mut pool) = HostPool::<Heap, 2>::new(Cfg, options) else {
        return;
    };
    let limits = DecodeLimits {
        max_instances: 4,
        max_decoded_bytes: 1 << 16,
        max_host_record_bytes: 64,
    };
    if let Ok(records) = pool.decode(rest, limits) {
        let ids: Vec<u32> = (0..u32::try_from(records.len()).unwrap_or(0)).collect();
        let _ = pool.import(&ids, &records);
    }
});

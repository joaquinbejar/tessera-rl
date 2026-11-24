// SPDX-License-Identifier: MIT OR Apache-2.0

//! A host environment with heap state and its own RNG, for the snapshot
//! tests (ADR-0073): a random walk whose path is a `Vec`, driven by a
//! splitmix generator kept in `self`. Its snapshot bytes are the RNG
//! state, the position and the path.

#![allow(dead_code, reason = "each test binary uses part of it")]

use tessera_rl_core::config::{CanonicalConfig, write_entry};
use tessera_rl_core::encoding::{Encoder, Sink};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot,
    SnapshotReject, SpaceError,
};

/// The configuration: the longest path, which bounds the snapshot.
#[derive(Debug, Clone, Copy)]
pub struct WalkConfig {
    pub length: u32,
}

impl EnvConfig for WalkConfig {}

impl CanonicalConfig for WalkConfig {
    const ADAPTIVE_FIELDS: &'static [&'static str] = &[];
    const STATIC_ENTRIES: u16 = 1;

    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
        write_entry(out, "length", &self.length);
    }
}

/// One walk.
#[derive(Debug)]
pub struct Walk {
    length: u32,
    rng: u64,
    pos: i32,
    path: Vec<i32>,
}

/// The fixed bytes of a snapshot: RNG, position, path length.
const FIXED: usize = 8 + 4 + 4;

/// Reasons `check_snapshot` gives.
pub const SHORT: u32 = 1;
pub const COUNT: u32 = 2;
pub const TOO_LONG: u32 = 3;

impl Walk {
    fn next(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn agent() -> AgentSlot {
    AgentSlot::new(0, 1).unwrap()
}

impl HostEnvironment for Walk {
    const ENV_ID: &'static str = "test/Walk-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = WalkConfig;

    fn new(cfg: &WalkConfig) -> Result<Self, HostError> {
        Ok(Self {
            length: cfg.length,
            rng: 0,
            pos: 0,
            path: Vec::new(),
        })
    }

    fn observation_space(_: &WalkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![3],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![i64::from(i32::MIN)],
                high: vec![i64::from(i32::MAX)],
            },
        })
    }

    fn action_space(_: &WalkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(&mut self, seed: HostSeed, _: &mut HostOut<'_>) {
        self.rng = seed.seed() ^ (u64::from(seed.episode_index()) << 40);
        self.pos = 0;
        self.path.clear();
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        let noise = i32::try_from(self.next() % 3).unwrap_or(0) - 1;
        let action = actions.discrete(agent(), 0).unwrap_or(1) - 1;
        self.pos = self.pos.saturating_add(action).saturating_add(noise);
        if self.path.len() < usize::try_from(self.length).unwrap_or(usize::MAX) {
            self.path.push(self.pos);
        }
        #[expect(clippy::cast_precision_loss, reason = "small test positions")]
        out.set_reward(agent(), self.pos as f32 + (self.next() % 7) as f32 / 8.0);
    }

    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let sum = self.path.iter().fold(0i32, |a, &p| a.wrapping_add(p));
        let values = [self.pos, i32::try_from(self.path.len()).unwrap_or(-1), sum];
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf.as_chunks_mut::<4>().0.iter_mut().zip(values) {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

impl HostSnapshot for Walk {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(cfg: &WalkConfig) -> usize {
        FIXED + 4 * usize::try_from(cfg.length).unwrap_or(0)
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.rng.to_le_bytes());
        out.extend_from_slice(&self.pos.to_le_bytes());
        out.extend_from_slice(&u32::try_from(self.path.len()).unwrap_or(0).to_le_bytes());
        for p in &self.path {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    fn check_snapshot(cfg: &WalkConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        let reject = |code, reason| Err(SnapshotReject { code, reason });
        let Some(count) = bytes.get(12..16) else {
            return reject(SHORT, "shorter than the fixed fields");
        };
        let count = u32::from_le_bytes(count.try_into().unwrap_or([0; 4]));
        if count > cfg.length {
            return reject(TOO_LONG, "a path longer than the config allows");
        }
        if bytes.len() != FIXED + 4 * usize::try_from(count).unwrap_or(usize::MAX) {
            return reject(COUNT, "the path count does not match the bytes");
        }
        Ok(())
    }

    fn load(&mut self, bytes: &[u8]) {
        let word = |at: usize| -> [u8; 4] {
            bytes
                .get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .unwrap_or([0; 4])
        };
        self.rng = u64::from_le_bytes(
            bytes
                .get(0..8)
                .and_then(|b| b.try_into().ok())
                .unwrap_or([0; 8]),
        );
        self.pos = i32::from_le_bytes(word(8));
        let count = usize::try_from(u32::from_le_bytes(word(12))).unwrap_or(0);
        self.path.clear();
        self.path
            .extend((0..count).map(|k| i32::from_le_bytes(word(FIXED + 4 * k))));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers of the environment tests: pools, a seeded sampler of
//! legal actions through the conditional mask layout, and a hash fold.

#![allow(
    dead_code,
    reason = "each test binary uses a different part of the helpers"
)]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, SyncOptions,
};
use tessera_rl_markets::actions::{
    CANCEL_ROWS, CANCELS, HEADS, KINDS, MASK_LEN, OFFSET_ROWS, OFFSETS, SIZE_ROWS, SIZES,
};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobConfig};
use tessera_rl_num::identity::Seed;

/// A masked pool of the environment.
pub type Pool = EnvPool<LimitOrderBook, Masked>;

/// A masked pool of `n` instances; `workers` adds a worker team.
pub fn pool(cfg: LobConfig, n: u32, seed: u64, workers: Option<usize>) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: IllegalActionPolicy::Noop,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p: Pool = EnvPool::with_masks(cfg, options).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p
}

/// A config with `agents` agents and `latency`.
pub fn config(agents: u32, latency: u32) -> LobConfig {
    LobConfig {
        agents,
        latency,
        steps: 200,
        initial_cash: 20_000,
        max_position: 12,
        bankrupt: 0,
        ..LobConfig::default()
    }
}

/// A seeded xorshift generator, independent of the engine.
pub struct Rng(pub u64);

impl Rng {
    /// Uniform in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(n.max(1)).unwrap()).unwrap()
    }

    /// A uniformly drawn legal index of `row`, or 0 if none is legal.
    pub fn pick(&mut self, row: &[Bool8]) -> i32 {
        let legal: Vec<usize> = (0..row.len()).filter(|&i| row[i].get()).collect();
        if legal.is_empty() {
            return 0;
        }
        i32::try_from(legal[self.below(legal.len())]).unwrap()
    }
}

/// One agent's legal action drawn uniformly, head by head, from its mask
/// row through the conditional layout; all zero for a dead agent.
pub fn sample(rng: &mut Rng, row: &[Bool8]) -> [i32; HEADS] {
    let k = rng.pick(&row[..KINDS as usize]);
    let ku = usize::try_from(k).unwrap();
    let offsets = OFFSETS as usize;
    let sizes = SIZES as usize;
    let cancels = CANCELS as usize;
    let o = rng.pick(&row[OFFSET_ROWS + ku * offsets..OFFSET_ROWS + (ku + 1) * offsets]);
    let ou = usize::try_from(o).unwrap();
    let base = SIZE_ROWS + ku * offsets * sizes + ou * sizes;
    let s = rng.pick(&row[base..base + sizes]);
    let c = rng.pick(&row[CANCEL_ROWS + ku * cancels..CANCEL_ROWS + (ku + 1) * cancels]);
    [k, o, s, c]
}

/// Legal actions for every agent of every instance of `n` instances.
pub fn legal_actions(rng: &mut Rng, masks: &[Bool8], n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n * MAX_AGENTS * HEADS);
    for row in 0..n * MAX_AGENTS {
        out.extend(sample(rng, &masks[row * MASK_LEN..(row + 1) * MASK_LEN]));
    }
    out
}

/// The engine input of discrete actions.
pub fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

/// FNV-1a over the little-endian bytes of `word`.
pub fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The FNV offset basis.
pub const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

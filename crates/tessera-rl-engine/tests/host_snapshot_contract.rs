// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check_snapshot_contract` (ADR-0073 item 8): it passes on the walk and
//! fails on an environment whose `load` drops state.

#![cfg(feature = "testing")]
#![expect(clippy::unwrap_used, reason = "a failure is the assertion")]

mod host_walk;

use core::num::NonZeroU32;

use host_walk::{Walk, WalkConfig};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot, SnapshotReject,
    SpaceError,
};
use tessera_rl_engine::PoolOptions;
use tessera_rl_engine::host_testing::{ContractError, ContractStep, check_snapshot_contract};
use tessera_rl_num::identity::Seed;

fn options() -> PoolOptions {
    let mut o = PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(8));
    o.max_episode_steps = NonZeroU32::new(64);
    o
}

fn steps() -> Vec<ContractStep> {
    (0..12)
        .map(|t| ((0..3).map(|i| (t + i) % 3).collect(), Vec::new()))
        .collect()
}

#[test]
fn test_host_snapshot_contract_walk_passes() {
    check_snapshot_contract::<Walk, 1>(&|| WalkConfig { length: 64 }, options(), &steps()).unwrap();
}

/// The walk, but `load` keeps only the RNG: the path is lost.
#[derive(Debug)]
struct Forgetful(Walk);

impl HostEnvironment for Forgetful {
    const ENV_ID: &'static str = "test/Forgetful-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = WalkConfig;

    fn new(cfg: &WalkConfig) -> Result<Self, HostError> {
        Walk::new(cfg).map(Self)
    }
    fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walk::observation_space(cfg)
    }
    fn action_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walk::action_space(cfg)
    }
    fn reset(&mut self, seed: HostSeed, out: &mut HostOut<'_>) {
        self.0.reset(seed, out);
    }
    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.0.step(actions, out);
    }
    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        self.0.observe(agent, obs);
    }
}

impl HostSnapshot for Forgetful {
    const SNAPSHOT_VERSION: u32 = 1;
    fn max_snapshot_len(cfg: &WalkConfig) -> usize {
        Walk::max_snapshot_len(cfg)
    }
    fn save(&self, out: &mut Vec<u8>) {
        self.0.save(out);
    }
    fn check_snapshot(cfg: &WalkConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        Walk::check_snapshot(cfg, bytes)
    }
    fn load(&mut self, bytes: &[u8]) {
        // Only the fixed fields, with an empty path.
        let mut head = bytes.get(..16).unwrap_or(&[]).to_vec();
        if let Some(count) = head.get_mut(12..16) {
            count.copy_from_slice(&0u32.to_le_bytes());
        }
        self.0.load(&head);
    }
}

#[test]
fn test_host_snapshot_contract_forgetful_load_fails() {
    let err =
        check_snapshot_contract::<Forgetful, 1>(&|| WalkConfig { length: 64 }, options(), &steps())
            .unwrap_err();
    assert!(matches!(err, ContractError::NotCanonical { .. }), "{err:?}");
}

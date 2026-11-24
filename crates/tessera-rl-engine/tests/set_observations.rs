// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Set` observations through the engine (ADR-0082): the exported leaves,
//! absent slots zero after `observe` (point 3) whatever the environment
//! wrote into them, or with `debug-checks` (and debug assertions) such a
//! write reported instead, and `Set` actions still refused (point 1).
//!
//! The fixture writes `SENTINEL` into every item row and marks present only
//! the slots whose bit is set in its state, so every absent row it leaves
//! nonzero must come out zero, inline and with workers.

#![expect(clippy::unwrap_used, reason = "tests: a failure is the assertion")]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, ObsSet, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{AutoReset, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

const SENTINEL: u8 = 0xAB;
const MAX: usize = 4;
/// Bytes per item row: `Box[3] u8`.
const ROW: usize = 3;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Leaky {
    present: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct LeakyConfig {
    /// 1: the action space is a `Set`, which the engine refuses.
    #[config(default = 0)]
    set_actions: u32,
}

#[derive(Observation)]
struct LeakyObs {
    #[space(box(low = 0, high = 255, dtype = u8, shape = [1]))]
    head: ObsLeaf<u8>,
    #[space(set(max = MAX, inner(box(low = 0, high = 255, dtype = u8, shape = [ROW]))))]
    items: ObsSet<u8>,
}

/// Writes the sentinel into every item row, present or not.
struct LeakyEnv;

impl Environment for LeakyEnv {
    type State = Leaky;
    type Config = LeakyConfig;
    type Obs = LeakyObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &LeakyConfig) -> Result<Space, SpaceError> {
        LeakyObs::space(cfg)
    }

    fn action_space(cfg: &LeakyConfig) -> Result<Space, SpaceError> {
        let discrete = Space::Discrete { n: 16 };
        if cfg.set_actions == 1 {
            return Ok(Space::Set {
                inner: Box::new(discrete),
                max: 2,
            });
        }
        Ok(discrete)
    }

    fn reset(_: &LeakyConfig, _: &mut StepRng, state: &mut Leaky) {
        *state = Leaky::default();
    }

    /// The action's bits are the present slots.
    fn step(
        _: &LeakyConfig,
        state: &mut Leaky,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        let bits = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0);
        state.present = u32::try_from(bits).unwrap_or(0);
    }

    fn observe(
        _: &LeakyConfig,
        state: &Leaky,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LeakyObs>,
    ) {
        if let Some(mut head) = out.leaf_mut(LeakyObs::HEAD) {
            head.fill(SENTINEL);
        }
        let Some(mut set) = out.set_mut(LeakyObs::ITEMS) else {
            return;
        };
        for slot in 0..MAX {
            set.set(slot, state.present & (1 << slot) != 0);
            if let Some(mut row) = set.item(slot) {
                row.fill(SENTINEL);
            }
        }
    }
}

fn pool(n: u32, workers: Option<usize>) -> EnvPool<LeakyEnv, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
    };
    let mut p = EnvPool::new(LeakyConfig::default(), options).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p
}

#[test]
fn test_set_layout_exports_items_then_mask() {
    let space = LeakyEnv::observation_space(&LeakyConfig::default()).unwrap();
    let Space::Dict(entries) = space else {
        panic!("a dict");
    };
    assert!(matches!(&entries[1].1, Space::Set { max: 4, .. }));
    let p = pool(1, None);
    let out = p.outputs().unwrap();
    assert_eq!(out.observation(0).unwrap().len(), 1);
    assert_eq!(out.observation(1).unwrap().len(), MAX * ROW);
    assert_eq!(out.observation(2).unwrap().len(), MAX);
    assert!(out.observation(3).is_none());
}

#[cfg(not(all(feature = "debug-checks", debug_assertions)))]
#[test]
fn test_set_absent_slots_written_by_observe_come_out_zero() {
    for workers in [None, Some(2)] {
        let mut p = pool(4, workers);
        // Reset: no slot present, every item row zero, the head written.
        let out = p.reset_all(Reseed::Keep).unwrap();
        assert!(out.observation(1).unwrap().iter().all(|&b| b == 0));
        assert!(out.observation(2).unwrap().iter().all(|&b| b == 0));
        assert_eq!(out.observation(0).unwrap(), [SENTINEL; 4]);
        let bits = [0b0101, 0b1111, 0b0000, 0b1000];
        let out = p
            .step(tessera_rl_engine::ActionInput {
                discrete: &bits,
                continuous: &[],
            })
            .unwrap();
        let items = out.observation(1).unwrap();
        let mask = out.observation(2).unwrap();
        for (i, &b) in bits.iter().enumerate() {
            for slot in 0..MAX {
                let present = b & (1 << slot) != 0;
                let row = &items[(i * MAX + slot) * ROW..(i * MAX + slot + 1) * ROW];
                let want = if present { [SENTINEL; ROW] } else { [0; ROW] };
                assert_eq!(row, want, "workers {workers:?} instance {i} slot {slot}");
                assert_eq!(mask[i * MAX + slot], u8::from(present));
            }
        }
    }
}

#[cfg(not(all(feature = "debug-checks", debug_assertions)))]
#[test]
fn test_set_rows_emptied_mid_episode_come_out_zero() {
    for workers in [None, Some(2)] {
        let mut p = pool(2, workers);
        p.reset_all(Reseed::Keep).unwrap();
        // Every slot present, then most slots emptied: the rows that were
        // filled one step earlier read as zeros (ADR-0082 point 3).
        for bits in [[0b1111, 0b1111], [0b0010, 0b0000], [0b0010, 0b1001]] {
            let out = p
                .step(tessera_rl_engine::ActionInput {
                    discrete: &bits,
                    continuous: &[],
                })
                .unwrap();
            let items = out.observation(1).unwrap();
            for (i, &b) in bits.iter().enumerate() {
                for slot in 0..MAX {
                    let present = b & (1 << slot) != 0;
                    let row = &items[(i * MAX + slot) * ROW..(i * MAX + slot + 1) * ROW];
                    let want = if present { [SENTINEL; ROW] } else { [0; ROW] };
                    assert_eq!(
                        row, want,
                        "workers {workers:?} bits {bits:?} instance {i} slot {slot}"
                    );
                }
            }
        }
    }
}

#[cfg(all(feature = "debug-checks", debug_assertions))]
#[test]
fn test_set_absent_slot_write_is_reported_under_debug_checks() {
    let mut p = pool(1, None);
    assert_eq!(
        p.reset_all(Reseed::Keep).err(),
        Some(tessera_rl_engine::EngineError::Panicked)
    );
}

#[test]
fn test_set_action_space_is_still_refused() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let cfg = LeakyConfig { set_actions: 1 };
    assert_eq!(
        EnvPool::<LeakyEnv, NoMasks>::new(cfg, options).err(),
        Some(tessera_rl_engine::EngineError::UnsupportedSpace {
            what: "set actions"
        })
    );
}

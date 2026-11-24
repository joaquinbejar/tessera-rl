// SPDX-License-Identifier: MIT OR Apache-2.0

//! Conditional heads and entity groups for host environments (#385): the
//! host pool builds the same mask layout an `EnvPool` builds from
//! `ActionMask`, reports it in its layout, writes the environment's
//! conditional mask rows, and keeps validating child heads by range only.

#![cfg(feature = "host")]

use core::num::{NonZeroU16, NonZeroU32};

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EntityHeads, EnvConfig, HeadSpec, HostEnvironment, HostError, HostOut, HostSeed,
    SpaceError,
};
use tessera_rl_engine::{ActionInput, EngineError, HostPool, MaskHeadShape, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

/// Two entities, each with a `kind` head (2 choices) and a `target` head
/// (3 choices) that is its child.
const HEADS: EntityHeads<2> = EntityHeads::new(2, [HeadSpec::root(2), HeadSpec::child(3, 0)]);

struct Cfg;
impl EnvConfig for Cfg {}

/// Masks: entity 0 may only pick kind 1 and, under kind 1, target 2;
/// entity 1 everything.
struct Squad;

impl HostEnvironment for Squad {
    const ENV_ID: &'static str = "test/HostSquad-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    const MASKS: bool = true;
    type Config = Cfg;

    fn new(_: &Cfg) -> Result<Self, HostError> {
        Ok(Self)
    }
    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(HEADS.space())
    }
    fn head_parents(_: &Cfg, head: usize) -> [Option<u16>; 2] {
        HEADS.head_parents(head)
    }
    fn entity_group_heads(_: &Cfg) -> Option<NonZeroU16> {
        HEADS.entity_group_heads()
    }
    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
    fn mask(&self, _: AgentSlot, mask: &mut MaskWriter<'_>) {
        mask.fill(true);
        let (Some(kind0), Some(target)) = (HEADS.mask_row(0, 0, 0), HEADS.mask_row(0, 1, 1)) else {
            return;
        };
        mask.set(kind0, false);
        for t in 0..2 {
            mask.set(target + t, false);
        }
    }
}

fn pool(budget: u64) -> Result<HostPool<Squad, 1>, EngineError> {
    let mut options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    options.mask_budget_bytes = budget;
    HostPool::new(Cfg, options)
}

#[test]
fn test_host_masks_layout_has_parents_and_groups() {
    let p = pool(1 << 20).unwrap();
    let layout = p.layout();
    assert_eq!(layout.entity_group_heads, 2);
    assert_eq!(layout.mask_len, HEADS.entity_mask_len() * 2);
    assert_eq!(
        layout.mask_heads,
        vec![
            MaskHeadShape {
                offset: 0,
                parent_sizes: [None, None],
                size: 2
            },
            MaskHeadShape {
                offset: 2,
                parent_sizes: [Some(2), None],
                size: 3
            },
            MaskHeadShape {
                offset: 8,
                parent_sizes: [None, None],
                size: 2
            },
            MaskHeadShape {
                offset: 10,
                parent_sizes: [Some(2), None],
                size: 3
            },
        ]
    );
}

#[test]
fn test_host_masks_rows_are_the_environments_conditional_rows() {
    let mut p = pool(1 << 20).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let masks: Vec<bool> = p
        .outputs()
        .unwrap()
        .masks()
        .iter()
        .map(|b| b.get())
        .collect();
    let mut want = vec![true; HEADS.entity_mask_len() * 2];
    want[0] = false;
    want[2 + 3] = false;
    want[2 + 3 + 1] = false;
    assert_eq!(masks, want);
}

#[test]
fn test_host_masks_child_heads_validated_by_range_only() {
    let mut p = pool(1 << 20).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // Masked choices are still accepted (masks are outputs for hosts).
    assert!(
        p.step(ActionInput {
            discrete: &[0, 0, 1, 2],
            continuous: &[],
        })
        .is_ok()
    );
    // A target outside its range is refused.
    assert!(matches!(
        p.step(ActionInput {
            discrete: &[1, 3, 0, 0],
            continuous: &[],
        }),
        Err(EngineError::DiscreteOutOfRange { .. })
    ));
}

#[test]
fn test_host_masks_budget_counts_conditional_blocks() {
    let bytes = u64::try_from(HEADS.entity_mask_len() * 2).unwrap();
    assert!(pool(bytes).is_ok());
    assert!(matches!(
        pool(bytes - 1),
        Err(EngineError::MaskBudgetExceeded { .. })
    ));
}

/// Groups that do not divide the heads: the host pool reads the
/// environment's groups and refuses them as an `EnvPool` does.
struct Uneven;

impl HostEnvironment for Uneven {
    const ENV_ID: &'static str = "test/HostUneven-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    const MASKS: bool = true;
    type Config = Cfg;

    fn new(_: &Cfg) -> Result<Self, HostError> {
        Ok(Self)
    }
    fn observation_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn action_space(_: &Cfg) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: vec![2, 2, 2],
        })
    }
    fn entity_group_heads(_: &Cfg) -> Option<NonZeroU16> {
        NonZeroU16::new(2)
    }
    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}
    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}
    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

#[test]
fn test_host_masks_uneven_groups_refused() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    assert!(matches!(
        HostPool::<Uneven, 1>::new(Cfg, options),
        Err(EngineError::InvalidMaskLayout { .. })
    ));
}

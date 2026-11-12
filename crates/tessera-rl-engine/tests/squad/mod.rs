// SPDX-License-Identifier: MIT OR Apache-2.0

//! The entity-group fixture of ADR-0051, shared by `entity_groups.rs` and
//! the allocation test `entity_groups_alloc.rs`: each agent commands three
//! units through a `type` head and a `target` head, the child of `type`.
//! Illegal: `type` 2 for unit 1 while the step count is even; a `target`
//! above its `type`.

#![allow(dead_code, reason = "each test binary uses part of it")]
#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "test fixture")]

use core::num::{NonZeroU16, NonZeroU32};

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, HeaderView, InvariantViolation, NoAdaptive, ObsLeaf,
    Observation, Persist, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, IllegalActionPolicy, Masked, PoolOptions,
    Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

pub const UNITS: usize = 3;
pub const HEADS: usize = UNITS * 2;

#[derive(EnvConfig, Debug, Clone)]
pub struct SquadConfig {
    /// `noop` writes only unit 0's type: the agent's index + 1 mod 3, so
    /// agents differ and other heads stay as the array holds them.
    #[config(default = false)]
    pub partial_noop: bool,
    /// Adds a `Box` head beside the discrete ones, which ADR-0051 refuses.
    #[config(default = false)]
    pub with_box: bool,
    /// Groups with no parents: `target` is a root, its four choices legal
    /// except 3.
    #[config(default = false)]
    pub flat: bool,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct SquadState {
    /// `type * 10 + target` of each unit, each agent.
    pub last: [[i32; 3]; 2],
    pub steps: u32,
    pub pad: i32,
}

#[derive(Observation)]
pub struct SquadObs {
    #[space(box(low = -100, high = 100, shape = [3]))]
    pub last: ObsLeaf<i32>,
}

pub struct Squad;

impl Environment for Squad {
    type State = SquadState;
    type Config = SquadConfig;
    type Obs = SquadObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &SquadConfig) -> Result<Space, SpaceError> {
        SquadObs::space(cfg)
    }

    fn action_space(cfg: &SquadConfig) -> Result<Space, SpaceError> {
        let orders = Space::MultiDiscrete {
            nvec: [3, 4].repeat(UNITS),
        };
        if !cfg.with_box {
            return Ok(orders);
        }
        Ok(Space::Tuple(vec![
            orders,
            Space::Box {
                shape: vec![1],
                dtype: DType::F32,
                bounds: BoxBounds::Float {
                    low: vec![-1.0],
                    high: vec![1.0],
                },
            },
        ]))
    }

    fn reset(_: &SquadConfig, _: &mut StepRng, state: &mut SquadState) {
        *state = SquadState::default();
    }

    fn step(
        _: &SquadConfig,
        state: &mut SquadState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        for a in 0..2u16 {
            let agent = AgentSlot::new(a, 2).unwrap();
            for u in 0..UNITS {
                let t = actions.discrete(agent, u * 2).unwrap();
                let c = actions.discrete(agent, u * 2 + 1).unwrap();
                state.last[usize::from(a)][u] = t * 10 + c;
            }
        }
        state.steps += 1;
    }

    fn observe(
        _: &SquadConfig,
        state: &SquadState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, SquadObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(SquadObs::LAST) {
            for (i, v) in state.last[agent.index()].iter().enumerate() {
                leaf.set(i, *v);
            }
        }
    }
}

impl ActionMask for Squad {
    fn action_mask(cfg: &SquadConfig, state: &SquadState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        if cfg.flat {
            // Per unit: type row (3), then one target row (4).
            for u in 0..UNITS {
                let base = u * 7;
                for t in 0..3usize {
                    out.set(
                        base + t,
                        !(u == 1 && t == 2 && state.steps.is_multiple_of(2)),
                    );
                }
                for c in 0..4usize {
                    out.set(base + 3 + c, c != 3);
                }
            }
            return;
        }
        // Per unit: type row (3), then four target choices per type (12).
        for u in 0..UNITS {
            let base = u * 15;
            for t in 0..3usize {
                let forbidden = u == 1 && t == 2 && state.steps.is_multiple_of(2);
                out.set(base + t, !forbidden);
                for c in 0..4usize {
                    out.set(base + 3 + t * 4 + c, c <= t);
                }
            }
        }
    }

    fn noop(cfg: &SquadConfig, _: &SquadState, agent: AgentSlot, out: &mut ActionWriter<'_>) {
        if cfg.partial_noop {
            out.set_discrete(0, i32::try_from(agent.index()).unwrap());
        }
        // Otherwise every head stays zero: type 0, target 0, legal.
    }

    fn head_parents(cfg: &SquadConfig, head: usize) -> [Option<u16>; 2] {
        if head % 2 == 1 && !cfg.flat {
            [Some(u16::try_from(head - 1).unwrap()), None]
        } else {
            [None, None]
        }
    }

    fn entity_group_heads(_: &SquadConfig) -> Option<NonZeroU16> {
        NonZeroU16::new(2)
    }
}

impl Persist for Squad {
    const ENV_ID: &'static str = "tessera/Squad-v0";
    const ENV_VERSION: u32 = 1;
}

impl Validate for Squad {
    fn validate(_: &SquadConfig, _: &SquadState) -> Result<(), InvariantViolation> {
        Ok(())
    }

    fn validate_record(
        _: &SquadConfig,
        _: &HeaderView<'_>,
        _: &SquadState,
    ) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

pub type Pool = EnvPool<Squad, Masked>;

pub fn options(n: u32, policy: IllegalActionPolicy) -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: policy,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(3))
    }
}

pub fn cfg(partial_noop: bool) -> SquadConfig {
    SquadConfig {
        partial_noop,
        with_box: false,
        flat: false,
    }
}

pub fn pool(policy: IllegalActionPolicy) -> Pool {
    let mut p = EnvPool::with_masks(cfg(false), options(1, policy)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

pub fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

/// A row for two agents, both sending `first`.
pub fn both(first: [i32; HEADS]) -> Vec<i32> {
    [first, first].concat()
}

pub fn illegal_count(p: &Pool) -> u32 {
    let record = DecodedRecord::from(p.snapshot(0).unwrap());
    record.header().agents[0].illegal_actions
}

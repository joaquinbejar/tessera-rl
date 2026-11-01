// SPDX-License-Identifier: MIT OR Apache-2.0

//! A failed scratch step (a contained kernel panic) poisons the scratch
//! pool: the expansion fails with the tree unchanged, and the next valid
//! expansion succeeds (R38 recovery).

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut,
    Tier, TurnStructure,
};
use tessera_rl_engine::{EngineError, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_search::{NodeHandle, SearchError, SearchOptions, SearchPool};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Count {
    total: u32,
    to_move: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct CountConfig {}

#[derive(Observation)]
struct CountObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    total: ObsLeaf<i32>,
}

/// Two players add 0 or 1 in turns; action 2 panics in the kernel.
struct Fragile;

impl Environment for Fragile {
    type State = Count;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn acting(_: &CountConfig, state: &Count, agent: AgentSlot) -> bool {
        agent.index() == state.to_move as usize
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Count) {
        *state = Count::default();
    }

    fn step(
        _: &CountConfig,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        let me = AgentSlot::new(u16::try_from(state.to_move).unwrap_or(0), 2)
            .unwrap_or(AgentSlot::FIRST);
        let a = actions.discrete(me, 0).unwrap_or(0);
        assert!(a != 2, "fixture: action 2 panics");
        state.total += u32::try_from(a).unwrap_or(0);
        state.to_move ^= 1;
    }

    fn observe(
        _: &CountConfig,
        state: &Count,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(CountObs::TOTAL) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
        }
    }
}

impl tessera_rl_core::Validate for Fragile {
    fn validate(_: &CountConfig, _: &Count) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}

#[test]
fn test_search_pool_recovers_the_scratch_pool_after_a_kernel_panic() {
    let mut p = EnvPool::<Fragile>::new(
        CountConfig {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut s = SearchPool::<Fragile, NoMasks>::new(
        CountConfig {},
        SearchOptions::new(NonZeroU32::new(8).unwrap(), NonZeroU32::new(2).unwrap()),
        &p.snapshot(0).unwrap(),
        EnvPool::new,
    )
    .unwrap();
    let root = s.root();
    let before = s.tree_hash();
    let mut out = [root; 2];
    let err = s.expand(&[(root, 1), (root, 2)], &mut out).unwrap_err();
    assert_eq!(err, SearchError::Engine(EngineError::Panicked));
    assert_eq!(
        s.tree_hash(),
        before,
        "the failed expansion changed the tree"
    );
    // The next valid expansion works, and gives the children a fresh pool
    // would give.
    let mut good: [NodeHandle; 2] = [root; 2];
    s.expand(&[(root, 0), (root, 1)], &mut good).unwrap();
    assert_eq!(s.record(good[0]).unwrap().state().total, 0);
    assert_eq!(s.record(good[1]).unwrap().state().total, 1);
    assert_eq!(s.record(good[1]).unwrap().state().to_move, 1);
}

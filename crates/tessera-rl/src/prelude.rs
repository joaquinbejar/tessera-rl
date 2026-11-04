// SPDX-License-Identifier: MIT OR Apache-2.0

//! The names an environment author and a pool user need, in one import
//! (SPEC 4.1 and 10): `use tessera_rl::prelude::*;`.
//!
//! The derive macros share their names with the traits they implement
//! (`EnvState`, `EnvConfig`, `Observation`, `StateEnum`); both are
//! exported, in their separate namespaces.

pub use crate::core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
pub use crate::core::space::{BoxBounds, DType, Space};
pub use crate::core::{
    ActionMask, AgentSlot, Bool8, Determinize, EnvAssets, EnvConfig, EnvState, Environment,
    EpisodeEnd, Heuristic, ImperfectInformation, InvariantViolation, LeafId, NoAdaptive, ObsLeaf,
    Observation, Pool, RecordHeader, Render, Reserved, SpaceError, StateEnum, StateHash, StepOut,
    Tier, TurnStructure, Validate,
};
pub use crate::engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, NoMasks, PoolContext,
    PoolOptions, Record, Reseed, SyncOptions,
};
pub use crate::macros::{EnvConfig, EnvState, Observation, StateEnum};
pub use crate::num::identity::Seed;
pub use crate::num::rng::{ObsRng, StepRng};
pub use crate::num::{Fx32, Fx64};

#[cfg(test)]
mod tests {
    //! An environment written with the prelude alone, as SPEC 10 shows it.

    use super::*;

    #[derive(EnvConfig, Debug, Clone)]
    struct WalkConfig {
        #[config(default = 8, range = 1..=100)]
        length: u32,
    }

    #[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
    #[repr(C)]
    struct WalkState {
        pos: Fx32,
        steps: u32,
        done: Bool8,
        reserved: Reserved<3>,
    }

    #[derive(Observation)]
    struct WalkObs {
        #[space(box(low = -1000, high = 1000, shape = [1]))]
        steps: ObsLeaf<i32>,
    }

    struct Walk;

    impl Environment for Walk {
        type State = WalkState;
        type Config = WalkConfig;
        type Obs = WalkObs;
        type Action = ();
        type Header = RecordHeader<1, NoAdaptive>;

        const MAX_AGENTS: usize = 1;
        const TIER: Tier = Tier::CrossPlatform;

        fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
            WalkObs::space(cfg)
        }

        fn action_space(_: &WalkConfig) -> Result<Space, SpaceError> {
            Ok(Space::Discrete { n: 2 })
        }

        fn reset(_: &WalkConfig, _: &mut StepRng, state: &mut WalkState) {
            *state = WalkState::default();
        }

        fn step(
            cfg: &WalkConfig,
            state: &mut WalkState,
            actions: &Actions<'_, ()>,
            _: &mut StepRng,
            out: &mut StepOut<'_>,
        ) {
            let right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
            state.steps += 1;
            out.set_reward(AgentSlot::FIRST, if right { 1.0 } else { 0.0 });
            if state.steps >= cfg.length {
                state.done = Bool8::TRUE;
                out.end_episode(EpisodeEnd::Terminated);
            }
        }

        fn observe(
            _: &WalkConfig,
            state: &WalkState,
            _: AgentSlot,
            _: &mut ObsRng,
            out: &mut ObsWriter<'_, '_, WalkObs>,
        ) {
            if let Some(mut leaf) = out.leaf_mut(WalkObs::STEPS) {
                leaf.set(0, i32::try_from(state.steps).unwrap_or(i32::MAX));
            }
        }
    }

    #[test]
    fn test_prelude_authors_and_runs_an_environment() {
        let options = PoolOptions::new(core::num::NonZeroU32::MIN, Seed::new(1));
        let Ok(mut pool) = EnvPool::<Walk, NoMasks>::new(WalkConfig::default(), options) else {
            panic!("pool construction failed");
        };
        assert!(pool.reset_all(Reseed::Keep).is_ok());
        let step = pool.step(ActionInput {
            discrete: &[1],
            continuous: &[],
        });
        assert_eq!(step.map(|out| out.rewards().to_vec()), Ok(vec![1.0]));
    }

    #[test]
    fn test_prelude_debug_checks_reaches_the_engine() {
        assert_eq!(crate::engine::DEBUG_CHECKS, cfg!(feature = "debug-checks"));
    }
}

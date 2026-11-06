// SPDX-License-Identifier: MIT OR Apache-2.0

//! A masked single-agent environment keeps its legality pass on the
//! single-discrete action copy: an illegal action is replaced by the no-op and
//! counted, as with any other shape.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Total {
    sum: i32,
}

#[derive(EnvConfig, Debug, Clone)]
struct TotalConfig {}

#[derive(Observation)]
struct TotalObs {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [1]))]
    sum: ObsLeaf<i32>,
}

/// One agent, one discrete head of 3: action 2 is masked out, the no-op is 0.
struct Adder;

impl Environment for Adder {
    type State = Total;
    type Config = TotalConfig;
    type Obs = TotalObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &TotalConfig) -> Result<Space, SpaceError> {
        TotalObs::space(cfg)
    }

    fn action_space(_: &TotalConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(_: &TotalConfig, _: &mut StepRng, state: &mut Total) {
        state.sum = 0;
    }

    fn step(
        _: &TotalConfig,
        state: &mut Total,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let a = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(-100);
        state.sum += a;
        #[expect(clippy::cast_precision_loss, reason = "small test values")]
        out.set_reward(AgentSlot::FIRST, a as f32);
    }

    fn observe(
        _: &TotalConfig,
        state: &Total,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, TotalObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(TotalObs::SUM) {
            leaf.set(0, state.sum);
        }
    }
}

impl ActionMask for Adder {
    fn action_mask(_: &TotalConfig, _: &Total, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(true);
        out.set(2, false);
    }

    fn noop(_: &TotalConfig, _: &Total, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

fn pool() -> EnvPool<Adder, Masked> {
    let options = PoolOptions {
        illegal_action: IllegalActionPolicy::Noop,
        auto_reset: AutoReset::NextStep,
        ..PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(3))
    };
    let mut p = EnvPool::with_masks(TotalConfig {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

#[test]
fn test_single_action_masked_pool_replaces_illegal_actions_by_the_noop() {
    let mut p = pool();
    let rewards = p
        .step(ActionInput {
            discrete: &[1, 2, 0, 2],
            continuous: &[],
        })
        .unwrap()
        .rewards()
        .to_vec();
    // The masked action 2 runs as the no-op 0, and is counted.
    assert_eq!(rewards, [1.0, 0.0, 0.0, 0.0]);
    let counts: Vec<u32> = (0..4)
        .map(|i| p.header(i).unwrap().agents[0].illegal_actions)
        .collect();
    assert_eq!(counts, [0, 1, 0, 1]);
}

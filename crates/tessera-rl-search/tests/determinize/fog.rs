// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fog: a three-agent imperfect-information test game with deaths and a
//! respawn, in a `Simultaneous` and a `Mixed` variant, for the `alive`,
//! `generation` and `acting` rows of a determinized record.
//!
//! - Each agent holds a hidden coin in `0..COINS`, dealt at reset.
//! - Every step rolls a public die. Slot 2 dies at step 1 and respawns at
//!   step 2 (generation 1); slot 1 dies at step 3. The episode ends at
//!   step `STEPS`.
//! - `acting`: alive agents whose coin plus the step is even. `Mixed`
//!   reads it; `Simultaneous` ignores it (every alive agent acts), so the
//!   two variants tell the turn rule apart.
//! - Information state: the observer's slot and coin, the step, the die
//!   and who is alive. The other coins are hidden.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Determinize, Environment, EpisodeEnd, ImperfectInformation,
    InvariantViolation, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut, Tier,
    TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

pub const COINS: u32 = 4;
pub const STEPS: u32 = 6;

#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct FogConfig {}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct FogState {
    pub coins: [u32; 3],
    pub life: [u32; 3],
    pub t: u32,
    pub die: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct FogInfo {
    pub me: u32,
    pub coin: u32,
    pub t: u32,
    pub die: u32,
    pub life: [u32; 3],
}

#[derive(Observation)]
pub struct FogObs {
    /// Own coin, step, die.
    #[space(box(low = 0, high = 100, dtype = u32, shape = [3]))]
    pub view: ObsLeaf<u32>,
}

/// Fog; `MIXED` selects the turn structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fog<const MIXED: bool>;

fn bound(n: u32) -> core::num::NonZeroU32 {
    core::num::NonZeroU32::new(n).unwrap_or(core::num::NonZeroU32::MIN)
}

fn slot(i: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(i).ok()?, 3)
}

impl<const MIXED: bool> Environment for Fog<MIXED> {
    type State = FogState;
    type Config = FogConfig;
    type Obs = FogObs;
    type Action = ();
    type Header = RecordHeader<3, NoAdaptive>;

    const MAX_AGENTS: usize = 3;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = false;
    const TURNS: TurnStructure = if MIXED {
        TurnStructure::Mixed
    } else {
        TurnStructure::Simultaneous
    };

    fn observation_space(cfg: &FogConfig) -> Result<Space, SpaceError> {
        FogObs::space(cfg)
    }

    fn action_space(_: &FogConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn acting(_: &FogConfig, s: &FogState, agent: AgentSlot) -> bool {
        let k = agent.index();
        s.life.get(k) == Some(&1) && (s.coins.get(k).copied().unwrap_or(0) + s.t).is_multiple_of(2)
    }

    fn reset(_: &FogConfig, rng: &mut StepRng, state: &mut FogState) {
        let mut coins = [0; 3];
        for c in &mut coins {
            *c = rng.below_u32(bound(COINS));
        }
        *state = FogState {
            coins,
            life: [1; 3],
            t: 0,
            die: 0,
        };
    }

    fn step(
        _: &FogConfig,
        state: &mut FogState,
        _: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.die = rng.below_u32(bound(6));
        state.t += 1;
        let (Some(one), Some(two)) = (slot(1), slot(2)) else {
            return;
        };
        match state.t {
            1 => {
                state.life[2] = 0;
                out.agent_died(two);
            }
            2 => {
                state.life[2] = 1;
                out.agent_spawned(two);
            }
            3 => {
                state.life[1] = 0;
                out.agent_died(one);
            }
            _ => {}
        }
        if state.t >= STEPS {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &FogConfig,
        state: &FogState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, FogObs>,
    ) {
        let coin = state.coins.get(agent.index()).copied().unwrap_or(0);
        let Some(mut leaf) = out.leaf_mut(FogObs::VIEW) else {
            return;
        };
        for (k, v) in [coin, state.t, state.die].into_iter().enumerate() {
            leaf.set(k, v);
        }
    }
}

impl<const MIXED: bool> ActionMask for Fog<MIXED> {
    fn action_mask(_: &FogConfig, _: &FogState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.set(0, true);
        out.set(1, true);
    }

    fn noop(_: &FogConfig, _: &FogState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

impl<const MIXED: bool> Validate for Fog<MIXED> {
    fn validate(_: &FogConfig, s: &FogState) -> Result<(), InvariantViolation> {
        if s.coins.iter().any(|&c| c >= COINS) || s.life.iter().any(|&l| l > 1) || s.die >= 6 {
            return Err(InvariantViolation {
                code: "a value out of range",
            });
        }
        Ok(())
    }
}

impl<const MIXED: bool> ImperfectInformation for Fog<MIXED> {
    type InfoState = FogInfo;

    fn info_state(_: &FogConfig, s: &FogState, agent: AgentSlot) -> FogInfo {
        let me = agent.index().min(2);
        FogInfo {
            me: u32::try_from(me).unwrap_or(0),
            coin: s.coins.get(me).copied().unwrap_or(0),
            t: s.t,
            die: s.die,
            life: s.life,
        }
    }
}

impl<const MIXED: bool> Determinize for Fog<MIXED> {
    fn sample(_: &FogConfig, info: &FogInfo, rng: &mut StepRng, state: &mut FogState) {
        let me = (info.me as usize).min(2);
        let mut coins = [0; 3];
        for (k, c) in coins.iter_mut().enumerate() {
            *c = if k == me {
                info.coin
            } else {
                rng.below_u32(bound(COINS))
            };
        }
        *state = FogState {
            coins,
            life: info.life,
            t: info.t,
            die: info.die,
        };
    }
}

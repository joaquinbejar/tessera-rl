// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every engine entry point runs in the canonical FP environment, even
//! when the caller's thread flushes subnormals (ADR-0037).

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Bool8, NoAdaptive, RecordHeader, Reserved, SpaceError, StepOut, Tier,
};
use tessera_rl_engine::{ActionInput, EngineError, EnvPool, PoolOptions, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::fp;
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, Default)]
#[repr(C)]
struct Last {
    action: f32,
    alive: Bool8,
    reserved: Reserved<3>,
}

#[derive(EnvConfig, Debug, Clone)]
struct ContConfig {
    #[config(default = false)]
    f64_actions: bool,
    #[config(default = false)]
    kill_each_step: bool,
}

struct Cont;

impl tessera_rl_core::Environment for Cont {
    type State = Last;
    type Config = ContConfig;
    type Obs = ();
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    /// Fails if construction does not run in the canonical environment.
    fn observation_space(_: &ContConfig) -> Result<Space, SpaceError> {
        if !fp::is_canonical() {
            return Err(SpaceError::Empty);
        }
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(cfg: &ContConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![1],
            dtype: if cfg.f64_actions {
                DType::F64
            } else {
                DType::F32
            },
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![0.0],
            },
        })
    }

    fn reset(_: &ContConfig, _: &mut StepRng, state: &mut Last) {
        state.alive = Bool8::TRUE;
    }

    fn step(
        cfg: &ContConfig,
        state: &mut Last,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.action = actions.continuous(AgentSlot::FIRST, 0).unwrap_or(f32::NAN);
        if cfg.kill_each_step {
            out.agent_died(AgentSlot::FIRST);
        }
    }

    fn observe(
        _: &ContConfig,
        _: &Last,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, ()>,
    ) {
    }
}

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::MIN, Seed::new(1))
}

#[test]
fn test_construction_runs_canonical_under_hostile_ftz() {
    let pool =
        fp::with_hostile_flush_to_zero(|| EnvPool::<Cont>::new(ContConfig::default(), options()));
    assert!(pool.is_ok());
}

#[test]
fn test_subnormal_action_is_validated_exactly_under_hostile_ftz() {
    let mut pool = EnvPool::<Cont>::new(ContConfig::default(), options()).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let tiny = [f32::from_bits(1)];
    let zero = [0.0f32];
    fp::with_hostile_flush_to_zero(|| {
        // 1.4e-45 is outside [0, 0]: a flushed comparison would accept it.
        let err = pool
            .step(ActionInput {
                discrete: &[],
                continuous: &tiny,
            })
            .map(|_| ());
        assert!(matches!(err, Err(EngineError::ContinuousOutOfRange { .. })));
        // The early return restored the caller's (hostile) environment.
        assert!(!fp::is_canonical());
        assert!(
            pool.step(ActionInput {
                discrete: &[],
                continuous: &zero
            })
            .is_ok()
        );
        assert!(!fp::is_canonical());
    });
    assert_eq!(pool.state(0).unwrap().action.to_bits(), 0);
}

#[test]
fn test_reset_runs_canonical_and_restores_under_hostile_ftz() {
    let mut pool = EnvPool::<Cont>::new(ContConfig::default(), options()).unwrap();
    fp::with_hostile_flush_to_zero(|| {
        assert!(pool.reset_all(Reseed::Keep).is_ok());
        assert!(pool.reset(&[3], Reseed::Keep).is_err());
        assert!(!fp::is_canonical());
    });
    assert!(fp::is_canonical());
}

#[test]
fn test_f64_box_actions_are_unsupported_in_phase_1a() {
    let cfg = ContConfig {
        f64_actions: true,
        ..ContConfig::default()
    };
    assert_eq!(
        EnvPool::<Cont>::new(cfg, options()).map(|_| ()),
        Err(EngineError::UnsupportedSpace {
            what: "f64 box actions"
        })
    );
}

/// In release builds (where the debug assertion is off) a second death of
/// the same slot is ignored: `terminated` is reported exactly once.
#[cfg(not(debug_assertions))]
#[test]
fn test_agent_died_on_dead_slot_reports_terminated_once() {
    let cfg = ContConfig {
        kill_each_step: true,
        ..ContConfig::default()
    };
    let mut pool = EnvPool::<Cont>::new(cfg, options()).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let zero = [0.0f32];
    let out = pool
        .step(ActionInput {
            discrete: &[],
            continuous: &zero,
        })
        .unwrap();
    assert_eq!(out.agent_terminated(), [Bool8::TRUE]);
    let out = pool
        .step(ActionInput {
            discrete: &[],
            continuous: &zero,
        })
        .unwrap();
    assert_eq!(out.agent_terminated(), [Bool8::FALSE]);
}

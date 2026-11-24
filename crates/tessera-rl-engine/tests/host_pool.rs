// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host pool (ADR-0062): construction, seeds (ADR-0007), the
//! validation boundary, auto-reset, panics and poisoning, masks, Sync mode
//! against Inline mode, and the floating-point controls after host code.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod host_support;

use core::num::{NonZeroU32, NonZeroUsize};

use host_support::{AGENTS, Book, BookConfig, LEVELS, SIDES};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, HostPool, MAX_HOST_HEADS, Outputs, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

type Pool = HostPool<Book, { Book::MAX_AGENTS }>;

fn options(n: u32, seed: u64) -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
}

fn pool(n: u32, seed: u64, cfg: BookConfig) -> Pool {
    Pool::new(cfg, options(n, seed)).unwrap()
}

/// The observation of agent `slot` of instance `i`:
/// `[position, orders, last price, step]`.
fn obs(out: &Outputs, i: usize, slot: usize) -> [i32; 4] {
    let leaf = out.observation(0).unwrap();
    let row = &leaf[(i * AGENTS + slot) * 16..(i * AGENTS + slot + 1) * 16];
    let mut v = [0i32; 4];
    for (x, b) in v.iter_mut().zip(row.as_chunks::<4>().0) {
        *x = i32::from_le_bytes(*b);
    }
    v
}

/// Every agent of every instance buys at level `level`.
fn buy(n: usize, level: i32) -> Vec<i32> {
    [1, level].repeat(n * AGENTS)
}

fn step(p: &mut Pool, actions: &[i32]) -> Result<(), EngineError> {
    p.step(ActionInput {
        discrete: actions,
        continuous: &[],
    })
    .map(|_| ())
}

/// Every output a host pool writes, for whole-call comparisons.
fn snapshot(out: &Outputs) -> Vec<Vec<u8>> {
    let flags = |f: &[Bool8]| f.iter().map(|b| u8::from(b.get())).collect::<Vec<u8>>();
    let floats = |f: &[f32]| f.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    vec![
        out.observation(0).unwrap().to_vec(),
        floats(out.rewards()),
        flags(out.agent_terminated()),
        flags(out.agent_truncated()),
        flags(out.alive()),
        flags(out.acting()),
        flags(out.episode_terminated()),
        flags(out.episode_truncated()),
        floats(out.episode_return()),
        floats(out.episode_length()),
        flags(out.masks()),
        out.generations()
            .iter()
            .flat_map(|g| g.to_le_bytes())
            .collect(),
    ]
}

#[test]
fn test_host_pool_new_refused_construction_maps_to_host_construction() {
    let cfg = BookConfig {
        refuse: true,
        ..BookConfig::default()
    };
    assert_eq!(
        Pool::new(cfg, options(2, 0)).err(),
        Some(EngineError::HostConstruction {
            reason: "the book refused".to_owned()
        })
    );
}

#[test]
fn test_host_pool_action_heads_and_mask_layout() {
    let mut p = pool(1, 0, BookConfig::default());
    assert_eq!(p.action_heads(), (2, 0));
    let out = p.reset_all(Reseed::Keep).unwrap();
    // One block per head: 3 sides, then 4 levels; selling is masked with no
    // position.
    let m: Vec<bool> = out.masks()[..7].iter().map(|b| b.get()).collect();
    assert_eq!(m, [true, true, false, true, true, true, true]);
    step(&mut p, &buy(1, 0)).unwrap();
    assert!(
        p.outputs().unwrap().masks()[2].get(),
        "a position allows selling"
    );
}

/// A host environment with `N` binary action heads per agent.
struct Wide<const N: u32>;

impl<const N: u32> HostEnvironment for Wide<N> {
    const ENV_ID: &'static str = "test/Wide-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = WideConfig;

    fn new(_: &WideConfig) -> Result<Self, HostError> {
        Ok(Self)
    }

    fn observation_space(_: &WideConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &WideConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiBinary { shape: vec![N] })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

struct WideConfig;

impl EnvConfig for WideConfig {}

#[test]
fn test_host_pool_new_head_count_bounded_by_max_host_heads() {
    const AT: u32 = 4096;
    const PAST: u32 = AT + 1;
    assert_eq!(usize::try_from(AT).unwrap(), MAX_HOST_HEADS);
    let at = HostPool::<Wide<AT>, 1>::new(WideConfig, options(1, 0)).unwrap();
    assert_eq!(at.action_heads(), (MAX_HOST_HEADS, 0));
    assert_eq!(
        HostPool::<Wide<PAST>, 1>::new(WideConfig, options(1, 0)).err(),
        Some(EngineError::UnsupportedSpace {
            what: "more than MAX_HOST_HEADS action heads per agent"
        })
    );
}

/// A host environment that declares a `Set` observation (ADR-0082).
struct Setty;

impl HostEnvironment for Setty {
    const ENV_ID: &'static str = "test/Setty-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    type Config = WideConfig;

    fn new(_: &WideConfig) -> Result<Self, HostError> {
        Ok(Self)
    }

    fn observation_space(_: &WideConfig) -> Result<Space, SpaceError> {
        Ok(Space::Dict(vec![
            ("x".to_owned(), Space::Discrete { n: 2 }),
            (
                "s".to_owned(),
                Space::Set {
                    inner: Box::new(Space::Discrete { n: 3 }),
                    max: 4,
                },
            ),
        ]))
    }

    fn action_space(_: &WideConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

#[test]
fn test_host_pool_new_set_observation_is_refused_at_construction() {
    assert_eq!(
        HostPool::<Setty, 1>::new(WideConfig, options(1, 0)).err(),
        Some(EngineError::UnsupportedSpace {
            what: "set observations in host protocol 1"
        })
    );
}

#[test]
fn test_host_pool_seeds_follow_adr_0007() {
    let mut p = pool(3, 40, BookConfig::default());
    // Construction: instance i gets pool_seed + i; the opening price is
    // seed % 97 + 1000 * episode.
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!([0, 1, 2].map(|i| obs(out, i, 0)[2]), [40, 41, 42]);
    // Keeping the seed advances the episode index.
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!([0, 1, 2].map(|i| obs(out, i, 0)[2]), [1040, 1041, 1042]);
    // reset(seed = s): instance i gets s + i, episode 0.
    let out = p.reset_all(Reseed::Base(Seed::new(90))).unwrap();
    assert_eq!([0, 1, 2].map(|i| obs(out, i, 0)[2]), [90, 91, 92]);
    // A list: instance i gets s_i; ids reseed only those instances.
    let seeds = [Seed::new(5), Seed::new(6)];
    let out = p.reset(&[2, 0], Reseed::List(&seeds)).unwrap();
    assert_eq!([0, 1, 2].map(|i| obs(out, i, 0)[2]), [6, 91, 5]);
}

#[test]
fn test_host_pool_invalid_actions_mutate_nothing() {
    let mut p = pool(2, 3, BookConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(2, 1)).unwrap();
    let before = snapshot(p.outputs().unwrap());
    // Wrong length, then a side out of range.
    assert!(step(&mut p, &[1, 1]).is_err());
    let mut bad = buy(2, 1);
    bad[0] = i32::try_from(SIDES).unwrap();
    assert!(step(&mut p, &bad).is_err());
    let mut bad = buy(2, 1);
    bad[1] = i32::try_from(LEVELS).unwrap();
    assert!(step(&mut p, &bad).is_err());
    assert_eq!(snapshot(p.outputs().unwrap()), before);
    // The next valid step continues as if the refused calls never happened.
    let mut q = pool(2, 3, BookConfig::default());
    q.reset_all(Reseed::Keep).unwrap();
    step(&mut q, &buy(2, 1)).unwrap();
    step(&mut p, &buy(2, 2)).unwrap();
    step(&mut q, &buy(2, 2)).unwrap();
    assert_eq!(
        snapshot(p.outputs().unwrap()),
        snapshot(q.outputs().unwrap())
    );
}

fn with_auto_reset(mode: AutoReset) -> Pool {
    let mut o = options(1, 7);
    o.auto_reset = mode;
    Pool::new(
        BookConfig {
            episode: 2,
            ..BookConfig::default()
        },
        o,
    )
    .unwrap()
}

#[test]
fn test_host_pool_auto_reset_next_step() {
    let mut p = with_auto_reset(AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    let out = p.outputs().unwrap();
    assert!(out.episode_terminated()[0].get());
    assert_eq!(out.episode_length()[0], 2.0);
    assert_eq!(obs(out, 0, 0), [2, 4, 7 + 4, 2], "the final observation");
    // The next step resets instead of stepping: episode 1, step 0.
    step(&mut p, &buy(1, 1)).unwrap();
    let out = p.outputs().unwrap();
    assert!(!out.episode_terminated()[0].get());
    assert_eq!(obs(out, 0, 0), [0, 0, 1007, 0]);
}

#[test]
fn test_host_pool_auto_reset_same_step_saves_terminal_observation() {
    let mut p = with_auto_reset(AutoReset::SameStep);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    let out = p.outputs().unwrap();
    assert!(out.episode_terminated()[0].get());
    let term = out.terminal_observation(0).unwrap();
    assert_eq!(term[..16], obs_bytes([2, 4, 11, 2]));
    assert_eq!(obs(out, 0, 0), [0, 0, 1007, 0], "the next episode's start");
}

fn obs_bytes(v: [i32; 4]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn test_host_pool_auto_reset_disabled_refuses_ended_instance() {
    let mut p = with_auto_reset(AutoReset::Disabled);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    assert!(matches!(
        step(&mut p, &buy(1, 1)),
        Err(EngineError::PendingReset { .. })
    ));
    p.reset(&[0], Reseed::Keep).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
}

#[test]
fn test_host_pool_panic_poisons_until_full_reset() {
    let mut p = pool(
        2,
        0,
        BookConfig {
            panic_at: 2,
            ..BookConfig::default()
        },
    );
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(2, 0)).unwrap();
    assert_eq!(step(&mut p, &buy(2, 0)), Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    assert_eq!(step(&mut p, &buy(2, 0)), Err(EngineError::Poisoned));
    assert!(p.outputs().is_err());
    assert!(p.reset(&[0], Reseed::Keep).is_err());
    p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
    step(&mut p, &buy(2, 0)).unwrap();
}

#[test]
fn test_host_pool_same_seeds_and_actions_give_same_outputs() {
    let run = || {
        let mut p = pool(4, 11, BookConfig::default());
        p.reset_all(Reseed::Keep).unwrap();
        let mut trace = vec![snapshot(p.outputs().unwrap())];
        for t in 0..12 {
            let actions: Vec<i32> = (0..4 * AGENTS)
                .flat_map(|k| {
                    [
                        i32::try_from((k + t) % 3).unwrap(),
                        i32::try_from(t % 4).unwrap(),
                    ]
                })
                .collect();
            step(&mut p, &actions).unwrap();
            trace.push(snapshot(p.outputs().unwrap()));
        }
        trace
    };
    assert_eq!(run(), run());
}

#[test]
fn test_host_pool_sync_workers_match_inline_bit_for_bit() {
    let trace = |workers: Option<usize>, chunk: u32| {
        let mut o = options(7, 2);
        o.chunk_size = NonZeroU32::new(chunk).unwrap();
        o.auto_reset = AutoReset::SameStep;
        let cfg = BookConfig {
            episode: 3,
            ..BookConfig::default()
        };
        let mut p = Pool::new(cfg, o).unwrap();
        if let Some(w) = workers {
            p = p
                .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
                .unwrap();
        }
        p.reset_all(Reseed::Keep).unwrap();
        let mut out = vec![snapshot(p.outputs().unwrap())];
        for t in 0..9 {
            let actions: Vec<i32> = (0..7 * AGENTS)
                .flat_map(|k| {
                    [
                        i32::try_from((k * 7 + t) % 3).unwrap(),
                        i32::try_from(t % 4).unwrap(),
                    ]
                })
                .collect();
            step(&mut p, &actions).unwrap();
            out.push(snapshot(p.outputs().unwrap()));
            out.push(vec![
                p.outputs()
                    .unwrap()
                    .terminal_observation(0)
                    .unwrap()
                    .to_vec(),
            ]);
        }
        out
    };
    let inline = trace(None, 2);
    assert_eq!(trace(Some(1), 2), inline);
    assert_eq!(trace(Some(3), 2), inline);
    assert_eq!(trace(Some(2), 1), inline);
}

#[test]
fn test_host_pool_host_fp_changes_are_undone_after_each_chunk() {
    let cfg = BookConfig {
        hostile_fp: true,
        ..BookConfig::default()
    };
    // Inline: the host code runs on this thread.
    let mut p = pool(2, 0, cfg);
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.host_fp_resets(), 0);
    step(&mut p, &buy(2, 0)).unwrap();
    assert_eq!(p.host_fp_resets(), 2, "after each of the two host steps");
    assert!(tessera_rl_num::fp::is_canonical());
    // Sync: two workers, chunks of one instance: one per host step.
    let mut o = options(4, 0);
    o.chunk_size = NonZeroU32::MIN;
    let mut p = Pool::new(cfg, o)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(4, 0)).unwrap();
    assert_eq!(p.host_fp_resets(), 4);
    assert!(tessera_rl_num::fp::is_canonical());
}

/// The terminal step's rewards and flags survive every auto-reset mode: a
/// `SameStep` reset writes scratch rows; a `NextStep` reset's rewards are
/// the outputs of the call after; `Disabled` never resets by itself.
#[test]
fn test_host_pool_terminal_rewards_kept_in_every_auto_reset_mode() {
    use host_support::RESET_REWARD;
    for mode in [
        AutoReset::NextStep,
        AutoReset::SameStep,
        AutoReset::Disabled,
    ] {
        let mut p = with_auto_reset(mode);
        let out = p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(
            out.rewards(),
            [RESET_REWARD; AGENTS],
            "{mode:?}: reset rewards"
        );
        step(&mut p, &buy(1, 1)).unwrap();
        step(&mut p, &buy(1, 1)).unwrap();
        let out = p.outputs().unwrap();
        // Buying at level 1 costs 1: the ended step's rewards and flags.
        assert_eq!(out.rewards(), [-1.0; AGENTS], "{mode:?}: terminal rewards");
        assert!(out.episode_terminated()[0].get(), "{mode:?}");
        assert!(out.agent_terminated().iter().all(|f| f.get()), "{mode:?}");
        assert_eq!(
            out.episode_return()[0],
            -4.0,
            "{mode:?}: two agents, two steps"
        );
        match mode {
            AutoReset::NextStep => {
                // The reset runs in the next call: its rewards are that call's.
                step(&mut p, &buy(1, 1)).unwrap();
                assert_eq!(p.outputs().unwrap().rewards(), [RESET_REWARD; AGENTS]);
            }
            AutoReset::Disabled => {
                let out = p.reset(&[0], Reseed::Keep).unwrap();
                assert_eq!(out.rewards(), [RESET_REWARD; AGENTS]);
            }
            AutoReset::SameStep => {}
        }
    }
}

#[cfg(feature = "debug-checks")]
#[test]
fn test_host_pool_nan_reward_poisons_with_instance_and_agent() {
    let cfg = BookConfig {
        nan_reward_at: 2,
        ..BookConfig::default()
    };
    let mut p = pool(3, 0, cfg);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(3, 0)).unwrap();
    assert_eq!(
        step(&mut p, &buy(3, 0)),
        Err(EngineError::NanDetected {
            instance: 0,
            site: tessera_rl_engine::NanSite::Reward { agent: 0 },
        })
    );
    assert!(p.is_poisoned());
    p.reset_all(Reseed::Keep).unwrap();
}

/// A `Sequential` host environment: agent 0 acts, then agent 1, unless
/// `both` (a violation: two acting agents).
struct Turns {
    both: bool,
    steps: u32,
}

#[derive(Clone, Copy)]
struct TurnsConfig {
    both: bool,
}

impl EnvConfig for TurnsConfig {}

impl HostEnvironment for Turns {
    const ENV_ID: &'static str = "test/Turns-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
    const TURNS: tessera_rl_core::TurnStructure = tessera_rl_core::TurnStructure::Sequential;
    type Config = TurnsConfig;

    fn new(cfg: &TurnsConfig) -> Result<Self, HostError> {
        Ok(Self {
            both: cfg.both,
            steps: 0,
        })
    }

    fn observation_space(_: &TurnsConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &TurnsConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.steps = 0;
    }

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {
        self.steps += 1;
    }

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}

    fn acting(&self, agent: AgentSlot) -> bool {
        self.both || agent.index() == usize::try_from(self.steps % 2).unwrap_or(0)
    }
}

#[test]
fn test_host_pool_sequential_turn_order_alternates_and_violation_poisons() {
    let mut p = HostPool::<Turns, 2>::new(TurnsConfig { both: false }, options(2, 0)).unwrap();
    let acting = |out: &Outputs| out.acting().iter().map(|f| f.get()).collect::<Vec<_>>();
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(acting(out), [true, false, true, false]);
    let none = [0; 4];
    let out = p
        .step(ActionInput {
            discrete: &none,
            continuous: &[],
        })
        .unwrap();
    assert_eq!(acting(out), [false, true, false, true]);
    let mut bad = HostPool::<Turns, 2>::new(TurnsConfig { both: true }, options(2, 0)).unwrap();
    assert!(matches!(
        bad.reset_all(Reseed::Keep),
        Err(EngineError::TurnOrder { instance: 0, .. })
    ));
    assert!(bad.is_poisoned());
}

/// Output ownership (ADR-0003, SPEC 5.3): one storage for the pool's life,
/// a write generation that moves on every call, memory that outlives the
/// pool, and borrowed rows that show the later call's values.
#[test]
fn test_host_pool_outputs_one_storage_generations_and_lifetime() {
    let mut p = pool(2, 0, BookConfig::default());
    let storage = p.reset_all(Reseed::Keep).unwrap().storage();
    let g0 = storage.generation();
    let before = obs(p.outputs().unwrap(), 0, 0);
    step(&mut p, &buy(2, 3)).unwrap();
    let out = p.outputs().unwrap();
    assert!(
        std::sync::Arc::ptr_eq(&storage, &out.storage()),
        "same memory"
    );
    assert!(storage.generation() > g0, "the step started a new window");
    let after = obs(out, 0, 0);
    assert_ne!(before, after, "the rows show the later call's values");
    let g1 = storage.generation();
    assert!(step(&mut p, &[0]).is_err());
    assert_eq!(storage.generation(), g1, "a refused call writes nothing");
    drop(p);
    assert!(storage.bytes() > 0, "the memory outlives the pool");
}

#[test]
fn test_host_pool_without_masks_has_no_mask_buffer() {
    // `Wide` keeps the default `MASKS = false`; `Book` sets it.
    let mut p = HostPool::<Wide<3>, 1>::new(WideConfig, options(2, 0)).unwrap();
    assert_eq!(p.reset_all(Reseed::Keep).unwrap().masks().len(), 0);
    let mut b = pool(2, 0, BookConfig::default());
    assert_eq!(
        b.reset_all(Reseed::Keep).unwrap().masks().len(),
        2 * AGENTS * 7
    );
}

/// A `SameStep` reset's observation is written on zeroed rows (ADR-0058):
/// the rows held the terminal observation, and an environment that writes
/// nothing at an episode's start must leave zeros, not the old bytes.
#[test]
fn test_host_pool_same_step_reset_observes_on_zeroed_rows() {
    let mut o = options(1, 7);
    o.auto_reset = AutoReset::SameStep;
    let cfg = BookConfig {
        episode: 2,
        blank_at_start: true,
        ..BookConfig::default()
    };
    let mut p = Pool::new(cfg, o).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    step(&mut p, &buy(1, 1)).unwrap();
    let out = p.outputs().unwrap();
    assert_eq!(
        out.terminal_observation(0).unwrap()[..16],
        obs_bytes([2, 4, 11, 2])
    );
    assert_eq!(
        obs(out, 0, 0),
        [0; 4],
        "the old bytes leaked into the reset's rows"
    );
}

/// A host environment with one `Discrete(n)` head per agent whose `mask`
/// panics unless `M` (its `MASKS`), and otherwise makes only choice 0
/// legal.
struct Choices<const M: bool>;

#[derive(Clone, Copy)]
struct ChoicesConfig {
    n: u32,
}

impl EnvConfig for ChoicesConfig {}

impl<const M: bool> HostEnvironment for Choices<M> {
    const ENV_ID: &'static str = "test/Choices-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;
    const MASKS: bool = M;
    type Config = ChoicesConfig;

    fn new(_: &ChoicesConfig) -> Result<Self, HostError> {
        Ok(Self)
    }

    fn observation_space(_: &ChoicesConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(cfg: &ChoicesConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: cfg.n })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}

    fn step(&mut self, _: &Actions<'_, ()>, _: &mut HostOut<'_>) {}

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}

    fn mask(&self, _: AgentSlot, mask: &mut tessera_rl_core::io::MaskWriter<'_>) {
        assert!(M, "mask called without MASKS");
        mask.fill(false);
        mask.set(0, true);
    }
}

#[test]
fn test_host_pool_masks_false_no_buffer_no_mask_calls_large_discrete_accepted() {
    // 2^31 choices: a mask row would be 2 GiB, past the default budget.
    let cfg = ChoicesConfig { n: 1 << 31 };
    let mut p = HostPool::<Choices<false>, 1>::new(cfg, options(2, 0)).unwrap();
    assert_eq!(p.action_heads(), (1, 0));
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(out.masks().len(), 0);
    let out = p
        .step(ActionInput {
            discrete: &[5, 1 << 30],
            continuous: &[],
        })
        .unwrap();
    assert_eq!(out.masks().len(), 0);
    assert!(!p.is_poisoned(), "mask was never called");
}

#[test]
fn test_host_pool_masks_true_layout_and_budget_check() {
    let small = ChoicesConfig { n: 4 };
    // With MASKS the environment's mask is written: only choice 0 is legal.
    let mut p = HostPool::<Choices<true>, 1>::new(small, options(2, 0)).unwrap();
    let masks: Vec<bool> = p
        .reset_all(Reseed::Keep)
        .unwrap()
        .masks()
        .iter()
        .map(|b| b.get())
        .collect();
    assert_eq!(masks, [true, false, false, false].repeat(2));
    // The mask buffer is sized and checked against the budget.
    let large = ChoicesConfig { n: 1 << 31 };
    assert!(matches!(
        HostPool::<Choices<true>, 1>::new(large, options(2, 0)).err(),
        Some(EngineError::MaskBudgetExceeded { .. })
    ));
    let mut tight = options(2, 0);
    tight.mask_budget_bytes = 7;
    assert!(matches!(
        HostPool::<Choices<true>, 1>::new(small, tight).err(),
        Some(EngineError::MaskBudgetExceeded {
            bytes: 8,
            budget: 7
        })
    ));
    tight.mask_budget_bytes = 8;
    assert!(HostPool::<Choices<true>, 1>::new(small, tight).is_ok());
}

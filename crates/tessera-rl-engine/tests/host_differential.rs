// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host pool's bookkeeping against `EnvPool`'s (ADR-0062 item 4): a
//! `HostEnvironment` that wraps an `Environment` (it holds the plain-data
//! state and drives the environment's kernels with the same RNG identity
//! the engine would use) must give the outputs `EnvPool` gives, output for
//! output, every step: observations, terminal observations, rewards, agent
//! and episode flags, alive and acting masks, slot generations and episode
//! statistics.
//!
//! The wrapped environment is the engine tests' `Counter`: two agents, RNG
//! draws in `step` and `observe`, agent 1 dies past `kill_at` and respawns
//! on the next step (deaths, spawns and generations), plus a step limit
//! from the pool options. It runs as written (`Simultaneous`) and as
//! `Alternating`, the same kernels with `Sequential` turns. Each runs under
//! `NextStep` and `SameStep` auto-reset, Inline and with workers.

#![cfg(all(not(loom), feature = "host"))]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a failure is the assertion"
)]

mod common;

use core::marker::PhantomData;
use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, CountObs, Counter, Counts};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, HostEnvironment, HostError, HostSeed, NoAdaptive, RecordHeader,
    SpaceError, StateInit, StepOut, Tier, TurnStructure,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, HostPool, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::{BranchSalt, LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{ObsRng, RngAuthority, StepRng};

/// `Counter` with `Sequential` turns: agent `steps % 2` acts, or agent 0
/// while agent 1 is dead.
struct Alternating;

impl Environment for Alternating {
    type State = Counts;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::observation_space(cfg)
    }

    fn action_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::action_space(cfg)
    }

    fn reset(cfg: &CountConfig, rng: &mut StepRng, state: &mut Counts) {
        Counter::reset(cfg, rng, state);
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Counter::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &CountConfig,
        state: &Counts,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        Counter::observe(cfg, state, agent, rng, out);
    }

    fn acting(_: &CountConfig, state: &Counts, agent: AgentSlot) -> bool {
        let turn = if state.dead.get() { 0 } else { state.steps % 2 };
        agent.index() == usize::try_from(turn).unwrap_or(0)
    }
}

/// A host environment driving `E`'s kernels over its own state, with the
/// RNG identity the engine gives an `E` record: the instance's seed, a zero
/// branch salt, the default logical id, the episode index and the step.
struct Hosted<E: Environment> {
    cfg: CountConfig,
    state: E::State,
    identity: RngIdentity,
    authority: RngAuthority,
    _env: PhantomData<E>,
}

impl<E> HostEnvironment for Hosted<E>
where
    E: Environment<Config = CountConfig, Action = (), Obs = CountObs> + Send + 'static,
    E::State: Send,
{
    const ENV_ID: &'static str = "test/Hosted-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
    const TURNS: TurnStructure = E::TURNS;
    // The first-party environment respawns agent 1 (ADR-0065).
    const SPAWNS: bool = true;
    type Config = CountConfig;

    fn new(cfg: &CountConfig) -> Result<Self, HostError> {
        Ok(Self {
            cfg: cfg.clone(),
            state: E::State::state_default(),
            identity: RngIdentity::new(Seed::new(0), LogicalId::DEFAULT),
            authority: RngAuthority::engine_only(),
            _env: PhantomData,
        })
    }

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        E::observation_space(cfg)
    }

    fn action_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        E::action_space(cfg)
    }

    fn reset(&mut self, seed: HostSeed, _: &mut StepOut<'_>) {
        self.identity = RngIdentity::from_parts(
            Seed::new(seed.seed()),
            BranchSalt::new(0),
            LogicalId::DEFAULT,
            seed.episode_index(),
            0,
        );
        self.state = E::State::state_default();
        let mut rng = StepRng::new(&self.authority, &self.identity, StreamId::DYNAMICS);
        E::reset(&self.cfg, &mut rng, &mut self.state);
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut StepOut<'_>) {
        let id = self.identity;
        self.identity = RngIdentity::from_parts(
            id.seed(),
            id.branch_salt(),
            id.logical_id(),
            id.episode_index(),
            id.step_index() + 1,
        );
        let mut rng = StepRng::new(&self.authority, &self.identity, StreamId::DYNAMICS);
        E::step(&self.cfg, &mut self.state, actions, &mut rng, out);
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        // `CountObs` has one leaf; hand `E` a typed writer over it.
        let Some(leaf) = obs.leaf(0) else {
            return;
        };
        let mut leaves = [leaf];
        let mut typed = ObsWriter::<CountObs>::new(&mut leaves);
        let mut rng = ObsRng::new(&self.authority, &self.identity);
        E::observe(&self.cfg, &self.state, agent, &mut rng, &mut typed);
    }

    fn acting(&self, agent: AgentSlot) -> bool {
        E::acting(&self.cfg, &self.state, agent)
    }
}

const N: u32 = 5;

fn config() -> CountConfig {
    CountConfig {
        // Agent 1 dies once its value passes 3 and respawns the next step.
        kill_at: 3,
        respawn: true,
        length: 7,
        ..CountConfig::default()
    }
}

fn options(auto_reset: AutoReset) -> PoolOptions {
    PoolOptions {
        auto_reset,
        chunk_size: NonZeroU32::new(2).unwrap(),
        // Below `length` some episodes are cut by the step limit (truncated).
        max_episode_steps: NonZeroU32::new(6),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(31))
    }
}

/// Every output the two pools share.
fn outputs(out: &Outputs) -> Vec<Vec<u8>> {
    let flags = |f: &[tessera_rl_core::Bool8]| f.iter().map(|b| u8::from(b.get())).collect();
    let floats = |f: &[f32]| f.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    vec![
        out.observation(0).unwrap().to_vec(),
        out.terminal_observation(0)
            .map(<[u8]>::to_vec)
            .unwrap_or_default(),
        floats(out.rewards()),
        flags(out.agent_terminated()),
        flags(out.agent_truncated()),
        flags(out.alive()),
        flags(out.acting()),
        out.generations()
            .iter()
            .flat_map(|g| g.to_le_bytes())
            .collect(),
        flags(out.episode_terminated()),
        flags(out.episode_truncated()),
        floats(out.episode_return()),
        floats(out.episode_length()),
        floats(out.episode_illegal_actions()),
        floats(out.episode_invalid_agent_events()),
    ]
}

/// Actions that vary by step, instance and agent: increments of 0 to 2.
fn actions(t: usize) -> Vec<i32> {
    (0..N as usize * 2)
        .map(|k| i32::try_from((k * 5 + t * 3) % 3).unwrap())
        .collect()
}

fn act(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

fn compare<E>(mode: AutoReset, workers: Option<usize>)
where
    E: Environment<
            Config = CountConfig,
            Action = (),
            Obs = CountObs,
            Header = RecordHeader<2, NoAdaptive>,
        > + Send
        + 'static,
    E::State: Send,
{
    let mut env = EnvPool::<E>::new(config(), options(mode)).unwrap();
    let mut host = HostPool::<Hosted<E>, 2>::new(&config(), options(mode)).unwrap();
    if let Some(w) = workers {
        let sync = SyncOptions::new(NonZeroUsize::new(w).unwrap());
        env = env.with_workers(sync.clone()).unwrap();
        host = host.with_workers(sync).unwrap();
    }
    let label = |what: &str, t: usize| format!("{mode:?}, {workers:?} workers, {what} {t}");
    assert_eq!(
        outputs(host.reset_all(Reseed::Keep).unwrap()),
        outputs(env.reset_all(Reseed::Keep).unwrap()),
        "{}",
        label("reset", 0)
    );
    let (mut deaths, mut spawns, mut truncations) = (0, 0, 0);
    for t in 0..40 {
        let a = actions(t);
        let e = env.step(act(&a)).expect("EnvPool step");
        let h = host.step(act(&a)).expect("HostPool step");
        assert_eq!(outputs(h), outputs(e), "{}", label("step", t));
        deaths += e.agent_terminated().iter().filter(|f| f.get()).count();
        spawns += e.generations().iter().filter(|&&g| g > 0).count();
        truncations += e.episode_truncated().iter().filter(|f| f.get()).count();
    }
    // The run covered what it claims to cover.
    assert!(
        deaths > 0 && spawns > 0 && truncations > 0,
        "{deaths} {spawns} {truncations}"
    );
    // An explicit reset of some instances with new seeds, then more steps.
    let seeds = [Seed::new(7), Seed::new(8)];
    assert_eq!(
        outputs(host.reset(&[3, 1], Reseed::List(&seeds)).unwrap()),
        outputs(env.reset(&[3, 1], Reseed::List(&seeds)).unwrap()),
        "{}",
        label("explicit reset", 40)
    );
    for t in 40..50 {
        let a = actions(t);
        let e = env.step(act(&a)).unwrap();
        let h = host.step(act(&a)).unwrap();
        assert_eq!(outputs(h), outputs(e), "{}", label("step", t));
    }
}

#[test]
fn test_host_differential_simultaneous_matches_env_pool() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        for workers in [None, Some(2)] {
            compare::<Counter>(mode, workers);
        }
    }
}

#[test]
fn test_host_differential_sequential_matches_env_pool() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        for workers in [None, Some(2)] {
            compare::<Alternating>(mode, workers);
        }
    }
}

/// Invalid agent events: an agent reported dead twice, alone and in the
/// same step as a valid respawn (the near-equivalent case: the death is
/// refused and counted, the spawn applied). Both pools mark them alike.
/// Under `debug-checks` both fail the call with the same
/// `InvalidAgentEvent` and poison; without it both count the event in
/// `episode_invalid_agent_events`. The step results are compared whole.
#[test]
fn test_host_differential_invalid_agent_events_match_env_pool() {
    for respawn in [false, true] {
        invalid_events_match(respawn);
    }
}

fn invalid_events_match(respawn: bool) {
    let cfg = CountConfig {
        die_twice: true,
        respawn,
        ..config()
    };
    let mut env = EnvPool::<Counter>::new(cfg.clone(), options(AutoReset::NextStep)).unwrap();
    let mut host = HostPool::<Hosted<Counter>, 2>::new(&cfg, options(AutoReset::NextStep)).unwrap();
    env.reset_all(Reseed::Keep).unwrap();
    host.reset_all(Reseed::Keep).unwrap();
    let mut events = 0;
    for t in 0..20 {
        let a = actions(t);
        let e = env.step(act(&a)).map(outputs);
        let h = host.step(act(&a)).map(outputs);
        assert_eq!(h, e, "respawn {respawn}, step {t}");
        if e.is_err() {
            events += 1;
            env.reset_all(Reseed::Keep).unwrap();
            host.reset_all(Reseed::Keep).unwrap();
        } else if let Ok(o) = &e {
            events += usize::from(o.get(13).is_some_and(|v| v.iter().any(|&b| b != 0)));
        }
    }
    assert!(
        events > 0,
        "respawn {respawn}: the run produced an invalid agent event"
    );
}

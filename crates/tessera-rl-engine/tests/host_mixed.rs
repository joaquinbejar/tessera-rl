// SPDX-License-Identifier: MIT OR Apache-2.0

//! One binary holding an `EnvPool` and a `HostPool` (ADR-0062 item 4 and
//! item 11), with `debug-checks` and the counting allocator installed:
//!
//! - every `Environment` chunk runs under canonical floating-point
//!   controls, although the host environment leaves flush-to-zero set
//!   after each of its steps;
//! - the host's allocations are counted in the host pool's statistic and
//!   never reach the `EnvPool`'s count, so its steps never fail with
//!   `AllocationInChunkStep`;
//! - the `EnvPool`'s outputs are exactly those of a run without the host
//!   pool.
//!
//! Inline mode runs both pools on this thread. Sync mode gives both pools
//! workers on the same cores: an explicit, shared `cpu_set` where the OS
//! reports core ids (Linux, Windows), unpinned otherwise (macOS). Each pool
//! owns its worker threads (ADR-0005), so what is shared is the cores.

#![cfg(all(not(loom), feature = "host", feature = "debug-checks"))]
#![expect(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "test helpers unwrap, and rewards are exact markers"
)]

mod host_support;

use core::num::{NonZeroU32, NonZeroUsize};

use host_support::{AGENTS, Book, BookConfig};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, HostEnvironment, NoAdaptive, ObsLeaf, Observation,
    RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_engine::debug::TrackingAllocator;
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, HostPool, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// An environment that records, in its reward, whether its step ran under
/// canonical floating-point controls: 1.0 if it did, -1.0 if not.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Probe {
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct ProbeConfig {
    #[config(default = 4)]
    length: u32,
}

#[derive(Observation)]
struct ProbeObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    steps: ObsLeaf<i32>,
}

struct FpProbe;

impl Environment for FpProbe {
    type State = Probe;
    type Config = ProbeConfig;
    type Obs = ProbeObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &ProbeConfig) -> Result<Space, SpaceError> {
        ProbeObs::space(cfg)
    }

    fn action_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &ProbeConfig, _: &mut StepRng, state: &mut Probe) {
        *state = Probe::default();
    }

    fn step(
        cfg: &ProbeConfig,
        state: &mut Probe,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        let canonical = tessera_rl_num::fp::is_canonical();
        out.set_reward(AgentSlot::FIRST, if canonical { 1.0 } else { -1.0 });
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &ProbeConfig,
        state: &Probe,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, ProbeObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(ProbeObs::STEPS) {
            leaf.set(0, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}

const N: u32 = 6;

fn options(seed: u64) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(2).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(seed))
    }
}

type Host = HostPool<Book, { Book::MAX_AGENTS }>;

fn host() -> Host {
    let cfg = BookConfig {
        hostile_fp: true,
        ..BookConfig::default()
    };
    Host::new(cfg, options(1)).unwrap()
}

fn probe() -> EnvPool<FpProbe> {
    EnvPool::new(ProbeConfig::default(), options(2)).unwrap()
}

const PROBE_ACTIONS: [i32; N as usize] = [0; N as usize];

fn host_actions() -> Vec<i32> {
    // Every agent buys: the book pushes an order (a heap allocation).
    [1, 1].repeat(N as usize * AGENTS)
}

fn act(discrete: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete,
        continuous: &[],
    }
}

fn probe_trace(out: &Outputs) -> (Vec<f32>, Vec<u8>, Vec<bool>) {
    (
        out.rewards().to_vec(),
        out.observation(0).unwrap().to_vec(),
        out.episode_terminated().iter().map(|b| b.get()).collect(),
    )
}

/// Steps `probe` `steps` times; with `host`, steps it before each probe
/// step and checks the host side. Returns the probe's trace.
fn run(
    probe: &mut EnvPool<FpProbe>,
    mut host: Option<&mut Host>,
    steps: usize,
) -> Vec<(Vec<f32>, Vec<u8>, Vec<bool>)> {
    probe.reset_all(Reseed::Keep).unwrap();
    if let Some(h) = host.as_deref_mut() {
        h.reset_all(Reseed::Keep).unwrap();
    }
    let mut trace = Vec::new();
    for _ in 0..steps {
        if let Some(h) = host.as_deref_mut() {
            h.step(act(&host_actions())).unwrap();
        }
        // An allocation charged to the probe's pool would fail this call
        // with `AllocationInChunkStep` (ADR-0055).
        let out = probe.step(act(&PROBE_ACTIONS));
        assert!(
            !matches!(out, Err(EngineError::AllocationInChunkStep { .. })),
            "host allocations reached the EnvPool's count"
        );
        let out = out.unwrap();
        assert!(
            // 1.0 for a step, 0.0 for a NextStep reset; -1.0 never.
            out.rewards().iter().all(|&r| r == 1.0 || r == 0.0),
            "an Environment chunk ran under non-canonical FP controls"
        );
        trace.push(probe_trace(out));
    }
    trace
}

#[test]
fn test_host_mixed_inline_probe_canonical_and_uncharged() {
    let alone = run(&mut probe(), None, 10);
    let mut h = host();
    let mixed = run(&mut probe(), Some(&mut h), 10);
    assert_eq!(
        mixed, alone,
        "the EnvPool's outputs changed beside a host pool"
    );
    // Every host `step` left flush-to-zero set and was put back right after
    // it: 6 instances per call. The book's episodes are 5 steps (NextStep):
    // call 6 resets, whose host code leaves the controls alone: 6 * 9.
    assert_eq!(h.host_fp_resets(), 54);
    // The host allocated (one order per buying agent) and it was counted
    // there.
    assert!(h.allocations_in_host_steps() > 0);
    assert!(tessera_rl_num::fp::is_canonical());
}

/// Two workers each, on the same cores: an explicit shared `cpu_set` where
/// the OS reports core ids, unpinned where it does not.
fn sync_options() -> SyncOptions {
    SyncOptions {
        cpu_set: Some(vec![0, 1]),
        ..SyncOptions::new(NonZeroUsize::new(2).unwrap())
    }
}

#[test]
fn test_host_mixed_sync_shared_cores_probe_canonical_and_uncharged() {
    let alone = run(&mut probe(), None, 10);
    let pinned = probe().with_workers(sync_options());
    let (mut p, mut h, shared) = match pinned {
        Ok(p) => (p, host().with_workers(sync_options()).unwrap(), true),
        // No core ids reported (macOS): both pools unpinned.
        Err(EngineError::Options {
            option: "cpu_set", ..
        }) => {
            let two = SyncOptions::new(NonZeroUsize::new(2).unwrap());
            (
                probe().with_workers(two.clone()).unwrap(),
                host().with_workers(two).unwrap(),
                false,
            )
        }
        Err(e) => panic!("{e}"),
    };
    if cfg!(target_os = "linux") {
        assert!(shared, "Linux reports core ids: the cpu_set must be shared");
    }
    let mixed = run(&mut p, Some(&mut h), 10);
    assert_eq!(
        mixed, alone,
        "the EnvPool's outputs changed beside a host pool"
    );
    // As in Inline mode, on the workers: 6 * 9.
    assert_eq!(h.host_fp_resets(), 54);
    assert!(h.allocations_in_host_steps() > 0);
    assert_eq!((p.workers(), h.workers()), (2, 2));
}

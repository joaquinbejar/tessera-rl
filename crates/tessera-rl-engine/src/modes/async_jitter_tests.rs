// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async mode under injected jitter (ADR-0047): every worker sleeps a
//! pseudo-random time before each claimed step, so completion order varies
//! far more than send order does; every instance's trajectory must still be
//! bit-identical to Sync with the same actions, for 1, 2, 7 and 64 workers.

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

use core::sync::atomic::{AtomicBool, Ordering};

use crate::buffer::batch::{
    GATE_AFTER_PUSH, GATE_BEFORE_PUSH, GATE_HOLD_ID, GATE_OPEN, GATE_REACHED, HELD_ID,
};
use crate::modes::async_mode::SNAPSHOT_HOOK;
use crate::sys::Faults;
use crate::{
    ActionInput, AsyncOptions, AutoReset, BatchToken, EnvPool, NoMasks, PoolOptions, Reseed,
};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Walk {
    value: i32,
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct WalkConfig {}

#[derive(Observation)]
struct WalkObs {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [1]))]
    value: ObsLeaf<i32>,
}

/// One agent adds its action and a random bit to a counter; the episode
/// ends after four steps. The random bits make each trajectory depend on the
/// instance's own RNG stream, so a mixed-up instance shows.
struct Walker;

impl Environment for Walker {
    type State = Walk;
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
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(_: &WalkConfig, rng: &mut StepRng, state: &mut Walk) {
        *state = Walk {
            value: i32::from(rng.next_u32().is_multiple_of(2)),
            steps: 0,
        };
    }

    fn step(
        _: &WalkConfig,
        state: &mut Walk,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let inc = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0)
            + i32::from(rng.next_u32().is_multiple_of(2));
        state.value += inc;
        state.steps += 1;
        #[expect(clippy::cast_precision_loss, reason = "small test values")]
        out.set_reward(AgentSlot::FIRST, inc as f32);
        if state.steps >= 4 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &WalkConfig,
        state: &Walk,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, WalkObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(WalkObs::VALUE) {
            leaf.set(0, state.value);
        }
    }
}

type Pool = EnvPool<Walker, NoMasks>;
type AsyncPoolOf = crate::AsyncPool<Walker, NoMasks>;

const N: u32 = 40;
const STEPS: usize = if cfg!(miri) { 2 } else { 6 };

fn options(mode: AutoReset) -> PoolOptions {
    PoolOptions {
        auto_reset: mode,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(91))
    }
}

fn row(t: usize, i: usize) -> [i32; 1] {
    [i32::try_from((i * 5 + t * 7) % 3).unwrap()]
}

/// Observations, rewards, alive, episode flags and the step counter of one
/// instance, as bits.
fn token_frame(t: &BatchToken<Walker, NoMasks>, id: u32) -> Vec<u64> {
    let mut v: Vec<u64> = t
        .observation(0, id)
        .unwrap()
        .iter()
        .map(|&b| u64::from(b))
        .collect();
    v.extend(
        t.rewards(id)
            .unwrap()
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    v.extend(t.alive(id).unwrap().iter().map(|f| u64::from(f.get())));
    v.push(u64::from(t.episode_terminated(id).unwrap().get()));
    v.push(u64::from(t.episode_return(id).unwrap().to_bits()));
    let (e, s) = t.step_counter(id).unwrap();
    v.extend([u64::from(e), u64::from(s)]);
    v
}

fn pool_frame(pool: &Pool, id: u32) -> Vec<u64> {
    let out = pool.outputs().unwrap();
    let i = usize::try_from(id).unwrap();
    let a = 1;
    let obs = out.observation(0).unwrap();
    let per = obs.len() / usize::try_from(N).unwrap();
    let mut v: Vec<u64> = obs[i * per..(i + 1) * per]
        .iter()
        .map(|&b| u64::from(b))
        .collect();
    v.extend(
        out.rewards()[i * a..(i + 1) * a]
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    v.extend(
        out.alive()[i * a..(i + 1) * a]
            .iter()
            .map(|f| u64::from(f.get())),
    );
    v.push(u64::from(out.episode_terminated()[i].get()));
    v.push(u64::from(out.episode_return()[i].to_bits()));
    let identity = tessera_rl_core::RecordLayout::engine(pool.header(id).unwrap()).identity;
    v.extend([
        u64::from(identity.episode_index()),
        u64::from(identity.step_index()),
    ]);
    v
}

#[test]
fn test_async_injected_jitter_matches_sync_per_instance_for_1_2_7_and_64_workers() {
    let count = usize::try_from(N).unwrap();
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        let mut reference = Pool::new(WalkConfig {}, options(mode)).unwrap();
        reference.reset_all(Reseed::Keep).unwrap();
        let mut frames: Vec<Vec<Vec<u64>>> =
            vec![(0..N).map(|i| pool_frame(&reference, i)).collect()];
        for step_t in 0..STEPS {
            let a: Vec<i32> = (0..count).flat_map(|i| row(step_t, i)).collect();
            reference
                .step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
            frames.push((0..N).map(|i| pool_frame(&reference, i)).collect());
        }
        // Under Miri, one small-worker run keeps the job short.
        let counts: &[usize] = if cfg!(miri) { &[2] } else { &[1, 2, 7, 64] };
        for &workers in counts {
            let faults = Faults {
                jitter_us: 300,
                ..Faults::default()
            };
            let mut pool = Pool::new(WalkConfig {}, options(mode))
                .unwrap()
                .async_faults(
                    AsyncOptions::new(
                        NonZeroUsize::new(workers).unwrap(),
                        NonZeroUsize::new(1).unwrap(),
                    ),
                    faults,
                )
                .unwrap();
            drop(pool.reset_all(Reseed::Keep).unwrap());
            let mut step = vec![0_usize; count];
            // Every instance is sent at once and resent as soon as it comes
            // back, so the order of completions is the workers' jitter.
            let ids: Vec<u32> = (0..N).collect();
            let a: Vec<i32> = (0..count).flat_map(|i| row(0, i)).collect();
            pool.send(
                &ids,
                ActionInput {
                    discrete: &a,
                    continuous: &[],
                },
            )
            .unwrap();
            let mut order = Vec::new();
            let mut done = 0;
            while done < count {
                let token = pool.recv().unwrap();
                let id = token.ids().next().unwrap();
                let i = usize::try_from(id).unwrap();
                step[i] += 1;
                order.push(id);
                assert_eq!(
                    token_frame(&token, id),
                    frames[step[i]][i],
                    "{workers} workers, {mode:?}, id {id}, step {}",
                    step[i]
                );
                if step[i] == STEPS {
                    done += 1;
                    drop(token);
                    continue;
                }
                let next = row(step[i], i);
                pool.send_batch(
                    token,
                    ActionInput {
                        discrete: &next,
                        continuous: &[],
                    },
                )
                .map_err(|e| e.error)
                .unwrap();
            }
            if workers > 1 {
                // The jitter reordered completions: not plain send order.
                let sorted = order.windows(2).filter(|w| w[0] > w[1]).count();
                assert!(sorted > 0, "{workers} workers: completions were reordered");
            }
        }
    }
}

/// Ids a send admitted with no job left to claim them (both jobs were past
/// their last claim) are still stepped when the pool is drained: `into_pool`
/// hands them back stepped, exactly as Sync steps them (TR m1 on #84).
#[test]
fn test_async_unpublished_ids_are_stepped_by_the_drain() {
    let count = usize::try_from(N).unwrap();
    let mut reference = Pool::new(WalkConfig {}, options(AutoReset::NextStep)).unwrap();
    reference.reset_all(Reseed::Keep).unwrap();
    let a: Vec<i32> = (0..count).flat_map(|i| row(0, i)).collect();
    reference
        .step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        ))
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    pool.send_unpublished(
        &ids,
        ActionInput {
            discrete: &a,
            continuous: &[],
        },
    )
    .unwrap();
    let back = pool.into_pool().map_err(|_| ()).unwrap();
    assert!(!back.is_poisoned());
    for id in 0..N {
        assert_eq!(
            back.state_hash(id).unwrap(),
            reference.state_hash(id).unwrap(),
            "id {id} was stepped once"
        );
    }
}

/// A pool left with admitted ids no job will claim (three ids, batches of
/// ten): `recv` and `reset_all` each make progress on them, so a caller
/// retrying them is never stuck (TR M1 on #84). `snapshot` of an idle id
/// needs no progress at all (issue #85), and an unclaimed id is `InFlight`.
#[test]
fn test_async_recv_and_reset_all_make_progress_on_unclaimed_ids() {
    let make = || {
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(10).unwrap(),
            ))
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        pool.send_unpublished(
            &[0, 1, 2],
            ActionInput {
                discrete: &[1, 1, 1],
                continuous: &[],
            },
        )
        .unwrap();
        pool
    };
    let tries = if cfg!(miri) { 200 } else { 100_000 };
    // `recv` refuses (three ids cannot fill a batch of ten), but steps them.
    let mut pool = make();
    for _ in 0..3 {
        assert!(matches!(
            pool.recv().map(|_| ()),
            Err(crate::EngineError::BatchUnreachable { .. })
        ));
    }
    let ok = (0..tries).any(|_| {
        std::thread::yield_now();
        pool.running_ids() == 0
    });
    assert!(ok, "recv stepped the unclaimed ids");
    // `reset_all` alone makes progress.
    let mut pool = make();
    let ok = (0..tries).any(|_| match pool.reset_all(Reseed::Keep) {
        Ok(t) => {
            drop(t);
            true
        }
        Err(crate::EngineError::NotQuiescent) => {
            std::thread::yield_now();
            false
        }
        Err(e) => panic!("unexpected {e:?}"),
    });
    assert!(ok, "reset_all stepped the unclaimed ids");
    // A snapshot of an idle id succeeds at once, and steps nothing; an
    // unclaimed id is refused.
    let mut pool = make();
    pool.snapshot(5).unwrap();
    assert_eq!(pool.running_ids(), 3);
    assert_eq!(
        pool.snapshot(1).map(|_| ()),
        Err(crate::EngineError::InFlight { id: 1 })
    );
}

/// A worker sleeps right after its id counts as stepped: with the entry made
/// visible first, the last `recv` would return while the id still counted as
/// running, and a writer or snapshot would refuse with nothing to wait for
/// (the CI flake of `alloc_async`, ADR-0047 section 3).
#[test]
fn test_async_a_received_id_is_never_counted_as_running() {
    let faults = Faults {
        jitter_us: 300,
        ..Faults::default()
    };
    let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_faults(
            AsyncOptions::new(
                NonZeroUsize::new(4).unwrap(),
                NonZeroUsize::new(N as usize).unwrap(),
            ),
            faults,
        )
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    let actions = vec![1; ids.len()];
    let rounds = if cfg!(miri) { 3 } else { 200 };
    for round in 0..rounds {
        pool.send(
            &ids,
            ActionInput {
                discrete: &actions,
                continuous: &[],
            },
        )
        .unwrap();
        // Polling, as a blocking `recv` waits for the job and hides the race.
        loop {
            if let Some(t) = pool.try_recv().unwrap() {
                drop(t);
                break;
            }
            std::thread::yield_now();
        }
        if round % 2 == 0 {
            pool.snapshot(0).unwrap();
        } else {
            pool.reset(&ids, Reseed::Keep).unwrap();
            drop(pool.recv().unwrap());
        }
    }
}

/// Serializes the tests that use the process-wide gates of `complete`.
static GATES: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Opens the gates even if an assertion fails, so that no worker stays held.
struct OpenGates;

impl Drop for OpenGates {
    fn drop(&mut self) {
        GATE_OPEN.store(true, Ordering::Release);
    }
}

fn gated(gate: u32) -> (AsyncPoolOf, OpenGates) {
    GATE_REACHED.store(false, Ordering::Release);
    GATE_OPEN.store(false, Ordering::Release);
    let open = OpenGates;
    let pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_faults(
            AsyncOptions::new(NonZeroUsize::MIN, NonZeroUsize::MIN),
            Faults {
                jitter_us: gate,
                ..Faults::default()
            },
        )
        .unwrap();
    (pool, open)
}

fn wait_for_gate() {
    while !GATE_REACHED.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
}

/// A receive that pops an id before its worker counts it leaves the ready
/// count below zero for a moment, which must not read as a ready batch.
#[test]
fn test_async_waiter_is_not_ready_while_the_count_is_below_zero() {
    let _lock = GATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut pool, open) = gated(GATE_AFTER_PUSH);
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let waiter = pool.waiter();
    pool.send(
        &[1],
        ActionInput {
            discrete: &[1],
            continuous: &[],
        },
    )
    .unwrap();
    wait_for_gate();
    let token = loop {
        if let Some(t) = pool.try_recv().unwrap() {
            break t;
        }
        std::thread::yield_now();
    };
    assert!(!waiter.wait_ready(core::time::Duration::ZERO));
    drop(token);
    drop(open);
}

/// Codex's probe of #96: with an id's rows written and counted as stepped but
/// its completion unpublished, nothing is delivered early, its own snapshot
/// still refuses, and a writer drains the job before it writes.
#[test]
fn test_async_id_stepped_but_unpublished_is_neither_delivered_nor_overwritten() {
    let _lock = GATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut pool, open) = gated(GATE_BEFORE_PUSH);
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool.send(
        &[1],
        ActionInput {
            discrete: &[1],
            continuous: &[],
        },
    )
    .unwrap();
    wait_for_gate();
    assert!(pool.try_recv().unwrap().is_none());
    let record = pool.snapshot(0).unwrap();
    assert!(matches!(
        pool.snapshot(1),
        Err(crate::EngineError::InFlight { .. })
    ));
    let mut records = vec![record; N as usize];
    assert!(matches!(
        pool.snapshot_all(&mut records),
        Err(crate::EngineError::NotQuiescent)
    ));
    let opener = std::thread::spawn(|| {
        std::thread::sleep(core::time::Duration::from_millis(1));
        GATE_OPEN.store(true, Ordering::Release);
    });
    drop(pool.reset_all(Reseed::Keep).unwrap());
    opener.join().unwrap();
    drop(open);
}

/// An id completed and pushed to the completion ring but not yet received
/// is still `InFlight` for `snapshot` (issue #85), while an idle id is
/// snapshotted unchanged; once received, the id is snapshotted too.
#[test]
fn test_async_snapshot_completed_but_unreceived_id_is_in_flight() {
    let _lock = GATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (mut pool, open) = gated(GATE_AFTER_PUSH);
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let idle = pool.snapshot(0).unwrap().to_bytes();
    pool.send(
        &[1],
        ActionInput {
            discrete: &[1],
            continuous: &[],
        },
    )
    .unwrap();
    wait_for_gate();
    assert_eq!(pool.running_ids(), 0);
    assert_eq!(
        pool.snapshot(1).map(|_| ()),
        Err(crate::EngineError::InFlight { id: 1 })
    );
    assert_eq!(pool.snapshot(0).unwrap().to_bytes(), idle);
    GATE_OPEN.store(true, Ordering::Release);
    let token = pool.recv().unwrap();
    assert!(token.contains(1));
    pool.snapshot(1).unwrap();
    drop(token);
    drop(open);
}

/// A step that waits for [`BOOM`], then panics with action 2.
struct Boomer;

static BOOM: AtomicBool = AtomicBool::new(false);
static BOOM_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Releases a step paused on [`BOOM`] on every exit path, a failed
/// assertion included, so the pool's `Drop` (which drains the paused job)
/// never waits forever. Declare it after the pool: it drops first.
struct ReleaseBoom;

impl Drop for ReleaseBoom {
    fn drop(&mut self) {
        BOOM.store(true, Ordering::Release);
    }
}

fn boom_serial() -> std::sync::MutexGuard<'static, ()> {
    BOOM_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Environment for Boomer {
    type State = Walk;
    type Config = WalkConfig;
    type Obs = WalkObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walker::observation_space(cfg)
    }

    fn action_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walker::action_space(cfg)
    }

    fn reset(cfg: &WalkConfig, rng: &mut StepRng, state: &mut Walk) {
        Walker::reset(cfg, rng, state);
    }

    fn step(
        cfg: &WalkConfig,
        state: &mut Walk,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        if actions.discrete(AgentSlot::FIRST, 0) == Some(2) {
            while !BOOM.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            std::panic::resume_unwind(Box::new("injected step panic"));
        }
        Walker::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &WalkConfig,
        state: &Walk,
        slot: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, WalkObs>,
    ) {
        Walker::observe(cfg, state, slot, rng, out);
    }
}

/// A step panics between `snapshot`'s poison check and its quiescence check:
/// nothing runs any more, but a row may be partly written (issue #97).
#[test]
fn test_async_snapshot_refuses_a_panic_between_its_two_checks() {
    let _serial = boom_serial();
    BOOM.store(false, Ordering::Release);
    let mut pool = EnvPool::<Boomer, NoMasks>::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_mode(AsyncOptions::new(NonZeroUsize::MIN, NonZeroUsize::MIN))
        .unwrap();
    let _release = ReleaseBoom;
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool.send(
        &[1],
        ActionInput {
            discrete: &[2],
            continuous: &[],
        },
    )
    .unwrap();
    SNAPSHOT_HOOK.set(Some(|| BOOM.store(true, Ordering::Release)));
    assert!(matches!(
        pool.snapshot(0),
        Err(crate::EngineError::Poisoned)
    ));
    drop(pool.reset_all(Reseed::Keep).unwrap());
}

/// With id 1 paused inside its step, a token of id 0 stays readable, its
/// generation unmoved and its snapshot unchanged; the release panics, and
/// recovery clears the poison.
#[test]
fn test_async_idle_token_readable_while_a_step_is_paused_then_panics() {
    let _serial = boom_serial();
    BOOM.store(false, Ordering::Release);
    let mut pool = EnvPool::<Boomer, NoMasks>::new(WalkConfig {}, options(AutoReset::NextStep))
        .unwrap()
        .async_mode(AsyncOptions::new(NonZeroUsize::MIN, NonZeroUsize::MIN))
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let act = |d| ActionInput {
        discrete: d,
        continuous: &[],
    };
    pool.send(&[0], act(&[1])).unwrap();
    let held = pool.recv().unwrap();
    let (rewards, generation) = (held.rewards(0).unwrap().to_vec(), held.generation(0));
    let quiescent = pool.snapshot(0).unwrap().to_bytes();
    let release = ReleaseBoom;
    pool.send(&[1], act(&[2])).unwrap();
    // Id 0 is idle (covered by `held`) while id 1 runs: its snapshot is
    // taken through its own row, byte-identical to the quiescent one. The
    // results are collected while id 1 is paused and checked after its
    // release, so a regression fails instead of hanging in `Drop`.
    let idle = pool.snapshot(0).map(|r| r.to_bytes());
    let running = pool.snapshot(1).map(|_| ());
    let (rewards_now, generation_now) = (held.rewards(0).map(<[f32]>::to_vec), held.generation(0));
    drop(release);
    assert_eq!(idle, Ok(quiescent));
    assert_eq!(running, Err(crate::EngineError::InFlight { id: 1 }));
    assert_eq!(rewards_now, Some(rewards));
    assert_eq!(generation_now, generation);
    assert!(matches!(
        pool.recv().map(|_| ()),
        Err(crate::EngineError::Panicked)
    ));
    assert!(pool.is_poisoned());
    drop(held);
    drop(pool.reset_all(Reseed::Keep).unwrap());
    assert!(!pool.is_poisoned());
}

/// How long a test of run claims may take before it fails by name instead
/// of hanging the binary (Miri runs far slower).
const WATCHDOG: core::time::Duration =
    core::time::Duration::from_secs(if cfg!(miri) { 1_200 } else { 30 });

/// Runs `f` on its own thread and fails, naming `what`, if it has not
/// finished within [`WATCHDOG`]: a broken claim or ring then fails the test
/// instead of hanging it. A panic in `f` is passed on.
fn within(what: &str, f: impl FnOnce() + Send + 'static) {
    let (done, finished) = std::sync::mpsc::channel();
    let body = std::thread::spawn(move || {
        f();
        let _ = done.send(());
    });
    match finished.recv_timeout(WATCHDOG) {
        Ok(()) => body.join().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what}: did not finish within {WATCHDOG:?}")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            if let Err(payload) = body.join() {
                std::panic::resume_unwind(payload);
            }
        }
    }
}

fn all_ids() -> (Vec<u32>, Vec<i32>) {
    ((0..N).collect(), vec![1; N as usize])
}

/// Clears the held id and opens the gates even if an assertion fails.
struct ReleaseHeld;

impl Drop for ReleaseHeld {
    fn drop(&mut self) {
        HELD_ID.store(u32::MAX, Ordering::Release);
        GATE_OPEN.store(true, Ordering::Release);
    }
}

/// With batches of one and two workers, a worker held before stepping id 0
/// holds only that id: a claim is at most `ceil(1 / 2) = 1` id, so every
/// other id is stepped and received through the other worker meanwhile
/// (issue #83). Under a claim of `ceil(found / workers)` the held worker
/// would hold half the ids and the receive would stall.
#[test]
fn test_async_held_id_does_not_hold_other_batches_of_one_received() {
    let _lock = GATES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _release = ReleaseHeld;
    within("held id with batches of one", || {
        let release = ReleaseHeld;
        GATE_REACHED.store(false, Ordering::Release);
        GATE_OPEN.store(false, Ordering::Release);
        HELD_ID.store(0, Ordering::Release);
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_faults(
                AsyncOptions::new(NonZeroUsize::new(2).unwrap(), NonZeroUsize::MIN),
                Faults {
                    jitter_us: GATE_HOLD_ID,
                    ..Faults::default()
                },
            )
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        let (ids, actions) = all_ids();
        pool.send(
            &ids,
            ActionInput {
                discrete: &actions,
                continuous: &[],
            },
        )
        .unwrap();
        wait_for_gate();
        // Collected while id 0 is held, checked after the release.
        let mut received = Vec::new();
        let mut failed = None;
        while received.len() + 1 < ids.len() {
            match pool.try_recv() {
                Ok(Some(token)) => received.extend(token.ids()),
                Ok(None) => std::thread::yield_now(),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        drop(release);
        assert_eq!(failed, None);
        let last: Vec<u32> = pool.recv().unwrap().ids().collect();
        assert!(!received.contains(&0), "the held id was not received early");
        assert_eq!(received.len(), ids.len() - 1);
        assert_eq!(last, [0]);
    });
}

/// No claim exceeds `ceil(batch_size / workers)`: with batches of 5 and two
/// workers, the first claim, which finds every id, takes exactly 3.
#[test]
fn test_async_claim_never_exceeds_the_batch_share_bounded() {
    within("claim size", || {
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_mode(AsyncOptions::new(
                NonZeroUsize::new(2).unwrap(),
                NonZeroUsize::new(5).unwrap(),
            ))
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        let (ids, actions) = all_ids();
        pool.send(
            &ids,
            ActionInput {
                discrete: &actions,
                continuous: &[],
            },
        )
        .unwrap();
        let mut received = 0;
        while received < ids.len() {
            received += pool.recv().unwrap().len();
        }
        assert_eq!(pool.max_claim(), 3);
    });
}

/// After one worker steps a run of 4 ids, the ready count is 4, so a
/// [`crate::BatchWaiter`] for batches of 4 sees a batch (issue #83: the
/// count moves by the run's length, not by one per run).
#[test]
fn test_async_waiter_sees_a_batch_after_one_run_ready() {
    within("waiter after a run", || {
        let mut pool = Pool::new(WalkConfig {}, options(AutoReset::NextStep))
            .unwrap()
            .async_mode(AsyncOptions::new(
                NonZeroUsize::MIN,
                NonZeroUsize::new(4).unwrap(),
            ))
            .unwrap();
        drop(pool.reset_all(Reseed::Keep).unwrap());
        let waiter = pool.waiter();
        pool.send(
            &[0, 1, 2, 3],
            ActionInput {
                discrete: &[1, 1, 1, 1],
                continuous: &[],
            },
        )
        .unwrap();
        assert!(waiter.wait_ready(core::time::Duration::from_secs(10)));
        assert_eq!(pool.max_claim(), 4, "one run of 4");
        assert_eq!(pool.recv().unwrap().len(), 4);
    });
}

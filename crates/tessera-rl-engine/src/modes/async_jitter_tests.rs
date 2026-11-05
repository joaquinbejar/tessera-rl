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
    let out = pool.outputs();
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
/// ten): `recv`, `reset_all` and `snapshot` each make progress on them, so
/// a caller retrying them is never stuck (TR M1 on #84).
#[test]
fn test_async_recv_reset_all_and_snapshot_make_progress_on_unclaimed_ids() {
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
    // `recv` refuses (three ids cannot fill a batch of ten), but steps them:
    // a snapshot then succeeds without any other call.
    let mut pool = make();
    for _ in 0..3 {
        assert!(matches!(
            pool.recv().map(|_| ()),
            Err(crate::EngineError::BatchUnreachable { .. })
        ));
    }
    let ok = (0..tries).any(|_| match pool.snapshot(5) {
        Ok(_) => true,
        Err(crate::EngineError::NotQuiescent) => {
            std::thread::yield_now();
            false
        }
        Err(e) => panic!("unexpected {e:?}"),
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
    // `snapshot` alone makes progress.
    let mut pool = make();
    let ok = (0..tries).any(|_| match pool.snapshot(5) {
        Ok(_) => true,
        Err(crate::EngineError::NotQuiescent) => {
            std::thread::yield_now();
            false
        }
        Err(e) => panic!("unexpected {e:?}"),
    });
    assert!(ok, "snapshot stepped the unclaimed ids");
}

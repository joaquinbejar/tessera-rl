// SPDX-License-Identifier: MIT OR Apache-2.0

//! External output storage (ADR-0084): CartPole and Snake (pixels and state observations) write byte-identical outputs
//! with and without a `Vec`-backed output allocator, after a reset and one
//! step, in every output region.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;
use std::sync::Arc;

use tessera_rl_classic::snake::{OBS_PIXELS, OBS_STATE};
use tessera_rl_classic::{CartPole, CartPoleConfig, Snake, SnakeConfig};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, NoMasks, OutputAllocator, OutputDtype, OutputField,
    OutputStorage, PoolOptions, Reseed, VecOutputAllocator,
};
use tessera_rl_num::identity::Seed;

/// Every field an `EnvPool` can lay out.
fn fields() -> Vec<OutputField> {
    let mut v = vec![
        OutputField::Rewards,
        OutputField::AgentTerminated,
        OutputField::AgentTruncated,
        OutputField::Died,
        OutputField::Spawned,
        OutputField::InvalidAgentEvent,
        OutputField::Alive,
        OutputField::Acting,
        OutputField::Generations,
        OutputField::EpisodeTerminated,
        OutputField::EpisodeTruncated,
        OutputField::EpisodeReturn,
        OutputField::EpisodeLength,
        OutputField::EpisodeIllegalActions,
        OutputField::EpisodeInvalidAgentEvents,
        OutputField::Masks,
    ];
    for leaf in 0..tessera_rl_engine::MAX_LEAVES {
        v.push(OutputField::Observation(leaf));
        v.push(OutputField::TerminalObservation(leaf));
    }
    v
}

/// The bytes of every region of `storage`, by field.
fn regions(storage: &OutputStorage) -> Vec<(OutputField, Vec<u8>)> {
    fields()
        .into_iter()
        .filter_map(|f| {
            let r = storage.region(f)?;
            let size = match r.dtype {
                OutputDtype::F32 | OutputDtype::U32 => 4,
                _ => 1,
            };
            // SAFETY: the region is `len * size` initialized bytes of a
            // storage kept alive by `storage`; no engine call runs.
            #[expect(unsafe_code, reason = "reading a borrowed output region")]
            let bytes = unsafe { core::slice::from_raw_parts(r.ptr, r.len * size) }.to_vec();
            Some((f, bytes))
        })
        .collect()
}

fn options(n: u32, mode: AutoReset, allocator: Option<Arc<dyn OutputAllocator>>) -> PoolOptions {
    PoolOptions {
        auto_reset: mode,
        output_allocator: allocator,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(21))
    }
}

/// The first legal action of each agent row (0 without masks), so the
/// step is accepted under `debug-checks` too.
fn legal(rows: usize, storage: &OutputStorage) -> Vec<i32> {
    let masks = regions(storage)
        .into_iter()
        .find(|(f, _)| *f == OutputField::Masks)
        .map(|(_, b)| b)
        .unwrap_or_default();
    let len = masks.len() / rows;
    (0..rows)
        .map(|r| {
            masks
                .get(r * len..(r + 1) * len)
                .and_then(|m| m.iter().position(|b| *b != 0))
                .map_or(0, |i| i32::try_from(i).unwrap())
        })
        .collect()
}

/// Builds a pool with `$build` with and without a `Vec`-backed allocator,
/// resets both, steps once with legal actions, and compares every region.
macro_rules! same_bytes {
    ($build:expr, $n:expr, $what:expr $(,)?) => {{
        let build = $build;
        let what: &str = $what;
        for mode in [AutoReset::NextStep, AutoReset::SameStep] {
            let alloc = Arc::new(VecOutputAllocator::new());
            let stats = alloc.stats();
            let mut plain = build(options($n, mode, None));
            let mut external = build(options($n, mode, Some(alloc)));
            assert_eq!(stats.allocations(), 1, "{what}");
            for p in [&mut plain, &mut external] {
                p.reset_all(Reseed::Keep).unwrap();
                let out = p.outputs().unwrap();
                let actions = legal(out.rewards().len(), &out.storage());
                p.step(ActionInput {
                    discrete: &actions,
                    continuous: &[],
                })
                .unwrap();
            }
            let a = regions(&plain.outputs().unwrap().storage());
            let b = regions(&external.outputs().unwrap().storage());
            assert!(a.len() >= 17, "{what} {mode:?}: {} regions", a.len());
            assert_eq!(a, b, "{what} {mode:?}");
            drop(external);
            assert_eq!(stats.drops(), 1, "{what}");
        }
    }};
}

#[test]
fn test_output_allocator_cartpole_outputs_byte_identical() {
    same_bytes!(
        |o| EnvPool::<CartPole, NoMasks>::new(CartPoleConfig::default(), o).unwrap(),
        8,
        "cartpole",
    );
}

#[test]
fn test_output_allocator_snake_pixels_outputs_byte_identical() {
    for (obs, what) in [(OBS_PIXELS, "snake pixels"), (OBS_STATE, "snake state")] {
        let cfg = SnakeConfig {
            size: 8,
            agents: 2,
            obs,
        };
        same_bytes!(
            |o| EnvPool::<Snake, Masked>::with_masks(cfg.clone(), o).unwrap(),
            3,
            what,
        );
    }
}

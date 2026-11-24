// SPDX-License-Identifier: MIT OR Apache-2.0

//! A search on a scratch pool with frame stacks and normalization never
//! allocates after construction (ADR-0078 item 7, REVIEW-headB-r1): root
//! evaluation, expansion, rollback of a failed call and a new root with
//! its stacks. One measuring test per binary (issue #43).

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

#[path = "support/counting.rs"]
mod counting;

use core::num::NonZeroU32;
use counting::window;

use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Record, Reseed, Transforms};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{Evaluator, EvaluatorError, Mcts, PuctConfig, SearchOptions, SearchPool};

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
}

fn transforms() -> Transforms {
    Transforms::new()
        .normalize("board", 3.0)
        .frame_stack("board", 4)
}

/// Uniform priors; from call `fail_at` on, a non-finite value (an error
/// that allocates nothing), so the call rolls back.
struct Scripted {
    calls: u32,
    fail_at: u32,
}

impl Evaluator<Connect4> for Scripted {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.calls += 1;
        priors.fill(1.0 / 7.0);
        values.fill(if self.calls >= self.fail_at {
            f64::NAN
        } else {
            0.0
        });
        Ok(())
    }
}

fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

fn test_search_with_frame_stacks_does_not_allocate() {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap()
    .with_transforms(transforms())
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let mut bytes = vec![0; p.transform_state_bytes()];
    p.snapshot_transforms(0, &mut bytes).unwrap();
    let context = p.pool_context().unwrap().transforms;
    let o = SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap());
    let mut pool = SearchPool::new_with_transforms(Connect4Config {}, o, &root, &bytes, |c, o| {
        EnvPool::with_masks(c, o)?.with_transforms(transforms())
    })
    .unwrap();
    pool.set_transform_context(context).unwrap();
    let mut cfg = PuctConfig::default();
    cfg.simulations = 200;
    cfg.batch_size = 8;
    let mut m = Mcts::new(pool, cfg).unwrap();
    let mut good = Scripted {
        calls: 0,
        fail_at: u32::MAX,
    };
    let mut failing = Scripted {
        calls: 0,
        fail_at: 3,
    };
    let n = allocations(|| {
        for _ in 0..2 {
            m.run(&mut good).unwrap();
            m.pool_mut()
                .set_root_with_transforms(&root, &bytes)
                .unwrap();
            failing.calls = 0;
            assert!(m.run(&mut failing).is_err(), "the call rolls back");
        }
    });
    assert_eq!(n, 0, "a stacked search allocated {n} times");
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_search_transforms_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_counter_detects_heap_use",
                test_counter_detects_heap_use,
            ),
            (
                "test_search_with_frame_stacks_does_not_allocate",
                test_search_with_frame_stacks_does_not_allocate,
            ),
        ],
    );
}

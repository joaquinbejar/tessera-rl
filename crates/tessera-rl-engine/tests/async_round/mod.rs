// SPDX-License-Identifier: MIT OR Apache-2.0

//! One Async round of the `async_instructions` bench, shared with the
//! `async_instructions_body` test so the bench body runs natively too (a
//! wrong buffer shape fails a test, not only a Valgrind run).

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use tessera_rl_engine::{
    ActionInput, AsyncOptions, AsyncPool, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
    WaitBackend,
};
use tessera_rl_num::identity::Seed;

use crate::common::{CountConfig, Counter};

/// Instances of the round.
pub const N: u32 = 256;
/// Ids per received batch.
pub const BATCH: usize = 64;

/// The pool, the ids sent and their action rows.
pub type Input = (AsyncPool<Counter, NoMasks>, Vec<u32>, Vec<i32>);

/// The round's setup: a reset pool on one worker (native wait, no spin),
/// every id, and one action per agent row of every id.
#[must_use]
pub fn round_256() -> Input {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(32).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    let mut probe = EnvPool::<Counter>::new(cfg.clone(), options).unwrap();
    probe.reset_all(Reseed::Keep).unwrap();
    // One action per agent row: the rewards buffer has one value per row.
    let rows = probe.outputs().unwrap().rewards().len();
    drop(probe);
    let sync = SyncOptions {
        spin: 0,
        wait: WaitBackend::Native,
        ..SyncOptions::new(NonZeroUsize::MIN)
    };
    let mut pool = EnvPool::new(cfg, options)
        .unwrap()
        .async_mode(AsyncOptions {
            sync,
            batch_size: NonZeroUsize::new(BATCH).unwrap(),
        })
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    (pool, ids, vec![1; rows])
}

/// One round: send every id, receive them in batches of [`BATCH`].
#[must_use]
pub fn round((mut pool, ids, actions): Input) -> Input {
    pool.send(
        black_box(&ids),
        ActionInput {
            discrete: black_box(&actions),
            continuous: &[],
        },
    )
    .unwrap();
    for _ in 0..ids.len() / BATCH {
        black_box(pool.recv().unwrap().len());
    }
    (pool, ids, actions)
}

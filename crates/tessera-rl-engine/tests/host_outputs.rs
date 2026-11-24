// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host-only outputs (ADR-0065 item 1): `aborted`, `abort_code` and
//! `pending_reset`, laid out for host pools only; `pending_reset` equals
//! the engine's pending state of every instance from construction on.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod host_support;

use core::num::NonZeroU32;

use host_support::{AGENTS, Book, BookConfig};
use tessera_rl_core::{Bool8, HostEnvironment};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, HostPool, OutputDtype, OutputField, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

type Pool = HostPool<Book, { Book::MAX_AGENTS }>;

fn pool(n: u32, mode: AutoReset, sync: bool) -> Pool {
    let mut o = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(5));
    o.auto_reset = mode;
    // Sync: chunks of 1 over 3 workers, so every worker runs chunks.
    o.chunk_size = NonZeroU32::MIN;
    let p = Pool::new(BookConfig::default(), o).unwrap();
    if sync {
        p.with_workers(SyncOptions::new(core::num::NonZeroUsize::new(3).unwrap()))
            .unwrap()
    } else {
        p
    }
}

fn pending(p: &Pool) -> Vec<bool> {
    p.outputs()
        .unwrap()
        .pending_reset()
        .iter()
        .map(|f| f.get())
        .collect()
}

fn hold(n: usize) -> Vec<i32> {
    vec![0; n * AGENTS * 2]
}

const MODES: [AutoReset; 3] = [
    AutoReset::NextStep,
    AutoReset::SameStep,
    AutoReset::Disabled,
];

#[test]
fn test_host_outputs_layout_dtypes_and_lengths() {
    let p = pool(8, AutoReset::NextStep, false);
    let out = p.outputs().unwrap();
    let storage = out.storage();
    for (field, dtype) in [
        (OutputField::Aborted, OutputDtype::Bool8),
        (OutputField::AbortCode, OutputDtype::U32),
        (OutputField::PendingReset, OutputDtype::Bool8),
    ] {
        let region = storage.region(field).unwrap();
        assert_eq!((region.len, region.dtype), (8, dtype), "{field:?}");
    }
    assert_eq!(out.aborted(), [Bool8::FALSE; 8]);
    assert_eq!(out.abort_code(), [0; 8]);
}

#[test]
fn test_host_outputs_pending_reset_exact_from_construction() {
    for mode in MODES {
        for sync in [false, true] {
            let mut p = pool(8, mode, sync);
            // Readable before any call: every instance is pending.
            assert_eq!(pending(&p), [true; 8], "{mode:?} {sync}");
            p.reset(&[1, 4], Reseed::Keep).unwrap();
            let mut want = [true; 8];
            want[1] = false;
            want[4] = false;
            assert_eq!(pending(&p), want, "{mode:?} {sync}");
            // Six instances were never reset: stepping raises in every mode,
            // and the outputs are unchanged.
            let err = p
                .step(ActionInput {
                    discrete: &hold(8),
                    continuous: &[],
                })
                .unwrap_err();
            assert!(matches!(err, EngineError::PendingReset { .. }), "{err:?}");
            assert_eq!(pending(&p), want, "{mode:?} {sync}");
            p.reset_all(Reseed::Keep).unwrap();
            assert_eq!(pending(&p), [false; 8], "{mode:?} {sync}");
        }
    }
}

#[test]
fn test_host_outputs_pending_reset_follows_episode_ends() {
    // The book terminates at step 5: NextStep and Disabled leave the
    // instance pending, SameStep resets it in the same call.
    for (mode, after_end) in [
        (AutoReset::NextStep, true),
        (AutoReset::Disabled, true),
        (AutoReset::SameStep, false),
    ] {
        for sync in [false, true] {
            let mut p = pool(3, mode, sync);
            p.reset_all(Reseed::Keep).unwrap();
            for t in 1..=5 {
                p.step(ActionInput {
                    discrete: &hold(3),
                    continuous: &[],
                })
                .unwrap();
                let want = if t == 5 { after_end } else { false };
                assert_eq!(pending(&p), [want; 3], "{mode:?} step {t}");
            }
            if mode == AutoReset::NextStep {
                p.step(ActionInput {
                    discrete: &hold(3),
                    continuous: &[],
                })
                .unwrap();
                assert_eq!(pending(&p), [false; 3]);
            }
        }
    }
}

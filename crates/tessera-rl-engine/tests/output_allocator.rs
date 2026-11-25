// SPDX-License-Identifier: MIT OR Apache-2.0

//! External output storage (ADR-0084): a pool built with an output
//! allocator writes the same bytes as one without, never moves its storage
//! after construction, refuses a failing allocator with a typed error, and
//! drops the allocator's owner once, on the thread of the last `Arc`.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use common::{CountConfig, Counter};
use tessera_rl_core::encoding::decode_from_slice;
use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{ObsLeaf, Observation, PrivilegedState, SpaceError};
use tessera_rl_engine::Transforms;
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, DecodedRecord, EngineError, EnvPool, ExternalAlloc,
    Masked, OutputAllocator, OutputDtype, OutputField, OutputStorage, PipelinedOptions,
    PoolOptions, Reseed, SyncOptions, VecOutputAllocator,
};
use tessera_rl_macros::Observation;
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

const N: u32 = 6;

fn options(allocator: Option<Arc<dyn OutputAllocator>>, mode: AutoReset) -> PoolOptions {
    PoolOptions {
        auto_reset: mode,
        // Two instances per chunk: three chunks, enough for Pipelined.
        chunk_size: NonZeroU32::new(2).unwrap(),
        output_allocator: allocator,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(9))
    }
}

fn pool(allocator: Option<Arc<dyn OutputAllocator>>, mode: AutoReset) -> Pool {
    EnvPool::with_masks(CountConfig::default(), options(allocator, mode)).unwrap()
}

fn step(p: &mut Pool) {
    let actions = vec![1; p.outputs().unwrap().rewards().len()];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

/// Every field a pool can lay out.
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

/// The bytes of every region `storage` has, by field.
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

/// The start of every region, by field.
fn pointers(storage: &OutputStorage) -> Vec<(OutputField, usize)> {
    fields()
        .into_iter()
        .filter_map(|f| Some((f, storage.region(f)?.ptr.addr())))
        .collect()
}

#[test]
fn test_output_allocator_pool_options_eq_by_identity_and_debug_flag() {
    let a: Arc<dyn OutputAllocator> = Arc::new(VecOutputAllocator::new());
    let b: Arc<dyn OutputAllocator> = Arc::new(VecOutputAllocator::new());
    let none = options(None, AutoReset::NextStep);
    let with_a = options(Some(Arc::clone(&a)), AutoReset::NextStep);
    assert_eq!(none, none.clone());
    assert_eq!(with_a, with_a.clone());
    assert_eq!(with_a, options(Some(Arc::clone(&a)), AutoReset::NextStep));
    assert_ne!(with_a, options(Some(b), AutoReset::NextStep));
    assert_ne!(with_a, none);
    assert_ne!(none, with_a);
    assert_ne!(none, options(None, AutoReset::SameStep));
    assert!(format!("{none:?}").contains("output_allocator: false"));
    assert!(format!("{with_a:?}").contains("output_allocator: true"));
    assert!(
        PoolOptions::new(NonZeroU32::MIN, Seed::new(0))
            .output_allocator
            .is_none()
    );
}

#[test]
fn test_output_allocator_outputs_byte_identical_with_and_without() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        let alloc = Arc::new(VecOutputAllocator::new());
        let stats = alloc.stats();
        let mut plain = pool(None, mode);
        let mut external = pool(Some(alloc), mode);
        assert_eq!(stats.allocations(), 1);
        for p in [&mut plain, &mut external] {
            p.reset_all(Reseed::Keep).unwrap();
            step(p);
        }
        let a = regions(&plain.outputs().unwrap().storage());
        let b = regions(&external.outputs().unwrap().storage());
        assert!(a.len() >= 17, "{mode:?}: {} regions", a.len());
        assert_eq!(a, b, "{mode:?}");
    }
}

#[test]
fn test_output_allocator_failing_allocator_refused_with_typed_error() {
    struct Nothing;
    impl OutputAllocator for Nothing {
        fn allocate(&self, _: usize, _: usize) -> Option<ExternalAlloc> {
            None
        }
    }
    let result = EnvPool::<Counter, Masked>::with_masks(
        CountConfig::default(),
        options(Some(Arc::new(Nothing)), AutoReset::NextStep),
    );
    assert_eq!(
        result.err(),
        Some(EngineError::OutputAllocation {
            reason: "the allocator returned no memory"
        })
    );
}

/// Asserts `p`'s storage is `storage`, with every region where it was.
fn same(what: &str, p: &Pool, storage: &Arc<OutputStorage>, ptrs: &[(OutputField, usize)]) {
    let now = p.outputs().unwrap().storage();
    assert!(Arc::ptr_eq(&now, storage), "{what}: another storage");
    assert_eq!(pointers(&now), ptrs, "{what}: a region moved");
}

/// ADR-0084 item 3b: every call writes into the same regions.
#[test]
fn test_output_allocator_storage_never_moves_across_calls() {
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let mut p = pool(Some(alloc), AutoReset::NextStep);
    let storage = p.outputs().unwrap().storage();
    let ptrs = pointers(&storage);
    assert!(ptrs.len() >= 17, "{} regions", ptrs.len());
    let same = |what: &str, p: &Pool| same(what, p, &storage, &ptrs);
    p.reset_all(Reseed::Keep).unwrap();
    same("reset_all", &p);
    p.reset(&[1], Reseed::Keep).unwrap();
    same("reset", &p);
    step(&mut p);
    same("step", &p);
    let snap = p.snapshot(0).unwrap();
    p.restore(1, &snap).unwrap();
    same("restore", &p);
    let all: Vec<_> = (0..N).map(|i| p.snapshot(i).unwrap()).collect();
    p.restore_all(&all).unwrap();
    same("restore_all", &p);
    let decoded: DecodedRecord<Counter> = decode_from_slice(&snap.to_bytes()).unwrap();
    p.import(&[2], &[decoded]).unwrap();
    same("import", &p);
    let decoded: Vec<DecodedRecord<Counter>> = all
        .iter()
        .map(|r| decode_from_slice(&r.to_bytes()).unwrap())
        .collect();
    p.import_all(&decoded).unwrap();
    same("import_all", &p);
    p.fork(0, &[3]).unwrap();
    same("fork", &p);
    p.fork_branched(0, &[1, 2], None).unwrap();
    same("fork_branched", &p);
    assert_eq!(stats.allocations(), 1, "the storage is allocated once");
    drop(p);
    assert_eq!(stats.drops(), 0, "the view still holds the storage");
    drop(storage);
    assert_eq!(stats.drops(), 1);
}

/// ADR-0084 item 3b: every mode conversion keeps the pool's storage. Each
/// conversion starts from an Inline pool, which they all require.
#[test]
fn test_output_allocator_storage_never_moves_across_modes() {
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let shared: Arc<dyn OutputAllocator> = alloc;
    let start = || {
        let mut p = pool(Some(Arc::clone(&shared)), AutoReset::NextStep);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p);
        let storage = p.outputs().unwrap().storage();
        let ptrs = pointers(&storage);
        (p, storage, ptrs)
    };

    let (p, storage, ptrs) = start();
    let mut p = p
        .with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    same("with_workers", &p, &storage, &ptrs);
    step(&mut p);
    same("step with workers", &p, &storage, &ptrs);

    let (p, storage, ptrs) = start();
    let piped = p
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    let mut p = piped.into_pool().map_err(|_| ()).unwrap();
    same("pipelined", &p, &storage, &ptrs);
    step(&mut p);
    same("step after pipelined", &p, &storage, &ptrs);

    let (p, storage, ptrs) = start();
    let asy = p
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        ))
        .unwrap();
    let mut p = asy.into_pool().map_err(|_| ()).unwrap();
    same("async_mode", &p, &storage, &ptrs);
    step(&mut p);
    same("step after async_mode", &p, &storage, &ptrs);

    assert_eq!(stats.allocations(), 3, "one allocation per pool");
}

/// ADR-0084 item 4: the owner is dropped once, by the thread that drops
/// the last `Arc`, never by a worker.
#[test]
fn test_output_allocator_owner_dropped_by_last_arc_thread_not_worker() {
    // The pool's `Arc` is the last one: dropped on this thread.
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let mut p = pool(Some(alloc), AutoReset::NextStep)
        .with_workers(SyncOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p);
    drop(p);
    assert_eq!(stats.drops(), 1);
    assert_eq!(stats.drop_threads(), vec![std::thread::current().id()]);

    // A view moved to another thread is the last one.
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let mut p = pool(Some(alloc), AutoReset::NextStep)
        .with_workers(SyncOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p);
    let view = p.outputs().unwrap().storage();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let holder = std::thread::spawn(move || {
        rx.recv().unwrap();
        drop(view);
        std::thread::current().id()
    });
    drop(p);
    assert_eq!(stats.drops(), 0, "the view keeps the owner alive");
    tx.send(()).unwrap();
    let holder = holder.join().unwrap();
    assert_eq!(stats.drops(), 1);
    assert_eq!(stats.drop_threads(), vec![holder]);
}

#[cfg(feature = "host")]
mod host_support;

/// Host pools take the allocator too (ADR-0084 item 1).
#[cfg(feature = "host")]
#[test]
fn test_output_allocator_host_pool_adopts_and_matches_plain() {
    use host_support::{Book, BookConfig};
    use tessera_rl_core::HostEnvironment;
    type Host = tessera_rl_engine::HostPool<Book, { Book::MAX_AGENTS }>;
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let opts = |a: Option<Arc<dyn OutputAllocator>>| PoolOptions {
        output_allocator: a,
        ..PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(5))
    };
    let mut plain = Host::new(BookConfig::default(), opts(None)).unwrap();
    let mut external = Host::new(BookConfig::default(), opts(Some(alloc))).unwrap();
    assert_eq!(stats.allocations(), 1);
    let storage = external.outputs().unwrap().storage();
    let ptrs = pointers(&storage);
    for p in [&mut plain, &mut external] {
        p.reset_all(Reseed::Keep).unwrap();
    }
    let host_fields = [
        OutputField::Aborted,
        OutputField::AbortCode,
        OutputField::PendingReset,
    ];
    let a = regions(&plain.outputs().unwrap().storage());
    let b = regions(&external.outputs().unwrap().storage());
    assert_eq!(a, b);
    for f in host_fields {
        assert_eq!(
            plain.outputs().unwrap().storage().region(f).map(|r| r.len),
            storage.region(f).map(|r| r.len),
            "{f:?}"
        );
    }
    assert_eq!(pointers(&external.outputs().unwrap().storage()), ptrs);
    drop(external);
    drop(storage);
    assert_eq!(stats.drops(), 1);
}

/// One privileged leaf: both counters.
#[derive(Observation)]
struct BothCounts {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [2]))]
    all: ObsLeaf<i32>,
}

impl PrivilegedState for Counter {
    fn state_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        BothCounts::space(cfg)
    }

    fn write_state(_: &CountConfig, state: &common::Counts, out: &mut ObsWriter<'_, '_, ()>) {
        if let Some(mut leaf) = out.leaf_mut(BothCounts::ALL) {
            leaf.set(0, state.value[0]);
            leaf.set(1, state.value[1]);
        }
    }
}

/// The name of the pool's first observation leaf.
fn first_leaf(p: &Pool) -> String {
    match p.observation_space().unwrap() {
        Space::Dict(entries) => entries.first().map_or_else(String::new, |e| e.0.clone()),
        _ => String::new(),
    }
}

fn transforms(p: &Pool) -> Transforms {
    Transforms::new().normalize(first_leaf(p), 1_000.0)
}

/// The two construction steps that lay the outputs out again, by name.
type Builder = fn(Pool) -> Result<Pool, EngineError>;

fn builders() -> [(&'static str, Builder); 2] {
    [
        ("with_transforms", |p| {
            let t = transforms(&p);
            p.with_transforms(t)
        }),
        ("with_privileged_state", Pool::with_privileged_state),
    ]
}

/// Without an export, each builder asks the allocator once more and
/// releases the storage it replaces.
#[test]
fn test_output_allocator_builders_reallocate_without_export() {
    let alloc = Arc::new(VecOutputAllocator::new());
    let stats = alloc.stats();
    let p = pool(Some(alloc), AutoReset::NextStep);
    assert_eq!(stats.allocations(), 1);
    let t = transforms(&p);
    let p = p.with_transforms(t).unwrap();
    assert_eq!((stats.allocations(), stats.drops()), (2, 1));
    let mut p = p.with_privileged_state().unwrap();
    assert_eq!((stats.allocations(), stats.drops()), (3, 2));
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p);
    drop(p);
    assert_eq!((stats.allocations(), stats.drops()), (3, 3));
}

/// ADR-0084 item 3b: with an allocator, a builder refuses to replace an
/// exported storage, and the exported storage is left as it was.
#[test]
fn test_output_allocator_builders_refuse_after_export() {
    for (name, build) in builders() {
        let alloc = Arc::new(VecOutputAllocator::new());
        let stats = alloc.stats();
        let p = pool(Some(alloc), AutoReset::NextStep);
        let exported = p.outputs().unwrap().storage();
        let ptrs = pointers(&exported);
        let bytes = regions(&exported);
        let generation = exported.generation();
        assert_eq!(
            build(p).err(),
            Some(EngineError::OutputAllocation {
                reason: "the output storage was exported and cannot be replaced"
            }),
            "{name}"
        );
        assert_eq!(stats.allocations(), 1, "{name}: no second allocation");
        assert_eq!(stats.drops(), 0, "{name}: the export keeps the storage");
        assert_eq!(pointers(&exported), ptrs, "{name}");
        assert_eq!(regions(&exported), bytes, "{name}");
        assert_eq!(exported.generation(), generation, "{name}");
        drop(exported);
        assert_eq!(stats.drops(), 1, "{name}");
    }
}

/// An export that was dropped again no longer blocks a builder; without
/// an allocator an export never does (the default path is unchanged).
#[test]
fn test_output_allocator_builders_accept_dropped_export_and_no_allocator() {
    for (name, build) in builders() {
        let alloc = Arc::new(VecOutputAllocator::new());
        let stats = alloc.stats();
        let p = pool(Some(alloc), AutoReset::NextStep);
        drop(p.outputs().unwrap().storage());
        let p = build(p).unwrap();
        assert_eq!((stats.allocations(), stats.drops()), (2, 1), "{name}");
        drop(p);

        let p = pool(None, AutoReset::NextStep);
        let exported = p.outputs().unwrap().storage();
        let p = build(p).unwrap();
        assert!(
            !Arc::ptr_eq(&exported, &p.outputs().unwrap().storage()),
            "{name}"
        );
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync mode never allocates after construction, on any thread (SPEC 5.5).
//!
//! A counting global allocator for this test binary only; test code, not
//! one of the audited `unsafe` modules. It counts on every thread while
//! enabled, so this binary holds a single test. The assertion is zero
//! allocations on every thread.
//!
//! Diagnostics for a failure (an unexplained intermittent count on CI,
//! job 124979520576 of run 41350533450):
//! - inside the allocator, a counted event first increments the count,
//!   then, under a per-thread reentrancy guard, claims a slot of a static
//!   preallocated table by an atomic index and stores raw data only: the
//!   size, the OS thread id, the kernel the thread is in (step, reset or
//!   none: park, wake and the engine's own code), how many kernel calls the
//!   thread has made, and up to [`FRAMES`] return addresses from
//!   `backtrace::trace_unsynchronized`, which does not allocate. No lock,
//!   no formatting, no symbolization;
//! - the unwinder is warmed up before counting starts;
//! - only after counting stops are the addresses symbolized and the thread
//!   names read, and the report goes into the failure message.
//!
//! The capture itself perturbs the scheduling it observes; a count of zero
//! is the result either way.

use core::cell::Cell;
use core::ffi::c_void;
use core::num::{NonZeroU32, NonZeroUsize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

mod common;

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, HeaderView, InvariantViolation, SpaceError, StepOut,
    Validate,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// Events recorded with their stacks.
const EVENTS: usize = 16;
/// Return addresses kept per event.
const FRAMES: usize = 32;

/// One recorded allocation, all atomics so that writing needs no lock.
struct Event {
    size: AtomicUsize,
    tid: AtomicU64,
    kernel: AtomicU32,
    kernel_calls: AtomicU32,
    frames: [AtomicUsize; FRAMES],
    depth: AtomicUsize,
}

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "the initializer of the static table, one copy per slot"
)]
const EMPTY: Event = Event {
    size: AtomicUsize::new(0),
    tid: AtomicU64::new(0),
    kernel: AtomicU32::new(0),
    kernel_calls: AtomicU32::new(0),
    frames: [const { AtomicUsize::new(0) }; FRAMES],
    depth: AtomicUsize::new(0),
};

static TABLE: [Event; EVENTS] = [EMPTY; EVENTS];
static NEXT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Set while this thread records an event: nested allocations are not
    /// counted.
    static RECORDING: Cell<bool> = const { Cell::new(false) };
    /// The kernel this thread is in: 0 none, 1 `step`, 2 `reset`.
    static KERNEL: Cell<u32> = const { Cell::new(0) };
    /// Kernel calls this thread has made.
    static KERNEL_CALLS: Cell<u32> = const { Cell::new(0) };
}

/// The OS id of the current thread (0 where not available).
fn os_tid() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: gettid has no preconditions and does not allocate.
        #[expect(unsafe_code, reason = "test-only thread id for diagnostics")]
        let tid = unsafe { libc::gettid() };
        u64::try_from(tid).unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Counts one allocation of `size` bytes and records it.
fn count(size: usize) {
    if !COUNTING.load(Ordering::Relaxed) || RECORDING.with(|r| r.replace(true)) {
        return;
    }
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let slot = NEXT.fetch_add(1, Ordering::Relaxed);
    if let Some(e) = TABLE.get(slot) {
        e.size.store(size, Ordering::Relaxed);
        e.tid.store(os_tid(), Ordering::Relaxed);
        e.kernel.store(KERNEL.with(Cell::get), Ordering::Relaxed);
        e.kernel_calls
            .store(KERNEL_CALLS.with(Cell::get), Ordering::Relaxed);
        let mut depth = 0;
        let mut keep = |frame: &backtrace::Frame| {
            if let Some(f) = e.frames.get(depth) {
                f.store(frame.ip() as usize, Ordering::Relaxed);
                depth += 1;
                true
            } else {
                false
            }
        };
        // SAFETY: the unsynchronized walk may race only with another
        // unsynchronized walk; walks of different threads touch their own
        // stacks, and the callback neither allocates nor unwinds.
        #[expect(unsafe_code, reason = "test-only allocation-free stack walk")]
        unsafe {
            backtrace::trace_unsynchronized(&mut keep);
        }
        e.depth.store(depth, Ordering::Relaxed);
    }
    RECORDING.with(|r| r.set(false));
}

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counters are atomics and const-initialized thread locals, which never
// allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// The engine's counter test environment, with its kernels tagged for the
/// diagnostics (the shared `Counter` stays untouched: benches use it).
struct Tagged;

/// Tags the thread as inside kernel `kind` until dropped.
struct InKernel;

impl InKernel {
    fn enter(kind: u32) -> Self {
        KERNEL.with(|k| k.set(kind));
        KERNEL_CALLS.with(|c| c.set(c.get().wrapping_add(1)));
        Self
    }
}

impl Drop for InKernel {
    fn drop(&mut self) {
        KERNEL.with(|k| k.set(0));
    }
}

impl Environment for Tagged {
    type State = <Counter as Environment>::State;
    type Config = <Counter as Environment>::Config;
    type Obs = <Counter as Environment>::Obs;
    type Action = <Counter as Environment>::Action;
    type Header = <Counter as Environment>::Header;

    const MAX_AGENTS: usize = Counter::MAX_AGENTS;
    const TIER: tessera_rl_core::Tier = Counter::TIER;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::observation_space(cfg)
    }

    fn action_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        Counter::action_space(cfg)
    }

    fn initial_agents(cfg: &CountConfig) -> usize {
        Counter::initial_agents(cfg)
    }

    fn reset(cfg: &CountConfig, rng: &mut StepRng, state: &mut Counts) {
        let _k = InKernel::enter(2);
        Counter::reset(cfg, rng, state);
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let _k = InKernel::enter(1);
        Counter::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &CountConfig,
        state: &Counts,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, Self::Obs>,
    ) {
        Counter::observe(cfg, state, agent, rng, out);
    }
}

impl Validate for Tagged {
    fn validate(cfg: &CountConfig, state: &Counts) -> Result<(), InvariantViolation> {
        Counter::validate(cfg, state)
    }

    fn validate_record(
        cfg: &CountConfig,
        header: &HeaderView<'_>,
        state: &Counts,
    ) -> Result<(), InvariantViolation> {
        Counter::validate_record(cfg, header, state)
    }
}

impl ActionMask for Tagged {
    fn action_mask(cfg: &CountConfig, state: &Counts, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        Counter::action_mask(cfg, state, agent, out);
    }

    fn noop(cfg: &CountConfig, state: &Counts, agent: AgentSlot, out: &mut ActionWriter<'_>) {
        Counter::noop(cfg, state, agent, out);
    }
}

/// The name of OS thread `tid` (Linux), read after counting stops.
fn thread_name(tid: u64) -> String {
    std::fs::read_to_string(format!("/proc/self/task/{tid}/comm"))
        .map_or_else(|_| "?".to_owned(), |s| s.trim().to_owned())
}

/// The report of every recorded event, symbolized.
fn report() -> String {
    let mut out = String::new();
    let recorded = NEXT.load(Ordering::Relaxed).min(EVENTS);
    for e in TABLE.iter().take(recorded) {
        let tid = e.tid.load(Ordering::Relaxed);
        let kernel = match e.kernel.load(Ordering::Relaxed) {
            1 => "in step",
            2 => "in reset",
            _ => "outside the kernels (park, wake, engine code)",
        };
        let _ = writeln!(
            out,
            "{} bytes on thread {tid} ({}), {kernel}, after {} kernel calls on this thread:",
            e.size.load(Ordering::Relaxed),
            thread_name(tid),
            e.kernel_calls.load(Ordering::Relaxed),
        );
        let depth = e.depth.load(Ordering::Relaxed).min(FRAMES);
        for f in e.frames.iter().take(depth) {
            let ip = f.load(Ordering::Relaxed);
            let mut name = String::from("?");
            backtrace::resolve(ip as *mut c_void, |symbol| {
                if let Some(n) = symbol.name() {
                    name = n.to_string();
                }
            });
            let _ = writeln!(out, "    {ip:#x} {name}");
        }
    }
    out
}

#[test]
fn test_sync_reset_and_step_do_not_allocate_on_any_thread() {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let mut pool = EnvPool::<Tagged, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![1i32; 128];
    // Warm up: the first epoch may initialize thread-local runtime state.
    pool.reset_all(Reseed::Keep).unwrap();
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    // Warm up the unwinder (its caches), outside the counted window.
    // SAFETY: as in `count`; no allocation happens in the callback.
    #[expect(unsafe_code, reason = "test-only allocation-free stack walk")]
    unsafe {
        backtrace::trace_unsynchronized(|_| true);
    }
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..20 {
        let _ = std::hint::black_box(pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        }));
    }
    let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
    COUNTING.store(false, Ordering::Relaxed);
    let n = ALLOCATIONS.load(Ordering::Relaxed);
    if n > 0 {
        // Symbolize while the workers still exist (their names).
        let details = report();
        drop(pool);
        panic!("{n} allocations in the measured window:\n{details}");
    }
}

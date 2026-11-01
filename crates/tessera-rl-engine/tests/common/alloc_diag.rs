// SPDX-License-Identifier: MIT OR Apache-2.0

//! A counting global allocator with bounded raw diagnostics, for the
//! allocation tests of the Sync scheduler (`alloc_sync.rs`) and the probes
//! of the diagnostics themselves (`alloc_sync_probe.rs`). Each binary that
//! includes this module installs the allocator.
//!
//! While counting is on, every allocation on every thread is counted
//! first. Then, under a per-thread reentrancy guard, it claims a slot of a
//! static table by an atomic index and stores raw data only: the size, a
//! thread tag, up to [`NAME`] bytes of the thread's name, the kernel the
//! thread is in (step, reset, or none: park, wake and the engine's own
//! code) and its kernel call count. No lock, formatting, backtrace or
//! allocation happens in the allocator; [`report`] formats the events
//! after counting stops. Events past [`EVENTS`] are counted but not
//! recorded, and the report says so.

use core::cell::Cell;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use super::common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, HeaderView, InvariantViolation, SpaceError, StepOut,
    Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Events recorded in full.
pub const EVENTS: usize = 16;
/// Bytes of the thread name kept per event.
pub const NAME: usize = 24;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static NEXT: AtomicUsize = AtomicUsize::new(0);
static THREAD_TAGS: AtomicU32 = AtomicU32::new(1);

/// One recorded allocation, all atomics so that writing needs no lock.
pub struct Event {
    size: AtomicUsize,
    tag: AtomicU32,
    kernel: AtomicU32,
    kernel_calls: AtomicU32,
    name: [AtomicU8; NAME],
    name_len: AtomicUsize,
}

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "the initializer of the static table, one copy per slot"
)]
const EMPTY: Event = Event {
    size: AtomicUsize::new(0),
    tag: AtomicU32::new(0),
    kernel: AtomicU32::new(0),
    kernel_calls: AtomicU32::new(0),
    name: [const { AtomicU8::new(0) }; NAME],
    name_len: AtomicUsize::new(0),
};

static TABLE: [Event; EVENTS] = [EMPTY; EVENTS];

thread_local! {
    /// Set while this thread records an event: nested allocations are not
    /// counted.
    pub static RECORDING: Cell<bool> = const { Cell::new(false) };
    /// This thread's tag (0 until its first counted allocation).
    static TAG: Cell<u32> = const { Cell::new(0) };
    /// The kernel this thread is in: 0 none, 1 `step`, 2 `reset`.
    static KERNEL: Cell<u32> = const { Cell::new(0) };
    /// Kernel calls this thread has made.
    static KERNEL_CALLS: Cell<u32> = const { Cell::new(0) };
}

/// When set, [`Tagged`]'s step kernel allocates (the probes' known
/// allocation).
pub static ALLOCATE_IN_STEP: AtomicBool = AtomicBool::new(false);

/// Starts a counted window: the count and the table are cleared.
pub fn start() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    NEXT.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Release);
}

/// Ends the counted window; returns the count.
pub fn stop() -> u64 {
    COUNTING.store(false, Ordering::Release);
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Counts one allocation of `size` bytes and records it.
fn count(size: usize) {
    if !COUNTING.load(Ordering::Acquire) || RECORDING.with(|r| r.replace(true)) {
        return;
    }
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let slot = NEXT.fetch_add(1, Ordering::Relaxed);
    if let Some(e) = TABLE.get(slot) {
        let tag = TAG.with(|t| {
            if t.get() == 0 {
                t.set(THREAD_TAGS.fetch_add(1, Ordering::Relaxed));
            }
            t.get()
        });
        e.size.store(size, Ordering::Relaxed);
        e.tag.store(tag, Ordering::Relaxed);
        e.kernel.store(KERNEL.with(Cell::get), Ordering::Relaxed);
        e.kernel_calls
            .store(KERNEL_CALLS.with(Cell::get), Ordering::Relaxed);
        // The current thread's handle: set at spawn for std threads, so
        // reading the name does not allocate (and, under the guard, an
        // allocation here would not be counted).
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("").as_bytes();
        let n = name.len().min(NAME);
        for (dst, &b) in e.name.iter().zip(name.iter().take(n)) {
            dst.store(b, Ordering::Relaxed);
        }
        e.name_len.store(n, Ordering::Relaxed);
    }
    RECORDING.with(|r| r.set(false));
}

/// The counting allocator.
pub struct Counting;

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

/// One recorded event, read back after the window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded {
    /// Bytes.
    pub size: usize,
    /// Thread tag.
    pub tag: u32,
    /// Thread name.
    pub thread: String,
    /// 0 none, 1 step, 2 reset.
    pub kernel: u32,
    /// Kernel calls of the thread at that point.
    pub kernel_calls: u32,
}

/// The recorded events of the last window, and how many were dropped past
/// the table. Call after [`stop`].
pub fn recorded() -> (Vec<Recorded>, u64) {
    let total = ALLOCATIONS.load(Ordering::Relaxed);
    let n = NEXT.load(Ordering::Relaxed).min(EVENTS);
    let events = TABLE
        .iter()
        .take(n)
        .map(|e| {
            let len = e.name_len.load(Ordering::Relaxed).min(NAME);
            let bytes: Vec<u8> = e
                .name
                .iter()
                .take(len)
                .map(|b| b.load(Ordering::Relaxed))
                .collect();
            Recorded {
                size: e.size.load(Ordering::Relaxed),
                tag: e.tag.load(Ordering::Relaxed),
                thread: String::from_utf8_lossy(&bytes).into_owned(),
                kernel: e.kernel.load(Ordering::Relaxed),
                kernel_calls: e.kernel_calls.load(Ordering::Relaxed),
            }
        })
        .collect::<Vec<_>>();
    let dropped = total.saturating_sub(u64::try_from(events.len()).unwrap_or(u64::MAX));
    (events, dropped)
}

/// A readable report of the last window. Call after [`stop`].
pub fn report() -> String {
    let (events, dropped) = recorded();
    let mut out = String::new();
    for e in &events {
        let kernel = match e.kernel {
            1 => "in step",
            2 => "in reset",
            _ => "outside the kernels (park, wake, engine code)",
        };
        let _ = writeln!(
            out,
            "{} bytes on thread #{} {:?}, {kernel}, after {} kernel calls on this thread",
            e.size, e.tag, e.thread, e.kernel_calls
        );
    }
    if dropped > 0 {
        let _ = writeln!(
            out,
            "... and {dropped} more, not recorded (table of {EVENTS})"
        );
    }
    out
}

/// The engine's counter test environment, with its kernels tagged for the
/// diagnostics (the shared `Counter` stays untouched: benches use it).
pub struct Tagged;

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
        if ALLOCATE_IN_STEP.load(Ordering::Relaxed) {
            drop(std::hint::black_box(vec![0u8; 3]));
        }
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

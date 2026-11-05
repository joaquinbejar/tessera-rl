// SPDX-License-Identifier: MIT OR Apache-2.0

//! The worker gang and its job protocol (ADR-0005, ADR-0047).
//!
//! # Startup
//!
//! Workers are spawned with fallible `Builder::spawn`; if a spawn fails,
//! the workers already started are shut down and joined. Every worker then
//! reports `ready` (after pinning and entering the canonical FP
//! environment, both caught), and `Gang::new` waits for all of them: a
//! failed worker shuts the gang down and returns a typed error, so a job
//! can never wait for a worker that does not exist.
//!
//! # Jobs by sequence, at most two in flight
//!
//! Jobs are numbered by a `u64` sequence starting at 1 (checked; overflow
//! is `SequenceExhausted`). Job `s` uses slot `s & 1`.
//!
//! 1. `submit` writes the slot's `panicked = false`, `pending = workers`
//!    and job pointer, then stores `published = s`, then increments the
//!    wake word, and wakes the workers only if some are parked (all
//!    SeqCst). A slot is reused for `s + 2` only after `wait` observed its
//!    `pending == 0`, so at most two jobs are ever in flight.
//! 2. Each worker keeps the next sequence it must run and runs every
//!    published job in order before it may sleep. To sleep it loads the
//!    wake word, then `published`, and blocks on the old word value; the
//!    publisher stores `published` before it changes the word, so a
//!    wake-up cannot be lost (a changed word only means "go and look").
//! 3. A worker decrements the slot's `pending` once per job, and that
//!    decrement is its last access to the job. The worker that brings it to
//!    zero wakes the waiter if one is parked.
//! 4. `wait` waits for the oldest job's `pending == 0` with SeqCst, which
//!    makes every worker's writes visible, and frees its slot.
//!
//! Waits spin for a bounded budget, then block. Blocking registers in a
//! parked counter with an RMW and then rechecks the condition; a waker
//! makes its change, then reads the counter with an RMW (`fetch_add(0)`).
//! Both RMWs are on the same counter, so whichever comes second in its
//! modification order sees the other's side: the waiter sees the change or
//! the waker sees the waiter. A waker therefore skips the wake syscall when
//! nobody is parked, without relying on store-load ordering. No waker is stored
//! for a job without one: with the park backend such a job's waiter polls
//! with `park_timeout(100 µs)`. A job with a waker (Sync's `run`, whose
//! submitter is its waiter) is unparked directly, as before.
//!
//! # Panics and a dead worker
//!
//! A panic in the job is caught and its payload disposed of (`sys::panics`)
//! before the decrement, and nothing that can panic runs after it. The
//! slot's `panicked` flag reports it to that job's wait, and the gang's
//! `poisoned` flag stays set until `clear_poison`. A job queued behind a
//! panicking one still runs.
//!
//! If anything else in the worker panics (outside the job's catch), the
//! backstop marks the gang `dead` (Release), makes the missing decrement for
//! the current job (reporting a panic), and the worker keeps draining:
//! until shutdown it consumes every published sequence and decrements its
//! `pending` without running the job, reporting a panic for it. No wait can
//! therefore block on a worker that stopped decrementing, even when a job
//! is published while the worker dies. `submit` returns `WorkerDied` once
//! it sees `dead`; after that only drop remains.
//!
//! Drop waits for every job in flight, then sets `shutdown`, changes the
//! wake word, wakes and joins every worker.
//!
//! # Unsafe invariants
//!
//! - A job pointer is valid from its `published` store until its slot's
//!   `pending` reaches zero: `submit`'s caller guarantees the `Job` outlives
//!   the matching `wait`. Workers read it only after seeing `published`
//!   reach its sequence, and never after their decrement.
//! - The job's data pointer is a `&F` for the `F` its `call` was
//!   instantiated with (`trampoline::<F>`), valid for the same window.
//! - Workers run concurrently, so `F: Sync`; the job itself is responsible
//!   for giving each worker disjoint data (see `engine::buffer`).

use core::num::NonZeroUsize;

use crate::error::EngineError;

use super::affinity::pin_current;
use super::panics::catch;
use super::sync::{
    Arc, AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering, spin_loop, thread,
};

/// How long a waiter without a stored waker sleeps per poll on the park
/// backend (ADR-0047).
#[cfg(not(loom))]
const PARK_POLL: core::time::Duration = core::time::Duration::from_micros(100);

/// Options of a gang.
#[derive(Debug, Clone)]
pub(crate) struct GangOptions {
    /// Number of workers.
    pub(crate) workers: NonZeroUsize,
    /// Spins before parking.
    pub(crate) spin: u32,
    /// Logical cores to pin workers to, in order (already validated);
    /// `None` leaves them unpinned.
    pub(crate) pin: Option<Vec<usize>>,
    /// The native wait primitive, or `None` for park and unpark.
    #[cfg(not(loom))]
    pub(crate) native: Option<super::Native>,
    /// Under loom: block with a futex-like mock instead of park.
    #[cfg(loom)]
    pub(crate) mock_futex: bool,
    /// Test-only fault injection.
    #[cfg(test)]
    pub(crate) faults: Faults,
}

/// Test-only fault injection for startup.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Faults {
    /// Fail the spawn of this worker.
    pub(crate) spawn: Option<usize>,
    /// Panic during the startup of this worker.
    pub(crate) startup: Option<usize>,
    /// Panic outside the job's catch, before the decrement, in this
    /// worker's first job (exercises the backstop).
    pub(crate) die_in_epoch: Option<usize>,
    /// Like `die_in_epoch`, but before the worker reads the job (and so
    /// before it clones the job's waker).
    pub(crate) die_before_job: Option<usize>,
}

/// A type-erased job: `call(data, worker)` on every worker.
pub(crate) struct Job {
    call: unsafe fn(*const (), usize),
    data: *const (),
    waker: Option<thread::Thread>,
}

impl Job {
    /// A job calling `f(worker)`. `waker` is the thread to unpark when it
    /// completes (park backend); `None` makes its waiter poll.
    pub(crate) fn new<F: Fn(usize) + Sync>(f: &F, waker: Option<thread::Thread>) -> Self {
        Self {
            call: trampoline::<F>,
            data: core::ptr::from_ref(f).cast::<()>(),
            waker,
        }
    }
}

/// A futex-like wait for loom, behind the same `wait` and `wake`
/// interface as the native primitive (loom has no address wait and no
/// `park_timeout`).
#[cfg(loom)]
struct MockFutex {
    lock: super::sync::Mutex<()>,
    cv: super::sync::Condvar,
}

#[cfg(loom)]
impl MockFutex {
    fn wait(&self, word: &AtomicU32, expected: u32) {
        let guard = self
            .lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if word.load(Ordering::SeqCst) == expected {
            drop(
                self.cv
                    .wait(guard)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }
    }

    fn wake(&self) {
        // Taking the lock orders this wake after a waiter's check.
        drop(
            self.lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        self.cv.notify_all();
    }
}

/// How waiting threads block and are woken.
#[derive(Debug, Clone, Copy)]
enum Backend {
    /// `thread::park` and `unpark` (portable).
    Park,
    /// Address waits on the 32-bit words below.
    #[cfg(not(loom))]
    Native(super::Native),
    /// The loom futex mock.
    #[cfg(loom)]
    Mock,
}

/// One of the two job slots.
struct JobSlot {
    job: AtomicPtr<Job>,
    pending: AtomicU32,
    /// Waiters parked on `pending`.
    waiters: AtomicU32,
    panicked: AtomicBool,
}

impl JobSlot {
    fn new() -> Self {
        Self {
            job: AtomicPtr::new(core::ptr::null_mut()),
            pending: AtomicU32::new(0),
            waiters: AtomicU32::new(0),
            panicked: AtomicBool::new(false),
        }
    }
}

struct Shared {
    published: AtomicU64,
    /// The wake word workers block on; only its change matters.
    word: AtomicU32,
    /// Workers parked on `word`.
    parked: AtomicU32,
    slots: [JobSlot; 2],
    shutdown: AtomicBool,
    poisoned: AtomicBool,
    dead: AtomicBool,
    ready: AtomicU32,
    failed: AtomicUsize,
    fp_resets: AtomicU64,
    creator: thread::Thread,
    spin: u32,
    backend: Backend,
    #[cfg(loom)]
    mock: MockFutex,
}

/// Persistent worker threads running published jobs in sequence.
pub(crate) struct Gang {
    shared: Arc<Shared>,
    workers: Vec<(thread::Thread, Option<thread::JoinHandle<()>>)>,
    /// `workers.len()` once every worker is spawned, checked to fit a u32.
    count: u32,
    /// The last sequence submitted, and the last one waited for.
    submitted: u64,
    completed: u64,
    /// Whether each slot's job has no waker (its waiter polls on park).
    polled: [bool; 2],
}

impl core::fmt::Debug for Gang {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Gang")
            .field("workers", &self.workers.len())
            .finish_non_exhaustive()
    }
}

#[cfg(not(loom))]
std::thread_local! {
    static IN_WORKER: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[cfg(not(loom))]
fn in_worker() -> bool {
    IN_WORKER.with(core::cell::Cell::get)
}

#[cfg(loom)]
fn in_worker() -> bool {
    false
}

/// Calls the job's closure for `worker`.
///
/// # Safety
///
/// `data` must point to a live `F` (the unsafe invariants of the module).
#[expect(unsafe_code, reason = "audited module: type-erased job call")]
unsafe fn trampoline<F: Fn(usize) + Sync>(data: *const (), worker: usize) {
    // SAFETY: `Job::new::<F>` stores `data = &f as *const F`, and the
    // submitter keeps `f` alive until every worker has decremented the
    // slot's `pending`, after this call.
    let f = unsafe { &*data.cast::<F>() };
    f(worker);
}

/// The slot of sequence `seq`.
fn slot_index(seq: u64) -> usize {
    usize::from(seq & 1 == 1)
}

impl Shared {
    /// Waits until `done(word)` holds: a bounded spin, then blocking. Used
    /// at startup, where the creator is the only waiter.
    fn wait_on(&self, word: &AtomicU32, mut done: impl FnMut(u32) -> bool) -> u32 {
        let mut spins = 0;
        loop {
            let value = word.load(Ordering::Acquire);
            if done(value) {
                return value;
            }
            if spins < self.spin {
                spins += 1;
                spin_loop();
                continue;
            }
            self.block(word, value, false);
        }
    }

    /// Waits until `ready()` holds: a bounded spin, then blocking on `word`
    /// with registration in `parked` and a recheck (SeqCst). The word is
    /// loaded before `ready` is rechecked, so a change made after the
    /// condition became true is never missed. `poll` makes the park backend
    /// sleep with a timeout instead of waiting for an unpark.
    fn wait_until(
        &self,
        word: &AtomicU32,
        parked: &AtomicU32,
        mut ready: impl FnMut() -> bool,
        poll: bool,
    ) {
        let mut spins = 0;
        loop {
            if ready() {
                return;
            }
            if spins < self.spin {
                spins += 1;
                spin_loop();
                continue;
            }
            let value = word.load(Ordering::SeqCst);
            parked.fetch_add(1, Ordering::SeqCst);
            if ready() {
                parked.fetch_sub(1, Ordering::SeqCst);
                return;
            }
            self.block(word, value, poll);
            parked.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Blocks once while `word == value`; may return spuriously.
    fn block(&self, word: &AtomicU32, value: u32, poll: bool) {
        match self.backend {
            Backend::Park => {
                let _ = (word, value);
                if poll {
                    #[cfg(not(loom))]
                    thread::park_timeout(PARK_POLL);
                    #[cfg(loom)]
                    thread::yield_now();
                } else {
                    thread::park();
                }
            }
            #[cfg(not(loom))]
            Backend::Native(native) => native.wait(word, value),
            #[cfg(loom)]
            Backend::Mock => self.mock.wait(word, value),
        }
    }

    /// Wakes the threads waiting on `word`: `parked` lists them for the
    /// park backend; the other backends wake by address.
    fn wake<'a>(&self, word: &AtomicU32, parked: impl IntoIterator<Item = &'a thread::Thread>) {
        match self.backend {
            Backend::Park => {
                // Park wakes threads, not addresses.
                let _ = word;
                for thread in parked {
                    thread.unpark();
                }
            }
            #[cfg(not(loom))]
            Backend::Native(native) => native.wake_all(word),
            #[cfg(loom)]
            Backend::Mock => {
                let _ = word;
                self.mock.wake();
            }
        }
    }
}

impl Gang {
    /// Spawns the workers and waits until every one is ready. Engines may
    /// not be nested inside a worker.
    pub(crate) fn new(options: &GangOptions) -> Result<Self, EngineError> {
        if in_worker() {
            return Err(EngineError::NestedEngine);
        }
        // `pending` and `ready` count workers in a u32.
        let count = u32::try_from(options.workers.get()).map_err(|_| EngineError::Options {
            option: "workers",
            reason: "more than u32::MAX workers",
        })?;
        let shared = Arc::new(Shared {
            published: AtomicU64::new(0),
            word: AtomicU32::new(0),
            parked: AtomicU32::new(0),
            slots: [JobSlot::new(), JobSlot::new()],
            shutdown: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            ready: AtomicU32::new(0),
            failed: AtomicUsize::new(usize::MAX),
            fp_resets: AtomicU64::new(0),
            creator: thread::current(),
            spin: options.spin,
            #[cfg(not(loom))]
            backend: options.native.map_or(Backend::Park, Backend::Native),
            #[cfg(loom)]
            backend: if options.mock_futex {
                Backend::Mock
            } else {
                Backend::Park
            },
            #[cfg(loom)]
            mock: MockFutex {
                lock: super::sync::Mutex::new(()),
                cv: super::sync::Condvar::new(),
            },
        });
        // Dropping `gang` on any early return shuts down and joins the
        // workers spawned so far (the construction guard).
        let mut gang = Self {
            shared,
            workers: Vec::with_capacity(options.workers.get()),
            count,
            submitted: 0,
            completed: 0,
            polled: [false; 2],
        };
        for w in 0..options.workers.get() {
            let shared = Arc::clone(&gang.shared);
            let core = options.pin.as_ref().and_then(|cores| cores.get(w).copied());
            #[cfg(test)]
            let faults = options.faults;
            #[cfg(test)]
            if faults.spawn == Some(w) {
                return Err(EngineError::WorkerStartup { worker: w });
            }
            let spawned = thread::Builder::new()
                .name(format!("tessera-worker-{w}"))
                .spawn(move || {
                    #[cfg(test)]
                    let (fail, die) = (
                        faults.startup == Some(w),
                        Death {
                            after_waker: faults.die_in_epoch == Some(w),
                            before_job: faults.die_before_job == Some(w),
                        },
                    );
                    #[cfg(not(test))]
                    let (fail, die) = (false, Death::default());
                    worker_main(&shared, w, core, fail, die);
                });
            match spawned {
                Ok(handle) => gang.workers.push((handle.thread().clone(), Some(handle))),
                Err(_) => return Err(EngineError::WorkerStartup { worker: w }),
            }
        }
        let shared = Arc::clone(&gang.shared);
        let n = gang.count;
        // Acquire (in `wait_on`): pairs with each worker's Release on `ready`.
        shared.wait_on(&shared.ready, |ready| ready == n);
        let failed = shared.failed.load(Ordering::Relaxed);
        if failed != usize::MAX {
            drop(gang);
            return Err(EngineError::WorkerStartup { worker: failed });
        }
        Ok(gang)
    }

    /// Number of workers.
    pub(crate) fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Jobs submitted and not yet waited for.
    pub(crate) fn in_flight(&self) -> u64 {
        // `completed <= submitted` always holds; `wrapping_sub` cannot wrap.
        self.submitted.wrapping_sub(self.completed)
    }

    /// Runs `f(worker)` on every worker and waits for all of them. Returns
    /// whether any call panicked, or an error if a worker died earlier.
    /// Needs no job in flight.
    pub(crate) fn run<F: Fn(usize) + Sync>(&mut self, f: &F) -> Result<bool, EngineError> {
        if self.in_flight() != 0 {
            return Err(EngineError::GangQueue {
                in_flight: self.in_flight(),
            });
        }
        let job = Job::new(f, Some(thread::current()));
        // SAFETY: `job` and `f` live on this stack until `wait` below
        // returns, which happens only after every worker has decremented
        // the slot's `pending`; nothing between the two calls can panic.
        #[expect(unsafe_code, reason = "audited module: job publication")]
        unsafe {
            self.submit(&job)?;
        }
        self.wait()
    }

    /// Publishes `job` without waiting for it (at most two in flight).
    ///
    /// # Safety
    ///
    /// `job`, and the closure it points to, must stay valid until the
    /// `wait` (or `try_wait`) that returns this job's result.
    #[expect(unsafe_code, reason = "audited module: job publication")]
    pub(crate) unsafe fn submit(&mut self, job: *const Job) -> Result<(), EngineError> {
        let shared = &*self.shared;
        if shared.dead.load(Ordering::Acquire) {
            return Err(EngineError::WorkerDied);
        }
        if self.in_flight() >= 2 {
            return Err(EngineError::GangQueue {
                in_flight: self.in_flight(),
            });
        }
        let seq = self
            .submitted
            .checked_add(1)
            .ok_or(EngineError::SequenceExhausted)?;
        let i = slot_index(seq);
        let slot = &shared.slots[i];
        // SAFETY: the caller keeps `job` valid until its wait (contract).
        self.polled[i] = unsafe { &*job }.waker.is_none();
        slot.panicked.store(false, Ordering::SeqCst);
        slot.pending.store(self.count, Ordering::SeqCst);
        slot.job.store(job.cast_mut(), Ordering::SeqCst);
        // Publication: `published` before the word (module docs, step 1).
        shared.published.store(seq, Ordering::SeqCst);
        shared.word.fetch_add(1, Ordering::SeqCst);
        // An RMW, not a load: it orders against a parking worker's RMW on
        // the same counter, so one of the two sees the other (no store-load
        // reordering, which SeqCst loads alone would rely on).
        if shared.parked.fetch_add(0, Ordering::SeqCst) > 0 {
            shared.wake(&shared.word, self.workers.iter().map(|(t, _)| t));
        }
        self.submitted = seq;
        Ok(())
    }

    /// Waits for the oldest job in flight and frees its slot. Returns
    /// whether any call of it panicked.
    pub(crate) fn wait(&mut self) -> Result<bool, EngineError> {
        let (seq, i) = self.oldest()?;
        let shared = &*self.shared;
        let slot = &shared.slots[i];
        shared.wait_until(
            &slot.pending,
            &slot.waiters,
            || slot.pending.load(Ordering::SeqCst) == 0,
            self.polled[i],
        );
        Ok(self.complete(seq, i))
    }

    /// Like `wait`, but returns `None` at once if the oldest job is still
    /// running.
    #[cfg_attr(
        any(not(test), loom),
        expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
    )]
    pub(crate) fn try_wait(&mut self) -> Result<Option<bool>, EngineError> {
        let (seq, i) = self.oldest()?;
        if self.shared.slots[i].pending.load(Ordering::SeqCst) != 0 {
            return Ok(None);
        }
        Ok(Some(self.complete(seq, i)))
    }

    fn oldest(&self) -> Result<(u64, usize), EngineError> {
        if self.in_flight() == 0 {
            return Err(EngineError::GangQueue { in_flight: 0 });
        }
        let seq = self
            .completed
            .checked_add(1)
            .ok_or(EngineError::SequenceExhausted)?;
        Ok((seq, slot_index(seq)))
    }

    fn complete(&mut self, seq: u64, i: usize) -> bool {
        let shared = &*self.shared;
        let slot = &shared.slots[i];
        slot.job.store(core::ptr::null_mut(), Ordering::SeqCst);
        self.completed = seq;
        let panicked = slot.panicked.load(Ordering::SeqCst);
        if panicked {
            shared.poisoned.store(true, Ordering::SeqCst);
        }
        panicked
    }

    /// Whether a job panicked since the last `clear_poison`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
    )]
    pub(crate) fn poisoned(&self) -> bool {
        self.shared.poisoned.load(Ordering::SeqCst)
    }

    /// Clears the sticky poison flag after a recovery; needs no job in
    /// flight.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
    )]
    pub(crate) fn clear_poison(&mut self) -> Result<(), EngineError> {
        if self.in_flight() != 0 {
            return Err(EngineError::GangQueue {
                in_flight: self.in_flight(),
            });
        }
        self.shared.poisoned.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// FP environment resets made by workers at batch starts (ADR-0037).
    pub(crate) fn fp_resets(&self) -> u64 {
        self.shared.fp_resets.load(Ordering::Relaxed)
    }
}

impl Drop for Gang {
    fn drop(&mut self) {
        // Drain: every job in flight completes, even after a death, because
        // a dead worker keeps decrementing.
        while self.in_flight() != 0 {
            if self.wait().is_err() {
                break;
            }
        }
        let shared = &*self.shared;
        shared.shutdown.store(true, Ordering::SeqCst);
        shared.word.fetch_add(1, Ordering::SeqCst);
        shared.wake(&shared.word, self.workers.iter().map(|(t, _)| t));
        for (_, join) in &mut self.workers {
            if let Some(join) = join.take()
                && join.join().is_err()
            {
                tracing::error!("an engine worker thread ended with a panic");
            }
        }
    }
}

/// Test-only injected deaths outside the job's catch (all false otherwise).
#[derive(Debug, Clone, Copy, Default)]
struct Death {
    after_waker: bool,
    before_job: bool,
}

/// Startup, then the job loop. A panic outside a job marks the gang dead and
/// turns the worker into a drainer.
fn worker_main(shared: &Shared, worker: usize, core: Option<usize>, fail: bool, die: Death) {
    let started = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert!(!fail, "injected worker startup failure");
        #[cfg(not(loom))]
        IN_WORKER.with(|flag| flag.set(true));
        if let Some(core) = core {
            pin_current(core);
        }
        #[cfg(not(loom))]
        tessera_rl_num::fp::establish_canonical();
    }));
    if started.is_err() {
        shared.failed.store(worker, Ordering::Relaxed);
    }
    // Release: publishes `failed`; the creator waits for every worker.
    shared.ready.fetch_add(1, Ordering::Release);
    shared.wake(&shared.ready, core::iter::once(&shared.creator));
    if started.is_err() {
        return;
    }
    let mut next: u64 = 1;
    let mut draining = false;
    loop {
        // Run every published job, in order, before sleeping (step 2).
        while shared.published.load(Ordering::SeqCst) >= next {
            let slot = &shared.slots[slot_index(next)];
            if draining {
                drain_job(shared, slot);
            } else {
                // Everything that can panic runs inside `run_job`'s catch,
                // before its decrement; this outer catch is a backstop.
                let mut seen = Seen::default();
                if !catch(|| run_job(shared, slot, worker, &mut seen, die)) {
                    shared.dead.store(true, Ordering::Release);
                    if !seen.decremented {
                        // The job may not have run for this worker's
                        // chunks: it reports a panic, so the pool poisons.
                        slot.panicked.store(true, Ordering::SeqCst);
                        let waker = seen.waker.take().or_else(|| job_waker(slot));
                        decrement(shared, slot, waker.as_ref());
                    }
                    draining = true;
                }
            }
            next = match next.checked_add(1) {
                Some(n) => n,
                // `submit` refuses the sequence that would need this.
                None => return,
            };
        }
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        shared.wait_until(
            &shared.word,
            &shared.parked,
            || {
                shared.published.load(Ordering::SeqCst) >= next
                    || shared.shutdown.load(Ordering::SeqCst)
            },
            false,
        );
    }
}

/// What a worker has done in the current job, for the backstop.
#[derive(Default)]
struct Seen {
    waker: Option<thread::Thread>,
    decremented: bool,
}

/// This worker's decrement of the slot's `pending`, its last access to the
/// job; the last one wakes the waiter if it is parked.
fn decrement(shared: &Shared, slot: &JobSlot, waker: Option<&thread::Thread>) {
    // SeqCst: releases this worker's writes (and `panicked`) to the waiter,
    // and orders the decrement before the `waiters` check (module docs).
    if slot.pending.fetch_sub(1, Ordering::SeqCst) == 1
        && slot.waiters.fetch_add(0, Ordering::SeqCst) > 0
    {
        // Park unparks the job's waker; a job without one has a polling
        // waiter. The other backends wake by address.
        shared.wake(&slot.pending, waker);
    }
}

/// A drainer's share of a job: no call, a reported panic, the decrement.
fn drain_job(shared: &Shared, slot: &JobSlot) {
    slot.panicked.store(true, Ordering::SeqCst);
    let waker = job_waker(slot);
    decrement(shared, slot, waker.as_ref());
}

/// The waker of the slot's job, read before this worker's decrement.
#[expect(unsafe_code, reason = "audited module: job pointer")]
fn job_waker(slot: &JobSlot) -> Option<thread::Thread> {
    let job = slot.job.load(Ordering::SeqCst);
    if job.is_null() {
        return None;
    }
    // SAFETY: this worker observed `published` reach the job's sequence and
    // has not decremented its `pending` yet, so the submitter keeps the
    // `Job` alive (module invariants).
    unsafe { &*job }.waker.clone()
}

#[expect(unsafe_code, reason = "audited module: job pointer and call")]
fn run_job(shared: &Shared, slot: &JobSlot, worker: usize, seen: &mut Seen, die: Death) {
    assert!(
        !die.before_job,
        "injected worker death before the job is read"
    );
    let job = slot.job.load(Ordering::SeqCst);
    // SAFETY: the submitter stored a pointer to a live `Job` before the
    // `published` store this worker observed, and keeps it alive until the
    // slot's `pending` reaches zero, which needs this worker's decrement
    // below; it is never null while the job is in flight.
    let job = unsafe { &*job };
    seen.waker.clone_from(&job.waker);
    let (call, data) = (job.call, job.data);
    assert!(
        !die.after_waker,
        "injected worker death outside the job's catch"
    );
    let completed = catch(|| {
        #[cfg(not(loom))]
        if tessera_rl_num::fp::ensure_canonical() {
            shared.fp_resets.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `call` is `trampoline::<F>` and `data` the matching `&F`,
        // valid for this job (module invariants).
        unsafe { call(data, worker) };
    });
    if !completed {
        slot.panicked.store(true, Ordering::SeqCst);
    }
    // The payload is already disposed of; nothing that can panic follows.
    seen.decremented = true;
    decrement(shared, slot, seen.waker.as_ref());
}

#[cfg(test)]
mod tests;

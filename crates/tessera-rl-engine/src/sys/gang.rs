// SPDX-License-Identifier: MIT OR Apache-2.0

//! The worker gang and its epoch protocol (ADR-0005).
//!
//! # Startup
//!
//! Workers are spawned with fallible `Builder::spawn`; if a spawn fails,
//! the workers already started are shut down and joined. Every worker then
//! reports `ready` (after pinning and entering the canonical FP
//! environment, both caught), and `Gang::new` waits for all of them: a
//! failed worker shuts the gang down and returns a typed error, so a run
//! can never wait for a worker that does not exist.
//!
//! # One epoch per `run`
//!
//! 1. the caller stores `pending = workers` and the job pointer (Relaxed),
//!    then increments `epoch` with Release, which publishes both, and
//!    unparks every worker;
//! 2. each worker waits until it sees a new epoch with Acquire, runs its
//!    share of the job, and decrements `pending` with `AcqRel`; the worker
//!    that brings it to zero unparks the caller;
//! 3. the caller waits until it reads `pending == 0` with Acquire, which
//!    makes every worker's writes visible, and only then returns.
//!
//! Waits spin for a bounded budget, then park; parking and waking happen
//! only at these batch boundaries. `unpark` before `park` is remembered by
//! the thread token, so a wake-up can never be lost, and spurious wake-ups
//! just re-check the counters. `run` takes `&mut self`, so only one epoch
//! is ever in flight.
//!
//! A worker always decrements `pending`, even after a panic in the job. A
//! panic anywhere else in its loop marks the gang dead (after
//! decrementing), and later runs return an error instead of waiting.
//!
//! Shutdown (on drop) sets `shutdown` and increments `epoch` with Release,
//! unparks and joins every worker.
//!
//! # Unsafe invariants
//!
//! - The job pointer is valid from the epoch increment until `pending`
//!   reaches zero: it points to a `Job` on the caller's stack, and `run`
//!   does not return before that. Workers read it only after seeing the new
//!   epoch, and never after their decrement.
//! - The job's data pointer is a `&F` for the `F` its `call` was
//!   instantiated with (`trampoline::<F>`), valid for the same window.
//! - Workers run concurrently, so `F: Sync`; the job itself is responsible
//!   for giving each worker disjoint data (see `engine::buffer`).

use core::num::NonZeroUsize;

use crate::error::EngineError;

use super::affinity::pin_current;
use super::sync::{
    Arc, AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering, spin_loop, thread,
};

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
}

struct Job {
    call: unsafe fn(*const (), usize),
    data: *const (),
    waker: thread::Thread,
}

struct Shared {
    epoch: AtomicU64,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    panicked: AtomicBool,
    dead: AtomicBool,
    ready: AtomicUsize,
    failed: AtomicUsize,
    fp_resets: AtomicU64,
    job: AtomicPtr<Job>,
    creator: thread::Thread,
    spin: u32,
}

/// Persistent worker threads running one job per epoch.
pub(crate) struct Gang {
    shared: Arc<Shared>,
    workers: Vec<(thread::Thread, Option<thread::JoinHandle<()>>)>,
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
    // SAFETY: `run::<F>` stores `data = &f as *const F` and keeps `f` alive
    // until every worker has decremented `pending`, after this call.
    let f = unsafe { &*data.cast::<F>() };
    f(worker);
}

/// Waits until `done()` holds: a bounded spin, then parking.
fn wait(spin: u32, mut done: impl FnMut() -> bool) {
    let mut spins = 0;
    while !done() {
        if spins < spin {
            spins += 1;
            spin_loop();
        } else {
            thread::park();
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
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            dead: AtomicBool::new(false),
            ready: AtomicUsize::new(0),
            failed: AtomicUsize::new(usize::MAX),
            fp_resets: AtomicU64::new(0),
            job: AtomicPtr::new(core::ptr::null_mut()),
            creator: thread::current(),
            spin: options.spin,
        });
        // Dropping `gang` on any early return shuts down and joins the
        // workers spawned so far (the construction guard).
        let mut gang = Self {
            shared,
            workers: Vec::with_capacity(options.workers.get()),
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
                    let fail = faults.startup == Some(w);
                    #[cfg(not(test))]
                    let fail = false;
                    worker_main(&shared, w, core, fail);
                });
            match spawned {
                Ok(handle) => gang.workers.push((handle.thread().clone(), Some(handle))),
                Err(_) => return Err(EngineError::WorkerStartup { worker: w }),
            }
        }
        let shared = Arc::clone(&gang.shared);
        let n = gang.workers.len();
        // Acquire: pairs with each worker's Release on `ready`.
        wait(shared.spin, || shared.ready.load(Ordering::Acquire) == n);
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

    /// Runs `f(worker)` on every worker and waits for all of them. Returns
    /// whether any call panicked, or an error if a worker died earlier.
    pub(crate) fn run<F: Fn(usize) + Sync>(&mut self, f: &F) -> Result<bool, EngineError> {
        let shared = &*self.shared;
        if shared.dead.load(Ordering::Acquire) {
            return Err(EngineError::WorkerDied);
        }
        let job = Job {
            call: trampoline::<F>,
            data: core::ptr::from_ref(f).cast::<()>(),
            waker: thread::current(),
        };
        shared.panicked.store(false, Ordering::Relaxed);
        shared.pending.store(self.workers.len(), Ordering::Relaxed);
        shared
            .job
            .store(core::ptr::from_ref(&job).cast_mut(), Ordering::Relaxed);
        // Release: publishes `pending`, `panicked` and the job (step 1).
        shared.epoch.fetch_add(1, Ordering::Release);
        for (handle, _) in &self.workers {
            handle.unpark();
        }
        // Acquire: sees every worker's writes once all have decremented.
        wait(shared.spin, || shared.pending.load(Ordering::Acquire) == 0);
        Ok(shared.panicked.load(Ordering::Relaxed))
    }

    /// FP environment resets made by workers at batch starts (ADR-0037).
    pub(crate) fn fp_resets(&self) -> u64 {
        self.shared.fp_resets.load(Ordering::Relaxed)
    }
}

impl Drop for Gang {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        // Release: publishes `shutdown` with the new epoch.
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for (handle, join) in &mut self.workers {
            handle.unpark();
            if let Some(join) = join.take()
                && join.join().is_err()
            {
                tracing::error!("an engine worker thread ended with a panic");
            }
        }
    }
}

/// Startup, then the epoch loop. A panic outside a job marks the gang dead.
fn worker_main(shared: &Shared, worker: usize, core: Option<usize>, fail: bool) {
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
    shared.creator.unpark();
    if started.is_err() {
        return;
    }
    let mut seen = 0;
    loop {
        let mut epoch = seen;
        wait(shared.spin, || {
            // Acquire: pairs with the caller's Release increment.
            epoch = shared.epoch.load(Ordering::Acquire);
            epoch != seen
        });
        seen = epoch;
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        // Everything that can panic runs inside `run_job`'s catch, before
        // its decrement; this outer catch is a backstop that marks the gang
        // dead so later runs fail instead of waiting for this worker.
        let alive =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_job(shared, worker)));
        if alive.is_err() {
            shared.dead.store(true, Ordering::Release);
            return;
        }
    }
}

#[expect(unsafe_code, reason = "audited module: job pointer and call")]
fn run_job(shared: &Shared, worker: usize) {
    let job = shared.job.load(Ordering::Relaxed);
    // SAFETY: the caller stored a pointer to a `Job` on its stack before the
    // Release epoch increment we observed with Acquire, and keeps it alive
    // until `pending` reaches zero, which needs this worker's decrement
    // below; it is never null during an epoch.
    let job = unsafe { &*job };
    let waker = job.waker.clone();
    let (call, data) = (job.call, job.data);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        #[cfg(not(loom))]
        if tessera_rl_num::fp::ensure_canonical() {
            shared.fp_resets.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `call` is `trampoline::<F>` and `data` the matching `&F`,
        // valid for this epoch (module invariants).
        unsafe { call(data, worker) };
    }));
    if result.is_err() {
        shared.panicked.store(true, Ordering::Relaxed);
    }
    // AcqRel: releases this worker's writes (and `panicked`) to the caller.
    if shared.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
        waker.unpark();
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroUsize;

    use super::{Faults, Gang, GangOptions};
    use crate::error::EngineError;
    use crate::sys::sync::{Arc, AtomicUsize, Ordering};

    fn options(workers: usize) -> GangOptions {
        GangOptions {
            workers: NonZeroUsize::new(workers).unwrap(),
            spin: if cfg!(loom) { 0 } else { 64 },
            pin: None,
            faults: Faults::default(),
        }
    }

    fn model(f: impl Fn() + Sync + Send + 'static) {
        #[cfg(loom)]
        loom::model(f);
        #[cfg(not(loom))]
        f();
    }

    /// Runs `epochs` jobs on `workers` workers and checks every worker ran
    /// each one and saw the writes of the previous epoch.
    fn epochs(workers: usize, epochs: usize) {
        let mut gang = Gang::new(&options(workers)).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        for epoch in 1..=epochs {
            let c = Arc::clone(&count);
            let panicked = gang
                .run(&move |_w: usize| {
                    c.fetch_add(1, Ordering::Relaxed);
                })
                .unwrap();
            assert!(!panicked);
            assert_eq!(count.load(Ordering::Relaxed), workers * epoch);
        }
    }

    // loom explores every interleaving, so its models stay small.
    #[test]
    fn test_gang_one_worker_two_epochs() {
        model(|| epochs(1, 2));
    }

    #[test]
    fn test_gang_two_workers_one_epoch() {
        model(|| epochs(2, 1));
    }

    #[test]
    fn test_gang_two_workers_two_epochs() {
        model(|| epochs(2, 2));
    }

    #[test]
    fn test_gang_drop_right_after_run() {
        model(|| {
            let mut gang = Gang::new(&options(2)).unwrap();
            assert!(!gang.run(&|_w: usize| {}).unwrap());
            drop(gang);
        });
    }

    #[cfg(not(loom))]
    #[test]
    fn test_gang_many_workers_many_epochs() {
        epochs(7, 50);
    }

    #[test]
    fn test_gang_shutdown_joins_idle_workers() {
        model(|| {
            let gang = Gang::new(&options(2)).unwrap();
            drop(gang);
        });
    }

    #[test]
    fn test_gang_worker_panic_is_reported_and_does_not_hang() {
        model(|| {
            let mut gang = Gang::new(&options(1)).unwrap();
            assert!(
                gang.run(&|_w: usize| panic!("injected worker panic"))
                    .unwrap()
            );
            assert!(!gang.run(&|_w: usize| {}).unwrap());
        });
    }

    /// Runs `f` on a helper thread and fails if it does not finish in time,
    /// so a hang shows up as a failure instead of a stuck test run.
    #[cfg(not(loom))]
    fn within_timeout(f: impl FnOnce() + Send + 'static) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            f();
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(20)).is_ok(),
            "timed out"
        );
    }

    #[cfg(not(loom))]
    #[test]
    fn test_gang_startup_failure_is_an_error_not_a_hang() {
        within_timeout(|| {
            let mut opts = options(3);
            opts.faults.startup = Some(1);
            assert!(matches!(
                Gang::new(&opts),
                Err(EngineError::WorkerStartup { .. })
            ));
        });
    }

    #[cfg(not(loom))]
    #[test]
    fn test_gang_failed_spawn_cleans_up_started_workers() {
        within_timeout(|| {
            let mut opts = options(3);
            opts.faults.spawn = Some(1);
            assert_eq!(
                Gang::new(&opts).map(|_| ()),
                Err(EngineError::WorkerStartup { worker: 1 })
            );
        });
    }

    #[cfg(not(loom))]
    #[test]
    fn test_gang_rejects_nested_engines() {
        let mut gang = Gang::new(&options(1)).unwrap();
        let nested = std::sync::Mutex::new(None);
        gang.run(&|_w: usize| {
            let result = Gang::new(&options(1)).map(|_| ());
            if let Ok(mut slot) = nested.lock() {
                *slot = Some(result);
            }
        })
        .unwrap();
        let result = nested.lock().ok().and_then(|mut s| s.take());
        assert_eq!(result, Some(Err(EngineError::NestedEngine)));
    }
}

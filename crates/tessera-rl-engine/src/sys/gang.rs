// SPDX-License-Identifier: MIT OR Apache-2.0

//! The worker gang and its epoch protocol (ADR-0005).
//!
//! One `run` is one epoch:
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
//! just re-check the counters. A worker always decrements `pending`, even
//! after a panic in the job, so the caller can never hang.
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

use crate::error::EngineError;

use super::affinity::pin_current;
use super::sync::{
    Arc, AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering, spin_loop, thread,
};

/// Options of a gang.
#[derive(Debug, Clone)]
pub(crate) struct GangOptions {
    /// Number of workers.
    pub(crate) workers: usize,
    /// Spins before parking (about 20 microseconds by default).
    pub(crate) spin: u32,
    /// Logical cores to pin workers to, in order; `None` leaves them unpinned.
    pub(crate) pin: Option<Vec<usize>>,
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
    fp_resets: AtomicU64,
    job: AtomicPtr<Job>,
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

impl Gang {
    /// Spawns the workers. Engines may not be nested inside a worker.
    pub(crate) fn new(options: &GangOptions) -> Result<Self, EngineError> {
        if in_worker() {
            return Err(EngineError::NestedEngine);
        }
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            fp_resets: AtomicU64::new(0),
            job: AtomicPtr::new(core::ptr::null_mut()),
            spin: options.spin,
        });
        let workers = (0..options.workers)
            .map(|w| {
                let shared = Arc::clone(&shared);
                let core = options.pin.as_ref().and_then(|cores| cores.get(w).copied());
                let handle = thread::spawn(move || worker_loop(&shared, w, core));
                (handle.thread().clone(), Some(handle))
            })
            .collect();
        Ok(Self { shared, workers })
    }

    /// Runs `f(worker)` on every worker and waits for all of them. Returns
    /// `true` if any call panicked.
    pub(crate) fn run<F: Fn(usize) + Sync>(&self, f: &F) -> bool {
        let job = Job {
            call: trampoline::<F>,
            data: core::ptr::from_ref(f).cast::<()>(),
            waker: thread::current(),
        };
        let shared = &*self.shared;
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
        let mut spins = 0;
        // Acquire: sees every worker's writes once all have decremented.
        while shared.pending.load(Ordering::Acquire) != 0 {
            if spins < shared.spin {
                spins += 1;
                spin_loop();
            } else {
                thread::park();
            }
        }
        shared.panicked.load(Ordering::Relaxed)
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
                tracing::error!("an engine worker thread panicked outside a job");
            }
        }
    }
}

fn worker_loop(shared: &Shared, worker: usize, core: Option<usize>) {
    #[cfg(not(loom))]
    IN_WORKER.with(|flag| flag.set(true));
    if let Some(core) = core {
        pin_current(core);
    }
    #[cfg(not(loom))]
    tessera_rl_num::fp::establish_canonical();
    let mut seen = 0;
    loop {
        let mut spins = 0;
        let epoch = loop {
            // Acquire: pairs with the caller's Release increment.
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen {
                break epoch;
            }
            if spins < shared.spin {
                spins += 1;
                spin_loop();
            } else {
                thread::park();
            }
        };
        seen = epoch;
        if shared.shutdown.load(Ordering::Relaxed) {
            return;
        }
        #[cfg(not(loom))]
        if tessera_rl_num::fp::ensure_canonical() {
            shared.fp_resets.fetch_add(1, Ordering::Relaxed);
        }
        run_job(shared, worker);
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
    use super::{Gang, GangOptions};
    use crate::sys::sync::{Arc, AtomicUsize, Ordering};

    fn options(workers: usize) -> GangOptions {
        GangOptions {
            workers,
            spin: if cfg!(loom) { 0 } else { 64 },
            pin: None,
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
        let gang = Gang::new(&options(workers)).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        for epoch in 1..=epochs {
            let c = Arc::clone(&count);
            let panicked = gang.run(&move |_w: usize| {
                c.fetch_add(1, Ordering::Relaxed);
            });
            assert!(!panicked);
            assert_eq!(count.load(Ordering::Relaxed), workers * epoch);
        }
    }

    // loom explores every interleaving, so its models stay small: one
    // worker over two epochs (epoch reuse, lost wake-ups of both sides) and
    // two workers over one epoch (the pending count).
    #[test]
    fn test_gang_one_worker_two_epochs() {
        model(|| epochs(1, 2));
    }

    #[test]
    fn test_gang_two_workers_one_epoch() {
        model(|| epochs(2, 1));
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
            let gang = Gang::new(&options(1)).unwrap();
            assert!(gang.run(&|_w: usize| panic!("injected worker panic")));
            assert!(!gang.run(&|_w: usize| {}));
        });
    }

    #[cfg(not(loom))]
    #[test]
    fn test_gang_rejects_nested_engines() {
        let gang = Gang::new(&options(1)).unwrap();
        let nested = std::sync::Mutex::new(None);
        gang.run(&|_w: usize| {
            let result = Gang::new(&options(1)).map(|_| ());
            if let Ok(mut slot) = nested.lock() {
                *slot = Some(result);
            }
        });
        let result = nested.lock().ok().and_then(|mut s| s.take());
        assert_eq!(result, Some(Err(crate::error::EngineError::NestedEngine)));
    }
}

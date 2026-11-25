// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the worker gang. Under `cfg(loom)` every `model` explores the
//! interleavings, so the models stay small.

use core::num::NonZeroUsize;

use super::{Faults, Gang, GangOptions, Job};
use crate::buffer::claim::Cursors;
use crate::error::EngineError;
use crate::sys::sync::{Arc, AtomicBool, AtomicUsize, Ordering, thread};

fn options(workers: usize) -> GangOptions {
    GangOptions {
        workers: NonZeroUsize::new(workers).unwrap(),
        spin: if cfg!(loom) { 0 } else { 64 },
        adaptive_cap_ns: None,
        pin: None,
        #[cfg(not(loom))]
        native: None,
        #[cfg(loom)]
        mock_futex: false,
        faults: Faults::default(),
    }
}

/// The same options blocking with the address wait: native outside loom,
/// the futex mock under loom.
fn address_options(workers: usize) -> GangOptions {
    // Miri cannot run the native address-wait syscalls: it uses park, so
    // only the wait primitive differs, never the job protocol.
    #[cfg(all(not(loom), not(miri)))]
    {
        let mut opts = options(workers);
        opts.native = crate::sys::Native::detect();
        opts
    }
    #[cfg(all(not(loom), miri))]
    {
        options(workers)
    }
    #[cfg(loom)]
    {
        let mut opts = options(workers);
        opts.mock_futex = true;
        opts
    }
}

/// Submits `job`. Every test keeps its jobs and closures alive until the
/// gang has waited for them: they are declared before the gang (dropped
/// after its draining drop), or outlive a joined waiter thread.
fn submit(gang: &mut Gang, job: &Job) -> Result<(), EngineError> {
    // SAFETY: see above; the job outlives the wait that returns its result.
    #[expect(unsafe_code, reason = "test of the audited submit")]
    unsafe {
        gang.submit(job)
    }
}

fn model(f: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    loom::model(f);
    #[cfg(not(loom))]
    f();
}

/// Like `model`, with a preemption bound of 2 for the models whose state
/// space is too large at the CI bound of 3 (two workers on the futex mock,
/// whose mutex and condvar add many interleavings).
fn small_model(f: impl Fn() + Sync + Send + 'static) {
    #[cfg(loom)]
    {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(2);
        builder.check(f);
    }
    #[cfg(not(loom))]
    f();
}

/// Runs `epochs` jobs on `workers` workers and checks every worker ran
/// each one and saw the writes of the previous one.
fn epochs(opts: &GangOptions, workers: usize, epochs: usize) {
    let mut gang = Gang::new(opts).unwrap();
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

#[test]
fn test_gang_one_worker_two_epochs() {
    model(|| epochs(&options(1), 1, 2));
}

#[test]
fn test_gang_two_workers_one_epoch() {
    model(|| epochs(&options(2), 2, 1));
}

#[test]
fn test_gang_two_workers_two_epochs() {
    model(|| epochs(&options(2), 2, 2));
}

#[test]
fn test_gang_address_wait_two_workers_one_epoch() {
    small_model(|| epochs(&address_options(2), 2, 1));
}

#[test]
fn test_gang_address_wait_one_worker_two_epochs() {
    model(|| epochs(&address_options(1), 1, 2));
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
    epochs(&options(7), 7, if cfg!(miri) { 3 } else { 50 });
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

/// A closure counting its calls, and a job for it, kept alive across the
/// gang's lifetime by the caller.
struct Counted {
    count: Arc<AtomicUsize>,
}

impl Counted {
    fn new() -> Self {
        Self {
            count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

/// Per-worker markers: `a[w]` is set when worker `w` has run job A, and
/// `violations` counts B calls on a worker that had not run A yet.
struct Order {
    a: Vec<AtomicUsize>,
    a_calls: AtomicUsize,
    b_calls: AtomicUsize,
    violations: AtomicUsize,
}

impl Order {
    fn new(workers: usize) -> Arc<Self> {
        Arc::new(Self {
            a: (0..workers).map(|_| AtomicUsize::new(0)).collect(),
            a_calls: AtomicUsize::new(0),
            b_calls: AtomicUsize::new(0),
            violations: AtomicUsize::new(0),
        })
    }
}

/// Two jobs published before any worker can observe either: both run,
/// each worker runs A before B (its own A marker is set when it starts B;
/// other workers may still be in A), and both waits return.
fn two_published(opts: &GangOptions) {
    let workers = opts.workers.get();
    let order = Order::new(workers);
    let (oa, ob) = (Arc::clone(&order), Arc::clone(&order));
    let fa = move |w: usize| {
        oa.a[w].store(1, Ordering::SeqCst);
        oa.a_calls.fetch_add(1, Ordering::SeqCst);
    };
    let fb = move |w: usize| {
        if ob.a[w].load(Ordering::SeqCst) != 1 {
            ob.violations.fetch_add(1, Ordering::SeqCst);
        }
        ob.b_calls.fetch_add(1, Ordering::SeqCst);
    };
    let ja = Job::new(&fa, None);
    let jb = Job::new(&fb, None);
    let mut gang = Gang::new(opts).unwrap();
    submit(&mut gang, &ja).unwrap();
    submit(&mut gang, &jb).unwrap();
    assert!(!gang.wait().unwrap());
    assert_eq!(order.a_calls.load(Ordering::SeqCst), workers);
    assert!(!gang.wait().unwrap());
    assert_eq!(order.b_calls.load(Ordering::SeqCst), workers);
    assert_eq!(
        order.violations.load(Ordering::SeqCst),
        0,
        "a worker ran B before its own A"
    );
    drop(gang);
}

/// Slot reuse with two jobs outstanding: A and B in flight, A waited, then
/// C reuses A's slot while B is still in flight. Each job runs exactly once
/// per worker with its own closure, so C cannot overwrite B's live slot.
fn slot_reuse(opts: &GangOptions) {
    let workers = opts.workers.get();
    let counts: Arc<Vec<AtomicUsize>> = Arc::new((0..3).map(|_| AtomicUsize::new(0)).collect());
    let (ca, cb, cc) = (
        Arc::clone(&counts),
        Arc::clone(&counts),
        Arc::clone(&counts),
    );
    let fa = move |_w: usize| {
        ca[0].fetch_add(1, Ordering::SeqCst);
    };
    let fb = move |_w: usize| {
        cb[1].fetch_add(1, Ordering::SeqCst);
    };
    let fc = move |_w: usize| {
        cc[2].fetch_add(1, Ordering::SeqCst);
    };
    let (ja, jb, jc) = (
        Job::new(&fa, None),
        Job::new(&fb, None),
        Job::new(&fc, None),
    );
    let mut gang = Gang::new(opts).unwrap();
    submit(&mut gang, &ja).unwrap();
    submit(&mut gang, &jb).unwrap();
    assert_eq!(
        submit(&mut gang, &jc),
        Err(EngineError::GangQueue { in_flight: 2 }),
        "a third job may not take a live slot"
    );
    assert!(!gang.wait().unwrap());
    assert_eq!(counts[0].load(Ordering::SeqCst), workers);
    submit(&mut gang, &jc).unwrap();
    assert!(!gang.wait().unwrap());
    assert_eq!(counts[1].load(Ordering::SeqCst), workers);
    assert!(!gang.wait().unwrap());
    assert_eq!(counts[2].load(Ordering::SeqCst), workers);
    assert_eq!(counts[0].load(Ordering::SeqCst), workers);
    drop(gang);
}

#[test]
fn test_gang_slot_reuse_with_two_jobs_outstanding_keeps_each_job() {
    model(|| slot_reuse(&options(1)));
}

#[cfg(not(loom))]
#[test]
fn test_gang_slot_reuse_on_many_workers_keeps_each_job() {
    for _ in 0..if cfg!(miri) { 2 } else { 200 } {
        slot_reuse(&options(3));
        slot_reuse(&address_options(3));
    }
}

#[test]
fn test_gang_two_jobs_published_before_any_worker_runs_both() {
    model(|| two_published(&options(1)));
}

#[test]
fn test_gang_two_jobs_published_on_two_workers_with_park_run_both() {
    small_model(|| two_published(&options(2)));
}

#[test]
fn test_gang_two_jobs_published_on_two_workers_with_address_wait_run_both() {
    small_model(|| two_published(&address_options(2)));
}

#[test]
fn test_gang_two_jobs_published_with_address_wait_run_both() {
    model(|| two_published(&address_options(1)));
}

#[cfg(not(loom))]
#[test]
fn test_gang_two_jobs_published_on_many_workers_run_both() {
    for _ in 0..if cfg!(miri) { 2 } else { 200 } {
        two_published(&options(4));
        two_published(&address_options(4));
    }
}

/// Dropping the gang with one or two jobs in flight drains them first.
fn drop_in_flight(jobs: usize) {
    let counted = Counted::new();
    let c = Arc::clone(&counted.count);
    let f = move |_w: usize| {
        c.fetch_add(1, Ordering::SeqCst);
    };
    let job = Job::new(&f, None);
    let mut gang = Gang::new(&address_options(1)).unwrap();
    for _ in 0..jobs {
        {
            submit(&mut gang, &job).unwrap();
        }
    }
    drop(gang);
    assert_eq!(counted.get(), jobs);
}

#[test]
fn test_gang_drop_with_one_job_in_flight_drains_it() {
    model(|| drop_in_flight(1));
}

#[test]
fn test_gang_drop_with_two_jobs_in_flight_drains_both() {
    model(|| drop_in_flight(2));
}

/// A panic in the first job with the second queued: the second still runs,
/// each wait reports its own job, and the poison stays until cleared.
#[test]
fn test_gang_panic_in_a_with_b_queued_runs_b_and_poison_sticks() {
    model(|| {
        let counted = Counted::new();
        let c = Arc::clone(&counted.count);
        let fa = |_w: usize| panic!("injected worker panic");
        let fb = move |_w: usize| {
            c.fetch_add(1, Ordering::SeqCst);
        };
        let (ja, jb) = (Job::new(&fa, None), Job::new(&fb, None));
        let mut gang = Gang::new(&options(1)).unwrap();
        {
            submit(&mut gang, &ja).unwrap();
            submit(&mut gang, &jb).unwrap();
        }
        assert!(gang.wait().unwrap());
        assert!(!gang.wait().unwrap());
        assert_eq!(counted.get(), 1);
        assert!(gang.poisoned());
        assert!(!gang.run(&|_w: usize| {}).unwrap());
        assert!(gang.poisoned(), "a clean job does not clear the poison");
        gang.clear_poison().unwrap();
        assert!(!gang.poisoned());
    });
}

/// A worker that dies outside the job's catch with a second job queued,
/// or published while it dies, keeps draining: both waits return (the
/// drained job reports a panic), and later submits report the death.
fn death_with_queued(publish_second_after_first_wait: bool) {
    // One worker that dies: the smallest gang with a drainer.
    let mut opts = options(1);
    opts.faults.die_in_epoch = Some(0);
    let f = |_w: usize| {};
    // B counts its calls: the dead worker must run no job code for it.
    let b_calls = Counted::new();
    let bc = Arc::clone(&b_calls.count);
    let fb = move |_w: usize| {
        bc.fetch_add(1, Ordering::SeqCst);
    };
    let (ja, jb) = (Job::new(&f, None), Job::new(&fb, None));
    let mut gang = Gang::new(&opts).unwrap();
    let queued = {
        submit(&mut gang, &ja).unwrap();
        // The worker may already be dead when the second job is published:
        // then the submit reports it; otherwise the job is drained.
        !publish_second_after_first_wait && submit(&mut gang, &jb).is_ok()
    };
    assert!(
        gang.wait().unwrap(),
        "the dying worker's job reports a panic"
    );
    if !publish_second_after_first_wait && !queued {
        return;
    }
    if publish_second_after_first_wait {
        let second = { submit(&mut gang, &jb) };
        assert_eq!(second, Err(EngineError::WorkerDied));
    } else {
        assert!(gang.wait().unwrap(), "the drained job reports a panic");
        assert_eq!(b_calls.get(), 0, "a dead worker ran a queued job's code");
        let third = { submit(&mut gang, &ja) };
        assert_eq!(third, Err(EngineError::WorkerDied));
    }
}

#[test]
fn test_gang_dead_worker_drains_a_queued_job() {
    model(|| death_with_queued(false));
}

#[test]
fn test_gang_publication_after_death_reports_worker_died() {
    model(|| death_with_queued(true));
}

/// A job submitted on one thread and waited for on another, with no waker
/// and no spinning: the waiter registers at wait time and rechecks, so the
/// completion is never missed (counted wakes on the address path).
fn cross_thread_wait(opts: &GangOptions) {
    let counted = Arc::new(Counted::new());
    let c = Arc::clone(&counted.count);
    let f = move |_w: usize| {
        c.fetch_add(1, Ordering::SeqCst);
    };
    let job = Job::new(&f, None);
    let mut gang = Gang::new(opts).unwrap();
    {
        submit(&mut gang, &job).unwrap();
    }
    let seen = Arc::clone(&counted);
    let waiter = crate::sys::sync::thread::spawn(move || {
        let mut gang = gang;
        let panicked = gang.wait();
        (panicked, seen.get())
    });
    let (panicked, count) = waiter.join().unwrap();
    assert_eq!(panicked, Ok(false));
    assert_eq!(count, 1);
    drop(job);
}

#[test]
fn test_gang_wait_on_another_thread_with_address_wait_completes() {
    model(|| {
        let mut opts = address_options(1);
        opts.spin = 0;
        cross_thread_wait(&opts);
    });
}

#[test]
fn test_gang_wait_on_another_thread_with_park_polling_completes() {
    model(|| {
        let mut opts = options(1);
        opts.spin = 0;
        cross_thread_wait(&opts);
    });
}

#[cfg(not(loom))]
#[test]
fn test_gang_try_wait_returns_none_then_the_result() {
    let gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let g = std::sync::Arc::clone(&gate);
    let f = move |_w: usize| {
        while !g.load(std::sync::atomic::Ordering::SeqCst) {
            std::hint::spin_loop();
        }
    };
    let job = Job::new(&f, None);
    let mut gang = Gang::new(&options(1)).unwrap();
    {
        submit(&mut gang, &job).unwrap();
    }
    assert_eq!(gang.try_wait(), Ok(None));
    gate.store(true, std::sync::atomic::Ordering::SeqCst);
    assert!(!gang.wait().unwrap());
    assert_eq!(gang.in_flight(), 0);
}

#[cfg(not(loom))]
#[test]
fn test_gang_queue_limits_are_typed_errors() {
    let f = |_w: usize| {};
    let job = Job::new(&f, None);
    let mut gang = Gang::new(&options(1)).unwrap();
    assert_eq!(gang.wait(), Err(EngineError::GangQueue { in_flight: 0 }));
    {
        submit(&mut gang, &job).unwrap();
        submit(&mut gang, &job).unwrap();
        assert_eq!(
            submit(&mut gang, &job),
            Err(EngineError::GangQueue { in_flight: 2 })
        );
    }
    assert_eq!(
        gang.run(&|_w: usize| {}),
        Err(EngineError::GangQueue { in_flight: 2 })
    );
    assert_eq!(
        gang.clear_poison(),
        Err(EngineError::GangQueue { in_flight: 2 })
    );
    gang.wait().unwrap();
    gang.wait().unwrap();
}

#[cfg(not(loom))]
#[test]
fn test_gang_sequence_overflow_is_sequence_exhausted() {
    let f = |_w: usize| {};
    let job = Job::new(&f, None);
    let mut gang = Gang::new(&options(1)).unwrap();
    // Skip ahead; no job is in flight, and the workers have run none.
    gang.submitted = u64::MAX;
    gang.completed = u64::MAX;
    let refused = { submit(&mut gang, &job) };
    assert_eq!(refused, Err(EngineError::SequenceExhausted));
    // Restore consistent counters so the drop's drain has nothing to do.
    gang.submitted = 0;
    gang.completed = 0;
}

/// How long a gang test may wait before it fails instead of hanging. It
/// catches hangs and does not bound speed: a starved process on a loaded
/// CI host can run these tests many times slower (as in #301).
#[cfg(not(loom))]
const HANG_GUARD: std::time::Duration = std::time::Duration::from_secs(300);

/// Runs `f` on a helper thread and fails if it does not finish within
/// [`HANG_GUARD`], so a hang shows up as a failure instead of a stuck test
/// run.
#[cfg(not(loom))]
fn within_timeout(f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    assert!(rx.recv_timeout(HANG_GUARD).is_ok(), "timed out");
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

struct Bomb;

impl Drop for Bomb {
    fn drop(&mut self) {
        std::panic::panic_any(());
    }
}

/// A job panic whose payload panics on drop is caught: the worker stays
/// alive and the next job completes.
#[test]
fn test_gang_survives_a_payload_whose_drop_panics() {
    model(|| {
        let mut gang = Gang::new(&options(1)).unwrap();
        assert!(gang.run(&|_w: usize| std::panic::panic_any(Bomb)).unwrap());
        assert!(!gang.run(&|_w: usize| {}).unwrap());
    });
}

/// A worker that dies before it reads the job, so before it clones
/// the job's waker, still wakes the run's caller, read from the job
/// itself: the gang is created on one thread and run on another, with
/// no spinning, so the caller really parks.
#[cfg(not(loom))]
#[test]
fn test_gang_death_before_the_job_wakes_a_caller_on_another_thread() {
    let mut opts = options(2);
    opts.spin = 0;
    opts.faults.die_before_job = Some(1);
    let gang = Gang::new(&opts).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let runner = std::thread::spawn(move || {
        let mut gang = gang;
        let first = gang.run(&|_w: usize| {});
        let second = gang.run(&|_w: usize| {}).map(|_| ());
        let _ = tx.send((first, second));
        gang
    });
    let outcome = rx.recv_timeout(HANG_GUARD);
    assert_eq!(outcome, Ok((Ok(true), Err(EngineError::WorkerDied))));
    drop(runner.join());
}

/// A worker that dies outside the job's catch still accounts for its
/// share: the current run completes and reports a panic (its chunks may
/// be untouched), and the next one reports the death instead of waiting.
#[test]
fn test_gang_death_mid_epoch_completes_the_run_then_reports() {
    model(|| {
        let mut opts = options(2);
        opts.faults.die_in_epoch = Some(1);
        let mut gang = Gang::new(&opts).unwrap();
        assert_eq!(gang.run(&|_w: usize| {}), Ok(true));
        assert_eq!(
            gang.run(&|_w: usize| {}).map(|_| ()),
            Err(EngineError::WorkerDied)
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

/// The adaptive policy (ADR-0074) with a small cap, so the mock clock of
/// loom reaches it: every wake-up is still seen, with workers that spin,
/// then block, across epochs.
fn adaptive_options(workers: usize, cap_ns: u64) -> GangOptions {
    let mut opts = options(workers);
    opts.adaptive_cap_ns = Some(cap_ns);
    opts
}

#[test]
fn test_gang_adaptive_wait_one_worker_two_epochs_never_loses_a_wake() {
    model(|| epochs(&adaptive_options(1, 3_000), 1, 2));
}

#[test]
fn test_gang_adaptive_wait_two_workers_one_epoch_never_loses_a_wake() {
    small_model(|| epochs(&adaptive_options(2, 2_000), 2, 1));
}

#[test]
fn test_gang_adaptive_wait_with_address_wait_never_loses_a_wake() {
    model(|| {
        let mut opts = address_options(1);
        opts.adaptive_cap_ns = Some(3_000);
        epochs(&opts, 1, 2);
    });
}

#[test]
fn test_gang_adaptive_shutdown_joins_spinning_workers() {
    model(|| {
        let mut gang = Gang::new(&adaptive_options(2, 2_000)).unwrap();
        assert!(!gang.run(&|_w: usize| {}).unwrap());
        drop(gang);
    });
}

/// Whether the one worker is parked `probe` after a job, once the gang
/// has seen `gap`-long waits between jobs, and how long the probe took.
#[cfg(all(not(loom), not(miri)))]
fn trial(
    adaptive_cap_ns: Option<u64>,
    gap: std::time::Duration,
    probe: std::time::Duration,
) -> (bool, std::time::Duration) {
    let mut opts = options(1);
    opts.spin = 0;
    opts.adaptive_cap_ns = adaptive_cap_ns;
    let mut gang = Gang::new(&opts).unwrap();
    let f = |_w: usize| {};
    for _ in 0..24 {
        assert!(!gang.run(&f).unwrap());
        std::thread::sleep(gap);
    }
    assert!(!gang.run(&f).unwrap());
    // The probe's real length: longer than `probe` when the main thread
    // is descheduled.
    let start = std::time::Instant::now();
    std::thread::sleep(probe);
    let parked = gang.shared.parked.load(Ordering::SeqCst) > 0;
    (parked, start.elapsed())
}

/// Fresh trials until one parks within the probe, at most `trials`. A
/// trial can miss only through preemption: the main thread descheduled
/// while it teaches the gap, which inflates the learned EWMA, or the worker
/// descheduled through the probe, so it never reaches its park. Neither
/// makes a worker park early, so retrying keeps the check's strength: an
/// implementation that keeps spinning past the probe fails every trial
/// (#471).
#[cfg(all(not(loom), not(miri)))]
fn parks_in_some_trial(
    adaptive_cap_ns: Option<u64>,
    gap: std::time::Duration,
    probe: std::time::Duration,
    trials: usize,
) -> bool {
    (0..trials).any(|_| trial(adaptive_cap_ns, gap, probe).0)
}

/// Whether the worker is still spinning after the probe, in the first of
/// `trials` fresh trials whose probe lasted less than `gap`; `false` if no
/// probe was that short, so a loaded host cannot pass the check vacuously
/// (#471). A longer probe can outlast the learned budget, and the worker
/// then parks by design.
#[cfg(all(not(loom), not(miri)))]
fn spins_through_some_probe(
    adaptive_cap_ns: Option<u64>,
    gap: std::time::Duration,
    probe: std::time::Duration,
    trials: usize,
) -> bool {
    for _ in 0..trials {
        let (parked, took) = trial(adaptive_cap_ns, gap, probe);
        if took < gap {
            return !parked;
        }
    }
    false
}

/// Trials for each expectation of the adaptive-wait test (#471).
#[cfg(all(not(loom), not(miri)))]
const PARK_TRIALS: usize = 8;

#[cfg(all(not(loom), not(miri)))]
#[test]
fn test_gang_adaptive_worker_spins_through_a_learned_gap() {
    let gap = std::time::Duration::from_millis(2);
    let probe = std::time::Duration::from_millis(1);
    // Fixed, no spins: parked at once.
    assert!(parks_in_some_trial(None, gap, probe, PARK_TRIALS));
    // Adaptive, cap 50 ms: the learned budget (about 2 x 2 ms) covers the
    // probe.
    assert!(spins_through_some_probe(
        Some(50_000_000),
        gap,
        probe,
        PARK_TRIALS
    ));
    // Adaptive, cap 0.2 ms: the cap ends the spinning before the probe.
    assert!(parks_in_some_trial(Some(200_000), gap, probe, PARK_TRIALS));
    // Adaptive, cap 50 ms, but back-to-back jobs taught a tiny EWMA: the
    // budget (2 x EWMA) ends the spinning long before the probe.
    assert!(parks_in_some_trial(
        Some(50_000_000),
        std::time::Duration::ZERO,
        probe,
        PARK_TRIALS
    ));
}

/// The owner-first Dynamic claim loop of `par_chunks` (ADR-0087), the real
/// [`Cursors::run`], over `chunks` chunks on `workers` workers: each worker
/// drains its own share, then steals from the others in rotation, and runs
/// each claimed chunk under the per-chunk catch. `panic_chunk` panics inside
/// a chunk (caught per chunk). `die_after_claim` makes that worker's first
/// claim end its job outside the per-chunk catch, after the claim and before
/// running the chunk. Returns the run's result and how many times each chunk
/// ran.
fn dynamic_claims(
    workers: usize,
    chunks: usize,
    panic_chunk: Option<usize>,
    die_after_claim: Option<usize>,
) -> (Result<bool, EngineError>, Vec<usize>) {
    let runs: Vec<AtomicUsize> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
    let claimed: Vec<AtomicBool> = (0..workers).map(|_| AtomicBool::new(false)).collect();
    let chunk_panicked = AtomicBool::new(false);
    let cursors = Cursors::new(workers);
    let result = {
        let mut gang = Gang::new(&options(workers)).unwrap();
        let job = |c: usize, w: usize| {
            let first = !claimed[w].swap(true, Ordering::Relaxed);
            assert!(
                !(first && die_after_claim == Some(w)),
                "injected death after claiming chunk {c}"
            );
            let ok = crate::sys::catch(|| {
                runs[c].fetch_add(1, Ordering::Relaxed);
                assert!(panic_chunk != Some(c), "injected chunk panic");
            });
            if !ok {
                chunk_panicked.store(true, Ordering::Relaxed);
            }
        };
        let result = cursors.run(&mut gang, chunks, &job);
        result.map(|gang_panicked| gang_panicked || chunk_panicked.load(Ordering::Relaxed))
    };
    let counts = runs.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    (result, counts)
}

/// Two workers, three chunks: shares {0, 2} and {1}, so worker 1 drains its
/// one chunk and then steals from share 0 while its owner claims from it.
#[test]
fn test_gang_dynamic_claims_run_every_chunk_exactly_once() {
    model(|| {
        let (result, counts) = dynamic_claims(2, 3, None, None);
        assert_eq!(result, Ok(false));
        assert_eq!(counts, vec![1, 1, 1]);
    });
}

/// The claim loop alone on three threads, four chunks: shares {0, 3}, {1}
/// and {2}, so two stealers race the owner of share 0 and each other.
#[test]
fn test_gang_dynamic_claims_on_three_workers_run_every_chunk_exactly_once() {
    model(|| {
        const CHUNKS: usize = 4;
        let cursors = Arc::new(Cursors::new(3));
        let runs: Arc<Vec<AtomicUsize>> =
            Arc::new((0..CHUNKS).map(|_| AtomicUsize::new(0)).collect());
        let spawn = |w: usize| {
            let (cursors, runs) = (Arc::clone(&cursors), Arc::clone(&runs));
            thread::spawn(move || {
                cursors.claim(w, CHUNKS, |c| {
                    runs[c].fetch_add(1, Ordering::Relaxed);
                });
            })
        };
        let (a, b) = (spawn(1), spawn(2));
        cursors.claim(0, CHUNKS, |c| {
            runs[c].fetch_add(1, Ordering::Relaxed);
        });
        a.join().unwrap();
        b.join().unwrap();
        let counts: Vec<usize> = runs.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        assert_eq!(counts, vec![1; CHUNKS]);
    });
}

/// Two jobs back to back on one cursor array (the reset between them is
/// `Cursors::run`'s, as in `par_chunks`): a slow stealer of the first job
/// never claims after the reset, and each chunk of each job runs exactly
/// once (ADR-0087 point 6).
#[test]
fn test_gang_dynamic_claims_back_to_back_jobs_run_each_chunk_once_per_job() {
    model(|| {
        const CHUNKS: usize = 3;
        let cursors = Cursors::new(2);
        let mut gang = Gang::new(&options(2)).unwrap();
        let first: Vec<AtomicUsize> = (0..CHUNKS).map(|_| AtomicUsize::new(0)).collect();
        let second: Vec<AtomicUsize> = (0..CHUNKS).map(|_| AtomicUsize::new(0)).collect();
        let first_done = AtomicBool::new(false);
        let late = AtomicBool::new(false);
        let job_a = |c: usize, _w: usize| {
            if first_done.load(Ordering::SeqCst) {
                late.store(true, Ordering::SeqCst);
            }
            first[c].fetch_add(1, Ordering::Relaxed);
        };
        assert_eq!(cursors.run(&mut gang, CHUNKS, &job_a), Ok(false));
        first_done.store(true, Ordering::SeqCst);
        let job_b = |c: usize, _w: usize| {
            second[c].fetch_add(1, Ordering::Relaxed);
        };
        assert_eq!(cursors.run(&mut gang, CHUNKS, &job_b), Ok(false));
        assert!(
            !late.load(Ordering::SeqCst),
            "a claim of job A after its run"
        );
        let a: Vec<usize> = first.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        let b: Vec<usize> = second.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        assert_eq!(a, vec![1; CHUNKS]);
        assert_eq!(b, vec![1; CHUNKS]);
    });
}

#[test]
fn test_gang_dynamic_claims_with_a_panicking_chunk_run_the_rest_and_poison() {
    model(|| {
        let (result, counts) = dynamic_claims(2, 3, Some(1), None);
        assert_eq!(result, Ok(true), "a panicking chunk is reported");
        assert_eq!(
            counts,
            vec![1, 1, 1],
            "the other chunks still run, each once"
        );
    });
}

#[test]
fn test_gang_dynamic_claims_worker_dying_after_a_claim_never_reports_success() {
    model(|| {
        let (result, counts) = dynamic_claims(2, 3, None, Some(0));
        assert!(
            counts.iter().all(|&c| c <= 1),
            "no chunk runs twice: {counts:?}"
        );
        let lost = counts.iter().filter(|&&c| c == 0).count();
        // Worker 0 dies only if it claims a chunk before worker 1 has taken
        // them all; when it does, exactly its claim is lost and the run
        // reports a panic, never success.
        match lost {
            0 => assert_eq!(result, Ok(false), "no claim, no death: {counts:?}"),
            1 => assert_eq!(
                result,
                Ok(true),
                "a death after claiming poisons: {counts:?}"
            ),
            _ => panic!("more than the dead worker's one claim lost: {counts:?}"),
        }
    });
}

#[test]
fn test_gang_dynamic_claims_with_a_dead_worker_backstop_complete_then_report() {
    model(|| {
        const CHUNKS: usize = 3;
        let mut opts = options(2);
        opts.faults.die_in_epoch = Some(1);
        let mut gang = Gang::new(&opts).unwrap();
        let cursors = Cursors::new(2);
        let runs: Vec<AtomicUsize> = (0..CHUNKS).map(|_| AtomicUsize::new(0)).collect();
        let job = |c: usize, _w: usize| {
            runs[c].fetch_add(1, Ordering::Relaxed);
        };
        // The dead worker never claims, so the live one runs every chunk once
        // (its own share, then the dead worker's); the run still reports the
        // death as a panic, never success.
        assert_eq!(cursors.run(&mut gang, CHUNKS, &job), Ok(true));
        let counts: Vec<usize> = runs.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        assert_eq!(counts, vec![1; CHUNKS]);
        assert_eq!(
            gang.run(&|_w: usize| {}).map(|_| ()),
            Err(EngineError::WorkerDied)
        );
    });
}

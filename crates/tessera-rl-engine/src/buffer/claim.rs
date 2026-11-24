// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owner-first chunk claiming for `Schedule::Dynamic` (ADR-0087, amending
//! ADR-0076).
//!
//! Worker `w` of `workers` owns the chunks `Static` would give it
//! (`c % workers == w`). Each owner share has its own claim cursor on its
//! own cache line; cursor value `k` of share `v` is chunk `v + k * workers`
//! while that is below the job's chunk count, and the share is exhausted
//! after that. A worker drains its own share, then the other shares in the
//! fixed rotation `w + 1, w + 2, ...` (mod `workers`), then stops.
//!
//! Each chunk belongs to exactly one share and each share's cursor hands
//! each value out once (an atomic read-modify-write), so each chunk is
//! claimed by exactly one worker; `Relaxed` is enough for that, and the
//! caller sees the chunk writes through the job's completion count.
//!
//! The cursors are allocated once, sized to the worker count, and reset to
//! zero before each job is published, only when no worker of the previous
//! job on the same array can still claim (ADR-0087 point 6): for gang jobs
//! by [`Cursors::run`], which resets before `Gang::run` and returns after
//! every worker's last access to the job (`sys::gang` invariant 3); for a
//! Pipelined half by its `publish`, after the half's per-worker `remaining`
//! count reached zero.

use crate::error::EngineError;
use crate::sys::Gang;
use crate::sys::sync::{AtomicUsize, Ordering};

/// One share's claim cursor, alone on its cache line (128 bytes covers the
/// adjacent-line prefetch pair on x86-64 and the line size on Apple
/// silicon).
#[derive(Debug, Default)]
#[repr(align(128))]
struct Cursor(AtomicUsize);

/// The claim cursors of one gang or one Pipelined half: one per owner
/// share, so one per worker.
#[derive(Debug)]
pub(crate) struct Cursors {
    cursors: Box<[Cursor]>,
}

impl Cursors {
    /// Cursors for `workers` owner shares (at least one), all at zero.
    #[must_use]
    pub(crate) fn new(workers: usize) -> Self {
        Self {
            cursors: (0..workers.max(1)).map(|_| Cursor::default()).collect(),
        }
    }

    /// The number of owner shares.
    #[must_use]
    pub(crate) fn shares(&self) -> usize {
        self.cursors.len()
    }

    /// Sets every cursor back to zero. Only called when no worker of the
    /// previous job on these cursors can still claim (module docs); the
    /// job's publication (`SeqCst` in `Gang::submit`) orders these stores
    /// before every claim of the next job.
    pub(crate) fn reset(&self) {
        for cursor in &self.cursors {
            cursor.0.store(0, Ordering::Relaxed);
        }
    }

    /// The chunk that cursor value `k` of share `share` stands for, or
    /// `None` once the share is exhausted (`chunks` chunks in the job).
    #[must_use]
    #[inline]
    fn chunk_of(&self, share: usize, k: usize, chunks: usize) -> Option<usize> {
        k.checked_mul(self.shares())
            .and_then(|offset| offset.checked_add(share))
            .filter(|&c| c < chunks)
    }

    /// Claims and runs, as worker `worker`, every chunk of a `chunks`-chunk
    /// job it can still get: its own share first, then the others in
    /// rotation. `run(c)` is called once per chunk this worker claimed. A
    /// worker index past the share count claims nothing.
    #[inline]
    pub(crate) fn claim(&self, worker: usize, chunks: usize, mut run: impl FnMut(usize)) {
        let shares = self.shares();
        if worker >= shares {
            return;
        }
        for step in 0..shares {
            // `worker` and `step` are both below `shares`: no overflow past
            // `2 * shares`, and the remainder is a valid share.
            let Some(share) = worker.checked_add(step).map(|s| s % shares) else {
                return;
            };
            let Some(cursor) = self.cursors.get(share) else {
                return;
            };
            loop {
                let k = cursor.0.fetch_add(1, Ordering::Relaxed);
                let Some(c) = self.chunk_of(share, k, chunks) else {
                    break;
                };
                run(c);
            }
        }
    }

    /// Runs `f(c, worker)` once for each of `chunks` chunks on `gang`'s
    /// workers under owner-first claiming. The cursors are reset before the
    /// job is published, and `Gang::run` returns only after every worker's
    /// decrement of the job's `pending`, which is its last access to the job
    /// (`sys::gang` invariant 3), so no worker of this job can claim after
    /// the next reset. Never two jobs in flight on one array: this calls
    /// `run`, never `submit` alone.
    ///
    /// # Errors
    ///
    /// The gang's errors, from [`Gang::run`].
    pub(crate) fn run<F>(&self, gang: &mut Gang, chunks: usize, f: &F) -> Result<bool, EngineError>
    where
        F: Fn(usize, usize) + Sync,
    {
        self.reset();
        gang.run(&|w: usize| self.claim(w, chunks, |c| f(c, w)))
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::Cursors;

    /// The chunks worker `worker` claims, in order, when it runs alone.
    fn alone(workers: usize, worker: usize, chunks: usize) -> Vec<usize> {
        let cursors = Cursors::new(workers);
        let mut order = Vec::new();
        cursors.claim(worker, chunks, |c| order.push(c));
        order
    }

    #[test]
    fn test_claim_alone_takes_own_share_then_rotation() {
        // 3 shares over 8 chunks: {0,3,6}, {1,4,7}, {2,5}.
        assert_eq!(alone(3, 0, 8), vec![0, 3, 6, 1, 4, 7, 2, 5]);
        assert_eq!(alone(3, 1, 8), vec![1, 4, 7, 2, 5, 0, 3, 6]);
        assert_eq!(alone(3, 2, 8), vec![2, 5, 0, 3, 6, 1, 4, 7]);
    }

    #[test]
    fn test_claim_fewer_chunks_than_workers_runs_each_once() {
        assert_eq!(alone(4, 3, 2), vec![0, 1]);
        assert_eq!(alone(4, 1, 0), Vec::<usize>::new());
    }

    #[test]
    fn test_claim_worker_past_shares_claims_nothing() {
        assert_eq!(alone(2, 2, 5), Vec::<usize>::new());
    }

    #[test]
    fn test_claim_zero_workers_is_one_share() {
        assert_eq!(alone(0, 0, 3), vec![0, 1, 2]);
    }

    #[test]
    fn test_claim_after_exhaustion_hands_nothing_out() {
        let cursors = Cursors::new(2);
        let mut first = Vec::new();
        cursors.claim(1, 7, |c| first.push(c));
        assert_eq!(first, vec![1, 3, 5, 0, 2, 4, 6]);
        let mut late = Vec::new();
        cursors.claim(0, 7, |c| late.push(c));
        assert!(late.is_empty(), "exhausted cursors hand nothing out");
    }

    #[test]
    fn test_claim_reset_hands_every_chunk_out_again() {
        let cursors = Cursors::new(2);
        let mut first = Vec::new();
        cursors.claim(0, 5, |c| first.push(c));
        cursors.reset();
        let mut second = Vec::new();
        cursors.claim(1, 5, |c| second.push(c));
        assert_eq!(first, vec![0, 2, 4, 1, 3]);
        assert_eq!(second, vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_claim_cursor_is_alone_on_its_cache_line() {
        assert_eq!(core::mem::align_of::<super::Cursor>(), 128);
        assert_eq!(core::mem::size_of::<super::Cursor>(), 128);
    }
}

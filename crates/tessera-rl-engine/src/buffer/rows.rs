// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-instance action rows for the split-phase sends (ADR-0047).
//!
//! An Async `send(ids)` validates and copies the actions of a few ids while
//! other instances are in flight, so it must never hold a slice over the
//! whole header or action buffers: a worker may be writing another row at
//! the same time, and a `&mut [T]` spanning that row is undefined behaviour
//! even if only admitted rows are touched. [`RawRows`] keeps raw parts and
//! hands out one instance's rows at a time, and only for the ids a caller
//! has admitted (invariants 3 and 4 of this module).

use super::Raw;
use crate::validate::{ActionRow, AdmittedRows};

/// Raw parts of the headers and action buffers, `[N]`, `[N, A, dh]` and
/// `[N, A, ch]`.
pub(crate) struct RawRows<H> {
    instances: usize,
    agents: usize,
    dh: usize,
    ch: usize,
    headers: Raw<H>,
    discrete: Raw<i32>,
    continuous: Raw<f32>,
}

impl<H: Copy> RawRows<H> {
    /// Takes the raw parts of the buffers, `[N]`, `[N, agents, dh]` and
    /// `[N, agents, ch]`. Build it while nothing is in flight: the `&mut`
    /// borrows end here, and the parts stay valid while the buffers are not
    /// moved, reallocated or borrowed mutably again (invariant 1).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
    )]
    pub(crate) fn new(
        headers: &mut [H],
        discrete: &mut [i32],
        continuous: &mut [f32],
        agents: usize,
        dh: usize,
        ch: usize,
    ) -> Self {
        let rows = headers.len().saturating_mul(agents);
        debug_assert_eq!(
            discrete.len(),
            rows.saturating_mul(dh),
            "discrete buffer size"
        );
        debug_assert_eq!(
            continuous.len(),
            rows.saturating_mul(ch),
            "continuous buffer size"
        );
        Self {
            instances: headers.len(),
            agents,
            dh,
            ch,
            headers: Raw::new(headers),
            discrete: Raw::new(discrete),
            continuous: Raw::new(continuous),
        }
    }

    /// The same rows from raw parts already taken by the caller (the
    /// Async mode takes them with the rest of its raw chunk, so no second
    /// `&mut` of the buffers is needed). The parts must describe buffers of
    /// `[instances]`, `[instances, agents, dh]` and `[instances, agents,
    /// ch]` (invariant 1).
    #[cfg(not(loom))]
    pub(super) const fn from_parts(
        instances: usize,
        agents: usize,
        (dh, ch): (usize, usize),
        headers: Raw<H>,
        discrete: Raw<i32>,
        continuous: Raw<f32>,
    ) -> Self {
        Self {
            instances,
            agents,
            dh,
            ch,
            headers,
            discrete,
            continuous,
        }
    }

    /// The rows of `ids`, by position.
    ///
    /// # Safety
    ///
    /// Invariants 1, 3 and 4: the buffers stay alive and in place for `'a`,
    /// and every row of every id in `ids` is exclusively the caller's for
    /// `'a` (covered by its admission, so no worker or other view touches
    /// it). `ids` must not repeat an id.
    #[expect(unsafe_code, reason = "audited module: token-guarded rows")]
    #[cfg_attr(
        loom,
        allow(dead_code, reason = "the split-phase modes are not built under loom")
    )]
    pub(crate) unsafe fn admitted<'a>(&'a self, ids: &'a [u32]) -> Admitted<'a, H> {
        Admitted { raw: self, ids }
    }

    /// The headers and action rows of `r` alone, or `None` if `r` is
    /// reversed or reaches past the last instance.
    ///
    /// # Safety
    ///
    /// As [`Self::admitted`], for every row of `r`.
    #[expect(unsafe_code, reason = "audited module: token-guarded rows")]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the split-phase modes (ADR-0047)")
    )]
    pub(crate) unsafe fn range<'a>(
        &self,
        r: core::ops::Range<usize>,
    ) -> Option<(&'a [H], &'a mut [i32], &'a mut [f32])> {
        if r.start > r.end || r.end > self.instances {
            return None;
        }
        let a = self.agents;
        // SAFETY: `r` lies inside the instances (checked above); the caller
        // guarantees its rows are exclusively its own for `'a` and the
        // buffers stay in place (invariants 1, 3 and 4).
        let headers = unsafe { self.headers.view(&r, 1) };
        // SAFETY: as above, `a * dh` values per instance.
        let discrete = unsafe { self.discrete.view(&r, a * self.dh) };
        // SAFETY: as above, `a * ch` values per instance.
        let continuous = unsafe { self.continuous.view(&r, a * self.ch) };
        Some((headers, discrete, continuous))
    }
}

/// The rows of the ids a send admitted (see [`RawRows::admitted`]).
pub(crate) struct Admitted<'a, H> {
    raw: &'a RawRows<H>,
    ids: &'a [u32],
}

impl<H: Copy> AdmittedRows<H> for Admitted<'_, H> {
    fn len(&self) -> usize {
        self.ids.len()
    }

    fn instances(&self) -> usize {
        self.raw.instances
    }

    fn id(&self, position: usize) -> Option<u32> {
        self.ids.get(position).copied()
    }

    #[expect(unsafe_code, reason = "audited module: token-guarded rows")]
    fn row(&mut self, position: usize) -> Option<ActionRow<'_, H>> {
        let id = usize::try_from(*self.ids.get(position)?).ok()?;
        if id >= self.raw.instances {
            return None;
        }
        let r = id..id + 1;
        let a = self.raw.agents;
        // SAFETY: `id` is in range (checked above) and is one of the ids the
        // caller declared exclusively its own when it built `Admitted`
        // (invariants 3 and 4). The returned borrows hold `&mut self`, so
        // at most one admitted row is live at a time; the header is only
        // read.
        let header = unsafe { self.raw.headers.view(&r, 1) };
        // SAFETY: as above, `a * dh` values of row `id`.
        let discrete = unsafe { self.raw.discrete.view(&r, a * self.raw.dh) };
        // SAFETY: as above, `a * ch` values of row `id`.
        let continuous = unsafe { self.raw.continuous.view(&r, a * self.raw.ch) };
        Some(ActionRow {
            header: header.first()?,
            discrete,
            continuous,
        })
    }
}

#[cfg(test)]
mod tests {
    use core::ptr;

    use tessera_rl_core::{Bool8, NoAdaptive, RecordHeader, StateInit};

    use super::RawRows;
    use crate::error::EngineError;
    use crate::layout::ActionLayout;
    use crate::options::AutoReset;
    use crate::validate::{
        ActionInput, check_and_copy_range_actions, check_and_copy_subset_actions,
    };

    type H = RecordHeader<2, NoAdaptive>;
    const N: usize = 4;
    const A: usize = 2;

    fn running() -> H {
        let mut h = H::state_default();
        h.engine.pending_reset = Bool8::FALSE;
        h.engine.fresh = Bool8::FALSE;
        for row in &mut h.agents {
            row.alive = Bool8::TRUE;
            row.acting = Bool8::TRUE;
        }
        h
    }

    fn layout() -> ActionLayout {
        ActionLayout {
            discrete: vec![3],
            continuous: Vec::new(),
            ..ActionLayout::default()
        }
    }

    /// One discrete head and one continuous head.
    fn layout_c() -> ActionLayout {
        ActionLayout {
            discrete: vec![3],
            continuous: vec![(-1.0, 1.0)],
            ..ActionLayout::default()
        }
    }

    /// A raw pointer that may cross threads: the test writes only rows the
    /// main thread never admits.
    struct Shared<T>(*mut T);
    // SAFETY: the pointer is used only for rows disjoint from the rows the
    // other thread borrows, and only while the buffer is alive.
    #[expect(unsafe_code, reason = "test of the token-guarded rows")]
    unsafe impl<T> Send for Shared<T> {}

    /// A worker writes the header and actions of instance 0 while the
    /// caller validates and copies ids 3 and 1. Under Miri, a slice over
    /// the whole buffer in the caller would be a data race or a retag
    /// conflict with the worker's writes.
    #[test]
    #[expect(unsafe_code, reason = "test of the token-guarded rows")]
    fn test_raw_rows_subset_copy_while_another_row_is_written_no_race() {
        let mut headers: Vec<H> = (0..N).map(|_| running()).collect();
        let mut discrete = vec![9_i32; N * A];
        let mut continuous = vec![9.0_f32; N * A];
        let raw = RawRows::new(&mut headers, &mut discrete, &mut continuous, A, 1, 1);
        let hdr_ptr = Shared(raw.headers.ptr);
        let act_ptr = Shared(raw.discrete.ptr);
        let cont_ptr = Shared(raw.continuous.ptr);
        let worker = std::thread::spawn(move || {
            let (h, d, c) = (hdr_ptr, act_ptr, cont_ptr);
            for i in 0..20_u32 {
                // SAFETY: row 0 is never admitted by the main thread, and the
                // buffers outlive the join below.
                let row = unsafe { &mut *h.0 };
                row.engine.episode_steps = i;
                // SAFETY: as above, the first action of row 0.
                unsafe { ptr::write(d.0, i32::try_from(i % 3).unwrap_or(0)) };
                // SAFETY: as above, the second action of row 0.
                let second = unsafe { d.0.add(1) };
                // SAFETY: as above.
                unsafe { ptr::write(second, 1) };
                // SAFETY: as above, the first continuous action of row 0.
                unsafe { ptr::write(c.0, 0.75) };
            }
        });
        let ids = [3_u32, 1];
        let acts = [1, 2, 0, 2];
        let conts = [0.5_f32, -0.5, 0.25, -0.25];
        let input = ActionInput {
            discrete: &acts,
            continuous: &conts,
        };
        for _ in 0..5 {
            // SAFETY: ids 3 and 1 are this thread's alone; the worker
            // touches only row 0, and the buffers outlive this scope.
            let mut rows = unsafe { raw.admitted(&ids) };
            check_and_copy_subset_actions(&layout_c(), AutoReset::NextStep, &mut rows, input)
                .unwrap();
        }
        worker.join().unwrap();
        assert_eq!(
            &continuous[2..4],
            &[0.25, -0.25],
            "id 1 takes source position 1"
        );
        assert_eq!(
            &continuous[6..8],
            &[0.5, -0.5],
            "id 3 takes source position 0"
        );
        assert_eq!(&continuous[4..6], &[9.0, 9.0], "row 2 was not admitted");
        assert_eq!(&discrete[2..4], &[0, 2]);
        assert_eq!(&discrete[6..8], &[1, 2]);
        assert_eq!(&discrete[4..6], &[9, 9], "row 2 was not admitted");
    }

    /// The Pipelined path borrows only the half: a worker writes half 0
    /// while the caller copies half 1.
    #[test]
    #[expect(unsafe_code, reason = "test of the token-guarded rows")]
    fn test_raw_rows_range_copy_while_the_other_half_is_written_no_race() {
        let mut headers: Vec<H> = (0..N).map(|_| running()).collect();
        let mut discrete = vec![9_i32; N * A];
        let mut continuous = vec![9.0_f32; N * A];
        let raw = RawRows::new(&mut headers, &mut discrete, &mut continuous, A, 1, 1);
        let hdr_ptr = Shared(raw.headers.ptr);
        let act_ptr = Shared(raw.discrete.ptr);
        let cont_ptr = Shared(raw.continuous.ptr);
        let worker = std::thread::spawn(move || {
            let (h, d, c) = (hdr_ptr, act_ptr, cont_ptr);
            for i in 0..20_u32 {
                // SAFETY: half 0 (rows 0 and 1) is never borrowed by the
                // main thread, and the buffers outlive the join below.
                let second_row = unsafe { h.0.add(1) };
                // SAFETY: as above, the header of row 1.
                unsafe { (*second_row).engine.episode_steps = i };
                // SAFETY: as above, the last action of row 1.
                let last = unsafe { d.0.add(3) };
                // SAFETY: as above.
                unsafe { ptr::write(last, 1) };
                // SAFETY: as above, the last continuous action of row 1.
                let last_c = unsafe { c.0.add(3) };
                // SAFETY: as above.
                unsafe { ptr::write(last_c, 0.75) };
            }
        });
        let acts = [2, 1, 0, 1];
        let conts = [0.1_f32, 0.2, 0.3, 0.4];
        let input = ActionInput {
            discrete: &acts,
            continuous: &conts,
        };
        for _ in 0..5 {
            // SAFETY: half 1 (rows 2 and 3) is this thread's alone.
            let (h, d, c) = unsafe { raw.range(2..4) }.unwrap();
            check_and_copy_range_actions(&layout_c(), AutoReset::NextStep, 2, h, input, d, c)
                .unwrap();
        }
        worker.join().unwrap();
        assert_eq!(&discrete[4..8], &[2, 1, 0, 1]);
        assert_eq!(&continuous[4..8], &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    #[expect(unsafe_code, reason = "test of the token-guarded rows")]
    fn test_raw_rows_bad_ranges_and_ids_are_refused() {
        let mut headers: Vec<H> = (0..N).map(|_| running()).collect();
        let mut discrete = vec![0_i32; N * A];
        let mut continuous: Vec<f32> = Vec::new();
        let raw = RawRows::new(&mut headers, &mut discrete, &mut continuous, A, 1, 0);
        let (end, start) = (2, 3);
        // SAFETY: nothing else uses the buffers in this test.
        assert!(unsafe { raw.range(start..end) }.is_none());
        // SAFETY: as above.
        assert!(unsafe { raw.range(2..5) }.is_none());
        let ids = [7_u32];
        // SAFETY: as above.
        let mut rows = unsafe { raw.admitted(&ids) };
        let err = check_and_copy_subset_actions(
            &layout(),
            AutoReset::NextStep,
            &mut rows,
            ActionInput {
                discrete: &[0, 0],
                continuous: &[],
            },
        );
        assert!(matches!(
            err,
            Err(EngineError::InstanceOutOfRange { id: 7, num_envs: 4 })
        ));
    }
}

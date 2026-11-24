//! The field types of a lane-wise state, and its column store (ADR-0090
//! decisions 1 and 2). **Audited `unsafe` module:** [`RawColumn::view`]
//! is the only unsafe code; the engine's buffer module calls it, through
//! [`LaneTuple::view`], for the disjoint chunk ranges of its invariants.
//!
//! A state with `#[state(lanes)]` is stored as one column per field. Each
//! field's type must implement [`LaneField`]: `f64`, `f32`, `i32`, `u32`,
//! `Fx32`, `Bool8`, and `#[derive(StateEnum)]` enums. Arrays,
//! `Pool<T, CAP>`, nested states and `Reserved<N>` do not, so the derive
//! refuses them at compile time. The state's `EnvState::Fields`, the tuple
//! of its field types, implements the sealed [`LaneTuple`]: one `Vec` per
//! element, each padded to a multiple of [`LANES`], and views that are
//! tuples of `&mut [T]`. Reading or writing one instance through a view is
//! safe and bounds-checked. States with `Fields = ()` store nothing.

use core::ops::Range;

use tessera_rl_num::Fx32;

use crate::bool8::Bool8;

/// Rows per lane group: columns are padded to a multiple of it (ADR-0090).
pub const LANES: usize = 8;

/// A scalar field type of a lane-wise state.
pub trait LaneField: Copy + Send + Sync + 'static {}

impl LaneField for f64 {}
impl LaneField for f32 {}
impl LaneField for i32 {}
impl LaneField for u32 {}
impl LaneField for Fx32 {}
impl LaneField for Bool8 {}

/// A raw pointer and length of one column. Only [`LaneTuple::view`]
/// dereferences it.
#[derive(Debug)]
pub struct RawColumn<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Clone for RawColumn<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for RawColumn<T> {}

impl<T> RawColumn<T> {
    fn new(column: &mut [T]) -> Self {
        Self {
            ptr: column.as_mut_ptr(),
            len: column.len(),
        }
    }

    /// The rows `r` of the column, clamped to its length (an out-of-range
    /// request yields an empty view).
    ///
    /// # Safety
    ///
    /// The column is alive and not reallocated while the view lives, and no
    /// other live view covers `r`.
    #[expect(unsafe_code, reason = "audited module: disjoint column views")]
    unsafe fn view<'a>(self, r: &Range<usize>) -> &'a mut [T] {
        let start = r.start.min(self.len);
        let end = r.end.min(self.len).max(start);
        // SAFETY: `start <= len`, so the offset stays inside (or one past)
        // the live allocation the caller guarantees.
        let first = unsafe { self.ptr.add(start) };
        // SAFETY: `start <= end <= len` keeps the view inside the column,
        // which is live and not reallocated while the view lives; the
        // caller guarantees no other live view overlaps `r`.
        unsafe { core::slice::from_raw_parts_mut(first, end - start) }
    }
}

/// A tuple of lane field types, stored as one column per element.
pub trait LaneTuple: Copy + Send + Sync + 'static + crate::sealed::LaneTupleSealed {
    /// One `Vec` per element.
    type Columns: Send + Sync;
    /// One raw column pointer per element. Not `Send`: the engine's
    /// audited buffer module carries it across threads under its invariants.
    type Raw: Copy;
    /// One `&mut` column slice per element.
    type View<'a>;

    /// Columns for `n` instances, every row set to `fill`; each column's
    /// length is `n` rounded up to a multiple of [`LANES`], the padding
    /// rows holding `fill` too.
    fn columns(n: usize, fill: Self) -> Self::Columns;

    /// Instance `i`'s fields, or `None` past the columns.
    fn get(columns: &Self::Columns, i: usize) -> Option<Self>;

    /// Writes instance `i`'s fields; nothing past the columns.
    fn set(columns: &mut Self::Columns, i: usize, value: Self);

    /// The raw column pointers, for chunk views.
    fn raw(columns: &mut Self::Columns) -> Self::Raw;

    /// A view of every row.
    fn view_all(columns: &mut Self::Columns) -> Self::View<'_>;

    /// The raw column pointers of a view, so that a pool-wide view can be
    /// split into chunk views ([`LaneTuple::view`]).
    fn view_raw(view: &mut Self::View<'_>) -> Self::Raw;

    /// The view of the instances in `r`, indexed from `r.start`. Called only
    /// by the engine's audited buffer module, for its disjoint chunk
    /// ranges; not part of the API.
    ///
    /// # Safety
    ///
    /// The caller guarantees, for the whole life of the view:
    ///
    /// - **Disjoint ranges:** no other live view of these columns, and no
    ///   reference to them, covers any row of `r`.
    /// - **No reallocation:** the columns `raw` was taken from are alive
    ///   and are neither moved, resized nor dropped (the epoch rule of the
    ///   buffer module).
    /// - **Bounds:** `r` lies within the padded column length. A range past
    ///   it is clamped to an empty or shorter view, never read out of
    ///   bounds.
    #[doc(hidden)]
    #[expect(unsafe_code, reason = "audited module: disjoint column views")]
    unsafe fn view<'a>(raw: Self::Raw, r: &Range<usize>) -> Self::View<'a>;

    /// The fields of row `i` of a view, or `None` past it.
    fn view_get(view: &Self::View<'_>, i: usize) -> Option<Self>;

    /// Writes row `i` of a view; nothing past it.
    fn view_set(view: &mut Self::View<'_>, i: usize, value: Self);
}

impl crate::sealed::LaneTupleSealed for () {}

impl LaneTuple for () {
    type Columns = ();
    type Raw = ();
    type View<'a> = ();

    fn columns(_: usize, (): Self) -> Self::Columns {}

    fn get((): &Self::Columns, _: usize) -> Option<Self> {
        Some(())
    }

    fn set((): &mut Self::Columns, _: usize, (): Self) {}

    fn raw((): &mut Self::Columns) -> Self::Raw {}

    fn view_all((): &mut Self::Columns) -> Self::View<'_> {}

    fn view_raw((): &mut Self::View<'_>) -> Self::Raw {}

    #[expect(
        unsafe_code,
        reason = "audited module: trait method, no unsafe operation"
    )]
    unsafe fn view<'a>((): Self::Raw, _: &Range<usize>) -> Self::View<'a> {}

    fn view_get((): &Self::View<'_>, _: usize) -> Option<Self> {
        Some(())
    }

    fn view_set((): &mut Self::View<'_>, _: usize, (): Self) {}
}

/// `n` rounded up to a multiple of [`LANES`]; `n` itself if that overflows
/// (a pool that large cannot be allocated anyway).
fn padded(n: usize) -> usize {
    n.div_ceil(LANES).checked_mul(LANES).unwrap_or(n)
}

macro_rules! field_tuple {
    ($($t:ident $i:tt),+) => {
        impl<$($t: LaneField),+> crate::sealed::LaneTupleSealed for ($($t,)+) {}

        impl<$($t: LaneField),+> LaneTuple for ($($t,)+) {
            type Columns = ($(Vec<$t>,)+);
            type Raw = ($(RawColumn<$t>,)+);
            type View<'a> = ($(&'a mut [$t],)+);

            fn columns(n: usize, fill: Self) -> Self::Columns {
                let len = padded(n);
                ($(vec![fill.$i; len],)+)
            }

            fn get(columns: &Self::Columns, i: usize) -> Option<Self> {
                Some(($(*columns.$i.get(i)?,)+))
            }

            fn set(columns: &mut Self::Columns, i: usize, value: Self) {
                $(
                    if let Some(slot) = columns.$i.get_mut(i) {
                        *slot = value.$i;
                    }
                )+
            }

            fn raw(columns: &mut Self::Columns) -> Self::Raw {
                ($(RawColumn::new(columns.$i.as_mut_slice()),)+)
            }

            fn view_all(columns: &mut Self::Columns) -> Self::View<'_> {
                ($(columns.$i.as_mut_slice(),)+)
            }

            fn view_raw(view: &mut Self::View<'_>) -> Self::Raw {
                ($(RawColumn::new(&mut *view.$i),)+)
            }

            #[expect(unsafe_code, reason = "audited module: disjoint column views")]
            unsafe fn view<'a>(raw: Self::Raw, r: &Range<usize>) -> Self::View<'a> {
                // SAFETY: each column is a live, never reallocated buffer
                // with one element per instance (invariant 1), and the
                // caller guarantees that no other live view covers `r`
                // (invariant 2), so each column's view of `r` is disjoint
                // from every other view of that column.
                ($(unsafe { raw.$i.view(r) },)+)
            }

            fn view_get(view: &Self::View<'_>, i: usize) -> Option<Self> {
                Some(($(*view.$i.get(i)?,)+))
            }

            fn view_set(view: &mut Self::View<'_>, i: usize, value: Self) {
                $(
                    if let Some(slot) = view.$i.get_mut(i) {
                        *slot = value.$i;
                    }
                )+
            }
        }
    };
}

field_tuple!(A 0);
field_tuple!(A 0, B 1);
field_tuple!(A 0, B 1, C 2);
field_tuple!(A 0, B 1, C 2, D 3);
field_tuple!(A 0, B 1, C 2, D 3, E 4);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
field_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
// Twelve: `std` implements `Copy` for tuples up to that arity, and
// `EnvState::Fields` must be `Copy`; a larger lane-wise state fails to
// compile here.

#[cfg(test)]
mod tests {
    use super::*;

    type F = (f64, u32, i32);

    #[test]
    fn test_columns_pad_to_lanes_and_round_trip() {
        let mut c = <F as LaneTuple>::columns(9, (0.0, 0, 0));
        assert_eq!(c.0.len(), 16);
        <F as LaneTuple>::set(&mut c, 8, (1.5, 2, -3));
        assert_eq!(<F as LaneTuple>::get(&c, 8), Some((1.5, 2, -3)));
        assert_eq!(<F as LaneTuple>::get(&c, 16), None);
        <F as LaneTuple>::set(&mut c, 99, (9.0, 9, 9));
        assert_eq!(<F as LaneTuple>::get(&c, 15), Some((0.0, 0, 0)));
    }

    #[test]
    #[expect(unsafe_code, reason = "test of the audited disjoint views")]
    #[expect(
        clippy::many_single_char_names,
        reason = "three chunk views and their rows"
    )]
    fn test_columns_disjoint_chunk_views_write_their_own_rows() {
        let n = 24;
        let mut c = <F as LaneTuple>::columns(n, (0.0, 0, 0));
        let raw = <F as LaneTuple>::raw(&mut c);
        // Three disjoint chunks of 8, live at the same time (Miri checks
        // that the views do not alias).
        // SAFETY: the columns outlive the views and the ranges are disjoint.
        let mut a = unsafe { <F as LaneTuple>::view(raw, &(0..8)) };
        // SAFETY: as above.
        let mut b = unsafe { <F as LaneTuple>::view(raw, &(8..16)) };
        // SAFETY: as above.
        let mut d = unsafe { <F as LaneTuple>::view(raw, &(16..24)) };
        for k in 0..8u32 {
            let x = f64::from(k);
            <F as LaneTuple>::view_set(&mut a, k as usize, (x, k, 1));
            <F as LaneTuple>::view_set(&mut b, k as usize, (x, k, 2));
            <F as LaneTuple>::view_set(&mut d, k as usize, (x, k, 3));
        }
        assert_eq!(<F as LaneTuple>::view_get(&b, 3), Some((3.0, 3, 2)));
        assert_eq!(<F as LaneTuple>::view_get(&b, 8), None);
        <F as LaneTuple>::view_set(&mut d, 8, (7.0, 7, 7));
        for i in 0..n {
            let k = u32::try_from(i % 8).unwrap();
            let chunk = i32::try_from(i / 8 + 1).unwrap();
            assert_eq!(<F as LaneTuple>::get(&c, i), Some((f64::from(k), k, chunk)));
        }
    }

    #[test]
    fn test_columns_unit_store_holds_nothing() {
        <() as LaneTuple>::columns(8, ());
        assert_eq!(<() as LaneTuple>::get(&(), 3), Some(()));
        <() as LaneTuple>::view_all(&mut ());
        assert_eq!(<() as LaneTuple>::view_get(&(), 5), Some(()));
    }
}

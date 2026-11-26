//! The narrow loads, stores and the byte gather of `I32Lanes` (ADR-0100
//! decision 4) at every level this CPU has, bit for bit against the scalar
//! conversions (`i32::from`, and back in range).

#![expect(clippy::indexing_slicing, reason = "test helpers over [_; 8]")]
#![expect(clippy::cast_possible_truncation, reason = "generator words")]

use tessera_rl_num::math::lane_int::I32Lanes;
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

/// splitmix64: a fixed, seeded word stream, the same on every level and ISA.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// One input set: eight values of each narrow type.
#[derive(Clone, Copy, Debug)]
struct Narrow {
    s16: [i16; 8],
    u16: [u16; 8],
    u8: [u8; 8],
}

/// What a level returns: the widened lanes, and the round trips.
#[derive(Debug, PartialEq, Eq)]
struct Widened {
    s16: [i32; 8],
    u16: [i32; 8],
    u8: [i32; 8],
    back_s16: [i16; 8],
    back_u16: [u16; 8],
    back_u8: [u8; 8],
}

impl LaneKernel for Narrow {
    type Output = Widened;

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Widened {
        let (a, b, c) = (
            S::I32::from_i16(self.s16),
            S::I32::from_u16(self.u16),
            S::I32::from_u8(self.u8),
        );
        Widened {
            s16: a.to_array(),
            u16: b.to_array(),
            u8: c.to_array(),
            back_s16: a.to_i16(),
            back_u16: b.to_u16(),
            back_u8: c.to_u8(),
        }
    }
}

fn want(n: Narrow) -> Widened {
    Widened {
        s16: n.s16.map(i32::from),
        u16: n.u16.map(i32::from),
        u8: n.u8.map(i32::from),
        back_s16: n.s16,
        back_u16: n.u16,
        back_u8: n.u8,
    }
}

fn edges() -> Vec<Narrow> {
    let s = [i16::MIN, i16::MIN + 1, -2, -1, 0, 1, i16::MAX - 1, i16::MAX];
    let u = [0, 1, 2, 0x7fff, 0x8000, 0xfffe, u16::MAX, 0x1234];
    let b = [0, 1, 2, 0x7f, 0x80, 0xfe, u8::MAX, 0x5a];
    let mut out = vec![Narrow {
        s16: s,
        u16: u,
        u8: b,
    }];
    // Every rotation, so each edge visits every lane.
    for r in 1..8 {
        let mut n = out[0];
        n.s16.rotate_left(r);
        n.u16.rotate_left(r);
        n.u8.rotate_left(r);
        out.push(n);
    }
    out
}

fn randoms() -> Vec<Narrow> {
    let mut rng = Rng(0x0759_0100_d4d4_0001);
    (0..10_000)
        .map(|_| {
            let mut n = Narrow {
                s16: [0; 8],
                u16: [0; 8],
                u8: [0; 8],
            };
            for i in 0..8 {
                let w = rng.next();
                n.s16[i] = w as i16;
                n.u16[i] = (w >> 16) as u16;
                n.u8[i] = (w >> 32) as u8;
            }
            n
        })
        .collect()
}

fn check(inputs: &[Narrow]) {
    for level in Resolved::available() {
        for &n in inputs {
            assert_eq!(
                level.dispatch_lanes(n),
                want(n),
                "{:?}: {n:?}",
                level.level()
            );
        }
    }
}

#[test]
fn test_narrow_lanes_edges_match_scalar_at_every_level() {
    check(&edges());
}

#[test]
fn test_narrow_lanes_random_match_scalar_at_every_level() {
    check(&randoms());
}

/// Gathers `base` at `idx` on one level.
struct Gather<'a>(&'a [u8], [i32; 8]);

impl LaneKernel for Gather<'_> {
    type Output = [i32; 8];

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) -> [i32; 8] {
        S::I32::gather_u8(self.0, self.1).to_array()
    }
}

#[test]
fn test_gather_u8_matches_direct_index_at_every_level() {
    let base: Vec<u8> = (0..=255u8).chain(0..=200u8).collect();
    let last = i32::try_from(base.len() - 1).unwrap_or(0);
    let mut rng = Rng(0x0759_0100_6a7e_0002);
    let mut sets = vec![[0, 1, 2, 3, last, last - 1, 255, 256], [last; 8], [0; 8]];
    for _ in 0..1_000 {
        let mut idx = [0i32; 8];
        for i in &mut idx {
            *i = (rng.next() % base.len() as u64) as i32;
        }
        sets.push(idx);
    }
    for level in Resolved::available() {
        for &idx in &sets {
            let want = idx.map(|i| i32::from(base[usize::try_from(i).unwrap_or(0)]));
            assert_eq!(
                level.dispatch_lanes(Gather(&base, idx)),
                want,
                "{:?}: {idx:?}",
                level.level()
            );
        }
    }
}

/// An out-of-range lane is a caller bug: debug builds assert.
struct ToI16([i32; 8]);

impl LaneKernel for ToI16 {
    type Output = [i16; 8];

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) -> [i16; 8] {
        S::I32::from_array(self.0).to_i16()
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "is outside i16")]
fn test_to_i16_out_of_range_asserts_in_debug() {
    let _ = Resolved::best().dispatch_lanes(ToI16([0, 0, 0, 40_000, 0, 0, 0, 0]));
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "is outside 0..")]
fn test_gather_u8_out_of_range_asserts_in_debug() {
    let base = [1u8; 4];
    let _ = Resolved::best().dispatch_lanes(Gather(&base, [0, 1, 2, 3, 4, 0, 0, 0]));
}

/// In release, an out-of-range store keeps the low bits (documented, not
/// relied on).
#[cfg(not(debug_assertions))]
#[test]
fn test_to_i16_out_of_range_keeps_low_bits_in_release() {
    for level in Resolved::available() {
        let got = level.dispatch_lanes(ToI16([40_000, -40_000, 65_536, 0, 0, 0, 0, 0]));
        assert_eq!(
            got[..3],
            [40_000i32 as i16, -40_000i32 as i16, 0],
            "{:?}",
            level.level()
        );
    }
}

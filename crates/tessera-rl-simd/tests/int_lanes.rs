//! `I32Lanes` and `Fx32Lanes` at every level this CPU has, bit for bit
//! against the scalar `i32`, `u32` and `Fx32` operations (#345 stage 2).

#![expect(clippy::indexing_slicing, reason = "test helpers over [_; 8]")]
#![expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "u32 views and generator words"
)]

use tessera_rl_num::Fx32;
use tessera_rl_num::fp::with_hostile_flush_to_zero;
use tessera_rl_num::math::lane_int::{Fx32Lanes, I32Lanes};
use tessera_rl_simd::{LaneKernel, Lanes, Level, Resolved};

/// Every operation's result on one pair of eight-lane operands.
#[derive(Debug, PartialEq, Eq)]
struct Out {
    words: Vec<[i32; 8]>,
    masks: Vec<u8>,
}

struct All([i32; 8], [i32; 8]);

impl LaneKernel for All {
    type Output = Out;

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Out {
        let (a, b) = (S::I32::from_array(self.0), S::I32::from_array(self.1));
        let (fa, fb) = (S::Fx32::from_raw(a), S::Fx32::from_raw(b));
        let bits = <S::I32 as I32Lanes>::mask_bits;
        let (oa, ma) = a.overflowing_add(b);
        let (os, ms) = a.overflowing_sub(b);
        let (foa, fma) = fa.overflowing_add(fb);
        let (fos, fms) = fa.overflowing_sub(fb);
        let gt = a.gt(b);
        let words = vec![
            a.to_array(),
            a.wrapping_add(b).to_array(),
            a.wrapping_sub(b).to_array(),
            a.wrapping_mul(b).to_array(),
            oa.to_array(),
            os.to_array(),
            a.and(b).to_array(),
            a.or(b).to_array(),
            a.xor(b).to_array(),
            a.not().to_array(),
            a.shl::<1>().to_array(),
            a.shl::<16>().to_array(),
            a.shl::<31>().to_array(),
            a.shr_arith::<1>().to_array(),
            a.shr_arith::<16>().to_array(),
            a.shr_arith::<31>().to_array(),
            a.shr_logical::<1>().to_array(),
            a.shr_logical::<16>().to_array(),
            a.shr_logical::<31>().to_array(),
            a.min(b).to_array(),
            a.max(b).to_array(),
            a.min_u(b).to_array(),
            a.max_u(b).to_array(),
            S::I32::select(gt, a, b).to_array(),
            [a.extract::<0>(), a.extract::<7>(), 0, 0, 0, 0, 0, 0],
            a.replace::<3>(-7).to_array(),
            fa.saturating_add(fb).to_raw().to_array(),
            fa.saturating_sub(fb).to_raw().to_array(),
            fa.saturating_mul(fb).to_raw().to_array(),
            foa.to_raw().to_array(),
            fos.to_raw().to_array(),
            fa.min(fb).to_raw().to_array(),
            fa.max(fb).to_raw().to_array(),
            // Grid's cell: `to_raw() >> 16`, floor.
            fa.to_raw().shr_arith::<16>().to_array(),
        ];
        let masks = vec![
            bits(ma),
            bits(ms),
            bits(fma),
            bits(fms),
            bits(a.eq(b)),
            bits(a.ne(b)),
            bits(gt),
            bits(a.lt(b)),
            bits(a.le(b)),
            bits(a.ge(b)),
            bits(a.gt_u(b)),
            bits(a.lt_u(b)),
            bits(a.le_u(b)),
            bits(a.ge_u(b)),
            bits(fa.eq(fb)),
            bits(fa.lt(fb)),
            bits(fa.le(fb)),
            bits(fa.gt(fb)),
            bits(fa.ge(fb)),
            bits(<S::I32 as I32Lanes>::mask_and(gt, a.eq(b))),
            bits(<S::I32 as I32Lanes>::mask_or(gt, a.eq(b))),
            bits(<S::I32 as I32Lanes>::mask_not(gt)),
            bits(<S::I32 as I32Lanes>::mask_from_bits(0b1011_0010)),
        ];
        Out { words, masks }
    }
}

fn per_lane(a: [i32; 8], b: [i32; 8], f: impl Fn(i32, i32) -> i32) -> [i32; 8] {
    std::array::from_fn(|i| f(a[i], b[i]))
}

fn bits(a: [i32; 8], b: [i32; 8], f: impl Fn(i32, i32) -> bool) -> u8 {
    (0..8).fold(0, |m, i| m | (u8::from(f(a[i], b[i])) << i))
}

fn fx(f: impl Fn(Fx32, Fx32) -> Fx32) -> impl Fn(i32, i32) -> i32 {
    move |a, b| f(Fx32::from_raw(a), Fx32::from_raw(b)).to_raw()
}

/// The reference, written on the scalar types directly (not on the scalar
/// lane level), so the scalar level is tested too.
fn reference(a: [i32; 8], b: [i32; 8]) -> Out {
    let u = |v: i32| v as u32;
    let un = |f: fn(i32) -> i32| per_lane(a, b, move |x, _| f(x));
    let gt = bits(a, b, |x, y| x > y);
    let eq = bits(a, b, |x, y| x == y);
    let mut replaced = a;
    replaced[3] = -7;
    // The overflow masks also equal their documented 3.3 decomposition.
    let add_sign = bits(a, b, |x, y| {
        let w = x.wrapping_add(y);
        (x ^ w) & (y ^ w) < 0
    });
    let sub_sign = bits(a, b, |x, y| {
        let w = x.wrapping_sub(y);
        (x ^ y) & (x ^ w) < 0
    });
    let add_none = bits(a, b, |x, y| x.checked_add(y).is_none());
    let sub_none = bits(a, b, |x, y| x.checked_sub(y).is_none());
    assert_eq!(add_sign, add_none);
    assert_eq!(sub_sign, sub_none);
    let fx_add_none = bits(a, b, |x, y| {
        Fx32::from_raw(x).checked_add(Fx32::from_raw(y)).is_none()
    });
    let fx_sub_none = bits(a, b, |x, y| {
        Fx32::from_raw(x).checked_sub(Fx32::from_raw(y)).is_none()
    });
    let words = vec![
        a,
        per_lane(a, b, i32::wrapping_add),
        per_lane(a, b, i32::wrapping_sub),
        per_lane(a, b, i32::wrapping_mul),
        per_lane(a, b, i32::wrapping_add),
        per_lane(a, b, i32::wrapping_sub),
        per_lane(a, b, |x, y| x & y),
        per_lane(a, b, |x, y| x | y),
        per_lane(a, b, |x, y| x ^ y),
        un(|x| !x),
        un(|x| x << 1),
        un(|x| x << 16),
        un(|x| x << 31),
        un(|x| x >> 1),
        un(|x| x >> 16),
        un(|x| x >> 31),
        un(|x| ((x as u32) >> 1) as i32),
        un(|x| ((x as u32) >> 16) as i32),
        un(|x| ((x as u32) >> 31) as i32),
        per_lane(a, b, i32::min),
        per_lane(a, b, i32::max),
        per_lane(a, b, |x, y| (x as u32).min(y as u32) as i32),
        per_lane(a, b, |x, y| (x as u32).max(y as u32) as i32),
        per_lane(a, b, |x, y| if x > y { x } else { y }),
        [a[0], a[7], 0, 0, 0, 0, 0, 0],
        replaced,
        per_lane(a, b, fx(Fx32::saturating_add)),
        per_lane(a, b, fx(Fx32::saturating_sub)),
        per_lane(a, b, fx(Fx32::saturating_mul)),
        per_lane(a, b, fx(Fx32::wrapping_add)),
        per_lane(a, b, fx(Fx32::wrapping_sub)),
        per_lane(a, b, fx(Fx32::min)),
        per_lane(a, b, fx(Fx32::max)),
        un(|x| x >> 16),
    ];
    let masks = vec![
        add_none,
        sub_none,
        fx_add_none,
        fx_sub_none,
        eq,
        bits(a, b, |x, y| x != y),
        gt,
        bits(a, b, |x, y| x < y),
        bits(a, b, |x, y| x <= y),
        bits(a, b, |x, y| x >= y),
        bits(a, b, |x, y| u(x) > u(y)),
        bits(a, b, |x, y| u(x) < u(y)),
        bits(a, b, |x, y| u(x) <= u(y)),
        bits(a, b, |x, y| u(x) >= u(y)),
        eq,
        bits(a, b, |x, y| x < y),
        bits(a, b, |x, y| x <= y),
        gt,
        bits(a, b, |x, y| x >= y),
        gt & eq,
        gt | eq,
        !gt,
        0b1011_0010,
    ];
    Out { words, masks }
}

/// Values at every boundary the multiply, the shifts and the cell
/// conversion have: the extremes, ±1, Fx32 ±0.5, ±1.0, ±1.5, ±2.0, and
/// raws whose product lands around bit 47 (the overflow boundary).
const EDGES: [i32; 30] = [
    i32::MIN,
    i32::MIN + 1,
    i32::MAX,
    i32::MAX - 1,
    -1,
    0,
    1,
    32_768,   // 0.5
    -32_768,  // -0.5 → cell -1
    65_536,   // 1.0
    -65_536,  // -1.0 → cell -1
    -98_304,  // -1.5 → cell -2
    -131_072, // -2.0 → cell -2
    -131_073, // just below -2.0 → cell -3
    98_304,   // 1.5
    131_072,  // 2.0
    1 << 15,
    -(1 << 15),
    (1 << 23) - 1,
    -(1 << 23),
    1 << 23,
    (1 << 24) + 1,
    46_341 << 8, // √2^31 · 2^8: products near 2^47
    -(46_341 << 8),
    46_340 << 8,
    -(46_340 << 8),
    0x7fff,
    -0x8001,
    0x5555_5555,
    -0x5555_5556,
];

/// splitmix64: a fixed, seeded word stream, the same on every level and ISA.
struct Words(u64);

impl Words {
    fn next(&mut self) -> i32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as i32
    }
}

/// Every edge against every edge, eight pairs at a time.
fn edge_pairs() -> Vec<([i32; 8], [i32; 8])> {
    let pairs: Vec<(i32, i32)> = EDGES
        .iter()
        .flat_map(|&x| EDGES.iter().map(move |&y| (x, y)))
        .collect();
    pairs
        .chunks(8)
        .map(|c| {
            let mut a = [0; 8];
            let mut b = [0; 8];
            for (i, &(x, y)) in c.iter().enumerate() {
                a[i] = x;
                b[i] = y;
            }
            (a, b)
        })
        .collect()
}

/// 2^18 random pairs; half with operands shrunk so products cross bit 47.
fn random_pairs() -> Vec<([i32; 8], [i32; 8])> {
    let mut w = Words(0x7e55_e2a0_0345_0002);
    (0..(1 << 18) / 8)
        .map(|n| {
            let shift = if n % 2 == 0 { 0 } else { (n / 2 % 17) as u32 };
            let a = std::array::from_fn(|_| w.next() >> shift);
            let b = std::array::from_fn(|_| w.next() >> shift);
            (a, b)
        })
        .collect()
}

/// The digest of every result over [`random_pairs`]: the value xaskar's
/// scalar, SSE4.2 and AVX2 levels agreed on in gate c1-r2 (45bb0296).
const PINNED_DIGEST: u64 = 0xf927_15bf_1e18_26a0;

/// FNV-1a over every result word and mask, in order.
fn digest(outs: &[Out]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |byte: u8| {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for o in outs {
        for w in o.words.iter().flatten() {
            w.to_le_bytes().into_iter().for_each(&mut eat);
        }
        o.masks.iter().copied().for_each(&mut eat);
    }
    h
}

fn check_level(level: Resolved, pairs: &[([i32; 8], [i32; 8])]) -> Vec<Out> {
    pairs
        .iter()
        .map(|&(a, b)| {
            let got = level.dispatch_lanes(All(a, b));
            let want = reference(a, b);
            for (k, (g, w)) in got.words.iter().zip(&want.words).enumerate() {
                assert_eq!(g, w, "{:?} word op {k}: a={a:?} b={b:?}", level.level());
            }
            for (k, (g, w)) in got.masks.iter().zip(&want.masks).enumerate() {
                assert_eq!(g, w, "{:?} mask op {k}: a={a:?} b={b:?}", level.level());
            }
            assert_eq!(got.words.len(), want.words.len());
            assert_eq!(got.masks.len(), want.masks.len());
            got
        })
        .collect()
}

#[test]
fn test_int_lanes_edges_match_scalar_at_every_level() {
    let pairs = edge_pairs();
    for level in Resolved::available() {
        check_level(level, &pairs);
    }
}

#[test]
fn test_int_lanes_random_match_scalar_at_every_level() {
    let pairs = random_pairs();
    let digests: Vec<u64> = Resolved::available()
        .map(|level| digest(&check_level(level, &pairs)))
        .collect();
    assert!(digests.windows(2).all(|d| d[0] == d[1]), "{digests:x?}");
    // The same committed value on x86-64 and arm64. Pinned from the first
    // gate run (the assert prints the computed value), then cross-checked.
    assert_eq!(digests[0], PINNED_DIGEST, "{:016x}", digests[0]);
}

#[test]
fn test_int_lanes_hostile_flush_to_zero_matches_scalar() {
    let pairs = edge_pairs();
    for level in Resolved::available() {
        with_hostile_flush_to_zero(|| check_level(level, &pairs));
    }
}

#[test]
fn test_fx32_cell_conversion_is_floor() {
    struct Cell([i32; 8]);
    impl LaneKernel for Cell {
        type Output = [i32; 8];
        #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
        #[inline(always)]
        fn run<S: Lanes>(self) -> [i32; 8] {
            S::Fx32::from_raw(S::I32::from_array(self.0))
                .to_raw()
                .shr_arith::<16>()
                .to_array()
        }
    }
    let cells: [(i32, i32); 8] = [
        (32_768, 0),
        (-32_768, -1),
        (-65_536, -1),
        (-98_304, -2),
        (-131_072, -2),
        (-131_073, -3),
        (i32::MIN, -32_768),
        (i32::MAX, 32_767),
    ];
    let raws = cells.map(|c| c.0);
    let want = cells.map(|c| c.1);
    for level in Resolved::available() {
        assert_eq!(
            level.dispatch_lanes(Cell(raws)),
            want,
            "{:?}",
            level.level()
        );
    }
}

#[test]
fn test_mask_bits_round_trip_at_every_level() {
    struct Trip;
    impl LaneKernel for Trip {
        type Output = bool;
        #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
        #[inline(always)]
        fn run<S: Lanes>(self) -> bool {
            (0..=u8::MAX).all(|b| {
                let m = <S::I32 as I32Lanes>::mask_from_bits(b);
                let arr = std::array::from_fn(|i| (b >> i) & 1 == 1);
                <S::I32 as I32Lanes>::mask_bits(m) == b
                    && <S::I32 as I32Lanes>::mask_bits(<S::I32 as I32Lanes>::mask_from_array(arr))
                        == b
            })
        }
    }
    for level in Resolved::available() {
        assert!(level.dispatch_lanes(Trip), "{:?}", level.level());
    }
}

#[test]
fn test_dispatch_lanes_reaches_forced_levels() {
    for level in [Level::Scalar, Level::Sse42, Level::Avx2, Level::Neon] {
        if let Ok(r) = Resolved::force(level) {
            assert_eq!(r.level(), level);
            check_level(r, &edge_pairs()[..4]);
        }
    }
}

/// On arm64 the differential must include NEON: proof in the gate log that
/// the NEON level ran, not only scalar.
#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_is_among_the_levels_on_aarch64() {
    assert!(Resolved::available().any(|r| r.level() == Level::Neon));
    assert_eq!(Resolved::best().level(), Level::Neon);
}

/// Each level enters its own lane family, not another one (the scalar
/// family gives bit-identical results, so only the type can tell).
#[test]
fn test_dispatch_lanes_enters_the_resolved_family() {
    struct Which;
    impl LaneKernel for Which {
        type Output = Level;
        #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
        #[inline(always)]
        fn run<S: Lanes>(self) -> Level {
            S::LEVEL
        }
    }
    for level in Resolved::available() {
        assert_eq!(level.dispatch_lanes(Which), level.level());
    }
}

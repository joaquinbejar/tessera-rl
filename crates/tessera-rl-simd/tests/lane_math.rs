//! Lane sine and cosine against the scalar functions, on every level this CPU has.

#![expect(clippy::indexing_slicing, reason = "test helpers")]

use tessera_rl_num::fp::with_canonical;
use tessera_rl_num::math::{self, lane};
use tessera_rl_simd::{Kernel, Level, LevelError, Resolved};

struct Eval([f64; 8]);

impl Kernel for Eval {
    type Output = ([f64; 8], [f64; 8], u8);

    #[expect(clippy::inline_always, reason = "the Kernel contract")]
    #[inline(always)]
    fn run<L: lane::F64Lanes>(self) -> Self::Output {
        let x = L::from_array(self.0);
        (
            lane::sin(x).to_array(),
            lane::cos(x).to_array(),
            L::mask_bits(lane::in_polynomial_domain(x)),
        )
    }
}

/// Lanes on the polynomial path, by the scalar's own test.
fn scalar_domain(x: f64) -> bool {
    (x.to_bits() >> 32) as u32 & 0x7fff_ffff <= 0x3fe9_21fb
}

fn check(values: &[f64]) {
    for level in Resolved::available() {
        for chunk in values.chunks(8) {
            let mut v = [0.0; 8];
            v[..chunk.len()].copy_from_slice(chunk);
            let (sin, cos, domain) = with_canonical(|| level.dispatch(Eval(v)));
            for i in 0..8 {
                assert_eq!(
                    domain >> i & 1 == 1,
                    scalar_domain(v[i]),
                    "domain {:x}",
                    v[i].to_bits()
                );
                if scalar_domain(v[i]) {
                    let (s, c) = with_canonical(|| (math::sin(v[i]), math::cos(v[i])));
                    assert_eq!(
                        sin[i].to_bits(),
                        s.to_bits(),
                        "{} sin {:x}",
                        level.level(),
                        v[i].to_bits()
                    );
                    assert_eq!(
                        cos[i].to_bits(),
                        c.to_bits(),
                        "{} cos {:x}",
                        level.level(),
                        v[i].to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn test_lane_trig_exact_boundaries_match_scalar() {
    let words: [u64; 6] = [
        0x3fe9_21fb_ffff_ffff,
        0x3fe9_21fc_0000_0000,
        0x3e4f_ffff_ffff_ffff,
        0x3e50_0000_0000_0000,
        0x3e46_a09d_ffff_ffff,
        0x3e46_a09e_0000_0000,
    ];
    let mut values = vec![
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::NAN,
        f64::INFINITY,
    ];
    for w in words {
        for d in [w.wrapping_sub(1), w, w + 1] {
            values.push(f64::from_bits(d));
            values.push(-f64::from_bits(d));
        }
    }
    check(&values);
}

#[test]
fn test_lane_trig_random_words_match_scalar() {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut values = Vec::new();
    for _ in 0..200_000 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let exponent = 0x3e0 + (state >> 54) % 0x20;
        values.push(f64::from_bits(
            (state >> 11 & 0x000f_ffff_ffff_ffff) | exponent << 52 | state << 63,
        ));
    }
    check(&values);
}

#[test]
fn test_force_refuses_a_level_without_fallback() {
    let absent = if cfg!(target_arch = "aarch64") {
        Level::Avx2
    } else {
        Level::Neon
    };
    assert_eq!(
        Resolved::force(absent),
        Err(LevelError::Unavailable(absent))
    );
}

#[test]
fn test_from_name_refuses_avx512_and_unknown() {
    assert!(matches!(
        Resolved::from_name(Some("avx512")),
        Err(LevelError::Unknown(_))
    ));
    assert!(matches!(
        Resolved::from_name(Some("nope")),
        Err(LevelError::Unknown(_))
    ));
    assert_eq!(
        Resolved::from_name(Some("scalar")).unwrap().level(),
        Level::Scalar
    );
}

#[test]
fn test_best_is_available() {
    assert!(Resolved::available().any(|l| l == Resolved::best()));
}

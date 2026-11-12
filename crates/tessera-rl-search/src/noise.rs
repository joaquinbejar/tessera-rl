// SPDX-License-Identifier: MIT OR Apache-2.0

//! The pinned root-noise sampler (ADR-0043 section 2): Dirichlet(`alpha`)
//! from Gamma(`alpha`, 1) variates by Marsaglia-Tsang, with normal
//! variates by Box-Muller.
//!
//! Pinned details, part of the search golden tests:
//! - a uniform `u` in `(0, 1]` is `1 - unit_f64()`;
//! - Box-Muller draws `u1` in `(0, 1]` then `u2` in `[0, 1)`, returns
//!   `r * cos(2 pi u2)` with `r = sqrt(-2 ln u1)`, and keeps `r * sin(2 pi
//!   u2)` for the next normal draw;
//! - `alpha < 1` draws Gamma(`alpha + 1`) then multiplies by
//!   `exp(ln(u) / alpha)`;
//! - Marsaglia-Tsang tries at most 1,000 candidates per variate and then
//!   returns `d` (the mode); the bound only makes the loop total, it is
//!   not reached in practice.
//!
//! The Gumbel sampler (design note for #69, step 1 and D2): a uniform
//! on the open interval, `u = (2k + 1) / 2^53` with `k = next_u64() >> 12`
//! (52 bits), so exactly that rational, smallest `2^-53`, largest
//! `1 - 2^-53`; then `g = -ln(-ln u)`, always finite. One `g` per action
//! index, in index order, legal or not, from the caller's search stream
//! (`StreamId::SEARCH` keyed by the call counter, never by storage slot).
//!
//! Every transcendental comes from `tessera_rl_num::math` (frozen libm,
//! ADR-0008); `sqrt` is the IEEE square root.

use tessera_rl_num::math;
use tessera_rl_num::rng::StepRng;

const TAU: f64 = core::f64::consts::TAU;
const MAX_TRIES: u32 = 1_000;

/// A normal variate source with the Box-Muller spare.
pub(crate) struct Normal {
    spare: Option<f64>,
}

impl Normal {
    pub(crate) const fn new() -> Self {
        Self { spare: None }
    }

    fn next(&mut self, rng: &mut StepRng) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = 1.0 - rng.unit_f64();
        let u2 = rng.unit_f64();
        let r = (-2.0 * math::ln(u1)).sqrt();
        let t = TAU * u2;
        self.spare = Some(r * math::sin(t));
        r * math::cos(t)
    }
}

/// A Gamma(`alpha`, 1) variate, `alpha > 0`.
#[expect(
    clippy::many_single_char_names,
    reason = "the names of Marsaglia and Tsang's paper"
)]
fn gamma(alpha: f64, rng: &mut StepRng, normal: &mut Normal) -> f64 {
    if alpha < 1.0 {
        let g = gamma(alpha + 1.0, rng, normal);
        let u = 1.0 - rng.unit_f64();
        return g * math::exp(math::ln(u) / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..MAX_TRIES {
        let x = normal.next(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = 1.0 - rng.unit_f64();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * (x2 * x2) {
            return d * v;
        }
        if math::ln(u) < 0.5 * x2 + d * (1.0 - v + math::ln(v)) {
            return d * v;
        }
    }
    d
}

/// Fills `out` with a Dirichlet(`alpha`) sample: one Gamma variate per
/// component, in order, then normalized. A zero sum gives the uniform
/// vector.
pub(crate) fn dirichlet(alpha: f64, rng: &mut StepRng, out: &mut [f64]) {
    let mut normal = Normal::new();
    let mut sum = 0.0;
    for x in out.iter_mut() {
        *x = gamma(alpha, rng, &mut normal);
        sum += *x;
    }
    if sum > 0.0 {
        for x in out.iter_mut() {
            *x /= sum;
        }
    } else if !out.is_empty() {
        let k = f64::from(u32::try_from(out.len()).unwrap_or(u32::MAX));
        out.fill(1.0 / k);
    }
}

/// The open uniform of the 52-bit `k`: exactly `(2k + 1) / 2^53`. Bits
/// above the 52nd are ignored.
#[must_use]
#[inline]
#[cfg_attr(not(test), expect(dead_code, reason = "used by the Gumbel root (G2)"))]
pub(crate) fn open_unit(k: u64) -> f64 {
    /// `2^-53`, the scale of the open uniform.
    const OPEN_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;
    /// The 52 low bits.
    const MASK_52: u64 = (1 << 52) - 1;
    // `2k + 1 < 2^53`: exact in f64, and the power-of-two scale is exact.
    #[expect(clippy::cast_precision_loss, reason = "the value is below 2^53")]
    let n = (((k & MASK_52) << 1) | 1) as f64;
    n * OPEN_SCALE
}

/// The Gumbel variate `-ln(-ln u)` of the 52-bit `k`.
#[must_use]
#[inline]
#[cfg_attr(not(test), expect(dead_code, reason = "used by the Gumbel root (G2)"))]
pub(crate) fn gumbel_of(k: u64) -> f64 {
    -math::ln(-math::ln(open_unit(k)))
}

/// Fills `out` with one Gumbel variate per action index, in index order:
/// one `next_u64` per entry, so the stream position afterwards depends
/// only on `out.len()`.
#[cfg_attr(not(test), expect(dead_code, reason = "used by the Gumbel root (G2)"))]
pub(crate) fn gumbel(rng: &mut StepRng, out: &mut [f64]) {
    for g in out.iter_mut() {
        *g = gumbel_of(rng.next_u64() >> 12);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::RngAuthority;

    fn rng(seed: u64) -> StepRng {
        let id = RngIdentity::new(Seed::new(seed), LogicalId::new(0));
        StepRng::new(&RngAuthority::engine_only(), &id, StreamId::SEARCH)
    }

    #[test]
    fn test_noise_dirichlet_sums_to_one_and_is_reproducible() {
        for alpha in [0.03, 0.3, 1.0, 2.5] {
            let (mut a, mut b) = ([0.0; 7], [0.0; 7]);
            dirichlet(alpha, &mut rng(5), &mut a);
            dirichlet(alpha, &mut rng(5), &mut b);
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
            let sum: f64 = a.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "alpha {alpha}: {sum}");
            assert!(a.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
    }

    #[test]
    fn test_noise_gamma_mean_is_close_to_alpha() {
        for alpha in [0.3, 1.0, 4.0] {
            let mut r = rng(11);
            let mut n = Normal::new();
            let mean = (0..20_000)
                .map(|_| gamma(alpha, &mut r, &mut n))
                .sum::<f64>()
                / 20_000.0;
            assert!(
                (mean - alpha).abs() < 0.05 * alpha.max(1.0),
                "alpha {alpha}: {mean}"
            );
        }
    }

    /// The `k` of the reference table (`tools/gumbel_reference.py`).
    const TABLE_K: [u64; 4] = [0, 1, 1 << 51, (1 << 52) - 1];

    /// The nearest f64 of the 60-digit reference `g` (the script's
    /// `float.hex` column), as bits:
    /// `k = 0`: -3.603778992970457507132029980796388300905714305074112339994452
    /// `k = 1`: -3.573417767444333889382367926359782344322571477281850781026611
    /// `k = 2^51`: 0.3665129205816646473550895397244538125731939500998852714637551
    /// `k = 2^52 - 1`: 36.73680056967710134360215120602552851891333120023787104515067
    const REFERENCE_NEAREST: [u64; 4] = [
        0xc00c_d48a_14a6_8f37, // -0x1.cd48a14a68f37p+1
        0xc00c_965c_0df0_f332, // -0x1.c965c0df0f332p+1
        0x3fd7_74f2_9bdd_6ba4, // 0x1.774f29bdd6ba4p-2
        0x4042_5e4f_7b27_37fa, // 0x1.25e4f7b2737fap+5
    ];

    /// `n / 2^e` as an f64, by integer arithmetic only: exact when `n`
    /// fits in 53 bits, compared by value against `open_unit`.
    fn exact_rational(numerator: u64, exp: i32) -> f64 {
        assert!(numerator < (1 << 53));
        // Build the IEEE bits directly: normalize n to 1.f * 2^p.
        let p = 63 - i32::try_from(numerator.leading_zeros()).unwrap_or(0);
        let frac = (numerator << (52 - p)) & ((1 << 52) - 1);
        let biased = u64::try_from(1023 + p - exp).unwrap_or(0);
        f64::from_bits((biased << 52) | frac)
    }

    #[test]
    fn test_noise_open_unit_known_k_is_exact_rational() {
        let mut ks = TABLE_K.to_vec();
        ks.extend([3, 12_345, 0x000f_0f0f_0f0f_0f0f, 0x0008_0000_0000_0001]);
        for k in ks {
            let u = open_unit(k);
            assert_eq!(
                u.to_bits(),
                exact_rational(2 * k + 1, 53).to_bits(),
                "k {k}"
            );
            assert!(u > 0.0 && u < 1.0);
        }
        // The extremes, as numbers.
        assert_eq!(open_unit(0), exact_rational(1, 53));
        assert_eq!(open_unit((1 << 52) - 1), 1.0 - exact_rational(1, 53));
        assert_eq!(open_unit(1 << 51), 0.5 + exact_rational(1, 53));
        // Bits above the 52nd are ignored.
        assert_eq!(
            open_unit(u64::MAX).to_bits(),
            open_unit((1 << 52) - 1).to_bits()
        );
    }

    #[test]
    fn test_noise_gumbel_variate_within_one_ulp_of_reference() {
        for (k, nearest) in TABLE_K.into_iter().zip(REFERENCE_NEAREST) {
            let g = gumbel_of(k);
            let reference = f64::from_bits(nearest);
            assert_eq!(g.signum(), reference.signum(), "k {k}");
            let ulps = g.to_bits().abs_diff(nearest);
            assert!(ulps <= 1, "k {k}: {g:e} vs {reference:e}, {ulps} ulps");
        }
    }

    #[test]
    fn test_noise_gumbel_variate_bits_pinned() {
        let bits = TABLE_K.map(|k| gumbel_of(k).to_bits());
        assert_eq!(bits, PINNED_BITS);
    }

    /// The bits `math::ln` produces for `TABLE_K`, on every platform. All
    /// equal the nearest f64 except `k = 2^51`, one ulp above it.
    const PINNED_BITS: [u64; 4] = [
        0xc00c_d48a_14a6_8f37,
        0xc00c_965c_0df0_f332,
        0x3fd7_74f2_9bdd_6ba5,
        0x4042_5e4f_7b27_37fa,
    ];

    #[test]
    fn test_noise_gumbel_draws_one_u64_per_index_in_order() {
        let mut out = [0.0; 5];
        gumbel(&mut rng(9), &mut out);
        let mut r = rng(9);
        for g in out {
            assert_eq!(g.to_bits(), gumbel_of(r.next_u64() >> 12).to_bits());
        }
        // The position afterwards depends only on the length.
        let mut a = rng(9);
        gumbel(&mut a, &mut out);
        assert_eq!(a.next_u64(), r.next_u64());
    }

    #[test]
    fn test_noise_gumbel_top1_reproduces_prior_within_tolerance() {
        // Seeds 0 to 99,999 (logical id 0, step 0, StreamId::SEARCH); a
        // fixed 7-action prior, action 1 illegal; tolerance 0.01 absolute
        // per action.
        const PI: [f64; 7] = [0.30, 0.0, 0.25, 0.20, 0.12, 0.08, 0.05];
        const N: u32 = 100_000;
        let logits = PI.map(|p| {
            if p > 0.0 {
                math::ln(p)
            } else {
                f64::NEG_INFINITY
            }
        });
        let mut counts = [0_u32; 7];
        let mut g = [0.0; 7];
        for seed in 0..u64::from(N) {
            gumbel(&mut rng(seed), &mut g);
            let mut best = 0;
            let mut best_score = f64::NEG_INFINITY;
            for (a, (&ga, &la)) in g.iter().zip(&logits).enumerate() {
                let score = ga + la;
                if la > f64::NEG_INFINITY && score > best_score {
                    best = a;
                    best_score = score;
                }
            }
            counts[best] += 1;
        }
        assert_eq!(counts[1], 0);
        for (a, (&c, &p)) in counts.iter().zip(&PI).enumerate() {
            let freq = f64::from(c) / f64::from(N);
            assert!((freq - p).abs() < 0.01, "action {a}: {freq} vs {p}");
        }
    }
}

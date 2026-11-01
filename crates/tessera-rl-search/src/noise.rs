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
}

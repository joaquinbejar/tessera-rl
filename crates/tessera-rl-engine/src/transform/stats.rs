// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running normalization statistics (ADR-0078 item 5): Welford per element
//! within a chunk, Chan's parallel merge across chunks in chunk order.

/// One element's running statistics: count, mean and M2, in `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    /// Values added.
    pub count: f64,
    /// Their mean.
    pub mean: f64,
    /// The sum of squared deviations from the mean.
    pub m2: f64,
}

/// Bytes of one element's accumulator (ADR-0078 item 5.2).
pub(crate) const MOMENT_BYTES: usize = 24;

impl Moments {
    /// Adds one value (Welford).
    #[inline]
    pub(crate) fn add(&mut self, x: f64) {
        self.count += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.count;
        self.m2 += delta * (x - self.mean);
    }

    /// Merges `other` into `self` (Chan et al.); `self` is the earlier
    /// part, so the merge order is fixed by the caller.
    #[inline]
    pub(crate) fn merge(&mut self, other: &Self) {
        if other.count == 0.0 {
            return;
        }
        if self.count == 0.0 {
            *self = *other;
            return;
        }
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean += delta * (other.count / count);
        self.m2 += other.m2 + delta * delta * (self.count * other.count / count);
        self.count = count;
    }

    /// The population variance, `1` before any value (so `Normalize` is
    /// the identity up to the clip before the first barrier, item 5.5).
    #[must_use]
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count == 0.0 {
            1.0
        } else {
            self.m2 / self.count
        }
    }
}

/// The statistics a `Normalize` applies within an epoch: per element, the
/// mean and `1 / sqrt(var + 1e-8)`, fixed at the last barrier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Applied {
    pub(crate) mean: f64,
    pub(crate) inv_std: f64,
}

impl Default for Applied {
    fn default() -> Self {
        Self::from(&Moments::default())
    }
}

impl From<&Moments> for Applied {
    fn from(m: &Moments) -> Self {
        let mean = if m.count == 0.0 { 0.0 } else { m.mean };
        Self {
            mean,
            inv_std: 1.0 / (m.variance() + 1e-8).sqrt(),
        }
    }
}

/// `clamp((x - mean) * inv_std, -clip, clip)` as `f32`; NaN stays NaN.
#[must_use]
#[inline]
pub(crate) fn normalize(x: f64, applied: Applied, clip: f32) -> f32 {
    let clip = f64::from(clip);
    let z = ((x - applied.mean) * applied.inv_std).clamp(-clip, clip);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is clamped to an f32 clip; rounding to f32 is the output"
    )]
    let out = z as f32;
    out
}

#[cfg(test)]
mod tests {
    use super::{Applied, Moments, normalize};

    #[test]
    fn test_moments_welford_matches_two_pass() {
        let xs = [1.0, 4.0, 4.0, 7.0, 10.0];
        let mut m = Moments::default();
        for x in xs {
            m.add(x);
        }
        assert_eq!(m.count, 5.0);
        assert_eq!(m.mean, 5.2);
        let var = xs.iter().map(|x| (x - 5.2) * (x - 5.2)).sum::<f64>() / 5.0;
        assert!((m.variance() - var).abs() < 1e-12);
    }

    #[test]
    fn test_moments_merge_matches_sequential_within_rounding() {
        let (a, b) = ([1.0, 2.0, 3.0], [10.0, 20.0]);
        let mut left = Moments::default();
        for &x in &a {
            left.add(x);
        }
        let mut right = Moments::default();
        for &x in &b {
            right.add(x);
        }
        let mut all = Moments::default();
        a.iter().chain(&b).for_each(|&x| all.add(x));
        left.merge(&right);
        assert_eq!(left.count, all.count);
        assert!((left.mean - all.mean).abs() < 1e-12);
        assert!((left.m2 - all.m2).abs() < 1e-9);
    }

    #[test]
    fn test_moments_merge_empty_sides_are_identity() {
        let mut m = Moments::default();
        m.add(3.0);
        let before = m;
        m.merge(&Moments::default());
        assert_eq!(m, before);
        let mut empty = Moments::default();
        empty.merge(&before);
        assert_eq!(empty, before);
    }

    #[test]
    fn test_normalize_before_first_barrier_is_identity_up_to_clip() {
        let fresh = Applied::default();
        assert_eq!(normalize(0.5, fresh, 5.0), 0.5);
        assert_eq!(normalize(9.0, fresh, 5.0), 5.0);
        assert_eq!(normalize(-9.0, fresh, 5.0), -5.0);
    }

    #[test]
    fn test_normalize_after_statistics_centres_and_scales() {
        let mut m = Moments::default();
        for x in [2.0, 4.0] {
            m.add(x);
        }
        let applied = Applied::from(&m);
        assert_eq!(normalize(3.0, applied, 10.0), 0.0);
        let one = normalize(4.0, applied, 10.0);
        assert!((one - 1.0).abs() < 1e-6, "{one}");
    }
}

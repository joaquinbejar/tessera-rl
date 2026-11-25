// SPDX-License-Identifier: MIT OR Apache-2.0

//! Matchmaking weights and the draw table (ADR-0088 item 5).

use tessera_rl_num::math;

use crate::{LeagueError, PolicyId};

/// The PFSP weighting of a win rate `p` (`AlphaStar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PfspWeighting {
    /// `(1 - p)^2`: focus on the hardest opponents.
    Hard,
    /// `p (1 - p)`: focus on even opponents.
    Variance,
    /// `1 - p`.
    Linear,
}

impl PfspWeighting {
    /// The weight of win rate `p`.
    #[must_use]
    pub fn weight(self, p: f64) -> f64 {
        match self {
            Self::Hard => (1.0 - p) * (1.0 - p),
            Self::Variance => p * (1.0 - p),
            Self::Linear => 1.0 - p,
        }
    }
}

/// How opponents are weighted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Matchmaking {
    /// Every active policy equally.
    Uniform,
    /// Prioritized fictitious self-play over the learner's win rates, with a
    /// (1 win, 1 loss) prior so an unseen opponent has `p = 0.5`.
    Pfsp(PfspWeighting),
    /// `exp(-|R_learner - R_opp| / temperature)` over Elo ratings.
    Elo {
        /// The temperature, in rating points; positive.
        temperature: f64,
    },
}

/// The draw table: one cumulative `u64` threshold per population entry.
///
/// A draw of `u` (uniform over all `u64`) picks the first entry with
/// `u < threshold`; if none, the last entry with a positive weight. Entries
/// whose threshold equals the previous one (zero weight, retired, or a
/// weight below the table's resolution) are never drawn.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Table {
    thresholds: Vec<u64>,
    last: Option<u32>,
    uniform_fallback: bool,
}

/// 2^64 as `f64`.
const TWO_64: f64 = 18_446_744_073_709_551_616.0;

impl Table {
    /// The table of `weights`, one per population entry; `None` entries
    /// (retired) and non-positive or non-finite weights weigh 0. If every
    /// weight is 0, every active entry weighs 1 (uniform fallback).
    ///
    /// The rounding rule: with `S` the sum of the weights and `c_i` the
    /// running sum, the threshold is `floor(c_i / S * 2^64)`, saturated to
    /// `u64::MAX`, and the last positive entry's threshold is `u64::MAX`.
    ///
    /// # Errors
    ///
    /// [`LeagueError::NoActivePolicy`] when no entry is active, and
    /// [`LeagueError::PopulationFull`] past `u32::MAX` entries.
    pub fn from_weights(weights: &[Option<f64>]) -> Result<Self, LeagueError> {
        let clean = |w: Option<f64>| match w {
            Some(w) if w.is_finite() && w > 0.0 => w,
            _ => 0.0,
        };
        let mut total: f64 = weights.iter().map(|&w| clean(w)).sum();
        let uniform_fallback = total <= 0.0;
        let weight = |w: Option<f64>| {
            if uniform_fallback {
                if w.is_some() { 1.0 } else { 0.0 }
            } else {
                clean(w)
            }
        };
        if uniform_fallback {
            total = weights.iter().map(|&w| weight(w)).sum();
        }
        if total <= 0.0 {
            return Err(LeagueError::NoActivePolicy);
        }
        let mut thresholds = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        let mut last = None;
        for (i, &w) in weights.iter().enumerate() {
            let w = weight(w);
            running += w;
            if w > 0.0 {
                last = Some(u32::try_from(i).map_err(|_| LeagueError::PopulationFull)?);
            }
            thresholds.push(to_threshold(running / total));
        }
        if let Some(l) = last
            && let Some(t) = thresholds.get_mut(l as usize)
        {
            *t = u64::MAX;
        }
        Ok(Self {
            thresholds,
            last,
            uniform_fallback,
        })
    }

    /// The entry a uniform `u64` draw `u` selects.
    #[must_use]
    pub fn draw(&self, u: u64) -> Option<PolicyId> {
        let i = self.thresholds.partition_point(|&t| t <= u);
        match u32::try_from(i) {
            Ok(i) if (i as usize) < self.thresholds.len() => Some(PolicyId(i)),
            _ => self.last.map(PolicyId),
        }
    }

    /// The last entry with a positive weight: the draw of any `u` past the
    /// thresholds.
    #[must_use]
    pub const fn last(&self) -> Option<u32> {
        self.last
    }

    /// The thresholds, one per population entry.
    #[must_use]
    pub fn thresholds(&self) -> &[u64] {
        &self.thresholds
    }

    /// This table with entry `reserved` given exactly `floor(share x
    /// 2^64)` of the range (a self-play share, ADR-0088 section 6): every
    /// other entry's width is scaled by `(2^64 - reserved width) / 2^64`,
    /// floored, in integers, and the last positive entry's threshold is
    /// `u64::MAX`. `reserved` must have weighed 0 in this table. The
    /// reserved width is exact; a last entry's `u64::MAX` threshold stands
    /// for 2^64, the end of the draw range. If `reserved` is not the last
    /// positive entry, the last one absorbs the floors' remainder.
    ///
    /// # Errors
    ///
    /// [`LeagueError::UnknownPolicy`] for an entry outside the table.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a share in (0, 1) scaled to 2^64 and floored by `as`"
    )]
    pub fn with_reserved_share(&self, reserved: PolicyId, share: f64) -> Result<Self, LeagueError> {
        let n = self.thresholds.len();
        if reserved.index() >= n {
            return Err(LeagueError::UnknownPolicy(reserved.0));
        }
        let full: u128 = 1 << 64;
        let width = u128::from((share * TWO_64).floor() as u64);
        let rest = full - width;
        // Each entry's width: the reserved one exact, the others this
        // table's widths scaled to `rest`, floored.
        let mut widths = Vec::with_capacity(n);
        let mut previous: u128 = 0;
        for (i, &t) in self.thresholds.iter().enumerate() {
            // This table's width of entry `i` (its last entry spans to 2^64).
            let end = if Some(i) == self.last.map(|l| l as usize) {
                full
            } else {
                u128::from(t)
            };
            let own = end.saturating_sub(previous);
            previous = previous.max(end);
            widths.push(if i == reserved.index() {
                width
            } else {
                own * rest / full
            });
        }
        // The floors' remainder goes to the last positive entry other than
        // the reserved one, so the reserved width stays exact.
        let sum: u128 = widths.iter().sum();
        let absorber = (0..n)
            .rev()
            .find(|&i| i != reserved.index() && widths.get(i).is_some_and(|&w| w > 0));
        if let Some(w) = absorber.and_then(|a| widths.get_mut(a)) {
            *w += full.saturating_sub(sum);
        }
        let mut thresholds = Vec::with_capacity(n);
        let mut running: u128 = 0;
        let mut last = None;
        for (i, &w) in widths.iter().enumerate() {
            running += w;
            if w > 0 {
                last = Some(u32::try_from(i).map_err(|_| LeagueError::PopulationFull)?);
            }
            thresholds.push(u64::try_from(running).unwrap_or(u64::MAX));
        }
        // The last positive entry's threshold is `u64::MAX`, standing for
        // 2^64.
        if let Some(l) = last
            && let Some(t) = thresholds.get_mut(l as usize)
        {
            *t = u64::MAX;
        }
        Ok(Self {
            thresholds,
            last,
            uniform_fallback: self.uniform_fallback,
        })
    }

    /// Whether every weight was zero and the table fell back to uniform.
    #[must_use]
    pub const fn uniform_fallback(&self) -> bool {
        self.uniform_fallback
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a fraction in [0, 1] scaled to 2^64, floored and saturated by `as`"
)]
fn to_threshold(fraction: f64) -> u64 {
    (fraction * TWO_64).floor() as u64
}

impl Matchmaking {
    /// The weight of an active opponent, given the learner's win rate `p`
    /// against it and both Elo ratings.
    #[must_use]
    pub fn weight(self, p: f64, learner_elo: f64, opponent_elo: f64) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Pfsp(w) => w.weight(p),
            Self::Elo { temperature } => {
                math::exp(-(learner_elo - opponent_elo).abs() / temperature)
            }
        }
    }
}

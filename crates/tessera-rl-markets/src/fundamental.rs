// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fundamental value: a bounded random walk on ticks
//! (`docs/reviews/381/DESIGN.md`, section 1).
//!
//! Each step it moves by -1 tick with probability `down_permille / 1000`,
//! by +1 with probability `up_permille / 1000`, and stays otherwise; one
//! uniform draw in `[0, 1000)` from `StepRng` decides. A move that would
//! leave `[min_tick, max_tick]` is reflected: it moves one tick the other
//! way instead. The reflection is a game rule, not a clamp; with
//! `min_tick == max_tick` the value never moves.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_num::rng::StepRng;

use crate::config::LobConfig;

/// The draws of one step are uniform in `[0, PERMILLE)`.
const PERMILLE: NonZeroU32 = match NonZeroU32::new(1000) {
    Some(n) => n,
    None => NonZeroU32::MIN,
};

/// The walk's parameters, taken from a validated [`LobConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RandomWalk {
    min_tick: u32,
    max_tick: u32,
    up_permille: u32,
    down_permille: u32,
}

impl RandomWalk {
    /// The walk of `cfg`. `cfg` must be valid (`min_tick <= max_tick`,
    /// `up_permille + down_permille <= 1000`).
    #[must_use]
    #[inline]
    pub const fn new(cfg: &LobConfig) -> Self {
        Self {
            min_tick: cfg.min_tick,
            max_tick: cfg.max_tick,
            up_permille: cfg.up_permille,
            down_permille: cfg.down_permille,
        }
    }

    /// One step from `price`, with one draw from `rng`.
    #[must_use]
    #[inline]
    pub fn step(&self, price: u32, rng: &mut StepRng) -> u32 {
        self.step_with(price, rng.below_u32(PERMILLE))
    }

    /// One step from `price` for the draw `draw` in `[0, 1000)`: below
    /// `down_permille` moves down, below `down_permille + up_permille`
    /// moves up, otherwise stays. A price outside the bounds is first
    /// brought back to the nearest bound (never in a valid state).
    #[must_use]
    #[inline]
    pub const fn step_with(&self, price: u32, draw: u32) -> u32 {
        let price = if price < self.min_tick {
            self.min_tick
        } else if price > self.max_tick {
            self.max_tick
        } else {
            price
        };
        if self.min_tick == self.max_tick {
            return price;
        }
        // `down_permille + up_permille <= 1000` by validation; never wraps.
        let up_until = self.down_permille + self.up_permille;
        if draw < self.down_permille {
            // Reflected at the lower bound: up instead of down.
            if price > self.min_tick {
                price - 1
            } else {
                price + 1
            }
        } else if draw < up_until {
            // Reflected at the upper bound: down instead of up.
            if price < self.max_tick {
                price + 1
            } else {
                price - 1
            }
        } else {
            price
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RandomWalk;
    use crate::config::LobConfig;

    fn walk(min_tick: u32, max_tick: u32, up: u32, down: u32) -> RandomWalk {
        RandomWalk::new(&LobConfig {
            min_tick,
            max_tick,
            up_permille: up,
            down_permille: down,
            ..LobConfig::default()
        })
    }

    #[test]
    fn test_walk_moves_by_draw_bands() {
        let w = walk(1, 100, 300, 200);
        assert_eq!(w.step_with(50, 0), 49);
        assert_eq!(w.step_with(50, 199), 49);
        assert_eq!(w.step_with(50, 200), 51);
        assert_eq!(w.step_with(50, 499), 51);
        assert_eq!(w.step_with(50, 500), 50);
        assert_eq!(w.step_with(50, 999), 50);
    }

    #[test]
    fn test_walk_reflects_at_bounds() {
        let w = walk(10, 20, 500, 500);
        assert_eq!(w.step_with(10, 0), 11);
        assert_eq!(w.step_with(20, 999), 19);
        assert_eq!(w.step_with(20, 0), 19);
        assert_eq!(walk(7, 7, 500, 500).step_with(7, 0), 7);
    }

    #[test]
    fn test_walk_stays_in_bounds_for_every_draw() {
        let w = walk(3, 6, 450, 450);
        for start in 3..=6 {
            for draw in 0..1000 {
                let next = w.step_with(start, draw);
                assert!((3..=6).contains(&next));
                assert!(next.abs_diff(start) <= 1);
            }
        }
    }
}

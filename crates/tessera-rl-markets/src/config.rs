// SPDX-License-Identifier: MIT OR Apache-2.0

//! `LobConfig` and its checked validation (`docs/reviews/381/DESIGN.md`,
//! sections 1 and 2).
//!
//! The field ranges bound every value; `check` adds the cross-field
//! rules and the overflow proofs of section 2 as `checked_*` arithmetic, so
//! that the kernel can use plain integer arithmetic afterwards:
//!
//! - `steps x MAX_AGENTS < 2^32`: the order id and arrival counters never
//!   wrap (one order or queued item per agent and step);
//! - `steps x agents x MAX_OPEN x MAX_LOT x max_tick + initial_cash < 2^62`:
//!   an upper bound on the cash any agent can reach, so `i64` cash never
//!   overflows (fills only move cash, so the sum over agents is constant);
//! - `max_position x max_tick` fits `i32`.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::ConfigError;
use tessera_rl_macros::EnvConfig;

use crate::limits::{MAX_AGENTS, MAX_LATENCY, MAX_LOT, MAX_OPEN, MAX_PRICE, MAX_STEPS};

// The literal ranges below repeat these bounds (the derive takes literals).
const _: () = assert!(
    MAX_AGENTS == 16
        && MAX_STEPS == 4096
        && MAX_LATENCY == 3
        && MAX_PRICE == 65_536
        && MAX_LOT == 1024
);

/// The size bucket fields, in order.
const SIZE_FIELDS: [&str; 4] = ["size_0", "size_1", "size_2", "size_3"];

/// The cash bound of section 2: `2^62`.
const CASH_BOUND: u64 = 1 << 62;

/// `tessera/LimitOrderBook-v1` configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
#[config(validate = check)]
pub struct LobConfig {
    /// Trading agents, slots `0..agents`.
    #[config(default = 8, range = 2..=16)]
    pub agents: u32,
    /// Steps per episode; the episode is truncated after the last, where
    /// every inventory is settled at the fundamental.
    #[config(default = 1000, range = 1..=4096)]
    pub steps: u32,
    /// Price levels per side in the observation (`K`).
    #[config(default = 10, range = 1..=16)]
    pub depth: u32,
    /// The fundamental's lower bound, in ticks.
    #[config(default = 1, range = 1..=65_536)]
    pub min_tick: u32,
    /// The fundamental's upper bound and the largest order price, in
    /// ticks. Order prices are in `[1, max_tick]`.
    #[config(default = 1024, range = 2..=65_536)]
    pub max_tick: u32,
    /// The fundamental at reset, in `[min_tick, max_tick]`.
    #[config(default = 512, range = 1..=65_536)]
    pub initial_price: u32,
    /// Per mille probability that the fundamental moves up one tick.
    #[config(default = 250, range = 0..=1000)]
    pub up_permille: u32,
    /// Per mille probability that it moves down one tick.
    #[config(default = 250, range = 0..=1000)]
    pub down_permille: u32,
    /// Steps between a submission and its arrival at the book.
    #[config(default = 0, range = 0..=3)]
    pub latency: u32,
    /// The lots of size bucket 0.
    #[config(default = 1, range = 1..=1024)]
    pub size_0: u32,
    /// The lots of size bucket 1.
    #[config(default = 2, range = 1..=1024)]
    pub size_1: u32,
    /// The lots of size bucket 2.
    #[config(default = 5, range = 1..=1024)]
    pub size_2: u32,
    /// The lots of size bucket 3.
    #[config(default = 10, range = 1..=1024)]
    pub size_3: u32,
    /// The largest absolute inventory, in lots.
    #[config(default = 50, range = 1..=1_000_000)]
    pub max_position: u32,
    /// Cash at reset, in tick-lots.
    #[config(default = 1_000_000, range = 0..=1_099_511_627_776)]
    pub initial_cash: i64,
    /// Whether buys may take cash below zero.
    #[config(default = false)]
    pub allow_short_cash: bool,
    /// An agent whose equity falls below `-bankrupt` tick-lots is
    /// terminated; 0 disables bankruptcy.
    #[config(default = 0, range = 0..=4_611_686_018_427_387_904)]
    pub bankrupt: u64,
    /// Half-width of the uniform noise on the private fundamental signal,
    /// in ticks.
    #[config(default = 2, range = 0..=64)]
    pub noise: u32,
    /// Rewards are divided by `2^reward_shift` (exact in `f32`).
    #[config(default = 0, range = 0..=32)]
    pub reward_shift: u32,
}

impl LobConfig {
    /// The lots of the four size buckets, smallest first.
    #[must_use]
    #[inline]
    pub const fn sizes(&self) -> [u32; 4] {
        [self.size_0, self.size_1, self.size_2, self.size_3]
    }

    /// The agent count as an index bound.
    #[must_use]
    #[inline]
    pub fn agent_count(&self) -> usize {
        usize::try_from(self.agents).map_or(MAX_AGENTS, |n| n.min(MAX_AGENTS))
    }
}

const fn error(field: &'static str, reason: &'static str) -> ConfigError {
    ConfigError { field, reason }
}

/// The cross-field rules and the overflow proofs; run after the range
/// checks.
fn check(cfg: &LobConfig) -> Result<(), ConfigError> {
    if cfg.min_tick > cfg.max_tick {
        return Err(error("min_tick", "must not exceed max_tick"));
    }
    if cfg.initial_price < cfg.min_tick || cfg.initial_price > cfg.max_tick {
        return Err(error("initial_price", "must be in [min_tick, max_tick]"));
    }
    if cfg
        .up_permille
        .checked_add(cfg.down_permille)
        .is_none_or(|p| p > 1000)
    {
        return Err(error(
            "down_permille",
            "up_permille + down_permille must not exceed 1000",
        ));
    }
    // The ranges already keep every bucket in `[1, MAX_LOT]`.
    let sizes = cfg.sizes();
    if let Some(i) = sizes.windows(2).position(|w| matches!(w, [a, b] if a >= b)) {
        let field = SIZE_FIELDS.get(i + 1).copied().unwrap_or("size_3");
        return Err(error(field, "size buckets must be strictly increasing"));
    }
    // Order ids and arrival ranks: at most one per agent and step.
    let agents_max = u32::try_from(MAX_AGENTS).ok();
    if agents_max.and_then(|a| cfg.steps.checked_mul(a)).is_none() {
        return Err(error("steps", "steps x MAX_AGENTS must fit u32"));
    }
    // The cash bound.
    let cash = u64::from(cfg.steps)
        .checked_mul(u64::from(cfg.agents))
        .and_then(|v| v.checked_mul(u64::try_from(MAX_OPEN).ok()?))
        .and_then(|v| v.checked_mul(u64::from(MAX_LOT)))
        .and_then(|v| v.checked_mul(u64::from(cfg.max_tick)))
        .and_then(|v| v.checked_add(u64::try_from(cfg.initial_cash).ok()?));
    if cash.is_none_or(|c| c >= CASH_BOUND) {
        return Err(error(
            "initial_cash",
            "steps x agents x MAX_OPEN x MAX_LOT x max_tick + initial_cash must be below 2^62",
        ));
    }
    let position = i32::try_from(cfg.max_position)
        .ok()
        .zip(i32::try_from(cfg.max_tick).ok())
        .and_then(|(p, t)| p.checked_mul(t));
    if position.is_none() {
        return Err(error(
            "max_position",
            "max_position x max_tick must fit i32",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::LobConfig;
    use tessera_rl_core::EnvConfig;

    /// The field a default config edited by `edit` is refused for.
    fn refused(edit: impl FnOnce(&mut LobConfig)) -> Option<&'static str> {
        let mut cfg = LobConfig::default();
        edit(&mut cfg);
        cfg.validate().err().map(|e| e.field)
    }

    #[test]
    fn test_config_default_is_valid() {
        assert_eq!(LobConfig::default().validate(), Ok(()));
        assert_eq!(LobConfig::default().sizes(), [1, 2, 5, 10]);
        assert_eq!(LobConfig::default().agent_count(), 8);
    }

    #[test]
    fn test_config_ranges_reject_out_of_bounds() {
        assert_eq!(refused(|c| c.agents = 1), Some("agents"));
        assert_eq!(refused(|c| c.agents = 17), Some("agents"));
        assert_eq!(refused(|c| c.steps = 4097), Some("steps"));
        assert_eq!(refused(|c| c.latency = 4), Some("latency"));
        assert_eq!(refused(|c| c.depth = 0), Some("depth"));
        assert_eq!(refused(|c| c.size_0 = 0), Some("size_0"));
        assert_eq!(refused(|c| c.max_tick = 65_537), Some("max_tick"));
    }

    #[test]
    fn test_config_cross_field_rules_reject() {
        assert_eq!(refused(|c| c.min_tick = 600), Some("initial_price"));
        let inverted = |c: &mut LobConfig| {
            c.min_tick = 100;
            c.max_tick = 50;
            c.initial_price = 60;
        };
        assert_eq!(refused(inverted), Some("min_tick"));
        let drift = |c: &mut LobConfig| {
            c.up_permille = 600;
            c.down_permille = 401;
        };
        assert_eq!(refused(drift), Some("down_permille"));
        assert_eq!(refused(|c| c.size_2 = 2), Some("size_2"));
    }

    #[test]
    fn test_config_overflow_proofs_reject_extremes() {
        // 32,768 x 65,536 = 2^31 does not fit i32.
        let wide = |position| {
            move |c: &mut LobConfig| {
                c.max_position = position;
                c.max_tick = 65_536;
            }
        };
        assert_eq!(refused(wide(32_768)), Some("max_position"));
        assert_eq!(refused(wide(32_767)), None);
        // The largest every other bound allows still keeps cash below 2^62.
        let big = |c: &mut LobConfig| {
            c.agents = 16;
            c.steps = 4096;
            c.max_tick = 65_536;
            c.initial_cash = 1_099_511_627_776;
        };
        assert_eq!(refused(big), None);
    }
}

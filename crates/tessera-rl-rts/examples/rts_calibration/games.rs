// SPDX-License-Identifier: MIT OR Apache-2.0

//! The calibration among scripted bots of RTS stage 5a
//! (`docs/reviews/rts-stage5-plan.md` r2, section 2): games, their outcomes
//! and a pairing's summary. Shared by the `rts_calibration` example and its
//! test. Scripted bots only: no learning result enters here.

use core::num::NonZeroU32;

use tessera_rl_core::AgentSlot;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig};

/// How a game ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The bot in seat Zero (agent 0) destroyed the other side's Cores.
    ZeroWins,
    /// The bot in seat One won.
    OneWins,
    /// Both sides lost their last Core in the same tick.
    Draw,
    /// The step limit came first.
    Truncated,
}

/// One game's outcome and its length in decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Game {
    pub outcome: Outcome,
    pub decisions: u32,
}

/// The bots' generators of game `seed`: one per seat, from the seed alone,
/// so a game is a function of (config, bots, seed).
fn rng(seed: u64, seat: u32) -> StepRng {
    let identity = RngIdentity::new(Seed::new(seed), LogicalId::new(seat));
    StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS)
}

/// Plays one game from reset with seed `seed`: `bots[0]` in seat Zero
/// (agent 0), `bots[1]` in seat One, until the episode ends.
///
/// # Panics
///
/// On an engine error: the tool's configs are valid by construction.
pub fn play(cfg: &RtsConfig, bots: [Bot; 2], seed: u64) -> Game {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        illegal_action: IllegalActionPolicy::Noop,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut pool: EnvPool<Rts, Masked> =
        EnvPool::with_masks(cfg.clone(), options).expect("a valid calibration config");
    pool.reset_all(Reseed::Keep).expect("reset");
    let per = usize::from(cfg.max_units) * HEADS_PER_UNIT;
    let mut actions = vec![0i32; 2 * per];
    let mut rngs = [rng(seed, 0), rng(seed, 1)];
    let mut decisions = 0u32;
    loop {
        let state = *pool.state(0).expect("instance 0");
        for (seat, bot) in bots.iter().enumerate() {
            let rows = &mut actions[seat * per..(seat + 1) * per];
            rows.fill(0);
            let mut w = ActionWriter::new(rows, &mut []);
            let slot = AgentSlot::new(u16::try_from(seat).expect("two seats"), 2).expect("slot");
            bot.act(cfg, &state, slot, &mut rngs[seat], &mut w);
        }
        let out = pool
            .step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .expect("legal scripted orders");
        decisions += 1;
        if out.episode_truncated()[0].get() {
            return Game {
                outcome: Outcome::Truncated,
                decisions,
            };
        }
        if out.episode_terminated()[0].get() {
            let r = out.rewards();
            let outcome = match (r[0] > 0.0, r[1] > 0.0) {
                (true, false) => Outcome::ZeroWins,
                (false, true) => Outcome::OneWins,
                _ => Outcome::Draw,
            };
            return Game { outcome, decisions };
        }
    }
}

/// A pairing's results over a seed list, both seats: game k is seed k with
/// bot A in seat Zero, and seed k with bot A in seat One.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    /// Per seat of bot A (0: seat Zero, 1: seat One): A's wins, B's wins,
    /// draws, truncations.
    pub by_seat: [[u32; 4]; 2],
    /// Every game's length, in play order.
    pub lengths: Vec<u32>,
    /// Lengths of the decided games (a win or a draw), in play order.
    pub decided_lengths: Vec<u32>,
}

/// Plays the pairing `a` against `b` over `seeds`, both seats.
pub fn pairing(cfg: &RtsConfig, first: Bot, second: Bot, seeds: core::ops::Range<u64>) -> Summary {
    let mut sum = Summary::default();
    for seed in seeds {
        for seat in 0..2 {
            let bots = if seat == 0 {
                [first, second]
            } else {
                [second, first]
            };
            let g = play(cfg, bots, seed);
            let a_won = matches!(
                (seat, g.outcome),
                (0, Outcome::ZeroWins) | (1, Outcome::OneWins)
            );
            let b_won = matches!(
                (seat, g.outcome),
                (0, Outcome::OneWins) | (1, Outcome::ZeroWins)
            );
            let k = if a_won {
                0
            } else if b_won {
                1
            } else if g.outcome == Outcome::Draw {
                2
            } else {
                3
            };
            sum.by_seat[seat][k] += 1;
            sum.lengths.push(g.decisions);
            if g.outcome != Outcome::Truncated {
                sum.decided_lengths.push(g.decisions);
            }
        }
    }
    sum
}

/// The Wilson 95% interval of `k` successes in `n` trials, as (low, high).
#[must_use]
pub fn wilson(k: u32, n: u32) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959_963_984_540_054_f64;
    let (k, n) = (f64::from(k), f64::from(n));
    let p = k / n;
    let denom = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denom;
    let half = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

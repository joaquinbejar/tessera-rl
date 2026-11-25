// SPDX-License-Identifier: MIT OR Apache-2.0

//! The market environments of phase 4 (SPEC 16): `tessera/LimitOrderBook-v1`,
//! a continuous double auction with price-time priority and up to 16
//! trading agents (`docs/reviews/381/DESIGN.md`).
//!
//! This crate is built in stages (design section 9). Stage 1: the
//! configuration ([`config`]), the order record ([`order`]), the book and
//! its matching ([`book`]), fills and accounts ([`fill`], [`account`]),
//! the worst-case commitment the masks check ([`exposure`]), the latency
//! queue ([`pending`]) and the fundamental's random walk ([`fundamental`]).
//! Stage 2: the state ([`state`]), actions and masks ([`actions`]), the
//! observation ([`observe`]) and the environment ([`env`](mod@env)).
//! Stage 3: the information boundary and `Determinize` ([`info`]), the
//! scripted baselines ([`heuristics`]) and the privileged state
//! ([`privileged`]).
//!
//! Everything here is integer arithmetic: no float enters a transition
//! (ADR-0008), which `clippy::float_arithmetic` enforces per module; the
//! one exception is the final `f32` reward conversion ([`reward`]).
//!
//! # Searching the order book on its profit and loss
//!
//! Each step rewards a trader with its mid-marked profit and loss. The
//! default `ValueMode::Terminal` of `tessera-rl-search` (ADR-0043) uses
//! rewards only at terminal nodes and only in `[-1, 1]`, so it sees none of
//! that signal. `ValueMode::Returns` (ADR-0096) backs up
//! `r + discount * V` on every edge: the searcher's step reward plus the
//! value below, the evaluator's at a leaf and 0 at the episode's end.
//! Selection uses each call's min-max of the edge means, so the reward
//! scale does not matter, and the evaluator's values are in reward units.
//!
//! `Returns` needs one mover on every path. The order book's traders act
//! at once, so it is searched through an ADR-0093 searcher pool, every
//! other trader playing its baseline from its information state:
//!
//! ```
//! use core::num::NonZeroU32;
//!
//! use tessera_rl_core::AgentSlot;
//! use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Record, Reseed};
//! use tessera_rl_markets::limits::MAX_AGENTS;
//! use tessera_rl_markets::{LimitOrderBook, LobConfig};
//! use tessera_rl_num::identity::Seed;
//! use tessera_rl_search::{
//!     Determinization, Evaluator, EvaluatorError, Mcts, PuctConfig, SearchOptions, SearchPool,
//!     ValueMode,
//! };
//!
//! /// Uniform priors over the 1,728 joint actions and a value of 0.
//! struct Uniform;
//!
//! impl Evaluator<LimitOrderBook> for Uniform {
//!     fn evaluate(
//!         &mut self,
//!         _: &[Record<LimitOrderBook>],
//!         priors: &mut [f64],
//!         values: &mut [f64],
//!     ) -> Result<(), EvaluatorError> {
//!         priors.fill(1.0 / 1728.0);
//!         values.fill(0.0);
//!         Ok(())
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = LobConfig {
//!     agents: 4,
//!     ..LobConfig::default()
//! };
//! let mut live = EnvPool::<LimitOrderBook, Masked>::with_masks(
//!     cfg.clone(),
//!     PoolOptions::new(NonZeroU32::MIN, Seed::new(11)),
//! )?;
//! live.reset_all(Reseed::Keep)?;
//! let root = live.snapshot(0)?;
//! // Trader 1 searches; traders 0, 2 and 3 play their baselines.
//! let how = Determinization {
//!     agent: AgentSlot::new(1, MAX_AGENTS).ok_or("slot")?,
//!     sampler_seed: Seed::new(3),
//!     sample_index: 0,
//! };
//! let opponents = |cfg, options| {
//!     EnvPool::<LimitOrderBook, Masked>::with_masks(cfg, options)?
//!         .with_info_heuristic_slots(&[0, 2, 3])
//! };
//! let options = SearchOptions::new(
//!     NonZeroU32::new(64).ok_or("capacity")?,
//!     NonZeroU32::new(4).ok_or("batch")?,
//! );
//! let pool = SearchPool::new_determinized(cfg, options, &root, how, opponents)?;
//! let mut puct = PuctConfig::default();
//! puct.simulations = 32;
//! puct.batch_size = 4;
//! puct.value = ValueMode::Returns { discount: 0.99 };
//! let mut search = Mcts::new(pool, puct)?;
//! search.run(&mut Uniform)?;
//! let mut visits = vec![0u32; 1728];
//! search.root_visits(&mut visits);
//! assert_eq!(visits.iter().sum::<u32>(), 32);
//! let _action = search.select_action(0.0)?;
//! # Ok(())
//! # }
//! ```
//!
//! A discount that is not finite or not in `[0, 1]` is
//! `SearchError::Options`, a pool whose paths change mover is
//! `SearchError::ReturnsNeedOneMover`, and a return that overflows is
//! `SearchError::NonFiniteReturn`, refused before any edge changes.

pub mod account;
pub mod actions;
pub mod book;
pub mod config;
pub mod env;
pub mod error;
pub mod exposure;
pub mod fill;
pub mod fundamental;
pub mod heuristics;
pub mod info;
pub mod limits;
pub mod observe;
pub mod order;
pub mod pending;
pub mod privileged;
pub mod reward;
pub mod state;

pub use account::{Account, apply_fill};
pub use book::{Book, Level, Outcome, Sweep};
pub use config::LobConfig;
pub use env::LimitOrderBook;
pub use error::{BookError, PendingError};
pub use exposure::Exposure;
pub use fill::{Fill, Fills};
pub use fundamental::RandomWalk;
pub use heuristics::{Baseline, act_info, baseline_of};
pub use info::{LobInfo, SampleReport, info_of, public_state, sample_with_report};
pub use observe::LobObs;
pub use order::{Kind, Order, Side};
pub use pending::{Pending, PendingQueue};
pub use privileged::LobPrivileged;
pub use state::{LobState, Touch, Trade};

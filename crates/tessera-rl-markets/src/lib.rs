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

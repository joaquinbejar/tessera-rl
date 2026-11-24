// SPDX-License-Identifier: MIT OR Apache-2.0

//! A reference PPO (clipped) on `tessera/CartPole-v1` with Burn (SPEC 14.2,
//! #367). A sanity example, not a trainer (SPEC 1.3).
//!
//! ```text
//! ppo-cartpole [--seed N] [--steps S] [--smoke] [--gate]
//! ```
//!
//! `--smoke` runs two rollouts and checks the losses are finite. `--gate` is
//! the example sanity gate: exit 0 when the mean return of the last 20
//! episodes reaches 195 within `--steps` (default 100 000).

mod model;
mod ppo;
mod rng;

use std::process::ExitCode;

/// The pinned Burn version (`integrations/burn/Cargo.toml`), printed by
/// every run.
pub const BURN_VERSION: &str = "0.22.1";

fn arg(args: &[String], name: &str) -> Option<u64> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let gate = args.iter().any(|a| a == "--gate");
    let config = ppo::Config {
        seed: arg(&args, "--seed").unwrap_or(1),
        total_steps: if smoke { 2 * ppo::ROLLOUT * ppo::NUM_ENVS } else { arg(&args, "--steps").unwrap_or(100_000) },
    };
    println!("ppo-cartpole: burn {BURN_VERSION}, backend flex (pure-Rust CPU) + autodiff, seed {}", config.seed);
    let report = match ppo::train(&config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ppo-cartpole: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "ppo-cartpole: steps {} episodes {} last-20 mean return {:.1} finite losses {}",
        report.steps, report.episodes, report.last_mean, report.finite
    );
    let ok = if smoke { report.finite } else if gate { report.finite && report.last_mean >= 195.0 } else { report.finite };
    if ok { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

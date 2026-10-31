// SPDX-License-Identifier: MIT OR Apache-2.0

use tessera_rl_core::{EnvConfig, ObsLeaf, Observation};
use tessera_rl_macros::{EnvConfig, Observation};

#[derive(EnvConfig)]
pub struct GridConfig {
    #[config(default = 16, range = 4..=64, shape)]
    size: u16,
    #[config(default = 200, range = 1..=10_000)]
    max_steps: u16,
    #[config(default = 0.5, adaptive)]
    noise: f32,
}

#[derive(Observation)]
pub struct GridObs {
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &GridConfig| [3, c.size, c.size]))]
    pixels: ObsLeaf<u8>,
}

fn main() {
    let cfg = GridConfig::default();
    assert!(cfg.validate().is_ok());
    assert!(GridObs::space(&cfg).is_ok());
    let _ = (cfg.max_steps, cfg.noise);
}

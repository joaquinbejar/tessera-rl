// SPDX-License-Identifier: MIT OR Apache-2.0

// A lane-wise state (ADR-0090): every allowed field type, one column each, and
// no padding (a lane state cannot use Reserved<N>),
// with `Fields` in declaration order.
use tessera_rl_core::{Bool8, EnvState, StateEnum};
use tessera_rl_macros::{EnvState, StateEnum};
use tessera_rl_num::Fx32;

#[derive(StateEnum, Clone, Copy, Default, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    #[default]
    Idle = 0,
    Run = 1,
}

#[derive(EnvState, Clone, Copy, Default)]
#[repr(C)]
#[state(lanes)]
pub struct Lanes {
    a: f64,
    b: f32,
    c: i32,
    d: u32,
    e: Fx32,
    f: Bool8,
    g: Phase,
    h: Bool8,
    i: Bool8,
    j: u32,
}

fn main() {
    let _ = <Phase as StateEnum>::NAME;
    assert!(<Lanes as EnvState>::LANES);
}

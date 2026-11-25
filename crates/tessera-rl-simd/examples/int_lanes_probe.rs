// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fixed `Fx32` multiply kernel for `scripts/check_simd_disasm.py`: its
//! `run_*_lanes` entry points must hold the widening multiply (`pmuldq`,
//! `vpmuldq` or `smull`) and no call to a lane operation.

use std::hint::black_box;

use tessera_rl_num::math::lane_int::{Fx32Lanes, I32Lanes};
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

struct FxMulProbe([i32; 8], [i32; 8]);

impl LaneKernel for FxMulProbe {
    type Output = [i32; 8];

    #[expect(clippy::inline_always, reason = "the LaneKernel contract")]
    #[inline(always)]
    fn run<S: Lanes>(self) -> [i32; 8] {
        let a = S::Fx32::from_raw(S::I32::from_array(self.0));
        let b = S::Fx32::from_raw(S::I32::from_array(self.1));
        a.saturating_mul(b).to_raw().to_array()
    }
}

fn main() {
    for level in Resolved::available() {
        black_box(level.dispatch_lanes(FxMulProbe(black_box([3; 8]), black_box([5; 8]))));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lane sine and cosine against the math version 1 reference table
//! (`tessera_rl_num::math::vectors`, libm 0.2.16), bit for bit, on every
//! SIMD level this CPU has, for the table's inputs on the lane polynomial
//! path (outside it a lane kernel takes the scalar transition). The
//! cross-ISA check is CI running this on x86-64 (scalar, SSE4.2, AVX2) and
//! aarch64 (scalar, NEON).

use tessera_rl_num::fp::with_canonical;
use tessera_rl_num::math::{self, lane, vectors};
use tessera_rl_simd::{Kernel, Resolved};

struct Eval([f64; 8]);

impl Kernel for Eval {
    type Output = ([f64; 8], [f64; 8], u8);

    #[expect(clippy::inline_always, reason = "the Kernel contract")]
    #[inline(always)]
    fn run<L: lane::F64Lanes>(self) -> Self::Output {
        let x = L::from_array(self.0);
        (
            lane::sin(x).to_array(),
            lane::cos(x).to_array(),
            L::mask_bits(lane::in_polynomial_domain(x)),
        )
    }
}

#[test]
fn test_lane_math_v1_table_matches_every_level() {
    assert_eq!(math::MATH_VERSION, 1);
    let inputs = vectors::INPUTS;
    let mut checked = 0usize;
    for level in Resolved::available() {
        for (start, chunk) in inputs.chunks(8).enumerate() {
            let mut v = [0.0; 8];
            for (slot, &bits) in v.iter_mut().zip(chunk) {
                *slot = f64::from_bits(bits);
            }
            let (sin, cos, domain) = with_canonical(|| level.dispatch(Eval(v)));
            for i in 0..chunk.len() {
                if domain >> i & 1 == 0 {
                    continue;
                }
                let k = start * 8 + i;
                assert_eq!(
                    sin[i].to_bits(),
                    vectors::SIN[k],
                    "{} sin {:x}",
                    level.level(),
                    inputs[k]
                );
                assert_eq!(
                    cos[i].to_bits(),
                    vectors::COS[k],
                    "{} cos {:x}",
                    level.level(),
                    inputs[k]
                );
                checked += 1;
            }
        }
    }
    // The table has inputs on the polynomial path at every level.
    assert!(
        checked >= Resolved::available().count() * 8,
        "checked {checked}"
    );
}

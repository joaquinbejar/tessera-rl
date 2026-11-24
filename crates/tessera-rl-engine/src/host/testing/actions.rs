// SPDX-License-Identifier: MIT OR Apache-2.0

//! Seeded legal actions for the host environment checks (#351).

use super::super::HostLayout;
use crate::outputs::Outputs;
use crate::validate::ActionInput;

/// Draws one joint action per call from a splitmix64 stream: a legal
/// choice of each discrete head whose mask block has no parent (any choice
/// in range otherwise, or without masks), and a point of each continuous
/// head's range.
#[derive(Debug, Clone)]
pub(crate) struct Sampler {
    state: u64,
    discrete: Vec<i32>,
    continuous: Vec<f32>,
}

impl Sampler {
    /// A sampler seeded `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            discrete: Vec::new(),
            continuous: Vec::new(),
        }
    }

    /// splitmix64: arithmetic modulo 2^64 by definition.
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; 0 for `n == 0`.
    fn below(&mut self, n: usize) -> usize {
        let n64 = u64::try_from(n).unwrap_or(u64::MAX);
        self.next()
            .checked_rem(n64)
            .map_or(0, |v| usize::try_from(v).unwrap_or(0))
    }

    /// The next joint action, `[N, A, heads]`, from the masks of `outputs`.
    pub(crate) fn fill(&mut self, layout: &HostLayout, outputs: &Outputs) -> ActionInput<'_> {
        let rows = layout.num_envs.saturating_mul(layout.agents);
        let heads = &layout.discrete_heads;
        let masks = outputs.masks();
        self.discrete.clear();
        self.continuous.clear();
        for row in 0..rows {
            let mask_row = row
                .checked_mul(layout.mask_len)
                .and_then(|start| masks.get(start..start.checked_add(layout.mask_len)?));
            for (h, &n) in heads.iter().enumerate() {
                let n = usize::try_from(n).unwrap_or(0);
                let block = layout
                    .mask_heads
                    .get(h)
                    .filter(|shape| shape.parent_sizes == [None, None])
                    .and_then(|shape| {
                        mask_row?.get(shape.offset..shape.offset.checked_add(shape.size)?)
                    });
                let choice = match block {
                    Some(block) => {
                        let legal = block.iter().filter(|b| b.get()).count();
                        if legal == 0 {
                            0
                        } else {
                            let k = self.below(legal);
                            block
                                .iter()
                                .enumerate()
                                .filter(|(_, b)| b.get())
                                .nth(k)
                                .map_or(0, |(i, _)| i)
                        }
                    }
                    None => self.below(n),
                };
                self.discrete.push(i32::try_from(choice).unwrap_or(0));
            }
            for &(low, high) in &layout.continuous_heads {
                // An unbounded side is one unit from the other side, or -1..0.
                let low = if low.is_finite() {
                    low
                } else if high.is_finite() {
                    high - 1.0
                } else {
                    -1.0
                };
                let high = if high.is_finite() { high } else { low + 1.0 };
                // A quarter point of the range: exact in f32 for small ranges.
                let k = u32::try_from(self.below(5)).unwrap_or(0);
                let v = low + (high - low) * f64::from(k) / 4.0;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a point of an f32 head's range"
                )]
                self.continuous.push(v as f32);
            }
        }
        ActionInput {
            discrete: &self.discrete,
            continuous: &self.continuous,
        }
    }
}

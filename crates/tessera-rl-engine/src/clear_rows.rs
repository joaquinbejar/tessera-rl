// SPDX-License-Identifier: MIT OR Apache-2.0

//! Clearing a one-agent instance's per-agent step rows (#757, #751): its
//! agent terminated, truncated, died, spawned and invalid-event flags and its
//! reward, by direct stores instead of a `fill` call on a one-element slice
//! per buffer (which compiled to a `memset` call each). The bytes written
//! are the same. Several agents keep the span fills in `clear_instance`.

use tessera_rl_core::Bool8;

/// Sets element `i` of each of `flags` to `Bool8::FALSE` and of `rewards`
/// to `0.0`; an index past a buffer's end writes nothing there.
#[inline]
pub(crate) fn clear_one_agent_rows(flags: [&mut [Bool8]; 5], rewards: &mut [f32], i: usize) {
    for f in flags.into_iter().filter_map(|b| b.get_mut(i)) {
        *f = Bool8::FALSE;
    }
    if let Some(r) = rewards.get_mut(i) {
        *r = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Bool8;

    use super::clear_one_agent_rows;

    const T: Bool8 = Bool8::TRUE;
    const F: Bool8 = Bool8::FALSE;

    fn run(len: usize, i: usize) -> ([Vec<Bool8>; 5], Vec<f32>) {
        let mut flags: [Vec<Bool8>; 5] = core::array::from_fn(|_| vec![T; len]);
        let mut rewards = vec![7.0f32; len];
        let [terminated, truncated, died, spawned, invalid] = &mut flags;
        clear_one_agent_rows(
            [
                terminated.as_mut_slice(),
                truncated.as_mut_slice(),
                died.as_mut_slice(),
                spawned.as_mut_slice(),
                invalid.as_mut_slice(),
            ],
            &mut rewards,
            i,
        );
        (flags, rewards)
    }

    #[test]
    fn test_clear_one_agent_rows_clears_only_its_row_in_every_buffer() {
        let (flags, rewards) = run(4, 2);
        for (k, buf) in flags.iter().enumerate() {
            assert_eq!(buf, &[T, T, F, T], "flag buffer {k}");
        }
        assert_eq!(rewards, [7.0, 7.0, 0.0, 7.0]);
    }

    #[test]
    fn test_clear_one_agent_rows_past_the_end_writes_nothing() {
        let (flags, rewards) = run(3, 3);
        for buf in &flags {
            assert_eq!(buf, &[T, T, T]);
        }
        assert_eq!(rewards, [7.0, 7.0, 7.0]);
    }
}

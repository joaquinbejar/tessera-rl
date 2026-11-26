// SPDX-License-Identifier: MIT OR Apache-2.0

//! Clearing one instance's per-agent step rows (#757, #751): its agent
//! terminated, truncated, died, spawned and invalid-event flags and its
//! rewards. A one-agent instance (most environments) stores each element
//! directly instead of calling `fill` on a one-element slice per buffer,
//! which compiled to a `memset` call each; the bytes written are the same.

use tessera_rl_core::Bool8;

/// Sets instance `i`'s rows (`agents` per instance) to `Bool8::FALSE` in each
/// of `flags` and to `0.0` in `rewards`; rows past a buffer's end are left
/// alone, as `get_mut` finds none.
#[inline]
pub(crate) fn clear_agent_rows(
    flags: [&mut [Bool8]; 5],
    rewards: &mut [f32],
    i: usize,
    agents: usize,
) {
    if agents == 1 {
        for f in flags.into_iter().filter_map(|b| b.get_mut(i)) {
            *f = Bool8::FALSE;
        }
        if let Some(r) = rewards.get_mut(i) {
            *r = 0.0;
        }
        return;
    }
    let range = i * agents..(i + 1) * agents;
    for buf in flags {
        if let Some(rows) = buf.get_mut(range.clone()) {
            rows.fill(Bool8::FALSE);
        }
    }
    if let Some(rows) = rewards.get_mut(range) {
        rows.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::Bool8;

    use super::clear_agent_rows;

    const T: Bool8 = Bool8::TRUE;
    const F: Bool8 = Bool8::FALSE;

    fn run(len: usize, i: usize, agents: usize) -> ([Vec<Bool8>; 5], Vec<f32>) {
        let mut flags: [Vec<Bool8>; 5] = core::array::from_fn(|_| vec![T; len]);
        let mut rewards = vec![7.0f32; len];
        let [terminated, truncated, died, spawned, invalid] = &mut flags;
        clear_agent_rows(
            [
                terminated.as_mut_slice(),
                truncated.as_mut_slice(),
                died.as_mut_slice(),
                spawned.as_mut_slice(),
                invalid.as_mut_slice(),
            ],
            &mut rewards,
            i,
            agents,
        );
        (flags, rewards)
    }

    #[test]
    fn test_clear_agent_rows_one_agent_clears_only_its_row_in_every_buffer() {
        let (flags, rewards) = run(4, 2, 1);
        for (k, buf) in flags.iter().enumerate() {
            assert_eq!(buf, &[T, T, F, T], "flag buffer {k}");
        }
        assert_eq!(rewards, [7.0, 7.0, 0.0, 7.0]);
    }

    #[test]
    fn test_clear_agent_rows_many_agents_clears_the_instance_span() {
        let (flags, rewards) = run(9, 1, 3);
        for (k, buf) in flags.iter().enumerate() {
            assert_eq!(buf, &[T, T, T, F, F, F, T, T, T], "flag buffer {k}");
        }
        assert_eq!(rewards, [7.0, 7.0, 7.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_clear_agent_rows_past_the_end_writes_nothing() {
        for agents in [1, 3] {
            let (flags, rewards) = run(3, 3, agents);
            for buf in &flags {
                assert_eq!(buf, &[T, T, T]);
            }
            assert_eq!(rewards, [7.0, 7.0, 7.0]);
        }
    }
}

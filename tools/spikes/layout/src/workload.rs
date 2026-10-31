// SPDX-License-Identifier: MIT OR Apache-2.0

//! Workloads shared by the equivalence test and the benches.

use crate::kernel::{SpikeEnv, Work};
use crate::layouts::Layout;

/// Deterministic actions for step `t`: splitmix64 of (t, row), legal for
/// both environments when masked by the caller's mask (Tag) or always
/// (CartPole, 2 choices).
#[must_use]
pub fn actions(rows: usize, choices: u32, t: u64) -> Vec<i32> {
    (0..rows)
        .map(|r| {
            let mut z = (t << 32 ^ r as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            i32::try_from((z ^ (z >> 31)) % u64::from(choices)).unwrap_or(0)
        })
        .collect()
}

/// Replaces every action the current mask forbids with 0 (stay).
pub fn legalize(acts: &mut [i32], work: &Work, choices: usize) {
    if work.masks.is_empty() {
        return;
    }
    for (r, a) in acts.iter_mut().enumerate() {
        let legal = usize::try_from(*a)
            .ok()
            .and_then(|c| work.masks.get(r * choices + c))
            .is_some_and(|m| m.get());
        if !legal {
            *a = 0;
        }
    }
}

/// Steps `layout` `steps` times from step `from` with the shared actions.
pub fn run<E: SpikeEnv, L: Layout<E>>(
    layout: &mut L,
    cfg: &E::Config,
    work: &mut Work,
    choices: u32,
    from: u64,
    steps: u64,
    limit: u32,
) {
    let rows = work.rewards.len();
    let width = usize::try_from(choices).unwrap_or(1);
    for t in from..from + steps {
        let mut acts = actions(rows, choices, t);
        legalize(&mut acts, work, width);
        layout.step_all(cfg, &acts, work, limit);
    }
}

/// The fork-heavy workload: snapshot `src`, fork it into `k` slots, step
/// the pool `steps` times, restore every destination from the snapshot.
pub fn fork_heavy<E: SpikeEnv, L: Layout<E>>(
    layout: &mut L,
    cfg: &E::Config,
    work: &mut Work,
    choices: u32,
    k: usize,
    steps: u64,
    limit: u32,
) {
    let Some((h, s)) = layout.get(0) else {
        return;
    };
    for d in 1..=k {
        layout.set(d, &h, &s);
    }
    run::<E, L>(layout, cfg, work, choices, 0, steps, limit);
    for d in 0..=k {
        layout.set(d, &h, &s);
    }
}

/// Every instance's (header, state) hash pair.
pub fn hashes<E: SpikeEnv, L: Layout<E>>(layout: &L, n: usize) -> Vec<(u64, u64)> {
    (0..n)
        .filter_map(|i| {
            let (h, s) = layout.get(i)?;
            Some((
                tessera_rl_core::StateHash::of(&h).get(),
                tessera_rl_core::StateHash::of(&s).get(),
            ))
        })
        .collect()
}

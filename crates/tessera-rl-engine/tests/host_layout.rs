// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HostPool::layout` (SPEC 14.4.3 draft, ADR-0065 item 5): the spaces,
//! agents and mask blocks a binding reads, fixed at construction.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod host_support;

use core::num::NonZeroU32;

use host_support::{AGENTS, Book, BookConfig, LEVELS, SIDES};
use tessera_rl_core::HostEnvironment;
use tessera_rl_engine::{AutoReset, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

type Pool = HostPool<Book, { Book::MAX_AGENTS }>;

fn pool(auto_reset: AutoReset) -> Pool {
    let mut options = PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(5));
    options.auto_reset = auto_reset;
    Pool::new(BookConfig::default(), options).unwrap()
}

#[test]
fn test_host_layout_matches_the_environment_and_the_outputs() {
    for mode in [
        AutoReset::NextStep,
        AutoReset::SameStep,
        AutoReset::Disabled,
    ] {
        let mut p = pool(mode);
        let cfg = BookConfig::default();
        let l = p.layout().clone();
        assert_eq!(l.num_envs, 3);
        assert_eq!(l.agents, AGENTS);
        assert_eq!(l.initial_agents, Book::initial_agents(&cfg));
        assert_eq!(l.spawns, Book::SPAWNS);
        assert_eq!(l.observation_space, Book::observation_space(&cfg).unwrap());
        assert_eq!(l.leaves.len(), 1);
        assert_eq!(
            (l.leaves[0].name.as_str(), l.leaves[0].shape.as_slice()),
            ("", [4].as_slice())
        );
        assert_eq!(l.action_space, Book::action_space(&cfg).unwrap());
        assert_eq!(l.terminal, mode == AutoReset::SameStep, "{mode:?}");
        assert_eq!(l.discrete_heads, [SIDES, LEVELS]);
        assert_eq!(l.continuous_heads, [] as [(f64, f64); 0]);
        assert_eq!(p.action_heads(), (2, 0));
        // One block per discrete head, laid out one after the other; the
        // row is the end of the last block.
        assert_eq!(l.mask_heads.len(), 2);
        let end = l
            .mask_heads
            .iter()
            .map(|m| m.offset + m.parent_sizes.iter().flatten().product::<usize>() * m.size)
            .max()
            .unwrap();
        assert_eq!(l.mask_len, end);
        let out = p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(out.masks().len(), l.num_envs * l.agents * l.mask_len);
        // Fixed: the same after a call.
        assert_eq!(p.layout(), &l);
    }
}

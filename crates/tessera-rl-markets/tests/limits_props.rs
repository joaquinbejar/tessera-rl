// SPDX-License-Identifier: MIT OR Apache-2.0

//! Review r1 of #381, correction 1, as a property: after any sequence of
//! actions drawn from the masks, every alive agent has
//! `|inventory| <= max_position` and, without `allow_short_cash`,
//! `cash >= 0` before the settlement, and every state passes `Validate`.
//! The same holds with arbitrary actions and no masks, because the step
//! decodes an illegal action as a no-op.

#![expect(
    clippy::indexing_slicing,
    reason = "property tests: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;

use proptest::prelude::*;
use support::{Rng, config, input, legal_actions, pool};
use tessera_rl_core::Validate;
use tessera_rl_engine::{EnvPool, PoolOptions, Reseed};
use tessera_rl_markets::actions::{CANCELS, HEADS, KINDS, OFFSETS, SIZES};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobConfig, LobState};
use tessera_rl_num::identity::Seed;

const INSTANCES: u32 = 3;

fn check(cfg: &LobConfig, s: &LobState) -> Result<(), TestCaseError> {
    prop_assert_eq!(LimitOrderBook::validate(cfg, s), Ok(()));
    let settled = s.step >= cfg.steps;
    for slot in 0..cfg.agent_count() {
        if !s.is_alive(slot) {
            continue;
        }
        let a = s.accounts[slot];
        prop_assert!(a.inventory.unsigned_abs() <= cfg.max_position);
        if !cfg.allow_short_cash && !settled {
            prop_assert!(a.cash >= 0, "slot {} cash {}", slot, a.cash);
        }
    }
    Ok(())
}

/// Tight limits, so the masks bind often.
fn lob_config() -> impl Strategy<Value = LobConfig> {
    (2..=6u32, 0..=3u32, 1..=6u32, 300..=3000i64).prop_map(|(agents, latency, pos, cash)| {
        LobConfig {
            max_position: pos,
            initial_cash: cash,
            steps: 40,
            ..config(agents, latency)
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn test_env_masked_actions_keep_position_and_cash_limits(
        cfg in lob_config(),
        seed in any::<u64>(),
        draws in any::<u64>(),
    ) {
        let mut p = pool(cfg.clone(), INSTANCES, seed, None);
        let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
        let mut rng = Rng(draws | 1);
        for _ in 0..60 {
            let a = legal_actions(&mut rng, &masks, INSTANCES as usize);
            let out = p.step(input(&a)).unwrap();
            prop_assert_eq!(out.episode_illegal_actions().iter().sum::<f32>().to_bits(), 0.0_f32.to_bits());
            masks = out.masks().to_vec();
            for i in 0..INSTANCES {
                check(&cfg, p.state(i).unwrap())?;
            }
        }
    }

    #[test]
    fn test_env_unmasked_actions_keep_position_and_cash_limits(
        cfg in lob_config(),
        seed in any::<u64>(),
        draws in any::<u64>(),
    ) {
        let options = PoolOptions::new(NonZeroU32::new(INSTANCES).unwrap(), Seed::new(seed));
        let mut p = EnvPool::<LimitOrderBook>::new(cfg.clone(), options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let mut rng = Rng(draws | 1);
        let sizes = [KINDS, OFFSETS, SIZES, CANCELS];
        for _ in 0..60 {
            let mut a = Vec::with_capacity(INSTANCES as usize * MAX_AGENTS * HEADS);
            for _ in 0..INSTANCES as usize * MAX_AGENTS {
                for n in sizes {
                    a.push(i32::try_from(rng.below(n as usize)).unwrap());
                }
            }
            p.step(input(&a)).unwrap();
            for i in 0..INSTANCES {
                check(&cfg, p.state(i).unwrap())?;
            }
        }
    }
}

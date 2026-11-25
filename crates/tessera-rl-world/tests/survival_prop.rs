// SPDX-License-Identifier: MIT OR Apache-2.0

//! Random configs and actions keep Survival-v1's invariants (`Validate`
//! after every step, through deaths and respawns) and never panic.

use core::num::NonZeroU32;

use proptest::prelude::*;
use tessera_rl_core::Validate;
use tessera_rl_engine::{ActionInput, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_world::survival::MAX_AGENTS;
use tessera_rl_world::{Survival, SurvivalConfig};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn test_survival_random_configs_and_actions_keep_the_invariants(
        agents in 16u32..=128,
        size in 32u32..=96,
        respawn in 0u32..=4,
        seed in 0u64..1_000,
        acts in proptest::collection::vec((0i32..5, 0i32..7), MAX_AGENTS),
    ) {
        let cfg = SurvivalConfig { agents, size, respawn, steps: 4096, ..SurvivalConfig::default() };
        let mut pool = EnvPool::<Survival, NoMasks>::new(cfg.clone(), PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        let mut a = vec![0i32; MAX_AGENTS * 2];
        for t in 0..40usize {
            for (i, &(m, x)) in acts.iter().enumerate() {
                // Rotate the per-agent choices each step.
                let k = (i + t) % acts.len();
                a[2 * i] = acts[k].0.max(m.min(0));
                a[2 * i + 1] = acts[k].1.max(x.min(0));
            }
            pool.step(ActionInput { discrete: &a, continuous: &[] }).unwrap();
            let s = pool.state(0).unwrap();
            prop_assert!(Survival::validate(&cfg, &s).is_ok(), "step {t}");
        }
    }
}

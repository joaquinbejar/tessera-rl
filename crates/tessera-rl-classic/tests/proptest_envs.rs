// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests of the classic environments (ADR-0024 layer 4): bounded
//! rewards, masks consistent with legality (the validation boundary agrees
//! with the mask), and configs whose canonical form and validation behave.

#![expect(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "property tests: a failure is the assertion; rewards are exact"
)]

use core::num::NonZeroU32;

use proptest::prelude::*;
use tessera_rl_classic::tag::grid::moved;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::EnvConfig;
use tessera_rl_core::config::{canonical_bytes, config_hash};
use tessera_rl_engine::{ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const A: usize = 8;

fn options(n: u32, seed: u64, limit: u32) -> PoolOptions {
    PoolOptions {
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    }
}

fn act(values: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: values,
        continuous: &[],
    }
}

/// Valid Tag configs: every field in range, at most 8 agents.
fn tag_config() -> impl Strategy<Value = TagConfig> {
    (
        3u32..=12,
        3u32..=12,
        1u32..=7,
        1u32..=7,
        0u32..=1000,
        0u32..=7,
    )
        .prop_filter("at most 8 agents", |(_, _, p, e, _, _)| p + e <= 8)
        .prop_map(
            |(width, height, pursuers, evaders, slip, radius)| TagConfig {
                width,
                height,
                pursuers,
                evaders,
                slip,
                radius,
            },
        )
}

/// Any Tag config, valid or not.
fn any_tag_config() -> impl Strategy<Value = TagConfig> {
    (0u32..80, 0u32..80, 0u32..10, 0u32..10, 0u32..1200, 0u32..10).prop_map(
        |(width, height, pursuers, evaders, slip, radius)| TagConfig {
            width,
            height,
            pursuers,
            evaders,
            slip,
            radius,
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn test_cartpole_rewards_are_one_for_any_actions(
        seed in any::<u64>(), acts in proptest::collection::vec(0..2i32, 1..300),
    ) {
        let mut p = EnvPool::<CartPole>::new(CartPoleConfig::default(), options(1, seed, 500)).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let mut ended = false;
        for a in acts {
            let out = p.step(act(&[a])).unwrap();
            // NextStep: the call after an episode end is the reset (reward 0).
            let expected = if ended { 0.0 } else { 1.0 };
            prop_assert_eq!(out.rewards(), [expected]);
            ended = out.episode_terminated()[0].get() || out.episode_truncated()[0].get();
        }
    }

    /// Rewards stay in [-1, pursuers]: an evader gets 0 or -1, a pursuer at
    /// most one point per evader it shares a cell with.
    #[test]
    fn test_tag_rewards_are_bounded_and_masks_match_legality(
        cfg in tag_config(), seed in any::<u64>(),
        choices in proptest::collection::vec(proptest::collection::vec(0..5i32, A), 1..60),
    ) {
        let mut p = EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options(1, seed, 200)).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        let evaders = f32::from(u16::try_from(cfg.evaders).unwrap());
        let mut ended = false;
        for row in choices {
            let state = *p.state(0).unwrap();
            let masks = p.outputs().masks().to_vec();
            let alive = p.outputs().alive().to_vec();
            // The mask is exactly legality from the agent's cell.
            for agent in 0..cfg.agents() {
                if !alive[agent].get() {
                    continue;
                }
                for a in 0..5i32 {
                    let legal = moved(state.pos[agent], a, cfg.width, cfg.height).is_some();
                    prop_assert_eq!(masks[agent * 5 + usize::try_from(a).unwrap()].get(), legal);
                }
            }
            // The validation boundary agrees with the mask.
            let all_legal = (0..cfg.agents()).all(|g| {
                !alive[g].get() || masks[g * 5 + usize::try_from(row[g]).unwrap()].get()
            });
            // A pending reset ignores every action (NextStep).
            let all_legal = all_legal || ended;
            match p.check_actions(act(&row)) {
                Ok(()) => prop_assert!(all_legal),
                Err(EngineError::IllegalAction { .. }) => prop_assert!(!all_legal),
                Err(e) => prop_assert!(false, "unexpected {e}"),
            }
            // Step only legal actions (stay replaces the rest).
            let legal_row: Vec<i32> = (0..A)
                .map(|g| {
                    if g < cfg.agents() && alive[g].get() && masks[g * 5 + usize::try_from(row[g]).unwrap()].get() {
                        row[g]
                    } else {
                        0
                    }
                })
                .collect();
            let out = p.step(act(&legal_row)).unwrap();
            ended = out.episode_terminated()[0].get() || out.episode_truncated()[0].get();
            for (g, r) in out.rewards().iter().enumerate() {
                if cfg.is_pursuer(g) {
                    prop_assert!((0.0..=evaders).contains(r), "pursuer reward {r}");
                } else {
                    prop_assert!(*r == 0.0 || *r == -1.0, "evader reward {r}");
                }
            }
        }
    }

    /// Validation accepts exactly the in-range configs with at most 8
    /// agents; equal configs have equal canonical bytes and hashes, and a
    /// changed field changes them.
    #[test]
    fn test_tag_config_validation_and_canonical_form(cfg in any_tag_config(), bump in 0usize..6) {
        let in_range = (3..=64).contains(&cfg.width)
            && (3..=64).contains(&cfg.height)
            && (1..=7).contains(&cfg.pursuers)
            && (1..=7).contains(&cfg.evaders)
            && cfg.slip <= 1000
            && cfg.radius <= 7
            && cfg.pursuers + cfg.evaders <= 8;
        prop_assert_eq!(cfg.validate().is_ok(), in_range);
        let built = EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options(1, 1, 10));
        prop_assert_eq!(built.is_ok(), in_range);
        let copy = cfg.clone();
        prop_assert_eq!(canonical_bytes(&copy), canonical_bytes(&cfg));
        let hash = |c: &TagConfig| config_hash("tessera/Tag-v1", 1, c).unwrap();
        prop_assert_eq!(hash(&copy), hash(&cfg));
        let mut other = cfg.clone();
        match bump {
            0 => other.width += 1,
            1 => other.height += 1,
            2 => other.pursuers += 1,
            3 => other.evaders += 1,
            4 => other.slip += 1,
            _ => other.radius += 1,
        }
        prop_assert_ne!(canonical_bytes(&other), canonical_bytes(&cfg));
        prop_assert_ne!(hash(&other), hash(&cfg));
    }
}

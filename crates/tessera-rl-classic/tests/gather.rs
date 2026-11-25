// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gather (`tessera/Gather-v1`, ADR-0082) through the engine: the `Set`
//! observation's leaves and bytes against the state, the baseline, and
//! import validation.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::gather::MAX_ITEMS;
use tessera_rl_classic::{Gather, GatherConfig, GatherState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic, Validate};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, Outputs, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

fn pool(cfg: GatherConfig, n: u32, seed: u64) -> EnvPool<Gather, NoMasks> {
    pool_with(cfg, n, seed, AutoReset::Disabled)
}

fn pool_with(
    cfg: GatherConfig,
    n: u32,
    seed: u64,
    auto_reset: AutoReset,
) -> EnvPool<Gather, NoMasks> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::new(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// Instance `i`'s observation leaves `(agent, items, mask)`.
fn observation(out: &Outputs, i: usize) -> (Vec<i32>, Vec<i32>, Vec<u8>) {
    let agent = &out.observation(0).unwrap()[i * 8..(i + 1) * 8];
    let items = &out.observation(1).unwrap()[i * 64..(i + 1) * 64];
    let mask = &out.observation(2).unwrap()[i * 8..(i + 1) * 8];
    (i32s(agent), i32s(items), mask.to_vec())
}

/// The observation the ADR-0082 contract gives for `s`.
fn expected(s: &GatherState) -> (Vec<i32>, Vec<i32>, Vec<u8>) {
    let mut items = vec![0; MAX_ITEMS * 2];
    let mut mask = vec![0; MAX_ITEMS];
    for (slot, (item, present)) in s.items.iter().zip(s.present).enumerate() {
        if present.get() {
            items[slot * 2] = i32::from(item[0]);
            items[slot * 2 + 1] = i32::from(item[1]);
            mask[slot] = 1;
        }
    }
    let agent = vec![i32::from(s.agent[0]), i32::from(s.agent[1])];
    (agent, items, mask)
}

fn baseline(cfg: &GatherConfig, s: &GatherState) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    Gather::act(cfg, s, AgentSlot::new(0, 1).unwrap(), &mut rng, &mut writer);
    discrete[0]
}

#[test]
fn test_gather_observation_space_is_agent_then_item_set() {
    let space = Gather::observation_space(&GatherConfig::default()).unwrap();
    let Space::Dict(entries) = space else {
        panic!("a dict");
    };
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["agent", "items"]);
    assert!(matches!(&entries[1].1, Space::Set { max: 8, .. }));
}

#[test]
fn test_gather_observation_matches_state_with_absent_slots_zero() {
    let cfg = GatherConfig {
        size: 6,
        items: 5,
        max_steps: 30,
    };
    // Same-step auto-reset: after a step the state and the observation are
    // both the next episode's when one ended.
    let mut p = pool_with(cfg.clone(), 8, 41, AutoReset::SameStep);
    for t in 0..40 {
        let actions: Vec<i32> = (0..8u32)
            .map(|i| baseline(&cfg, &p.state(i).unwrap()))
            .map(|a| if t % 5 == 0 { (a + 1) % 4 } else { a })
            .collect();
        let states: Vec<GatherState> = {
            p.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
            (0..8u32).map(|i| p.state(i).unwrap()).collect()
        };
        let out = p.outputs().unwrap();
        for (i, s) in states.iter().enumerate() {
            assert_eq!(observation(out, i), expected(s), "step {t} instance {i}");
        }
    }
}

#[test]
fn test_gather_baseline_collects_one_item_by_the_shortest_walk() {
    let cfg = GatherConfig {
        size: 8,
        items: 1,
        max_steps: 100,
    };
    for seed in 0..200 {
        let mut p = pool(cfg.clone(), 1, seed);
        let s = p.state(0).unwrap();
        let shortest = (s.items[0][0] - s.agent[0]).abs() + (s.items[0][1] - s.agent[1]).abs();
        let mut steps = 0;
        loop {
            let a = baseline(&cfg, &p.state(0).unwrap());
            let out = p
                .step(ActionInput {
                    discrete: &[a],
                    continuous: &[],
                })
                .unwrap();
            steps += 1;
            if out.episode_terminated()[0].get() {
                assert_eq!(out.rewards()[0], 1.0);
                break;
            }
            assert!(steps <= shortest, "seed {seed}");
        }
        assert_eq!(steps, shortest, "seed {seed}");
    }
}

#[test]
fn test_gather_validate_refuses_inconsistent_states() {
    let cfg = GatherConfig::default();
    let mut s = pool(cfg.clone(), 1, 3).state(0).unwrap();
    assert!(Gather::validate(&cfg, &s).is_ok());
    s.left += 1;
    assert_eq!(
        Gather::validate(&cfg, &s).unwrap_err().code,
        "a left count that disagrees with the items present"
    );
    s.left -= 1;
    s.agent = [8, 0];
    assert_eq!(
        Gather::validate(&cfg, &s).unwrap_err().code,
        "an agent off the grid"
    );
    let few = GatherConfig {
        items: 2,
        ..GatherConfig::default()
    };
    // Version 2: `items` is adaptive, so a state from a slot whose target
    // is 6 holds items past the config's 2, and is valid.
    let mut s = pool(few.clone(), 1, 3).state(0).unwrap();
    s.present[5] = Bool8::TRUE;
    s.items[5] = [5, 5];
    s.left += 1;
    if s.items[5] == s.agent {
        s.items[5] = [4, 4];
    }
    assert_eq!(Gather::validate(&few, &s), Ok(()));
}

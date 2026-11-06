// SPDX-License-Identifier: MIT OR Apache-2.0

//! Edge fixtures the classic environments add to the engine's (ADR-0024
//! layer 3): ignored actions for caught evaders and for slots outside
//! `initial_agents`, and illegal actions under the penalize policy.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::tag::grid::moved;
use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const A: usize = 8;

fn pool(cfg: TagConfig, policy: IllegalActionPolicy, seed: u64) -> EnvPool<Tag, Masked> {
    let options = PoolOptions {
        illegal_action: policy,
        max_episode_steps: NonZeroU32::new(200),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    EnvPool::with_masks(cfg, options).unwrap()
}

fn act(values: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: values,
        continuous: &[],
    }
}

#[test]
fn test_edge_tag_actions_of_empty_and_caught_slots_are_ignored() {
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 2,
        evaders: 2,
        slip: 0,
        ..TagConfig::default()
    };
    let mut p = pool(cfg, IllegalActionPolicy::Noop, 3);
    p.reset_all(Reseed::Keep).unwrap();
    // Slots 4..8 are empty: any value, even out of range, is not checked.
    let mut acts = [0; A];
    acts[4..].fill(99);
    p.check_actions(act(&acts)).unwrap();
    p.step(act(&acts)).unwrap();
    // Chase until an evader is caught but the episode goes on.
    let mut caught = None;
    for _ in 0..60 {
        let s = *p.state(0).unwrap();
        let mut acts = [0; A];
        for (slot, a) in acts.iter_mut().enumerate().take(2) {
            let target = (2..4).find(|&e| !s.caught[e].get()).unwrap_or(2);
            let [dx, dy] = [
                s.pos[target][0] - s.pos[slot][0],
                s.pos[target][1] - s.pos[slot][1],
            ];
            *a = match (dx.signum(), dy.signum()) {
                (1, _) => 3,
                (-1, _) => 4,
                (_, 1) => 2,
                (_, -1) => 1,
                _ => 0,
            };
        }
        acts[4..].fill(99);
        let out = p.step(act(&acts)).unwrap();
        if out.episode_terminated()[0].get() {
            break;
        }
        if let Some(e) = (2..4).find(|&e| p.state(0).unwrap().caught[e].get()) {
            caught = Some(e);
            break;
        }
    }
    let e = caught.expect("a capture that does not end the episode");
    assert!(!p.outputs().unwrap().alive()[e].get());
    // The caught evader's action is ignored: out of range is accepted.
    let mut acts = [0; A];
    acts[e] = -7;
    acts[4..].fill(99);
    p.check_actions(act(&acts)).unwrap();
    let before = p.state(0).unwrap().pos[e];
    p.step(act(&acts)).unwrap();
    assert_eq!(
        p.state(0).unwrap().pos[e],
        before,
        "a caught evader stays put"
    );
}

#[test]
fn test_edge_tag_illegal_move_is_penalized_or_rejected() {
    let cfg = TagConfig {
        slip: 0,
        ..TagConfig::default()
    };
    // The first seed that starts agent 0 on an edge of the grid.
    let (mut p, illegal) = (0..1_000)
        .find_map(|seed| {
            let mut p = pool(cfg.clone(), IllegalActionPolicy::Penalize(-0.5), seed);
            p.reset_all(Reseed::Keep).unwrap();
            let cell = p.state(0).unwrap().pos[0];
            let illegal = (1..5).find(|&a| moved(cell, a, cfg.width, cfg.height).is_none())?;
            Some((p, illegal))
        })
        .unwrap();
    let s = *p.state(0).unwrap();
    let mut acts = [0; A];
    acts[0] = illegal;
    assert!(matches!(
        p.check_actions(act(&acts)),
        Err(EngineError::IllegalAction { agent: 0, .. })
    ));
    match p.step(act(&acts)) {
        // `debug-checks`: rejected atomically.
        Err(e) => assert!(matches!(e, EngineError::IllegalAction { agent: 0, .. })),
        // Default: replaced by stay, penalized and counted.
        Ok(out) => {
            assert_eq!(out.rewards()[0], -0.5);
            assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 1);
            assert_eq!(p.state(0).unwrap().pos[0], s.pos[0]);
        }
    }
}

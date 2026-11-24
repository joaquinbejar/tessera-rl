// SPDX-License-Identifier: MIT OR Apache-2.0

//! Archive resets on a multi-agent environment with deaths (ADR-0075 item
//! 3): an entry saved after an evader was caught starts the episode with
//! that slot dead, the alive slots acting, and the instance's own
//! generations kept.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::tag::MAX_RADIUS;
use tessera_rl_classic::{Tag, TagConfig, TagState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, RecordLayout};
use tessera_rl_engine::{
    ActionInput, Archive, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed, ResetSource,
};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const A: usize = 8;

fn cfg() -> TagConfig {
    TagConfig {
        pursuers: 3,
        evaders: 3,
        ..TagConfig::default()
    }
}

fn pool(seed: u64) -> EnvPool<Tag, Masked> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        max_episode_steps: NonZeroU32::new(500),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    EnvPool::with_masks(cfg(), options).unwrap()
}

/// The baseline with a window that covers the whole grid (grids up to 8
/// cells wide), so it sees every agent: the chase these tests were written
/// for, whatever the pool's observation radius (ADR-0079, #437).
fn heuristic(cfg: &TagConfig, state: &TagState, agent: usize) -> i32 {
    let cfg = &TagConfig {
        radius: MAX_RADIUS,
        ..cfg.clone()
    };
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(u16::try_from(agent).unwrap(), A).unwrap();
    Tag::act(cfg, state, slot, &mut rng, &mut writer);
    discrete[0]
}

/// A record of instance 0 with at least one evader caught: pursuers chase,
/// evaders stay.
fn caught_record() -> Record<Tag> {
    let c = cfg();
    let mut p = pool(4);
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..400 {
        let state = p.state(0).unwrap();
        let mut actions = vec![0i32; A];
        for (a, slot) in actions.iter_mut().enumerate().take(c.agents()) {
            if c.is_pursuer(a) {
                *slot = heuristic(&c, &state, a);
            }
        }
        let masks = p.outputs().unwrap().masks().to_vec();
        for (row, a) in actions.iter_mut().enumerate() {
            if !masks[row * 5 + usize::try_from(*a).unwrap()].get() {
                *a = 0;
            }
        }
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
        let r = p.snapshot(0).unwrap();
        let rows = r.header().agents();
        let dead_evader = (0..c.agents()).any(|a| !c.is_pursuer(a) && !rows[a].alive.get());
        if dead_evader && !r.header().engine().pending_reset.get() {
            return r;
        }
    }
    panic!("no evader was caught in 400 steps");
}

#[test]
fn test_archive_resets_tag_entry_alive_slots_start_the_episode() {
    let entry = caught_record();
    let entry_alive: Vec<bool> = entry
        .header()
        .agents()
        .iter()
        .map(|r| r.alive.get())
        .collect();
    assert!(entry_alive.iter().take(cfg().agents()).any(|a| !a));
    let archive = Archive::new(&cfg(), &[entry]).unwrap();
    let mut p = pool(9);
    p.reset_all(Reseed::Keep).unwrap();
    let generations: Vec<u32> = p
        .snapshot(0)
        .unwrap()
        .header()
        .agents()
        .iter()
        .map(|r| r.generation)
        .collect();
    p.set_reset_source(ResetSource::Archive {
        archive,
        weights: None,
    })
    .unwrap();
    p.reset_from(&[0], &[0]).unwrap();
    let now = p.snapshot(0).unwrap();
    let rows = now.header().agents();
    let alive: Vec<bool> = rows.iter().map(|r| r.alive.get()).collect();
    assert_eq!(alive, entry_alive, "the entry's alive slots");
    // Tag is simultaneous: every alive agent acts, no dead one does.
    for r in rows {
        assert_eq!(r.acting.get(), r.alive.get());
        assert_eq!(r.episode_return, 0.0);
    }
    let after: Vec<u32> = rows.iter().map(|r| r.generation).collect();
    assert_eq!(after, generations, "generations are the instance's own");
    assert_eq!(now.state().pos, entry.state().pos);
}

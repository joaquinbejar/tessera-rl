// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reseeding tests (SPEC 8.5, ADR-0007) on CartPole and Tag: reseeding one
//! instance leaves the others alone, restore after a different reseed,
//! duplicate seeds and lineages, a changed first list entry, and seed
//! wrapping at 2^64.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;

use support::{Pooled, actions};
use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::{Heuristic, StateHash};
use tessera_rl_engine::{
    ActionInput, EnvPool, MAX_LEAVES, MaskProvider, Masked, NoMasks, PoolOptions, Reseed,
};
use tessera_rl_num::identity::{LogicalId, Seed};

const SEED: u64 = 5;

/// Observations and state hash of one instance.
fn view<E: Pooled, M: MaskProvider<E>>(p: &EnvPool<E, M>, id: u32) -> (Vec<u8>, u64) {
    let out = p.outputs().unwrap();
    let n = out.rewards().len() / E::MAX_AGENTS;
    let i = usize::try_from(id).unwrap();
    let mut obs = Vec::new();
    for leaf in 0..MAX_LEAVES {
        let Some(bytes) = out.observation(leaf) else {
            break;
        };
        let per = bytes.len() / n;
        obs.extend_from_slice(&bytes[i * per..(i + 1) * per]);
    }
    (obs, StateHash::of(p.state(id).unwrap()).get())
}

struct Suite<E: Pooled, M: MaskProvider<E>> {
    cfg: E::Config,
    build: fn(E::Config, PoolOptions) -> EnvPool<E, M>,
    limit: u32,
}

impl<E, M> Suite<E, M>
where
    E: Pooled + Heuristic,
    E::Config: Clone,
    M: MaskProvider<E>,
{
    fn pool(&self, n: u32) -> EnvPool<E, M> {
        let options = PoolOptions {
            max_episode_steps: NonZeroU32::new(self.limit),
            ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(SEED))
        };
        (self.build)(self.cfg.clone(), options)
    }

    /// One step with generated actions; rows in `copy` (dst, src) take the
    /// actions of another instance in the same state.
    fn step(&self, p: &mut EnvPool<E, M>, t: u32, copy: &[(usize, usize)]) {
        let a = E::MAX_AGENTS;
        let mut acts = actions(p, &self.cfg, SEED, t);
        for &(dst, src) in copy {
            let row = acts[src * a..(src + 1) * a].to_vec();
            acts[dst * a..(dst + 1) * a].copy_from_slice(&row);
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    }

    fn run(&self, p: &mut EnvPool<E, M>, from: u32, steps: u32, id: u32) -> Vec<(Vec<u8>, u64)> {
        (from..from + steps)
            .map(|t| {
                self.step(p, t, &[]);
                view(p, id)
            })
            .collect()
    }

    fn reseeding_one_instance_leaves_the_others_alone(&self) {
        let (mut a, mut b) = (self.pool(4), self.pool(4));
        a.reset_all(Reseed::Keep).unwrap();
        b.reset_all(Reseed::Keep).unwrap();
        self.run(&mut a, 1, 20, 0);
        self.run(&mut b, 1, 20, 0);
        b.reset(&[1], Reseed::Base(Seed::new(123))).unwrap();
        for id in [0, 2, 3] {
            assert_eq!(view(&a, id), view(&b, id), "instance {id} after the reseed");
        }
        for t in 21..60 {
            self.step(&mut a, t, &[]);
            self.step(&mut b, t, &[]);
            for id in [0, 2, 3] {
                assert_eq!(view(&a, id), view(&b, id), "instance {id} at {t}");
            }
        }
        assert_ne!(
            StateHash::of(a.state(1).unwrap()),
            StateHash::of(b.state(1).unwrap())
        );
    }

    fn restore_after_a_different_reseed_continues_the_snapshot(&self) {
        let mut p = self.pool(2);
        p.reset_all(Reseed::Keep).unwrap();
        self.run(&mut p, 1, 15, 0);
        let record = p.snapshot(0).unwrap();
        let expected = self.run(&mut p, 16, 40, 0);
        p.reset(&[0], Reseed::Base(Seed::new(777))).unwrap();
        self.run(&mut p, 100, 10, 0);
        p.restore(0, &record).unwrap();
        // Instance 1 drew different actions meanwhile, but instance 0's
        // actions depend only on its own state and the step.
        assert_eq!(self.run(&mut p, 16, 40, 0), expected);
    }

    fn duplicate_seeds_match_until_lineages_differ(&self) {
        let mut p = self.pool(2);
        let seeds = [Seed::new(42), Seed::new(42)];
        p.reset_all(Reseed::List(&seeds)).unwrap();
        assert_eq!(view(&p, 0), view(&p, 1));
        for t in 1..80 {
            self.step(&mut p, t, &[(1, 0)]);
            assert_eq!(view(&p, 0), view(&p, 1), "step {t}");
        }
        p.set_lineage(1, LogicalId::new(9)).unwrap();
        p.reset_all(Reseed::List(&seeds)).unwrap();
        assert_ne!(view(&p, 0).1, view(&p, 1).1, "distinct lineages");
    }

    fn changing_the_first_list_entry_changes_only_instance_zero(&self) {
        let (mut a, mut b) = (self.pool(4), self.pool(4));
        let seeds = [11, 12, 13, 14].map(Seed::new);
        let mut first = seeds;
        first[0] = Seed::new(99);
        a.reset_all(Reseed::List(&seeds)).unwrap();
        b.reset_all(Reseed::List(&first)).unwrap();
        assert_ne!(view(&a, 0).1, view(&b, 0).1);
        for t in 1..40 {
            self.step(&mut a, t, &[]);
            self.step(&mut b, t, &[]);
            for id in 1..4 {
                assert_eq!(view(&a, id), view(&b, id), "instance {id} at {t}");
            }
        }
    }

    fn base_seeds_wrap_at_two_to_the_64(&self) {
        let (mut a, mut b) = (self.pool(2), self.pool(2));
        a.reset_all(Reseed::Base(Seed::new(u64::MAX))).unwrap();
        b.reset_all(Reseed::List(&[Seed::new(u64::MAX), Seed::new(0)]))
            .unwrap();
        for id in 0..2 {
            assert_eq!(view(&a, id), view(&b, id), "instance {id}");
        }
    }

    fn all(&self) {
        self.reseeding_one_instance_leaves_the_others_alone();
        self.restore_after_a_different_reseed_continues_the_snapshot();
        self.duplicate_seeds_match_until_lineages_differ();
        self.changing_the_first_list_entry_changes_only_instance_zero();
        self.base_seeds_wrap_at_two_to_the_64();
    }
}

#[test]
fn test_reseed_cartpole_suite_passes() {
    Suite::<CartPole, NoMasks> {
        cfg: CartPoleConfig::default(),
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        limit: MAX_EPISODE_STEPS,
    }
    .all();
}

#[test]
fn test_reseed_tag_suite_passes() {
    Suite::<Tag, Masked> {
        cfg: TagConfig {
            slip: 200,
            ..TagConfig::default()
        },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        limit: 200,
    }
    .all();
}

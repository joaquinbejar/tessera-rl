// SPDX-License-Identifier: MIT OR Apache-2.0

//! Continuation tests (SPEC 9.1, normative; ADR-0010): snapshot one
//! instance at step t, run K steps, then restore it into the same slot, a
//! different slot and a different pool with a different seed (by `restore`
//! and by `import`), replay the same actions, and require identical
//! observations, rewards, masks, agent and episode flags, generations,
//! episode statistics, headers and state hashes. `fork` continues the same
//! way; `fork_branched` diverges.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::panic,
    clippy::type_complexity,
    reason = "test helpers: a failure is the assertion"
)]

mod support;

use core::num::NonZeroU32;

use support::{Pooled, actions};
use tessera_rl_classic::cartpole::MAX_EPISODE_STEPS;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::encoding::StateHasher;
use tessera_rl_core::{Heuristic, StateHash};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, MAX_LEAVES, MaskProvider, Masked, NoMasks,
    PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 8;
const SRC: u32 = 2;
const OTHER_SLOT: u32 = 5;
const OTHER_POOL_SLOT: u32 = 1;
const K: u32 = 50;
const SEED: u64 = 77;

/// Everything one instance reports after a call, plus its record.
#[derive(Debug, Clone, PartialEq)]
struct Frame {
    obs: Vec<u8>,
    terminal_obs: Vec<u8>,
    rewards: Vec<u32>,
    masks: Vec<u8>,
    flags: Vec<u8>,
    generations: Vec<u32>,
    stats: [u32; 4],
    episode: [u8; 2],
    record: u64,
    state: u64,
}

fn frame<E: Pooled, M: MaskProvider<E>>(p: &EnvPool<E, M>, id: u32, same_step: bool) -> Frame {
    let out = p.outputs();
    let a = E::MAX_AGENTS;
    let n = out.rewards().len() / a;
    let i = usize::try_from(id).unwrap();
    let rows = i * a..(i + 1) * a;
    let mut obs = Vec::new();
    let mut terminal_obs = Vec::new();
    let ended = out.episode_terminated()[i].get() || out.episode_truncated()[i].get();
    for leaf in 0..MAX_LEAVES {
        let Some(bytes) = out.observation(leaf) else {
            break;
        };
        let per = bytes.len() / n;
        obs.extend_from_slice(&bytes[i * per..(i + 1) * per]);
        if same_step && ended {
            let t = out.terminal_observation(leaf).unwrap();
            terminal_obs.extend_from_slice(&t[i * per..(i + 1) * per]);
        }
    }
    let masks = if out.masks().is_empty() {
        Vec::new()
    } else {
        let per = out.masks().len() / n;
        out.masks()[i * per..(i + 1) * per]
            .iter()
            .map(|m| m.to_byte())
            .collect()
    };
    let mut flags = Vec::new();
    for buf in [
        out.agent_terminated(),
        out.agent_truncated(),
        out.alive(),
        out.acting(),
    ] {
        flags.extend(buf[rows.clone()].iter().map(|f| f.to_byte()));
    }
    let mut hasher = StateHasher::new();
    hasher.add(p.header(id).unwrap());
    hasher.add(p.state(id).unwrap());
    Frame {
        obs,
        terminal_obs,
        rewards: out.rewards()[rows.clone()]
            .iter()
            .map(|r| r.to_bits())
            .collect(),
        masks,
        flags,
        generations: out.generations()[rows].to_vec(),
        stats: [
            out.episode_return()[i].to_bits(),
            out.episode_length()[i].to_bits(),
            out.episode_illegal_actions()[i].to_bits(),
            out.episode_invalid_agent_events()[i].to_bits(),
        ],
        episode: [
            out.episode_terminated()[i].to_byte(),
            out.episode_truncated()[i].to_byte(),
        ],
        record: hasher.finish().get(),
        state: StateHash::of(p.state(id).unwrap()).get(),
    }
}

fn step<E: Pooled, M: MaskProvider<E>>(p: &mut EnvPool<E, M>, acts: &[i32]) {
    p.step(ActionInput {
        discrete: acts,
        continuous: &[],
    })
    .unwrap();
}

/// Generated actions for `p` with instance `dst`'s row replaced by `row`.
fn with_row<E, M>(p: &EnvPool<E, M>, cfg: &E::Config, t: u32, dst: u32, row: &[i32]) -> Vec<i32>
where
    E: Pooled + Heuristic,
    M: MaskProvider<E>,
{
    let a = E::MAX_AGENTS;
    let mut acts = actions(p, cfg, SEED ^ 0x5eed, t);
    let d = usize::try_from(dst).unwrap();
    acts[d * a..(d + 1) * a].copy_from_slice(row);
    acts
}

struct Case<E: Pooled, M: MaskProvider<E>> {
    cfg: E::Config,
    build: fn(E::Config, PoolOptions) -> EnvPool<E, M>,
    auto_reset: AutoReset,
    limit: u32,
    /// An agent dies before its episode ends (Tag captures).
    deaths: bool,
}

impl<E, M> Case<E, M>
where
    E: Pooled + Heuristic,
    E::Config: Clone,
    M: MaskProvider<E>,
{
    fn pool(&self, n: u32, seed: u64) -> EnvPool<E, M> {
        let options = PoolOptions {
            auto_reset: self.auto_reset,
            max_episode_steps: NonZeroU32::new(self.limit),
            chunk_size: NonZeroU32::new(2).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
        };
        (self.build)(self.cfg.clone(), options)
    }

    fn same_step(&self) -> bool {
        self.auto_reset == AutoReset::SameStep
    }

    /// The reference run: snapshot `SRC` after `t` steps, then K more
    /// steps; returns the record, the K action rows of `SRC` and frames.
    fn reference(
        &self,
        t: u32,
    ) -> (
        tessera_rl_engine::Record<E>,
        Vec<Vec<i32>>,
        Vec<Frame>,
        EnvPool<E, M>,
    ) {
        let a = E::MAX_AGENTS;
        let s = usize::try_from(SRC).unwrap();
        let mut p = self.pool(N, SEED);
        p.reset_all(Reseed::Keep).unwrap();
        for k in 1..=t {
            let acts = actions(&p, &self.cfg, SEED, k);
            step(&mut p, &acts);
        }
        let record = p.snapshot(SRC).unwrap();
        let (mut rows, mut frames) = (Vec::new(), Vec::new());
        for k in 1..=K {
            let acts = actions(&p, &self.cfg, SEED, t + k);
            rows.push(acts[s * a..(s + 1) * a].to_vec());
            step(&mut p, &acts);
            frames.push(frame(&p, SRC, self.same_step()));
        }
        (record, rows, frames, p)
    }

    fn replay(&self, p: &mut EnvPool<E, M>, dst: u32, rows: &[Vec<i32>]) -> Vec<Frame> {
        rows.iter()
            .enumerate()
            .map(|(k, row)| {
                let t = u32::try_from(k).unwrap();
                let acts = with_row(p, &self.cfg, t, dst, row);
                step(p, &acts);
                frame(p, dst, self.same_step())
            })
            .collect()
    }

    fn check_at(&self, t: u32)
    where
        E: tessera_rl_core::Validate,
    {
        let (record, rows, expected, mut p) = self.reference(t);
        assert!(
            expected.windows(2).any(|w| w[0] != w[1]),
            "t {t}: the reference frames never change"
        );
        let what = |w: &str| format!("t {t}, {w}");
        // (a) The same slot.
        p.restore(SRC, &record).unwrap();
        assert_eq!(
            self.replay(&mut p, SRC, &rows),
            expected,
            "{}",
            what("same slot")
        );
        // (b) A different slot.
        p.restore(OTHER_SLOT, &record).unwrap();
        assert_eq!(
            self.replay(&mut p, OTHER_SLOT, &rows),
            expected,
            "{}",
            what("other slot")
        );
        // (c) A different pool with a different seed, by restore and by
        // import (the untrusted path).
        let mut q = self.pool(4, SEED + 1_000);
        q.reset_all(Reseed::Keep).unwrap();
        q.restore(OTHER_POOL_SLOT, &record).unwrap();
        assert_eq!(
            self.replay(&mut q, OTHER_POOL_SLOT, &rows),
            expected,
            "{}",
            what("other pool, restore")
        );
        let mut q = self.pool(4, SEED + 2_000);
        q.reset_all(Reseed::Keep).unwrap();
        let decoded = DecodedRecord::from_parts(*record.header(), *record.state()).unwrap();
        q.import(&[OTHER_POOL_SLOT], &[decoded]).unwrap();
        assert_eq!(
            self.replay(&mut q, OTHER_POOL_SLOT, &rows),
            expected,
            "{}",
            what("other pool, import")
        );
    }

    /// `fork` into another slot continues identically; `fork_branched`
    /// (unlabelled) diverges within `horizon` steps.
    fn check_forks(&self, t: u32, horizon: u32) {
        let mut p = self.pool(N, SEED);
        p.reset_all(Reseed::Keep).unwrap();
        for k in 1..=t {
            let acts = actions(&p, &self.cfg, SEED, k);
            step(&mut p, &acts);
        }
        p.fork(SRC, &[OTHER_SLOT]).unwrap();
        let mut b = self.pool(N, SEED);
        b.reset_all(Reseed::Keep).unwrap();
        for k in 1..=t {
            let acts = actions(&b, &self.cfg, SEED, k);
            step(&mut b, &acts);
        }
        b.fork_branched(SRC, &[OTHER_SLOT], None).unwrap();
        let a = E::MAX_AGENTS;
        let (s, o) = (
            usize::try_from(SRC).unwrap(),
            usize::try_from(OTHER_SLOT).unwrap(),
        );
        let mut diverged = false;
        for k in 1..=horizon {
            let mut acts = actions(&p, &self.cfg, SEED, t + k);
            let row = acts[s * a..(s + 1) * a].to_vec();
            acts[o * a..(o + 1) * a].copy_from_slice(&row);
            step(&mut p, &acts);
            let (x, y) = (
                frame(&p, SRC, self.same_step()),
                frame(&p, OTHER_SLOT, self.same_step()),
            );
            // The whole frame: masks, terminal obs, flags, generations,
            // stats and the record hash too.
            assert_eq!(x, y, "fork at {t}, step {k}");

            let mut acts = actions(&b, &self.cfg, SEED, t + k);
            let row = acts[s * a..(s + 1) * a].to_vec();
            acts[o * a..(o + 1) * a].copy_from_slice(&row);
            // The rows may disagree on legality once the branches diverge.
            let legal = support::actions(&b, &self.cfg, SEED ^ 1, t + k);
            if frame(&b, SRC, false).masks != frame(&b, OTHER_SLOT, false).masks {
                acts[o * a..(o + 1) * a].copy_from_slice(&legal[o * a..(o + 1) * a]);
            }
            step(&mut b, &acts);
            diverged |= frame(&b, SRC, false).state != frame(&b, OTHER_SLOT, false).state;
        }
        assert!(
            diverged,
            "fork_branched at {t} never diverged in {horizon} steps"
        );
    }

    /// The first call (after `t = 0`) where instance `SRC` ends an episode,
    /// and the first where one of its agents dies without ending it.
    fn landmarks(&self) -> (u32, Option<u32>) {
        let mut p = self.pool(N, SEED);
        p.reset_all(Reseed::Keep).unwrap();
        let a = E::MAX_AGENTS;
        let s = usize::try_from(SRC).unwrap();
        let mut death = None;
        for k in 1..=2_000 {
            let acts = actions(&p, &self.cfg, SEED, k);
            step(&mut p, &acts);
            let out = p.outputs();
            let ended = out.episode_terminated()[s].get() || out.episode_truncated()[s].get();
            if ended {
                return (k, death);
            }
            if death.is_none()
                && out.agent_terminated()[s * a..(s + 1) * a]
                    .iter()
                    .any(|f| f.get())
            {
                death = Some(k);
            }
        }
        panic!("instance {SRC} never ended an episode");
    }

    fn check_all(&self, horizon: u32)
    where
        E: tessera_rl_core::Validate,
    {
        let (end, death) = self.landmarks();
        assert_eq!(death.is_some(), self.deaths, "death landmark");
        // Mid-episode, the step before the end, the end itself (a pending
        // reset in NextStep, a terminal observation in SameStep) and just
        // after it.
        let mut points = vec![3, end - 1, end, end + 1];
        points.extend(death);
        if let Some(d) = death {
            points.push(d - 1);
        }
        for t in points {
            self.check_at(t);
        }
        self.check_forks(end / 2, horizon);
    }
}

fn cartpole(auto_reset: AutoReset) -> Case<CartPole, NoMasks> {
    Case {
        cfg: CartPoleConfig::default(),
        build: |cfg, options| EnvPool::new(cfg, options).unwrap(),
        auto_reset,
        limit: MAX_EPISODE_STEPS,
        deaths: false,
    }
}

fn tag(auto_reset: AutoReset) -> Case<Tag, Masked> {
    Case {
        cfg: TagConfig {
            pursuers: 3,
            evaders: 3,
            slip: 200,
            ..TagConfig::default()
        },
        build: |cfg, options| EnvPool::with_masks(cfg, options).unwrap(),
        auto_reset,
        limit: 200,
        deaths: true,
    }
}

#[test]
fn test_continuation_cartpole_next_step_restores_identically_everywhere() {
    // CartPole draws only at reset: a branch diverges after the next reset.
    cartpole(AutoReset::NextStep).check_all(600);
}

#[test]
fn test_continuation_cartpole_same_step_restores_identically_everywhere() {
    cartpole(AutoReset::SameStep).check_all(600);
}

#[test]
fn test_continuation_tag_next_step_restores_identically_everywhere() {
    tag(AutoReset::NextStep).check_all(50);
}

#[test]
fn test_continuation_tag_same_step_restores_identically_everywhere() {
    tag(AutoReset::SameStep).check_all(50);
}

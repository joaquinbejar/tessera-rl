// SPDX-License-Identifier: MIT OR Apache-2.0

//! The check of `ActionMask::head_active` (ADR-0086 decision 6): an
//! environment that declares a head inactive must ignore its value, so a
//! canonical joint and its copies with other values in the inactive heads
//! are the same game action. The engine cannot check this per step; this
//! function checks it on sampled records, for environment tests.

use core::num::NonZeroU32;

use tessera_rl_core::space::{JointHeads, MAX_JOINT_HEADS};
use tessera_rl_core::{ActionMask, AdaptiveValues, Environment, RecordLayout, StateHash};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, IllegalActionPolicy, MaskProvider, Masked,
    PoolOptions, Record,
};
use tessera_rl_num::identity::Seed;
use thiserror::Error;

use crate::error::SearchError;
use crate::pool::joint_heads;

/// What [`check_head_activity`] covered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeadActivityReport {
    /// Records checked.
    pub roots: usize,
    /// (root, acting agent, legal canonical joint) cases stepped.
    pub cases: u64,
    /// Cases whose joint has at least one inactive head, so the copies
    /// differ from it.
    pub with_inactive: u64,
}

/// Why [`check_head_activity`] failed.
#[derive(Debug, Error)]
pub enum HeadActivityError {
    /// The action space has no joint index (as for search).
    #[error(transparent)]
    Actions(#[from] SearchError),
    /// The engine pool failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A copy with other values in the inactive heads stepped differently.
    #[error(
        "record {root}, agent {agent}: joint {joint} and its copy {copy} (inactive heads changed) differ in {what}"
    )]
    Differs {
        /// Index into the records.
        root: usize,
        /// The agent whose joint was varied.
        agent: usize,
        /// The canonical joint.
        joint: u32,
        /// The copy's joint.
        copy: u32,
        /// The first output that differs.
        what: &'static str,
    },
}

/// A seeded xorshift generator: the check needs no engine randomness.
struct Draw(u64);

impl Draw {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        u32::try_from(self.0 % u64::from(n.max(1))).unwrap_or(0)
    }
}

/// Checks `E`'s `head_active` declaration against its step (ADR-0086
/// decision 6).
///
/// For every record in `roots`, every acting agent and every legal
/// canonical joint of that agent, it steps the joint (the other acting
/// agents playing their first legal canonical joint) and `copies` copies
/// in which every agent's inactive heads take random values, and checks
/// that each copy's record hash, rewards, agent and episode flags,
/// observations and masks equal those of the canonical step.
///
/// # Errors
///
/// [`HeadActivityError::Differs`] for the first copy that differs; an
/// action space search does not take; an engine error.
pub fn check_head_activity<E>(
    cfg: &E::Config,
    roots: &[Record<E>],
    copies: NonZeroU32,
    seed: u64,
) -> Result<HeadActivityReport, HeadActivityError>
where
    E: ActionMask<
            Config: Clone + Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
{
    let joint = joint_heads::<E, Masked>(cfg)?;
    let heads = joint.heads();
    let agents = E::MAX_AGENTS;
    let rows = copies.get().saturating_add(1);
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        illegal_action: IllegalActionPolicy::Noop,
        ..PoolOptions::new(
            NonZeroU32::new(rows).unwrap_or(NonZeroU32::MIN),
            Seed::new(0),
        )
    };
    let mut pool: EnvPool<E, Masked> = EnvPool::with_masks(cfg.clone(), options)?;
    let rows = usize::try_from(rows).unwrap_or(usize::MAX);
    let mut draw = Draw(seed | 1);
    let mut report = HeadActivityReport::default();
    let mut actions = vec![0i32; rows.saturating_mul(agents).saturating_mul(heads)];
    let mut base = vec![0u32; agents];
    let mut legal = vec![false; usize::try_from(joint.joint()).unwrap_or(0)];
    let width = joint.mask_len();
    for (r, root) in roots.iter().enumerate() {
        let acting: Vec<bool> = root
            .header()
            .agents()
            .iter()
            .map(|a| a.acting.get())
            .collect();
        let records = vec![*root; rows];
        pool.restore_all(&records)?;
        // Each acting agent's legal canonical joints at the root.
        let masks = pool.outputs()?.masks().to_vec();
        let mut legal_of = vec![Vec::new(); agents];
        for (k, list) in legal_of.iter_mut().enumerate() {
            if !acting.get(k).copied().unwrap_or(false) {
                continue;
            }
            crate::pool::fill_legal(&joint, &masks, Some(k * width), &mut legal);
            list.extend(
                (0..joint.joint()).filter(|&j| legal.get(j as usize).copied().unwrap_or(false)),
            );
            if let (Some(b), Some(&first)) = (base.get_mut(k), list.first()) {
                *b = first;
            }
        }
        for k in 0..agents {
            let Some(list) = legal_of.get(k) else {
                continue;
            };
            for &j in list {
                report.cases = report.cases.saturating_add(1);
                if has_inactive(&joint, j) {
                    report.with_inactive = report.with_inactive.saturating_add(1);
                }
                pool.restore_all(&records)?;
                fill_actions(&joint, &acting, &base, k, j, rows, &mut draw, &mut actions);
                pool.step(ActionInput {
                    discrete: &actions,
                    continuous: &[],
                })?;
                if let Some((row, what)) = first_difference(&pool, rows, agents)? {
                    let at = row * agents * heads + k * heads;
                    let copy = actions
                        .get(at..at + heads)
                        .and_then(|v| joint.encode(v))
                        .unwrap_or(u32::MAX);
                    return Err(HeadActivityError::Differs {
                        root: r,
                        agent: k,
                        joint: j,
                        copy,
                        what,
                    });
                }
            }
        }
        report.roots = report.roots.saturating_add(1);
    }
    Ok(report)
}

fn has_inactive(joint: &JointHeads, j: u32) -> bool {
    let mut buf = [0i32; MAX_JOINT_HEADS];
    let Some(v) = buf.get_mut(..joint.heads()) else {
        return false;
    };
    joint.decode(j, v);
    let v = &*v;
    (0..joint.heads()).any(|h| !joint.is_active(h, v))
}

/// Row 0: agent `k` plays `j`, every other acting agent its base joint.
/// Rows 1..: the same with every inactive head of every agent drawn at
/// random.
#[expect(clippy::too_many_arguments, reason = "one internal call site")]
fn fill_actions(
    joint: &JointHeads,
    acting: &[bool],
    base: &[u32],
    k: usize,
    j: u32,
    rows: usize,
    draw: &mut Draw,
    actions: &mut [i32],
) {
    let heads = joint.heads();
    let agents = acting.len();
    actions.fill(0);
    for row in 0..rows {
        for (a, &on) in acting.iter().enumerate() {
            if !on {
                continue;
            }
            let chosen = if a == k {
                j
            } else {
                base.get(a).copied().unwrap_or(0)
            };
            let at = (row * agents + a) * heads;
            let Some(v) = actions.get_mut(at..at + heads) else {
                continue;
            };
            joint.decode(chosen, v);
            if row == 0 {
                continue;
            }
            for h in 0..heads {
                if joint.is_active(h, v) {
                    continue;
                }
                let n = joint.head_size(h).unwrap_or(1);
                if let Some(x) = v.get_mut(h) {
                    *x = i32::try_from(draw.below(n)).unwrap_or(0);
                }
            }
        }
    }
}

/// The first copy row whose outputs differ from row 0, and what differs.
fn first_difference<E, M: MaskProvider<E>>(
    pool: &EnvPool<E, M>,
    rows: usize,
    agents: usize,
) -> Result<Option<(usize, &'static str)>, EngineError>
where
    E: Environment<
            Config: Sync,
            State: Send,
            Header: Send + RecordLayout<Adaptive: AdaptiveValues<E::Config>>,
        >,
{
    let hash = |row: usize| -> Result<StateHash, EngineError> {
        pool.state_hash(u32::try_from(row).unwrap_or(u32::MAX))
    };
    let out = pool.outputs()?;
    let same = |len: usize, row: usize, eq: &dyn Fn(usize, usize) -> bool| {
        (0..len).all(|i| eq(i, row * len + i))
    };
    let first = hash(0)?;
    for row in 1..rows {
        if hash(row)? != first {
            return Ok(Some((row, "the record")));
        }
        let rewards = out.rewards();
        if !same(agents, row, &|a, b| {
            rewards.get(a).map(|x| x.to_bits()) == rewards.get(b).map(|x| x.to_bits())
        }) {
            return Ok(Some((row, "the rewards")));
        }
        for (what, flags) in [
            ("agent terminated", out.agent_terminated()),
            ("agent truncated", out.agent_truncated()),
            ("alive", out.alive()),
            ("acting", out.acting()),
        ] {
            if !same(agents, row, &|a, b| {
                flags.get(a).map(|f| f.get()) == flags.get(b).map(|f| f.get())
            }) {
                return Ok(Some((row, what)));
            }
        }
        for (what, flags) in [
            ("episode terminated", out.episode_terminated()),
            ("episode truncated", out.episode_truncated()),
        ] {
            if flags.first().map(|f| f.get()) != flags.get(row).map(|f| f.get()) {
                return Ok(Some((row, what)));
            }
        }
        let masks = out.masks();
        let m = masks.len() / rows.max(1);
        if !same(m, row, &|a, b| {
            masks.get(a).map(|f| f.get()) == masks.get(b).map(|f| f.get())
        }) {
            return Ok(Some((row, "the masks")));
        }
        for leaf in 0..tessera_rl_engine::MAX_LEAVES {
            let Some(obs) = out.observation(leaf) else {
                break;
            };
            let n = obs.len() / rows.max(1);
            if obs.get(..n) != obs.get(row * n..(row + 1) * n) {
                return Ok(Some((row, "the observations")));
            }
        }
    }
    Ok(None)
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check_continuation` and `check_snapshot_continuation` (#351, SPEC
//! 10.2): an episode continued from a reseed, or from a restored snapshot,
//! is the episode a fresh run gives.

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::{HostEnvironment, HostSnapshot};
use tessera_rl_num::identity::Seed;

use super::super::HostPool;
use super::super::record::HostRecord;
use super::actions::Sampler;
use super::error::CheckError;
use super::hash::{Cover, FieldHashes};
use super::run::{CheckRun, build};
use crate::pool::Reseed;
use crate::validate::ActionInput;

/// What a continuation check saw.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContinuationReport {
    /// Steps run before the reseed or the snapshot.
    pub at: usize,
    /// One hash per compared call (see `DeterminismReport::call_hashes`).
    pub call_hashes: Vec<u64>,
}

/// The seed the continued pool is reseeded with.
const RESEED: u64 = 0xC0DE_5EED_0000_0351;

/// Checks that a reseed starts the same episode as a fresh pool.
///
/// Pool `a` resets, steps `at` times, then resets every instance with a
/// new base seed and steps `run.calls` more times. Pool `b` is built with
/// that seed, resets and steps `run.calls` times with the same seeded
/// actions. Every output but the slot generations (which count spawns over
/// the pool's life) must agree after every call from the reseed on: an
/// environment whose `reset` keeps anything from the previous episode
/// fails. For environments without snapshots this is the continuation
/// check; with snapshots, also run [`check_snapshot_continuation`].
///
/// # Errors
///
/// [`CheckError::Diverged`] at the first call and output that differ, or
/// [`CheckError::Engine`] for a failed pool call.
pub fn check_continuation<E: HostEnvironment, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
    at: usize,
) -> Result<ContinuationReport, CheckError> {
    let seed = run.seed ^ RESEED;
    let mut a = build::<E, A>(cfg, run, run.seed)?;
    let mut warm = Sampler::new(run.action_seed.rotate_left(17));
    a.reset_all(Reseed::Keep)?;
    for _ in 0..at {
        let input = warm.fill(a.layout(), a.outputs()?);
        a.step(input)?;
    }
    a.reset_all(Reseed::Base(Seed::new(seed)))?;
    let mut b = build::<E, A>(cfg, run, seed)?;
    b.reset_all(Reseed::Keep)?;
    let mut call_hashes = Vec::with_capacity(run.calls.saturating_add(1));
    call_hashes.push(same(0, &a, &b, Cover::Episode, "a fresh pool")?);
    let (mut sa, mut sb) = (Sampler::new(run.action_seed), Sampler::new(run.action_seed));
    for call in 1..=run.calls {
        let input = sa.fill(a.layout(), a.outputs()?);
        a.step(input)?;
        let input = sb.fill(b.layout(), b.outputs()?);
        b.step(input)?;
        call_hashes.push(same(call, &a, &b, Cover::Episode, "a fresh pool")?);
    }
    Ok(ContinuationReport { at, call_hashes })
}

/// Checks that a restored snapshot continues as the original did.
///
/// One Inline pool resets, steps `at` times and snapshots every instance.
/// It then steps `run.calls` times, recording the actions and the hash of
/// every output after every call; restores the snapshots; replays the
/// recorded actions; and compares every output after every call. Run it
/// next to [`check_snapshot_contract`](super::check_snapshot_contract),
/// which restores into a second pool.
///
/// # Errors
///
/// [`CheckError::Diverged`] at the first call and output that differ, or
/// [`CheckError::Engine`] for a failed pool call, snapshot or restore.
pub fn check_snapshot_continuation<E, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
    at: usize,
) -> Result<ContinuationReport, CheckError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let mut pool = build::<E, A>(cfg, run, run.seed)?;
    let mut sampler = Sampler::new(run.action_seed);
    pool.reset_all(Reseed::Keep)?;
    for _ in 0..at {
        let input = sampler.fill(pool.layout(), pool.outputs()?);
        pool.step(input)?;
    }
    let n = pool.layout().num_envs;
    let mut records: Vec<HostRecord<A>> = (0..n).map(|_| HostRecord::new()).collect();
    pool.snapshot_all(&mut records)?;
    let mut script: Vec<(Vec<i32>, Vec<f32>)> = Vec::with_capacity(run.calls);
    let mut first: Vec<FieldHashes> = Vec::with_capacity(run.calls);
    for _ in 0..run.calls {
        let input = sampler.fill(pool.layout(), pool.outputs()?);
        script.push((input.discrete.to_vec(), input.continuous.to_vec()));
        pool.step(input)?;
        first.push(FieldHashes::of(pool.outputs()?, Cover::All));
    }
    pool.restore_all(&records)?;
    let mut call_hashes = Vec::with_capacity(run.calls);
    for (k, ((discrete, continuous), want)) in script.iter().zip(&first).enumerate() {
        pool.step(ActionInput {
            discrete,
            continuous,
        })?;
        let got = FieldHashes::of(pool.outputs()?, Cover::All);
        if let Some(what) = want.first_difference(&got) {
            return Err(CheckError::Diverged {
                call: at.saturating_add(k).saturating_add(1),
                left: "the first run",
                right: "the run from the restored snapshot",
                what,
            });
        }
        call_hashes.push(got.total());
    }
    Ok(ContinuationReport { at, call_hashes })
}

/// The call's hash, if the pools agree on the covered fields.
fn same<E: HostEnvironment, const A: usize>(
    call: usize,
    a: &HostPool<E, A>,
    b: &HostPool<E, A>,
    cover: Cover,
    right: &'static str,
) -> Result<u64, CheckError> {
    let ha = FieldHashes::of(a.outputs()?, cover);
    let hb = FieldHashes::of(b.outputs()?, cover);
    match ha.first_difference(&hb) {
        Some(what) => Err(CheckError::Diverged {
            call,
            left: "the reseeded pool",
            right,
            what,
        }),
        None => Ok(ha.total()),
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check_determinism` (#351, SPEC 10.2): equal seeds and actions give
//! equal outputs, Inline and with workers.

use tessera_rl_core::HostEnvironment;

use super::super::HostPool;
use super::actions::Sampler;
use super::error::CheckError;
use super::hash::{Cover, FieldHashes};
use super::run::{CheckRun, build};
use crate::options::SyncOptions;
use crate::pool::Reseed;

/// What [`check_determinism`] saw.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeterminismReport {
    /// One hash per call, the first reset first: of every output of every
    /// instance (observations, terminal observations, rewards, the agent
    /// events as alive flags and slot generations, the agent and episode
    /// done rows, aborts and pending resets, episode statistics, masks
    /// and refusals). Stable across platforms: a crate can commit them as
    /// goldens.
    pub call_hashes: Vec<u64>,
    /// Workers of the Sync-mode pool compared.
    pub workers: usize,
}

/// Checks that `E` is deterministic on `run`.
///
/// Builds three pools from `cfg()` with the same seed: two Inline and one
/// in Sync mode with `run.workers` workers. Resets them, then steps them
/// `run.calls` times with the same seeded legal actions (drawn from the
/// first pool's masks), and after every call compares the hash of every
/// output field (see [`DeterminismReport::call_hashes`]): the two Inline
/// pools catch state that leaks between instances or runs (a global, the
/// clock, an unseeded RNG), and the Sync pool catches state that depends
/// on the thread or the chunk.
///
/// # Errors
///
/// [`CheckError::Diverged`] naming the first call and output that differ,
/// or [`CheckError::Engine`] for a failed pool call.
pub fn check_determinism<E: HostEnvironment, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
) -> Result<DeterminismReport, CheckError> {
    let mut a = build::<E, A>(cfg, run, run.seed)?;
    let mut b = build::<E, A>(cfg, run, run.seed)?;
    let mut w = build::<E, A>(cfg, run, run.seed)?.with_workers(SyncOptions::new(run.workers))?;
    let mut sampler = Sampler::new(run.action_seed);
    let mut call_hashes = Vec::with_capacity(run.calls.saturating_add(1));
    a.reset_all(Reseed::Keep)?;
    b.reset_all(Reseed::Keep)?;
    w.reset_all(Reseed::Keep)?;
    call_hashes.push(agree(0, &a, &b, &w)?);
    for call in 1..=run.calls {
        let input = sampler.fill(a.layout(), a.outputs()?);
        a.step(input)?;
        b.step(input)?;
        w.step(input)?;
        call_hashes.push(agree(call, &a, &b, &w)?);
    }
    Ok(DeterminismReport {
        call_hashes,
        workers: w.workers(),
    })
}

/// The call's hash, if the three pools agree on every field.
fn agree<E: HostEnvironment, const A: usize>(
    call: usize,
    a: &HostPool<E, A>,
    b: &HostPool<E, A>,
    w: &HostPool<E, A>,
) -> Result<u64, CheckError> {
    let ha = FieldHashes::of(a.outputs()?, Cover::All);
    let hb = FieldHashes::of(b.outputs()?, Cover::All);
    let hw = FieldHashes::of(w.outputs()?, Cover::All);
    if let Some(what) = ha.first_difference(&hb) {
        return Err(CheckError::Diverged {
            call,
            left: "one Inline run",
            right: "another Inline run",
            what,
        });
    }
    if let Some(what) = ha.first_difference(&hw) {
        return Err(CheckError::Diverged {
            call,
            left: "Inline",
            right: "Sync with workers",
            what,
        });
    }
    Ok(ha.total())
}

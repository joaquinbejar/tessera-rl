// SPDX-License-Identifier: MIT OR Apache-2.0

//! A check of the `HostSnapshot` contract on an author's own trajectories
//! (ADR-0073 item 8), for an environment crate's tests.
//!
//! The engine cannot prove that `save` is canonical or that `load` restores
//! everything that matters; it can test both on the actions an author
//! supplies. [`check_snapshot_contract`] runs them in one pool, snapshots
//! every instance at a midpoint, restores those records into a second pool
//! built with another seed, and replays the rest of the actions in both.
//!
//! The checks `cargo tessera check` runs (#351, SPEC 10.2) live in the
//! submodules: [`check_determinism`] (Inline and with workers, every output
//! hashed per call), [`check_continuation`] and
//! [`check_snapshot_continuation`], [`check_spaces`] and
//! [`check_invariants`]. Each builds its pools from a configuration factory
//! and a [`CheckRun`], drives them with seeded legal actions, and returns a
//! report or the first [`CheckError`].

mod actions;
mod continuation;
mod determinism;
mod error;
mod hash;
mod invariants;
mod run;
mod spaces;

pub use continuation::{ContinuationReport, check_continuation, check_snapshot_continuation};
pub use determinism::{DeterminismReport, check_determinism};
pub use error::CheckError;
pub use invariants::{InvariantsReport, check_invariants};
pub use run::CheckRun;
pub use spaces::{SpacesReport, check_spaces};

use tessera_rl_core::HostSnapshot;
use tessera_rl_core::config::CanonicalConfig;

use super::HostPool;
use super::record::HostRecord;
use crate::error::EngineError;
use crate::options::PoolOptions;
use crate::pool::Reseed;
use crate::validate::ActionInput;

/// One call's actions: discrete and continuous, as [`ActionInput`].
pub type ContractStep = (Vec<i32>, Vec<f32>);

/// How the contract failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ContractError {
    /// A pool call failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A restored instance saves other bytes than the record it was
    /// restored from: `save` is not canonical, or `load` dropped state.
    #[error("instance {instance}: save after load differs from the restored record")]
    NotCanonical {
        /// The instance.
        instance: u32,
    },
    /// After restoring, the same actions gave another state or output.
    #[error("instance {instance} diverged {step} steps after the restore: {what}")]
    Diverged {
        /// Steps after the restore.
        step: usize,
        /// The instance.
        instance: u32,
        /// What differed.
        what: &'static str,
    },
}

/// Checks the `HostSnapshot` contract of `E` on `steps` (ADR-0073 item 8).
///
/// Builds a pool from `cfg()` and `options`, resets it, runs the first half
/// of `steps` and snapshots every instance. Then it builds a second pool
/// from `cfg()` with the seed changed, restores the records into it, and
/// checks:
///
/// - each restored instance saves exactly the bytes it was restored from;
/// - running the rest of `steps` in both pools gives the same state hash,
///   rewards and observations for every instance after every call.
///
/// # Errors
///
/// The first [`ContractError`]: a pool error (including the environment's
/// refusal of its own record), bytes that are not canonical, or a
/// divergence.
pub fn check_snapshot_contract<E, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    options: PoolOptions,
    steps: &[ContractStep],
) -> Result<(), ContractError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let mut other = options.clone();
    other.seed = tessera_rl_num::identity::Seed::new(options.seed.get() ^ 0x5EED_5EED);
    let mut a = HostPool::<E, A>::new(cfg(), options)?;
    let mut b = HostPool::<E, A>::new(cfg(), other)?;
    a.reset_all(Reseed::Keep)?;
    b.reset_all(Reseed::Keep)?;
    let (before, after) = steps.split_at(steps.len() / 2);
    for (discrete, continuous) in before {
        a.step(ActionInput {
            discrete,
            continuous,
        })?;
    }
    let n = a.headers.len();
    let mut records: Vec<HostRecord<A>> = (0..n).map(|_| HostRecord::new()).collect();
    a.snapshot_all(&mut records)?;
    b.restore_all(&records)?;
    let mut again: Vec<HostRecord<A>> = (0..n).map(|_| HostRecord::new()).collect();
    b.snapshot_all(&mut again)?;
    for (k, (r, s)) in records.iter().zip(&again).enumerate() {
        if r.state_bytes() != s.state_bytes() {
            return Err(ContractError::NotCanonical {
                instance: u32::try_from(k).unwrap_or(u32::MAX),
            });
        }
    }
    for (step, (discrete, continuous)) in after.iter().enumerate() {
        let input = ActionInput {
            discrete,
            continuous,
        };
        a.step(input)?;
        b.step(input)?;
        compare(&mut a, &mut b, step + 1)?;
    }
    Ok(())
}

/// Compares every instance's outputs and state hash after one call.
fn compare<E, const A: usize>(
    a: &mut HostPool<E, A>,
    b: &mut HostPool<E, A>,
    step: usize,
) -> Result<(), ContractError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let n = a.headers.len();
    for i in 0..n {
        let instance = u32::try_from(i).unwrap_or(u32::MAX);
        let diverged = |what| ContractError::Diverged {
            step,
            instance,
            what,
        };
        let (oa, ob) = (a.outputs()?, b.outputs()?);
        let rows = i * A..(i + 1) * A;
        if oa.rewards().get(rows.clone()) != ob.rewards().get(rows) {
            return Err(diverged("rewards"));
        }
        let mut leaf = 0;
        while let (Some(la), Some(lb)) = (oa.observation(leaf), ob.observation(leaf)) {
            let per = la.len() / n.max(1);
            if la.get(i * per..(i + 1) * per) != lb.get(i * per..(i + 1) * per) {
                return Err(diverged("observations"));
            }
            leaf += 1;
        }
        if a.state_hash(instance)? != b.state_hash(instance)? {
            return Err(diverged("state hash"));
        }
    }
    Ok(())
}

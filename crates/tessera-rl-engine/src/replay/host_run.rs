// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replaying a host pool's `from_reset_only` file (ADR-0072 section 5,
//! ADR-0069 item 8, ADR-0070 item 9).

use tessera_rl_core::{HostEnvironment, Tier};

use crate::error::EngineError;
use crate::host::HostPool;
use crate::persist::{BUILD_ID, PersistError, host_identity};
use crate::validate::ActionInput;

use super::outputs_hash::{outputs_hash, refusal_rows};
use super::reader::{Event, Ids, RefusalRow, Replay};
use super::run::ReplaySummary;
use super::{DivergenceKind, ReplayError};

const fn diverged(event_index: u64, kind: DivergenceKind, instance: Option<u32>) -> ReplayError {
    ReplayError::Diverged {
        event_index,
        kind,
        instance,
        last_matching_checkpoint: None,
    }
}

/// How a re-issued call ended.
enum Outcome {
    /// Accepted: compare its outputs.
    Ok,
    /// It poisoned the pool: the next event must be `Poisoned`.
    Poisoned,
}

impl Replay<'_> {
    /// Replays a host pool's file into `pool` (ADR-0072 section 5). The file
    /// starts from a reset of every instance, so `pool` may be fresh. Every
    /// call is re-issued through the pool's public methods: each `Reset` and
    /// `Step` must succeed with the recorded outputs hash and, for a
    /// refusing host, the recorded refusal rows; each `Rejected` event is
    /// re-submitted through `step` and must be rejected with the same
    /// instance and code, reaching `on_rejected` through the engine as live
    /// (replay never calls it directly); `Poisoned` must follow a call that
    /// poisons.
    ///
    /// The pool is the caller's, checked against the header: environment id
    /// and version, options, instance count, action layout and the host
    /// flags. A host config is not canonical, so it is not compared; a pool
    /// built with another config shows up as a divergence. A caller-built
    /// pool is a staged step; rebuilding from the header is issue #377. The
    /// whole file is held in memory, not streamed (ADR-0072 section 8; a
    /// known gap).
    ///
    /// There are no checkpoints, so `last_matching_checkpoint` is always
    /// `None` in a divergence.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] if the pool does not match the header,
    /// [`ReplayError::Diverged`] at the first failing check, and other typed
    /// errors for a malformed event.
    pub fn run_host<E: HostEnvironment, const A: usize>(
        &self,
        pool: &mut HostPool<E, A>,
    ) -> Result<ReplaySummary, ReplayError> {
        self.check_host_pool(pool)?;
        let n = usize::try_from(self.header.num_envs).map_err(|_| ReplayError::Allocation)?;
        let mut summary = ReplaySummary::default();
        let mut awaiting_poison: Option<u64> = None;
        let mut last_call: Option<u64> = None;
        for item in self.events() {
            let (index, event) = item?;
            let malformed = |field| ReplayError::Malformed {
                event_index: index,
                field,
            };
            if summary.poisoned {
                return Err(malformed("an event after Poisoned"));
            }
            if let Some(call) = awaiting_poison.take() {
                if event != Event::Poisoned {
                    return Err(diverged(call, DivergenceKind::UnexpectedPoison, None));
                }
                summary.poisoned = true;
                summary.events = index.saturating_add(1);
                continue;
            }
            let outcome = match event {
                Event::Reset {
                    ids,
                    reseed,
                    outputs_hash: recorded,
                } => {
                    let recorded = recorded.ok_or_else(|| malformed("outputs hash"))?;
                    let result = match &ids {
                        Ids::All => pool.reset_all(reseed.as_reseed()).map(|_| ()),
                        Ids::List(ids) => pool.reset(ids, reseed.as_reseed()).map(|_| ()),
                    };
                    let outcome = outcome(pool, index, &result)?;
                    if let Outcome::Ok = outcome {
                        check_reset(pool, n, index, &ids, recorded)?;
                    }
                    outcome
                }
                Event::Step {
                    ids,
                    discrete,
                    continuous_bits,
                    outputs_hash: recorded,
                    refusals,
                } => {
                    if ids != Ids::All {
                        return Err(ReplayError::Unsupported {
                            feature: "subset steps",
                        });
                    }
                    let recorded = recorded.ok_or_else(|| malformed("outputs hash"))?;
                    let result = step(pool, &discrete, &continuous_bits);
                    let outcome = outcome(pool, index, &result)?;
                    if let Outcome::Ok = outcome {
                        check_step(pool, n, index, recorded, &refusals)?;
                    }
                    outcome
                }
                Event::Rejected {
                    discrete,
                    continuous_bits,
                    instance,
                    code,
                } => rejected(pool, index, &discrete, &continuous_bits, instance, code)?,
                Event::Poisoned => {
                    return Err(match last_call {
                        Some(call) => diverged(call, DivergenceKind::NotPoisoned, None),
                        None => malformed("Poisoned before any call"),
                    });
                }
                Event::StateWrite { .. } | Event::Keyframe { .. } | Event::Checkpoint { .. } => {
                    return Err(ReplayError::Unsupported {
                        feature: "keyframes, checkpoints and state writes in host replay files",
                    });
                }
            };
            summary.calls = summary.calls.saturating_add(1);
            last_call = Some(index);
            if let Outcome::Poisoned = outcome {
                awaiting_poison = Some(index);
            }
            summary.events = index.saturating_add(1);
        }
        if let Some(call) = awaiting_poison {
            return Err(diverged(call, DivergenceKind::UnexpectedPoison, None));
        }
        Ok(summary)
    }

    fn check_host_pool<E: HostEnvironment, const A: usize>(
        &self,
        pool: &HostPool<E, A>,
    ) -> Result<(), ReplayError> {
        let incompatible = |field: &'static str| -> Result<(), ReplayError> {
            Err(PersistError::Incompatible { field }.into())
        };
        let Some(host) = self.header.host else {
            return incompatible("host_protocol");
        };
        let id = host_identity(E::ENV_ID, E::ENV_VERSION, E::MASKS, pool.options());
        self.file
            .check_against(&id, Tier::CrossPlatform, BUILD_ID)?;
        if self.header.num_envs != pool.options().num_envs.get() {
            return incompatible("instance_count");
        }
        if !self.header.config.is_empty() {
            return incompatible("config");
        }
        let (discrete, continuous) = pool.action_heads();
        let same = |n: u32, m: usize| usize::try_from(n).is_ok_and(|n| n == m);
        if !same(self.header.agents, A)
            || !same(self.header.discrete_heads, discrete)
            || !same(self.header.continuous_heads, continuous)
        {
            return incompatible("action_layout");
        }
        if host.notifies != E::NOTIFIES_REJECTION {
            return incompatible("notifies_rejection");
        }
        if host.refuses != E::REFUSES_AGENTS {
            return incompatible("refuses_agents");
        }
        Ok(())
    }
}

/// Re-submits a recorded joint action through `step`.
fn step<E: HostEnvironment, const A: usize>(
    pool: &mut HostPool<E, A>,
    discrete: &[i32],
    continuous_bits: &[u32],
) -> Result<(), EngineError> {
    let continuous: Vec<f32> = continuous_bits.iter().map(|&b| f32::from_bits(b)).collect();
    pool.step(ActionInput {
        discrete,
        continuous: &continuous,
    })
    .map(|_| ())
}

/// Compares an accepted reset's outputs hash, over the reset ids in order.
fn check_reset<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    n: usize,
    index: u64,
    ids: &Ids,
    recorded: u64,
) -> Result<(), ReplayError> {
    let out = pool.outputs()?;
    let hash = match ids {
        Ids::All => outputs_hash(out, n, 0..n, E::REFUSES_AGENTS),
        Ids::List(ids) => outputs_hash(
            out,
            n,
            ids.iter().filter_map(|&id| usize::try_from(id).ok()),
            E::REFUSES_AGENTS,
        ),
    };
    if hash == recorded {
        Ok(())
    } else {
        Err(diverged(index, DivergenceKind::OutputsMismatch, None))
    }
}

/// Compares an accepted step's refusal rows (refusing hosts) and outputs
/// hash with the recorded ones.
fn check_step<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    n: usize,
    index: u64,
    recorded: u64,
    refusals: &[RefusalRow],
) -> Result<(), ReplayError> {
    let out = pool.outputs()?;
    if E::REFUSES_AGENTS {
        let replayed = refusal_rows(out, n);
        if replayed != refusals {
            let first = replayed
                .iter()
                .zip(refusals)
                .find(|(a, b)| a != b)
                .map(|(a, _)| a)
                .or_else(|| replayed.get(refusals.len()))
                .or_else(|| refusals.get(replayed.len()));
            return Err(diverged(
                index,
                DivergenceKind::RefusalMismatch,
                first.map(|r| r.instance),
            ));
        }
    }
    if outputs_hash(out, n, 0..n, E::REFUSES_AGENTS) != recorded {
        return Err(diverged(index, DivergenceKind::OutputsMismatch, None));
    }
    Ok(())
}

/// A re-issued call of a recorded accepted call: accepted, poisoned, or
/// diverged (a rejection is [`DivergenceKind::RejectionMismatch`], any other
/// refusal [`DivergenceKind::CallFailed`]).
fn outcome<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    index: u64,
    result: &Result<(), EngineError>,
) -> Result<Outcome, ReplayError> {
    match result {
        Ok(()) => Ok(Outcome::Ok),
        Err(_) if pool.is_poisoned() => Ok(Outcome::Poisoned),
        Err(EngineError::ActionRejected { instance, .. }) => Err(diverged(
            index,
            DivergenceKind::RejectionMismatch,
            Some(*instance),
        )),
        Err(_) => Err(diverged(index, DivergenceKind::CallFailed, None)),
    }
}

/// Re-submits a recorded rejected call (ADR-0069 item 8): it must be rejected
/// again, through `step`, with the same instance and code.
fn rejected<E: HostEnvironment, const A: usize>(
    pool: &mut HostPool<E, A>,
    index: u64,
    discrete: &[i32],
    continuous_bits: &[u32],
    instance: u32,
    code: u32,
) -> Result<Outcome, ReplayError> {
    match step(pool, discrete, continuous_bits) {
        Err(EngineError::ActionRejected {
            instance: i,
            code: c,
            ..
        }) if i == instance && c == code && !pool.is_poisoned() => Ok(Outcome::Ok),
        Err(_) if pool.is_poisoned() => Ok(Outcome::Poisoned),
        _ => Err(diverged(
            index,
            DivergenceKind::RejectionMismatch,
            Some(instance),
        )),
    }
}

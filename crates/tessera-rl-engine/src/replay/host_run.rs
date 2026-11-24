// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replaying a host pool's file (ADR-0072 section 5, ADR-0069 item 8,
//! ADR-0070 item 9, ADR-0073).

use tessera_rl_core::config::{CanonicalConfig, canonical_bytes, config_hash};
use tessera_rl_core::{HostEnvironment, HostSnapshot, StateHash, Tier};

use crate::error::EngineError;
use crate::host::{HostPool, HostRecord};
use crate::persist::{BUILD_ID, PersistError, host_identity};
use crate::validate::ActionInput;

use super::format::StateWriteOp;
use super::outputs_hash::{outputs_hash, refusal_rows};
use super::reader::{Event, Ids, RefusalRow, Replay};
use super::run::ReplaySummary;
use super::{DivergenceKind, ReplayError};

/// A pool's canonical config bytes and hash.
type ConfigFn<E, const A: usize> = fn(&HostPool<E, A>) -> Result<(Vec<u8>, u64), EngineError>;
/// `HostPool::import`.
type ImportFn<E, const A: usize> =
    fn(&mut HostPool<E, A>, &[u32], &[HostRecord<A>]) -> Result<(), EngineError>;

/// The snapshot operations replay needs from a host with `HostSnapshot`, as
/// function pointers, so one loop serves both kinds of file.
struct RunOps<E: HostEnvironment, const A: usize> {
    identity: fn(&mut HostPool<E, A>) -> Result<u64, EngineError>,
    max_len: fn(&mut HostPool<E, A>) -> Result<usize, EngineError>,
    config: ConfigFn<E, A>,
    hash: fn(&mut HostPool<E, A>, u32) -> Result<StateHash, EngineError>,
    import: ImportFn<E, A>,
    restore_all: fn(&mut HostPool<E, A>, &[HostRecord<A>]) -> Result<(), EngineError>,
}

impl<E: HostEnvironment, const A: usize> Clone for RunOps<E, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: HostEnvironment, const A: usize> Copy for RunOps<E, A> {}

/// The canonical config bytes and hash of a snapshot host's pool.
fn config_of<E, const A: usize>(pool: &HostPool<E, A>) -> Result<(Vec<u8>, u64), EngineError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let cfg = pool.config();
    Ok((
        canonical_bytes(cfg),
        config_hash(E::ENV_ID, E::ENV_VERSION, cfg)?,
    ))
}

impl<E, const A: usize> RunOps<E, A>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    fn of() -> Self {
        Self {
            identity: HostPool::<E, A>::snapshot_identity,
            max_len: HostPool::<E, A>::max_snapshot_len,
            config: config_of::<E, A>,
            hash: HostPool::<E, A>::state_hash,
            import: HostPool::<E, A>::import,
            restore_all: HostPool::<E, A>::restore_all,
        }
    }
}

const fn diverged(
    event_index: u64,
    kind: DivergenceKind,
    instance: Option<u32>,
    last_matching_checkpoint: Option<u64>,
) -> ReplayError {
    ReplayError::Diverged {
        event_index,
        kind,
        instance,
        last_matching_checkpoint,
    }
}

/// How a re-issued call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Accepted.
    Ok,
    /// It poisoned the pool: the next event must be `Poisoned`.
    Poisoned,
}

/// Where the replay stands between events.
struct Run {
    n: usize,
    summary: ReplaySummary,
    awaiting_poison: Option<u64>,
    last_call: Option<u64>,
    last_checkpoint: Option<u64>,
    /// The ids of a state write whose keyframe comes next.
    write: Option<(u64, Ids)>,
    /// Most author bytes of one record: the pool's `max_snapshot_len`, capped
    /// by `DecodeLimits::max_host_record_bytes`.
    max_record: usize,
}

impl Run {
    const fn diverged(
        &self,
        index: u64,
        kind: DivergenceKind,
        instance: Option<u32>,
    ) -> ReplayError {
        diverged(index, kind, instance, self.last_checkpoint)
    }
}

impl Replay<'_> {
    /// Replays a host pool's `from_reset_only` file into `pool` (ADR-0072
    /// section 5). The file starts from a reset of every instance, so `pool`
    /// may be fresh. Every call is re-issued through the pool's public
    /// methods: each `Reset` and `Step` must succeed with the recorded
    /// outputs hash and, for a refusing host, the recorded refusal rows;
    /// each `Rejected` event is re-submitted through `step` and must be
    /// rejected with the same instance and code, reaching `on_rejected`
    /// through the engine as live (replay never calls it directly);
    /// `Poisoned` must follow a call that poisons. A file with snapshots
    /// needs [`Replay::run_host_snapshots`].
    ///
    /// The pool is the caller's, checked against the header: environment id
    /// and version, options, instance count, action layout and the host
    /// flags. The config of a `from_reset_only` file is not in it (host
    /// configs need not be canonical), so a pool built with another config
    /// shows up as a divergence. A caller-built pool is a staged step;
    /// rebuilding from the header is issue #377. The whole file is held in
    /// memory, not streamed (ADR-0072 section 8; a known gap).
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] if the pool does not match the header (a file
    /// with snapshots included), [`ReplayError::Diverged`] at the first
    /// failing check, and other typed errors for a malformed event.
    pub fn run_host<E: HostEnvironment, const A: usize>(
        &self,
        pool: &mut HostPool<E, A>,
    ) -> Result<ReplaySummary, ReplayError> {
        if self.header.host.is_some_and(|h| h.identity().is_some()) {
            return Err(PersistError::Incompatible {
                field: "host_snapshots: use run_host_snapshots",
            }
            .into());
        }
        self.run_host_inner(pool, None)
    }

    /// Replays a host pool's file of either kind into `pool`, for a host
    /// with snapshots (ADR-0073). A file with snapshots is checked at open
    /// against the pool's identity digest (`Incompatible { field:
    /// "identity" }`) and canonical config (`config_hash`, `config`); its
    /// first keyframe is restored through `restore_all` (which runs
    /// `check_host_record`), state-write keyframes go through `import`,
    /// periodic keyframes are compared by hash, and checkpoints by the host
    /// `StateHash`, naming the last matching checkpoint on a divergence.
    /// Calls are checked as in [`Replay::run_host`]. Same staged-step and
    /// in-memory notes.
    ///
    /// # Errors
    ///
    /// As [`Replay::run_host`].
    pub fn run_host_snapshots<E, const A: usize>(
        &self,
        pool: &mut HostPool<E, A>,
    ) -> Result<ReplaySummary, ReplayError>
    where
        E: HostSnapshot,
        E::Config: CanonicalConfig,
    {
        self.run_host_inner(pool, Some(RunOps::of()))
    }

    fn run_host_inner<E: HostEnvironment, const A: usize>(
        &self,
        pool: &mut HostPool<E, A>,
        ops: Option<RunOps<E, A>>,
    ) -> Result<ReplaySummary, ReplayError> {
        let max_record = self.check_host_pool(pool, ops)?;
        let mut run = Run {
            n: usize::try_from(self.header.num_envs).map_err(|_| ReplayError::Allocation)?,
            summary: ReplaySummary::default(),
            awaiting_poison: None,
            last_call: None,
            last_checkpoint: None,
            write: None,
            max_record,
        };
        for item in self.events() {
            let (index, event) = item?;
            if run.summary.poisoned || run.summary.unverified_poison {
                return Err(ReplayError::Malformed {
                    event_index: index,
                    field: "an event after Poisoned",
                });
            }
            if let Some(call) = run.awaiting_poison.take() {
                if event != Event::Poisoned {
                    return Err(run.diverged(call, DivergenceKind::UnexpectedPoison, None));
                }
                run.summary.poisoned = true;
            } else if let Some((write, _)) = &run.write
                && !matches!(event, Event::Keyframe { .. })
            {
                if event != Event::Poisoned {
                    return Err(ReplayError::Malformed {
                        event_index: *write,
                        field: "a state write without its keyframe",
                    });
                }
                // A state write that poisoned: its source is not re-run.
                run.summary.unverified_poison = true;
                run.write = None;
            } else {
                Self::apply_host(pool, ops, &mut run, index, event)?;
            }
            run.summary.events = index.saturating_add(1);
        }
        if let Some(call) = run.awaiting_poison {
            return Err(run.diverged(call, DivergenceKind::UnexpectedPoison, None));
        }
        if let Some((write, _)) = run.write {
            return Err(ReplayError::Malformed {
                event_index: write,
                field: "a state write without its keyframe",
            });
        }
        Ok(run.summary)
    }

    fn apply_host<E: HostEnvironment, const A: usize>(
        pool: &mut HostPool<E, A>,
        ops: Option<RunOps<E, A>>,
        run: &mut Run,
        index: u64,
        event: Event<'_>,
    ) -> Result<(), ReplayError> {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        let n = run.n;
        let outcome = match event {
            Event::Reset {
                ids,
                reseed,
                outputs_hash: recorded,
            } => {
                let result = match &ids {
                    Ids::All => pool.reset_all(reseed.as_reseed()).map(|_| ()),
                    Ids::List(ids) => pool.reset(ids, reseed.as_reseed()).map(|_| ()),
                };
                let outcome = outcome(pool, run, index, &result)?;
                if let (Outcome::Ok, Some(recorded)) = (outcome, recorded) {
                    check_reset(pool, run, index, &ids, recorded)?;
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
                let result = step(pool, &discrete, &continuous_bits);
                let outcome = outcome(pool, run, index, &result)?;
                if outcome == Outcome::Ok {
                    check_step(pool, run, index, recorded, &refusals)?;
                }
                outcome
            }
            Event::Rejected {
                discrete,
                continuous_bits,
                instance,
                code,
            } => match step(pool, &discrete, &continuous_bits) {
                Err(EngineError::ActionRejected {
                    instance: i,
                    code: c,
                    ..
                }) if i == instance && c == code && !pool.is_poisoned() => Outcome::Ok,
                Err(_) if pool.is_poisoned() => Outcome::Poisoned,
                _ => {
                    return Err(run.diverged(
                        index,
                        DivergenceKind::RejectionMismatch,
                        Some(instance),
                    ));
                }
            },
            Event::Poisoned => {
                return Err(match run.last_call {
                    Some(call) => run.diverged(call, DivergenceKind::NotPoisoned, None),
                    None => malformed("Poisoned before any call"),
                });
            }
            Event::StateWrite { op, ids, .. } => {
                if op == StateWriteOp::RestoreAll && ids != Ids::All {
                    return Err(malformed("restore_all ids"));
                }
                run.write = Some((index, ids));
                Outcome::Ok
            }
            Event::Keyframe { ids, records, .. } => {
                let ops = ops.ok_or_else(|| malformed("a keyframe without snapshots"))?;
                return Self::host_keyframe(pool, ops, run, index, &ids, records);
            }
            Event::Checkpoint { ids, hashes } => {
                let ops = ops.ok_or_else(|| malformed("a checkpoint without snapshots"))?;
                for (id, hash) in ids_of(&ids, n).into_iter().zip(hashes) {
                    if (ops.hash)(pool, id)? != hash {
                        return Err(run.diverged(index, DivergenceKind::Checkpoint, Some(id)));
                    }
                }
                run.last_checkpoint = Some(index);
                run.summary.checkpoints = run.summary.checkpoints.saturating_add(1);
                return Ok(());
            }
        };
        run.summary.calls = run.summary.calls.saturating_add(1);
        run.last_call = Some(index);
        if outcome == Outcome::Poisoned {
            run.awaiting_poison = Some(index);
        }
        Ok(())
    }

    /// The first keyframe is restored with `restore_all`; a state write's
    /// keyframe with `import`; a periodic keyframe is compared by hash.
    fn host_keyframe<E: HostEnvironment, const A: usize>(
        pool: &mut HostPool<E, A>,
        ops: RunOps<E, A>,
        run: &mut Run,
        index: u64,
        ids: &Ids,
        bytes: &[u8],
    ) -> Result<(), ReplayError> {
        let records = decode_records::<A>(bytes, run.max_record, index)?;
        let id_list = ids_of(ids, run.n);
        if records.len() != id_list.len() {
            return Err(ReplayError::Malformed {
                event_index: index,
                field: "host records",
            });
        }
        run.summary.keyframes = run.summary.keyframes.saturating_add(1);
        if index == 0 {
            if *ids != Ids::All {
                return Err(ReplayError::Malformed {
                    event_index: index,
                    field: "first keyframe ids",
                });
            }
            (ops.restore_all)(pool, &records)
                .map_err(|_| run.diverged(index, DivergenceKind::KeyframeRejected, None))?;
            return Ok(());
        }
        if let Some((write, written)) = run.write.take() {
            if written != *ids {
                return Err(ReplayError::Malformed {
                    event_index: index,
                    field: "a keyframe that differs from its state write",
                });
            }
            (ops.import)(pool, &id_list, &records)
                .map_err(|_| run.diverged(write, DivergenceKind::KeyframeRejected, None))?;
            return Ok(());
        }
        for (id, record) in id_list.into_iter().zip(&records) {
            if (ops.hash)(pool, id)? != record.hash() {
                return Err(run.diverged(index, DivergenceKind::Keyframe, Some(id)));
            }
        }
        Ok(())
    }

    /// Checks the pool against the header; returns the record cap of a file
    /// with snapshots (0 otherwise).
    fn check_host_pool<E: HostEnvironment, const A: usize>(
        &self,
        pool: &mut HostPool<E, A>,
        ops: Option<RunOps<E, A>>,
    ) -> Result<usize, ReplayError> {
        let incompatible = |field: &'static str| -> Result<usize, ReplayError> {
            Err(PersistError::Incompatible { field }.into())
        };
        let Some(host) = self.header.host else {
            return incompatible("host_protocol");
        };
        let (config, hash) = match (host.config_checked(), ops) {
            (true, Some(ops)) => (ops.config)(pool)?,
            (true, None) => return incompatible("config: the host has no snapshots"),
            (false, _) => (Vec::new(), 0),
        };
        let id = host_identity(E::ENV_ID, E::ENV_VERSION, E::MASKS, pool.options(), hash);
        // `Tier::CrossPlatform` only makes `check_against` expect the empty
        // descriptor host replay files carry; the tier field itself is
        // compared with the identity's `HOST_TIER`.
        self.file
            .check_against(&id, Tier::CrossPlatform, BUILD_ID)?;
        if self.header.config != config.as_slice() {
            return incompatible("config");
        }
        if self.header.num_envs != pool.options().num_envs.get() {
            return incompatible("instance_count");
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
        match (host.identity(), ops) {
            (Some(identity), Some(ops)) => {
                if (ops.identity)(pool)? != identity {
                    return incompatible("identity");
                }
                let max = (ops.max_len)(pool)?;
                let cap = usize::try_from(self.limits.max_host_record_bytes).unwrap_or(usize::MAX);
                Ok(max.min(cap))
            }
            (Some(_), None) => incompatible("host_snapshots"),
            (None, _) => Ok(0),
        }
    }
}

/// Splits a keyframe's `len:u32 record` items and decodes each record,
/// author bytes capped by `max` before allocating. Untrusted until a pool
/// checks them.
fn decode_records<const A: usize>(
    mut bytes: &[u8],
    max: usize,
    index: u64,
) -> Result<Vec<HostRecord<A>>, ReplayError> {
    let malformed = || ReplayError::Malformed {
        event_index: index,
        field: "host records",
    };
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let (len, rest) = bytes.split_at_checked(4).ok_or_else(malformed)?;
        let len = usize::try_from(u32::from_le_bytes(len.try_into().map_err(|_| malformed())?))
            .map_err(|_| malformed())?;
        let (record, rest) = rest.split_at_checked(len).ok_or_else(malformed)?;
        out.try_reserve(1).map_err(|_| ReplayError::Allocation)?;
        out.push(HostRecord::from_bytes(record, max).map_err(|_| malformed())?);
        bytes = rest;
    }
    Ok(out)
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
    run: &Run,
    index: u64,
    ids: &Ids,
    recorded: u64,
) -> Result<(), ReplayError> {
    let out = pool.outputs()?;
    let n = run.n;
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
        Err(run.diverged(index, DivergenceKind::OutputsMismatch, None))
    }
}

/// Compares an accepted step's refusal rows (refusing hosts) and, when
/// recorded, its outputs hash.
fn check_step<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    run: &Run,
    index: u64,
    recorded: Option<u64>,
    refusals: &[RefusalRow],
) -> Result<(), ReplayError> {
    let out = pool.outputs()?;
    let n = run.n;
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
            return Err(run.diverged(
                index,
                DivergenceKind::RefusalMismatch,
                first.map(|r| r.instance),
            ));
        }
    }
    if let Some(recorded) = recorded
        && outputs_hash(out, n, 0..n, E::REFUSES_AGENTS) != recorded
    {
        return Err(run.diverged(index, DivergenceKind::OutputsMismatch, None));
    }
    Ok(())
}

/// A re-issued call of a recorded accepted call: accepted, poisoned, or
/// diverged (a rejection is [`DivergenceKind::RejectionMismatch`], any other
/// refusal [`DivergenceKind::CallFailed`]).
fn outcome<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    run: &Run,
    index: u64,
    result: &Result<(), EngineError>,
) -> Result<Outcome, ReplayError> {
    match result {
        Ok(()) => Ok(Outcome::Ok),
        Err(_) if pool.is_poisoned() => Ok(Outcome::Poisoned),
        Err(EngineError::ActionRejected { instance, .. }) => {
            Err(run.diverged(index, DivergenceKind::RejectionMismatch, Some(*instance)))
        }
        Err(_) => Err(run.diverged(index, DivergenceKind::CallFailed, None)),
    }
}

/// The ids of an event as a list.
fn ids_of(ids: &Ids, n: usize) -> Vec<u32> {
    match ids {
        Ids::All => (0..u32::try_from(n).unwrap_or(0)).collect(),
        Ids::List(ids) => ids.clone(),
    }
}

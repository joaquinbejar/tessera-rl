// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replaying a file into a pool and the divergence check (ADR-0072 section
//! 5).

use tessera_rl_core::config::{CanonicalConfig, canonical_bytes};
use tessera_rl_core::encoding::{Canonical, decode_from_slice};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout, StateHash, Validate};

use crate::masks::PersistMask;
use crate::persist::{BUILD_ID, PersistError, identity};
use crate::pool::EnvPool;
use crate::record::{DecodedRecord, PoolContext};
use crate::validate::ActionInput;

use super::format::StateWriteOp;
use super::reader::{Event, Ids, Replay};
use super::{DivergenceKind, ReplayError};

/// What a replay checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplaySummary {
    /// Events replayed, the end event excluded.
    pub events: u64,
    /// Calls re-issued (`Reset`, `Step` and `StateWrite` events).
    pub calls: u64,
    /// Checkpoints that matched.
    pub checkpoints: u64,
    /// Keyframes imported or checked.
    pub keyframes: u64,
    /// Whether the file ends with a `Poisoned` event that the replay
    /// reproduced.
    pub poisoned: bool,
    /// Whether the file ends with a state write that poisoned the recorded
    /// pool. Replay does not re-run the source of a state write, so this
    /// poison is not reproduced and the replay pool is left unpoisoned.
    pub unverified_poison: bool,
}

/// Where the replay stands between events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// Any call, keyframe or checkpoint.
    Any,
    /// The keyframe of the state write at this event.
    Keyframe(u64),
    /// The `Poisoned` event: the call at this event poisoned the replay pool.
    Poison(u64),
    /// Nothing: the file ended with `Poisoned`.
    Nothing,
}

struct State {
    summary: ReplaySummary,
    expect: Expect,
    last_checkpoint: Option<u64>,
    last_call: Option<u64>,
    /// The ids and branch counter of the pending state write.
    write: Option<(Ids, u64)>,
}

impl State {
    const fn diverged(
        &self,
        event_index: u64,
        kind: DivergenceKind,
        instance: Option<u32>,
    ) -> ReplayError {
        ReplayError::Diverged {
            event_index,
            kind,
            instance,
            last_matching_checkpoint: self.last_checkpoint,
        }
    }
}

impl Replay<'_> {
    /// Replays the file into `pool` and checks it (ADR-0072 section 5): the
    /// pool is checked against the header, the first keyframe is imported
    /// through the validated `import_all` path, every call is re-issued
    /// through the pool's public methods, state-write keyframes are imported
    /// through `import`, and checkpoints and periodic keyframes are compared
    /// with the replayed instances.
    ///
    /// The pool is checked against the header first: identity fields
    /// (ADR-0011), the canonical config bytes, the instance count and the
    /// action layout; the execution mode is free. To rebuild the pool from
    /// the header instead, as ADR-0072 describes, use
    /// [`Replay::run_rebuilt`] (or [`Replay::build_pool`]).
    ///
    /// Known gap: ADR-0072 section 8 says replay streams events. This reads
    /// them from the slice given to [`Replay::open`], so the whole file is
    /// held in memory.
    ///
    /// On a divergence the pool is left as the replay left it, for
    /// inspection.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] if the pool does not match the header,
    /// [`ReplayError::Diverged`] at the first failing check, and other typed
    /// errors for a malformed event.
    pub fn run<E, M>(&self, pool: &mut EnvPool<E, M>) -> Result<ReplaySummary, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        self.check_pool(pool)?;
        let mut st = State {
            summary: ReplaySummary::default(),
            expect: Expect::Any,
            last_checkpoint: None,
            last_call: None,
            write: None,
        };
        for item in self.events() {
            let (index, event) = item?;
            self.apply(pool, &mut st, index, event)?;
            st.summary.events = index.saturating_add(1);
        }
        match st.expect {
            Expect::Any | Expect::Nothing => Ok(st.summary),
            Expect::Keyframe(index) => Err(ReplayError::Malformed {
                event_index: index,
                field: "a state write without its keyframe",
            }),
            Expect::Poison(index) => {
                Err(st.diverged(index, DivergenceKind::UnexpectedPoison, None))
            }
        }
    }

    pub(crate) fn check_pool<E, M>(&self, pool: &EnvPool<E, M>) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        if self.header.async_pool {
            return Err(PersistError::Incompatible {
                field: "async: use run_async",
            }
            .into());
        }
        self.check_parts::<E, M>(&pool.cfg, &pool.options, pool.action_heads())
    }

    /// Checks a first-party pool's config, options and action heads against
    /// the header.
    pub(super) fn check_parts<E, M>(
        &self,
        cfg: &E::Config,
        options: &crate::options::PoolOptions,
        (discrete, continuous): (usize, usize),
    ) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig,
        M: PersistMask<E>,
    {
        let incompatible = |field: &'static str| -> Result<(), ReplayError> {
            Err(PersistError::Incompatible { field }.into())
        };
        if self.header.host.is_some() {
            return incompatible("host_protocol");
        }
        self.file
            .check_against(&identity::<E, M>(cfg, options)?, E::TIER, BUILD_ID)?;
        if self.header.num_envs != options.num_envs.get() {
            return incompatible("instance_count");
        }
        if self.header.config != canonical_bytes(cfg).as_slice() {
            return incompatible("config");
        }
        let same = |n: u32, m: usize| usize::try_from(n).is_ok_and(|n| n == m);
        if !same(self.header.agents, E::MAX_AGENTS)
            || !same(self.header.discrete_heads, discrete)
            || !same(self.header.continuous_heads, continuous)
        {
            return incompatible("action_layout");
        }
        if <<E::Header as RecordLayout>::Adaptive as Canonical>::ENCODED_LEN != 0 {
            return Err(ReplayError::AdaptiveUnsupported);
        }
        Ok(())
    }

    fn apply<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        st: &mut State,
        index: u64,
        event: Event<'_>,
    ) -> Result<(), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        match (st.expect, &event) {
            (Expect::Nothing, _) => return Err(malformed("an event after Poisoned")),
            (Expect::Poison(_), Event::Poisoned) => {
                st.summary.poisoned = true;
                st.expect = Expect::Nothing;
                return Ok(());
            }
            (Expect::Poison(call), _) => {
                return Err(st.diverged(call, DivergenceKind::UnexpectedPoison, None));
            }
            (Expect::Keyframe(_), Event::Keyframe { .. }) | (Expect::Any, _) => {}
            (Expect::Keyframe(_), Event::Poisoned) => {
                // A state write that poisoned: its source is not re-run.
                st.summary.unverified_poison = true;
                st.expect = Expect::Nothing;
                return Ok(());
            }
            (Expect::Keyframe(_), _) => {
                return Err(malformed("a state write without its keyframe"));
            }
        }
        match event {
            Event::Reset { ids, reseed, .. } => {
                let result = match &ids {
                    Ids::All => pool.reset_all(reseed.as_reseed()).map(|_| ()),
                    Ids::List(ids) => pool.reset(ids, reseed.as_reseed()).map(|_| ()),
                };
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::Step {
                ids,
                discrete,
                continuous_bits,
                ..
            } => {
                if ids != Ids::All {
                    return Err(ReplayError::Unsupported {
                        feature: "subset steps",
                    });
                }
                let continuous: Vec<f32> =
                    continuous_bits.iter().map(|&b| f32::from_bits(b)).collect();
                let result = pool
                    .step(ActionInput {
                        discrete: &discrete,
                        continuous: &continuous,
                    })
                    .map(|_| ());
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::StateWrite {
                op,
                ids,
                branch_counter,
            } => {
                if op == StateWriteOp::RestoreAll && ids != Ids::All {
                    return Err(malformed("restore_all ids"));
                }
                st.summary.calls = st.summary.calls.saturating_add(1);
                st.last_call = Some(index);
                st.expect = Expect::Keyframe(index);
                st.write = Some((ids, branch_counter));
                Ok(())
            }
            Event::Keyframe {
                branch_counter,
                ids,
                records,
            } => self.apply_keyframe(pool, st, index, branch_counter, &ids, records),
            Event::Checkpoint { ids, hashes } => {
                for (id, hash) in ids_of(&ids, self.header.num_envs).into_iter().zip(hashes) {
                    if pool.state_hash(id)? != hash {
                        return Err(st.diverged(index, DivergenceKind::Checkpoint, Some(id)));
                    }
                }
                st.last_checkpoint = Some(index);
                st.summary.checkpoints = st.summary.checkpoints.saturating_add(1);
                Ok(())
            }
            Event::Rejected { .. } => Err(malformed("a Rejected event in a first-party file")),
            Event::Recv { .. } => Err(malformed("a Recv event outside an async file")),
            Event::Poisoned => match st.last_call {
                Some(call) => Err(st.diverged(call, DivergenceKind::NotPoisoned, None)),
                None => Err(malformed("Poisoned before any call")),
            },
        }
    }

    /// The first keyframe is imported with `import_all`; a state write's
    /// keyframe with `import`; a periodic keyframe is compared with the pool.
    fn apply_keyframe<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        st: &mut State,
        index: u64,
        branch_counter: u64,
        ids: &Ids,
        records: &[u8],
    ) -> Result<(), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        let records = self.decode_records::<E>(index, records)?;
        st.summary.keyframes = st.summary.keyframes.saturating_add(1);
        if index == 0 {
            if *ids != Ids::All {
                return Err(malformed("first keyframe ids"));
            }
            pool.import_all(&records)
                .map_err(|_| st.diverged(index, DivergenceKind::KeyframeRejected, None))?;
            pool.restore_pool_context(PoolContext { branch_counter })?;
            return Ok(());
        }
        if let Expect::Keyframe(write) = st.expect {
            st.expect = Expect::Any;
            if st.write.take() != Some((ids.clone(), branch_counter)) {
                return Err(malformed("a keyframe that differs from its state write"));
            }
            let id_list = ids_of(ids, self.header.num_envs);
            pool.import(&id_list, &records)
                .map_err(|_| st.diverged(write, DivergenceKind::KeyframeRejected, None))?;
            pool.restore_pool_context(PoolContext { branch_counter })?;
            return Ok(());
        }
        // A periodic keyframe: check it against the replayed pool.
        if pool.pool_context()?.branch_counter != branch_counter {
            return Err(st.diverged(index, DivergenceKind::BranchCounter, None));
        }
        for (id, record) in ids_of(ids, self.header.num_envs).into_iter().zip(&records) {
            if pool.state_hash(id)? != StateHash::of(record) {
                return Err(st.diverged(index, DivergenceKind::Keyframe, Some(id)));
            }
        }
        Ok(())
    }

    /// Decodes a keyframe's untrusted records; the import validates them.
    pub(super) fn decode_records<E: Persist>(
        &self,
        index: u64,
        bytes: &[u8],
    ) -> Result<Vec<DecodedRecord<E>>, ReplayError> {
        let len = usize::try_from(self.header.record_len)
            .ok()
            .filter(|&len| len == <DecodedRecord<E> as Canonical>::ENCODED_LEN && len > 0)
            .ok_or(PersistError::Incompatible {
                field: "record_len",
            })?;
        let n = bytes.len() / len;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| ReplayError::Allocation)?;
        for (i, chunk) in bytes.chunks_exact(len).enumerate() {
            let record = decode_from_slice(chunk).map_err(|source| ReplayError::Record {
                event_index: index,
                index: i,
                source,
            })?;
            out.push(record);
        }
        Ok(out)
    }
}

/// Records the outcome of a re-issued call: a success, a poison (the next
/// event must be `Poisoned`), or a divergence.
fn call_outcome<E, M>(
    pool: &EnvPool<E, M>,
    st: &mut State,
    index: u64,
    ok: bool,
) -> Result<(), ReplayError>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    st.summary.calls = st.summary.calls.saturating_add(1);
    st.last_call = Some(index);
    if ok {
        return Ok(());
    }
    if pool.is_poisoned() {
        st.expect = Expect::Poison(index);
        return Ok(());
    }
    Err(st.diverged(index, DivergenceKind::CallFailed, None))
}

/// The ids of an event as a list.
fn ids_of(ids: &Ids, num_envs: u32) -> Vec<u32> {
    match ids {
        Ids::All => (0..num_envs).collect(),
        Ids::List(ids) => ids.clone(),
    }
}

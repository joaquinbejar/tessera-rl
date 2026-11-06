// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase side of an environment object (ADR-0047 section 9): the
//! Pipelined pool and its stored tokens behind a binding mutex. No `unsafe`
//! here.
//!
//! The mutex is held only for non-waiting sections and always taken with
//! the interpreter released, and no Python code runs under it, so a thread
//! blocked on it never holds the interpreter. `recv` waits outside it, so
//! another thread may `send` meanwhile.

use core::any::Any;
use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};
use tessera_rl_core::Persist;
use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_engine::{DecodeLimits, PersistMask};

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{
    ActionInput, CycleState, EngineError, HalfGenerations, HalfToken, MAX_LEAVES, MaskProvider,
    OutputStorage, PipelinedPool, Record, Reseed, Waiter,
};
use tessera_rl_num::identity::Seed;

use crate::env::{BindError, Layout, Owned, PoolOps, ResetSeed};

/// Whether a half view's window has ended: its half has a new generation, a
/// writer is running, or the pool is poisoned, whose rows a failed writer may
/// have partly written. Neither advances a generation (ADR-0047 section 5).
pub(crate) fn half_stale(
    generations: &HalfGenerations,
    half: usize,
    taken: u64,
    unsettled: &AtomicBool,
) -> bool {
    // A row read before this call is not ordered by an Acquire load alone: this
    // fence pairs with the writer's fence after it sets the flag or moves the
    // generation (a seqlock reader). Python reads rows without Rust atomics, so
    // at that boundary this is best effort, not a guarantee.
    core::sync::atomic::fence(Ordering::Acquire);
    unsettled.load(Ordering::Acquire) || generations.half(half) != Some(taken)
}

/// One half's outputs: which half, its generation captured under the mutex
/// together with its token, and a copy of its rows in `owned` mode.
#[derive(Debug, Clone)]
pub(crate) struct HalfOutput {
    pub(crate) half: usize,
    /// The delivery number of a `recv` (0 for `reset` and `restore_all`).
    pub(crate) seq: u64,
    pub(crate) generation: u64,
    pub(crate) owned: Option<Arc<Owned>>,
}

/// What one non-waiting receive found.
#[derive(Debug)]
pub(crate) enum Polled {
    /// A half, new or not yet acknowledged.
    Half(HalfOutput),
    /// The half that must finish next is still running.
    Running(usize),
    /// Another thread has a delivery it has not acknowledged yet.
    Busy,
}

/// A whole-pool snapshot, opaque outside the pool that made it.
pub(crate) type Snapshot = Arc<dyn Any + Send + Sync>;

/// What a split-phase environment object can do, whatever the environment.
pub(crate) trait SplitOps: Send + Sync {
    fn storage(&self) -> Arc<OutputStorage>;
    fn halves(&self) -> [Range<usize>; 2];
    fn half_generations(&self) -> HalfGenerations;
    /// Set while a writer runs or the pool is poisoned; views read it.
    fn unsettled_flag(&self) -> Arc<AtomicBool>;
    /// Held by the one running `recv`.
    fn receiving(&self) -> &AtomicBool;
    fn waiter(&self) -> &Waiter;
    /// Resets every instance; the stored tokens are released first, and
    /// both halves' new tokens are stored.
    fn reset(&self, seed: &ResetSeed, owned: bool) -> Result<[HalfOutput; 2], BindError>;
    /// Starts `half`'s next step. `delivered` acknowledges deliveries up to
    /// that number; a half delivered but not acknowledged counts as in
    /// flight, since its caller may never have seen it.
    fn send(&self, half: u32, actions: &[i32], delivered: u64) -> Result<(), BindError>;
    /// The oldest half, without waiting. A delivery stays stored until
    /// `delivered` acknowledges it, and repeats until then, so a signal
    /// raised after the native call returned loses nothing.
    fn try_recv(&self, owned: bool, delivered: u64) -> Result<Polled, BindError>;
    /// Captures every record and the cycle state; no work may be in flight.
    fn snapshot_all(&self) -> Result<Snapshot, BindError>;
    /// Restores a snapshot of this same pool and returns both halves.
    fn restore_all(&self, snapshot: &Snapshot, owned: bool) -> Result<[HalfOutput; 2], BindError>;
    /// The persisted file of `ids`; no work may be in flight.
    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError>;
    /// Loads a persisted file into every instance and returns both halves. A
    /// refused file leaves tokens, views and deliveries as they were.
    fn load(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
        owned: bool,
    ) -> Result<[HalfOutput; 2], BindError>;
    fn is_poisoned(&self) -> bool;
    /// Drains and joins the workers; idempotent.
    fn close(&self);
}

struct State<E: Environment, M: MaskProvider<E>> {
    pool: Option<PipelinedPool<E, M>>,
    /// The token of each returned half, the borrow behind its NumPy views.
    tokens: [Option<HalfToken<E, M>>; 2],
    /// The last delivery, until acknowledged, and the thread it went to.
    pending: Option<(HalfOutput, ThreadId)>,
    last_seq: u64,
}

impl<E: Environment, M: MaskProvider<E>> State<E, M> {
    fn acknowledge(&mut self, delivered: u64) {
        if self
            .pending
            .as_ref()
            .is_some_and(|(out, _)| out.seq <= delivered)
        {
            self.pending = None;
        }
    }
}

struct Taken<E: Environment> {
    owner: Arc<OutputStorage>,
    records: Vec<Record<E>>,
    cycles: CycleState,
}

struct SplitPool<E: Environment, M: MaskProvider<E>> {
    state: Mutex<State<E, M>>,
    storage: Arc<OutputStorage>,
    halves: [Range<usize>; 2],
    generations: HalfGenerations,
    unsettled: Arc<AtomicBool>,
    /// A record to size snapshot buffers with.
    template: Record<E>,
    receiving: AtomicBool,
    waiter: Waiter,
    terminal: bool,
}

fn closed() -> BindError {
    BindError::Argument("the environment is closed".to_owned())
}

fn flags(v: &[tessera_rl_core::Bool8]) -> Vec<u8> {
    v.iter().map(|f| f.to_byte()).collect()
}

fn copy_token<E: Environment, M: MaskProvider<E>>(
    token: &HalfToken<E, M>,
    terminal: bool,
) -> Owned {
    Owned {
        observations: (0..MAX_LEAVES)
            .map_while(|leaf| token.observation(leaf).map(<[u8]>::to_vec))
            .collect(),
        terminal_observations: if terminal {
            (0..MAX_LEAVES)
                .map_while(|leaf| token.terminal_observation(leaf).map(<[u8]>::to_vec))
                .collect()
        } else {
            Vec::new()
        },
        rewards: token
            .rewards()
            .iter()
            .flat_map(|r| r.to_le_bytes())
            .collect(),
        agent_terminated: flags(token.agent_terminated()),
        agent_truncated: flags(token.agent_truncated()),
        alive: flags(token.alive()),
        acting: flags(token.acting()),
        episode_terminated: flags(token.episode_terminated()),
        episode_truncated: flags(token.episode_truncated()),
        masks: flags(token.masks()),
    }
}

impl<E, M> SplitPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn lock(&self) -> Result<MutexGuard<'_, State<E, M>>, BindError> {
        // Poisoned only by a panic in a section of this module.
        self.state
            .lock()
            .map_err(|_| BindError::Engine(EngineError::Poisoned))
    }

    /// Views read stale from here until `settle`: a writer rewrites rows
    /// before the engine moves their generations (ADR-0047 section 5). The
    /// fence orders this store before the writer's row writes.
    fn unsettle(&self) {
        self.unsettled.store(true, Ordering::Release);
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Publishes the outcome before the call returns, after anything else a
    /// view reads has moved: the pool's poison keeps views stale.
    fn settle(&self, pool: &PipelinedPool<E, M>) {
        self.unsettled.store(pool.is_poisoned(), Ordering::Release);
    }

    fn output(&self, half: usize, seq: u64, token: &HalfToken<E, M>, owned: bool) -> HalfOutput {
        HalfOutput {
            half,
            seq,
            generation: self.generations.half(half).unwrap_or(0),
            owned: owned.then(|| Arc::new(copy_token(token, self.terminal))),
        }
    }
}

impl<E, M> SplitOps for SplitPool<E, M>
where
    E: Persist,
    M: PersistMask<E> + Send + Sync,
    E::Config: CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn storage(&self) -> Arc<OutputStorage> {
        Arc::clone(&self.storage)
    }

    fn halves(&self) -> [Range<usize>; 2] {
        self.halves.clone()
    }

    fn half_generations(&self) -> HalfGenerations {
        self.generations.clone()
    }

    fn unsettled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.unsettled)
    }

    fn receiving(&self) -> &AtomicBool {
        &self.receiving
    }

    fn waiter(&self) -> &Waiter {
        &self.waiter
    }

    fn reset(&self, seed: &ResetSeed, owned: bool) -> Result<[HalfOutput; 2], BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        // The FFI release (ADR-0047 section 9): NumPy views are not Rust
        // references and keep the overwrite semantics.
        state.tokens = [None, None];
        let seeds: Vec<Option<Seed>>;
        let reseed = match seed {
            ResetSeed::Keep => Reseed::Keep,
            ResetSeed::Base(s) => Reseed::Base(Seed::new(*s)),
            ResetSeed::List(list) => {
                seeds = list.iter().map(|s| s.map(Seed::new)).collect();
                Reseed::Each(&seeds)
            }
        };
        self.unsettle();
        let result = pool.reset_all(reseed);
        self.settle(pool);
        let [a, b] = result?;
        let out = [self.output(0, 0, &a, owned), self.output(1, 0, &b, owned)];
        state.tokens = [Some(a), Some(b)];
        state.pending = None;
        Ok(out)
    }

    fn send(&self, half: u32, actions: &[i32], delivered: u64) -> Result<(), BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        let index = usize::try_from(half).unwrap_or(usize::MAX);
        if state
            .pending
            .as_ref()
            .is_some_and(|(out, _)| out.half == index)
        {
            return Err(BindError::Engine(EngineError::HalfInFlight {
                half: u8::try_from(half).unwrap_or(u8::MAX),
            }));
        }
        let stored = state.tokens.get_mut(index).and_then(Option::take);
        let token = match stored {
            Some(token) => token,
            None => pool.idle_token(half)?,
        };
        let result = pool.send(
            token,
            ActionInput {
                discrete: actions,
                continuous: &[],
            },
        );
        self.settle(pool);
        result.map_err(|e| {
            if let Some(slot) = state.tokens.get_mut(index) {
                *slot = e.token;
            }
            BindError::Engine(e.error)
        })
    }

    fn try_recv(&self, owned: bool, delivered: u64) -> Result<Polled, BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        let me = std::thread::current().id();
        // Health first: a cached delivery is not returned after a panic. It is
        // lost; recovery clears it, and the next delivery has a new seq.
        if pool.is_poisoned() {
            let refused = pool.try_recv().map(|_| ());
            self.settle(pool);
            refused?;
            return Err(BindError::Engine(EngineError::Poisoned));
        }
        if let Some((out, owner)) = &state.pending {
            return Ok(if *owner == me {
                Polled::Half(out.clone())
            } else {
                Polled::Busy
            });
        }
        let received = pool.try_recv();
        self.settle(pool);
        let Some(token) = received? else {
            let oldest = pool.oldest_half().unwrap_or(0);
            return Ok(Polled::Running(usize::try_from(oldest).unwrap_or(0)));
        };
        let half = usize::try_from(token.half()).unwrap_or(usize::MAX);
        state.last_seq = state.last_seq.saturating_add(1);
        let out = self.output(half, state.last_seq, &token, owned);
        if let Some(slot) = state.tokens.get_mut(half) {
            *slot = Some(token);
        }
        state.pending = Some((out.clone(), me));
        Ok(Polled::Half(out))
    }

    fn snapshot_all(&self) -> Result<Snapshot, BindError> {
        let guard = self.lock()?;
        let pool = guard.pool.as_ref().ok_or_else(closed)?;
        let n = self.halves.iter().map(ExactSizeIterator::len).sum();
        let mut records = vec![self.template; n];
        let cycles = pool.snapshot_all(&mut records)?;
        Ok(Arc::new(Taken::<E> {
            owner: Arc::clone(&self.storage),
            records,
            cycles,
        }))
    }

    fn restore_all(&self, snapshot: &Snapshot, owned: bool) -> Result<[HalfOutput; 2], BindError> {
        let taken = snapshot
            .downcast_ref::<Taken<E>>()
            .filter(|t| Arc::ptr_eq(&t.owner, &self.storage))
            .ok_or_else(|| {
                BindError::Argument("a snapshot restores only the pool that took it".to_owned())
            })?;
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        // The FFI release, as in `reset`.
        state.tokens = [None, None];
        self.unsettle();
        let restored = pool.restore_all(&taken.records, &taken.cycles);
        self.settle(pool);
        restored?;
        let a = pool.idle_token(0)?;
        let b = pool.idle_token(1)?;
        let out = [self.output(0, 0, &a, owned), self.output(1, 0, &b, owned)];
        state.tokens = [Some(a), Some(b)];
        state.pending = None;
        Ok(out)
    }

    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError> {
        let guard = self.lock()?;
        let pool = guard.pool.as_ref().ok_or_else(closed)?;
        Ok(pool.encode(ids)?)
    }

    fn load(
        &self,
        bytes: &[u8],
        limits: DecodeLimits,
        owned: bool,
    ) -> Result<[HalfOutput; 2], BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        // Everything that can refuse the file comes before the tokens go.
        let records = pool.decode(bytes, limits)?;
        pool.check_records(&records)?;
        // The FFI release, as in `reset`.
        state.tokens = [None, None];
        self.unsettle();
        let imported = pool.import_all(&records);
        self.settle(pool);
        imported?;
        let a = pool.idle_token(0)?;
        let b = pool.idle_token(1)?;
        let out = [self.output(0, 0, &a, owned), self.output(1, 0, &b, owned)];
        state.tokens = [Some(a), Some(b)];
        state.pending = None;
        Ok(out)
    }

    fn is_poisoned(&self) -> bool {
        self.lock().map_or(true, |s| {
            s.pool.as_ref().is_some_and(PipelinedPool::is_poisoned)
        })
    }

    fn close(&self) {
        let taken = self.lock().ok().map(|mut state| {
            state.pending = None;
            let tokens = core::mem::take(&mut state.tokens);
            (tokens, state.pool.take())
        });
        // Outside the mutex: this drains and joins the workers.
        drop(taken);
    }
}

/// A Pipelined environment object behind [`PoolOps`]. Every operation of
/// the mode goes through [`SplitOps`]; the rest are refused.
struct SplitHandle {
    ops: Arc<dyn SplitOps>,
    layout: Layout,
}

fn wrong_mode(call: &'static str) -> BindError {
    BindError::Engine(EngineError::WrongMode {
        call,
        mode: "pipelined",
    })
}

impl PoolOps for SplitHandle {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn reset(&mut self, _ids: Option<&[u32]>, _seed: &ResetSeed) -> Result<(), BindError> {
        Err(wrong_mode("reset"))
    }

    fn step(&mut self, _actions: &[i32]) -> Result<(), BindError> {
        Err(wrong_mode("step"))
    }

    fn outputs(&self) -> Result<Owned, BindError> {
        Ok(Owned::default())
    }

    fn storage(&self) -> Arc<OutputStorage> {
        self.ops.storage()
    }

    fn snapshot(&self, _id: u32) -> Result<Vec<u8>, BindError> {
        Err(wrong_mode("snapshot"))
    }

    fn import(&mut self, _id: u32, _bytes: &[u8]) -> Result<(), BindError> {
        Err(wrong_mode("import_record"))
    }

    fn encode(&self, _ids: &[u32]) -> Result<Vec<u8>, BindError> {
        Err(wrong_mode("encode"))
    }

    fn load(
        &mut self,
        _bytes: &[u8],
        _ids: Option<&[u32]>,
        _limits: tessera_rl_engine::DecodeLimits,
    ) -> Result<(), BindError> {
        Err(wrong_mode("load"))
    }

    fn fork(&mut self, _src: u32, _dst: &[u32]) -> Result<(), BindError> {
        Err(wrong_mode("fork"))
    }

    fn is_poisoned(&self) -> bool {
        self.ops.is_poisoned()
    }

    fn split(&self) -> Option<Arc<dyn SplitOps>> {
        Some(Arc::clone(&self.ops))
    }
}

/// The environment object of a Pipelined `pool`, whose outputs live in
/// `storage`.
pub(crate) fn pipelined<E, M>(
    pool: PipelinedPool<E, M>,
    storage: Arc<OutputStorage>,
    template: Record<E>,
    layout: Layout,
) -> Box<dyn PoolOps>
where
    E: Persist,
    M: PersistMask<E> + Send + Sync + 'static,
    E: 'static,
    E::Config: CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let usize_of = |v: u32| usize::try_from(v).unwrap_or(usize::MAX);
    let halves = pool.halves().map(|r| usize_of(r.start)..usize_of(r.end));
    let waiter = pool.waiter();
    let generations = pool.half_generations();
    let ops: Arc<dyn SplitOps> = Arc::new(SplitPool {
        state: Mutex::new(State {
            pool: Some(pool),
            tokens: [None, None],
            pending: None,
            last_seq: 0,
        }),
        storage,
        halves,
        generations,
        unsettled: Arc::new(AtomicBool::new(false)),
        template,
        receiving: AtomicBool::new(false),
        waiter,
        terminal: layout.terminal,
    });
    Box::new(SplitHandle { ops, layout })
}

#[cfg(any(test, feature = "test-hooks"))]
#[path = "../../tessera-rl-engine/tests/common/mod.rs"]
pub(crate) mod common;

#[cfg(test)]
pub(crate) mod reset_probe {
    //! Reads a flag from inside a writer: the fixture's `panic_on_reset`
    //! asserts during the reset, which calls the panic hook there.

    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, PoisonError};

    /// Serializes the tests that panic in a writer, as the hook is global.
    pub(crate) static RESET_PANICS: Mutex<()> = Mutex::new(());

    /// The value of `flag` at the first injected reset panic in `run`.
    pub(crate) fn flag_at_reset_panic(flag: &Arc<AtomicBool>, run: impl FnOnce()) -> Option<bool> {
        flag_at_panic(flag, "injected reset panic", run)
    }

    /// The value of `flag` at the first injected restore panic in `run`,
    /// a panic in `observe` on this thread.
    pub(crate) fn flag_at_restore_panic(
        flag: &Arc<AtomicBool>,
        run: impl FnOnce(),
    ) -> Option<bool> {
        struct Off;
        impl Drop for Off {
            fn drop(&mut self) {
                crate::observe_panic::PANIC_IN_OBSERVE.with(|p| p.set(false));
            }
        }
        crate::observe_panic::PANIC_IN_OBSERVE.with(|p| p.set(true));
        let _off = Off;
        flag_at_panic(flag, crate::observe_panic::OBSERVE_PANIC, run)
    }

    fn flag_at_panic(
        flag: &Arc<AtomicBool>,
        wanted: &'static str,
        run: impl FnOnce(),
    ) -> Option<bool> {
        let _serial = RESET_PANICS.lock().unwrap_or_else(PoisonError::into_inner);
        let seen: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let previous = Arc::new(std::panic::take_hook());
        {
            let (flag, seen, previous) =
                (Arc::clone(flag), Arc::clone(&seen), Arc::clone(&previous));
            std::panic::set_hook(Box::new(move |info| {
                let message = info
                    .payload()
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| info.payload().downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                if message.contains(wanted) {
                    let mut slot = seen.lock().unwrap_or_else(PoisonError::into_inner);
                    if slot.is_none() {
                        *slot = Some(flag.load(Ordering::Acquire));
                    }
                } else {
                    previous(info);
                }
            }));
        }
        run();
        drop(std::panic::take_hook());
        if let Ok(previous) = Arc::try_unwrap(previous) {
            std::panic::set_hook(previous);
        }
        *seen.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{PoolArgs, cartpole};

    fn split() -> (Box<dyn PoolOps>, Arc<dyn SplitOps>) {
        let ops = cartpole(&PoolArgs {
            num_envs: 8,
            seed: 1,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: Some(2),
            workers: 1,
            pipelined: true,
            asynced: None,
        })
        .unwrap();
        let split = ops.split().unwrap();
        (ops, split)
    }

    fn received(split: &dyn SplitOps, delivered: u64) -> HalfOutput {
        loop {
            match split.try_recv(false, delivered).unwrap() {
                Polled::Half(out) => return out,
                Polled::Running(h) => {
                    let _ = split
                        .waiter()
                        .wait_half(h, core::time::Duration::from_millis(50));
                }
                Polled::Busy => panic!("no other thread receives"),
            }
        }
    }

    /// The generation a delivery carries is the one under the mutex: a send
    /// on another thread before its views are built cannot make them fresh.
    #[test]
    fn test_split_delivery_generation_predates_a_later_send() {
        let (_ops, split) = split();
        split.reset(&ResetSeed::Keep, false).unwrap();
        split.send(0, &[0; 4], 0).unwrap();
        let out = received(&*split, 0);
        assert_eq!(out.generation, split.half_generations().half(0).unwrap());
        split.send(0, &[0; 4], out.seq).unwrap();
        assert_ne!(out.generation, split.half_generations().half(0).unwrap());
        split.close();
    }

    #[test]
    fn test_split_generation_moves_on_accepted_send_only() {
        let (_ops, split) = split();
        split.reset(&ResetSeed::Keep, false).unwrap();
        let before = split.half_generations().half(0).unwrap();
        assert!(split.send(0, &[0; 3], 0).is_err());
        assert_eq!(split.half_generations().half(0).unwrap(), before);
        split.send(0, &[0; 4], 0).unwrap();
        assert_ne!(split.half_generations().half(0).unwrap(), before);
        split.close();
    }

    #[test]
    fn test_split_unacknowledged_delivery_repeats_and_blocks_its_send() {
        let (_ops, split) = split();
        split.reset(&ResetSeed::Keep, false).unwrap();
        split.send(0, &[0; 4], 0).unwrap();
        let first = received(&*split, 0);
        let again = received(&*split, 0);
        assert_eq!((first.seq, first.half), (again.seq, again.half));
        assert!(split.send(0, &[0; 4], 0).is_err());
        split.send(0, &[0; 4], first.seq).unwrap();
        split.close();
    }

    fn counter(cfg: &super::common::CountConfig) -> Arc<dyn SplitOps> {
        env_pool::<super::common::Counter>(cfg)
    }

    fn env_pool<E>(cfg: &super::common::CountConfig) -> Arc<dyn SplitOps>
    where
        E: Environment<Config = super::common::CountConfig> + Persist + 'static,
        E::Header: Send + Sync,
        E::State: Send + Sync,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<super::common::CountConfig>,
    {
        let mut options = tessera_rl_engine::PoolOptions::new(
            core::num::NonZeroU32::new(8).unwrap(),
            Seed::new(1),
        );
        options.chunk_size = core::num::NonZeroU32::new(2).unwrap();
        let pool =
            tessera_rl_engine::EnvPool::<E, tessera_rl_engine::NoMasks>::new(cfg.clone(), options)
                .unwrap();
        let args = PoolArgs {
            num_envs: 8,
            seed: 1,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: Some(2),
            workers: 2,
            pipelined: true,
            asynced: None,
        };
        crate::env::finish(cfg, pool, &args, 8, false)
            .unwrap()
            .split()
            .unwrap()
    }

    /// Healthy, the half not sent stays fresh while the other half runs.
    #[test]
    fn test_split_idle_half_stays_fresh_while_healthy() {
        let split = counter(&super::common::CountConfig {
            length: 9,
            ..super::common::CountConfig::default()
        });
        split.reset(&ResetSeed::Keep, false).unwrap();
        let (gens, flag) = (split.half_generations(), split.unsettled_flag());
        let idle = gens.half(1).unwrap();
        let rows = split.halves()[0].len() * <super::common::Counter as Environment>::MAX_AGENTS;
        split.send(0, &vec![0; rows], 0).unwrap();
        assert!(!half_stale(&gens, 1, idle, &flag));
        split.close();
    }

    /// A panic during a reset advances no generation, yet a view must not
    /// report fresh over rows the reset may have partly written.
    #[test]
    fn test_split_load_refuses_a_bad_file_and_recovers_from_poison() {
        let split = counter(&super::common::CountConfig {
            panic_at: 1,
            length: 9,
            ..super::common::CountConfig::default()
        });
        split.reset(&ResetSeed::Keep, true).unwrap();
        let ids: Vec<u32> = (0..8).collect();
        let file = split.encode(&ids).unwrap();
        let limits = tessera_rl_engine::DecodeLimits::default();
        let mut bad = file.clone();
        if let Some(last) = bad.last_mut() {
            *last ^= 1;
        }
        // A refused file changes no generation, so views stay fresh.
        assert!(split.load(&bad, limits, true).is_err());
        assert!(
            !split
                .unsettled_flag()
                .load(core::sync::atomic::Ordering::Acquire)
        );
        let width = split.halves()[0].len() * 2;
        split.send(0, &vec![0; width], 0).unwrap();
        while !split.is_poisoned() {
            let _ = split.try_recv(true, 0);
            std::thread::yield_now();
        }
        assert!(split.encode(&[]).is_err());
        assert!(split.load(&bad, limits, true).is_err());
        assert!(split.is_poisoned());
        split.load(&file, limits, true).unwrap();
        assert!(!split.is_poisoned());
        assert_eq!(split.encode(&ids).unwrap(), file);
        split.close();
    }

    /// `file` with a record that decodes and `Validate` refuses, resealed.
    fn invariant_refused(file: &[u8]) -> Vec<u8> {
        let mut bad = file.to_vec();
        let state = bad.len() - 8 - size_of::<super::common::Counts>();
        // `steps`, after the two `value` fields.
        bad[state + 8..state + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        let n = bad.len() - 8;
        let sum = tessera_rl_core::encoding::checksum(&bad[..n]);
        bad[n..].copy_from_slice(&sum.to_le_bytes());
        bad
    }

    #[test]
    fn test_split_load_refuses_before_giving_up_the_tokens() {
        let split = counter(&super::common::CountConfig {
            length: 9,
            ..super::common::CountConfig::default()
        });
        split.reset(&ResetSeed::Keep, true).unwrap();
        let ids: Vec<u32> = (0..8).collect();
        let file = split.encode(&ids).unwrap();
        let limits = tessera_rl_engine::DecodeLimits::default();
        let width = split.halves()[0].len() * 2;
        // A record `Validate` refuses: both halves can still be sent.
        let bad = invariant_refused(&file);
        assert!(matches!(
            split.load(&bad, limits, true),
            Err(BindError::Engine(
                tessera_rl_engine::EngineError::Invariant { .. }
            ))
        ));
        split.send(0, &vec![0; width], 0).unwrap();
        // A good file with work in flight: retryable, and the idle half's token stays.
        assert!(matches!(
            split.load(&file, limits, true),
            Err(BindError::Engine(
                tessera_rl_engine::EngineError::NotQuiescent
            ))
        ));
        split.send(1, &vec![0; width], 0).unwrap();
        let mut seq = 0;
        for _ in 0..2 {
            loop {
                if let Polled::Half(out) = split.try_recv(true, seq).unwrap() {
                    seq = out.seq;
                    break;
                }
                std::thread::yield_now();
            }
        }
        let _ = split.try_recv(true, seq);
        split.load(&file, limits, true).unwrap();
        assert_eq!(split.encode(&ids).unwrap(), file);
        split.close();
    }

    #[test]
    fn test_split_flag_is_set_inside_a_writer() {
        let split = counter(&super::common::CountConfig {
            panic_on_reset: true,
            ..super::common::CountConfig::default()
        });
        let flag = split.unsettled_flag();
        let seen = super::reset_probe::flag_at_reset_panic(&flag, || {
            assert!(split.reset(&ResetSeed::Keep, false).is_err());
        });
        assert_eq!(seen, Some(true), "views must read stale during the write");
        split.close();
    }

    #[test]
    fn test_split_flag_is_set_inside_restore_all() {
        let split = env_pool::<crate::observe_panic::ObservePanicCounter>(
            &super::common::CountConfig::default(),
        );
        split.reset(&ResetSeed::Keep, false).unwrap();
        let taken = split.snapshot_all().unwrap();
        let flag = split.unsettled_flag();
        let seen = super::reset_probe::flag_at_restore_panic(&flag, || {
            assert!(split.restore_all(&taken, false).is_err());
        });
        assert_eq!(seen, Some(true), "views must read stale during the write");
        split.close();
    }

    #[test]
    fn test_split_panic_during_reset_makes_views_stale() {
        let _serial = super::reset_probe::RESET_PANICS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let split = counter(&super::common::CountConfig {
            panic_on_reset: true,
            ..super::common::CountConfig::default()
        });
        let gens = split.half_generations();
        let taken = gens.half(0).unwrap();
        let flag = split.unsettled_flag();
        assert!(!half_stale(&gens, 0, taken, &flag));
        assert!(split.reset(&ResetSeed::Keep, false).is_err());
        assert!(split.is_poisoned());
        assert_eq!(gens.half(0), Some(taken), "no generation advanced");
        assert!(half_stale(&gens, 0, taken, &flag));
        split.close();
    }

    /// A step's panic marks the other half's views too; a recovering reset
    /// gives new views that are fresh, and old ones stay stale.
    #[test]
    fn test_split_poison_ends_views_and_recovery_clears_it() {
        let split = counter(&super::common::CountConfig {
            panic_at: 1,
            length: 9,
            ..super::common::CountConfig::default()
        });
        split.reset(&ResetSeed::Keep, false).unwrap();
        let gens = split.half_generations();
        let flag = split.unsettled_flag();
        let other = gens.half(1).unwrap();
        let rows = split.halves()[0].len() * <super::common::Counter as Environment>::MAX_AGENTS;
        split.send(0, &vec![0; rows], 0).unwrap();
        let failed = loop {
            match split.try_recv(false, 0) {
                Ok(Polled::Running(h)) => {
                    let _ = split
                        .waiter()
                        .wait_half(h, core::time::Duration::from_millis(1));
                }
                Ok(other) => panic!("expected a panic, got {other:?}"),
                Err(e) => break e,
            }
        };
        assert!(matches!(
            failed,
            BindError::Engine(EngineError::Panicked | EngineError::Poisoned)
        ));
        assert!(half_stale(&gens, 1, other, &flag));
        split.reset(&ResetSeed::Keep, false).unwrap();
        assert!(half_stale(&gens, 1, other, &flag), "old view stays stale");
        let fresh = gens.half(1).unwrap();
        assert!(!half_stale(&gens, 1, fresh, &flag), "new view is fresh");
        split.close();
    }

    /// Pipelined learns of a panic only on collecting the job, so a half cached
    /// and unacknowledged is still delivered before that: its rows are the other
    /// half's, not written by the panic. Acknowledging it collects the panic, and
    /// from then on every poll refuses, the old ack included.
    #[test]
    fn test_split_cached_delivery_then_collected_panic_refuses() {
        let split = counter(&super::common::CountConfig {
            panic_at: 2,
            length: 9,
            ..super::common::CountConfig::default()
        });
        split.reset(&ResetSeed::Keep, false).unwrap();
        let rows = |h: usize| {
            vec![0; split.halves()[h].len() * <super::common::Counter as Environment>::MAX_AGENTS]
        };
        let receive = |ack| loop {
            match split.try_recv(false, ack).unwrap() {
                Polled::Half(out) => return out,
                Polled::Running(_) => std::thread::yield_now(),
                Polled::Busy => panic!("one thread only"),
            }
        };
        split.send(0, &rows(0), 0).unwrap();
        let first = receive(0);
        split.send(1, &rows(1), first.seq).unwrap();
        let second = receive(first.seq);
        assert_eq!(second.half, 1);
        split.send(0, &rows(0), first.seq).unwrap();
        assert!(matches!(
            split.try_recv(false, first.seq),
            Ok(Polled::Half(out)) if out.seq == second.seq
        ));
        assert!(!split.is_poisoned());
        let collected = loop {
            match split.try_recv(false, second.seq) {
                Ok(Polled::Running(_)) => std::thread::yield_now(),
                other => break other,
            }
        };
        assert!(matches!(
            collected,
            Err(BindError::Engine(EngineError::Panicked))
        ));
        assert!(split.is_poisoned());
        assert!(matches!(
            split.try_recv(false, first.seq),
            Err(BindError::Engine(EngineError::Poisoned))
        ));
        split.reset(&ResetSeed::Keep, false).unwrap();
        split.close();
    }
}

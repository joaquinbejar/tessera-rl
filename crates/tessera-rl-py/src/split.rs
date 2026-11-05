// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split-phase side of an environment object (ADR-0047 section 9): the
//! Pipelined pool and its stored tokens behind a binding mutex. No `unsafe`
//! here.
//!
//! The mutex is held only for non-waiting sections and always taken with
//! the interpreter released, and no Python code runs under it, so a thread
//! blocked on it never holds the interpreter. `recv` waits outside it, so
//! another thread may `send` meanwhile.

use core::ops::Range;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use std::sync::{Arc, Mutex, MutexGuard};

use tessera_rl_core::{AdaptiveValues, Environment, RecordLayout};
use tessera_rl_engine::{
    ActionInput, EngineError, HalfToken, MAX_LEAVES, MaskProvider, OutputStorage, PipelinedPool,
    Reseed, Waiter,
};
use tessera_rl_num::identity::Seed;

use crate::env::{BindError, Layout, Owned, PoolOps, ResetSeed};

/// The caller-side generations of the two halves (ADR-0047 section 5),
/// mirrored from the engine for views to read without the mutex.
pub(crate) type HalfGenerations = Arc<[AtomicU64; 2]>;

/// One received half: which one, and a copy of its rows in `owned` mode.
#[derive(Debug)]
pub(crate) struct HalfOutput {
    pub(crate) half: usize,
    pub(crate) owned: Option<Owned>,
}

/// What a split-phase environment object can do, whatever the environment.
pub(crate) trait SplitOps: Send + Sync {
    fn storage(&self) -> Arc<OutputStorage>;
    fn halves(&self) -> [Range<usize>; 2];
    fn half_generations(&self) -> HalfGenerations;
    /// Held by the one running `recv`.
    fn receiving(&self) -> &AtomicBool;
    fn waiter(&self) -> &Waiter;
    /// Resets every instance; the stored tokens are released first, and
    /// both halves' new tokens are stored.
    fn reset(&self, seed: &ResetSeed, owned: bool) -> Result<[HalfOutput; 2], BindError>;
    fn send(&self, half: u32, actions: &[i32]) -> Result<(), BindError>;
    fn try_recv(&self, owned: bool) -> Result<Option<HalfOutput>, BindError>;
    fn is_poisoned(&self) -> bool;
    /// Drains and joins the workers; idempotent.
    fn close(&self);
}

struct State<E: Environment, M: MaskProvider<E>> {
    pool: Option<PipelinedPool<E, M>>,
    /// The token of each returned half, the borrow behind its NumPy views.
    tokens: [Option<HalfToken<E, M>>; 2],
}

struct SplitPool<E: Environment, M: MaskProvider<E>> {
    state: Mutex<State<E, M>>,
    storage: Arc<OutputStorage>,
    halves: [Range<usize>; 2],
    generations: HalfGenerations,
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

    /// Mirrors the engine's generations, whatever the call's outcome.
    fn mirror(&self, pool: &PipelinedPool<E, M>) {
        for (h, slot) in (0u32..).zip(self.generations.iter()) {
            if let Ok(g) = pool.generation(h) {
                slot.store(g, Ordering::Release);
            }
        }
    }

    fn output(&self, half: usize, token: &HalfToken<E, M>, owned: bool) -> HalfOutput {
        HalfOutput {
            half,
            owned: owned.then(|| copy_token(token, self.terminal)),
        }
    }
}

impl<E, M> SplitOps for SplitPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Send + Sync,
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
        Arc::clone(&self.generations)
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
        let result = pool.reset_all(reseed);
        self.mirror(pool);
        let [a, b] = result?;
        let out = [self.output(0, &a, owned), self.output(1, &b, owned)];
        state.tokens = [Some(a), Some(b)];
        Ok(out)
    }

    fn send(&self, half: u32, actions: &[i32]) -> Result<(), BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        let index = usize::try_from(half).unwrap_or(usize::MAX);
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
        self.mirror(pool);
        result.map_err(|e| {
            if let Some(slot) = state.tokens.get_mut(index) {
                *slot = e.token;
            }
            BindError::Engine(e.error)
        })
    }

    fn try_recv(&self, owned: bool) -> Result<Option<HalfOutput>, BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        let Some(token) = pool.try_recv()? else {
            return Ok(None);
        };
        let half = usize::try_from(token.half()).unwrap_or(usize::MAX);
        let out = self.output(half, &token, owned);
        if let Some(slot) = state.tokens.get_mut(half) {
            *slot = Some(token);
        }
        Ok(Some(out))
    }

    fn is_poisoned(&self) -> bool {
        self.lock().map_or(true, |s| {
            s.pool.as_ref().is_some_and(PipelinedPool::is_poisoned)
        })
    }

    fn close(&self) {
        let taken = self.lock().ok().map(|mut state| {
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

    fn outputs(&self) -> Owned {
        Owned::default()
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
    layout: Layout,
) -> Box<dyn PoolOps>
where
    E: Environment,
    M: MaskProvider<E> + Send + Sync + 'static,
    E: 'static,
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let usize_of = |v: u32| usize::try_from(v).unwrap_or(usize::MAX);
    let halves = pool.halves().map(|r| usize_of(r.start)..usize_of(r.end));
    let waiter = pool.waiter();
    let ops: Arc<dyn SplitOps> = Arc::new(SplitPool {
        state: Mutex::new(State {
            pool: Some(pool),
            tokens: [None, None],
        }),
        storage,
        halves,
        generations: Arc::new([AtomicU64::new(0), AtomicU64::new(0)]),
        receiving: AtomicBool::new(false),
        waiter,
        terminal: layout.terminal,
    });
    Box::new(SplitHandle { ops, layout })
}

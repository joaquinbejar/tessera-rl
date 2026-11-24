// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host pool's chunk kernels (ADR-0062): reset, step and observe of
//! host instances over the engine's output rows. The bookkeeping (episode
//! steps, agent rows, episode end, statistics, auto-reset) is the
//! `EnvPool` kernels', on the same header type and through the same
//! helpers; only the calls into the environment differ.

use core::sync::atomic::{AtomicU64, Ordering};

use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::{
    AbortCode, ActionReject, AgentSlot, Bool8, EpisodeEnd, HostEnvironment, HostOut, HostSeed,
    NoAdaptive, RecordHeader, RecordLayout, Refusals, StepOut, TurnStructure,
};
use tessera_rl_num::fp;
use tessera_rl_num::identity::{RngIdentity, Seed};

use crate::layout::{ActionLayout, MAX_LEAVES};
use crate::options::AutoReset;
use crate::outputs::OutputsView;
use crate::pool::{
    Ending, Reseed, Rows, clear_chunk, clear_terminal, finish_episode, save_terminal, set,
    with_step, write_agent_masks,
};

/// A host pool's record header: the engine fields and one row per agent,
/// no adaptive config.
pub(crate) type HostHeader<const A: usize> = RecordHeader<A, NoAdaptive>;

/// The rows of a host pool: headers and host instances.
pub(crate) type HostRows<'a, E, const A: usize> = Rows<'a, HostHeader<A>, E>;

/// Read-only context of a host step, shared by every chunk.
pub(crate) struct HostCtx<'a> {
    pub(crate) layout: &'a ActionLayout,
    pub(crate) limit: Option<u32>,
    pub(crate) auto_reset: AutoReset,
    /// Agent slots alive after a reset (`HostEnvironment::initial_agents`).
    pub(crate) initial: usize,
    /// Times a host chunk left the thread's FP controls non-canonical.
    pub(crate) fp_resets: &'a AtomicU64,
    /// Aborts the engine applied (ADR-0065 item 1).
    pub(crate) aborts: &'a AtomicU64,
    /// Instances whose `refuse` refused an agent (ADR-0070): the
    /// `refused` and `refuse_code` rows need clearing only if it moved.
    pub(crate) refusals: &'a AtomicU64,
    /// The lowest [`misuse_key`] of a refusal of a slot that was not acting
    /// in this call, or `u64::MAX` (ADR-0070 item 2).
    pub(crate) misuse: &'a AtomicU64,
    /// The pool's host allocation tally (ADR-0062 item 4).
    #[cfg(feature = "debug-checks")]
    pub(crate) allocs: &'a crate::sys::alloc_track::AllocTally,
}

/// Puts the thread back under canonical FP controls after host code ran
/// (after each host `reset` and `step`, and at the end of each chunk,
/// before any other chunk runs on the thread), and counts the times host
/// code had changed them (ADR-0062 item 4).
fn recanonicalize(fp_resets: &AtomicU64) {
    if fp::ensure_canonical() {
        fp_resets.fetch_add(1, Ordering::Relaxed);
    }
}

/// The key of a refusal of slot `agent` of pool instance `instance` that
/// was not acting: ordered by instance, then slot (ADR-0070 item 2).
#[must_use]
pub(crate) fn misuse_key(instance: usize, agent: usize) -> u64 {
    let instance = u64::try_from(instance).unwrap_or(u64::from(u32::MAX));
    let agent = u64::try_from(agent).unwrap_or(u64::from(u32::MAX));
    (instance.min(u64::from(u32::MAX)) << 32) | agent.min(u64::from(u32::MAX))
}

/// Steps every instance of `rows` (after validation); `first` is the pool
/// index of the chunk's first instance.
pub(crate) fn step_chunk<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    rows: &mut HostRows<'_, E, A>,
    first: usize,
) {
    {
        #[cfg(feature = "debug-checks")]
        let _bracket = crate::sys::alloc_track::HostBracket::enter(ctx.allocs);
        step_rows::<E, A>(ctx, rows, first);
    }
    recanonicalize(ctx.fp_resets);
}

fn step_rows<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    rows: &mut HostRows<'_, E, A>,
    first_instance: usize,
) {
    let (dh, ch) = (ctx.layout.discrete.len(), ctx.layout.continuous.len());
    let Rows {
        headers,
        states: instances,
        discrete,
        continuous,
        outputs,
        ..
    } = rows;
    clear_chunk(outputs, headers.len());
    for (i, (header, inst)) in headers.iter_mut().zip(instances.iter_mut()).enumerate() {
        if header.engine().pending_reset.get() {
            // NextStep: reset instead of stepping (validated above).
            let aborted = reset_one::<E, A>(ctx, header, inst, None, outputs, i, ResetRows::Live);
            write_agent_masks(outputs, i, header);
            if !aborted {
                observe_rows::<E, A, false>(header, inst, outputs, i);
            }
            write_pending(outputs, i, header);
            continue;
        }
        let first = i * A;
        let view = Actions::new(
            discrete.get(first * dh..(first + A) * dh).unwrap_or(&[]),
            dh,
            continuous.get(first * ch..(first + A) * ch).unwrap_or(&[]),
            ch,
        );
        let mut codes = [0u32; A];
        let view = if E::REFUSES_AGENTS {
            if let Err(slot) = refuse_one::<E, A>(ctx, header, inst, &view, &mut codes, outputs, i)
            {
                // Not stepped; the pool is poisoned after the call.
                ctx.misuse.fetch_min(
                    misuse_key(first_instance.saturating_add(i), slot),
                    Ordering::Relaxed,
                );
                continue;
            }
            view.with_refusals(&codes)
        } else {
            view
        };
        let aborted = step_one::<E, A>(header, inst, &view, outputs, i, ctx.limit, ctx.fp_resets);
        if aborted {
            ctx.aborts.fetch_add(1, Ordering::Relaxed);
            // No `observe` or `mask` on an aborted state: zero rows (the
            // observation rows were zeroed by `clear_chunk`), false masks.
            zero_rows::<E, A>(outputs, i);
        } else {
            observe_rows::<E, A, false>(header, inst, outputs, i);
        }
        if ctx.auto_reset == AutoReset::SameStep && header.engine().pending_reset.get() {
            save_terminal(outputs, i);
            // Flags, rewards and stats of the ended episode stay set: the
            // reset writes scratch rows.
            let aborted =
                reset_one::<E, A>(ctx, header, inst, None, outputs, i, ResetRows::Scratch);
            write_agent_masks(outputs, i, header);
            if !aborted {
                // The rows hold the terminal observation: zero them again.
                observe_rows::<E, A, true>(header, inst, outputs, i);
            }
            write_pending(outputs, i, header);
        }
        // `step_one` wrote `pending_reset` where the step changed it (an
        // ended or aborted episode); a step whose episode runs on leaves
        // the instance not pending, as its output already says.
    }
}

/// Runs instance `i`'s `refuse` (ADR-0070) into `codes` and, if it
/// refused an agent, copies the verdicts to the instance's `refused` and
/// `refuse_code` rows (zero on entry). Returns the first slot it refused
/// while that slot was not acting, without writing any row.
fn refuse_one<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    header: &HostHeader<A>,
    inst: &E,
    actions: &Actions<'_, ()>,
    codes: &mut [u32; A],
    outputs: &mut OutputsView<'_>,
    i: usize,
) -> Result<(), usize> {
    let mut refused = [Bool8::FALSE; A];
    let misuse = {
        let mut out = Refusals::new(codes, &mut refused, header.agents());
        inst.refuse(actions, &mut out);
        out.misuse()
    };
    recanonicalize(ctx.fp_resets);
    if let Some(slot) = misuse {
        codes.fill(0);
        return Err(slot.index());
    }
    if codes.iter().any(|&c| c != 0) {
        ctx.refusals.fetch_add(1, Ordering::Relaxed);
        if let Some(row) = outputs.refuse_code.get_mut(i * A..(i + 1) * A) {
            row.copy_from_slice(codes);
        }
        if let Some(row) = outputs.refused.get_mut(i * A..(i + 1) * A) {
            row.copy_from_slice(&refused);
        }
    }
    Ok(())
}

/// Whether instance `header` steps in this call: not pending a reset,
/// with at least one required action (an alive, acting slot).
fn steps<const A: usize>(header: &HostHeader<A>) -> bool {
    !header.engine().pending_reset.get()
        && header
            .agents()
            .iter()
            .any(|r| r.alive.get() && r.acting.get())
}

/// Validates the joint action of every instance of `rows` that steps
/// (ADR-0065 item 2), in instance order, stopping at the first rejection,
/// which it returns with the instance's index in `rows`. Runs inside the
/// host allocation bracket and puts the FP controls back after host code,
/// as a step does.
pub(crate) fn validate_chunk<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    rows: &HostRows<'_, E, A>,
) -> Option<(usize, ActionReject)> {
    let rejected = {
        #[cfg(feature = "debug-checks")]
        let _bracket = crate::sys::alloc_track::HostBracket::enter(ctx.allocs);
        validate_rows::<E, A>(ctx, rows)
    };
    recanonicalize(ctx.fp_resets);
    rejected
}

/// Runs `on_rejected` on one instance (ADR-0069) inside the host
/// allocation bracket, and puts the FP controls back after host code, as a
/// step does.
pub(crate) fn notify_rejected<E: HostEnvironment>(
    ctx: &HostCtx<'_>,
    inst: &mut E,
    reject: &ActionReject,
) {
    {
        #[cfg(feature = "debug-checks")]
        let _bracket = crate::sys::alloc_track::HostBracket::enter(ctx.allocs);
        inst.on_rejected(reject);
    }
    recanonicalize(ctx.fp_resets);
}

fn validate_rows<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    rows: &HostRows<'_, E, A>,
) -> Option<(usize, ActionReject)> {
    let (dh, ch) = (ctx.layout.discrete.len(), ctx.layout.continuous.len());
    for (i, (header, inst)) in rows.headers.iter().zip(rows.states.iter()).enumerate() {
        if !steps::<A>(header) {
            continue;
        }
        let first = i * A;
        let view = Actions::new(
            rows.discrete
                .get(first * dh..(first + A) * dh)
                .unwrap_or(&[]),
            dh,
            rows.continuous
                .get(first * ch..(first + A) * ch)
                .unwrap_or(&[]),
            ch,
        );
        let result = inst.validate(&view);
        recanonicalize(ctx.fp_resets);
        if let Err(reject) = result {
            return Some((i, reject));
        }
    }
    None
}

/// Resets every instance of `rows`; `first` is the position of the chunk's
/// first instance among the reset instances (for the seeds).
pub(crate) fn reset_chunk<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    rows: &mut HostRows<'_, E, A>,
    reseed: Reseed<'_>,
    first: usize,
) {
    {
        #[cfg(feature = "debug-checks")]
        let _bracket = crate::sys::alloc_track::HostBracket::enter(ctx.allocs);
        let Rows {
            headers,
            states: instances,
            outputs,
            ..
        } = rows;
        clear_chunk(outputs, headers.len());
        for (i, (header, inst)) in headers.iter_mut().zip(instances.iter_mut()).enumerate() {
            let aborted = reset_one::<E, A>(
                ctx,
                header,
                inst,
                reseed.seed(first + i),
                outputs,
                i,
                ResetRows::Live,
            );
            write_agent_masks(outputs, i, header);
            if !aborted {
                observe_rows::<E, A, false>(header, inst, outputs, i);
            }
            write_pending(outputs, i, header);
        }
    }
    recanonicalize(ctx.fp_resets);
}

/// Copies instance `i`'s pending-reset flag from its header to the
/// `pending_reset` output (ADR-0065 item 1), at the end of every call that
/// runs the instance, so the output equals the header for every instance
/// at all times.
pub(crate) fn write_pending<const A: usize>(
    outputs: &mut OutputsView<'_>,
    i: usize,
    header: &HostHeader<A>,
) {
    set(
        outputs.pending_reset,
        i,
        header.engine().pending_reset.get(),
    );
}

/// Resets instance `i`: reseeds it (or advances its episode index), makes
/// slots `0..initial` alive and the others empty, and calls the environment's `reset` with the
/// instance's seed. Rewards the reset writes are its outputs; an episode
/// end it reports is ignored.
pub(crate) fn reset_one<E: HostEnvironment, const A: usize>(
    ctx: &HostCtx<'_>,
    header: &mut HostHeader<A>,
    inst: &mut E,
    seed: Option<Seed>,
    outputs: &mut OutputsView<'_>,
    i: usize,
    rows: ResetRows,
) -> bool {
    let initial = ctx.initial;
    let engine = header.engine();
    let mut episode = engine.identity.episode_index();
    if let Some(seed) = seed {
        let id = engine.identity;
        header.engine_mut().identity =
            RngIdentity::from_parts(seed, id.branch_salt(), id.logical_id(), 0, 0);
        episode = 0;
    } else if !engine.fresh.get() {
        // Checked at the boundary.
        episode = episode.saturating_add(1);
    }
    let identity = with_step(header, episode, 0);
    let engine = header.engine_mut();
    // The ended step's end: a `SameStep` reset that aborts restores it, as
    // its instance stays pending with that episode ended (issue #290).
    let ended = engine.end_kind;
    engine.fresh = Bool8::FALSE;
    engine.pending_reset = Bool8::FALSE;
    engine.end_kind = 0;
    engine.episode_steps = 0;
    // `initial` is checked at construction (ADR-0065).
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let alive = Bool8::from(slot < initial);
        row.alive = alive;
        row.acting = alive;
        row.episode_return = 0.0;
        row.illegal_actions = 0;
        row.invalid_agent_events = 0;
    }
    let seed = HostSeed::new(identity.seed().get(), identity.episode_index());
    let mut abort = None;
    match rows {
        ResetRows::Live => {
            let range = i * A..(i + 1) * A;
            let OutputsView {
                rewards,
                agent_terminated,
                agent_truncated,
                died,
                spawned,
                ..
            } = outputs;
            if let (
                Some(reward_row),
                Some(term_row),
                Some(trunc_row),
                Some(died_row),
                Some(spawned_row),
            ) = (
                rewards.get_mut(range.clone()),
                agent_terminated.get_mut(range.clone()),
                agent_truncated.get_mut(range.clone()),
                died.get_mut(range.clone()),
                spawned.get_mut(range),
            ) {
                let mut out = HostOut::new(StepOut::new(
                    reward_row,
                    term_row,
                    trunc_row,
                    died_row,
                    spawned_row,
                ));
                inst.reset(seed, &mut out);
                abort = out.aborted();
            }
        }
        ResetRows::Scratch => {
            // On the stack: the hot path allocates nothing.
            let mut rewards = [0.0f32; A];
            let (mut term, mut trunc, mut died, mut spawned) = (
                [Bool8::FALSE; A],
                [Bool8::FALSE; A],
                [Bool8::FALSE; A],
                [Bool8::FALSE; A],
            );
            let mut out = HostOut::new(StepOut::new(
                &mut rewards,
                &mut term,
                &mut trunc,
                &mut died,
                &mut spawned,
            ));
            inst.reset(seed, &mut out);
            abort = out.aborted();
        }
    }
    recanonicalize(ctx.fp_resets);
    if let Some(code) = abort {
        ctx.aborts.fetch_add(1, Ordering::Relaxed);
        abort_reset::<E, A>(header, outputs, i, rows, code, ended);
        return true;
    }
    update_acting::<E, A>(header, inst);
    false
}

/// Applies the abort of instance `i`'s reset (ADR-0065 item 1). The
/// instance stays pending a reset, no slot acts, and its rows are zeros
/// with false masks (no `observe`, `mask` or `acting`).
///
/// - `Live` rows (an explicit, `NextStep` or retried reset): the row is the
///   reset's: every live agent (slots `0..initial_agents`) truncated, the
///   rewards the reset wrote kept, its deaths and spawns discarded, a
///   zero-length episode and the reset's `abort_code`.
/// - `Scratch` rows (a `SameStep` reset): the row is the ended step's and
///   keeps its flags and rewards; it reports the reset's code unless the
///   step itself aborted, whose code it keeps. No episode is reported for
///   the reset. The header keeps the ended step's `end_kind` (`ended`),
///   so a pending instance always records an episode end (issue #290).
fn abort_reset<E: HostEnvironment, const A: usize>(
    header: &mut HostHeader<A>,
    outputs: &mut OutputsView<'_>,
    i: usize,
    rows: ResetRows,
    code: AbortCode,
    ended: u8,
) {
    for row in header.agents_mut() {
        row.acting = Bool8::FALSE;
    }
    match rows {
        ResetRows::Live => {
            let mut total = 0.0f32;
            for (slot, row) in header.agents_mut().iter_mut().enumerate() {
                let k = i * A + slot;
                let reward = outputs.rewards.get(k).copied().unwrap_or(0.0);
                row.episode_return += reward;
                total += row.episode_return;
                set(outputs.died, k, false);
                set(outputs.spawned, k, false);
                set(outputs.agent_terminated, k, false);
                set(outputs.agent_truncated, k, row.alive.get());
            }
            let end = Ending {
                terminated: false,
                truncated: true,
            };
            finish_episode(header, outputs, i, end, 0, total);
            mark_aborted(outputs, i, code);
            // The row is this reset's: its terminal observation is the
            // aborted state's, zeros, not an earlier episode's (ADR-0065
            // item 1). A no-op outside `SameStep` (no terminal buffer).
            clear_terminal(outputs, i);
        }
        ResetRows::Scratch => {
            let engine = header.engine_mut();
            engine.pending_reset = Bool8::TRUE;
            // The step that ended the episode set it (`finish_episode`).
            debug_assert_ne!(ended, 0, "a same_step reset runs after an ended step");
            engine.end_kind = ended;
            if !outputs.aborted.get(i).is_some_and(|f| f.get()) {
                mark_aborted(outputs, i, code);
            }
        }
    }
    zero_rows::<E, A>(outputs, i);
}

/// Where a reset's `StepOut` writes (the `HostEnvironment::reset` doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResetRows {
    /// The instance's output rows, cleared before: an explicit reset or a
    /// `NextStep` auto-reset, whose outputs these are.
    Live,
    /// Scratch rows, discarded: a `SameStep` auto-reset, whose instance rows
    /// hold the ended step's rewards and flags.
    Scratch,
}

/// Sets the acting flag of every alive agent from the turn structure
/// (ADR-0013), after the step's agent events, as `EnvPool` does.
pub(crate) fn update_acting<E: HostEnvironment, const A: usize>(
    header: &mut HostHeader<A>,
    inst: &E,
) {
    if E::TURNS == TurnStructure::Simultaneous {
        return;
    }
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let on = row.alive.get()
            && u16::try_from(slot)
                .ok()
                .and_then(|s| AgentSlot::new(s, A))
                .is_some_and(|s| inst.acting(s));
        row.acting = Bool8::from(on);
    }
}

/// One step of instance `i`: the environment's `step`, then the engine's
/// bookkeeping exactly as `EnvPool`'s (step limit, agent events, terminal
/// flush, episode end and statistics).
fn step_one<E: HostEnvironment, const A: usize>(
    header: &mut HostHeader<A>,
    inst: &mut E,
    actions: &Actions<'_, ()>,
    outputs: &mut OutputsView<'_>,
    i: usize,
    limit: Option<u32>,
    fp_resets: &AtomicU64,
) -> bool {
    let range = i * A..(i + 1) * A;
    let t = header.engine().episode_steps.saturating_add(1);
    let episode = header.engine().identity.episode_index();
    with_step(header, episode, t);
    // The caller cleared instance `i`'s rows (`clear_chunk`) and its abort
    // rows (`HostPool::clear_abort_rows`).
    let end = {
        let OutputsView {
            rewards,
            agent_terminated,
            agent_truncated,
            died,
            spawned,
            ..
        } = outputs;
        let (Some(reward_row), Some(term_row), Some(trunc_row), Some(died_row), Some(spawned_row)) = (
            rewards.get_mut(range.clone()),
            agent_terminated.get_mut(range.clone()),
            agent_truncated.get_mut(range.clone()),
            died.get_mut(range.clone()),
            spawned.get_mut(range),
        ) else {
            return false;
        };
        let mut out = HostOut::new(StepOut::new(
            reward_row,
            term_row,
            trunc_row,
            died_row,
            spawned_row,
        ));
        inst.step(actions, &mut out);
        let end = (out.episode_end(), out.aborted());
        recanonicalize(fp_resets);
        end
    };
    let (end, abort) = end;
    if let Some(code) = abort {
        abort_step::<A>(header, outputs, i, t, code);
        return true;
    }
    // TimeLimit semantics (SPEC 5.4), as in `EnvPool`.
    let end = Ending {
        terminated: end == Some(EpisodeEnd::Terminated),
        truncated: end == Some(EpisodeEnd::Truncated) || limit == Some(t),
    };
    let end = (end.terminated || end.truncated).then_some(end);
    header.engine_mut().episode_steps = t;
    let mut total = 0.0f32;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * A + slot;
        let reward = outputs.rewards.get(k).copied().unwrap_or(0.0);
        row.episode_return += reward;
        total += row.episode_return;
        let mut died = outputs.died.get(k).is_some_and(|f| f.get());
        let spawned = outputs.spawned.get(k).is_some_and(|f| f.get());
        if died && !row.alive.get() {
            // An invalid agent event: never applied, counted and marked.
            died = false;
            row.invalid_agent_events = row.invalid_agent_events.saturating_add(1);
            set(outputs.invalid_agent_event, k, true);
            set(outputs.agent_terminated, k, false);
        }
        if died {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
        if spawned {
            // Without `SPAWNS`, every spawn is invalid (ADR-0065).
            let valid = E::SPAWNS && !died && !row.alive.get();
            if !valid {
                row.invalid_agent_events = row.invalid_agent_events.saturating_add(1);
                set(outputs.invalid_agent_event, k, true);
                set(outputs.spawned, k, false);
            }
            if valid {
                row.alive = Bool8::TRUE;
                row.acting = Bool8::TRUE;
                // Generations count occupants modulo 2^32: an explicit,
                // documented wrap (unreachable in practice).
                row.generation = row.generation.wrapping_add(1);
            }
        }
    }
    // Terminal flush (SPEC 12).
    if let Some(end) = end {
        for (slot, row) in header.agents().iter().enumerate() {
            if row.alive.get() {
                let k = i * A + slot;
                if end.terminated {
                    set(outputs.agent_terminated, k, true);
                }
                if end.truncated {
                    set(outputs.agent_truncated, k, true);
                }
            }
        }
    }
    update_acting::<E, A>(header, inst);
    if let Some(end) = end {
        finish_episode(header, outputs, i, end, t, total);
        write_pending(outputs, i, header);
    }
    write_agent_masks(outputs, i, header);
    false
}

/// Applies the abort of instance `i`'s step `t` (ADR-0065 item 1): the
/// episode ends with every agent alive on entry truncated and none
/// terminated, the step's rewards are kept, its deaths and spawns are
/// discarded (never applied, not charged), and no slot acts. The instance
/// is pending a reset.
fn abort_step<const A: usize>(
    header: &mut HostHeader<A>,
    outputs: &mut OutputsView<'_>,
    i: usize,
    t: u32,
    code: AbortCode,
) {
    header.engine_mut().episode_steps = t;
    let mut total = 0.0f32;
    for (slot, row) in header.agents_mut().iter_mut().enumerate() {
        let k = i * A + slot;
        let reward = outputs.rewards.get(k).copied().unwrap_or(0.0);
        row.episode_return += reward;
        total += row.episode_return;
        set(outputs.died, k, false);
        set(outputs.spawned, k, false);
        set(outputs.agent_terminated, k, false);
        set(outputs.agent_truncated, k, row.alive.get());
        row.acting = Bool8::FALSE;
    }
    let end = Ending {
        terminated: false,
        truncated: true,
    };
    finish_episode(header, outputs, i, end, t, total);
    write_agent_masks(outputs, i, header);
    mark_aborted(outputs, i, code);
    write_pending(outputs, i, header);
}

/// Clears instance `i`'s `aborted` and `abort_code` outputs, for a call
/// that runs only some instances (an explicit reset).
pub(crate) fn clear_abort_row(outputs: &mut OutputsView<'_>, i: usize) {
    set(outputs.aborted, i, false);
    if let Some(c) = outputs.abort_code.get_mut(i) {
        *c = 0;
    }
}

/// Sets instance `i`'s `aborted` and `abort_code` outputs.
fn mark_aborted(outputs: &mut OutputsView<'_>, i: usize, code: AbortCode) {
    set(outputs.aborted, i, true);
    if let Some(c) = outputs.abort_code.get_mut(i) {
        *c = code.get();
    }
}

/// Zeroes instance `i`'s observation rows and sets its masks false: the
/// rows of an aborted state, never observed (ADR-0065 item 1).
pub(crate) fn zero_rows<E: HostEnvironment, const A: usize>(
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    let OutputsView {
        observations,
        leaf_bytes,
        masks,
        mask_len,
        ..
    } = outputs;
    for (buf, &bytes) in observations.iter_mut().zip(leaf_bytes.iter()) {
        if let Some(rows) = buf.get_mut(i * A * bytes..(i + 1) * A * bytes) {
            rows.fill(0);
        }
    }
    if E::MASKS
        && let Some(m) = masks.get_mut(i * A * *mask_len..(i + 1) * A * *mask_len)
    {
        m.fill(Bool8::FALSE);
    }
}

/// `observe_rows` for rows that may hold other bytes (an explicit reset).
pub(crate) fn observe_instance<E: HostEnvironment, const A: usize>(
    header: &HostHeader<A>,
    inst: &E,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    observe_rows::<E, A, true>(header, inst, outputs, i);
}

/// Writes instance `i`'s observations and masks: every leaf row zero on
/// entry (ADR-0058; `ZERO` when the rows may hold other bytes), the mask
/// all false for dead agents, and the environment's `observe` and `mask`
/// for alive ones.
fn observe_rows<E: HostEnvironment, const A: usize, const ZERO: bool>(
    header: &HostHeader<A>,
    inst: &E,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    let OutputsView {
        observations,
        leaf_bytes,
        masks,
        mask_len,
        ..
    } = outputs;
    for (slot, row) in header.agents().iter().enumerate() {
        let k = i * A + slot;
        let mut views: [&mut [u8]; MAX_LEAVES] = Default::default();
        let mut used = 0;
        for ((buf, &bytes), view) in observations
            .iter_mut()
            .zip(leaf_bytes.iter())
            .zip(views.iter_mut())
        {
            if let Some(rows) = buf.get_mut(k * bytes..(k + 1) * bytes) {
                if ZERO {
                    rows.fill(0);
                }
                *view = rows;
            }
            used += 1;
        }
        let mut mask = if E::MASKS {
            masks.get_mut(k * *mask_len..(k + 1) * *mask_len)
        } else {
            None
        };
        if let Some(m) = mask.as_deref_mut() {
            m.fill(Bool8::FALSE);
        }
        if !row.alive.get() {
            continue;
        }
        let (Some(agent), Some(leaves)) = (
            AgentSlot::new(u16::try_from(slot).unwrap_or(u16::MAX), A),
            views.get_mut(..used),
        ) else {
            continue;
        };
        inst.observe(agent, &mut ObsWriter::new(leaves));
        if let Some(m) = mask
            && !m.is_empty()
        {
            inst.mask(agent, &mut MaskWriter::new(m));
        }
    }
}

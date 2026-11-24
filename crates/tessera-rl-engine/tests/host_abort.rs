// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aborted episodes in host pools (ADR-0065 item 1): aborts in `step` and
//! in `reset`, in every auto-reset mode. Every test runs in Inline mode and
//! in two Sync shapes (2 workers with chunks of 1, 3 workers with chunks of
//! 2, over 8 instances, so chunks spread over workers), and each Sync run's
//! outputs must equal the Inline run's, call for call.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod host_script;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use host_script::{
    ABORT, ABORT_THEN_TERMINATE, BAD_EVENT, BAD_EVENT_THEN_ABORT, DIE, NAN, NOTHING, RESET_CODE,
    RESET_REWARD, SLOTS, STEP_CODE, Script, ScriptConfig, Shared, TERMINATE, TERMINATE_THEN_ABORT,
    TRUNCATE,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, HostPool, OutputField, Outputs, PoolOptions, Reseed,
    SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: usize = 8;
const MODES: [AutoReset; 3] = [
    AutoReset::NextStep,
    AutoReset::SameStep,
    AutoReset::Disabled,
];

type Pool<const T: u8> = HostPool<Script<T, false>, SLOTS>;

/// Where a test runs: Inline (`workers == 0`) or Sync.
#[derive(Clone, Copy, Debug)]
struct Rt {
    workers: usize,
    chunk: u32,
}

const RTS: [Rt; 3] = [
    Rt {
        workers: 0,
        chunk: 64,
    },
    Rt {
        workers: 2,
        chunk: 1,
    },
    Rt {
        workers: 3,
        chunk: 2,
    },
];

fn options(mode: AutoReset, rt: Rt) -> PoolOptions {
    let mut o = PoolOptions::new(
        NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
        Seed::new(9),
    );
    o.auto_reset = mode;
    o.chunk_size = NonZeroU32::new(rt.chunk).unwrap();
    o
}

fn pool_with<const T: u8>(shared: &Arc<Shared>, o: PoolOptions, rt: Rt) -> Pool<T> {
    let p = Pool::<T>::new(ScriptConfig::new(shared), o).unwrap();
    if rt.workers > 0 {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(rt.workers).unwrap()))
            .unwrap()
    } else {
        p
    }
}

fn pool<const T: u8>(shared: &Arc<Shared>, mode: AutoReset, rt: Rt) -> Pool<T> {
    pool_with(shared, options(mode, rt), rt)
}

/// Runs `case` in every runtime; each returns its trace (outputs after the
/// calls it chose), which must equal the Inline run's.
fn each_rt(case: impl Fn(Rt) -> Vec<Vec<Vec<u8>>>) {
    let traces: Vec<_> = RTS.iter().map(|&rt| case(rt)).collect();
    for (rt, trace) in RTS.iter().zip(&traces).skip(1) {
        assert_eq!(trace, &traces[0], "{rt:?} differs from Inline");
    }
}

/// `[instance + 1, step + 1, episode index + 1]`, as the script observes
/// it, of agent `slot` in observation rows `rows`.
fn obs_values(rows: &[u8], slot: usize) -> [i32; 3] {
    let at = |k: usize| {
        let o = slot * 12 + k * 4;
        i32::from_le_bytes(rows[o..o + 4].try_into().unwrap())
    };
    [at(0), at(1), at(2)]
}

/// Instance `i`'s row of a per-agent flag output read through its storage
/// region (`died` and `spawned` have no typed accessor).
fn region_flags(out: &Outputs, field: OutputField, i: usize) -> Vec<bool> {
    let storage = out.storage();
    let region = storage.region(field).unwrap();
    // SAFETY: the region is `len` initialized bytes (Bool8) of a storage kept
    // alive by `storage`, and no engine call runs meanwhile.
    #[expect(unsafe_code, reason = "reading a borrowed output region")]
    let all = unsafe { core::slice::from_raw_parts(region.ptr, region.len) };
    all[i * SLOTS..(i + 1) * SLOTS]
        .iter()
        .map(|&b| b != 0)
        .collect()
}

fn terminal_row(out: &Outputs, i: usize) -> &[u8] {
    let bytes = 3 * 4;
    &out.terminal_observation(0).unwrap()[i * SLOTS * bytes..(i + 1) * SLOTS * bytes]
}

/// Every agent of every instance does nothing, except `(instance, slot,
/// action)` entries.
fn actions(special: &[(usize, usize, i32)]) -> Vec<i32> {
    let mut a = vec![NOTHING; N * SLOTS];
    for &(i, s, v) in special {
        a[i * SLOTS + s] = v;
    }
    a
}

fn step<const T: u8>(p: &mut Pool<T>, special: &[(usize, usize, i32)]) -> Result<(), EngineError> {
    p.step(ActionInput {
        discrete: &actions(special),
        continuous: &[],
    })
    .map(|_| ())
}

fn flags(f: &[tessera_rl_core::Bool8], i: usize) -> Vec<bool> {
    f[i * SLOTS..(i + 1) * SLOTS]
        .iter()
        .map(|b| b.get())
        .collect()
}

fn obs_row(out: &Outputs, i: usize) -> &[u8] {
    let bytes = 3 * 4;
    &out.observation(0).unwrap()[i * SLOTS * bytes..(i + 1) * SLOTS * bytes]
}

fn mask_row(out: &Outputs, i: usize) -> &[tessera_rl_core::Bool8] {
    let len = out.masks().len() / (N * SLOTS);
    &out.masks()[i * SLOTS * len..(i + 1) * SLOTS * len]
}

/// Every output a host pool writes, for whole-call comparisons.
fn snapshot(out: &Outputs) -> Vec<Vec<u8>> {
    let b = |f: &[tessera_rl_core::Bool8]| f.iter().map(|x| u8::from(x.get())).collect::<Vec<_>>();
    let f = |f: &[f32]| {
        f.iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut v = vec![
        out.observation(0).unwrap().to_vec(),
        out.terminal_observation(0)
            .map(<[u8]>::to_vec)
            .unwrap_or_default(),
        f(out.rewards()),
        b(out.agent_terminated()),
        b(out.agent_truncated()),
        b(out.alive()),
        b(out.acting()),
        b(out.episode_terminated()),
        b(out.episode_truncated()),
        f(out.episode_return()),
        f(out.episode_length()),
        b(out.masks()),
        b(out.aborted()),
        out.abort_code()
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect(),
        b(out.pending_reset()),
    ];
    v.push(
        out.generations()
            .iter()
            .flat_map(|g| g.to_le_bytes())
            .collect(),
    );
    v
}

#[test]
fn test_host_abort_in_step_truncates_live_agents_and_keeps_rewards() {
    for mode in MODES {
        each_rt(|rt| {
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            p.reset_all(Reseed::Keep).unwrap();
            // Instance 1: agent 2 dies first, then a step where agent 0
            // dies, spawns its own slot and aborts while agent 1 dies (all
            // discarded). Agent 2 is dead, so it does not act.
            step(&mut p, &[(1, 2, DIE)]).unwrap();
            let steps_before = Shared::get(&shared.steps, 1);
            step(&mut p, &[(1, 0, BAD_EVENT_THEN_ABORT), (1, 1, DIE)]).unwrap();
            assert_eq!(Shared::get(&shared.steps, 1), steps_before + 1);
            let out = p.outputs().unwrap();
            let ctx = format!("{mode:?} {rt:?}");
            assert!(out.aborted()[1].get(), "{ctx}");
            assert_eq!(out.abort_code()[1], STEP_CODE, "{ctx}");
            for other in (0..N).filter(|&i| i != 1) {
                assert!(!out.aborted()[other].get(), "{ctx}");
                assert_eq!(out.abort_code()[other], 0, "{ctx}");
            }
            assert_eq!(out.rewards()[SLOTS..2 * SLOTS], [1.0, 2.0, 3.0], "{ctx}");
            assert!(out.episode_truncated()[1].get(), "{ctx}");
            assert!(!out.episode_terminated()[1].get(), "{ctx}");
            assert_eq!(flags(out.agent_terminated(), 1), [false; 3], "{ctx}");
            // The step's death and spawn are discarded from the rows too.
            assert_eq!(region_flags(out, OutputField::Died, 1), [false; 3], "{ctx}");
            assert_eq!(
                region_flags(out, OutputField::Spawned, 1),
                [false; 3],
                "{ctx}"
            );
            // Every agent alive on entry is truncated, in every mode.
            assert_eq!(
                flags(out.agent_truncated(), 1),
                [true, true, false],
                "{ctx}"
            );
            if mode == AutoReset::SameStep {
                // Reset in the same call: the new episode's rows.
                assert_eq!(flags(out.alive(), 1), [true; 3], "{ctx}");
                assert!(!out.pending_reset()[1].get(), "{ctx}");
            } else {
                // Liveness is that of entry: agent 1's death and the spawn
                // of slot 2 were discarded; agent 2 was dead on entry.
                assert_eq!(
                    flags(out.agent_truncated(), 1),
                    [true, true, false],
                    "{ctx}"
                );
                assert_eq!(flags(out.alive(), 1), [true, true, false], "{ctx}");
                assert_eq!(flags(out.acting(), 1), [false; 3], "{ctx}");
                assert!(out.pending_reset()[1].get(), "{ctx}");
                assert!(obs_row(out, 1).iter().all(|&b| b == 0), "{ctx}");
                assert!(mask_row(out, 1).iter().all(|m| !m.get()), "{ctx}");
            }
            assert_eq!(p.aborts(), 1, "{ctx}");
            assert!(!p.is_poisoned(), "{ctx}");
            vec![snapshot(out)]
        });
    }
}

#[test]
fn test_host_abort_wins_over_end_episode_in_either_order() {
    each_rt(|rt| {
        let mut trace = Vec::new();
        for action in [TERMINATE_THEN_ABORT, ABORT_THEN_TERMINATE] {
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
            p.reset_all(Reseed::Keep).unwrap();
            step(&mut p, &[(0, 1, action)]).unwrap();
            let out = p.outputs().unwrap();
            assert!(out.aborted()[0].get());
            assert_eq!(out.abort_code()[0], STEP_CODE + 1);
            assert!(out.episode_truncated()[0].get() && !out.episode_terminated()[0].get());
            assert_eq!(flags(out.agent_terminated(), 0), [false; 3]);
            assert_eq!(flags(out.agent_truncated(), 0), [true; 3]);
            trace.push(snapshot(out));
        }
        // A later agent's abort overrides an earlier one: the last code wins.
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(0, 0, ABORT), (0, 2, ABORT)]).unwrap();
        assert_eq!(p.outputs().unwrap().abort_code()[0], STEP_CODE + 2);
        assert_eq!(p.aborts(), 1);
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_calls_no_observe_mask_or_acting_on_the_aborted_state() {
    each_rt(|rt| {
        let shared = Shared::new();
        // Mixed, so `acting` is consulted.
        let mut p = pool::<2>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        let (o, m, a) = (
            Shared::get(&shared.observes, 2),
            Shared::get(&shared.masks, 2),
            Shared::get(&shared.actings, 2),
        );
        let o3 = Shared::get(&shared.observes, 3);
        step(&mut p, &[(2, 0, ABORT)]).unwrap();
        assert_eq!(Shared::get(&shared.observes, 2), o);
        assert_eq!(Shared::get(&shared.masks, 2), m);
        assert_eq!(Shared::get(&shared.actings, 2), a);
        // A step that does not abort observes every alive agent.
        assert_eq!(Shared::get(&shared.observes, 3), o3 + 3);
        let out = p.outputs().unwrap();
        assert!(obs_row(out, 2).iter().all(|&b| b == 0));
        // Observed at step 1 of episode 0.
        assert_eq!(obs_values(obs_row(out, 3), 0), [4, 2, 1]);
        assert!(mask_row(out, 2).iter().all(|m| !m.get()));
        assert_eq!(flags(out.acting(), 2), [false; 3]);
        vec![snapshot(out)]
    });
}

#[test]
fn test_host_abort_next_step_resets_on_the_next_call_disabled_raises() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(0, 0, ABORT)]).unwrap();
        let (steps, resets) = (
            Shared::get(&shared.steps, 0),
            Shared::get(&shared.resets, 0),
        );
        // The next call resets instance 0 instead of stepping it; its
        // actions are ignored (an out-of-range action is accepted).
        let mut a = actions(&[]);
        a[0] = 1000;
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
        assert_eq!(Shared::get(&shared.steps, 0), steps);
        assert_eq!(Shared::get(&shared.resets, 0), resets + 1);
        let out = p.outputs().unwrap();
        assert!(!out.aborted()[0].get() && out.abort_code()[0] == 0);
        assert!(!out.pending_reset()[0].get());
        assert_eq!(flags(out.agent_truncated(), 0), [false; 3]);
        assert_eq!(flags(out.acting(), 0), [true; 3]);
        let trace = vec![snapshot(out)];
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::Disabled, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(0, 0, ABORT)]).unwrap();
        let err = step(&mut p, &[]).unwrap_err();
        assert!(matches!(err, EngineError::PendingReset { .. }), "{err:?}");
        assert!(!p.is_poisoned());
        trace
    });
}

#[test]
fn test_host_abort_same_step_reset_succeeds_with_the_steps_code() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::SameStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(3, 1, ABORT)]).unwrap();
        let out = p.outputs().unwrap();
        assert!(out.aborted()[3].get());
        assert_eq!(out.abort_code()[3], STEP_CODE + 1);
        assert!(!out.pending_reset()[3].get());
        // The terminal observation is the zero row; the observation rows
        // hold the new episode's start.
        assert!(terminal_row(out, 3).iter().all(|&b| b == 0));
        // The new episode (index 1) at its start (step 0).
        assert_eq!(obs_values(obs_row(out, 3), 0), [4, 1, 2]);
        assert_eq!(flags(out.acting(), 3), [true; 3]);
        let mut trace = vec![snapshot(out)];
        let steps = Shared::get(&shared.steps, 3);
        step(&mut p, &[]).unwrap();
        assert_eq!(Shared::get(&shared.steps, 3), steps + 1);
        assert!(!p.outputs().unwrap().aborted()[3].get());
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_mixed_same_step_reset_into_no_acting_agent_is_not_pending() {
    // The case a rule over `aborted` and `acting` gets wrong: aborted, no
    // slot acts, yet the instance is not pending, and the next call steps
    // it.
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<2>(&shared, AutoReset::SameStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        shared.mixed_acting.store(false, Ordering::Relaxed);
        step(&mut p, &[(1, 0, ABORT)]).unwrap();
        let out = p.outputs().unwrap();
        assert!(out.aborted()[1].get());
        assert_eq!(flags(out.acting(), 1), [false; 3]);
        assert!(!out.pending_reset()[1].get());
        let mut trace = vec![snapshot(out)];
        let steps = Shared::get(&shared.steps, 1);
        step(&mut p, &[]).unwrap();
        assert_eq!(Shared::get(&shared.steps, 1), steps + 1);
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_sequential_passes_the_turn_check() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        each_rt(|rt| {
            let shared = Shared::new();
            let mut p = pool::<1>(&shared, mode, rt);
            p.reset_all(Reseed::Keep).unwrap();
            step(&mut p, &[(0, 0, ABORT)]).unwrap();
            assert!(!p.is_poisoned());
            let mut trace = vec![snapshot(p.outputs().unwrap())];
            // The next call's turn check holds.
            step(&mut p, &[]).unwrap();
            assert!(!p.is_poisoned(), "{mode:?}");
            trace.push(snapshot(p.outputs().unwrap()));
            trace
        });
    }
}

#[test]
fn test_host_aborts_counts_applied_aborts_only() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(0, 0, ABORT), (6, 1, ABORT)]).unwrap();
        assert_eq!(p.aborts(), 2);
        let mut trace = vec![snapshot(p.outputs().unwrap())];
        // A call that does not run instance 0 adds nothing.
        p.reset(&[1], Reseed::Keep).unwrap();
        assert_eq!(p.aborts(), 2);
        trace.push(snapshot(p.outputs().unwrap()));
        // The retries succeed: no abort.
        step(&mut p, &[]).unwrap();
        assert_eq!(p.aborts(), 2);
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_episode_stats_count_the_steps_run_and_kept_rewards() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[]).unwrap();
        step(&mut p, &[(0, 0, ABORT)]).unwrap();
        let out = p.outputs().unwrap();
        assert_eq!(out.episode_length()[0], 2.0);
        // Two steps of 1 + 2 + 3 per step; the reset's 0.5 is not part of
        // the episode's return (it is the reset call's output).
        assert_eq!(out.episode_return()[0], 12.0);
        vec![snapshot(out)]
    });
}

#[test]
fn test_host_abort_debug_checks_nan_poisons_discarded_events_do_not() {
    each_rt(|rt| {
        // A NaN reward written before an abort still poisons (debug-checks).
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        let r = step(&mut p, &[(0, 1, NAN), (0, 0, ABORT)]);
        if cfg!(feature = "debug-checks") {
            assert!(r.is_err() && p.is_poisoned(), "{r:?}");
        } else {
            assert!(r.is_ok() && !p.is_poisoned());
        }
        // An invalid event written before an abort is discarded: no poison.
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[(5, 1, BAD_EVENT_THEN_ABORT)]).unwrap();
        assert!(!p.is_poisoned());
        let trace = vec![snapshot(p.outputs().unwrap())];
        // The same event in a call that does not abort still poisons.
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        let r = step(&mut p, &[(5, 1, BAD_EVENT)]);
        if cfg!(feature = "debug-checks") {
            assert!(
                matches!(r, Err(EngineError::InvalidAgentEvent { instance: 5, .. })),
                "{r:?}"
            );
        } else {
            assert!(r.is_ok());
        }
        trace
    });
}

#[test]
fn test_host_abort_mixed_script_in_every_mode() {
    // A longer script of aborts, deaths and episode ends over all eight
    // instances; `each_rt` compares every call's outputs with Inline.
    for mode in MODES {
        each_rt(|rt| {
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            p.reset_all(Reseed::Keep).unwrap();
            let script: [&[(usize, usize, i32)]; 5] = [
                &[(0, 0, ABORT), (3, 2, DIE), (7, 1, ABORT)],
                &[(1, 1, ABORT), (2, 0, TERMINATE), (6, 0, TRUNCATE)],
                &[(3, 0, ABORT_THEN_TERMINATE), (4, 2, ABORT)],
                &[(5, 1, TERMINATE_THEN_ABORT)],
                &[],
            ];
            let mut trace = Vec::new();
            for special in script {
                if step(&mut p, special).is_err() {
                    // Disabled: reset the pending instances, then go on.
                    let pending: Vec<u32> = p
                        .outputs()
                        .unwrap()
                        .pending_reset()
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.get())
                        .map(|(i, _)| u32::try_from(i).unwrap())
                        .collect();
                    p.reset(&pending, Reseed::Keep).unwrap();
                }
                trace.push(snapshot(p.outputs().unwrap()));
            }
            trace.push(vec![p.aborts().to_le_bytes().to_vec()]);
            trace
        });
    }
}

// -- aborts in reset ------------------------------------------------------------------

fn rewards(out: &Outputs, i: usize) -> Vec<f32> {
    out.rewards()[i * SLOTS..(i + 1) * SLOTS].to_vec()
}

/// The row of an aborted reset of instance `i` with `Live` rows (ADR-0065
/// item 8): under `SameStep` its terminal observation row is zeros too.
fn assert_reset_abort_row(out: &Outputs, i: usize, ctx: &str) {
    assert!(out.aborted()[i].get(), "{ctx}");
    assert_eq!(
        out.abort_code()[i],
        RESET_CODE + u32::try_from(i).unwrap(),
        "{ctx}"
    );
    assert_eq!(flags(out.agent_truncated(), i), [true; 3], "{ctx}");
    assert_eq!(flags(out.agent_terminated(), i), [false; 3], "{ctx}");
    assert_eq!(region_flags(out, OutputField::Died, i), [false; 3], "{ctx}");
    assert_eq!(
        region_flags(out, OutputField::Spawned, i),
        [false; 3],
        "{ctx}"
    );
    assert!(
        out.episode_truncated()[i].get() && !out.episode_terminated()[i].get(),
        "{ctx}"
    );
    assert_eq!(rewards(out, i), [RESET_REWARD, 0.0, 0.0], "{ctx}");
    assert_eq!(out.episode_length()[i], 0.0, "{ctx}");
    assert_eq!(out.episode_return()[i], RESET_REWARD, "{ctx}");
    assert_eq!(flags(out.acting(), i), [false; 3], "{ctx}");
    assert!(obs_row(out, i).iter().all(|&b| b == 0), "{ctx}");
    assert!(mask_row(out, i).iter().all(|m| !m.get()), "{ctx}");
    if out.terminal_observation(0).is_some() {
        assert!(
            terminal_row(out, i).iter().all(|&b| b == 0),
            "{ctx}: terminal row"
        );
    }
    assert!(out.pending_reset()[i].get(), "{ctx}");
}

#[test]
fn test_host_abort_explicit_reset_row_and_retry() {
    for mode in MODES {
        each_rt(|rt| {
            let ctx = format!("{mode:?} {rt:?}");
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            shared.abort_resets(2, 1);
            // The reset writes a termination, a death and a spawn, then
            // aborts: all three are discarded.
            shared.reset_events.store(true, Ordering::Relaxed);
            let out = p.reset_all(Reseed::Keep).unwrap();
            assert_reset_abort_row(out, 2, &ctx);
            shared.reset_events.store(false, Ordering::Relaxed);
            for other in (0..N).filter(|&i| i != 2) {
                assert!(
                    !out.aborted()[other].get() && !out.pending_reset()[other].get(),
                    "{ctx}"
                );
            }
            let mut trace = vec![snapshot(out)];
            assert_eq!(p.aborts(), 1, "{ctx}");
            assert_eq!(Shared::get(&shared.observes, 2), 0, "{ctx}");
            if mode == AutoReset::Disabled {
                let err = step(&mut p, &[]).unwrap_err();
                assert!(
                    matches!(err, EngineError::PendingReset { .. }),
                    "{ctx} {err:?}"
                );
                return trace;
            }
            // The next call retries the reset only, with the next episode
            // index (a reset without a new seed advances it).
            let steps = Shared::get(&shared.steps, 2);
            step(&mut p, &[]).unwrap();
            assert_eq!(Shared::get(&shared.steps, 2), steps, "{ctx}");
            assert_eq!(shared.episode_index[2].load(Ordering::Relaxed), 1, "{ctx}");
            let out = p.outputs().unwrap();
            assert!(!out.aborted()[2].get() && out.abort_code()[2] == 0, "{ctx}");
            assert!(!out.pending_reset()[2].get(), "{ctx}");
            assert_eq!(flags(out.agent_truncated(), 2), [false; 3], "{ctx}");
            assert_eq!(flags(out.acting(), 2), [true; 3], "{ctx}");
            // The new episode's observation: episode index 1 (the retry).
            assert_eq!(obs_values(obs_row(out, 2), 0), [3, 1, 2], "{ctx}");
            trace.push(snapshot(out));
            trace
        });
    }
}

#[test]
fn test_host_abort_reset_all_clears_every_abort_row() {
    for mode in MODES {
        each_rt(|rt| {
            let ctx = format!("{mode:?} {rt:?}");
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            p.reset_all(Reseed::Keep).unwrap();
            // Instance 1 aborts in a step, instance 5 in an explicit reset.
            step(&mut p, &[(1, 0, ABORT)]).unwrap();
            shared.abort_resets(5, 1);
            let out = p.reset(&[5], Reseed::Keep).unwrap();
            assert!(out.aborted()[1].get() && out.aborted()[5].get(), "{ctx}");
            assert_eq!(out.abort_code()[1], STEP_CODE, "{ctx}");
            assert_eq!(out.abort_code()[5], RESET_CODE + 5, "{ctx}");
            let mut trace = vec![snapshot(out)];
            // A reset of every instance clears every abort row.
            let out = p.reset_all(Reseed::Keep).unwrap();
            for i in 0..N {
                assert!(!out.aborted()[i].get(), "{ctx} instance {i}");
                assert_eq!(out.abort_code()[i], 0, "{ctx} instance {i}");
            }
            trace.push(snapshot(out));
            trace
        });
    }
}

#[test]
fn test_host_abort_explicit_seed_restarts_the_episode_index() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        shared.abort_resets(0, 1);
        p.reset_all(Reseed::Keep).unwrap();
        p.reset(&[0], Reseed::Base(Seed::new(77))).unwrap();
        assert_eq!(shared.episode_index[0].load(Ordering::Relaxed), 0);
        assert!(!p.outputs().unwrap().pending_reset()[0].get());
        vec![snapshot(p.outputs().unwrap())]
    });
}

#[test]
fn test_host_abort_next_step_retry_that_aborts_again_keeps_pending() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        // Instance 1's step aborts; its next two resets abort too.
        step(&mut p, &[(1, 0, ABORT)]).unwrap();
        shared.abort_resets(1, 2);
        let mut trace = Vec::new();
        step(&mut p, &[]).unwrap();
        assert_reset_abort_row(p.outputs().unwrap(), 1, "first retry");
        trace.push(snapshot(p.outputs().unwrap()));
        step(&mut p, &[]).unwrap();
        assert_reset_abort_row(p.outputs().unwrap(), 1, "second retry");
        trace.push(snapshot(p.outputs().unwrap()));
        step(&mut p, &[]).unwrap();
        let out = p.outputs().unwrap();
        assert!(!out.aborted()[1].get() && !out.pending_reset()[1].get());
        assert_eq!(p.aborts(), 3);
        trace.push(snapshot(out));
        trace
    });
}

#[test]
fn test_host_abort_same_step_reset_aborts_after_an_aborted_step() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::SameStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        shared.abort_resets(0, 1);
        step(&mut p, &[(0, 1, ABORT)]).unwrap();
        let out = p.outputs().unwrap();
        // The row is the step's: its flags, rewards, aborted and code.
        assert!(out.aborted()[0].get());
        assert_eq!(out.abort_code()[0], STEP_CODE + 1);
        assert_eq!(flags(out.agent_truncated(), 0), [true; 3]);
        assert_eq!(rewards(out, 0), [1.0, 2.0, 3.0]);
        assert!(obs_row(out, 0).iter().all(|&b| b == 0));
        assert!(terminal_row(out, 0).iter().all(|&b| b == 0));
        assert!(mask_row(out, 0).iter().all(|m| !m.get()));
        assert_eq!(flags(out.acting(), 0), [false; 3]);
        assert!(out.pending_reset()[0].get());
        // An aborted step and its aborted same-call reset count 2.
        assert_eq!(p.aborts(), 2);
        let mut trace = vec![snapshot(out)];
        // The next call is reset-only: no step, the reset's rewards only.
        let steps = Shared::get(&shared.steps, 0);
        step(&mut p, &[]).unwrap();
        assert_eq!(Shared::get(&shared.steps, 0), steps);
        let out = p.outputs().unwrap();
        assert!(!out.aborted()[0].get() && !out.pending_reset()[0].get());
        assert_eq!(rewards(out, 0), [RESET_REWARD, 0.0, 0.0]);
        assert_eq!(flags(out.acting(), 0), [true; 3]);
        trace.push(snapshot(out));
        trace
    });
}

#[test]
fn test_host_abort_same_step_normal_end_then_reset_aborts() {
    // Terminated, truncated and the step limit: the row keeps the step's
    // flags, rewards and real terminal observation, with the reset's code.
    for (end, limit) in [(TERMINATE, None), (TRUNCATE, None), (NOTHING, Some(1))] {
        each_rt(|rt| {
            let shared = Shared::new();
            let mut o = options(AutoReset::SameStep, rt);
            o.max_episode_steps = limit.map(|l| NonZeroU32::new(l).unwrap());
            let mut p = pool_with::<0>(&shared, o, rt);
            p.reset_all(Reseed::Keep).unwrap();
            shared.abort_resets(3, 1);
            step(&mut p, &[(3, 0, end)]).unwrap();
            let out = p.outputs().unwrap();
            let ctx = format!("{end} {limit:?} {rt:?}");
            assert!(out.aborted()[3].get(), "{ctx}");
            assert_eq!(out.abort_code()[3], RESET_CODE + 3, "{ctx}");
            assert_eq!(rewards(out, 3), [1.0, 2.0, 3.0], "{ctx}");
            let terminated = end == TERMINATE;
            assert_eq!(out.episode_terminated()[3].get(), terminated, "{ctx}");
            assert_eq!(out.episode_truncated()[3].get(), !terminated, "{ctx}");
            assert_eq!(flags(out.agent_terminated(), 3), [terminated; 3], "{ctx}");
            // The real terminal observation: episode 0 after its step 1.
            assert_eq!(
                obs_values(terminal_row(out, 3), 0),
                [4, 2, 1],
                "{ctx}: the real terminal observation"
            );
            assert!(obs_row(out, 3).iter().all(|&b| b == 0), "{ctx}");
            assert!(mask_row(out, 3).iter().all(|m| !m.get()), "{ctx}");
            assert_eq!(flags(out.acting(), 3), [false; 3], "{ctx}");
            assert!(out.pending_reset()[3].get(), "{ctx}");
            let mut trace = vec![snapshot(out)];
            let steps = Shared::get(&shared.steps, 3);
            step(&mut p, &[]).unwrap();
            assert_eq!(Shared::get(&shared.steps, 3), steps, "{ctx}: reset-only");
            trace.push(snapshot(p.outputs().unwrap()));
            trace
        });
    }
}

#[test]
fn test_host_abort_same_step_terminal_row_zero_after_a_normal_end_and_aborted_retries() {
    // A normal end whose same-call reset aborts keeps the real terminal
    // observation; the retries that abort again give their own rows, whose
    // terminal observation is zeros, not that earlier episode's.
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::SameStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        shared.abort_resets(4, 3);
        step(&mut p, &[(4, 0, TERMINATE)]).unwrap();
        let out = p.outputs().unwrap();
        assert_eq!(
            obs_values(terminal_row(out, 4), 0),
            [5, 2, 1],
            "{rt:?}: the real one"
        );
        let mut trace = vec![snapshot(out)];
        for retry in 1..=2 {
            step(&mut p, &[]).unwrap();
            let out = p.outputs().unwrap();
            assert_reset_abort_row(out, 4, &format!("retry {retry} {rt:?}"));
            trace.push(snapshot(out));
        }
        step(&mut p, &[]).unwrap();
        assert!(!p.outputs().unwrap().pending_reset()[4].get());
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_same_step_explicit_reset_abort_zeroes_the_terminal_row() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::SameStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        // An episode ends normally and resets: a real terminal row.
        step(&mut p, &[(6, 0, TERMINATE)]).unwrap();
        assert_eq!(
            obs_values(terminal_row(p.outputs().unwrap(), 6), 0),
            [7, 2, 1]
        );
        let mut trace = vec![snapshot(p.outputs().unwrap())];
        // An explicit reset of it that aborts: its row's terminal
        // observation is zeros.
        shared.abort_resets(6, 1);
        let out = p.reset(&[6, 2], Reseed::Keep).unwrap();
        assert_reset_abort_row(out, 6, &format!("{rt:?}"));
        trace.push(snapshot(out));
        trace
    });
}

#[test]
fn test_host_abort_reset_reward_kept_live_discarded_in_same_step() {
    each_rt(|rt| {
        // Explicit reset: the reward it wrote stays in its row.
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::SameStep, rt);
        shared.abort_resets(1, 1);
        let out = p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(rewards(out, 1), [RESET_REWARD, 0.0, 0.0]);
        let mut trace = vec![snapshot(out)];
        p.reset_all(Reseed::Keep).unwrap();
        // A SameStep reset writes scratch rows: its reward is discarded.
        shared.abort_resets(1, 1);
        step(&mut p, &[(1, 0, TERMINATE)]).unwrap();
        assert_eq!(rewards(p.outputs().unwrap(), 1), [1.0, 2.0, 3.0]);
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_same_step_reset_reports_no_zero_length_episode() {
    // The same trajectory: SameStep reports one episode end (the step's),
    // NextStep two (the step's and the aborted retry's zero-length one).
    each_rt(|rt| {
        let mut ends = Vec::new();
        for mode in [AutoReset::SameStep, AutoReset::NextStep] {
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            p.reset_all(Reseed::Keep).unwrap();
            shared.abort_resets(0, 1);
            let mut n = 0;
            let mut zero = 0;
            for special in [&[(0, 0, TERMINATE)][..], &[], &[]] {
                step(&mut p, special).unwrap();
                let out = p.outputs().unwrap();
                if out.episode_terminated()[0].get() || out.episode_truncated()[0].get() {
                    n += 1;
                    if out.episode_length()[0] == 0.0 {
                        zero += 1;
                    }
                }
            }
            ends.push((n, zero, p.aborts()));
        }
        assert_eq!(ends, [(1, 0, 1), (2, 1, 1)]);
        vec![vec![format!("{ends:?}").into_bytes()]]
    });
}

#[test]
fn test_host_abort_in_reset_sequential_passes_the_turn_check() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<1>(&shared, AutoReset::NextStep, rt);
        shared.abort_resets(0, 1);
        p.reset_all(Reseed::Keep).unwrap();
        assert!(!p.is_poisoned());
        let mut trace = vec![snapshot(p.outputs().unwrap())];
        step(&mut p, &[]).unwrap();
        assert!(!p.is_poisoned());
        trace.push(snapshot(p.outputs().unwrap()));
        trace
    });
}

#[test]
fn test_host_abort_resets_script_in_every_mode() {
    for mode in MODES {
        each_rt(|rt| {
            let shared = Shared::new();
            let mut p = pool::<0>(&shared, mode, rt);
            shared.abort_resets(1, 1);
            shared.abort_resets(5, 1);
            p.reset_all(Reseed::Keep).unwrap();
            let mut trace = vec![snapshot(p.outputs().unwrap())];
            shared.abort_resets(2, 2);
            shared.abort_resets(3, 1);
            shared.abort_resets(7, 1);
            let script: [&[(usize, usize, i32)]; 4] = [
                &[(2, 0, ABORT), (3, 1, TERMINATE), (7, 2, TRUNCATE)],
                &[(0, 0, ABORT)],
                &[],
                &[],
            ];
            for special in script {
                if step(&mut p, special).is_err() {
                    let pending: Vec<u32> = p
                        .outputs()
                        .unwrap()
                        .pending_reset()
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.get())
                        .map(|(i, _)| u32::try_from(i).unwrap())
                        .collect();
                    p.reset(&pending, Reseed::Keep).unwrap();
                }
                trace.push(snapshot(p.outputs().unwrap()));
            }
            trace.push(vec![p.aborts().to_le_bytes().to_vec()]);
            trace
        });
    }
}

#[test]
fn test_host_abort_explicit_reset_ids_not_observed_and_a_later_reset_clears_the_flags() {
    each_rt(|rt| {
        let shared = Shared::new();
        let mut p = pool::<0>(&shared, AutoReset::NextStep, rt);
        p.reset_all(Reseed::Keep).unwrap();
        let observed = Shared::get(&shared.observes, 5);
        shared.abort_resets(5, 1);
        let out = p.reset(&[5, 1], Reseed::Keep).unwrap();
        assert_reset_abort_row(out, 5, &format!("{rt:?}"));
        // No observe on the aborted state.
        assert_eq!(Shared::get(&shared.observes, 5), observed);
        let mut trace = vec![snapshot(out)];
        // A later reset that succeeds clears aborted and abort_code.
        let out = p.reset(&[5], Reseed::Keep).unwrap();
        assert!(!out.aborted()[5].get());
        assert_eq!(out.abort_code()[5], 0);
        assert!(!out.pending_reset()[5].get());
        assert_eq!(Shared::get(&shared.observes, 5), observed + 3);
        trace.push(snapshot(out));
        trace
    });
}

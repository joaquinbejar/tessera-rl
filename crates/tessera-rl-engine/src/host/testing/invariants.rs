// SPDX-License-Identifier: MIT OR Apache-2.0

//! `check_invariants` (#351, SPEC 10.2): the host rules an environment's
//! outputs must keep, beyond its spaces.

use tessera_rl_core::{Bool8, HostEnvironment};

use super::super::HostLayout;
use super::actions::Sampler;
use super::error::CheckError;
use super::run::{CheckRun, build};
use crate::outputs::Outputs;
use crate::pool::Reseed;

/// What [`check_invariants`] saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvariantsReport {
    /// Calls checked, the first reset included.
    pub calls: usize,
    /// Episodes that ended (terminated, truncated or aborted).
    pub episodes_ended: u64,
    /// Aborts the pool applied (`HostPool::aborts`).
    pub aborts: u64,
    /// Times the engine had to restore the canonical floating-point
    /// controls after the environment's code (`HostPool::host_fp_resets`):
    /// not a failure, but a sign the environment changes them.
    pub fp_resets: u64,
}

/// Checks the host invariants of `E` on a seeded run.
///
/// Builds one Inline pool, resets it and steps it `run.calls` times with
/// seeded legal actions. After every call, for every instance:
///
/// - no episode that ended counted an invalid agent event (a spawn
///   without `HostEnvironment::SPAWNS`, a spawn into an alive slot, a
///   death of an empty one): the engine reports them when the episode
///   ends, so the run must end episodes (`CheckRun::max_episode_steps`);
/// - after a reset that did not abort, exactly slots
///   `0..initial_agents` are alive;
/// - every acting agent is alive;
/// - an agent is refused only if it was acting in the call;
/// - an abort code is set exactly where an instance aborted.
///
/// The pool must not be poisoned at any point.
/// Under the engine's `debug-checks` feature an invalid agent event
/// poisons the pool in the call that reports it, so it surfaces as
/// [`CheckError::Engine`] instead.
///
/// # Errors
///
/// [`CheckError::Invariant`] for the first broken rule, or
/// [`CheckError::Engine`] (`Poisoned` included).
pub fn check_invariants<E: HostEnvironment, const A: usize>(
    cfg: &dyn Fn() -> E::Config,
    run: &CheckRun,
) -> Result<InvariantsReport, CheckError> {
    let mut pool = build::<E, A>(cfg, run, run.seed)?;
    let mut sampler = Sampler::new(run.action_seed);
    let mut ended = 0u64;
    pool.reset_all(Reseed::Keep)?;
    check_call(0, pool.layout(), pool.outputs()?, None, true)?;
    let mut acting: Vec<Bool8> = pool.outputs()?.acting().to_vec();
    for call in 1..=run.calls {
        let input = sampler.fill(pool.layout(), pool.outputs()?);
        pool.step(input)?;
        let o = pool.outputs()?;
        ended = ended.saturating_add(count_ended(o));
        check_call(call, pool.layout(), o, Some(&acting), false)?;
        acting.clear();
        acting.extend_from_slice(o.acting());
    }
    Ok(InvariantsReport {
        calls: run.calls.saturating_add(1),
        episodes_ended: ended,
        aborts: pool.aborts(),
        fp_resets: pool.host_fp_resets(),
    })
}

fn flag(v: &[Bool8], i: usize) -> bool {
    v.get(i).is_some_and(|b| b.get())
}

fn count_ended(o: &Outputs) -> u64 {
    let n = o.episode_terminated().len();
    (0..n)
        .filter(|&i| {
            flag(o.episode_terminated(), i)
                || flag(o.episode_truncated(), i)
                || flag(o.aborted(), i)
        })
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Checks one call: `before` is the acting rows the call stepped with
/// (`None` for a reset), `reset` whether every instance was just reset.
fn check_call(
    call: usize,
    layout: &HostLayout,
    o: &Outputs,
    before: Option<&[Bool8]>,
    reset: bool,
) -> Result<(), CheckError> {
    let a = layout.agents.max(1);
    for instance in 0..layout.num_envs {
        let fail = |what| {
            Err(CheckError::Invariant {
                call,
                instance,
                what,
            })
        };
        let aborted = flag(o.aborted(), instance);
        if o.episode_invalid_agent_events()
            .get(instance)
            .is_some_and(|&v| v > 0.0)
        {
            return fail(
                "an invalid agent event (a spawn without SPAWNS, into an alive slot, or a death of an empty slot)",
            );
        }
        let code = o.abort_code().get(instance).copied().unwrap_or(0);
        if !o.aborted().is_empty() && aborted != (code != 0) {
            return fail("an abort code is set exactly where the instance aborted");
        }
        for agent in 0..a {
            let row = instance.saturating_mul(a).saturating_add(agent);
            let alive = flag(o.alive(), row);
            if reset && !aborted && alive != (agent < layout.initial_agents) {
                return fail("after a reset exactly slots 0..initial_agents are alive");
            }
            if flag(o.acting(), row) && !alive {
                return fail("an acting agent is alive");
            }
            if let Some(before) = before
                && flag(o.refused(), row)
                && !flag(before, row)
            {
                return fail("only an agent acting in the call is refused");
            }
        }
    }
    Ok(())
}

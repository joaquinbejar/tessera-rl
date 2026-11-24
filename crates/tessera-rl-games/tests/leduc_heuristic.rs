// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leduc's `InfoHeuristic` (ADR-0079 item 4.5): `act` is unchanged and
//! equals `act_info` on sampled states; states with one information state
//! give one action (noninterference, item 4.4); a Leduc heuristic slot
//! needs `InfoHeuristic` unless set as privileged, and plays `act_info` at
//! any worker count.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{
    AgentSlot, Determinize, Heuristic, ImperfectInformation, InfoHeuristic, PolicyGroup,
    RecordLayout,
};
use tessera_rl_engine::{
    ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed, Schedule, SyncOptions,
};
use tessera_rl_games::{Leduc, LeducConfig, LeducState};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const CFG: LeducConfig = LeducConfig {};
const N: u32 = 12;
/// `N` as a count.
const NU: usize = 12;

fn rng(seed: u64) -> StepRng {
    let id = RngIdentity::new(Seed::new(seed), LogicalId::DEFAULT);
    StepRng::new(&RngAuthority::engine_only(), &id, StreamId::DYNAMICS)
}

fn agent(i: usize) -> AgentSlot {
    AgentSlot::new(u16::try_from(i).unwrap(), 2).unwrap()
}

fn act(s: &LeducState, me: usize) -> i32 {
    let mut a = [0];
    Leduc::act(
        &CFG,
        s,
        agent(me),
        &mut rng(0),
        &mut ActionWriter::new(&mut a, &mut []),
    );
    a[0]
}

fn act_info(s: &LeducState, me: usize) -> i32 {
    let mut a = [0];
    let info = Leduc::info_state(&CFG, s, agent(me));
    Leduc::act_info(
        &CFG,
        &info,
        agent(me),
        &mut rng(0),
        &mut ActionWriter::new(&mut a, &mut []),
    );
    a[0]
}

fn options(schedule: Schedule) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        schedule,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(17))
    }
}

/// A legal learner move for seat 0: check or call (always legal), or a
/// raise when the mask allows it, alternating with `t`.
fn learner(p: &EnvPool<Leduc, Masked>, t: usize) -> Vec<i32> {
    let masks = p.outputs().unwrap().masks();
    (0..NU * 2)
        .map(|row| {
            if row % 2 == 1 {
                return -5;
            }
            if t.is_multiple_of(2) && masks[row * 3 + 2].get() {
                2
            } else {
                1
            }
        })
        .collect()
}

#[test]
fn test_leduc_act_equals_act_info_on_sampled_states() {
    let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options(Schedule::Static)).unwrap();
    p.reset_all(Reseed::Base(Seed::new(3))).unwrap();
    let mut checked = 0;
    for t in 0..200 {
        for i in 0..N {
            let s = p.state(i).unwrap();
            for me in 0..2 {
                assert_eq!(act(s, me), act_info(s, me), "t {t}, instance {i}");
                checked += 1;
            }
        }
        // Both seats play the learner's moves here.
        let mut acts = learner(&p, t);
        for (k, a) in acts.iter_mut().enumerate() {
            if k % 2 == 1 {
                *a = 1;
            }
        }
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    }
    assert!(checked > 1000);
}

#[test]
fn test_leduc_info_heuristic_noninterference_on_determinized_states() {
    let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options(Schedule::Static)).unwrap();
    p.reset_all(Reseed::Base(Seed::new(8))).unwrap();
    for i in 0..N {
        let s = *p.state(i).unwrap();
        for me in 0..2 {
            let info = Leduc::info_state(&CFG, &s, agent(me));
            let want = act_info(&s, me);
            for k in 0..16 {
                let mut other = LeducState::default();
                Leduc::sample(&CFG, &info, &mut rng(k), &mut other);
                assert_eq!(Leduc::info_state(&CFG, &other, agent(me)), info);
                assert_eq!(act(&other, me), want);
                assert_eq!(act_info(&other, me), want);
            }
        }
    }
}

#[test]
fn test_leduc_heuristic_slot_needs_info_heuristic_unless_privileged() {
    let build = || EnvPool::<Leduc, Masked>::with_masks(CFG, options(Schedule::Static)).unwrap();
    assert!(matches!(
        build().with_heuristic_slots(&[1]),
        Err(EngineError::PolicyMap { .. })
    ));
    let info = build().with_info_heuristic_slots(&[1]).unwrap();
    assert_eq!(
        info.policy_groups(),
        &[PolicyGroup::Learner, PolicyGroup::Heuristic]
    );
    let privileged = build().with_privileged_heuristic_slots(&[1]).unwrap();
    assert_eq!(
        privileged.policy_groups(),
        &[PolicyGroup::Learner, PolicyGroup::PrivilegedHeuristic]
    );
}

fn run(workers: Option<usize>, schedule: Schedule) -> Vec<u64> {
    let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options(schedule))
        .unwrap()
        .with_info_heuristic_slots(&[1])
        .unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    for t in 0..120 {
        let mut want = vec![0; NU * 2];
        for i in 0..N {
            let header = p.header(i).unwrap();
            let row = &header.agents()[1];
            if !header.engine().pending_reset.get() && row.alive.get() && row.acting.get() {
                want[usize::try_from(i).unwrap() * 2 + 1] = act_info(p.state(i).unwrap(), 1);
            }
        }
        let acts = learner(&p, t);
        p.step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
        assert_eq!(p.heuristic_actions().unwrap(), want.as_slice(), "t {t}");
    }
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

#[test]
fn test_leduc_heuristic_slot_equals_act_info_at_every_worker_count_and_schedule() {
    let inline = run(None, Schedule::Static);
    for w in [1, 2, 7] {
        assert_eq!(run(Some(w), Schedule::Static), inline);
        assert_eq!(run(Some(w), Schedule::Dynamic), inline);
    }
}

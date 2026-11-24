// SPDX-License-Identifier: MIT OR Apache-2.0

//! JobShop-v1 rules: instance generation, dispatch timing, masks, the
//! return as minus the makespan, termination, observations, validation and
//! the dispatching rules.

#![expect(
    clippy::unwrap_used,
    reason = "test scenarios: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::{Environment, Validate};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_ops::jobshop::baseline::{self, Rule};
use tessera_rl_ops::jobshop::{JOB_FEATURES, MACHINE_FEATURES, MAX_JOBS, MAX_MACHINES};
use tessera_rl_ops::{JobShop, JobShopConfig, JobShopState};

fn cfg(jobs: u32, machines: u32) -> JobShopConfig {
    JobShopConfig {
        jobs,
        machines,
        max_duration: 9,
    }
}

fn pool(c: JobShopConfig, n: u32, seed: u64) -> EnvPool<JobShop, Masked> {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    let mut p = EnvPool::with_masks(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn state(p: &EnvPool<JobShop, Masked>, i: u32) -> JobShopState {
    *p.state(i).unwrap()
}

fn step(p: &mut EnvPool<JobShop, Masked>, actions: &[i32]) -> (Vec<f32>, Vec<bool>) {
    let out = p
        .step(ActionInput {
            discrete: actions,
            continuous: &[],
        })
        .unwrap();
    (
        out.rewards().to_vec(),
        out.episode_terminated().iter().map(|f| f.get()).collect(),
    )
}

#[test]
fn test_jobshop_reset_routes_are_permutations_and_durations_in_range() {
    let c = cfg(7, 5);
    let p = pool(c.clone(), 8, 3);
    for i in 0..8 {
        let s = state(&p, i);
        assert_eq!((s.jobs, s.machines), (7, 5));
        for j in 0..7 {
            let mut route = s.route[j][..5].to_vec();
            route.sort_unstable();
            assert_eq!(route, vec![0, 1, 2, 3, 4]);
            assert!(s.duration[j][..5].iter().all(|&d| (1..=9).contains(&d)));
            assert!(
                s.route[j][5..]
                    .iter()
                    .chain(&s.duration[j][5..])
                    .all(|&b| b == 0)
            );
        }
        assert!(JobShop::validate(&c, &s).is_ok());
    }
    // Instances differ across seeds.
    assert_ne!(state(&p, 0).route, state(&p, 1).route);
}

#[test]
fn test_jobshop_dispatch_starts_at_the_later_of_job_and_machine() {
    let c = cfg(2, 2);
    let mut p = pool(c, 1, 11);
    let s0 = state(&p, 0);
    // Dispatch job 0's first operation, then job 1's first.
    step(&mut p, &[0]);
    let s1 = state(&p, 0);
    let (m0, d0) = (usize::from(s0.route[0][0]), u32::from(s0.duration[0][0]));
    assert_eq!(s1.job_ready[0], d0);
    assert_eq!(s1.machine_free[m0], d0);
    step(&mut p, &[1]);
    let s2 = state(&p, 0);
    let (m1, d1) = (usize::from(s0.route[1][0]), u32::from(s0.duration[1][0]));
    let start = if m1 == m0 { d0 } else { 0 };
    assert_eq!(s2.job_ready[1], start + d1);
    assert_eq!(s2.makespan, d0.max(start + d1));
}

#[test]
fn test_jobshop_return_is_minus_the_makespan_and_episode_ends_on_time() {
    let c = cfg(6, 4);
    let mut p = pool(c.clone(), 4, 5);
    let mut returns = [0f32; 4];
    let total = 6 * 4;
    for t in 0..total {
        let before: Vec<JobShopState> = (0..4).map(|i| state(&p, i)).collect();
        let actions: Vec<i32> = before
            .iter()
            .map(|s| i32::try_from(baseline::act(Rule::Spt, s)).unwrap())
            .collect();
        let (rewards, ended) = step(&mut p, &actions);
        for (i, r) in rewards.iter().enumerate() {
            returns[i] += r;
        }
        assert!(ended.iter().all(|&e| e == (t + 1 == total)), "step {t}");
        if t + 1 == total {
            for (i, s) in before.iter().enumerate() {
                // The final makespan: the one before the last step plus
                // minus its reward.
                let makespan = f32::from(u16::try_from(s.makespan).unwrap()) - rewards[i];
                assert_eq!(returns[i], -makespan);
                assert!(makespan > 0.0);
            }
        }
    }
}

#[test]
fn test_jobshop_masks_follow_unfinished_jobs_and_illegal_is_the_noop() {
    let c = cfg(3, 2);
    let mut p = pool(c, 1, 2);
    // Job 0 twice: then it is done and masked.
    step(&mut p, &[0]);
    step(&mut p, &[0]);
    let masks: Vec<bool> = p
        .outputs()
        .unwrap()
        .masks()
        .iter()
        .map(|f| f.get())
        .collect();
    assert_eq!(masks, vec![false, true, true]);
    // A third dispatch of job 0 is illegal: the engine's policy replaces it
    // (ADR-0028) and the kernel would read it as the no-op, job 1.
    let s = state(&p, 0);
    assert!(!s.legal(0) && s.legal(1));
    assert_eq!(s.noop(), 1);
}

#[test]
fn test_jobshop_observation_shapes_and_relative_times() {
    let c = cfg(4, 3);
    let space = JobShop::observation_space(&c).unwrap();
    let text = format!("{space:?}");
    assert!(text.contains("[4, 5]") && text.contains("[3, 2]"), "{text}");
    let mut p = pool(c, 1, 9);
    step(&mut p, &[2]);
    let s = state(&p, 0);
    let out = p.outputs().unwrap();
    // Leaf 0 is `jobs`, leaf 1 `machines`, in declaration order; one
    // instance and one agent, so each leaf is the whole buffer.
    let ints = |b: &[u8]| -> Vec<i32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_ne_bytes(*c))
            .collect()
    };
    let jobs = ints(out.observation(0).unwrap());
    let machines = ints(out.observation(1).unwrap());
    assert_eq!(jobs.len(), 4 * JOB_FEATURES);
    assert_eq!(machines.len(), 3 * MACHINE_FEATURES);
    let row = &jobs[2 * JOB_FEATURES..3 * JOB_FEATURES];
    // Job 2 ran one operation: its next is the second, it ends the makespan.
    assert_eq!(row[0], i32::from(s.route[2][1]));
    assert_eq!(row[1], i32::from(s.duration[2][1]));
    assert_eq!(row[3], 0);
    assert_eq!(row[4], 0);
    assert!(jobs.iter().chain(&machines).all(|&v| v >= 0));
    let load: i32 = machines.iter().skip(1).step_by(2).sum();
    let work: i32 = jobs.iter().skip(2).step_by(JOB_FEATURES).sum();
    assert_eq!(
        load, work,
        "remaining loads and remaining work count the same operations"
    );
}

#[test]
fn test_jobshop_validate_refuses_broken_states() {
    let c = cfg(3, 3);
    let p = pool(c.clone(), 1, 4);
    let good = state(&p, 0);
    assert!(JobShop::validate(&c, &good).is_ok());
    assert!(JobShop::validate(&c, &JobShopState::default()).is_ok());
    let broken: [fn(&mut JobShopState); 6] = [
        |s| s.route[0][1] = s.route[0][0],
        |s| s.duration[1][0] = 0,
        |s| s.next_op[0] = 4,
        |s| s.makespan = 1,
        |s| s.done = 1,
        |s| s.route[MAX_JOBS - 1][MAX_MACHINES - 1] = 1,
    ];
    for (k, f) in broken.iter().enumerate() {
        let mut s = good;
        f(&mut s);
        assert!(JobShop::validate(&c, &s).is_err(), "case {k}");
    }
    assert!(
        JobShop::validate(&cfg(4, 3), &good).is_err(),
        "another shape"
    );
}

#[test]
fn test_jobshop_rules_pick_by_their_keys_with_lowest_index_ties() {
    let p = pool(cfg(5, 4), 1, 21);
    let s = state(&p, 0);
    let spt = baseline::act(Rule::Spt, &s);
    let best = (0..5).map(|j| s.duration[j][0]).min().unwrap();
    assert_eq!(s.duration[spt][0], best);
    assert!((0..spt).all(|j| s.duration[j][0] > best));
    let mwkr = baseline::act(Rule::Mwkr, &s);
    let most = (0..5).map(|j| s.remaining_work(j)).max().unwrap();
    assert_eq!(s.remaining_work(mwkr), most);
    // Every job is ready at 0 at reset: FIFO takes job 0.
    assert_eq!(baseline::act(Rule::Fifo, &s), 0);
}

#[test]
fn test_jobshop_largest_shape_runs_to_the_end() {
    let c = JobShopConfig {
        jobs: 64,
        machines: 32,
        max_duration: 255,
    };
    let mut p = pool(c.clone(), 2, 8);
    let mut last = (vec![], vec![]);
    for _ in 0..64 * 32 {
        let actions: Vec<i32> = (0..2)
            .map(|i| i32::try_from(baseline::act(Rule::Mwkr, &state(&p, i))).unwrap())
            .collect();
        last = step(&mut p, &actions);
    }
    assert!(last.1.iter().all(|&e| e));
}

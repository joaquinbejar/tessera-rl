// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search over `MultiDiscrete` actions (ADR-0086) on a one-agent
//! environment with conditional heads: the joint index, canonical joints,
//! legality from the conditional masks, the caps, PUCT and Gumbel runs,
//! and `check_head_activity`.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

#[path = "support/joint_fixture.rs"]
mod joint_fixture;
#[path = "support/picker.rs"]
mod picker;

use core::num::NonZeroU32;

use picker::{NVEC, Picker, PickerConfig};
use tessera_rl_core::ActionMask;
use tessera_rl_core::space::JointHeads;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, NoMasks, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, GumbelConfig, HeadActivityError, MAX_JOINT_ACTIONS, Mcts,
    PuctConfig, RootSelection, SearchError, SearchOptions, SearchPool, check_head_activity,
};

fn cfg() -> PickerConfig {
    PickerConfig::default()
}

/// The joint index of head values (first head most significant).
fn j(v: [u32; 3]) -> u32 {
    (v[0] * NVEC[1] + v[1]) * NVEC[2] + v[2]
}

fn root(cfg: &PickerConfig, steps: &[[i32; 3]]) -> Record<Picker> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(5))
    };
    let mut p = EnvPool::<Picker, Masked>::with_masks(cfg.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for a in steps {
        p.step(ActionInput {
            discrete: a,
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn options(capacity: u32) -> SearchOptions {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(8).unwrap(),
    );
    o.eviction = Eviction::Strict;
    o
}

fn masked(capacity: u32) -> SearchPool<Picker, Masked> {
    SearchPool::new(
        cfg(),
        options(capacity),
        &root(&cfg(), &[]),
        EnvPool::with_masks,
    )
    .unwrap()
}

/// The canonical joints legal at a total of 0: pass, add 0 to 2, bonus
/// with any amount and either bonus value.
fn legal_at_start() -> Vec<u32> {
    let mut v = vec![j([0, 0, 0])];
    v.extend((0..3).map(|a| j([1, a, 0])));
    for a in 0..4 {
        v.extend((0..2).map(|b| j([2, a, b])));
    }
    v.sort_unstable();
    v
}

struct Uniform;

impl Evaluator<Picker> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<Picker>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 24.0);
        values.fill(0.0);
        Ok(())
    }
}

#[test]
fn test_multi_discrete_pool_has_one_edge_per_joint() {
    let s = masked(16);
    assert_eq!(s.actions(), 24);
}

#[test]
fn test_multi_discrete_expand_steps_the_decoded_heads() {
    let mut s = masked(16);
    let root = s.root();
    let mut out = [root; 2];
    s.expand(&[(root, j([1, 2, 0])), (root, j([2, 1, 1]))], &mut out)
        .unwrap();
    assert_eq!(s.record(out[0]).unwrap().state().total, 3);
    assert_eq!(s.record(out[1]).unwrap().state().total, 21);
    assert_eq!(s.rewards(out[1]).unwrap(), &[21.0]);
}

#[test]
fn test_multi_discrete_non_canonical_and_masked_joints_are_illegal() {
    let mut s = masked(16);
    let root = s.root();
    let before = s.tree_hash();
    let mut out = [root];
    for bad in [
        j([0, 1, 0]), // pass with an amount: amount is inactive
        j([1, 0, 1]), // add with a bonus: bonus is inactive
        j([1, 3, 0]), // add 3: masked
    ] {
        assert_eq!(
            s.expand(&[(root, bad)], &mut out).unwrap_err(),
            SearchError::IllegalAction { action: bad }
        );
    }
    assert_eq!(s.tree_hash(), before);
    // The child's legal joints come from its own mask: bonus is illegal
    // from a total of 20.
    s.expand(&[(root, j([2, 3, 1]))], &mut out).unwrap();
    let mut again = [out[0]];
    assert!(matches!(
        s.expand(&[(out[0], j([2, 0, 0]))], &mut again),
        Err(SearchError::IllegalAction { .. })
    ));
    s.expand(&[(out[0], j([1, 0, 0]))], &mut again).unwrap();
}

#[test]
fn test_multi_discrete_puct_visits_only_legal_canonical_joints() {
    let mut c = PuctConfig::default();
    c.simulations = 64;
    let mut m = Mcts::new(masked(256), c).unwrap();
    m.run(&mut Uniform).unwrap();
    let mut visits = [0u32; 24];
    m.root_visits(&mut visits);
    let visited: Vec<u32> = (0..24).filter(|&a| visits[a as usize] > 0).collect();
    assert_eq!(visited, legal_at_start());
    assert_eq!(visits.iter().sum::<u32>(), 64);
    assert!(legal_at_start().contains(&m.select_action(0.0).unwrap()));
}

#[test]
fn test_multi_discrete_gumbel_considers_only_legal_canonical_joints() {
    let mut c = PuctConfig::default();
    c.simulations = 32;
    c.root = RootSelection::Gumbel(GumbelConfig::default());
    let mut m = Mcts::new(masked(256), c).unwrap();
    m.run(&mut Uniform).unwrap();
    let legal = legal_at_start();
    let considered = m.considered().unwrap();
    assert_ne!(considered.len(), 0);
    assert!(
        considered.iter().all(|a| legal.contains(a)),
        "{considered:?}"
    );
    assert!(legal.contains(&m.select_action(0.0).unwrap()));
}

#[test]
fn test_multi_discrete_without_masks_every_joint_is_legal() {
    // `NoMasks` declares no parents and no activity (ADR-0086 decision 3:
    // pools without masks are unaffected), so every joint is its own
    // action and legal.
    let mut s =
        SearchPool::<Picker, NoMasks>::new(cfg(), options(64), &root(&cfg(), &[]), EnvPool::new)
            .unwrap();
    assert_eq!(s.actions(), 24);
    let root = s.root();
    let mut out = [root; 2];
    s.expand(&[(root, j([1, 3, 0])), (root, j([0, 2, 1]))], &mut out)
        .unwrap();
    assert_eq!(s.record(out[0]).unwrap().state().total, 4);
    assert_eq!(s.record(out[1]).unwrap().state().total, 0);
}

#[test]
fn test_multi_discrete_above_the_cap_is_refused_and_discrete_is_not_capped() {
    let wide = PickerConfig { wide: 1, cheat: 0 };
    let r = SearchPool::<Picker, NoMasks>::new(
        wide.clone(),
        options(8),
        &root(&cfg(), &[]),
        EnvPool::new,
    );
    assert_eq!(
        r.err(),
        Some(SearchError::TooManyJointActions {
            joint: 64 * 65,
            max: MAX_JOINT_ACTIONS
        })
    );
    let discrete = PickerConfig { wide: 2, cheat: 0 };
    let s =
        SearchPool::<Picker, NoMasks>::new(discrete, options(8), &root(&cfg(), &[]), EnvPool::new)
            .unwrap();
    assert_eq!(s.actions(), 5000);
}

#[test]
fn test_check_head_activity_passes_an_honest_env_and_catches_a_cheat() {
    let roots = [
        root(&cfg(), &[]),
        root(&cfg(), &[[2, 3, 1]]),
        root(&cfg(), &[[1, 1, 0], [0, 0, 0]]),
    ];
    let report =
        check_head_activity::<Picker>(&cfg(), &roots, NonZeroU32::new(4).unwrap(), 7).unwrap();
    assert_eq!(report.roots, 3);
    // 12 legal joints at a total of 0 (twice), 4 above 20.
    assert_eq!(report.cases, 12 + 4 + 12);
    assert!(report.with_inactive > 0);

    let cheat = PickerConfig { wide: 0, cheat: 1 };
    let err = check_head_activity::<Picker>(&cheat, &roots[..1], NonZeroU32::new(4).unwrap(), 7)
        .unwrap_err();
    assert!(
        matches!(
            err,
            HeadActivityError::Differs {
                joint: 0,
                what: "the record",
                ..
            }
        ),
        "{err}"
    );
}

/// The native joints of the test environment, for the Python cross-check
/// of `joint_log_prior` (review r1 of #454).
#[test]
fn test_multi_discrete_joint_fixture_matches_the_native_joints() {
    let c = cfg();
    let parents: Vec<_> = (0..NVEC.len())
        .map(|h| Picker::head_parents(&c, h))
        .collect();
    let active = |h: usize, p: [u32; 2]| Picker::head_active(&c, h, p);
    let joint = JointHeads::new(&NVEC, &parents, u32::MAX, active).unwrap();
    let mut text = String::new();
    for (name, steps) in [("start", &[][..]), ("bonus_spent", &[[2, 3, 1]][..])] {
        let options = PoolOptions {
            auto_reset: AutoReset::Disabled,
            ..PoolOptions::new(NonZeroU32::MIN, Seed::new(5))
        };
        let mut p = EnvPool::<Picker, Masked>::with_masks(c.clone(), options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        p.restore(0, &root(&c, steps)).unwrap();
        let row = p.outputs().unwrap().masks().to_vec();
        text += &joint_fixture::joint_case(
            &format!("picker_{name}"),
            &NVEC,
            &parents,
            &active,
            &joint,
            &|j| joint.is_legal(j, |at| row.get(at).is_some_and(|m| m.get())),
        );
    }
    joint_fixture::check_fixture("picker.txt", &text);
}

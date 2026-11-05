// SPDX-License-Identifier: MIT OR Apache-2.0

//! The eight normative tests of ADR-0046 on Leduc hold'em: the opponent's
//! card is hidden, and the public card is chance drawn from `StepRng` at
//! step time. Every test asserts on the output record: the opponent's
//! card is the sampler's first draw, the public card of a child is the
//! rollout's draw, and the header is the ADR's table.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test helpers: a failure is the assertion; counts and slots are small"
)]

use std::collections::{BTreeMap, BTreeSet};

use core::num::NonZeroU32;

use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::{AgentSlot, Environment, ImperfectInformation, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, Masked, PoolOptions, Record, Reseed,
};
use tessera_rl_games::leduc::{CALL, FOLD, NO_CARD, RAISE, draw};
use tessera_rl_games::{Leduc, LeducConfig};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_search::{
    Determinization, Evaluator, EvaluatorError, Mcts, PuctConfig, SearchError, SearchOptions,
    SearchPool, rollout_identity, rollout_salt, sampler_identity,
};

type Pool = SearchPool<Leduc, Masked>;
const CFG: LeducConfig = LeducConfig {};

fn agent(i: u16) -> AgentSlot {
    AgentSlot::new(i, 2).unwrap()
}

fn options(envs: u32, seed: u64) -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(envs).unwrap(), Seed::new(seed))
    }
}

fn play(p: &mut EnvPool<Leduc, Masked>, id: usize, moves: &[u32]) {
    let envs = p.options().num_envs.get() as usize;
    for &a in moves {
        let mut act = vec![1; envs * 2];
        act[id * 2 + p.state(id as u32).unwrap().mover()] = i32::try_from(a).unwrap();
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
}

fn root_with(seed: u64, lineage: u32, extra: bool, branched: bool, moves: &[u32]) -> Record<Leduc> {
    let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options(2, seed)).unwrap();
    p.set_lineage(0, LogicalId::new(lineage)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    if extra {
        p.reset_all(Reseed::Keep).unwrap();
    }
    let id = if branched {
        p.fork_branched(0, &[1], None).unwrap();
        1
    } else {
        0
    };
    play(&mut p, id, moves);
    p.snapshot(id as u32).unwrap()
}

fn root(seed: u64, moves: &[u32]) -> Record<Leduc> {
    root_with(seed, 0, false, false, moves)
}

fn how(seed: u64, sample_index: u64) -> Determinization {
    how_as(0, seed, sample_index)
}

fn how_as(a: u16, seed: u64, sample_index: u64) -> Determinization {
    Determinization {
        agent: agent(a),
        sampler_seed: Seed::new(seed),
        sample_index,
    }
}

fn search_options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap())
}

fn det(root: &Record<Leduc>, how: Determinization) -> Pool {
    SearchPool::new_determinized(CFG, search_options(), root, how, EnvPool::with_masks).unwrap()
}

fn out(s: &Pool) -> Record<Leduc> {
    *s.record(s.root()).unwrap()
}

fn info_bytes(r: &Record<Leduc>, a: u16) -> Vec<u8> {
    encode_to_vec(&Leduc::info_state(&CFG, r.state(), agent(a)))
}

fn step_rng(id: &RngIdentity) -> StepRng {
    StepRng::new(&RngAuthority::engine_only(), id, StreamId::DYNAMICS)
}

/// The opponent's card the sampler must draw for `how.agent`.
fn sampled_card(r: &Record<Leduc>, how: Determinization) -> u32 {
    let info = Leduc::info_state(&CFG, r.state(), how.agent);
    let id = sampler_identity(how.sampler_seed, how.sample_index);
    draw(&mut step_rng(&id), &[info.card, info.public])
}

/// The searching agents and root histories: agent 0 to move after two
/// checks (the public card dealt), agent 1 to move after one check, and
/// agent 1 not to move after two checks.
const CASES: [(u16, &[u32]); 3] = [(0, &[CALL, CALL]), (1, &[CALL]), (1, &[CALL, CALL])];

/// The largest group of roots with one information state for agent `a`
/// after `moves`, varying the hidden
/// card, the seed, the lineage, the episode index and the salt. The
/// accumulators are zero in every running hand (rewards come at the end,
/// and an illegal action is an error under `debug-checks`): Hidden Dice
/// varies them.
fn equal_info_roots(a: u16, moves: &[u32]) -> Vec<Record<Leduc>> {
    let mut groups: BTreeMap<Vec<u8>, Vec<Record<Leduc>>> = BTreeMap::new();
    for seed in 0..200 {
        for (lineage, extra, branched) in [
            (0, false, false),
            (1, false, false),
            (0, true, false),
            (0, false, true),
            (2, true, true),
        ] {
            let r = root_with(seed, lineage, extra, branched, moves);
            groups.entry(info_bytes(&r, a)).or_default().push(r);
        }
    }
    groups.into_values().max_by_key(Vec::len).unwrap()
}

fn distinct<T: Ord>(rs: &[Record<Leduc>], f: impl Fn(&Record<Leduc>) -> T) -> usize {
    rs.iter().map(f).collect::<BTreeSet<_>>().len()
}

/// ADR-0046 test 1, on the output record: identical bytes, and the
/// opponent's card is the sampler's draw. `acting`, reserved bytes and
/// (here) the accumulators cannot vary between valid roots.
#[test]
fn test_leduc_sample_noninterference_equal_info_gives_identical_records() {
    for (a, moves) in CASES {
        let (me, them) = (usize::from(a), 1 - usize::from(a));
        let roots = equal_info_roots(a, moves);
        assert!(roots.len() >= 8, "{}", roots.len());
        assert!(
            distinct(&roots, |r| r.state().cards[them]) > 1,
            "hidden card"
        );
        let e = |r: &Record<Leduc>| r.header().engine().identity;
        assert!(distinct(&roots, |r| e(r).seed().get()) > 1);
        assert!(distinct(&roots, |r| e(r).logical_id().get()) > 1);
        assert!(distinct(&roots, |r| e(r).branch_salt().get()) > 1);
        assert!(distinct(&roots, |r| e(r).episode_index()) > 1);
        for index in 0..6 {
            let h = how_as(a, 77, index);
            let first = out(&det(&roots[0], h));
            assert_eq!(first.state().cards[them], sampled_card(&roots[0], h));
            assert_eq!(first.state().cards[me], roots[0].state().cards[me]);
            for r in &roots[1..] {
                assert_eq!(out(&det(r, h)).to_bytes(), first.to_bytes());
            }
        }
    }
}

/// Priors and value from each leaf's record hash.
struct Canned;

impl Evaluator<Leduc> for Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Leduc>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        for (i, r) in leaves.iter().enumerate() {
            let h = r.hash().get();
            let w: Vec<f64> = (0..3)
                .map(|k| f64::from(1 + ((h >> (8 * k)) & 0xff) as u32))
                .collect();
            let sum: f64 = w.iter().sum();
            for k in 0..3 {
                priors[i * 3 + k] = w[k] / sum;
            }
            values[i] = f64::from((h >> 48) as u32 & 0xffff) / 32_767.5 - 1.0;
        }
        Ok(())
    }
}

fn searched(a: u16, root: &Record<Leduc>, index: u64) -> (u64, Vec<u32>) {
    let mut c = PuctConfig::default();
    c.simulations = 96;
    c.batch_size = 4;
    let mut pool = det(root, how_as(a, 5, index + 1000));
    pool.set_root_determinized(root, how_as(a, 5, index))
        .unwrap();
    let mut m = Mcts::new(pool, c).unwrap();
    m.run(&mut Canned).unwrap();
    assert!(!m.is_privileged());
    let mut visits = vec![0; 3];
    m.root_visits(&mut visits);
    (m.pool().tree_hash(), visits)
}

/// ADR-0046 test 2.
#[test]
fn test_leduc_search_noninterference_equal_info_gives_identical_search() {
    for (a, moves) in CASES {
        let roots = equal_info_roots(a, moves);
        for index in 0..2 {
            let first = searched(a, &roots[0], index);
            assert!(first.1.iter().sum::<u32>() > 0);
            for r in &roots[1..] {
                assert_eq!(searched(a, r, index), first, "agent {a} sample {index}");
            }
        }
    }
}

/// ADR-0046 test 3, a fixed-seed fixture with a declared tolerance. From
/// a root where agent 1 closes the first round, the public card of each
/// child is the rollout's draw over the cards left, its frequencies
/// follow the deck as agent 0 sees it, and the true public card is not
/// over-represented.
#[test]
fn test_leduc_no_future_chance_public_card_follows_the_deck() {
    const N: u64 = 900;
    const TOL: f64 = 0.25;
    let mut r = root(11, &[CALL]);
    let mut seed = 11;
    // A root whose agent 0 holds a king, so the expected shares are known.
    while r.state().cards[0] != 2 {
        seed += 1;
        r = root(seed, &[CALL]);
    }
    let mut real = EnvPool::<Leduc, Masked>::with_masks(CFG, options(1, 0)).unwrap();
    real.restore(0, &r).unwrap();
    play(&mut real, 0, &[CALL]);
    let true_public = real.state(0).unwrap().public;

    let mut s = det(&r, how(99, 0));
    let mut publics = [0_u32; 3];
    for i in 0..N {
        let h = s.set_root_determinized(&r, how(99, i)).unwrap();
        // The engine keys a step on the step it produces.
        let steps = s.record(h).unwrap().header().engine().episode_steps + 1;
        let mut o = [h];
        s.expand(&[(h, CALL)], &mut o).unwrap();
        let child = *s.record(o[0]).unwrap();
        let public = child.state().public;
        let want = draw(
            &mut step_rng(&rollout_identity(Seed::new(99), i, steps)),
            &child.state().cards,
        );
        assert_eq!(public, want, "sample {i}: the rollout's draw");
        assert_ne!(public, NO_CARD);
        publics[public as usize] += 1;
    }
    // Agent 0 holds one king: of the five cards it cannot see, two jacks,
    // two queens and one king; the opponent takes one, the public another.
    let expected = [0.4, 0.4, 0.2].map(|p| p * N as f64);
    for (c, e) in publics.iter().zip(expected) {
        assert!((f64::from(*c) - e).abs() <= TOL * e, "{publics:?}");
    }
    let share = f64::from(publics[true_public as usize]) / expected[true_public as usize];
    assert!(share <= 1.0 + TOL, "{publics:?} true {true_public}");
}

/// ADR-0046 test 4: the opponent's card is the first draw of the sampler
/// identity on the dynamics stream, at samples where the rollout identity
/// or the observation stream would give another card; the rollout salt
/// is in the output header.
#[test]
fn test_leduc_domain_separation_sampler_draw_is_pinned() {
    for (a, moves) in CASES {
        let them = 1 - usize::from(a);
        let r = root(6, moves);
        let info = Leduc::info_state(&CFG, r.state(), agent(a));
        let seen = [info.card, info.public];
        let mut telling = 0;
        for i in 0..48 {
            let sampler = sampler_identity(Seed::new(13), i);
            let expected = draw(&mut step_rng(&sampler), &seen);
            let rollout = draw(&mut step_rng(&rollout_identity(Seed::new(13), i, 0)), &seen);
            let obs = draw(
                &mut StepRng::new(&RngAuthority::engine_only(), &sampler, StreamId::OBS_NOISE),
                &seen,
            );
            let d = out(&det(&r, how_as(a, 13, i)));
            assert_eq!(d.state().cards[them], expected, "agent {a} sample {i}");
            assert_eq!(
                d.state().cards[usize::from(a)],
                r.state().cards[usize::from(a)]
            );
            assert_eq!(d.header().engine().identity.branch_salt(), rollout_salt(i));
            assert_ne!(sampler.branch_salt(), rollout_salt(i));
            telling += usize::from(expected != rollout && expected != obs);
        }
        assert!(telling >= 8, "{telling}");
    }
}

fn projection(r: &Record<Leduc>) -> Vec<u8> {
    let e = r.header().engine();
    let mut v = e.identity.step_index().to_le_bytes().to_vec();
    v.extend(e.episode_steps.to_le_bytes());
    for row in r.header().agents() {
        v.push(u8::from(row.alive.get()));
        v.extend(row.generation.to_le_bytes());
    }
    v
}

/// ADR-0046 test 5: over rollouts with varied seeds and policies, the
/// copied projection is single-valued per information state of each
/// agent. Leduc has no adaptive field.
#[test]
fn test_leduc_copied_fields_are_single_valued_per_info_state() {
    let mut seen: [BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>>; 2] = Default::default();
    for seed in 0..300_u64 {
        let mut p = EnvPool::<Leduc, Masked>::with_masks(CFG, options(1, seed)).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        for t in 0..8 {
            if p.state(0).unwrap().done != 0 {
                break;
            }
            let r = p.snapshot(0).unwrap();
            for (a, map) in seen.iter_mut().enumerate() {
                map.entry(info_bytes(&r, a as u16))
                    .or_default()
                    .insert(projection(&r));
            }
            let s = *p.state(0).unwrap();
            let pick = match [CALL, RAISE, CALL, FOLD][((seed >> (2 * t)) & 3) as usize] {
                RAISE if s.raises >= 2 => CALL,
                FOLD if !s.facing_bet() => CALL,
                other => other,
            };
            play(&mut p, 0, &[pick]);
        }
    }
    for map in &seen {
        assert!(map.len() > 50, "{}", map.len());
        assert!(map.values().all(|v| v.len() == 1));
    }
}

/// ADR-0046 test 6, on the output record and header.
#[test]
fn test_leduc_consistency_samples_keep_the_info_state_and_pass_checks() {
    for (seed, moves) in [
        (1, &[][..]),
        (2, &[RAISE]),
        (3, &[CALL, CALL]),
        (4, &[RAISE, RAISE, CALL, RAISE]),
    ] {
        let rt = root(seed, moves);
        for (a, i) in [0, 1].into_iter().flat_map(|a| (0..8).map(move |i| (a, i))) {
            let plan = how_as(a, seed + 40, i);
            let sample = out(&det(&rt, plan));
            assert_eq!(info_bytes(&sample, a), info_bytes(&rt, a));
            assert_eq!(
                sample.state().cards[1 - usize::from(a)],
                sampled_card(&rt, plan)
            );
            assert_eq!(
                sample.state().cards[usize::from(a)],
                rt.state().cards[usize::from(a)]
            );
            let (o, from) = (sample.header().engine(), rt.header().engine());
            assert_eq!(o.identity.step_index(), from.identity.step_index());
            assert_eq!(o.episode_steps, moves.len() as u32);
            assert_eq!(o.identity.seed(), Seed::new(seed + 40));
            assert_eq!(o.identity.logical_id(), LogicalId::DEFAULT);
            assert_eq!(o.identity.episode_index(), 0);
            assert!(!o.pending_reset.get() && o.end_kind == 0 && !o.fresh.get());
            for (k, row) in sample.header().agents().iter().enumerate() {
                assert_eq!(
                    row.acting.get(),
                    Leduc::acting(&CFG, sample.state(), agent(k as u16))
                );
                assert!(row.alive.get());
                assert_eq!(row.generation, rt.header().agents()[k].generation);
                assert_eq!(
                    (
                        row.episode_return,
                        row.illegal_actions,
                        row.invalid_agent_events
                    ),
                    (0.0, 0, 0)
                );
            }
            let mut check = EnvPool::<Leduc, Masked>::with_masks(CFG, options(1, 0)).unwrap();
            check.import(&[0], &[DecodedRecord::from(sample)]).unwrap();
        }
    }
}

/// ADR-0046 test 7: the same root and index give the same output in any
/// order, the root is not changed, and no buffer keeps the full root.
#[test]
fn test_leduc_sampling_leaves_the_root_unchanged_in_any_order() {
    let r = root(9, &[RAISE, CALL]);
    let before = r.to_bytes();
    let mut s = det(&r, how(1, 0));
    let mut forward = Vec::new();
    for i in [0, 1, 2, 3] {
        s.set_root_determinized(&r, how(1, i)).unwrap();
        forward.push(out(&s).to_bytes());
        assert!(!s.holds_record(&r));
    }
    for i in [3, 0, 2, 1, 3] {
        s.set_root_determinized(&r, how(1, i)).unwrap();
        assert_eq!(out(&s).to_bytes(), forward[i as usize]);
    }
    assert_eq!(r.to_bytes(), before);
}

/// ADR-0046 test 8.
#[test]
fn test_leduc_refused_roots_and_operations_give_typed_errors() {
    let r = root(4, &[RAISE]);
    let finished = root(4, &[RAISE, FOLD]);
    assert!(finished.header().engine().pending_reset.get());
    let refused = SearchPool::new_determinized(
        CFG,
        search_options(),
        &finished,
        how(1, 0),
        EnvPool::with_masks,
    );
    assert!(matches!(refused, Err(SearchError::PendingResetRoot)));

    let mut s = det(&r, how(1, 0));
    let root_h = s.root();
    let mut children = [root_h];
    s.expand(&[(root_h, CALL)], &mut children).unwrap();
    let before = s.tree_hash();
    assert_eq!(
        s.set_root_determinized(&finished, how(1, 1)).unwrap_err(),
        SearchError::PendingResetRoot
    );
    assert_eq!(
        s.reroot(children[0]).unwrap_err(),
        SearchError::DeterminizedReroot
    );
    assert_eq!(s.set_root(&r).unwrap_err(), SearchError::DeterminizedPool);
    assert_eq!(s.tree_hash(), before);

    let plain = SearchPool::<Leduc, Masked>::new(CFG, search_options(), &r, EnvPool::with_masks);
    assert!(matches!(plain, Err(SearchError::ImperfectInformation)));
    let privileged =
        SearchPool::<Leduc, Masked>::new_privileged(CFG, search_options(), &r, EnvPool::with_masks)
            .unwrap();
    assert!(privileged.is_privileged());
    assert_eq!(out(&privileged).to_bytes(), r.to_bytes(), "the full root");
}

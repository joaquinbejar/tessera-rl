// SPDX-License-Identifier: MIT OR Apache-2.0

//! Determinized search roots (ADR-0035, ADR-0046) on Hidden Dice, a test
//! game with hidden cards, a die drawn from `StepRng` and observation
//! noise drawn from `ObsRng` at step time: the eight normative tests of
//! ADR-0046, the reference vectors of its labels, and the privileged and
//! perfect-information constructors.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "test helpers: a failure is the assertion; counts and slots are small"
)]

mod env;
mod fog;

use std::collections::{BTreeMap, BTreeSet};

use core::num::NonZeroU32;

use env::{CARDS, DIE, Dice, DiceConfig, DiceWith, Level, MOVES, NOISE};
use fog::{Fog, FogConfig};
use tessera_rl_core::encoding::encode_to_vec;
use tessera_rl_core::{AgentSlot, Environment, ImperfectInformation, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, Masked, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::{LogicalId, Seed, StreamId};
use tessera_rl_num::key::derive_key;
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_search::{
    Determinization, Evaluator, EvaluatorError, Mcts, PuctConfig, ROLLOUT, SAMPLER, SearchError,
    SearchOptions, SearchPool, rollout_identity, rollout_salt, sampler_identity, sampler_salt,
};

type Pool = SearchPool<Dice, Masked>;

const fn cfg() -> DiceConfig {
    DiceConfig { sabotage: 0 }
}

fn agent(i: u16) -> AgentSlot {
    AgentSlot::new(i, 2).unwrap()
}

fn options(envs: u32, seed: u64) -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(envs).unwrap(), Seed::new(seed))
    }
}

/// Plays `moves` (actions of the side to move) on instance `id`.
fn play(p: &mut EnvPool<Dice, Masked>, id: usize, moves: &[u32]) {
    let envs = p.options().num_envs.get() as usize;
    for &a in moves {
        let mut act = vec![0; envs * 2];
        let me = (p.state(id as u32).unwrap().moves % 2) as usize;
        act[id * 2 + me] = i32::try_from(a).unwrap();
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
}

/// A root after `moves`, with the given lineage, an extra reset (episode
/// index 1) and a branch salt (a `fork_branched` copy) when asked.
fn root_with(
    seed: u64,
    lineage: u32,
    extra_reset: bool,
    branched: bool,
    moves: &[u32],
) -> Record<Dice> {
    let mut p = EnvPool::<Dice, Masked>::with_masks(cfg(), options(2, seed)).unwrap();
    p.set_lineage(0, LogicalId::new(lineage)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    if extra_reset {
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

fn root(seed: u64, moves: &[u32]) -> Record<Dice> {
    root_with(seed, 0, false, false, moves)
}

fn how(seed: u64, sample_index: u64) -> Determinization {
    Determinization {
        agent: agent(0),
        sampler_seed: Seed::new(seed),
        sample_index,
    }
}

fn search_options() -> SearchOptions {
    SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap())
}

fn det_with(c: DiceConfig, root: &Record<Dice>, how: Determinization) -> Result<Pool, SearchError> {
    SearchPool::new_determinized(c, search_options(), root, how, EnvPool::with_masks)
}

fn det(root: &Record<Dice>, how: Determinization) -> Pool {
    det_with(cfg(), root, how).unwrap()
}

fn err<T>(r: Result<T, SearchError>) -> SearchError {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected an error"),
    }
}

fn root_bytes(s: &Pool) -> Vec<u8> {
    s.record(s.root()).unwrap().to_bytes()
}

fn info_bytes(r: &Record<Dice>, a: u16) -> Vec<u8> {
    encode_to_vec(&Dice::info_state(&cfg(), r.state(), agent(a)))
}

/// Roots with one information state for agent 0, varying everything
/// else: the opponent's card, the seed, the lineage, the episode index,
/// the branch salt and the accumulators. The largest such group.
fn equal_info_roots() -> Vec<Record<Dice>> {
    let mut groups: BTreeMap<Vec<u8>, Vec<Record<Dice>>> = BTreeMap::new();
    for seed in 0..300 {
        for (lineage, extra, branched) in [
            (0, false, false),
            (1, false, false),
            (0, true, false),
            (0, false, true),
            (2, true, true),
        ] {
            let r = root_with(seed, lineage, extra, branched, &[1, 0]);
            groups.entry(info_bytes(&r, 0)).or_default().push(r);
        }
    }
    groups.into_values().max_by_key(Vec::len).unwrap()
}

fn distinct<T: Ord>(rs: &[Record<Dice>], f: impl Fn(&Record<Dice>) -> T) -> usize {
    rs.iter().map(f).collect::<BTreeSet<_>>().len()
}

fn assert_group_varies(rs: &[Record<Dice>]) {
    assert!(distinct(rs, |r| r.state().cards[1]) > 1, "hidden state");
    assert!(
        distinct(rs, |r| r.header().engine().identity.seed().get()) > 1,
        "seed"
    );
    assert!(
        distinct(rs, |r| r.header().engine().identity.logical_id().get()) > 1,
        "lineage"
    );
    assert!(
        distinct(rs, |r| r.header().engine().identity.branch_salt().get()) > 1,
        "salt"
    );
    assert!(
        distinct(rs, |r| r.header().engine().identity.episode_index()) > 1,
        "episode"
    );
    assert!(
        distinct(rs, |r| r.header().agents()[1].episode_return.to_bits()) > 1,
        "accumulators"
    );
}

/// ADR-0046 test 1. `acting` cannot differ between valid roots with one
/// information state here (the engine checks it against the turn order,
/// which the information state determines), and no valid root has
/// nonzero reserved bytes (decoding refuses them): both are fixed by
/// construction rather than varied.
#[test]
fn test_determinize_sample_noninterference_equal_info_gives_identical_records() {
    let roots = equal_info_roots();
    assert!(roots.len() >= 8, "a large group: {}", roots.len());
    assert_group_varies(&roots);
    for index in 0..4 {
        let first = root_bytes(&det(&roots[0], how(77, index)));
        for r in &roots[1..] {
            assert_eq!(root_bytes(&det(r, how(77, index))), first, "sample {index}");
        }
    }
}

/// Priors and value from the leaf's record hash: the search depends on
/// every byte of every leaf it sees.
struct Canned;

impl Evaluator<Dice> for Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Dice>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        for (i, r) in leaves.iter().enumerate() {
            let h = r.hash().get();
            let w0 = f64::from(1 + (h & 0xff) as u32);
            let w1 = f64::from(1 + ((h >> 8) & 0xff) as u32);
            priors[i * 2] = w0 / (w0 + w1);
            priors[i * 2 + 1] = w1 / (w0 + w1);
            values[i] = f64::from((h >> 48) as u32 & 0xffff) / 32_767.5 - 1.0;
        }
        Ok(())
    }
}

fn searched(root: &Record<Dice>, index: u64) -> (u64, Vec<u32>, Vec<u8>) {
    let mut c = PuctConfig::default();
    c.simulations = 96;
    c.batch_size = 4;
    // Through `set_root_determinized`, from a pool made on another sample.
    let mut pool = det(root, how(5, index + 1000));
    pool.set_root_determinized(root, how(5, index)).unwrap();
    let mut m = Mcts::new(pool, c).unwrap();
    m.run(&mut Canned).unwrap();
    let mut visits = vec![0; 2];
    m.root_visits(&mut visits);
    let stats: Vec<u8> = (0..2)
        .flat_map(|a| {
            let e = m.pool().edge_stats(m.pool().root(), a).unwrap();
            let mut b = e.visits.to_le_bytes().to_vec();
            b.extend(e.value.to_bits().to_le_bytes());
            b
        })
        .collect();
    (m.pool().tree_hash(), visits, stats)
}

/// ADR-0046 test 2.
#[test]
fn test_determinize_search_noninterference_equal_info_gives_identical_search() {
    let roots = equal_info_roots();
    for index in 0..2 {
        let first = searched(&roots[0], index);
        assert!(first.1.iter().sum::<u32>() > 0);
        for r in &roots[1..] {
            assert_eq!(searched(r, index), first, "sample {index}");
        }
    }
}

/// Agent 0's noise in the observation of `record`.
fn noise(record: &Record<Dice>) -> u32 {
    let mut p = EnvPool::<Dice, Masked>::with_masks(cfg(), options(1, 0)).unwrap();
    p.restore(0, record).unwrap();
    let bytes = p.outputs().observation(0).unwrap();
    u32::from_le_bytes(bytes[12..16].try_into().unwrap())
}

/// ADR-0046 test 3: a fixed-seed fixture with declared tolerances. Over
/// `N` samples the simulated die and noise of the first child are spread
/// over their ranges (each bin within `TOL` of its expectation) and are
/// not concentrated on the true next events.
#[test]
fn test_determinize_no_future_chance_simulated_events_follow_the_distribution() {
    const N: u32 = 600;
    const TOL: f64 = 0.4;
    let r = root(7, &[1]);
    let mut real = EnvPool::<Dice, Masked>::with_masks(cfg(), options(1, 0)).unwrap();
    real.restore(0, &r).unwrap();
    play(&mut real, 0, &[0]);
    let true_child = real.snapshot(0).unwrap();
    let (true_die, true_noise) = (true_child.state().die, noise(&true_child));

    let mut s = det(&r, how(99, 0));
    let mut dice = vec![0_u32; DIE as usize];
    let mut noises = vec![0_u32; NOISE as usize];
    let mut cards = vec![0_u32; CARDS as usize];
    for i in 0..u64::from(N) {
        let root_h = s.set_root_determinized(&r, how(99, i)).unwrap();
        cards[s.record(root_h).unwrap().state().cards[1] as usize] += 1;
        let mut out = [root_h];
        s.expand(&[(root_h, 0)], &mut out).unwrap();
        let child = *s.record(out[0]).unwrap();
        dice[child.state().die as usize] += 1;
        noises[noise(&child) as usize] += 1;
    }
    let within = |bins: &[u32]| {
        let expected = f64::from(N) / bins.len() as f64;
        bins.iter()
            .all(|&c| (f64::from(c) - expected).abs() <= TOL * expected)
    };
    assert!(within(&dice), "dice {dice:?}");
    assert!(within(&noises), "noise {noises:?}");
    assert!(within(&cards), "hidden cards {cards:?}");
    let share = |c: u32, bins: usize| f64::from(c) / (f64::from(N) / bins as f64);
    assert!(share(dice[true_die as usize], dice.len()) <= 1.0 + TOL);
    assert!(share(noises[true_noise as usize], noises.len()) <= 1.0 + TOL);
}

/// ADR-0046 test 4: the sampler and the rollout of one sample use
/// distinct RNG coordinates (salts, hence keys), checked against the
/// reference vectors; the record is repeatable.
#[test]
fn test_determinize_domain_separation_sampler_and_rollout_keys_differ() {
    assert_eq!(
        (SAMPLER, ROLLOUT),
        (0x5453_4454_5341_4D50, 0x5453_4454_524F_4C4C)
    );
    assert_eq!(sampler_salt(0).get(), 0x6B5D_1A48_306D_A7B8);
    assert_eq!(sampler_salt(1).get(), 0x6766_544A_FFA0_CC49);
    assert_eq!(rollout_salt(0).get(), 0x058A_FC2A_50F7_395F);
    assert_eq!(rollout_salt(1).get(), 0x0D27_A07C_566A_AF22);
    let r = root(3, &[1, 1, 0]);
    for i in 0..4 {
        let s = det(&r, how(11, i));
        let id = s.record(s.root()).unwrap().header().engine().identity;
        assert_eq!(id.branch_salt(), rollout_salt(i));
        assert_eq!(
            (id.seed(), id.logical_id(), id.episode_index()),
            (Seed::new(11), LogicalId::DEFAULT, 0)
        );
        let sampler = sampler_identity(Seed::new(11), i);
        assert_eq!(sampler.branch_salt(), sampler_salt(i));
        assert_ne!(sampler.branch_salt(), id.branch_salt());
        for stream in [StreamId::DYNAMICS, StreamId::OBS_NOISE] {
            assert_ne!(
                derive_key(sampler.seed(), StreamId::DYNAMICS, sampler.branch_salt()),
                derive_key(id.seed(), stream, id.branch_salt()),
            );
        }
        assert_eq!(
            root_bytes(&det(&r, how(11, i))),
            root_bytes(&s),
            "repeatable"
        );
    }
}

/// The copied projection of a root: step index, episode steps, the
/// `alive` and `generation` rows and the adaptive values.
fn projection(r: &Record<Dice>) -> Vec<u8> {
    let e = r.header().engine();
    let mut out = e.identity.step_index().to_le_bytes().to_vec();
    out.extend(e.episode_steps.to_le_bytes());
    for row in r.header().agents() {
        out.push(u8::from(row.alive.get()));
        out.extend(row.generation.to_le_bytes());
    }
    let mut h = *r.header();
    out.extend(encode_to_vec(h.adaptive_mut()));
    out
}

/// ADR-0046 test 5: over rollouts with varied seeds and policies (the
/// game has no adaptive field), the copied projection is single-valued
/// per information state of each agent.
#[test]
fn test_determinize_copied_fields_are_single_valued_per_info_state() {
    let mut seen: [BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>>; 2] = Default::default();
    for seed in 0..120_u64 {
        let mut p = EnvPool::<Dice, Masked>::with_masks(cfg(), options(1, seed)).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        for t in 0..MOVES {
            let r = p.snapshot(0).unwrap();
            for (a, map) in seen.iter_mut().enumerate() {
                map.entry(info_bytes(&r, a as u16))
                    .or_default()
                    .insert(projection(&r));
            }
            play(&mut p, 0, &[u32::from((seed >> t) & 1 == 1)]);
        }
    }
    for map in &seen {
        assert!(map.len() > 50);
        assert!(map.values().all(|v| v.len() == 1));
    }
}

/// ADR-0046 test 6.
#[test]
fn test_determinize_consistency_samples_keep_the_info_state_and_pass_checks() {
    for (seed, moves) in [
        (1, &[][..]),
        (2, &[1]),
        (3, &[0, 1, 1]),
        (4, &[1, 1, 1, 0, 0]),
    ] {
        let r = root(seed, moves);
        for i in 0..8 {
            let s = det(&r, how(seed + 40, i));
            let d = *s.record(s.root()).unwrap();
            assert_eq!(info_bytes(&d, 0), info_bytes(&r, 0));
            // The output header: the copied step clock (nonzero once a
            // move was played) and the rollout identity.
            let (out, from) = (d.header().engine(), r.header().engine());
            assert_eq!(out.identity.step_index(), from.identity.step_index());
            assert_eq!(out.episode_steps, from.episode_steps);
            assert_eq!(out.episode_steps, moves.len() as u32);
            assert_eq!(out.identity.seed(), Seed::new(seed + 40));
            assert_eq!(out.identity.branch_salt(), rollout_salt(i));
            assert!(!out.pending_reset.get() && out.end_kind == 0 && !out.fresh.get());
            for (k, row) in d.header().agents().iter().enumerate() {
                let slot = agent(k as u16);
                assert_eq!(
                    row.acting.get(),
                    row.alive.get() && Dice::acting(&cfg(), d.state(), slot)
                );
                assert_eq!(
                    (
                        row.episode_return,
                        row.illegal_actions,
                        row.invalid_agent_events
                    ),
                    (0.0, 0, 0)
                );
            }
            let mut p = EnvPool::<Dice, Masked>::with_masks(cfg(), options(1, 0)).unwrap();
            p.import(&[0], &[DecodedRecord::from(d)]).unwrap();
        }
    }
}

/// ADR-0046 test 7.
#[test]
fn test_determinize_sampling_leaves_the_root_unchanged_in_any_order() {
    let r = root(9, &[0, 1]);
    let mut s = det(&r, how(1, 0));
    let mut forward = Vec::new();
    for i in [0, 1, 2] {
        s.set_root_determinized(&r, how(1, i)).unwrap();
        forward.push(root_bytes(&s));
    }
    for (i, k) in [(2, 2), (0, 0), (1, 1), (2, 2)] {
        s.set_root_determinized(&r, how(1, i)).unwrap();
        assert_eq!(root_bytes(&s), forward[k]);
    }
    assert_ne!(forward[0], forward[1], "indices give different samples");
}

/// ADR-0046 test 8: each refusal is its typed error and leaves the tree
/// unchanged.
#[test]
fn test_determinize_refused_roots_and_operations_give_typed_errors() {
    let r = root(4, &[1]);
    let finished = root(4, &[0; MOVES as usize]);
    assert!(finished.header().engine().pending_reset.get());
    assert_eq!(
        err(det_with(cfg(), &finished, how(1, 0))),
        SearchError::PendingResetRoot
    );

    let mut s = det(&r, how(1, 0));
    let h = s.root();
    let mut out = [h];
    s.expand(&[(h, 1)], &mut out).unwrap();
    let before = s.tree_hash();
    assert_eq!(
        s.set_root_determinized(&finished, how(1, 1)).unwrap_err(),
        SearchError::PendingResetRoot
    );
    assert_eq!(
        s.reroot(out[0]).unwrap_err(),
        SearchError::DeterminizedReroot
    );
    assert_eq!(s.set_root(&r).unwrap_err(), SearchError::DeterminizedPool);
    assert_eq!(s.tree_hash(), before);

    // Readiness: a scratch pool whose step limit the root exceeds.
    let longer = root(4, &[1, 1, 1]);
    let short = |c, mut o: PoolOptions| {
        o.max_episode_steps = NonZeroU32::new(2);
        EnvPool::with_masks(c, o)
    };
    let e = err(SearchPool::new_determinized(
        cfg(),
        search_options(),
        &longer,
        how(1, 0),
        short,
    ));
    assert!(matches!(e, SearchError::RootNotReady { .. }), "{e:?}");

    assert_eq!(
        err(det_with(DiceConfig { sabotage: 2 }, &r, how(1, 0))),
        SearchError::SampleNotSingleActor { acting: 0 }
    );
    assert!(matches!(
        err(det_with(DiceConfig { sabotage: 1 }, &r, how(1, 0))),
        SearchError::InconsistentSample { .. }
    ));
}

#[test]
fn test_determinize_new_refuses_and_new_privileged_labels_imperfect_information() {
    let r = root(2, &[1]);
    let plain = SearchPool::<Dice, Masked>::new(cfg(), search_options(), &r, EnvPool::with_masks);
    assert_eq!(err(plain), SearchError::ImperfectInformation);
    let mut p = SearchPool::<Dice, Masked>::new_privileged(
        cfg(),
        search_options(),
        &r,
        EnvPool::with_masks,
    )
    .unwrap();
    assert!(p.is_privileged() && !p.is_determinized());
    assert_eq!(
        p.record(p.root()).unwrap().to_bytes(),
        r.to_bytes(),
        "the full root"
    );
    assert_eq!(
        p.set_root_determinized(&r, how(1, 0)).unwrap_err(),
        SearchError::NotDeterminized
    );
    let d = det(&r, how(1, 0));
    assert!(d.is_determinized() && !d.is_privileged());
}

/// The privileged label reaches the MCTS output and the tree hash: a
/// privileged and a non-privileged tree over the same record never hash
/// equal, while non-privileged hashes are unchanged (the golden hashes
/// of the perfect-information tests check that).
#[test]
fn test_determinize_privileged_label_is_in_the_mcts_results() {
    use tessera_rl_games::{Connect4, Connect4Config};

    let rec = root(2, &[1]);
    let mut puct = PuctConfig::default();
    puct.simulations = 16;
    puct.batch_size = 4;
    let privileged = SearchPool::<Dice, Masked>::new_privileged(
        cfg(),
        search_options(),
        &rec,
        EnvPool::with_masks,
    )
    .unwrap();
    let mut mcts = Mcts::new(privileged, puct).unwrap();
    mcts.run(&mut Canned).unwrap();
    assert!(mcts.is_privileged());
    let plain_det = Mcts::new(det(&rec, how(1, 0)), puct).unwrap();
    assert!(!plain_det.is_privileged());

    // The same perfect-information record, through `new` and through
    // `new_privileged` (which does not label it): equal hashes. A
    // privileged tree differs from the same tree unlabelled.
    let mut p4 = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(3)),
    )
    .unwrap();
    p4.reset_all(Reseed::Keep).unwrap();
    let r4 = p4.snapshot(0).unwrap();
    let opts = search_options();
    let plain = SearchPool::new(Connect4Config {}, opts, &r4, EnvPool::with_masks).unwrap();
    let same =
        SearchPool::new_privileged(Connect4Config {}, opts, &r4, EnvPool::with_masks).unwrap();
    assert!(!same.is_privileged());
    assert_eq!(plain.tree_hash(), same.tree_hash());

    // A privileged and an unlabelled tree over one record differ only
    // by the label: the unit test in the crate compares them.
}

/// ADR-0046 test 4, the real path: the sampler draws with the sampler
/// identity on the dynamics stream. The opponent's card equals the first
/// draw of a `StepRng` built independently from `sampler_identity`, at
/// samples where the rollout identity or the observation stream would
/// give another card.
#[test]
fn test_determinize_sampler_draws_with_the_sampler_identity_on_dynamics() {
    let r = root(6, &[1, 0, 1]);
    let auth = RngAuthority::engine_only();
    let card = |id: &tessera_rl_num::identity::RngIdentity, stream| {
        StepRng::new(&auth, id, stream).below_u32(NonZeroU32::new(CARDS).unwrap())
    };
    let mut telling = 0;
    for i in 0..32 {
        let expected = card(&sampler_identity(Seed::new(13), i), StreamId::DYNAMICS);
        let rollout = card(&rollout_identity(Seed::new(13), i, 0), StreamId::DYNAMICS);
        let obs = card(&sampler_identity(Seed::new(13), i), StreamId::OBS_NOISE);
        let s = det(&r, how(13, i));
        assert_eq!(
            s.record(s.root()).unwrap().state().cards[1],
            expected,
            "sample {i}"
        );
        telling += usize::from(expected != rollout && expected != obs);
    }
    assert!(telling >= 8, "{telling} samples tell the identities apart");
}

/// The copied step clock has a consequence: a root one step before the
/// step limit gives a determinized child truncated at the same step as
/// the real one.
#[test]
fn test_determinize_step_clock_truncates_at_the_same_step() {
    let limit = NonZeroU32::new(3);
    let mut real = EnvPool::<Dice, Masked>::with_masks(
        cfg(),
        PoolOptions {
            max_episode_steps: limit,
            ..options(1, 8)
        },
    )
    .unwrap();
    real.reset_all(Reseed::Keep).unwrap();
    play(&mut real, 0, &[1, 1]);
    let r = real.snapshot(0).unwrap();
    play(&mut real, 0, &[0]);
    let true_end = real.snapshot(0).unwrap().header().engine().end_kind;
    assert_eq!(true_end, 2, "the real step truncates");

    let limited = |c, mut o: PoolOptions| {
        o.max_episode_steps = limit;
        EnvPool::with_masks(c, o)
    };
    let mut s =
        SearchPool::new_determinized(cfg(), search_options(), &r, how(3, 0), limited).unwrap();
    let root_h = s.root();
    assert_eq!(s.record(root_h).unwrap().header().engine().episode_steps, 2);
    let mut out = [root_h];
    s.expand(&[(root_h, 0)], &mut out).unwrap();
    assert!(s.is_terminal(out[0]).unwrap());
    let child = s.record(out[0]).unwrap().header().engine();
    assert_eq!((child.end_kind, child.episode_steps), (true_end, 3));
}

fn fog_root<const MIXED: bool>(seed: u64, steps: u32) -> Record<Fog<MIXED>> {
    let mut p = EnvPool::<Fog<MIXED>, Masked>::with_masks(FogConfig {}, options(1, seed)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..steps {
        p.step(ActionInput {
            discrete: &[0; 3],
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn fog_det<const MIXED: bool>(r: &Record<Fog<MIXED>>, i: u64) -> Record<Fog<MIXED>> {
    let s = SearchPool::<Fog<MIXED>, Masked>::new_determinized(
        FogConfig {},
        search_options(),
        r,
        how(17, i),
        EnvPool::with_masks,
    )
    .unwrap();
    *s.record(s.root()).unwrap()
}

/// The output rows of a root with a dead slot (1) and a respawned slot
/// (2, generation 1): `alive` and `generation` are copied, and `acting`
/// follows the recomputation rule. Under `Mixed`, `acting` is
/// `Environment::acting` of the sample, which differs from the root's
/// when the hidden coins differ; a dead slot never acts.
#[test]
fn test_determinize_rows_copy_alive_and_generation_and_recompute_mixed_acting() {
    let r = fog_root::<true>(4, 3);
    let rows = r.header().agents();
    assert_eq!(
        rows.iter().map(|x| x.alive.get()).collect::<Vec<_>>(),
        [true, false, true]
    );
    assert_eq!(
        rows.iter().map(|x| x.generation).collect::<Vec<_>>(),
        [0, 0, 1]
    );
    let mut differs = 0;
    for i in 0..16 {
        let d = fog_det(&r, i);
        for (k, (out, from)) in d.header().agents().iter().zip(rows).enumerate() {
            assert_eq!((out.alive, out.generation), (from.alive, from.generation));
            let expected = out.alive.get()
                && <Fog<true> as Environment>::acting(&FogConfig {}, d.state(), agent3(k));
            assert_eq!(out.acting.get(), expected, "sample {i} slot {k}");
            differs += usize::from(out.acting != from.acting);
        }
        assert!(
            !d.header().agents()[1].acting.get(),
            "the dead slot does not act"
        );
    }
    assert!(differs > 0, "some sample recomputes another acting row");
}

/// Under `Simultaneous`, every alive slot acts and the dead one does not,
/// whatever `Environment::acting` says of the sample.
#[test]
fn test_determinize_simultaneous_acting_is_every_alive_slot() {
    let r = fog_root::<false>(4, 3);
    let mut acting_false = 0;
    for i in 0..16 {
        let d = fog_det(&r, i);
        for (k, out) in d.header().agents().iter().enumerate() {
            assert_eq!(out.acting, out.alive, "sample {i} slot {k}");
            acting_false += usize::from(
                out.alive.get()
                    && !<Fog<false> as Environment>::acting(&FogConfig {}, d.state(), agent3(k)),
            );
        }
        assert_eq!(d.header().agents()[2].generation, 1);
    }
    assert!(acting_false > 0, "the rule is not Environment::acting");
}

fn agent3(k: usize) -> AgentSlot {
    AgentSlot::new(k as u16, 3).unwrap()
}

/// Author code runs contained and on a validated sample: a panicking
/// sampler is a typed error, and so is a sample that fails `Validate`.
#[test]
fn test_determinize_author_panics_and_invalid_samples_are_typed_errors() {
    let r = root(4, &[1]);
    assert_eq!(
        err(det_with(DiceConfig { sabotage: 3 }, &r, how(1, 0))),
        SearchError::AuthorPanicked
    );
    assert!(matches!(
        err(det_with(DiceConfig { sabotage: 4 }, &r, how(1, 0))),
        SearchError::SampleInvariant { .. }
    ));
}

/// ADR-0046 Rule: an environment with an adaptive field cannot be
/// determinized yet.
#[test]
fn test_determinize_adaptive_environments_are_refused() {
    let mut p = EnvPool::<DiceWith<Level>, Masked>::with_masks(cfg(), options(1, 2)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let r = p.snapshot(0).unwrap();
    let made = SearchPool::<DiceWith<Level>, Masked>::new_determinized(
        cfg(),
        search_options(),
        &r,
        how(1, 0),
        EnvPool::with_masks,
    );
    assert_eq!(err(made), SearchError::AdaptiveDeterminize);
}

/// No node slot, staging buffer or scratch row keeps the full root,
/// after construction and after `set_root_determinized`.
#[test]
fn test_determinize_no_buffer_keeps_the_full_root() {
    let r = root(5, &[1, 0]);
    let mut s = det(&r, how(2, 0));
    assert!(!s.holds_record(&r));
    let other = root(6, &[0]);
    s.set_root_determinized(&other, how(2, 1)).unwrap();
    assert!(!s.holds_record(&other) && !s.holds_record(&r));
    let sampled = *s.record(s.root()).unwrap();
    assert!(s.buffers_hold_only(&sampled), "refilled with the sample");
    let h = s.root();
    let mut out = [h];
    s.expand(&[(h, 0)], &mut out).unwrap();
    assert!(!s.holds_record(&other));
    // A sample refused after the root passed the readiness checks.
    let mut late = det_with(DiceConfig { sabotage: 5 }, &r, how(2, 0)).unwrap();
    let three = root(5, &[1, 0, 1]);
    let refused = late.set_root_determinized(&three, how(2, 1));
    assert!(matches!(
        err(refused),
        SearchError::InconsistentSample { .. }
    ));
    assert!(!late.holds_record(&three));
    // A refused root does not stay either.
    let short = |c, mut o: PoolOptions| {
        o.max_episode_steps = NonZeroU32::new(2);
        EnvPool::with_masks(c, o)
    };
    let mut limited =
        SearchPool::new_determinized(cfg(), search_options(), &r, how(2, 0), short).unwrap();
    let longer = root(5, &[1, 0, 1]);
    let refused = limited.set_root_determinized(&longer, how(2, 1));
    assert!(matches!(err(refused), SearchError::RootNotReady { .. }));
    assert!(!limited.holds_record(&longer) && !limited.holds_record(&r));
}

/// A failed expansion poisons the scratch pool (a contained kernel
/// panic); the next determinization recovers it before its checks.
#[test]
fn test_determinize_recovers_a_scratch_pool_poisoned_by_a_failed_step() {
    let c = DiceConfig { sabotage: 6 };
    let mut s = det_with(c, &root(4, &[1]), how(1, 0)).unwrap();
    let h = s.root();
    let mut out = [h];
    assert!(matches!(
        s.expand(&[(h, 1)], &mut out),
        Err(SearchError::Engine(_))
    ));
    let other = root(5, &[0, 1]);
    let new_root = s.set_root_determinized(&other, how(1, 1)).unwrap();
    let rec = *s.record(new_root).unwrap();
    assert_eq!(info_bytes(&rec, 0), info_bytes(&other, 0));
    assert!(s.buffers_hold_only(&rec));
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Go 9x9 through the engine: spaces, turns, observations, scoring, the
//! mask, the cap and import validation. The rules are checked against an
//! array-based reference written only for these tests, sharing no code with
//! the bitboards, and against positions whose result is known by hand.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    reason = "test helpers on a 9 x 9 board: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Bool8, EnvConfig, Environment, Heuristic, RecordLayout, Validate,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_games::go9::{ACTIONS, MAX_PLIES, PASS, pack, unpack};
use tessera_rl_games::{Go9, Go9Config, Go9State};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

fn cfg() -> Go9Config {
    Go9Config::default()
}

/// The action of the point at `row`, `col`.
const fn at(row: u32, col: u32) -> u32 {
    row * 9 + col
}

/// `SplitMix64`, for reproducible random games.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn pick(&mut self, items: &[u32]) -> u32 {
        items[(self.next() % items.len() as u64) as usize]
    }
}

/// The reference: `cells[row * 9 + col]` is 0 empty, 1 black, 2 white. It
/// finds groups by walking neighbours and keeps every position in a `Vec`.
#[derive(Clone)]
struct Ref {
    cells: [u8; 81],
    side: u8,
    passes: u32,
    plies: u32,
    seen: Vec<[u8; 81]>,
}

fn adjacent(i: usize) -> Vec<usize> {
    let (r, c) = (i / 9, i % 9);
    let mut out = Vec::new();
    if r > 0 {
        out.push(i - 9);
    }
    if r < 8 {
        out.push(i + 9);
    }
    if c > 0 {
        out.push(i - 1);
    }
    if c < 8 {
        out.push(i + 1);
    }
    out
}

/// The group of `i` and whether it has a liberty.
fn group_of(cells: &[u8; 81], i: usize) -> (Vec<usize>, bool) {
    let colour = cells[i];
    let (mut stack, mut found, mut liberty) = (vec![i], vec![i], false);
    while let Some(k) = stack.pop() {
        for n in adjacent(k) {
            if cells[n] == 0 {
                liberty = true;
            } else if cells[n] == colour && !found.contains(&n) {
                found.push(n);
                stack.push(n);
            }
        }
    }
    (found, liberty)
}

impl Ref {
    fn new() -> Self {
        Self {
            cells: [0; 81],
            side: 1,
            passes: 0,
            plies: 0,
            seen: Vec::new(),
        }
    }

    /// The cells after `side` places on `i`, if legal.
    fn after(&self, i: usize) -> Option<[u8; 81]> {
        if self.cells[i] != 0 {
            return None;
        }
        let mut cells = self.cells;
        cells[i] = self.side;
        for n in adjacent(i) {
            if cells[n] == 3 - self.side {
                let (g, liberty) = group_of(&cells, n);
                if !liberty {
                    for k in g {
                        cells[k] = 0;
                    }
                }
            }
        }
        if !group_of(&cells, i).1 || self.seen.contains(&cells) {
            return None;
        }
        Some(cells)
    }

    fn actions(&self) -> Vec<u32> {
        let mut out: Vec<u32> = (0..81)
            .filter(|&i| self.after(i as usize).is_some())
            .collect();
        out.push(PASS);
        out
    }

    fn play(&mut self, a: u32) {
        if a == PASS {
            self.passes += 1;
        } else {
            self.cells = self.after(a as usize).unwrap();
            self.seen.push(self.cells);
            self.passes = 0;
        }
        self.plies += 1;
        self.side = 3 - self.side;
    }

    fn is_over(&self) -> bool {
        self.passes >= 2 || self.plies >= MAX_PLIES
    }

    fn stones(&self) -> (u128, u128) {
        let (mut b, mut w) = (0, 0);
        for (i, &v) in self.cells.iter().enumerate() {
            match v {
                1 => b |= 1u128 << i,
                2 => w |= 1u128 << i,
                _ => {}
            }
        }
        (b, w)
    }

    /// Area scores by walking each empty region.
    fn areas(&self) -> (u32, u32) {
        let (mut b, mut w) = (0, 0);
        let mut done = [false; 81];
        for i in 0..81 {
            match self.cells[i] {
                1 => b += 1,
                2 => w += 1,
                _ if !done[i] => {
                    let (mut stack, mut size, mut touch) = (vec![i], 0, [false; 3]);
                    done[i] = true;
                    while let Some(k) = stack.pop() {
                        size += 1;
                        for n in adjacent(k) {
                            let v = self.cells[n];
                            touch[usize::from(v)] = true;
                            if v == 0 && !done[n] {
                                done[n] = true;
                                stack.push(n);
                            }
                        }
                    }
                    match (touch[1], touch[2]) {
                        (true, false) => b += size,
                        (false, true) => w += size,
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        (b, w)
    }
}

fn legal_actions(s: &Go9State) -> Vec<u32> {
    (0..ACTIONS).filter(|&a| s.is_legal(a)).collect()
}

fn perft_state(s: &Go9State, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if s.is_over() {
        return 0;
    }
    legal_actions(s)
        .into_iter()
        .map(|a| {
            let mut t = *s;
            t.play(a);
            perft_state(&t, depth - 1)
        })
        .sum()
}

fn perft_ref(r: &Ref, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if r.is_over() {
        return 0;
    }
    r.actions()
        .into_iter()
        .map(|a| {
            let mut t = r.clone();
            t.play(a);
            perft_ref(&t, depth - 1)
        })
        .sum()
}

#[test]
fn test_go9_perft_matches_closed_forms_and_the_reference() {
    let s = Go9State::default();
    let r = Ref::new();
    assert_eq!(perft_state(&s, 1), 82);
    // 81 placements with 81 replies each (80 points and the pass), and the
    // pass with 82.
    assert_eq!(perft_state(&s, 2), 81 * 81 + 82);
    assert_eq!(perft_state(&s, 3), perft_ref(&r, 3));
}

/// A random game, checked ply by ply against the reference. `pass_per_mille`
/// biases long games; returns every state and how many placements superko
/// alone barred.
fn random_game(seed: u64, pass_per_mille: u64) -> (Vec<Go9State>, u32) {
    let mut rng = Rng(seed);
    let (mut s, mut r) = (Go9State::default(), Ref::new());
    let mut states = vec![s];
    let mut barred = 0;
    while !s.is_over() {
        let legal = legal_actions(&s);
        assert_eq!(legal, r.actions(), "seed {seed}, ply {}", s.moves);
        let (me, opp) = s.sides();
        for sq in 0..81u32 {
            let bit = 1u128 << sq;
            if (me | opp) & bit == 0 && !legal.contains(&sq) {
                let mut free = r.clone();
                free.seen.clear();
                barred += u32::from(free.after(sq as usize).is_some());
            }
        }
        let a = if rng.next() % 1000 < pass_per_mille || legal.len() == 1 {
            PASS
        } else {
            let placements = &legal[..legal.len() - 1];
            rng.pick(placements)
        };
        s.play(a);
        r.play(a);
        assert_eq!(s.sets(), r.stones(), "seed {seed}, ply {}", s.moves);
        assert_eq!(s.is_over(), r.is_over(), "seed {seed}, ply {}", s.moves);
        states.push(s);
    }
    assert_eq!(s.areas(), r.areas(), "seed {seed}");
    (states, barred)
}

#[test]
fn test_go9_random_games_match_the_reference_and_stay_valid() {
    let (mut barred, mut capped) = (0, 0);
    for seed in 0..60 {
        let (states, b) = random_game(seed, 5 * (seed % 3));
        barred += b;
        for s in &states {
            assert_eq!(Go9::validate(&cfg(), s), Ok(()), "seed {seed}");
        }
        capped += u32::from(states[states.len() - 1].moves == MAX_PLIES);
    }
    assert!(capped > 0, "no game reached the cap");
    assert!(barred > 0, "superko never barred a move");
}

fn pool_with(auto_reset: AutoReset, c: Go9Config) -> EnvPool<Go9, Masked> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(11))
    };
    let mut p = EnvPool::with_masks(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool() -> EnvPool<Go9, Masked> {
    pool_with(AutoReset::Disabled, cfg())
}

fn play(p: &mut EnvPool<Go9, Masked>, action: u32) {
    let to_move = p.state(0).unwrap().to_move();
    let mut act = [PASS.cast_signed(); 2];
    act[to_move] = action.cast_signed();
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn mask(p: &EnvPool<Go9, Masked>, agent: usize) -> Vec<u32> {
    let n = ACTIONS as usize;
    p.outputs().unwrap().masks()[agent * n..(agent + 1) * n]
        .iter()
        .enumerate()
        .filter(|(_, f)| f.get())
        .map(|(a, _)| u32::try_from(a).unwrap())
        .collect()
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

fn import(p: &mut EnvPool<Go9, Masked>, state: &Go9State) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    for (k, row) in header.agents_mut().iter_mut().enumerate() {
        let slot = AgentSlot::new(u16::try_from(k).unwrap(), 2).unwrap();
        row.acting = Bool8::from(Go9::acting(&cfg(), state, slot));
    }
    let rec = DecodedRecord::from_parts(header, *state).unwrap();
    p.import(&[0], &[rec])
}

/// Plays a state through `actions`, each checked legal.
fn state_of(actions: &[u32]) -> Go9State {
    let mut s = Go9State::default();
    for &a in actions {
        assert!(!s.is_over(), "ply {}: the game is over", s.moves);
        assert!(s.is_legal(a), "ply {}: {a}", s.moves);
        s.play(a);
    }
    s
}

/// Black and white stones in turn; the side with more then plays alone, the
/// other passing, so any shape can be built without two passes in a row.
fn setup(black: &[u32], white: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let shared = black.len().min(white.len());
    for k in 0..shared {
        out.push(black[k]);
        out.push(white[k]);
    }
    for &b in &black[shared..] {
        out.push(b);
        out.push(PASS);
    }
    for &w in &white[shared..] {
        out.push(PASS);
        out.push(w);
    }
    out
}

#[test]
fn test_go9_spaces_are_two_u8_planes_and_82_actions() {
    let expected = Space::Dict(vec![(
        "board".to_owned(),
        Space::Box {
            shape: vec![2, 9, 9],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![1],
            },
        },
    )]);
    assert_eq!(Go9::observation_space(&cfg()).unwrap(), expected);
    assert_eq!(
        Go9::action_space(&cfg()).unwrap(),
        Space::Discrete { n: 82 }
    );
}

#[test]
fn test_go9_config_rejects_komi_out_of_range() {
    assert!(
        Go9Config {
            komi_half_points: 163
        }
        .validate()
        .is_err()
    );
    assert!(
        Go9Config {
            komi_half_points: -163
        }
        .validate()
        .is_err()
    );
    assert!(
        Go9Config {
            komi_half_points: -162
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn test_go9_observation_is_from_each_players_view() {
    let mut p = pool();
    play(&mut p, at(4, 4));
    play(&mut p, at(0, 0));
    let obs = p.outputs().unwrap().observation(0).unwrap().to_vec();
    assert_eq!(obs.len(), 2 * 162);
    let cell =
        |agent: usize, channel: usize, sq: u32| obs[agent * 162 + channel * 81 + sq as usize];
    assert_eq!(cell(0, 0, at(4, 4)), 1);
    assert_eq!(cell(0, 1, at(0, 0)), 1);
    assert_eq!(cell(1, 0, at(0, 0)), 1);
    assert_eq!(cell(1, 1, at(4, 4)), 1);
    assert_eq!(obs.iter().map(|&v| u32::from(v)).sum::<u32>(), 4);
}

#[test]
fn test_go9_capture_removes_a_corner_stone() {
    let s = state_of(&[at(0, 0), at(0, 1), PASS, at(1, 0)]);
    let (black, white) = s.sets();
    assert_eq!(black, 0);
    assert_eq!(white.count_ones(), 2);
    assert_eq!(s.last_capture, 3);
}

#[test]
fn test_go9_capture_takes_two_groups_at_once() {
    // White stones at (0,0) and (0,2), each short of one liberty at (0,1);
    // black (0,1) takes both.
    let s = state_of(&[
        at(1, 0),
        at(0, 0),
        at(1, 2),
        at(0, 2),
        at(0, 3),
        PASS,
        at(0, 1),
    ]);
    let (black, white) = s.sets();
    assert_eq!(white, 0);
    assert_eq!(black.count_ones(), 4);
}

#[test]
fn test_go9_suicide_is_illegal_and_masked() {
    let s = state_of(&setup(&[], &[at(0, 1), at(1, 0)]));
    assert_eq!(s.to_move(), 0);
    assert!(!s.is_legal(at(0, 0)));
    assert!(s.is_legal(at(8, 8)));
    let mut p = pool();
    import(&mut p, &s).unwrap();
    assert!(!mask(&p, 0).contains(&at(0, 0)));
}

#[test]
fn test_go9_multi_stone_suicide_is_illegal() {
    // Black (0,0) would join (0,1), and the pair has no liberty left.
    let s = state_of(&[at(0, 1), at(0, 2), PASS, at(1, 1), PASS, at(1, 0)]);
    assert_eq!(s.to_move(), 0);
    assert!(!s.is_legal(at(0, 0)));
}

#[test]
fn test_go9_capture_that_would_otherwise_be_suicide_is_legal() {
    // White (0,0) has only the liberty (0,1); black there has none of its
    // own until the capture.
    let s = state_of(&[at(1, 0), at(0, 0), at(1, 1), at(0, 2), PASS, at(1, 2)]);
    assert_eq!(s.to_move(), 0);
    assert!(s.is_legal(at(0, 1)));
    let mut t = s;
    t.play(at(0, 1));
    assert_eq!(t.sets().1 & (1u128 << at(0, 0)), 0, "the corner is taken");
}

/// One ko at rows `r` to `r + 2`: black (r, 1), (r+1, 0), (r+2, 1) and white
/// (r, 2), (r+1, 3), (r+2, 2); its points are (r+1, 1) and (r+1, 2).
fn ko_shape(r: u32) -> ([u32; 3], [u32; 3]) {
    (
        [at(r, 1), at(r + 1, 0), at(r + 2, 1)],
        [at(r, 2), at(r + 1, 3), at(r + 2, 2)],
    )
}

#[test]
fn test_go9_simple_ko_cannot_be_retaken_at_once_but_can_after_a_move() {
    let (b, w) = ko_shape(0);
    let mut moves = setup(&b, &w);
    moves.extend([at(1, 2), at(1, 1)]); // Black in, white takes.
    let s = state_of(&moves);
    assert_eq!(s.to_move(), 0);
    assert!(!s.is_legal(at(1, 2)), "the retake repeats the position");
    let mut t = s;
    t.play(at(8, 8));
    t.play(at(8, 0));
    assert!(t.is_legal(at(1, 2)), "new stones make the position new");
}

/// Two kos cycle: the retake repeats a position from four plies back, with
/// other plies in between, so a rule that only bars the immediate retake
/// would allow it. Positional superko bars it.
#[test]
fn test_go9_double_ko_cycle_is_barred_by_superko_alone() {
    let (b1, w1) = ko_shape(0);
    let (b2, w2) = ko_shape(4);
    let black: Vec<u32> = b1.iter().chain(&b2).copied().collect();
    let white: Vec<u32> = w1.iter().chain(&w2).copied().collect();
    let mut moves = setup(&black, &white);
    // Black enters both kos, white taking the first: A held by white, B
    // held by black. This is the position S, black to move.
    moves.extend([at(5, 2), PASS, at(1, 2), at(1, 1)]);
    let s_position = state_of(&moves);
    assert_eq!(s_position.to_move(), 0);
    let mut s = s_position;
    // Black must wait a ply before retaking A; a far stone would change the
    // position for good, so white's next capture comes after a black pass.
    s.play(PASS);
    assert_eq!(s.to_move(), 1);
    // White takes B: A and B are both white's.
    assert!(s.is_legal(at(5, 1)));
    s.play(at(5, 1));
    // Black retakes A with a position not seen before.
    assert!(s.is_legal(at(1, 2)));
    s.play(at(1, 2));
    // White cannot retake A at once (simple ko), so it passes.
    assert!(!s.is_legal(at(1, 1)));
    s.play(PASS);
    // Black retaking B would give A black and B black, the position after
    // black's (1,2) entry earlier: not the last position, yet a repeat.
    let mut reference = Ref::new();
    let mut replay = Go9State::default();
    for &a in moves.iter().chain(&[PASS, at(5, 1), at(1, 2), PASS]) {
        reference.play(a);
        replay.play(a);
    }
    assert_eq!(replay.sets(), s.sets());
    assert!(!s.is_legal(at(5, 2)), "superko bars the cycle");
    assert!(reference.after(at(5, 2) as usize).is_none());
    let mut free = reference.clone();
    free.seen.clear();
    assert!(
        free.after(at(5, 2) as usize).is_some(),
        "only superko bars it"
    );
    assert!(s.is_legal(PASS));
}

/// A non-capturing placement that repeats a position: white takes two stones
/// with one, black takes that one back and then fills the second point. The
/// random games never reach this, so it is the only check that a placement
/// without a capture still looks behind the last capture.
#[test]
fn test_go9_non_capturing_placement_is_barred_by_superko() {
    let s = state_of(&[
        at(0, 0),
        at(1, 0),
        at(0, 1),
        at(1, 1),
        at(0, 3),
        PASS,
        at(1, 2),
        at(0, 2), // White takes (0,0) and (0,1).
        at(0, 1), // Black takes (0,2).
        PASS,
    ]);
    assert_eq!(s.to_move(), 0);
    assert_eq!(
        s.sets().1 & (1u128 << at(0, 2)),
        0,
        "the white stone is gone"
    );
    assert!(!s.is_legal(at(0, 0)), "this would repeat the position");
    let mut other = s;
    other.play(at(8, 8));
    other.play(PASS);
    assert!(other.is_legal(at(0, 0)), "with a new stone it is new");
}

#[test]
fn test_go9_two_passes_end_with_komi_deciding_an_empty_board() {
    let mut p = pool();
    play(&mut p, PASS);
    assert!(!p.outputs().unwrap().episode_terminated()[0].get());
    play(&mut p, PASS);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
    assert_eq!(
        p.outputs().unwrap().rewards()[..2],
        [-1.0, 1.0],
        "white by komi"
    );
}

#[test]
fn test_go9_area_scoring_counts_stones_and_single_colour_regions() {
    // Black wall on column 3, white wall on column 5: the columns left of
    // black are black's, right of white are white's, column 4 is neutral.
    let black: Vec<u32> = (0..9).map(|r| at(r, 3)).collect();
    let white: Vec<u32> = (0..9).map(|r| at(r, 5)).collect();
    let mut moves = Vec::new();
    for k in 0..9 {
        moves.push(black[k]);
        moves.push(white[k]);
    }
    let s = state_of(&moves);
    assert_eq!(s.areas(), (36, 36));
    assert_eq!(s.margin_half_points(15), -15);
    assert_eq!(s.margin_half_points(0), 0);
}

#[test]
fn test_go9_even_komi_tie_pays_nothing() {
    let c = Go9Config {
        komi_half_points: 0,
    };
    let mut p = pool_with(AutoReset::Disabled, c);
    play(&mut p, PASS);
    play(&mut p, PASS);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
    assert_eq!(p.outputs().unwrap().rewards()[..2], [0.0, 0.0]);
}

#[test]
fn test_go9_cap_scores_as_a_termination_at_ply_162() {
    let mut rng = Rng(9);
    let mut p = pool();
    let mut s = Go9State::default();
    let mut capped = false;
    for _ in 0..40 {
        p.reset_all(Reseed::Keep).unwrap();
        s = Go9State::default();
        while !s.is_over() {
            let legal = legal_actions(&s);
            let a = if legal.len() > 1 {
                rng.pick(&legal[..legal.len() - 1])
            } else {
                PASS
            };
            s.play(a);
            assert!(!p.outputs().unwrap().episode_terminated()[0].get());
            play(&mut p, a);
        }
        if s.moves == MAX_PLIES {
            capped = true;
            break;
        }
    }
    assert!(capped, "no game reached the cap");
    assert_eq!(s.moves, 162);
    assert!(s.passes_in_row < 2);
    assert!(p.outputs().unwrap().episode_terminated()[0].get());
    assert!(!p.outputs().unwrap().episode_truncated()[0].get());
    let margin = s.margin_half_points(15);
    let r = f32::from(i8::from(margin > 0) - i8::from(margin < 0));
    assert_eq!(p.outputs().unwrap().rewards()[..2], [r, -r]);
}

#[test]
fn test_go9_illegal_point_is_replaced_by_the_pass_or_rejected() {
    let mut p = pool();
    play(&mut p, at(0, 0));
    play(&mut p, PASS);
    let before = *p.state(0).unwrap();
    let result = p.step(ActionInput {
        discrete: &[at(0, 0).cast_signed(), PASS.cast_signed()],
        continuous: &[],
    });
    if let Err(e) = result {
        assert!(matches!(e, EngineError::IllegalAction { agent: 0, .. }));
        assert_eq!(*p.state(0).unwrap(), before);
    } else {
        assert_eq!(p.state(0).unwrap().sets(), before.sets());
        assert_eq!(p.state(0).unwrap().passes_in_row, 2, "a second pass");
        assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 1);
    }
}

#[test]
fn test_go9_mask_equals_the_reference_along_random_games() {
    for seed in 0..8 {
        let mut rng = Rng(seed);
        let mut p = pool();
        let mut r = Ref::new();
        while !r.is_over() {
            let actions = r.actions();
            assert_eq!(mask(&p, usize::from(r.side - 1)), actions, "seed {seed}");
            let a = if actions.len() > 1 {
                rng.pick(&actions[..actions.len() - 1])
            } else {
                PASS
            };
            play(&mut p, a);
            r.play(a);
        }
    }
}

#[test]
fn test_go9_snapshot_and_restore_keep_the_superko_history() {
    let (b, w) = ko_shape(0);
    let mut moves = setup(&b, &w);
    moves.extend([at(1, 2), at(1, 1)]);
    let mut p = pool();
    for &a in &moves {
        play(&mut p, a);
    }
    let snap = p.snapshot(0).unwrap();
    let fixed = *p.state(0).unwrap();
    play(&mut p, at(8, 8));
    p.restore(0, &snap).unwrap();
    assert_eq!(*p.state(0).unwrap(), fixed);
    assert!(!mask(&p, 0).contains(&at(1, 2)), "the ko is still barred");
}

#[test]
fn test_go9_pack_round_trips_both_sets_exactly() {
    let black = (1u128 << 80) | (1u128 << 64) | 1;
    let white = (1u128 << 79) | (1u128 << 47) | 2;
    assert_eq!(unpack(pack(black, white)), (black, white));
}

fn rejected(s: &Go9State, code: &str) {
    assert_eq!(Go9::validate(&cfg(), s).unwrap_err().code, code);
}

#[test]
fn test_go9_validate_rejects_each_broken_counter() {
    let base = state_of(&[at(0, 0), at(0, 1), PASS, at(1, 0)]);
    assert_eq!(Go9::validate(&cfg(), &base), Ok(()));
    rejected(
        &Go9State { moves: 163, ..base },
        "more plies or placements than the cap",
    );
    rejected(
        &Go9State {
            placements: 163,
            ..base
        },
        "more plies or placements than the cap",
    );
    rejected(
        &Go9State {
            passes_in_row: 3,
            ..base
        },
        "more than two passes in a row",
    );
    rejected(
        &Go9State {
            moves: base.moves + 2,
            ..base
        },
        "plies that the history and the passes do not give",
    );
    rejected(
        &Go9State {
            passes_in_row: 1,
            ..base
        },
        "plies that the history and the passes do not give",
    );
    rejected(
        &Go9State {
            last_capture: 0,
            ..base
        },
        "a last capture that the history does not give",
    );
    rejected(
        &Go9State {
            placements: 2,
            ..base
        },
        "a non-zero unused history entry",
    );
}

#[test]
fn test_go9_validate_rejects_a_broken_history() {
    let base = state_of(&[at(0, 0), at(0, 1), PASS, at(1, 0)]);
    let mut stray = base;
    stray.history[0][2] |= 1 << 60;
    rejected(&stray, "a history entry off the board");
    let mut floating = base;
    floating.history[1] = pack(1u128 << at(0, 0), 1u128 << at(8, 8) | 1u128 << at(7, 7));
    rejected(&floating, "a history entry that is not one legal placement");
    // Black placed on a white point, with white keeping it: the new stone is
    // single and nothing is captured, so only the empty-point check refuses.
    let mut on_white = state_of(&[at(0, 0), at(0, 1)]);
    let (b, w) = on_white.sets();
    let (b2, w2) = (b | 1u128 << at(0, 1), w);
    on_white.history[2] = pack(b2, w2);
    on_white.placements = 3;
    on_white.moves = 3;
    on_white.stones = [
        [b2 as u64, (b2 >> 64) as u64],
        [w2 as u64, (w2 >> 64) as u64],
    ];
    rejected(&on_white, "a history entry that is not one legal placement");
    let mut stones = base;
    stones.stones = [[1, 0], [0, 0]];
    rejected(&stones, "stones other than the last history entry");
    // A repeat needs a real capture cycle: the ko retake written in by hand.
    let (b, w) = ko_shape(0);
    let mut moves = setup(&b, &w);
    moves.extend([at(1, 2), at(1, 1)]);
    let s = state_of(&moves);
    let mut repeated = s;
    let n = s.placements as usize;
    repeated.history[n] = s.history[n - 2];
    repeated.placements += 1;
    let (black, white) = unpack(s.history[n - 2]);
    repeated.stones = [
        [black as u64, (black >> 64) as u64],
        [white as u64, (white >> 64) as u64],
    ];
    repeated.moves += 1;
    repeated.last_capture = repeated.placements;
    rejected(&repeated, "a position repeated in the history");
}

/// Field perturbations of reachable states: `Validate` accepts only values a
/// replay gives, so what it lets through is the same state or one the
/// perturbation left reachable, and it plays on within the cap.
#[test]
fn test_go9_perturbed_imports_stay_total_when_accepted() {
    let (mut rejected_n, mut accepted) = (0u32, 0u32);
    let mut rng = Rng(0x0e11_0000);
    for seed in 0..8 {
        let (states, _) = random_game(seed, 5);
        for base in states {
            let mut s = base;
            match rng.next() % 6 {
                0 => s.stones[0][0] ^= 1u64 << (rng.next() % 64),
                1 => s.stones[1][1] ^= 1u64 << (rng.next() % 64),
                2 => s.moves = (rng.next() % 200) as u32,
                3 => s.passes_in_row = (rng.next() % 4) as u32,
                4 => s.last_capture = (rng.next() % 170) as u32,
                _ => {
                    let k = (rng.next() % 162) as usize;
                    s.history[k][(rng.next() % 3) as usize] ^= 1u64 << (rng.next() % 64);
                }
            }
            if Go9::validate(&cfg(), &s).is_err() {
                rejected_n += 1;
                continue;
            }
            accepted += 1;
            let mut t = s;
            while !t.is_over() {
                let legal = legal_actions(&t);
                t.play(rng.pick(&legal));
                assert!(t.moves <= MAX_PLIES, "{:?}", t.moves);
                assert_eq!(Go9::validate(&cfg(), &t), Ok(()));
            }
        }
    }
    assert!(rejected_n > 500, "only {rejected_n} rejected");
    assert!(accepted > 0, "nothing accepted");
}

#[test]
fn test_go9_import_rejects_a_finished_game_as_running_and_a_dead_player() {
    let mut p = pool();
    play(&mut p, at(4, 4));
    let finished = state_of(&[PASS, PASS]);
    assert_eq!(Go9::validate(&cfg(), &finished), Ok(()));
    let before = p.state_hash(0).unwrap();
    assert!(invalid(&import(&mut p, &finished)));
    assert_eq!(p.state_hash(0).unwrap(), before);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(invalid(&p.import(&[0], &[bad])));
}

#[test]
fn test_go9_terminal_record_imports_as_pending_reset() {
    let mut p = pool();
    play(&mut p, PASS);
    play(&mut p, PASS);
    assert!(p.header(0).unwrap().engine.pending_reset.get());
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut q = pool();
    assert!(q.import(&[0], &[rec]).is_ok());
    assert_eq!(q.state(0), p.state(0));
}

#[test]
fn test_go9_never_reset_record_round_trips_and_refuses_anything_else() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(11));
    let p = EnvPool::<Go9, Masked>::with_masks(cfg(), options).unwrap();
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    assert_eq!(*rec.state(), Go9State::default());
    let mut q = EnvPool::<Go9, Masked>::with_masks(cfg(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    assert_eq!(*q.state(0).unwrap(), Go9State::default());
    let mut r = EnvPool::<Go9, Masked>::with_masks(cfg(), options).unwrap();
    let header = *DecodedRecord::from(p.snapshot(0).unwrap()).header();
    let bad = DecodedRecord::from_parts(header, state_of(&[at(0, 0)])).unwrap();
    assert!(invalid(&r.import(&[0], &[bad])));
}

fn baseline(state: &Go9State) -> u32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(u16::try_from(state.to_move()).unwrap(), 2).unwrap();
    Go9::act(&cfg(), state, slot, &mut rng, &mut writer);
    u32::try_from(discrete[0]).unwrap()
}

#[test]
fn test_go9_baseline_takes_the_centre_captures_rescues_and_keeps_eyes() {
    assert_eq!(baseline(&Go9State::default()), at(4, 4));
    // White in the corner has one liberty at (1,0): black takes it.
    let capture = state_of(&[at(0, 1), at(0, 0)]);
    assert_eq!(baseline(&capture), at(1, 0));
    // Black in the corner has one liberty at (1,0): black extends.
    let rescue = state_of(&[at(0, 0), at(0, 1), PASS, PASS - 1]);
    assert_eq!(baseline(&rescue), at(1, 0));
    for seed in 0..10 {
        let (states, _) = random_game(seed, 5);
        for s in states.iter().filter(|s| !s.is_over()) {
            assert!(s.is_legal(baseline(s)));
        }
    }
}

/// The baseline against a seeded uniform player, by colour, under komi.
#[test]
fn test_go9_baseline_beats_a_uniform_player() {
    let mut wins = 0;
    let games = 200u64;
    for game in 0..games {
        let bot = usize::try_from(game % 2).unwrap();
        let mut rng = Rng(0xba5e_0000 + game);
        let mut s = Go9State::default();
        while !s.is_over() {
            let a = if s.to_move() == bot {
                baseline(&s)
            } else {
                let legal = legal_actions(&s);
                if legal.len() > 1 {
                    rng.pick(&legal[..legal.len() - 1])
                } else {
                    PASS
                }
            };
            s.play(a);
        }
        let margin = s.margin_half_points(15);
        wins += u32::from(if bot == 0 { margin > 0 } else { margin < 0 });
    }
    assert_eq!(wins, 200, "{wins} wins of {games}");
}

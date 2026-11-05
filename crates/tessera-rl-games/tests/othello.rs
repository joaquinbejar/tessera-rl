// SPDX-License-Identifier: MIT OR Apache-2.0

//! Othello through the engine: spaces, turns, observations, rewards, the
//! mask and the no-op, the episode bound and import validation. The rules
//! are checked against an array-based reference written only for these
//! tests, which shares no code with the bitboards, and against the
//! published perft counts.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test helpers on an 8 x 8 board: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, Environment, Heuristic, RecordLayout, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_games::othello::board::square;
use tessera_rl_games::othello::{ACTIONS, MAX_PLIES, PASS, START_DISCS};
use tessera_rl_games::{Othello, OthelloConfig, OthelloState};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const CFG: OthelloConfig = OthelloConfig {};

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

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[usize::try_from(self.next() % items.len() as u64).unwrap()]
    }
}

/// The array-based reference: `board[row][col]` is 0 empty, 1 black, 2
/// white. It walks each line square by square.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Ref {
    board: [[u8; 8]; 8],
    side: u8,
}

const LINES: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

impl Ref {
    fn start() -> Self {
        let mut board = [[0; 8]; 8];
        board[3][3] = 2;
        board[4][4] = 2;
        board[3][4] = 1;
        board[4][3] = 1;
        Self { board, side: 1 }
    }

    /// The squares flipped by `side` placing at `row`, `col`.
    fn flipped(&self, side: u8, row: usize, col: usize) -> Vec<(usize, usize)> {
        let mut all = Vec::new();
        if self.board[row][col] != 0 {
            return all;
        }
        for (dr, dc) in LINES {
            let mut line = Vec::new();
            let (mut r, mut c) = (row as i32 + dr, col as i32 + dc);
            while (0..8).contains(&r) && (0..8).contains(&c) {
                let v = self.board[r as usize][c as usize];
                if v == 3 - side {
                    line.push((r as usize, c as usize));
                } else {
                    if v == side {
                        all.extend(&line);
                    }
                    break;
                }
                r += dr;
                c += dc;
            }
        }
        all
    }

    fn actions_of(&self, side: u8) -> Vec<u32> {
        let mut out = Vec::new();
        for row in 0..8 {
            for col in 0..8 {
                if !self.flipped(side, row, col).is_empty() {
                    out.push(u32::try_from(row * 8 + col).unwrap());
                }
            }
        }
        out
    }

    fn actions(&self) -> Vec<u32> {
        let placements = self.actions_of(self.side);
        if placements.is_empty() {
            vec![PASS]
        } else {
            placements
        }
    }

    fn is_over(&self) -> bool {
        self.actions_of(1).is_empty() && self.actions_of(2).is_empty()
    }

    fn play(&mut self, action: u32) {
        if action != PASS {
            let (row, col) = (action as usize / 8, action as usize % 8);
            for (r, c) in self.flipped(self.side, row, col) {
                self.board[r][c] = self.side;
            }
            self.board[row][col] = self.side;
        }
        self.side = 3 - self.side;
    }

    fn stones(&self) -> [u64; 2] {
        let mut out = [0; 2];
        for row in 0..8 {
            for col in 0..8 {
                let v = self.board[row][col];
                if v != 0 {
                    out[usize::from(v - 1)] |= 1u64 << (row * 8 + col);
                }
            }
        }
        out
    }
}

/// The legal actions of a state, in order, read from `is_legal`.
fn legal_actions(s: &OthelloState) -> Vec<u32> {
    (0..ACTIONS).filter(|&a| s.is_legal(a)).collect()
}

fn perft_state(s: &OthelloState, depth: u32) -> u64 {
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
            let mut t = *r;
            t.play(a);
            perft_ref(&t, depth - 1)
        })
        .sum()
}

/// Perft from the start, depths 0 to 8 (OEIS A124004; a pass is a ply).
const PERFT: [u64; 9] = [1, 4, 12, 56, 244, 1_396, 8_200, 55_092, 390_216];

#[test]
fn test_othello_perft_matches_the_reference_and_the_published_counts() {
    let (s, r) = (OthelloState::start(), Ref::start());
    for (depth, &expected) in PERFT.iter().enumerate() {
        let depth = u32::try_from(depth).unwrap();
        assert_eq!(perft_state(&s, depth), expected, "bitboards at {depth}");
        assert_eq!(perft_ref(&r, depth), expected, "reference at {depth}");
    }
}

/// One random game checked ply by ply against the reference: legal
/// actions, the board and the end. Returns the states and whether each ply
/// was a pass.
fn random_game(seed: u64) -> (Vec<OthelloState>, Vec<bool>) {
    let mut rng = Rng(seed);
    let (mut s, mut r) = (OthelloState::start(), Ref::start());
    let (mut states, mut passes) = (vec![s], Vec::new());
    while !s.is_over() {
        let legal = legal_actions(&s);
        assert_eq!(legal, r.actions(), "seed {seed}, ply {}", s.moves);
        let a = rng.pick(&legal);
        s.play(a);
        r.play(a);
        assert_eq!(s.stones, r.stones(), "seed {seed}, ply {}", s.moves);
        assert_eq!(s.is_over(), r.is_over(), "seed {seed}, ply {}", s.moves);
        states.push(s);
        passes.push(a == PASS);
    }
    (states, passes)
}

#[test]
fn test_othello_random_games_match_the_reference_and_stay_valid() {
    let mut passes_seen = 0;
    for seed in 0..2_000 {
        let (states, passes) = random_game(seed);
        for s in &states {
            assert_eq!(Othello::validate(&CFG, s), Ok(()), "seed {seed}: {s:?}");
        }
        passes_seen += passes.iter().filter(|&&p| p).count();
    }
    assert!(passes_seen > 100, "only {passes_seen} passes exercised");
}

/// The bound of `MAX_PLIES`: no game opens or ends on a pass, and no two
/// passes follow each other, so the passes fit in the gaps between at most
/// 60 placements.
#[test]
fn test_othello_episode_length_is_bounded_by_placements_and_gaps() {
    let mut longest = 0;
    for seed in 0..2_000 {
        let (states, passes) = random_game(seed);
        assert!(!passes[0], "seed {seed} opened on a pass");
        assert!(!passes[passes.len() - 1], "seed {seed} ended on a pass");
        assert!(
            passes.windows(2).all(|w| !(w[0] && w[1])),
            "seed {seed} passed twice in a row"
        );
        let last = states[states.len() - 1];
        let placements = passes.iter().filter(|&&p| !p).count();
        assert!(placements <= 60);
        assert_eq!(
            placements,
            last.stones.map(u64::count_ones).iter().sum::<u32>() as usize - 4
        );
        assert!(last.moves <= MAX_PLIES);
        longest = longest.max(last.moves);
    }
    assert!(
        longest > 60,
        "no game with passes reached the longer lengths"
    );
    assert_eq!(MAX_PLIES, 60 + 59);
}

/// The first position of the seeded random games whose mover must pass.
fn first_pass_position() -> OthelloState {
    (0..2_000)
        .find_map(|seed| {
            let (states, passes) = random_game(seed);
            passes.iter().position(|&p| p).map(|k| states[k])
        })
        .unwrap()
}

fn pool_with(auto_reset: AutoReset) -> EnvPool<Othello, Masked> {
    let options = PoolOptions {
        auto_reset,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(11))
    };
    let mut p = EnvPool::with_masks(CFG, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool() -> EnvPool<Othello, Masked> {
    pool_with(AutoReset::Disabled)
}

fn play(p: &mut EnvPool<Othello, Masked>, action: u32) {
    let to_move = p.state(0).unwrap().to_move();
    let mut act = [PASS.cast_signed(); 2];
    act[to_move] = action.cast_signed();
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn mask(p: &EnvPool<Othello, Masked>, agent: usize) -> Vec<u32> {
    let n = ACTIONS as usize;
    p.outputs().masks()[agent * n..(agent + 1) * n]
        .iter()
        .enumerate()
        .filter(|(_, f)| f.get())
        .map(|(a, _)| u32::try_from(a).unwrap())
        .collect()
}

fn acting(p: &EnvPool<Othello, Masked>) -> Vec<bool> {
    p.outputs().acting().iter().map(|f| f.get()).collect()
}

/// Imports `state` under the header of the running pool, with the acting
/// flags that `state` implies.
fn import(p: &mut EnvPool<Othello, Masked>, state: OthelloState) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    for (k, row) in header.agents_mut().iter_mut().enumerate() {
        let slot = AgentSlot::new(u16::try_from(k).unwrap(), 2).unwrap();
        row.acting = Bool8::from(Othello::acting(&CFG, &state, slot));
    }
    let rec = DecodedRecord::from_parts(header, state).unwrap();
    p.import(&[0], &[rec])
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

const LAST_PLY: &str = "a last ply that the board cannot explain";
const D3: u32 = 19;
const C3: u32 = 18;
const C4: u32 = 26;
const F5: u32 = 37;
const E6: u32 = 44;

#[test]
fn test_othello_spaces_are_two_u8_planes_and_65_actions() {
    assert_eq!(
        Othello::action_space(&CFG).unwrap(),
        Space::Discrete { n: 65 }
    );
    let expected = Space::Dict(vec![(
        "board".to_owned(),
        Space::Box {
            shape: vec![2, 8, 8],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![1],
            },
        },
    )]);
    assert_eq!(Othello::observation_space(&CFG).unwrap(), expected);
}

#[test]
fn test_othello_start_has_the_published_setup_and_opening_moves() {
    let mut p = pool();
    let s = *p.state(0).unwrap();
    assert_eq!(s.stones[0], square(4, 3) | square(3, 4), "black e4, d5");
    assert_eq!(s.stones[1], square(3, 3) | square(4, 4), "white d4, e5");
    assert_eq!(mask(&p, 0), [D3, C4, F5, E6]);
    assert_eq!(acting(&p), [true, false]);
    play(&mut p, D3);
    assert_eq!(acting(&p), [false, true]);
    assert_eq!(p.state(0).unwrap().moves, 1);
}

#[test]
fn test_othello_placement_flips_every_bracketed_line_in_all_directions() {
    // White rays from c3..: black at the centre of a star, white at its tips.
    let at = square(3, 3);
    let mut me = 0;
    let mut opp = 0;
    let mut expected = 0;
    for (dr, dc) in LINES {
        let (r1, c1) = ((3 + dr) as u32, (3 + dc) as u32);
        let (r2, c2) = ((3 + 2 * dr) as u32, (3 + 2 * dc) as u32);
        opp |= square(c1, r1);
        expected |= square(c1, r1);
        me |= square(c2, r2);
    }
    let mut s = OthelloState {
        stones: [me, opp],
        ..OthelloState::default()
    };
    assert!(s.is_legal(at.trailing_zeros()));
    s.play(at.trailing_zeros());
    assert_eq!(s.stones[0], me | at | expected);
    assert_eq!(s.stones[1], 0);
}

#[test]
fn test_othello_observation_is_from_each_players_view() {
    let p = pool();
    let obs = p.outputs().observation(0).unwrap().to_vec();
    assert_eq!(obs.len(), 2 * 128);
    let at = |agent: usize, channel: usize, sq: usize| obs[agent * 128 + channel * 64 + sq];
    let (d4, e4) = (27, 28);
    assert_eq!(at(0, 0, e4), 1, "black's own disc");
    assert_eq!(at(0, 1, d4), 1, "black sees white");
    assert_eq!(at(1, 0, d4), 1, "white's own disc");
    assert_eq!(at(1, 1, e4), 1, "white sees black");
    assert_eq!(obs.iter().map(|&v| u32::from(v)).sum::<u32>(), 8);
}

/// The published nine-ply wipeout: f5 d6 c5 f4 e3 f6 g5 e6 e7.
const WIPEOUT: [u32; 9] = [37, 43, 34, 29, 20, 45, 38, 44, 52];

fn wipeout_game() -> OthelloState {
    let mut s = OthelloState::start();
    for a in WIPEOUT {
        assert!(s.is_legal(a), "ply {}: {a}", s.moves);
        s.play(a);
    }
    s
}

#[test]
fn test_othello_win_by_wipeout_rewards_and_ends_with_empty_squares() {
    let mut p = pool();
    for a in WIPEOUT {
        assert_eq!(p.outputs().rewards()[..2], [0.0, 0.0], "before {a}");
        play(&mut p, a);
    }
    let end = wipeout_game();
    assert_eq!(*p.state(0).unwrap(), end);
    assert_eq!(end.stones[1], 0, "white holds nothing");
    assert!(end.is_over());
    assert!(p.outputs().episode_terminated()[0].get());
    assert_eq!(p.outputs().rewards()[..2], [1.0, -1.0]);
}

#[test]
fn test_othello_equal_discs_at_the_end_is_a_draw() {
    // Found by a seeded search over random games.
    let mut drawn = None;
    for seed in 0..20_000 {
        let (states, _) = random_game(seed);
        let last = states[states.len() - 1];
        if last.stones[0].count_ones() == last.stones[1].count_ones() {
            drawn = Some(seed);
            break;
        }
    }
    let seed = drawn.unwrap();
    let mut p = pool();
    let mut rng = Rng(seed);
    let mut s = OthelloState::start();
    while !s.is_over() {
        let a = rng.pick(&legal_actions(&s));
        s.play(a);
        play(&mut p, a);
    }
    assert!(p.outputs().episode_terminated()[0].get());
    assert_eq!(p.outputs().rewards()[..2], [0.0, 0.0]);
}

#[test]
fn test_othello_forced_pass_is_the_only_legal_action() {
    let s = first_pass_position();
    assert_eq!(legal_actions(&s), [PASS]);
    assert!(!s.is_over());
    let mut p = pool();
    import(&mut p, s).unwrap();
    let mover = s.to_move();
    assert_eq!(mask(&p, mover), [PASS]);
    let mut t = s;
    t.play(PASS);
    play(&mut p, PASS);
    assert_eq!(*p.state(0).unwrap(), t);
    assert_eq!(t.stones, s.stones);
    assert!(
        !legal_actions(&t).contains(&PASS),
        "a pass hands over a move"
    );
}

/// Sends `action` at the start: under `debug-checks` it is rejected
/// atomically, otherwise the no-op `d3` plays and the action is counted.
fn check_illegal_at_start(action: u32) {
    let mut p = pool();
    let start = *p.state(0).unwrap();
    let result = p.step(ActionInput {
        discrete: &[action.cast_signed(), PASS.cast_signed()],
        continuous: &[],
    });
    if let Err(e) = result {
        assert!(matches!(e, EngineError::IllegalAction { agent: 0, .. }));
        assert_eq!(*p.state(0).unwrap(), start);
    } else {
        let mut expected = start;
        expected.play(D3);
        assert_eq!(*p.state(0).unwrap(), expected);
        assert_eq!(p.header(0).unwrap().agents[0].illegal_actions, 1);
    }
}

#[test]
fn test_othello_pass_with_a_placement_available_is_replaced_or_rejected() {
    check_illegal_at_start(PASS);
}

#[test]
fn test_othello_illegal_square_is_replaced_or_rejected() {
    check_illegal_at_start(0);
}

#[test]
fn test_othello_mask_equals_the_reference_along_random_games() {
    for seed in 0..50 {
        let mut rng = Rng(seed);
        let mut p = pool();
        let mut r = Ref::start();
        while !r.is_over() {
            let mover = usize::from(r.side - 1);
            let actions = r.actions();
            assert_eq!(mask(&p, mover), actions, "seed {seed}");
            assert!(acting(&p)[mover]);
            assert!(!acting(&p)[1 - mover]);
            let a = rng.pick(&actions);
            play(&mut p, a);
            r.play(a);
        }
        assert_eq!(acting(&p), [false, false]);
    }
}

#[test]
fn test_othello_auto_reset_starts_a_fresh_board() {
    let mut p = pool_with(AutoReset::SameStep);
    for a in WIPEOUT {
        play(&mut p, a);
    }
    assert!(p.outputs().episode_terminated()[0].get());
    assert_eq!(*p.state(0).unwrap(), OthelloState::start());
}

#[test]
fn test_othello_snapshot_and_restore_continue_the_game() {
    let mut p = pool();
    play(&mut p, D3);
    let snap = p.snapshot(0).unwrap();
    let fixed = *p.state(0).unwrap();
    play(&mut p, legal_actions(&fixed)[0]);
    let after = *p.state(0).unwrap();
    p.restore(0, &snap).unwrap();
    assert_eq!(*p.state(0).unwrap(), fixed);
    play(&mut p, legal_actions(&fixed)[0]);
    assert_eq!(*p.state(0).unwrap(), after);
}

#[test]
fn test_othello_import_rejects_overlap_missing_centre_and_gaps() {
    let mut p = pool();
    play(&mut p, D3);
    let base = *p.state(0).unwrap();
    let overlap = OthelloState {
        stones: [base.stones[0] | square(3, 3), base.stones[1] | square(3, 3)],
        ..base
    };
    assert!(invalid(&import(&mut p, overlap)));
    let no_centre = OthelloState {
        stones: [
            base.stones[0] & !square(4, 4),
            base.stones[1] & !square(4, 4),
        ],
        ..base
    };
    assert!(invalid(&import(&mut p, no_centre)));
    let gap = OthelloState {
        stones: [base.stones[0] | square(0, 7), base.stones[1]],
        moves: 2,
        ..base
    };
    assert!(invalid(&import(&mut p, gap)));
    assert!(import(&mut p, base).is_ok());
}

#[test]
fn test_othello_validate_rejects_plies_that_do_not_match_the_discs() {
    let mut s = OthelloState::start();
    s.play(D3);
    assert_eq!(Othello::validate(&CFG, &s), Ok(()));
    let few = OthelloState { moves: 0, ..s };
    assert!(Othello::validate(&CFG, &few).is_err(), "fewer plies");
    let many = OthelloState { moves: 3, ..s };
    assert!(Othello::validate(&CFG, &many).is_err(), "more passes");
}

#[test]
fn test_othello_validate_rejects_another_board_at_ply_zero() {
    let start = OthelloState::start();
    let swapped = OthelloState {
        stones: [start.stones[1], start.stones[0]],
        ..start
    };
    assert!(Othello::validate(&CFG, &swapped).is_err());
}

#[test]
fn test_othello_validate_rejects_a_pass_with_a_placement_available() {
    // After d3 the count allows one pass, but white, which would have
    // passed, has placements and too few discs to have placed.
    let mut s = OthelloState::start();
    s.play(D3);
    let passed = OthelloState { moves: 2, ..s };
    assert_eq!(
        Othello::validate(&CFG, &passed).unwrap_err().code,
        "a last ply that the board cannot explain"
    );
}

#[test]
fn test_othello_validate_rejects_more_plies_than_the_bound() {
    let full = OthelloState {
        stones: [0x0000_0000_ffff_ffff, 0xffff_ffff_0000_0000],
        moves: 120,
        ..OthelloState::default()
    };
    assert_eq!(Othello::validate(&CFG, &full).unwrap_err().code, LAST_PLY);
}

#[test]
fn test_othello_validate_rejects_a_last_placement_with_a_pass_per_placement() {
    let mut s = OthelloState::start();
    s.play(D3);
    s.play(C3);
    assert_eq!(Othello::validate(&CFG, &s), Ok(()));
    // Two placements and two passes, white last with c3, d4, e5 in line.
    let padded = OthelloState { moves: 4, ..s };
    assert_eq!(Othello::validate(&CFG, &padded).unwrap_err().code, LAST_PLY);
}

/// Only connectivity rejects this: the corner disc is far from the rest,
/// and its count, with one pass, fits the plies.
#[test]
fn test_othello_validate_rejects_a_disconnected_disc() {
    let mut s = OthelloState::start();
    s.play(D3);
    s.play(C3);
    s.stones[0] |= square(7, 7);
    s.moves = 4;
    assert_eq!(
        Othello::validate(&CFG, &s).unwrap_err().code,
        "discs that are not one connected group"
    );
    s.stones[0] &= !square(7, 7);
    s.moves = 2;
    assert_eq!(Othello::validate(&CFG, &s), Ok(()));
}

/// The last mover holds two adjacent discs but no three in line, and the
/// count leaves no pass, so nothing explains its placement.
#[test]
fn test_othello_validate_rejects_a_placement_without_three_in_line() {
    let start = OthelloState::start();
    let s = OthelloState {
        stones: [start.stones[0], start.stones[1] | square(2, 2)],
        moves: 1,
        ..start
    };
    assert_eq!(Othello::validate(&CFG, &s).unwrap_err().code, LAST_PLY);
}

#[test]
fn test_othello_validate_accepts_the_pass_a_game_really_played() {
    let mut s = first_pass_position();
    s.play(PASS);
    assert_eq!(Othello::validate(&CFG, &s), Ok(()));
}

/// The engine refuses a running record without an actor, whether the
/// header's acting flags are stale or follow the finished board.
#[test]
fn test_othello_import_rejects_a_running_episode_on_a_finished_board() {
    let mut p = pool();
    play(&mut p, WIPEOUT[0]);
    let target = wipeout_game();
    assert_eq!(Othello::validate(&CFG, &target), Ok(()));
    let before = p.state_hash(0).unwrap();
    assert!(invalid(&import(&mut p, target)));
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let stale = DecodedRecord::from_parts(*good.header(), target).unwrap();
    assert!(p.import(&[0], &[stale]).is_err());
    assert!(!p.is_poisoned());
    assert_eq!(p.state_hash(0).unwrap(), before);
}

#[test]
fn test_othello_terminal_record_imports_as_pending_reset() {
    let mut p = pool();
    for a in WIPEOUT {
        play(&mut p, a);
    }
    assert!(p.header(0).unwrap().engine.pending_reset.get());
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut q = pool();
    assert!(q.import(&[0], &[rec]).is_ok());
    assert_eq!(q.state(0), p.state(0));
}

#[test]
fn test_othello_import_rejects_a_dead_player() {
    let mut p = pool();
    play(&mut p, D3);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(invalid(&p.import(&[0], &[bad])));
}

#[test]
fn test_othello_never_reset_record_round_trips_and_continues() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(11));
    let p = EnvPool::<Othello, Masked>::with_masks(CFG, options).unwrap();
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    assert_eq!(*rec.state(), OthelloState::default());
    let mut q = EnvPool::<Othello, Masked>::with_masks(CFG, options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    assert_eq!(*q.state(0).unwrap(), OthelloState::start());
    let bad = DecodedRecord::from_parts(
        *DecodedRecord::from(p.snapshot(0).unwrap()).header(),
        OthelloState::start(),
    )
    .unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
}

/// Bit and field perturbations of reachable states: whatever `Validate`
/// lets through is a total position, whose mask is never empty and whose
/// play ends within the bound for an imported count.
#[test]
fn test_othello_perturbed_imports_stay_total_when_accepted() {
    let (mut rejected, mut accepted) = (0u32, 0u32);
    let mut rng = Rng(0x0e11_0000);
    for seed in 0..200 {
        let (states, _) = random_game(seed);
        for base in states {
            let mut s = base;
            match rng.next() % 3 {
                0 => s.stones[0] ^= 1u64 << (rng.next() % 64),
                1 => s.stones[1] ^= 1u64 << (rng.next() % 64),
                _ => s.moves = u32::try_from(rng.next() % 130).unwrap(),
            }
            if Othello::validate(&CFG, &s).is_err() {
                rejected += 1;
                continue;
            }
            accepted += 1;
            let discs: u32 = s.stones.map(u64::count_ones).iter().sum();
            assert!(discs >= START_DISCS);
            let mut t = s;
            assert!(s.moves <= MAX_PLIES, "{s:?}");
            while !t.is_over() {
                let legal = legal_actions(&t);
                assert!(!legal.is_empty(), "{t:?}");
                t.play(rng.pick(&legal));
                assert!(t.moves <= MAX_PLIES, "{s:?} passes the bound");
            }
        }
    }
    assert!(rejected > 1_000, "only {rejected} rejected");
    assert!(accepted > 100, "only {accepted} accepted");
}

fn baseline(state: &OthelloState) -> u32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let slot = AgentSlot::new(u16::try_from(state.to_move()).unwrap(), 2).unwrap();
    Othello::act(&CFG, state, slot, &mut rng, &mut writer);
    u32::try_from(discrete[0]).unwrap()
}

#[test]
fn test_othello_baseline_is_legal_prefers_corners_and_passes_when_forced() {
    let forced = first_pass_position();
    assert_eq!(baseline(&forced), PASS);
    let mut corners = 0;
    for seed in 0..200 {
        let (states, _) = random_game(seed);
        for s in states.iter().filter(|s| !s.is_over()) {
            let pick = baseline(s);
            assert!(s.is_legal(pick), "{s:?}: {pick}");
            let corner = [0u32, 7, 56, 63].into_iter().find(|&c| s.is_legal(c));
            if corner.is_some() {
                corners += 1;
                assert!([0, 7, 56, 63].contains(&pick), "{s:?}: {pick}");
            }
        }
    }
    assert!(corners > 0);
}

/// The baseline against a seeded uniform player, 500 games per colour.
#[test]
fn test_othello_baseline_beats_a_uniform_player() {
    let mut wins = 0;
    let games = 1_000u64;
    for game in 0..games {
        let bot_side = usize::try_from(game % 2).unwrap();
        let mut rng = Rng(0xba5e_0000 + game);
        let mut s = OthelloState::start();
        while !s.is_over() {
            let a = if s.to_move() == bot_side {
                baseline(&s)
            } else {
                rng.pick(&legal_actions(&s))
            };
            s.play(a);
        }
        let counts = s.stones.map(u64::count_ones);
        wins += u32::from(counts[bot_side] > counts[1 - bot_side]);
    }
    assert_eq!(wins, 866, "{wins} wins of {games}");
}

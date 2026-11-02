// SPDX-License-Identifier: MIT OR Apache-2.0

//! MCTS strength on Connect Four (P2-M2 acceptance): PUCT with uniform
//! priors and a zero value (the search is driven by terminal values
//! only), 400 simulations per move, against uniform random play and
//! against the depth-1 bot, both colours, from the empty board.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::connect4::board::can_play;
use tessera_rl_games::connect4::bot::Bot;
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, Mcts, PuctConfig, SearchOptions, SearchPool,
};

struct Uniform;

impl Evaluator<Connect4> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

/// Uniform priors and the result of one random playout from the leaf
/// (seeded by the leaf's record hash), from the view of the side to move.
struct Rollout;

impl Evaluator<Connect4> for Rollout {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 7.0);
        for (v, r) in values.iter_mut().zip(leaves) {
            let mut s = *r.state();
            let me = s.to_move();
            let mut x = r.hash().get();
            *v = 0.0;
            while !s.is_over() {
                let legal: Vec<u32> = (0..7).filter(|&c| can_play(s.occupied(), c)).collect();
                x = mix(x);
                let side = s.to_move();
                s.play(legal[usize::try_from(x % legal.len() as u64).unwrap()]);
                if tessera_rl_games::connect4::board::has_four(s.stones[side]) {
                    *v = if side == me { 1.0 } else { -1.0 };
                }
            }
        }
        Ok(())
    }
}

/// splitmix64, for the random opponent.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Clone, Copy)]
enum Opponent {
    Random,
    Depth1,
}

/// One game; returns +1 if MCTS wins, 0 for a draw, -1 for a loss.
fn game(opponent: Opponent, mcts_first: bool, seed: u64, sims: u32, rollout: bool) -> i32 {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut env = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    env.reset_all(Reseed::Keep).unwrap();
    let mut cfg = PuctConfig::default();
    cfg.simulations = sims;
    let mut o = SearchOptions::new(NonZeroU32::new(8_192).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Oldest;
    let pool = SearchPool::new(
        Connect4Config {},
        o,
        &env.snapshot(0).unwrap(),
        EnvPool::with_masks,
    )
    .unwrap();
    let mut mcts = Mcts::new(pool, cfg).unwrap();
    let bot = Bot::with_depth(1, seed);
    let mut ply = 0u64;
    loop {
        let s = *env.state(0).unwrap();
        let me = s.to_move();
        let mcts_turn = (me == 0) == mcts_first;
        let col = if mcts_turn {
            mcts.pool_mut().set_root(&env.snapshot(0).unwrap()).unwrap();
            if rollout {
                mcts.run(&mut Rollout).unwrap();
            } else {
                mcts.run(&mut Uniform).unwrap();
            }
            mcts.select_action(0.0).unwrap()
        } else {
            match opponent {
                Opponent::Depth1 => bot.choose(&s).unwrap(),
                Opponent::Random => {
                    let legal: Vec<u32> = (0..7).filter(|&c| can_play(s.occupied(), c)).collect();
                    legal[usize::try_from(mix(seed ^ (ply << 32)) % legal.len() as u64).unwrap()]
                }
            }
        };
        let mut act = [0, 0];
        act[me] = i32::try_from(col).unwrap();
        let out = env
            .step(ActionInput {
                discrete: &act,
                continuous: &[],
            })
            .unwrap();
        if out.episode_terminated()[0].get() {
            let r = out.rewards()[me];
            let mine = if mcts_turn { r } else { -r };
            return if mine > 0.0 {
                1
            } else if mine < 0.0 {
                -1
            } else {
                0
            };
        }
        ply += 1;
    }
}

fn tally(opponent: Opponent, games: u64, sims: u32, rollout: bool) -> (u32, u32, u32) {
    let (mut w, mut d, mut l) = (0, 0, 0);
    for g in 0..games {
        for first in [true, false] {
            match game(opponent, first, 0x5713_0000 + g, sims, rollout) {
                1 => w += 1,
                0 => d += 1,
                _ => l += 1,
            }
        }
    }
    (w, d, l)
}

/// Against random play, uniform priors and a zero value, 400 simulations
/// per move: at least 48 of 2 x 25 games (measured: 50/0/0).
#[test]
fn test_strength_mcts_beats_random_play() {
    let (w, d, l) = tally(Opponent::Random, 25, 400, false);
    assert!(w >= 48, "vs random: {w}/{d}/{l}");
}

/// Against the depth-1 bot, with a random-playout evaluator and 4,000
/// simulations per move: at least 35 of 2 x 25 games (measured: 39/1/10).
/// A zero-value evaluator is tactically sound (see the solver test) but
/// strategically blind, and loses to the bot's window heuristic.
#[test]
#[ignore = "long in debug builds: run with make search-strength (release)"]
fn test_strength_mcts_with_rollouts_beats_the_depth_one_bot() {
    let (w, d, l) = tally(Opponent::Depth1, 25, 4_000, true);
    assert!(w >= 35, "vs depth 1: {w}/{d}/{l}");
}

/// Tactics against the perfect solver: in every position of the
/// calibration set with a forced win within about four plies, the search
/// (uniform priors, zero value, 2,000 simulations) plays a winning move
/// (measured: 49 of 49; at 400 simulations, 47).
#[test]
#[ignore = "long in debug builds: run with make search-strength (release)"]
fn test_strength_mcts_plays_a_winning_move_in_near_win_positions() {
    use tessera_rl_connect4_ref::calibrate::position_set;
    use tessera_rl_connect4_ref::solver::Solver;
    let mut solver = Solver::new();
    let (mut n, mut ok) = (0, 0);
    for p in position_set(400) {
        let scores = solver.column_scores(&p);
        let best = scores.iter().flatten().copied().max().unwrap();
        if best < (43 - i32::try_from(p.moves()).unwrap()) / 2 - 2 {
            continue;
        }
        n += 1;
        let options = PoolOptions {
            auto_reset: AutoReset::Disabled,
            ..PoolOptions::new(NonZeroU32::MIN, Seed::new(1))
        };
        let mut env = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
        env.reset_all(Reseed::Keep).unwrap();
        for c in reconstruct(&p) {
            let me = env.state(0).unwrap().to_move();
            let mut a = [0, 0];
            a[me] = c;
            env.step(ActionInput {
                discrete: &a,
                continuous: &[],
            })
            .unwrap();
        }
        let mut cfg = PuctConfig::default();
        cfg.simulations = 2_000;
        let mut o = SearchOptions::new(
            NonZeroU32::new(16_384).unwrap(),
            NonZeroU32::new(8).unwrap(),
        );
        o.eviction = Eviction::Oldest;
        let pool = SearchPool::new(
            Connect4Config {},
            o,
            &env.snapshot(0).unwrap(),
            EnvPool::with_masks,
        )
        .unwrap();
        let mut m = Mcts::new(pool, cfg).unwrap();
        m.run(&mut Uniform).unwrap();
        let col = m.select_action(0.0).unwrap();
        if scores[col as usize].unwrap() > 0 {
            ok += 1;
        }
    }
    assert_eq!((n, ok), (49, 49));
}

/// A move sequence reaching `p` (DFS over the reference board).
fn reconstruct(p: &tessera_rl_connect4_ref::board::Position) -> Vec<i32> {
    use tessera_rl_connect4_ref::board::Position;
    fn go(cur: Position, target: &Position, seq: &mut Vec<i32>) -> bool {
        if cur.moves() == target.moves() {
            return cur == *target;
        }
        for c in 0..7u32 {
            if !cur.can_play(c) || cur.is_winning_col(c) {
                continue;
            }
            let mut n = cur;
            n.play_col(c);
            // Prune: the new stone must be in the target.
            let mine_before = cur.current();
            let placed = (n.opponent()) & !mine_before;
            let target_side = if cur.moves() % 2 == target.moves() % 2 {
                target.current()
            } else {
                target.opponent()
            };
            if placed & !target_side != 0 {
                continue;
            }
            seq.push(i32::try_from(c).unwrap());
            if go(n, target, seq) {
                return true;
            }
            seq.pop();
        }
        false
    }
    let mut seq = Vec::new();
    assert!(go(Position::new(), p, &mut seq));
    seq
}

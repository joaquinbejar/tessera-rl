// SPDX-License-Identifier: MIT OR Apache-2.0

//! Brick wall through the engine, against a reference written only for these
//! tests in raw `i64` arithmetic: serve timing, walls, bricks and the deep side
//! contact, the paddle rule and its rounding, misses, refills and both terminal
//! states, frameskip, both observations, import validation by phase and the
//! baseline.

#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "test helpers on a small world: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::brickwall::{
    FULL, LEFT, OBS_PIXELS, OBS_STATE, PIXELS, RIGHT, SERVE_VX, SHADE_BANDS, SHADE_OBJECT, STAY,
    fx, vector,
};
use tessera_rl_classic::{BrickWall, BrickWallConfig, BrickWallState};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{AgentSlot, Bool8, EnvConfig, Environment, Heuristic, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodeLimits, DecodedRecord, EngineError, EnvPool, NoMasks,
    PoolOptions, Reseed,
};
use tessera_rl_num::Fx32;
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};

type Pool = EnvPool<BrickWall, NoMasks>;

const ONE: i64 = 1 << 16;

fn config(frameskip: u32, lives: u32, obs: u32) -> BrickWallConfig {
    BrickWallConfig {
        frameskip,
        lives,
        obs,
    }
}

fn pool_with(c: &BrickWallConfig, seed: u64, auto_reset: AutoReset, limit: u32) -> Pool {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::new(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(c: &BrickWallConfig, seed: u64) -> Pool {
    pool_with(c, seed, AutoReset::Disabled, 1_000_000)
}

fn state(p: &Pool) -> BrickWallState {
    *p.state(0).unwrap()
}

fn step(p: &mut Pool, action: i32) -> f32 {
    p.step(ActionInput {
        discrete: &[action],
        continuous: &[],
    })
    .unwrap()
    .rewards()[0]
}

fn terminated(p: &Pool) -> bool {
    p.outputs().unwrap().episode_terminated()[0].get()
}

fn import(p: &mut Pool, s: BrickWallState) -> Result<(), EngineError> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let rec = DecodedRecord::from_parts(*good.header(), s).unwrap();
    p.import(&[0], &[rec])
}

fn invalid(r: &Result<(), EngineError>) -> bool {
    matches!(
        r,
        Err(EngineError::Invariant { index: 0, .. } | EngineError::InvalidRecord { index: 0, .. })
    )
}

fn raw(v: i64) -> Fx32 {
    Fx32::from_raw(i32::try_from(v).unwrap())
}

/// A moving ball at `(x, y)` with speeds, in whole units unless raw is used.
fn moving(x: i32, y: i32, vx: i32, vy: i32) -> BrickWallState {
    let c = config(1, 5, OBS_STATE);
    let mut s = state(&pool(&c, 1));
    s.x = fx(x);
    s.y = fx(y);
    s.vx = fx(vx);
    s.vy = fx(vy);
    s.wait = 0;
    s
}

/// Imports `s` into a frameskip-1 pool.
fn fixture(s: BrickWallState) -> Pool {
    let c = config(1, 5, OBS_STATE);
    let mut p = pool(&c, 1);
    assert_eq!(BrickWall::validate(&c, &s), Ok(()), "{s:?}");
    import(&mut p, s).unwrap();
    p
}

fn clear(s: &mut BrickWallState, row: i32, col: i32) {
    let k = (row * 18 + col) as usize;
    s.bricks[k / 64] &= !(1 << (k % 64));
    s.score += [3, 3, 2, 2, 1, 1][row as usize];
}

fn has(s: &BrickWallState, row: i32, col: i32) -> bool {
    let k = (row * 18 + col) as usize;
    s.bricks[k / 64] >> (k % 64) & 1 == 1
}

#[test]
fn test_brickwall_state_is_48_bytes() {
    assert_eq!(size_of::<BrickWallState>(), 48);
}

#[test]
fn test_brickwall_spaces_follow_obs_and_the_action_is_three() {
    let leaf = |space: Space| match space {
        Space::Dict(mut entries) => entries.remove(0).1,
        other => other,
    };
    assert_eq!(
        leaf(BrickWall::observation_space(&config(4, 5, OBS_PIXELS)).unwrap()),
        Space::Box {
            shape: vec![1, PIXELS, PIXELS],
            dtype: DType::U8,
            bounds: BoxBounds::UInt {
                low: vec![0],
                high: vec![255]
            },
        }
    );
    assert_eq!(
        leaf(BrickWall::observation_space(&config(4, 5, OBS_STATE)).unwrap()),
        Space::Box {
            shape: vec![116],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0]
            },
        }
    );
    assert_eq!(
        BrickWall::action_space(&BrickWallConfig::default()).unwrap(),
        Space::Discrete { n: 3 }
    );
}

#[test]
fn test_brickwall_config_rejects_values_out_of_range() {
    for c in [
        config(0, 5, 0),
        config(9, 5, 0),
        config(4, 0, 0),
        config(4, 10, 0),
        config(4, 5, 2),
    ] {
        assert!(c.validate().is_err(), "{c:?}");
    }
    assert_eq!(BrickWallConfig::default(), config(4, 5, OBS_PIXELS));
}

#[test]
fn test_brickwall_reset_is_the_full_wall_with_a_parked_ball() {
    let s = state(&pool(&config(4, 7, OBS_STATE), 3));
    assert_eq!(s.bricks, FULL);
    assert_eq!((s.px, s.x, s.y), (fx(257), fx(285), fx(200)));
    assert_eq!((s.vx, s.vy), (Fx32::ZERO, Fx32::ZERO));
    assert_eq!((s.wait, s.lives, s.refills, s.score), (30, 7, 0, 0));
}

#[test]
fn test_brickwall_serve_waits_exactly_30_frames_then_draws_once() {
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..64 {
        let mut p = pool(&config(1, 5, OBS_STATE), seed);
        for frame in 1..30 {
            step(&mut p, RIGHT);
            let s = state(&p);
            assert!(s.parked(), "frame {frame}");
            assert_eq!(s.wait, 30 - frame as u8);
        }
        step(&mut p, STAY);
        let s = state(&p);
        assert_eq!((s.wait, s.vy, s.x, s.y), (0, fx(4), fx(285), fx(200)));
        let vx = s.vx.to_raw() >> 16;
        assert!(SERVE_VX.contains(&vx), "{vx}");
        seen.insert(vx);
        step(&mut p, STAY);
        let t = state(&p);
        assert_eq!(
            (t.x, t.y),
            (fx(285 + vx), fx(204)),
            "it moves from frame 31"
        );
        assert_eq!(
            t.px,
            fx(257 + 29 * 8).min(fx(514)),
            "the paddle moved while waiting"
        );
    }
    assert_eq!(seen.len(), 4, "every serve speed is drawn");
}

#[test]
fn test_brickwall_walls_reflect_at_0_570_and_the_top() {
    let mut p = fixture(moving(2, 100, -5, 4));
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.x, s.vx), (fx(3), fx(5)));
    let mut p = fixture(moving(567, 100, 6, 4));
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.x, s.vx), (fx(567), fx(-6)), "1140 - 573");
    let mut p = fixture(moving(0, 100, 0, 4));
    step(&mut p, STAY);
    assert_eq!(state(&p).x, fx(0), "x 0 is in range and does not reflect");
    // The top: a cleared column, so no brick is in the way.
    let mut s = moving(0, 2, 0, -4);
    for row in 0..6 {
        clear(&mut s, row, 0);
    }
    let mut p = fixture(s);
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.y, s.vy), (fx(2), fx(4)));
}

#[test]
fn test_brickwall_brick_touching_is_no_hit_and_overlap_is() {
    // The ball's top edge reaches 112, the bottom of row 5 at y 100..112.
    let mut p = fixture(moving(100, 116, 0, -4));
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.y, s.vy, s.score), (fx(112), fx(-4), 0), "touching only");
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!(
        (s.y, s.vy, s.score),
        (fx(108), fx(4), 1),
        "now inside row 5"
    );
    assert!(!has(&s, 5, 3));
}

#[test]
fn test_brickwall_straddling_two_bricks_clears_the_first_only() {
    // Columns 2 and 3 meet at x 96; the ball spans 93..99.
    let mut p = fixture(moving(93, 113, 0, -4));
    step(&mut p, STAY);
    let s = state(&p);
    assert!(!has(&s, 5, 2) && has(&s, 5, 3));
    assert_eq!((s.score, s.vy), (1, fx(4)));
    // Rows 4 and 5 meet at y 100; a ball spanning both clears row 4 first.
    let mut s = moving(100, 101, 0, -4);
    clear(&mut s, 5, 3);
    let mut p = fixture(s);
    step(&mut p, STAY);
    let s = state(&p);
    assert!(!has(&s, 4, 3) && has(&s, 5, 2));
}

#[test]
fn test_brickwall_deep_side_contact_stays_in_the_band() {
    // Codex's fixture: row 0, column 1, with column 0 cleared.
    let mut s = moving(26, 43, 6, 4);
    for row in 0..6 {
        clear(&mut s, row, 0);
    }
    let mut p = fixture(s);
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.x, s.y), (fx(32), fx(47)));
    assert!(!has(&s, 0, 1) && s.vy == fx(-4));
    step(&mut p, STAY);
    let s = state(&p);
    assert_eq!((s.x, s.y), (fx(38), fx(43)), "still inside the band");
}

/// `vx + floor(d * 5782 / 65536)`, clamped, in raw units.
fn bounce(vx: i64, d: i64) -> i64 {
    (vx + (d * 5782).div_euclid(ONE)).clamp(-6 * ONE, 6 * ONE)
}

#[test]
fn test_brickwall_paddle_bounce_crossing_rounding_and_clamp() {
    // Ball bottom at 308, then 312: it crosses the paddle top at 310.
    for (px, x_raw, vx) in [
        (257, (257 + 31 - 3 + 10) * ONE, 3 * ONE),
        (257, (257 + 31 - 3) * ONE - 6 * ONE - ONE / 2, -2 * ONE),
        (100, 100 * ONE - 5 * ONE, -6 * ONE),
        (100, 100 * ONE + 61 * ONE, 6 * ONE),
    ] {
        let mut s = moving(0, 302, 0, 4);
        s.px = fx(px);
        s.x = raw(x_raw);
        s.vx = raw(vx);
        // The step moves x first, so place it one step back.
        s.x = raw(x_raw - vx);
        let mut p = fixture(s);
        step(&mut p, STAY);
        let t = state(&p);
        let d = x_raw + 3 * ONE - (i64::from(px) + 31) * ONE;
        assert_eq!((t.y, t.vy), (fx(304), fx(-4)), "{px} {x_raw}");
        assert_eq!(i64::from(t.vx.to_raw()), bounce(vx, d), "{px} {x_raw}");
    }
    // Just outside in x: the ball's right edge at the paddle's left edge.
    let mut s = moving(251 - 2, 302, 2, 4);
    s.px = fx(257);
    let mut p = fixture(s);
    step(&mut p, STAY);
    assert_eq!(state(&p).vy, fx(4), "x 251..257 does not meet 257..319");
    // Already below the top: no bounce.
    let mut s = moving(280, 306, 0, 4);
    s.px = fx(257);
    let mut p = fixture(s);
    step(&mut p, STAY);
    assert_eq!(
        state(&p).vy,
        fx(4),
        "308 + 2 is not a crossing from at most 310"
    );
}

#[test]
fn test_brickwall_miss_costs_a_ball_and_the_last_terminates() {
    let mut s = moving(10, 328, 0, 4);
    s.px = fx(400);
    s.lives = 2;
    let c = config(1, 5, OBS_STATE);
    let mut p = pool(&c, 1);
    import(&mut p, s).unwrap();
    step(&mut p, STAY);
    let t = state(&p);
    assert_eq!((t.lives, t.wait, t.parked()), (1, 30, true));
    assert!(!terminated(&p));
    let mut s = t;
    s.wait = 0;
    s.x = fx(10);
    s.y = fx(328);
    s.vy = fx(4);
    import(&mut p, s).unwrap();
    assert_eq!(step(&mut p, STAY), 0.0, "no penalty");
    let t = state(&p);
    assert!(terminated(&p));
    assert_eq!((t.lives, t.wait, t.parked()), (0, 0, true));
    assert_eq!(BrickWall::validate(&c, &t), Ok(()));
}

#[test]
fn test_brickwall_first_empty_wall_refills_and_the_second_terminates() {
    let mut s = moving(0, 114, 0, -4);
    s.bricks = [0, 0];
    s.score = 0;
    // Only brick (5, 0) left, worth 1.
    s.bricks[(5 * 18) / 64] |= 1 << ((5 * 18) % 64);
    s.score = 215;
    let mut p = fixture(s);
    step(&mut p, STAY);
    let t = state(&p);
    assert_eq!((t.bricks, t.refills, t.score), (FULL, 1, 216));
    assert_eq!((t.parked(), t.wait), (true, 30));
    assert!(!terminated(&p));
    let mut s = t;
    s.bricks = [0, 0];
    s.bricks[(5 * 18) / 64] |= 1 << ((5 * 18) % 64);
    s.score = 216 + 215;
    s.x = fx(0);
    s.y = fx(114);
    s.vy = fx(-4);
    s.wait = 0;
    import(&mut p, s).unwrap();
    assert_eq!(step(&mut p, STAY), 1.0);
    let t = state(&p);
    assert!(terminated(&p));
    assert_eq!((t.bricks, t.refills, t.score), ([0, 0], 1, 432));
    assert_eq!((t.parked(), t.wait), (true, 0));
}

/// The reference: raw `i64` coordinates, the rules written out once more.
#[derive(Clone, Debug, PartialEq)]
struct Ref {
    bricks: [[bool; 18]; 6],
    px: i64,
    x: i64,
    y: i64,
    vx: i64,
    vy: i64,
    wait: i64,
    lives: i64,
    refills: i64,
    score: i64,
}

impl Ref {
    fn of(s: &BrickWallState) -> Self {
        let mut bricks = [[false; 18]; 6];
        for (r, row) in bricks.iter_mut().enumerate() {
            for (c, b) in row.iter_mut().enumerate() {
                *b = has(s, r as i32, c as i32);
            }
        }
        Self {
            bricks,
            px: i64::from(s.px.to_raw()),
            x: i64::from(s.x.to_raw()),
            y: i64::from(s.y.to_raw()),
            vx: i64::from(s.vx.to_raw()),
            vy: i64::from(s.vy.to_raw()),
            wait: i64::from(s.wait),
            lives: i64::from(s.lives),
            refills: i64::from(s.refills),
            score: i64::from(s.score),
        }
    }

    fn park(&mut self, wait: i64) {
        (self.x, self.y, self.vx, self.vy, self.wait) = (285 * ONE, 200 * ONE, 0, 0, wait);
    }

    /// One frame. `serve` is the speed the engine drew, if this frame serves.
    /// Returns points and whether the frame is terminal.
    fn frame(&mut self, action: i32, serve: i64) -> (i64, bool) {
        let shift = [-8, 0, 8][action as usize] * ONE;
        self.px = (self.px + shift).clamp(0, 514 * ONE);
        if self.vy == 0 {
            if self.wait > 0 {
                self.wait -= 1;
                if self.wait == 0 {
                    (self.vx, self.vy) = (serve, 4 * ONE);
                }
            }
            return (0, false);
        }
        self.x += self.vx;
        if self.x < 0 {
            (self.x, self.vx) = (-self.x, -self.vx);
        }
        if self.x > 570 * ONE {
            (self.x, self.vx) = (1140 * ONE - self.x, -self.vx);
        }
        let old = self.y;
        self.y += self.vy;
        if self.y < 0 {
            (self.y, self.vy) = (-self.y, 4 * ONE);
        }
        let mut gained = 0;
        let hit = (0..6).find_map(|r| {
            (0..18)
                .find(|&c| {
                    let (bx, by) = (i64::from(32 * c) * ONE, i64::from(40 + 12 * r) * ONE);
                    self.bricks[r as usize][c as usize]
                        && self.x < bx + 32 * ONE
                        && bx < self.x + 6 * ONE
                        && self.y < by + 12 * ONE
                        && by < self.y + 6 * ONE
                })
                .map(|c| (r, c))
        });
        if let Some((r, c)) = hit {
            self.bricks[r as usize][c as usize] = false;
            gained = [3, 3, 2, 2, 1, 1][r as usize];
            self.score += gained;
            self.vy = -self.vy;
            if self.bricks.iter().flatten().all(|b| !b) {
                if self.refills == 0 {
                    self.refills = 1;
                    self.bricks = [[true; 18]; 6];
                    self.park(30);
                    return (gained, false);
                }
                self.park(0);
                return (gained, true);
            }
        }
        let crossing = old + 6 * ONE <= 310 * ONE && 310 * ONE < self.y + 6 * ONE;
        let over = self.x < self.px + 62 * ONE && self.px < self.x + 6 * ONE;
        if self.vy == 4 * ONE && crossing && over {
            self.y = 304 * ONE;
            self.vy = -4 * ONE;
            let d = self.x + 3 * ONE - (self.px + 31 * ONE);
            self.vx = bounce(self.vx, d);
        }
        if self.y > 330 * ONE {
            self.lives -= 1;
            if self.lives == 0 {
                self.park(0);
                return (gained, true);
            }
            self.park(30);
        }
        (gained, false)
    }
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(0x5851_f42d_4c95_7f2d)
            .wrapping_add(0x1405_7b7e_f767_814f);
        (self.0 >> 33) % n
    }
}

/// Runs one decision of `frames` frames on the reference; a serve inside it
/// takes whichever drawn speed makes the reference agree with `engine`.
fn decide(r: &mut Ref, action: i32, frames: u32, engine: &BrickWallState) -> (i64, bool) {
    let run = |r: &mut Ref, serve: i64| {
        let mut total = 0;
        for _ in 0..frames {
            let (p, end) = r.frame(action, serve);
            total += p;
            if end {
                return (total, true);
            }
        }
        (total, false)
    };
    let want = Ref::of(engine);
    for serve in SERVE_VX.map(|v| i64::from(v) * ONE) {
        let mut trial = r.clone();
        let out = run(&mut trial, serve);
        if trial == want {
            *r = trial;
            return out;
        }
    }
    run(r, 0)
}

#[test]
fn test_brickwall_frameskip_sums_frames_and_a_terminal_ends_the_decision() {
    let mut ends = 0;
    for (seed, frames) in [(1u64, 4u32), (2, 8), (3, 3), (4, 2)] {
        let c = config(frames, 1, OBS_STATE);
        let mut p = pool(&c, seed);
        let mut r = Ref::of(&state(&p));
        let mut rng = Lcg(seed);
        for _ in 0..20_000 {
            let s = state(&p);
            let action = if rng.below(3) == 0 {
                rng.below(3) as i32
            } else {
                baseline(&c, &s)
            };
            let got = step(&mut p, action);
            let (want, end) = decide(&mut r, action, frames, &state(&p));
            assert_eq!(Ref::of(&state(&p)), r, "seed {seed}");
            assert_eq!(got, want as f32);
            assert_eq!(terminated(&p), end);
            if end {
                ends += 1;
                break;
            }
        }
    }
    assert_eq!(
        ends, 4,
        "each run reaches its terminal frame inside a decision"
    );
}

fn baseline(c: &BrickWallConfig, s: &BrickWallState) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    BrickWall::act(c, s, AgentSlot::new(0, 1).unwrap(), &mut rng, &mut writer);
    discrete[0]
}

#[test]
fn test_brickwall_play_matches_the_reference_frame_by_frame() {
    let c = config(1, 3, OBS_STATE);
    let (mut bricks, mut bounces, mut misses, mut refills, mut ends) = (0, 0, 0, 0, 0);
    for seed in 0..40 {
        let mut p = pool(&c, seed);
        let mut r = Ref::of(&state(&p));
        let mut rng = Lcg(seed + 99);
        for _ in 0..40_000 {
            let s = state(&p);
            let action = if rng.below(5) == 0 {
                rng.below(3) as i32
            } else {
                baseline(&c, &s)
            };
            let got = step(&mut p, action);
            let t = state(&p);
            let serve = if s.parked() && s.wait == 1 {
                i64::from(t.vx.to_raw())
            } else {
                0
            };
            let before = r.clone();
            let (want, end) = r.frame(action, serve);
            assert_eq!(Ref::of(&t), r, "seed {seed}");
            assert_eq!(got, want as f32, "seed {seed}");
            assert_eq!(terminated(&p), end, "seed {seed}");
            assert_eq!(BrickWall::validate(&c, &t), Ok(()), "seed {seed}");
            bricks += i32::from(want > 0);
            bounces += i32::from(before.vy > 0 && r.vy < 0 && r.y == 304 * ONE);
            misses += i32::from(r.lives < before.lives);
            refills += i32::from(r.refills > before.refills);
            if end {
                ends += 1;
                break;
            }
        }
    }
    assert!(
        bricks > 1_000 && bounces > 500 && misses > 20 && ends > 10,
        "{bricks} {bounces} {misses} {refills} {ends}"
    );
}

#[test]
fn test_brickwall_extreme_play_stays_in_range_and_never_saturates() {
    // Saturation would leave a committed range or break the reference match;
    // validate checks the ranges after every frame, for every action pattern.
    let c = config(1, 9, OBS_STATE);
    for (seed, pattern) in [(1u64, 0), (2, 1), (3, 2), (4, 3)] {
        let mut p = pool(&c, seed);
        let mut rng = Lcg(seed);
        for t in 0..30_000u64 {
            let action = match pattern {
                0 => LEFT,
                1 => RIGHT,
                2 => [LEFT, RIGHT][(t / 40 % 2) as usize],
                _ => rng.below(3) as i32,
            };
            step(&mut p, action);
            let s = state(&p);
            assert_eq!(BrickWall::validate(&c, &s), Ok(()), "{pattern} {t}");
            if terminated(&p) {
                break;
            }
        }
    }
}

/// A rasterizer of its own: each pixel asks which objects' boxes reach it.
fn raster(s: &BrickWallState) -> Vec<u8> {
    let cells = |v0: i64, v1: i64, world: i64| {
        let lo = v0.div_euclid(ONE);
        let hi = (v1 + ONE - 1).div_euclid(ONE);
        let a = (lo * 84).div_euclid(world).clamp(0, 84);
        let b = (hi * 84 + world - 1).div_euclid(world).clamp(0, 84);
        a..b.max(a)
    };
    let mut out = vec![0u8; 84 * 84];
    let mut boxes = Vec::new();
    for r in 0..6i64 {
        for c in 0..18i64 {
            if has(s, r as i32, c as i32) {
                let (x, y) = (32 * c * ONE, (40 + 12 * r) * ONE);
                boxes.push((
                    x,
                    x + 32 * ONE,
                    y,
                    y + 12 * ONE,
                    SHADE_BANDS[(r / 2) as usize],
                ));
            }
        }
    }
    let px = i64::from(s.px.to_raw());
    boxes.push((px, px + 62 * ONE, 310 * ONE, 318 * ONE, SHADE_OBJECT));
    if !s.parked() {
        let (x, y) = (i64::from(s.x.to_raw()), i64::from(s.y.to_raw()));
        boxes.push((x, x + 6 * ONE, y, y + 6 * ONE, SHADE_OBJECT));
    }
    for (x0, x1, y0, y1, shade) in boxes {
        for row in cells(y0, y1, 330) {
            for col in cells(x0, x1, 576) {
                out[(row * 84 + col) as usize] = shade;
            }
        }
    }
    out
}

fn pixels(p: &Pool) -> Vec<u8> {
    p.outputs().unwrap().observation(0).unwrap().to_vec()
}

#[test]
fn test_brickwall_pixels_match_the_reference_rasterizer() {
    let c = config(1, 3, OBS_PIXELS);
    let mut rng = Lcg(5);
    for seed in 0..6 {
        let mut p = pool(&c, seed);
        for _ in 0..3_000 {
            assert_eq!(pixels(&p), raster(&state(&p)), "seed {seed}");
            let s = state(&p);
            let a = if rng.below(4) == 0 {
                rng.below(3) as i32
            } else {
                baseline(&c, &s)
            };
            step(&mut p, a);
            if terminated(&p) {
                break;
            }
        }
    }
}

#[test]
fn test_brickwall_pixels_clip_a_ball_at_the_bottom_and_cover_a_fractional_one() {
    let c = config(1, 5, OBS_PIXELS);
    for (x, y) in [
        (10 * ONE, 327 * ONE),
        (10 * ONE + 3, 100 * ONE + ONE / 3),
        (0, 0),
    ] {
        let mut s = moving(0, 0, 0, 4);
        s.x = raw(x);
        s.y = raw(y);
        for row in 0..6 {
            clear(&mut s, row, 0);
        }
        let mut p = pool(&c, 1);
        import(&mut p, s).unwrap();
        let got = pixels(&p);
        assert_eq!(got, raster(&s));
        let lit = got.contains(&SHADE_OBJECT);
        assert!(lit, "the ball and paddle cover pixels");
    }
}

#[test]
fn test_brickwall_state_vector_follows_the_recipe_at_extremes() {
    let c = config(1, 5, OBS_STATE);
    let mut s = moving(570, 330, -6, -4);
    s.px = fx(514);
    s.lives = 5;
    let v = vector(&c, &s);
    assert_eq!(&v[..8], &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let mut s = moving(0, 0, 6, 4);
    s.px = fx(0);
    s.x = raw(285 * ONE / 2);
    s.lives = 1;
    let v = vector(&c, &s);
    let expect = |num: f64, den: f64| (num / den) as f32;
    assert_eq!(v[0], 0.0);
    assert_eq!(v[1], expect(142.5, 570.0));
    assert_eq!((v[3], v[4]), (1.0, 1.0));
    assert_eq!(v[5], expect(1.0, 5.0));
    assert!(v[8..].iter().all(|&f| f == 1.0));
    let parked = state(&pool(&c, 1));
    let v = vector(&c, &parked);
    assert_eq!((v[3], v[4], v[6]), (0.5, 0.5, 1.0));
    let p = pool(&c, 1);
    let leaf: Vec<f32> = p
        .outputs()
        .unwrap()
        .observation(0)
        .unwrap()
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(leaf, v.to_vec());
}

#[test]
fn test_brickwall_validate_names_each_rejection() {
    let c = config(1, 5, OBS_STATE);
    let good = moving(100, 150, 2, -4);
    assert_eq!(BrickWall::validate(&c, &good), Ok(()));
    let rejected = |s: &BrickWallState, code: &str| {
        assert_eq!(BrickWall::validate(&c, s).unwrap_err().code, code);
    };
    let edit = |f: &dyn Fn(&mut BrickWallState)| {
        let mut s = good;
        f(&mut s);
        s
    };
    rejected(&edit(&|s| s.px = fx(515)), "a paddle out of range");
    rejected(&edit(&|s| s.px = raw(-1)), "a paddle out of range");
    rejected(&edit(&|s| s.x = raw(570 * ONE + 1)), "a ball out of range");
    rejected(&edit(&|s| s.y = fx(331)), "a ball out of range");
    rejected(
        &edit(&|s| s.vx = raw(6 * ONE + 1)),
        "a horizontal speed out of range",
    );
    rejected(
        &edit(&|s| s.vy = fx(3)),
        "a vertical speed other than -4, 0 or 4",
    );
    rejected(&edit(&|s| s.wait = 31), "a wait out of range");
    rejected(&edit(&|s| s.lives = 6), "lives or refills out of range");
    rejected(&edit(&|s| s.refills = 2), "lives or refills out of range");
    rejected(
        &edit(&|s| s.bricks[1] |= 1 << 50),
        "a brick bit beyond the wall",
    );
    rejected(
        &edit(&|s| s.vy = Fx32::ZERO),
        "a parked ball off the serve position or moving",
    );
    rejected(&edit(&|s| s.wait = 3), "a wait while the ball moves");
    // Codex: an empty wall with a moving ball, or with refills 0.
    rejected(
        &edit(&|s| {
            s.bricks = [0, 0];
            s.refills = 1;
            s.score = 432;
        }),
        "an empty wall outside the final terminal state",
    );
    let mut parked = state(&pool(&c, 1));
    parked.wait = 0;
    parked.bricks = [0, 0];
    parked.score = 216;
    rejected(&parked, "an empty wall outside the final terminal state");
    rejected(&edit(&|s| s.lives = 0), "no balls outside a terminal state");
    let mut idle = state(&pool(&c, 1));
    idle.wait = 0;
    rejected(&idle, "a parked ball with no wait in play");
    rejected(
        &edit(&|s| s.score = 1),
        "a score that disagrees with the cleared bricks",
    );
}

fn header_with(p: &Pool, alive: bool, acting: bool) -> DecodedRecord<BrickWall> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[0].alive = Bool8::new(alive);
    header.agents[0].acting = Bool8::new(acting);
    DecodedRecord::from_parts(header, *good.state()).unwrap()
}

#[test]
fn test_brickwall_import_by_phase_direct_and_persisted() {
    let c = config(1, 1, OBS_STATE);
    // Never reset: the default state only.
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let fresh = Pool::new(c.clone(), options).unwrap();
    let rec = DecodedRecord::from(fresh.snapshot(0).unwrap());
    let mut q = Pool::new(c.clone(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let bad = DecodedRecord::from_parts(*rec.header(), moving(1, 1, 0, 4)).unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
    // Active: a terminal state is refused, atomically.
    let mut p = pool(&c, 1);
    let before = p.state_hash(0).unwrap();
    let mut lost = state(&p);
    lost.wait = 0;
    lost.lives = 0;
    assert_eq!(BrickWall::validate(&c, &lost), Ok(()));
    match import(&mut p, lost) {
        Err(EngineError::Invariant { violation, .. }) => {
            assert_eq!(violation.code, "a running episode in a terminal state");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(p.state_hash(0).unwrap(), before);
    // The agent must be alive and acting after a reset.
    assert!(invalid(&p.import(&[0], &[header_with(&p, true, false)])));
    // Pending reset: the last-ball terminal state, then a truncation mid-wait,
    // each round-trips directly and through a persisted file.
    let mut p = pool(&c, 1);
    let mut s = moving(10, 328, 0, 4);
    s.px = fx(400);
    s.lives = 1;
    import(&mut p, s).unwrap();
    step(&mut p, STAY);
    assert!(terminated(&p));
    let mut t = pool_with(&c, 1, AutoReset::Disabled, 3);
    for _ in 0..3 {
        step(&mut t, STAY);
    }
    assert!(t.outputs().unwrap().episode_truncated()[0].get() && state(&t).parked());
    for (src, limit) in [(&p, 1_000_000), (&t, 3)] {
        let rec = DecodedRecord::from(src.snapshot(0).unwrap());
        let mut q = pool_with(&c, 7, AutoReset::Disabled, limit);
        q.import(&[0], &[rec]).unwrap();
        assert_eq!(q.state_hash(0).unwrap(), src.state_hash(0).unwrap());
        let bytes = src.encode(&[0]).unwrap();
        let decoded = q.decode(&bytes, DecodeLimits::default()).unwrap();
        let mut r = pool_with(&c, 8, AutoReset::Disabled, limit);
        r.import(&[0], &decoded).unwrap();
        assert_eq!(r.state_hash(0).unwrap(), src.state_hash(0).unwrap());
    }
}

/// The baseline's points within a bound, summed over seeds; frozen for v1, as
/// it feeds the goldens: a change to `Heuristic::act` moves this pin and needs
/// a new `ENV_VERSION` with new fixtures.
#[test]
fn test_brickwall_baseline_points_are_pinned() {
    let c = config(4, 5, OBS_STATE);
    let mut total = 0.0;
    for seed in 0..50 {
        let mut p = pool(&c, seed);
        for _ in 0..4_000 {
            let s = state(&p);
            total += step(&mut p, baseline(&c, &s));
            if terminated(&p) {
                break;
            }
        }
    }
    assert_eq!(total, 19_415.0);
}

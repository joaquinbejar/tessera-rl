// SPDX-License-Identifier: MIT OR Apache-2.0

//! Paddle duel through the engine, against a reference written only for these
//! tests in raw `i64` arithmetic: serve timing, walls, crossing hits and their
//! spin, goals and the winning point, frameskip and the reward recipe, the
//! unused slot with one agent, the mirrored view in both observations, import
//! validation by phase and the baseline.

#![expect(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "test helpers on a small world: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::paddleduel::{
    DOWN, OBS_PIXELS, OBS_STATE, PIXELS, SERVE_VY, SHADE_OBJECT, SHADE_OTHER, STAY, UP, fx, vector,
    view,
};
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig, PaddleDuelState};
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

type Pool = EnvPool<PaddleDuel, NoMasks>;

const ONE: i64 = 1 << 16;

fn config(agents: u32, frameskip: u32, obs: u32) -> PaddleDuelConfig {
    PaddleDuelConfig {
        agents,
        frameskip,
        max_score: 3,
        bounce: 1,
        obs,
    }
}

fn pool_with(c: &PaddleDuelConfig, seed: u64, limit: u32) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        max_episode_steps: NonZeroU32::new(limit),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
    };
    let mut p = EnvPool::new(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(c: &PaddleDuelConfig, seed: u64) -> Pool {
    pool_with(c, seed, 1_000_000)
}

fn state(p: &Pool) -> PaddleDuelState {
    *p.state(0).unwrap()
}

fn step(p: &mut Pool, acts: [i32; 2]) -> [f32; 2] {
    let out = p
        .step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    [out.rewards()[0], out.rewards()[1]]
}

fn terminated(p: &Pool) -> bool {
    p.outputs().unwrap().episode_terminated()[0].get()
}

fn import(p: &mut Pool, s: PaddleDuelState) -> Result<(), EngineError> {
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

/// A moving ball, paddles at 215.
fn moving(x: i32, y: i32, vx: i32, vy: i32) -> PaddleDuelState {
    let mut s = state(&pool(&config(2, 1, OBS_STATE), 1));
    s.x = fx(x);
    s.y = fx(y);
    s.vx = fx(vx);
    s.vy = fx(vy);
    s.wait = 0;
    s
}

fn fixture(agents: u32, s: PaddleDuelState) -> Pool {
    let c = config(agents, 1, OBS_STATE);
    let mut p = pool(&c, 1);
    assert_eq!(PaddleDuel::validate(&c, &s), Ok(()), "{s:?}");
    import(&mut p, s).unwrap();
    p
}

#[test]
fn test_paddleduel_state_is_32_bytes() {
    assert_eq!(size_of::<PaddleDuelState>(), 32);
}

#[test]
fn test_paddleduel_spaces_config_and_reset() {
    let leaf = |space: Space| match space {
        Space::Dict(mut entries) => entries.remove(0).1,
        other => other,
    };
    assert_eq!(
        leaf(PaddleDuel::observation_space(&config(1, 4, OBS_PIXELS)).unwrap()),
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
        leaf(PaddleDuel::observation_space(&config(2, 4, OBS_STATE)).unwrap()),
        Space::Box {
            shape: vec![9],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0]
            },
        }
    );
    assert_eq!(
        PaddleDuel::action_space(&PaddleDuelConfig::default()).unwrap(),
        Space::Discrete { n: 3 }
    );
    assert_eq!((UP, STAY, DOWN), (0, 1, 2));
    for bad in [
        PaddleDuelConfig {
            agents: 0,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            agents: 3,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            frameskip: 9,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            max_score: 22,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            max_score: 0,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            bounce: 11,
            ..PaddleDuelConfig::default()
        },
        PaddleDuelConfig {
            obs: 2,
            ..PaddleDuelConfig::default()
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    let d = PaddleDuelConfig::default();
    assert_eq!(
        (d.agents, d.frameskip, d.max_score, d.bounce, d.obs),
        (1, 4, 11, 1, 0)
    );
    let s = state(&pool(&config(1, 1, OBS_STATE), 3));
    assert_eq!((s.ly, s.ry, s.x, s.y), (fx(215), fx(215), fx(304), fx(234)));
    assert_eq!(
        (s.vx, s.vy, s.wait, s.score),
        (Fx32::ZERO, Fx32::ZERO, 30, [0, 0])
    );
}

#[test]
fn test_paddleduel_serve_waits_30_frames_and_draws_once() {
    let mut seen = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut p = pool(&config(2, 1, OBS_STATE), seed);
        for frame in 1..30u8 {
            step(&mut p, [DOWN, UP]);
            let s = state(&p);
            assert!(s.parked() && s.wait == 30 - frame);
        }
        step(&mut p, [STAY, STAY]);
        let s = state(&p);
        assert_eq!((s.wait, s.x, s.y), (0, fx(304), fx(234)));
        let (vx, vy) = (s.vx.to_raw() >> 16, s.vy.to_raw() >> 16);
        assert!([6, -6].contains(&vx) && SERVE_VY.contains(&vy));
        seen.insert((vx, vy));
        step(&mut p, [STAY, STAY]);
        let t = state(&p);
        assert_eq!(
            (t.x, t.y),
            (fx(304 + vx), fx(234 + vy)),
            "moves from frame 31"
        );
        assert_eq!((t.ly, t.ry), (fx(430), fx(0)), "29 frames of 8, clamped");
    }
    assert_eq!(seen.len(), 8, "every serve is drawn");
}

#[test]
fn test_paddleduel_walls_reflect_and_a_paddle_hit_in_the_same_frame() {
    let mut p = fixture(2, moving(300, 3, 6, -5));
    step(&mut p, [STAY, STAY]);
    let s = state(&p);
    assert_eq!((s.y, s.vy), (fx(2), fx(5)));
    let mut p = fixture(2, moving(300, 466, 6, 5));
    step(&mut p, [STAY, STAY]);
    let s = state(&p);
    assert_eq!((s.y, s.vy), (fx(465), fx(-5)), "936 - 471");
    // Top reflection, then the left paddle at 0 hits the ball.
    let mut s = moving(22, 3, -6, -5);
    s.ly = fx(0);
    let mut p = fixture(2, s);
    step(&mut p, [STAY, STAY]);
    let s = state(&p);
    assert_eq!((s.x, s.vx, s.y, s.vy), (fx(20), fx(6), fx(2), fx(5)));
}

#[test]
fn test_paddleduel_hits_need_a_crossing_and_an_overlap() {
    // Left: from 22 to 16 crosses 20; paddle 215..285, ball 250..282.
    let mut p = fixture(2, moving(22, 250, -6, 0));
    assert_eq!(step(&mut p, [STAY, STAY]), [0.1, 0.0]);
    assert_eq!((state(&p).x, state(&p).vx), (fx(20), fx(6)));
    // Touching: the ball's bottom edge at the paddle's top, 183 + 32 = 215.
    let mut p = fixture(2, moving(22, 183, -6, 0));
    step(&mut p, [STAY, STAY]);
    assert_eq!(state(&p).vx, fx(-6), "touching is no overlap");
    // One pixel of overlap hits.
    let mut p = fixture(2, moving(22, 184, -6, 0));
    step(&mut p, [STAY, STAY]);
    assert_eq!(state(&p).vx, fx(6));
    // Already past the face: no hit even with the paddle on it.
    let mut p = fixture(2, moving(18, 250, -6, 0));
    step(&mut p, [STAY, STAY]);
    assert_eq!(
        (state(&p).x, state(&p).vx),
        (fx(12), fx(-6)),
        "it goes through"
    );
    // Right, mirrored: from 586 to 592, 618 <= 620 < 624.
    let mut p = fixture(2, moving(586, 250, 6, 0));
    assert_eq!(step(&mut p, [STAY, STAY]), [0.0, 0.1]);
    assert_eq!((state(&p).x, state(&p).vx), (fx(588), fx(-6)));
    let mut p = fixture(2, moving(589, 250, 6, 0));
    step(&mut p, [STAY, STAY]);
    assert_eq!(state(&p).vx, fx(6), "past the right face");
}

#[test]
fn test_paddleduel_spin_uses_the_applied_move_after_the_clamp() {
    let hit = |ly: i32, action: i32, vy: i32| {
        let mut s = moving(22, ly + 20, -6, vy);
        s.ly = fx(ly);
        let mut p = fixture(2, s);
        step(&mut p, [action, STAY]);
        state(&p).vy.to_raw() >> 16
    };
    assert_eq!(hit(100, DOWN, 0), 2);
    assert_eq!(hit(100, UP, 0), -2);
    assert_eq!(hit(100, STAY, 3), 3);
    assert_eq!(hit(0, UP, 0), 0, "fully clamped: no move, no spin");
    assert_eq!(hit(4, UP, 0), -2, "a partial move still spins");
    assert_eq!(hit(430, DOWN, 1), 1, "clamped at the bottom");
    assert_eq!(hit(100, DOWN, 7), 8, "vy clamps at 8");
    assert_eq!(hit(100, UP, -7), -8);
}

#[test]
fn test_paddleduel_goals_score_park_and_the_last_point_terminates() {
    let mut s = moving(2, 100, -6, 0);
    s.ly = fx(400);
    let mut p = fixture(2, s);
    assert_eq!(step(&mut p, [STAY, STAY]), [-1.0, 1.0]);
    let t = state(&p);
    assert_eq!((t.score, t.wait, t.parked()), ([0, 1], 30, true));
    assert!(!terminated(&p));
    let mut s = moving(606, 100, 6, 0);
    s.ry = fx(400);
    s.score = [2, 1];
    let mut p = fixture(2, s);
    assert_eq!(step(&mut p, [STAY, STAY]), [1.0, -1.0]);
    let t = state(&p);
    assert!(terminated(&p));
    assert_eq!((t.score, t.wait, t.parked()), ([3, 1], 0, true));
    assert_eq!(PaddleDuel::validate(&config(2, 1, OBS_STATE), &t), Ok(()));
}

#[test]
fn test_paddleduel_one_agent_leaves_slot_1_zero_and_ignores_its_action() {
    // The scripted right paddle bounces the ball, then later scores.
    let c = config(1, 1, OBS_STATE);
    let mut s = moving(586, 250, 6, 0);
    s.ry = fx(230);
    let mut p = pool(&c, 1);
    import(&mut p, s).unwrap();
    assert_eq!(
        step(&mut p, [STAY, DOWN]),
        [0.0, 0.0],
        "no bounce reward to anyone"
    );
    assert_eq!(state(&p).vx, fx(-6));
    let mut s = moving(2, 100, -6, 0);
    s.ly = fx(400);
    import(&mut p, s).unwrap();
    assert_eq!(step(&mut p, [STAY, UP]), [-1.0, 0.0]);
    assert_eq!(state(&p).score, [0, 1]);
    // Its action changes nothing: the scripted paddle moves by the script.
    let mut a = pool(&c, 4);
    let mut b = pool(&c, 4);
    for t in 0..400 {
        step(&mut a, [t % 3, UP]);
        step(&mut b, [t % 3, DOWN]);
        assert_eq!(state(&a), state(&b));
    }
    let out = a.outputs().unwrap();
    assert!(!out.alive()[1].get() && !out.acting()[1].get());
    assert!(!out.agent_terminated()[1].get());
    let obs = out.observation(0).unwrap();
    assert!(obs[36..].iter().all(|&b| b == 0), "slot 1's row stays zero");
}

#[test]
fn test_paddleduel_reward_is_tenths_converted_once() {
    // A hit worth 3 tenths, inside a decision of 8 frames: 0.3 as f32 from
    // f64 division, not a sum of f32 frame rewards.
    let c = PaddleDuelConfig {
        agents: 2,
        frameskip: 8,
        max_score: 3,
        bounce: 3,
        obs: OBS_STATE,
    };
    let mut p = pool(&c, 1);
    import(&mut p, moving(22, 250, -6, 0)).unwrap();
    assert_eq!(step(&mut p, [STAY, STAY]), [(3.0f64 / 10.0) as f32, 0.0]);
    // Nine tenths: f64 division gives 0.9f32; 9f32 * 0.1f32 would not.
    let c9 = PaddleDuelConfig {
        bounce: 9,
        ..c.clone()
    };
    let mut q = pool(&c9, 1);
    import(&mut q, moving(22, 250, -6, 0)).unwrap();
    let r = step(&mut q, [STAY, STAY]);
    assert_eq!(r[0], 0.9f32);
    assert_ne!(r[0], 9.0f32 * 0.1f32);
    // A point: exactly 1 and -1.
    let mut s = moving(606, 100, 6, 0);
    s.ry = fx(400);
    import(&mut p, s).unwrap();
    assert_eq!(step(&mut p, [STAY, STAY]), [1.0, -1.0]);
}

/// The reference: raw `i64` values, the rules written out once more.
#[derive(Clone, Debug, PartialEq)]
struct Ref {
    ly: i64,
    ry: i64,
    x: i64,
    y: i64,
    vx: i64,
    vy: i64,
    score: [i64; 2],
    wait: i64,
}

impl Ref {
    fn of(s: &PaddleDuelState) -> Self {
        let r = |v: Fx32| i64::from(v.to_raw());
        Self {
            ly: r(s.ly),
            ry: r(s.ry),
            x: r(s.x),
            y: r(s.y),
            vx: r(s.vx),
            vy: r(s.vy),
            score: [i64::from(s.score[0]), i64::from(s.score[1])],
            wait: i64::from(s.wait),
        }
    }

    fn park(&mut self, wait: i64) {
        (self.x, self.y, self.vx, self.vy, self.wait) = (304 * ONE, 234 * ONE, 0, 0, wait);
    }

    fn script(&self) -> i32 {
        if self.vx == 0 {
            return STAY;
        }
        let (ball, pad) = (self.y + 16 * ONE, self.ry + 35 * ONE);
        if ball > pad + 4 * ONE {
            DOWN
        } else if ball < pad - 4 * ONE {
            UP
        } else {
            STAY
        }
    }

    /// One frame; `serve` is the drawn (vx, vy) if this frame serves.
    fn frame(
        &mut self,
        acts: [i32; 2],
        speeds: [i64; 2],
        c: &PaddleDuelConfig,
        serve: (i64, i64),
    ) -> ([i64; 2], bool) {
        let mv = |y: i64, a: i32, sp: i64| {
            let to = (y + [-sp, 0, sp][a as usize] * ONE).clamp(0, 430 * ONE);
            (to, (to - y).signum())
        };
        let (ly, ml) = mv(self.ly, acts[0], speeds[0]);
        let (ry, mr) = mv(self.ry, acts[1], speeds[1]);
        (self.ly, self.ry) = (ly, ry);
        let mut t = [0i64; 2];
        if self.vx == 0 {
            if self.wait > 0 {
                self.wait -= 1;
                if self.wait == 0 {
                    (self.vx, self.vy) = serve;
                }
            }
            return (t, false);
        }
        self.y += self.vy;
        if self.y < 0 {
            (self.y, self.vy) = (-self.y, -self.vy);
        }
        if self.y > 468 * ONE {
            (self.y, self.vy) = (936 * ONE - self.y, -self.vy);
        }
        let old = self.x;
        self.x += self.vx;
        let over = |pad: i64| self.y < pad + 70 * ONE && pad < self.y + 32 * ONE;
        let spin = |vy: i64, m: i64| (vy + 2 * m * ONE).clamp(-8 * ONE, 8 * ONE);
        if self.vx < 0 && old >= 20 * ONE && self.x < 20 * ONE && over(self.ly) {
            (self.x, self.vx, self.vy) = (20 * ONE, 6 * ONE, spin(self.vy, ml));
            t[0] += i64::from(c.bounce);
        } else if self.vx > 0
            && old + 32 * ONE <= 620 * ONE
            && 620 * ONE < self.x + 32 * ONE
            && over(self.ry)
        {
            (self.x, self.vx, self.vy) = (588 * ONE, -6 * ONE, spin(self.vy, mr));
            t[1] += i64::from(c.bounce);
        }
        let scorer = if self.x < 0 {
            Some(1)
        } else if self.x > 608 * ONE {
            Some(0)
        } else {
            None
        };
        if let Some(k) = scorer {
            t[k] += 10;
            t[1 - k] -= 10;
            self.score[k] += 1;
            if self.score[k] >= i64::from(c.max_score) {
                self.park(0);
                return (t, true);
            }
            self.park(30);
        }
        (t, false)
    }

    /// One decision; tries each serve until the engine's state matches.
    fn decide(
        &mut self,
        c: &PaddleDuelConfig,
        acts: [i32; 2],
        engine: &PaddleDuelState,
    ) -> ([i64; 2], bool) {
        let run = |r: &mut Self, serve: (i64, i64)| {
            let mut total = [0i64; 2];
            for _ in 0..c.frameskip {
                let (right, speed) = if c.agents == 2 {
                    (acts[1], 8)
                } else {
                    (r.script(), 6)
                };
                let (t, end) = r.frame([acts[0], right], [8, speed], c, serve);
                total = [total[0] + t[0], total[1] + t[1]];
                if end {
                    return (total, true);
                }
            }
            (total, false)
        };
        let want = Self::of(engine);
        for vx in [6, -6] {
            for vy in SERVE_VY {
                let mut trial = self.clone();
                let out = run(&mut trial, (vx * ONE, i64::from(vy) * ONE));
                if trial == want {
                    *self = trial;
                    return out;
                }
            }
        }
        run(self, (0, 0))
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

fn baseline(c: &PaddleDuelConfig, s: &PaddleDuelState, k: u16) -> i32 {
    let mut discrete = [0];
    let mut writer = ActionWriter::new(&mut discrete, &mut []);
    let identity = RngIdentity::new(Seed::new(0), LogicalId::new(0));
    let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    PaddleDuel::act(c, s, AgentSlot::new(k, 2).unwrap(), &mut rng, &mut writer);
    discrete[0]
}

#[test]
fn test_paddleduel_play_matches_the_reference_by_decision() {
    let (mut hits, mut goals, mut ends) = (0, 0, 0);
    for (seed, agents, frameskip) in [
        (1u64, 1u32, 1u32),
        (2, 2, 1),
        (3, 1, 4),
        (4, 2, 4),
        (5, 2, 8),
        (6, 1, 3),
    ] {
        let c = config(agents, frameskip, OBS_STATE);
        for run in 0..6u64 {
            let mut p = pool(&c, seed * 10 + run);
            let mut r = Ref::of(&state(&p));
            let mut rng = Lcg(seed + run);
            for _ in 0..30_000 {
                let s = state(&p);
                let pick = |rng: &mut Lcg, k: u16| {
                    if rng.below(3) == 0 {
                        rng.below(3) as i32
                    } else {
                        baseline(&c, &s, k)
                    }
                };
                let acts = [pick(&mut rng, 0), pick(&mut rng, 1)];
                let got = step(&mut p, acts);
                let (want, end) = r.decide(&c, acts, &state(&p));
                assert_eq!(Ref::of(&state(&p)), r, "seed {seed}");
                let expect = |t: i64| (t as f64 / 10.0) as f32;
                assert_eq!(got[0], expect(want[0]));
                assert_eq!(got[1], if agents == 2 { expect(want[1]) } else { 0.0 });
                assert_eq!(terminated(&p), end);
                assert_eq!(PaddleDuel::validate(&c, &state(&p)), Ok(()));
                hits += i32::from(want.iter().any(|&t| t.rem_euclid(10) != 0));
                goals += i32::from(want.iter().any(|&t| t.abs() >= 9));
                if end {
                    ends += 1;
                    break;
                }
            }
        }
    }
    assert!(
        hits > 500 && goals > 100 && ends > 20,
        "{hits} {goals} {ends}"
    );
}

#[test]
fn test_paddleduel_extreme_play_stays_valid() {
    let c = PaddleDuelConfig {
        agents: 2,
        frameskip: 1,
        max_score: 21,
        bounce: 10,
        obs: OBS_STATE,
    };
    let mut rng = Lcg(3);
    for pattern in 0..4u64 {
        let mut p = pool(&c, pattern);
        for t in 0..40_000u64 {
            let acts = match pattern {
                0 => [UP, DOWN],
                1 => [DOWN, DOWN],
                2 => [
                    [UP, DOWN][(t / 30 % 2) as usize],
                    [DOWN, UP][(t / 50 % 2) as usize],
                ],
                _ => [rng.below(3) as i32, rng.below(3) as i32],
            };
            step(&mut p, acts);
            assert_eq!(
                PaddleDuel::validate(&c, &state(&p)),
                Ok(()),
                "{pattern} {t}"
            );
            if terminated(&p) {
                break;
            }
        }
    }
}

/// The world mirrored left to right with the players swapped: what slot 1
/// must see as slot 0 would.
fn mirrored(s: &PaddleDuelState) -> PaddleDuelState {
    let mut m = *s;
    (m.ly, m.ry) = (s.ry, s.ly);
    m.x = fx(608).saturating_sub(s.x);
    m.vx = Fx32::ZERO.saturating_sub(s.vx);
    m.score = [s.score[1], s.score[0]];
    m
}

fn raster(s: &PaddleDuelState) -> Vec<u8> {
    let cells = |v0: i64, v1: i64, world: i64| {
        let lo = v0.div_euclid(ONE);
        let hi = (v1 + ONE - 1).div_euclid(ONE);
        let a = (lo * 84).div_euclid(world).clamp(0, 84);
        let b = (hi * 84 + world - 1).div_euclid(world).clamp(0, 84);
        a..b.max(a)
    };
    let mut out = vec![0u8; 84 * 84];
    let r = |v: Fx32| i64::from(v.to_raw());
    let mut boxes = vec![
        (
            620 * ONE,
            640 * ONE,
            r(s.ry),
            r(s.ry) + 70 * ONE,
            SHADE_OTHER,
        ),
        (0, 20 * ONE, r(s.ly), r(s.ly) + 70 * ONE, SHADE_OBJECT),
    ];
    if !s.parked() {
        boxes.push((
            r(s.x),
            r(s.x) + 32 * ONE,
            r(s.y),
            r(s.y) + 32 * ONE,
            SHADE_OBJECT,
        ));
    }
    for (x0, x1, y0, y1, shade) in boxes {
        for row in cells(y0, y1, 500) {
            for col in cells(x0, x1, 640) {
                out[(row * 84 + col) as usize] = shade;
            }
        }
    }
    out
}

#[test]
fn test_paddleduel_slot_1_sees_the_mirror_in_both_observations() {
    let px = config(2, 1, OBS_PIXELS);
    let st = config(2, 1, OBS_STATE);
    let mut rng = Lcg(11);
    for seed in 0..4 {
        let mut a = pool(&px, seed);
        let mut b = pool(&st, seed);
        for _ in 0..2_000 {
            let s = state(&a);
            assert_eq!(s, state(&b));
            let pix = a.outputs().unwrap().observation(0).unwrap().to_vec();
            let side = (PIXELS * PIXELS) as usize;
            assert_eq!(pix[..side], raster(&s)[..]);
            assert_eq!(pix[side..], raster(&mirrored(&s))[..], "slot 1 pixels");
            assert_eq!(
                vector(&st, &s, 1),
                vector(&st, &mirrored(&s), 0),
                "slot 1 state"
            );
            let v = view(&s, 1);
            assert_eq!((v.own_score, v.other_score), (s.score[1], s.score[0]));
            let vals: Vec<f32> = b
                .outputs()
                .unwrap()
                .observation(0)
                .unwrap()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(vals[..9], vector(&st, &s, 0));
            assert_eq!(vals[9..], vector(&st, &s, 1));
            let acts = [rng.below(3) as i32, rng.below(3) as i32];
            step(&mut a, acts);
            step(&mut b, acts);
            if terminated(&a) {
                break;
            }
        }
    }
    // Actions keep world up and down: slot 1 pressing UP moves its paddle up.
    let mut p = fixture(2, moving(300, 100, 6, 0));
    step(&mut p, [STAY, UP]);
    assert_eq!(state(&p).ry, fx(207));
}

#[test]
fn test_paddleduel_state_vector_at_extremes() {
    let c = config(2, 1, OBS_STATE);
    let mut s = moving(608, 468, 6, 8);
    (s.ly, s.ry, s.score) = (fx(430), fx(0), [2, 0]);
    assert_eq!(
        vector(&c, &s, 0),
        [
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            (2.0f64 / 3.0) as f32,
            0.0
        ]
    );
    assert_eq!(
        vector(&c, &s, 1),
        [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            (2.0f64 / 3.0) as f32
        ]
    );
    let parked = state(&pool(&c, 1));
    let v = vector(&c, &parked, 0);
    assert_eq!((v[4], v[5], v[6]), (0.5, 0.5, 1.0));
}

#[test]
fn test_paddleduel_validate_names_each_rejection() {
    let c = config(2, 1, OBS_STATE);
    let good = moving(300, 100, 6, -3);
    assert_eq!(PaddleDuel::validate(&c, &good), Ok(()));
    let rejected = |s: &PaddleDuelState, code: &str| {
        assert_eq!(PaddleDuel::validate(&c, s).unwrap_err().code, code);
    };
    let edit = |f: &dyn Fn(&mut PaddleDuelState)| {
        let mut s = good;
        f(&mut s);
        s
    };
    rejected(
        &edit(&|s| s.x = Fx32::from_raw(s.x.to_raw() + 1)),
        "a fractional position or speed",
    );
    rejected(
        &edit(&|s| s.vy = Fx32::from_raw(3 << 15)),
        "a fractional position or speed",
    );
    rejected(&edit(&|s| s.ly = fx(431)), "a paddle out of range");
    rejected(&edit(&|s| s.ry = fx(-1)), "a paddle out of range");
    rejected(&edit(&|s| s.x = fx(609)), "a ball out of range");
    rejected(&edit(&|s| s.y = fx(469)), "a ball out of range");
    rejected(
        &edit(&|s| s.vx = fx(5)),
        "a horizontal speed other than -6, 0 or 6",
    );
    rejected(&edit(&|s| s.vy = fx(9)), "a vertical speed out of range");
    rejected(&edit(&|s| s.wait = 31), "a wait out of range");
    rejected(
        &edit(&|s| s.vx = Fx32::ZERO),
        "a parked ball off the centre or moving",
    );
    rejected(&edit(&|s| s.wait = 2), "a wait while the ball moves");
    rejected(&edit(&|s| s.score = [4, 0]), "a score above max_score");
    let mut both = state(&pool(&c, 1));
    both.wait = 0;
    both.score = [3, 3];
    rejected(&both, "both scores at max_score");
    rejected(
        &edit(&|s| s.score = [3, 1]),
        "a winning score exactly when parked with no wait, violated",
    );
    let mut idle = state(&pool(&c, 1));
    idle.wait = 0;
    rejected(
        &idle,
        "a winning score exactly when parked with no wait, violated",
    );
}

fn with_flags(p: &Pool, flags: [(bool, bool); 2]) -> DecodedRecord<PaddleDuel> {
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    for (k, (alive, acting)) in flags.iter().enumerate() {
        header.agents[k].alive = Bool8::new(*alive);
        header.agents[k].acting = Bool8::new(*acting);
    }
    DecodedRecord::from_parts(header, *good.state()).unwrap()
}

#[test]
fn test_paddleduel_import_by_phase_direct_and_persisted() {
    let c1 = config(1, 1, OBS_STATE);
    let c2 = config(2, 1, OBS_STATE);
    // Never reset: the default state only.
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let fresh = Pool::new(c2.clone(), options).unwrap();
    let rec = DecodedRecord::from(fresh.snapshot(0).unwrap());
    let mut q = Pool::new(c2.clone(), options).unwrap();
    q.import(&[0], &[rec]).unwrap();
    let bad = DecodedRecord::from_parts(*rec.header(), moving(300, 100, 6, 0)).unwrap();
    assert!(invalid(&q.import(&[0], &[bad])));
    // Active: a winner's score is refused, atomically.
    let mut p = pool(&c2, 1);
    let before = p.state_hash(0).unwrap();
    let mut won = state(&p);
    won.wait = 0;
    won.score = [3, 0];
    assert_eq!(PaddleDuel::validate(&c2, &won), Ok(()));
    match import(&mut p, won) {
        Err(EngineError::Invariant { violation, .. }) => {
            assert_eq!(violation.code, "a running episode in a terminal state");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(p.state_hash(0).unwrap(), before);
    // Header flags: used slots alive and acting, an unused slot neither.
    assert!(invalid(
        &p.import(&[0], &[with_flags(&p, [(true, true), (true, false)])])
    ));
    let mut one = pool(&c1, 1);
    assert!(invalid(&one.import(
        &[0],
        &[with_flags(&one, [(true, true), (true, true)])]
    )));
    assert!(invalid(&one.import(
        &[0],
        &[with_flags(&one, [(true, false), (false, false)])]
    )));
    // An unused slot acting but not alive: the engine refuses it before
    // Validate runs, so the environment's acting check there is defence.
    match one.import(&[0], &[with_flags(&one, [(true, true), (false, true)])]) {
        Err(EngineError::InvalidRecord { reason, .. }) => {
            assert_eq!(reason, "an acting agent that is not alive");
        }
        other => panic!("{other:?}"),
    }
    // Pending reset: the winning point, then a truncation mid-wait, each
    // round-trips directly and through a persisted file.
    let mut w = pool(&c2, 1);
    let mut s = moving(606, 100, 6, 0);
    s.ry = fx(400);
    s.score = [2, 0];
    import(&mut w, s).unwrap();
    step(&mut w, [STAY, STAY]);
    assert!(terminated(&w));
    let mut t = pool_with(&c2, 1, 3);
    for _ in 0..3 {
        step(&mut t, [STAY, STAY]);
    }
    assert!(t.outputs().unwrap().episode_truncated()[0].get() && state(&t).parked());
    for (src, limit) in [(&w, 1_000_000), (&t, 3)] {
        let rec = DecodedRecord::from(src.snapshot(0).unwrap());
        let mut q = pool_with(&c2, 7, limit);
        q.import(&[0], &[rec]).unwrap();
        assert_eq!(q.state_hash(0).unwrap(), src.state_hash(0).unwrap());
        let bytes = src.encode(&[0]).unwrap();
        let decoded = q.decode(&bytes, DecodeLimits::default()).unwrap();
        let mut r = pool_with(&c2, 8, limit);
        r.import(&[0], &decoded).unwrap();
        assert_eq!(r.state_hash(0).unwrap(), src.state_hash(0).unwrap());
    }
}

/// The baseline's reward in tenths for slot 0 against the scripted paddle within a
/// bound, summed over seeds; frozen for v1, as it feeds the goldens: a change
/// to `Heuristic::act` or the script moves this pin and needs a new
/// `ENV_VERSION` with new fixtures.
#[test]
fn test_paddleduel_baseline_points_are_pinned() {
    let c = PaddleDuelConfig::default();
    let mut tenths = 0i64;
    for seed in 0..50 {
        let mut p = pool(&c, seed);
        for _ in 0..5_000 {
            let s = state(&p);
            tenths +=
                (f64::from(step(&mut p, [baseline(&c, &s, 0), STAY])[0]) * 10.0).round() as i64;
            if terminated(&p) {
                break;
            }
        }
    }
    assert_eq!(tenths, 7_303);
}

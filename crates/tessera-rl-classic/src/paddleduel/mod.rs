// SPDX-License-Identifier: MIT OR Apache-2.0

//! Paddle duel (`tessera/PaddleDuel-v1`): two paddles and a ball, for one agent
//! against a scripted paddle or two agents. Cross-platform tier: positions and
//! speeds are `Fx32`, integers in every generated state. Original rules and
//! rendering; the full rules are in the arcade design note, revision 8.
//!
//! - World 640 x 500, y downward, walls top and bottom, goals left and right.
//!   Paddles 20 x 70 at x 0 and 620; ball 32 x 32. Boxes are half-open.
//! - Action: `Discrete(3)`: 0 up, 1 stay, 2 down, 8 per frame. No masks.
//! - A decision runs `frameskip` frames; rewards sum as integer tenths and
//!   convert once. A terminal frame ends the decision.
//! - Serve: the ball parks at the centre for 30 frames, then starts with `vx`
//!   6 or -6 and `vy` from {-3, -2, 2, 3}, one draw.
//! - A paddle hits a ball crossing its face; `vy` gains twice the sign of the
//!   paddle's applied move, clamped to 8. A ball past a goal line scores.
//! - Slot 0 is left, slot 1 right and sees the world mirrored left to right.
//!
//! Kernel arithmetic uses `saturating_*` throughout, as kernels are infallible:
//! every value is bounded far inside `Fx32` by the rules, so none saturates.

mod baseline;
mod validate;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, Reserved,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::Fx32;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/PaddleDuel-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// Move up.
pub const UP: i32 = 0;
/// Stay; the no-op.
pub const STAY: i32 = 1;
/// Move down.
pub const DOWN: i32 = 2;
/// Choices of the action head.
pub const ACTIONS: u32 = 3;
/// `obs`: grayscale pixels.
pub const OBS_PIXELS: u32 = 0;
/// `obs`: the state vector.
pub const OBS_STATE: u32 = 1;
/// Pixels on a side of the `pixels` observation.
pub const PIXELS: u32 = 84;
/// Fields of the `state` observation.
pub const STATE_FIELDS: u32 = 9;

/// World width.
pub const WIDTH: i32 = 640;
/// World height.
pub const HEIGHT: i32 = 500;
/// Paddle width.
pub const PADDLE_W: i32 = 20;
/// Paddle height.
pub const PADDLE_H: i32 = 70;
/// Largest paddle y.
pub const PADDLE_MAX: i32 = HEIGHT - PADDLE_H;
/// Paddle start y.
pub const PADDLE_START: i32 = 215;
/// Agent paddle speed per frame.
pub const PADDLE_SPEED: i32 = 8;
/// Scripted paddle speed per frame.
pub const SCRIPT_SPEED: i32 = 6;
/// Ball side.
pub const BALL: i32 = 32;
/// Largest ball x.
pub const BALL_MAX_X: i32 = WIDTH - BALL;
/// Largest ball y.
pub const BALL_MAX_Y: i32 = HEIGHT - BALL;
/// The left paddle's face.
pub const LEFT_FACE: i32 = PADDLE_W;
/// The right paddle's face.
pub const RIGHT_FACE: i32 = WIDTH - PADDLE_W;
/// Centre position of a parked ball.
pub const CENTRE: (i32, i32) = (304, 234);
/// Frames a serve waits.
pub const SERVE_WAIT: u8 = 30;
/// Horizontal speed.
pub const SPEED_X: i32 = 6;
/// Largest vertical speed.
pub const SPEED_Y: i32 = 8;
/// Serve vertical speeds.
pub const SERVE_VY: [i32; 4] = [-3, -2, 2, 3];
/// Shade of the opponent's paddle.
pub const SHADE_OTHER: u8 = 160;
/// Shade of the own paddle and the ball.
pub const SHADE_OBJECT: u8 = 255;
/// Reward of a point, in tenths.
pub const POINT_TENTHS: i32 = 10;

/// An integer as `Fx32`.
#[must_use]
pub const fn fx(v: i32) -> Fx32 {
    Fx32::from_raw(v << 16)
}

/// Paddle duel configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct PaddleDuelConfig {
    /// Agents: 1 against a scripted right paddle, or 2.
    #[config(default = 1, range = 1..=2)]
    pub agents: u32,
    /// Frames per decision.
    #[config(default = 4, range = 1..=8)]
    pub frameskip: u32,
    /// Points that win.
    #[config(default = 11, range = 1..=21)]
    pub max_score: u32,
    /// Reward of an own hit, in tenths.
    #[config(default = 1, range = 0..=10)]
    pub bounce: u32,
    /// The observation: [`OBS_PIXELS`] or [`OBS_STATE`].
    #[config(default = 0, range = 0..=1)]
    pub obs: u32,
}

/// Paddle duel state: 32 bytes, no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct PaddleDuelState {
    /// Left paddle top edge.
    pub ly: Fx32,
    /// Right paddle top edge.
    pub ry: Fx32,
    /// Ball left edge.
    pub x: Fx32,
    /// Ball top edge.
    pub y: Fx32,
    /// Ball horizontal speed: 6, -6, or 0 when parked.
    pub vx: Fx32,
    /// Ball vertical speed.
    pub vy: Fx32,
    /// Points of left and right.
    pub score: [u8; 2],
    /// Frames left before a parked ball serves.
    pub wait: u8,
    /// Zero.
    pub reserved: Reserved<5>,
}

fn meets(a0: Fx32, a1: Fx32, b0: Fx32, b1: Fx32) -> bool {
    a0 < b1 && b0 < a1
}

fn neg(v: Fx32) -> Fx32 {
    Fx32::ZERO.saturating_sub(v)
}

/// The sign of a paddle's applied move.
fn sign(before: Fx32, after: Fx32) -> i32 {
    match after.cmp(&before) {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    }
}

impl PaddleDuelState {
    /// Whether the ball is parked.
    #[must_use]
    pub fn parked(&self) -> bool {
        self.vx == Fx32::ZERO
    }

    fn park(&mut self, wait: u8) {
        self.x = fx(CENTRE.0);
        self.y = fx(CENTRE.1);
        self.vx = Fx32::ZERO;
        self.vy = Fx32::ZERO;
        self.wait = wait;
    }

    /// The scripted right paddle's move: its centre toward the ball's.
    #[must_use]
    pub fn script(&self) -> i32 {
        if self.parked() {
            return STAY;
        }
        let ball = self.y.saturating_add(fx(BALL / 2));
        let paddle = self.ry.saturating_add(fx(PADDLE_H / 2));
        if ball > paddle.saturating_add(fx(4)) {
            DOWN
        } else if ball < paddle.saturating_sub(fx(4)) {
            UP
        } else {
            STAY
        }
    }

    /// One frame with the left and right paddles' actions and speeds.
    /// Returns each side's reward in tenths and whether it is terminal.
    pub fn frame(
        &mut self,
        actions: [i32; 2],
        speeds: [i32; 2],
        cfg: &PaddleDuelConfig,
        rng: &mut StepRng,
    ) -> ([i32; 2], bool) {
        let shift = |action: i32, speed: i32| match action {
            UP => -speed,
            DOWN => speed,
            _ => 0,
        };
        let range = |v: Fx32| v.clamp(Fx32::ZERO, fx(PADDLE_MAX));
        let (l0, r0) = (self.ly, self.ry);
        self.ly = range(self.ly.saturating_add(fx(shift(actions[0], speeds[0]))));
        self.ry = range(self.ry.saturating_add(fx(shift(actions[1], speeds[1]))));
        let moves = [sign(l0, self.ly), sign(r0, self.ry)];
        let mut tenths = [0i32; 2];
        if self.parked() {
            if self.wait > 0 {
                self.wait -= 1;
                if self.wait == 0 {
                    let pick = rng.below_u32(core::num::NonZeroU32::MIN.saturating_add(7));
                    self.vx = fx(if pick < 4 { SPEED_X } else { -SPEED_X });
                    self.vy = fx(SERVE_VY.get((pick % 4) as usize).copied().unwrap_or(2));
                }
            }
            return (tenths, false);
        }
        // y, then x.
        self.y = self.y.saturating_add(self.vy);
        if self.y < Fx32::ZERO {
            self.y = neg(self.y);
            self.vy = neg(self.vy);
        } else if self.y > fx(BALL_MAX_Y) {
            self.y = fx(2 * BALL_MAX_Y).saturating_sub(self.y);
            self.vy = neg(self.vy);
        }
        let old = self.x;
        self.x = self.x.saturating_add(self.vx);
        let spin = |vy: Fx32, m: i32| {
            vy.saturating_add(fx(2 * m))
                .clamp(fx(-SPEED_Y), fx(SPEED_Y))
        };
        let y1 = self.y.saturating_add(fx(BALL));
        let bounce = i32::try_from(cfg.bounce).unwrap_or(0);
        if self.vx < Fx32::ZERO
            && fx(LEFT_FACE) <= old
            && self.x < fx(LEFT_FACE)
            && meets(self.y, y1, self.ly, self.ly.saturating_add(fx(PADDLE_H)))
        {
            self.x = fx(LEFT_FACE);
            self.vx = fx(SPEED_X);
            self.vy = spin(self.vy, moves[0]);
            tenths[0] += bounce;
        } else if self.vx > Fx32::ZERO
            && old.saturating_add(fx(BALL)) <= fx(RIGHT_FACE)
            && fx(RIGHT_FACE) < self.x.saturating_add(fx(BALL))
            && meets(self.y, y1, self.ry, self.ry.saturating_add(fx(PADDLE_H)))
        {
            self.x = fx(RIGHT_FACE - BALL);
            self.vx = fx(-SPEED_X);
            self.vy = spin(self.vy, moves[1]);
            tenths[1] += bounce;
        }
        // Goals.
        let scorer = if self.x < Fx32::ZERO {
            Some(1)
        } else if self.x > fx(BALL_MAX_X) {
            Some(0)
        } else {
            None
        };
        if let Some(k) = scorer {
            let delta = if k == 0 { POINT_TENTHS } else { -POINT_TENTHS };
            tenths[0] += delta;
            tenths[1] -= delta;
            let won = self.score.get_mut(k).is_some_and(|s| {
                *s = s.saturating_add(1);
                u32::from(*s) >= cfg.max_score
            });
            if won {
                self.park(0);
                return (tenths, true);
            }
            self.park(SERVE_WAIT);
        }
        (tenths, false)
    }
}

/// Grayscale pixels, one leaf.
#[derive(Observation)]
pub struct PaddleDuelPixels {
    /// `[1, row, column]`.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [1, PIXELS, PIXELS]))]
    pub obs: ObsLeaf<u8>,
}

/// The state vector, one leaf.
#[derive(Observation)]
pub struct PaddleDuelVector {
    /// Nine fields, in the observer's view.
    #[space(box(low = 0.0, high = 1.0, dtype = f32, shape = [STATE_FIELDS]))]
    pub obs: ObsLeaf<f32>,
}

/// Paddle duel observation: one leaf whose type `obs` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaddleDuelObs;

impl Observation<PaddleDuelConfig> for PaddleDuelObs {
    const LEAVES: usize = 1;

    fn space(cfg: &PaddleDuelConfig) -> Result<Space, SpaceError> {
        if cfg.obs == OBS_STATE {
            <PaddleDuelVector as Observation<PaddleDuelConfig>>::space(cfg)
        } else {
            <PaddleDuelPixels as Observation<PaddleDuelConfig>>::space(cfg)
        }
    }
}

/// The paddle duel environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PaddleDuel;

/// The world as `me` sees it: slot 1 mirrored left to right, own paddle left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    /// Own paddle top edge.
    pub own: Fx32,
    /// Other paddle top edge.
    pub other: Fx32,
    /// Ball x in the view.
    pub x: Fx32,
    /// Ball y.
    pub y: Fx32,
    /// Ball horizontal speed in the view.
    pub vx: Fx32,
    /// Ball vertical speed.
    pub vy: Fx32,
    /// Whether the ball is parked.
    pub parked: bool,
    /// Own points.
    pub own_score: u8,
    /// Other points.
    pub other_score: u8,
}

/// `s` seen from slot `me`.
#[must_use]
pub fn view(s: &PaddleDuelState, me: usize) -> View {
    let [left, right] = s.score;
    if me == 0 {
        View {
            own: s.ly,
            other: s.ry,
            x: s.x,
            y: s.y,
            vx: s.vx,
            vy: s.vy,
            parked: s.parked(),
            own_score: left,
            other_score: right,
        }
    } else {
        View {
            own: s.ry,
            other: s.ly,
            x: fx(BALL_MAX_X).saturating_sub(s.x),
            y: s.y,
            vx: neg(s.vx),
            vy: s.vy,
            parked: s.parked(),
            own_score: right,
            other_score: left,
        }
    }
}

fn span(a0: Fx32, a1: Fx32, world: i32) -> (usize, usize) {
    let side = PIXELS.cast_signed();
    let lo = a0.to_raw() >> 16;
    let hi = (a1.to_raw().saturating_add(0xFFFF)) >> 16;
    let c0 = (lo * side).div_euclid(world).clamp(0, side);
    let c1 = (hi * side + world - 1).div_euclid(world).clamp(0, side);
    let index = |v: i32| usize::try_from(v).unwrap_or(0);
    (index(c0), index(c1.max(c0)))
}

fn draw_pixels(v: &View, bytes: &mut [u8]) {
    let side = PIXELS as usize;
    bytes.fill(0);
    let mut paint = |x0: Fx32, x1: Fx32, y0: Fx32, y1: Fx32, shade: u8| {
        let (c0, c1) = span(x0, x1, WIDTH);
        let (r0, r1) = span(y0, y1, HEIGHT);
        for row in r0..r1 {
            if let Some(line) = bytes.get_mut(row * side + c0..row * side + c1) {
                line.fill(shade);
            }
        }
    };
    let tall = |y: Fx32| y.saturating_add(fx(PADDLE_H));
    paint(
        fx(RIGHT_FACE),
        fx(WIDTH),
        v.other,
        tall(v.other),
        SHADE_OTHER,
    );
    paint(Fx32::ZERO, fx(LEFT_FACE), v.own, tall(v.own), SHADE_OBJECT);
    if !v.parked {
        paint(
            v.x,
            v.x.saturating_add(fx(BALL)),
            v.y,
            v.y.saturating_add(fx(BALL)),
            SHADE_OBJECT,
        );
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the observation is f32 by its space; the cast is the recipe's last step"
)]
fn scaled(v: Fx32, offset: f64, scale: f64) -> f32 {
    ((v.to_f64() + offset) / scale) as f32
}

/// The state vector seen from `me`, in field order.
#[must_use]
pub fn vector(cfg: &PaddleDuelConfig, s: &PaddleDuelState, me: usize) -> [f32; 9] {
    let v = view(s, me);
    let top = f64::from(PADDLE_MAX);
    let max = f64::from(cfg.max_score);
    [
        scaled(v.own, 0.0, top),
        scaled(v.other, 0.0, top),
        scaled(v.x, 0.0, f64::from(BALL_MAX_X)),
        scaled(v.y, 0.0, f64::from(BALL_MAX_Y)),
        scaled(v.vx, f64::from(SPEED_X), f64::from(2 * SPEED_X)),
        scaled(v.vy, f64::from(SPEED_Y), f64::from(2 * SPEED_Y)),
        if v.parked { 1.0 } else { 0.0 },
        scaled(fx(i32::from(v.own_score)), 0.0, max),
        scaled(fx(i32::from(v.other_score)), 0.0, max),
    ]
}

fn slot(index: usize) -> Option<AgentSlot> {
    u16::try_from(index)
        .ok()
        .and_then(|i| AgentSlot::new(i, MAX_AGENTS))
}

/// Tenths of a point as the `f32` reward: `f64` division, one cast.
#[expect(
    clippy::cast_possible_truncation,
    reason = "rewards are f32; the cast is the recipe's last step"
)]
fn reward(tenths: i32) -> f32 {
    (f64::from(tenths) / 10.0) as f32
}

impl Environment for PaddleDuel {
    type State = PaddleDuelState;
    type Config = PaddleDuelConfig;
    type Obs = PaddleDuelObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &PaddleDuelConfig) -> Result<Space, SpaceError> {
        PaddleDuelObs::space(cfg)
    }

    fn action_space(_: &PaddleDuelConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn initial_agents(cfg: &PaddleDuelConfig) -> usize {
        cfg.agents as usize
    }

    /// No draw: the serve draws when its wait ends.
    fn reset(_: &PaddleDuelConfig, _: &mut StepRng, s: &mut PaddleDuelState) {
        *s = PaddleDuelState {
            ly: fx(PADDLE_START),
            ry: fx(PADDLE_START),
            ..PaddleDuelState::default()
        };
        s.park(SERVE_WAIT);
    }

    fn step(
        cfg: &PaddleDuelConfig,
        s: &mut PaddleDuelState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let two = cfg.agents == 2;
        let act = |k: usize| slot(k).and_then(|a| actions.discrete(a, 0)).unwrap_or(STAY);
        let left = act(0);
        let mut total = [0i32; 2];
        let mut over = false;
        for _ in 0..cfg.frameskip {
            let (right, speed) = if two {
                (act(1), PADDLE_SPEED)
            } else {
                (s.script(), SCRIPT_SPEED)
            };
            let (tenths, terminal) = s.frame([left, right], [PADDLE_SPEED, speed], cfg, rng);
            total[0] += tenths[0];
            total[1] += tenths[1];
            if terminal {
                over = true;
                break;
            }
        }
        for (k, t) in total.iter().enumerate().take(cfg.agents as usize) {
            if let Some(agent) = slot(k) {
                out.set_reward(agent, reward(*t));
            }
        }
        if over {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &PaddleDuelConfig,
        s: &PaddleDuelState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, PaddleDuelObs>,
    ) {
        let Some(bytes) = out.leaf(0) else {
            return;
        };
        let me = agent.index();
        if cfg.obs == OBS_STATE {
            for (chunk, v) in bytes
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(vector(cfg, s, me))
            {
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        } else {
            draw_pixels(&view(s, me), bytes);
        }
    }
}

impl tessera_rl_core::Persist for PaddleDuel {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}

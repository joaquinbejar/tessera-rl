// SPDX-License-Identifier: MIT OR Apache-2.0

//! Brick wall (`tessera/BrickWall-v1`): one paddle, a ball and a wall of
//! bricks. Cross-platform tier: positions and speeds are `Fx32`. Original rules
//! text, layout, scoring and rendering; the full rules are in the arcade design
//! note, revision 6, and are summarized here.
//!
//! - World 576 x 330, y downward, walls left, right and top. Bricks 6 x 18 of
//!   32 x 12 from y 40; paddle 62 x 8 at y 310; ball 6 x 6. Boxes are half-open.
//! - Action: `Discrete(3)`: left, stay, right, 8 per frame. No masks.
//! - A decision runs `frameskip` frames with one action; rewards sum, and a
//!   terminal frame ends the decision.
//! - Serve: the ball parks at (285, 200) for 30 frames, then starts down with
//!   `vx` drawn from {-3, -2, 2, 3}; that draw is the only randomness.
//! - Bricks: the first overlapped in row-major order clears and negates `vy`,
//!   one per frame. Points 3, 2, 1 by pairs of rows. The first empty wall
//!   refills; the second terminates. Losing the last of `lives` terminates.
//! - Paddle: a ball crossing its top bounces up, with `vx` changed by the
//!   offset from the paddle centre times 3/34, clamped to 6.
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
pub const ENV_ID: &str = "tessera/BrickWall-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(BrickWall, id = "tessera/BrickWall-v1");
/// Move left.
pub const LEFT: i32 = 0;
/// Stay; the no-op.
pub const STAY: i32 = 1;
/// Move right.
pub const RIGHT: i32 = 2;
/// Choices of the action head.
pub const ACTIONS: u32 = 3;
/// `obs`: grayscale pixels.
pub const OBS_PIXELS: u32 = 0;
/// `obs`: the state vector.
pub const OBS_STATE: u32 = 1;
/// Pixels on a side of the `pixels` observation.
pub const PIXELS: u32 = 84;
/// Fields of the `state` observation.
pub const STATE_FIELDS: u32 = 116;

/// World width.
pub const WIDTH: i32 = 576;
/// World height.
pub const HEIGHT: i32 = 330;
/// Brick rows.
pub const ROWS: i32 = 6;
/// Brick columns.
pub const COLS: i32 = 18;
/// Bricks in a wall.
pub const BRICKS: u32 = 108;
/// Brick width.
pub const BRICK_W: i32 = 32;
/// Brick height.
pub const BRICK_H: i32 = 12;
/// Top of brick row 0.
pub const WALL_TOP: i32 = 40;
/// Paddle width.
pub const PADDLE_W: i32 = 62;
/// Paddle height.
pub const PADDLE_H: i32 = 8;
/// Top of the paddle.
pub const PADDLE_Y: i32 = 310;
/// Largest paddle x.
pub const PADDLE_MAX: i32 = WIDTH - PADDLE_W;
/// Paddle start x.
pub const PADDLE_START: i32 = 257;
/// Paddle speed per frame.
pub const PADDLE_SPEED: i32 = 8;
/// Ball side.
pub const BALL: i32 = 6;
/// Largest ball x: the right reflection plane.
pub const BALL_MAX_X: i32 = WIDTH - BALL;
/// Serve position.
pub const SERVE: (i32, i32) = (285, 200);
/// Frames a serve waits.
pub const SERVE_WAIT: u8 = 30;
/// Vertical speed.
pub const SPEED_Y: i32 = 4;
/// Largest horizontal speed.
pub const SPEED_X: i32 = 6;
/// Serve speeds, drawn uniformly.
pub const SERVE_VX: [i32; 4] = [-3, -2, 2, 3];
/// The paddle coefficient: floor of 3/34 in Q16.16.
pub const PADDLE_K_RAW: i32 = 5782;
/// Points of a wall.
pub const WALL_POINTS: u16 = 216;
/// Points of the game.
pub const MAX_SCORE: u16 = 2 * WALL_POINTS;
/// Shades of rows 0 and 1, 2 and 3, 4 and 5.
pub const SHADE_BANDS: [u8; 3] = [200, 150, 100];
/// Shade of the paddle and the ball.
pub const SHADE_OBJECT: u8 = 255;

/// The full wall: 108 bits.
pub const FULL: [u64; 2] = [u64::MAX, (1 << (BRICKS - 64)) - 1];

/// An integer as `Fx32`.
#[must_use]
pub const fn fx(v: i32) -> Fx32 {
    Fx32::from_raw(v << 16)
}

/// Points of a brick in `row`.
#[must_use]
pub const fn points(row: i32) -> u16 {
    match row {
        0 | 1 => 3,
        2 | 3 => 2,
        _ => 1,
    }
}

/// Brick wall configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct BrickWallConfig {
    /// Frames per decision.
    #[config(default = 4, range = 1..=8)]
    pub frameskip: u32,
    /// Balls per game.
    #[config(default = 5, range = 1..=9)]
    pub lives: u32,
    /// The observation: [`OBS_PIXELS`] or [`OBS_STATE`].
    #[config(default = 0, range = 0..=1)]
    pub obs: u32,
}

/// Brick wall state: 48 bytes, no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct BrickWallState {
    /// Present bricks, bit `18 * row + col`; bits 108 and above zero.
    pub bricks: [u64; 2],
    /// Paddle left edge.
    pub px: Fx32,
    /// Ball left edge.
    pub x: Fx32,
    /// Ball top edge.
    pub y: Fx32,
    /// Ball horizontal speed.
    pub vx: Fx32,
    /// Ball vertical speed: 4, -4, or 0 when parked.
    pub vy: Fx32,
    /// Points so far.
    pub score: u16,
    /// Frames left before a parked ball serves.
    pub wait: u8,
    /// Balls left.
    pub lives: u8,
    /// Walls refilled: 0 or 1.
    pub refills: u8,
    /// Zero.
    pub reserved: Reserved<7>,
}

/// Whether two half-open intervals `[a0, a1)` and `[b0, b1)` intersect.
fn meets(a0: Fx32, a1: Fx32, b0: Fx32, b1: Fx32) -> bool {
    a0 < b1 && b0 < a1
}

impl BrickWallState {
    /// Whether the brick at `row`, `col` is present.
    #[must_use]
    pub fn has(&self, row: i32, col: i32) -> bool {
        let Ok(k) = u32::try_from(row * COLS + col) else {
            return false;
        };
        self.bricks
            .get((k / 64) as usize)
            .is_some_and(|w| w >> (k % 64) & 1 == 1)
    }

    fn clear(&mut self, row: i32, col: i32) {
        if let Ok(k) = u32::try_from(row * COLS + col)
            && let Some(w) = self.bricks.get_mut((k / 64) as usize)
        {
            *w &= !(1 << (k % 64));
        }
    }

    /// Whether the ball is parked.
    #[must_use]
    pub fn parked(&self) -> bool {
        self.vy == Fx32::ZERO
    }

    /// Whether the wall is empty.
    #[must_use]
    pub fn empty(&self) -> bool {
        self.bricks == [0, 0]
    }

    fn park(&mut self, wait: u8) {
        self.x = fx(SERVE.0);
        self.y = fx(SERVE.1);
        self.vx = Fx32::ZERO;
        self.vy = Fx32::ZERO;
        self.wait = wait;
    }

    /// The first present brick the ball overlaps, in row-major order.
    fn hit(&self) -> Option<(i32, i32)> {
        let (x1, y1) = (
            self.x.saturating_add(fx(BALL)),
            self.y.saturating_add(fx(BALL)),
        );
        for row in 0..ROWS {
            let top = fx(WALL_TOP + BRICK_H * row);
            if !meets(self.y, y1, top, top.saturating_add(fx(BRICK_H))) {
                continue;
            }
            for col in 0..COLS {
                let left = fx(BRICK_W * col);
                if self.has(row, col) && meets(self.x, x1, left, left.saturating_add(fx(BRICK_W))) {
                    return Some((row, col));
                }
            }
        }
        None
    }

    /// One frame; returns its points and whether it is terminal.
    pub fn frame(&mut self, action: i32, rng: &mut StepRng) -> (u16, bool) {
        let shift = match action {
            LEFT => -PADDLE_SPEED,
            RIGHT => PADDLE_SPEED,
            _ => 0,
        };
        self.px = self
            .px
            .saturating_add(fx(shift))
            .clamp(Fx32::ZERO, fx(PADDLE_MAX));
        if self.parked() {
            if self.wait > 0 {
                self.wait -= 1;
                if self.wait == 0 {
                    let pick = rng.below_u32(core::num::NonZeroU32::MIN.saturating_add(3));
                    let vx = SERVE_VX.get(pick as usize).copied().unwrap_or(2);
                    self.vx = fx(vx);
                    self.vy = fx(SPEED_Y);
                }
            }
            return (0, false);
        }
        // x, then y.
        self.x = self.x.saturating_add(self.vx);
        if self.x < Fx32::ZERO {
            self.x = Fx32::ZERO.saturating_sub(self.x);
            self.vx = Fx32::ZERO.saturating_sub(self.vx);
        } else if self.x > fx(BALL_MAX_X) {
            self.x = fx(2 * BALL_MAX_X).saturating_sub(self.x);
            self.vx = Fx32::ZERO.saturating_sub(self.vx);
        }
        let old = self.y;
        self.y = self.y.saturating_add(self.vy);
        if self.y < Fx32::ZERO {
            self.y = Fx32::ZERO.saturating_sub(self.y);
            self.vy = fx(SPEED_Y);
        }
        // Bricks.
        let mut gained = 0;
        if let Some((row, col)) = self.hit() {
            self.clear(row, col);
            gained = points(row);
            self.score = self.score.saturating_add(gained);
            self.vy = Fx32::ZERO.saturating_sub(self.vy);
            if self.empty() {
                if self.refills == 0 {
                    self.refills = 1;
                    self.bricks = FULL;
                    self.park(SERVE_WAIT);
                    return (gained, false);
                }
                self.park(0);
                return (gained, true);
            }
        }
        // Paddle.
        let bottom = |y: Fx32| y.saturating_add(fx(BALL));
        if self.vy == fx(SPEED_Y)
            && bottom(old) <= fx(PADDLE_Y)
            && fx(PADDLE_Y) < bottom(self.y)
            && meets(
                self.x,
                self.x.saturating_add(fx(BALL)),
                self.px,
                self.px.saturating_add(fx(PADDLE_W)),
            )
        {
            self.y = fx(PADDLE_Y - BALL);
            self.vy = fx(-SPEED_Y);
            let d = self
                .x
                .saturating_add(fx(BALL / 2))
                .saturating_sub(self.px.saturating_add(fx(PADDLE_W / 2)));
            let dv = d.saturating_mul(Fx32::from_raw(PADDLE_K_RAW));
            self.vx = self.vx.saturating_add(dv).clamp(fx(-SPEED_X), fx(SPEED_X));
        }
        // Miss.
        if self.y > fx(HEIGHT) {
            self.lives = self.lives.saturating_sub(1);
            if self.lives == 0 {
                self.park(0);
                return (gained, true);
            }
            self.park(SERVE_WAIT);
        }
        (gained, false)
    }
}

/// Grayscale pixels, one leaf.
#[derive(Observation)]
pub struct BrickWallPixels {
    /// `[1, row, column]`.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [1, PIXELS, PIXELS]))]
    pub obs: ObsLeaf<u8>,
}

/// The state vector, one leaf.
#[derive(Observation)]
pub struct BrickWallVector {
    /// Eight scalars, then 108 brick flags.
    #[space(box(low = 0.0, high = 1.0, dtype = f32, shape = [STATE_FIELDS]))]
    pub obs: ObsLeaf<f32>,
}

/// Brick wall observation: one leaf whose type `obs` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BrickWallObs;

impl Observation<BrickWallConfig> for BrickWallObs {
    const LEAVES: usize = 1;

    fn space(cfg: &BrickWallConfig) -> Result<Space, SpaceError> {
        if cfg.obs == OBS_STATE {
            <BrickWallVector as Observation<BrickWallConfig>>::space(cfg)
        } else {
            <BrickWallPixels as Observation<BrickWallConfig>>::space(cfg)
        }
    }
}

/// The brick wall environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BrickWall;

/// Output cells `[lo, hi)` covered by the world interval `[a0, a1)` of a side
/// `world` long, clipped to the frame.
fn span(a0: Fx32, a1: Fx32, world: i32) -> (usize, usize) {
    let side = PIXELS.cast_signed();
    let lo = a0.to_raw() >> 16;
    let hi = (a1.to_raw().saturating_add(0xFFFF)) >> 16;
    let c0 = (lo * side).div_euclid(world).clamp(0, side);
    let c1 = (hi * side + world - 1).div_euclid(world).clamp(0, side);
    let index = |v: i32| usize::try_from(v).unwrap_or(0);
    (index(c0), index(c1.max(c0)))
}

/// `bytes` is zero on entry (ADR-0058): only the lit pixels are written.
fn draw_pixels(s: &BrickWallState, bytes: &mut [u8]) {
    let side = PIXELS as usize;
    let mut paint = |x0: Fx32, x1: Fx32, y0: Fx32, y1: Fx32, shade: u8| {
        let (c0, c1) = span(x0, x1, WIDTH);
        let (r0, r1) = span(y0, y1, HEIGHT);
        for row in r0..r1 {
            if let Some(line) = bytes.get_mut(row * side + c0..row * side + c1) {
                line.fill(shade);
            }
        }
    };
    for row in 0..ROWS {
        let shade = SHADE_BANDS
            .get(usize::try_from(row / 2).unwrap_or(0))
            .copied()
            .unwrap_or(0);
        for col in 0..COLS {
            if s.has(row, col) {
                let (x, y) = (BRICK_W * col, WALL_TOP + BRICK_H * row);
                paint(fx(x), fx(x + BRICK_W), fx(y), fx(y + BRICK_H), shade);
            }
        }
    }
    paint(
        s.px,
        s.px.saturating_add(fx(PADDLE_W)),
        fx(PADDLE_Y),
        fx(PADDLE_Y + PADDLE_H),
        SHADE_OBJECT,
    );
    if !s.parked() {
        paint(
            s.x,
            s.x.saturating_add(fx(BALL)),
            s.y,
            s.y.saturating_add(fx(BALL)),
            SHADE_OBJECT,
        );
    }
}

/// The frozen recipe: exact `f64`, an offset, a scale, one cast.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the observation is f32 by its space; the cast is the recipe's last step"
)]
fn scaled(v: Fx32, offset: f64, scale: f64) -> f32 {
    ((v.to_f64() + offset) / scale) as f32
}

/// The state vector of `s`, in field order.
#[must_use]
pub fn vector(cfg: &BrickWallConfig, s: &BrickWallState) -> [f32; STATE_FIELDS as usize] {
    let mut v = [0.0f32; STATE_FIELDS as usize];
    let lives = f64::from(cfg.lives);
    let head = [
        scaled(s.px, 0.0, f64::from(PADDLE_MAX)),
        scaled(s.x, 0.0, f64::from(BALL_MAX_X)),
        scaled(s.y, 0.0, f64::from(HEIGHT)),
        scaled(s.vx, f64::from(SPEED_X), f64::from(2 * SPEED_X)),
        scaled(s.vy, f64::from(SPEED_Y), f64::from(2 * SPEED_Y)),
        scaled(fx(i32::from(s.lives)), 0.0, lives),
        if s.parked() { 1.0 } else { 0.0 },
        scaled(fx(i32::from(s.score)), 0.0, f64::from(MAX_SCORE)),
    ];
    for (slot, value) in v.iter_mut().zip(head) {
        *slot = value;
    }
    let mut k = 8;
    for row in 0..ROWS {
        for col in 0..COLS {
            if let Some(slot) = v.get_mut(k) {
                *slot = if s.has(row, col) { 1.0 } else { 0.0 };
            }
            k += 1;
        }
    }
    v
}

impl Environment for BrickWall {
    type State = BrickWallState;
    type Config = BrickWallConfig;
    type Obs = BrickWallObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &BrickWallConfig) -> Result<Space, SpaceError> {
        BrickWallObs::space(cfg)
    }

    fn action_space(_: &BrickWallConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    /// No draw: the serve draws when its wait ends.
    fn reset(cfg: &BrickWallConfig, _: &mut StepRng, s: &mut BrickWallState) {
        *s = BrickWallState {
            bricks: FULL,
            px: fx(PADDLE_START),
            lives: u8::try_from(cfg.lives).unwrap_or(1),
            ..BrickWallState::default()
        };
        s.park(SERVE_WAIT);
    }

    fn step(
        cfg: &BrickWallConfig,
        s: &mut BrickWallState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(agent) = AgentSlot::new(0, 1) else {
            return;
        };
        let action = actions.discrete(agent, 0).unwrap_or(STAY);
        let mut gained = 0u32;
        let mut over = false;
        for _ in 0..cfg.frameskip {
            let (p, terminal) = s.frame(action, rng);
            gained += u32::from(p);
            if terminal {
                over = true;
                break;
            }
        }
        out.set_reward(agent, f32::from(u16::try_from(gained).unwrap_or(u16::MAX)));
        if over {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &BrickWallConfig,
        s: &BrickWallState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, BrickWallObs>,
    ) {
        let Some(bytes) = out.leaf(0) else {
            return;
        };
        if cfg.obs == OBS_STATE {
            for (chunk, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(vector(cfg, s)) {
                chunk.copy_from_slice(&v.to_le_bytes());
            }
        } else {
            draw_pixels(s, bytes);
        }
    }
}

impl tessera_rl_core::Persist for BrickWall {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}

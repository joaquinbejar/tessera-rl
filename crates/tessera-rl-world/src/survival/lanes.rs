// SPDX-License-Identifier: MIT OR Apache-2.0

//! Survival-v1's agent-lane step (ADR-0100 decision 2): the order-free
//! per-agent work of each phase runs as lane kernels over groups of 8 agents
//! of the instance, and everything slot-ordered or scattered stays scalar.
//!
//! `Survival::step` (`mod.rs`) is the reference and is not edited
//! (ADR-0100 decision 2, team-lead's choice (a) for head 3). The scalar
//! remainder here is a copy of it; each copied block names the lines it was
//! copied from, and the whole-step differential and the drift mutants keep
//! the copy equal to the reference. Survival does not set
//! `Environment::AGENT_LANE` until ADR-0100's stop point passes (erratum 2).
//!
//! Lanes of agents at or past the agent count, and of dead agents, are
//! masked: they read only in-range indices (a masked lane's coordinates are
//! replaced by 0 before the gather, as the gather's contract requires) and
//! write back their old values.
//!
//! No `i32` lane operation here can wrap: positions are in `-1..=96`,
//! stocks and health in `-85..=130`, counters at most 4,097.

use core::num::NonZeroU32;

use tessera_rl_core::io::Actions;
use tessera_rl_core::{AgentLaneEnvironment, Bool8, EpisodeEnd, LANES, NoAdaptive, StepOut};
use tessera_rl_num::math::lane_int::I32Lanes;
use tessera_rl_num::rng::StepRng;
use tessera_rl_simd::{LaneKernel, Lanes, Resolved};

use super::{
    ATTACK, DRINK, FORAGE, FOREST, GRASS, MAX_AGENTS, MAX_HEALTH, MAX_SIZE, MAX_STOCK,
    RESPAWN_TRIES, STARVE, Survival, SurvivalConfig, SurvivalState, WATER, cells, slot,
};

/// The occupancy grid stores an agent index plus one in a `u8`.
const _: () = assert!(MAX_AGENTS < 255);
/// Every flat tile index `y * MAX_SIZE + x` fits an `i32` lane.
const STRIDE: i32 = 96;
const _: () = assert!(MAX_SIZE == 96);

/// A `[[u8; MAX_SIZE]; MAX_SIZE]` grid, `[y][x]`.
pub type Grid = [[u8; MAX_SIZE]; MAX_SIZE];

/// Agent index plus one on every tile with a live agent of `s`, 0 elsewhere;
/// the lowest index wins on a shared tile, as `SurvivalState::agent_at`'s
/// `find` does.
#[must_use]
pub fn occupancy(s: &SurvivalState) -> Grid {
    let mut grid = [[0u8; MAX_SIZE]; MAX_SIZE];
    for a in 0..s.agent_count() {
        if !s.is_alive(a) {
            continue;
        }
        let (x, y) = s.pos(a);
        if let (Ok(cx), Ok(cy)) = (usize::try_from(x), usize::try_from(y))
            && let Some(cell) = grid.get_mut(cy).and_then(|r| r.get_mut(cx))
            && *cell == 0
        {
            // `a < MAX_AGENTS < 255`.
            *cell = u8::try_from(a + 1).unwrap_or(u8::MAX);
        }
    }
    grid
}

/// Lanes `base..base + 8` of `v`.
fn window<T: Copy + Default>(v: &[T; MAX_AGENTS], base: usize) -> [T; LANES] {
    core::array::from_fn(|l| v.get(base + l).copied().unwrap_or_default())
}

/// Writes `w` into lanes `base..base + 8` of `v`, for the lanes `keep`
/// marks.
fn put<T: Copy>(v: &mut [T; MAX_AGENTS], base: usize, w: [T; LANES], keep: [bool; LANES]) {
    for (l, (value, k)) in w.into_iter().zip(keep).enumerate() {
        if k && let Some(e) = v.get_mut(base + l) {
            *e = value;
        }
    }
}

/// Which lanes of the group at `base` hold an agent alive in `s`.
fn alive_lanes(s: &SurvivalState, base: usize) -> [bool; LANES] {
    core::array::from_fn(|l| s.is_alive(base + l))
}

/// Which lanes of the group at `base` hold an agent (`< n`).
fn agent_lanes(n: usize, base: usize) -> [bool; LANES] {
    core::array::from_fn(|l| base + l < n)
}

/// One head of every agent, as the reference reads it (`head` in
/// `Survival::step`).
#[must_use]
pub fn heads(actions: &Actions<'_, ()>, n: usize, h: usize) -> [i32; MAX_AGENTS] {
    core::array::from_fn(|a| {
        let v = if a < n {
            slot(a)
                .and_then(|s| actions.discrete(s, h))
                .and_then(|v| u32::try_from(v).ok())
                .unwrap_or(0)
        } else {
            0
        };
        i32::try_from(v).unwrap_or(0)
    })
}

/// The cell one step from `(x, y)` in `dir` (1 N, 2 E, 3 S, 4 W, else
/// stay), per lane: `super::toward`.
#[expect(
    clippy::inline_always,
    reason = "called from LaneKernel::run, inside its target_feature context"
)]
#[inline(always)]
fn toward_lanes<L: I32Lanes>(x: L, y: L, dir: L) -> (L, L, L::Mask) {
    let one = L::splat(1);
    let north = dir.eq(L::splat(1));
    let east = dir.eq(L::splat(2));
    let south = dir.eq(L::splat(3));
    let west = dir.eq(L::splat(4));
    let tx = L::select(
        east,
        x.wrapping_add(one),
        L::select(west, x.wrapping_sub(one), x),
    );
    let ty = L::select(
        north,
        y.wrapping_sub(one),
        L::select(south, y.wrapping_add(one), y),
    );
    let moving = L::mask_or(L::mask_or(north, south), L::mask_or(east, west));
    (tx, ty, moving)
}

/// Whether `(x, y)` is on a map of side `side`, per lane.
#[expect(
    clippy::inline_always,
    reason = "called from LaneKernel::run, inside its target_feature context"
)]
#[inline(always)]
fn on_map<L: I32Lanes>(x: L, y: L, side: L) -> L::Mask {
    let zero = L::splat(0);
    L::mask_and(
        L::mask_and(x.ge(zero), y.ge(zero)),
        L::mask_and(x.lt(side), y.lt(side)),
    )
}

/// The flat index of `(x, y)` where `inside` holds, and 0 elsewhere, so
/// every lane's index is in range (the gather's contract).
#[expect(
    clippy::inline_always,
    reason = "called from LaneKernel::run, inside its target_feature context"
)]
#[inline(always)]
fn flat<L: I32Lanes>(x: L, y: L, inside: L::Mask) -> [i32; LANES] {
    let zero = L::splat(0);
    let (x, y) = (L::select(inside, x, zero), L::select(inside, y, zero));
    y.wrapping_mul(L::splat(STRIDE)).wrapping_add(x).to_array()
}

// --- Phase 1: move targets -------------------------------------------------

struct MoveKernel<'a> {
    x: [i16; LANES],
    y: [i16; LANES],
    alive: [bool; LANES],
    dir: [i32; LANES],
    side: i32,
    tiles: &'a [u8],
    occ: &'a [u8],
}

impl LaneKernel for MoveKernel<'_> {
    type Output = ([i16; LANES], [i16; LANES]);

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Self::Output {
        let (x, y) = (S::I32::from_i16(self.x), S::I32::from_i16(self.y));
        let (tx, ty, moving) = toward_lanes(x, y, S::I32::from_array(self.dir));
        let inside = on_map(tx, ty, S::I32::splat(self.side));
        let idx = flat(tx, ty, inside);
        let tile = S::I32::gather_u8(self.tiles, idx);
        let passable = S::I32::mask_or(
            tile.eq(S::I32::splat(i32::from(GRASS))),
            tile.eq(S::I32::splat(i32::from(FOREST))),
        );
        // T3: occupied at the start of the step, or not passable.
        let empty = S::I32::gather_u8(self.occ, idx).eq(S::I32::splat(0));
        let free = S::I32::mask_and(S::I32::mask_and(inside, passable), empty);
        let go = S::I32::mask_and(
            S::I32::mask_and(S::I32::mask_from_array(self.alive), moving),
            free,
        );
        // Targets are on the map or the start position: in `i16`'s range.
        (
            S::I32::select(go, tx, x).to_i16(),
            S::I32::select(go, ty, y).to_i16(),
        )
    }
}

/// Phase 1's lane part: each agent's move target from the start-of-step
/// state, before the T1 conflicts (the first loop of `Survival::step`'s
/// phase 1).
#[must_use]
pub fn move_targets(
    level: Resolved,
    start: &SurvivalState,
    dirs: &[i32; MAX_AGENTS],
) -> [(i16, i16); MAX_AGENTS] {
    let n = start.agent_count();
    let occ = occupancy(start);
    let side = i32::try_from(start.side()).unwrap_or(0);
    let mut target = [(0i16, 0i16); MAX_AGENTS];
    for base in (0..n).step_by(LANES) {
        let (tx, ty) = level.dispatch_lanes(MoveKernel {
            x: window(&start.x, base),
            y: window(&start.y, base),
            alive: alive_lanes(start, base),
            dir: window(dirs, base),
            side,
            tiles: start.tiles.as_flattened(),
            occ: occ.as_flattened(),
        });
        let keep = agent_lanes(n, base);
        for (l, k) in keep.into_iter().enumerate() {
            if k && let (Some(t), Some(&x), Some(&y)) =
                (target.get_mut(base + l), tx.get(l), ty.get(l))
            {
                *t = (x, y);
            }
        }
    }
    target
}

// --- Phase 2: attack victims -----------------------------------------------

struct AttackKernel<'a> {
    x: [i16; LANES],
    y: [i16; LANES],
    alive: [bool; LANES],
    act: [i32; LANES],
    side: i32,
    occ: &'a [u8],
}

impl LaneKernel for AttackKernel<'_> {
    type Output = [i32; LANES];

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Self::Output {
        let (x, y) = (S::I32::from_i16(self.x), S::I32::from_i16(self.y));
        let act = S::I32::from_array(self.act);
        let attack = S::I32::mask_and(act.ge(S::I32::splat(3)), act.le(S::I32::splat(6)));
        let (tx, ty, _) = toward_lanes(x, y, act.wrapping_sub(S::I32::splat(2)));
        let inside = on_map(tx, ty, S::I32::splat(self.side));
        let held = S::I32::gather_u8(self.occ, flat(tx, ty, inside));
        let hit = S::I32::mask_and(
            S::I32::mask_and(S::I32::mask_from_array(self.alive), attack),
            S::I32::mask_and(inside, held.gt(S::I32::splat(0))),
        );
        S::I32::select(hit, held.wrapping_sub(S::I32::splat(1)), S::I32::splat(-1)).to_array()
    }
}

/// Phase 2's lane part: the live agent each attacking agent hits on the
/// post-move positions, or -1 (the `agent_at` lookup of `Survival::step`'s
/// phase 2).
#[must_use]
pub fn attack_victims(
    level: Resolved,
    moved: &SurvivalState,
    acts: &[i32; MAX_AGENTS],
) -> [i32; MAX_AGENTS] {
    let n = moved.agent_count();
    let occ = occupancy(moved);
    let side = i32::try_from(moved.side()).unwrap_or(0);
    let mut victim = [-1i32; MAX_AGENTS];
    for base in (0..n).step_by(LANES) {
        let v = level.dispatch_lanes(AttackKernel {
            x: window(&moved.x, base),
            y: window(&moved.y, base),
            alive: alive_lanes(moved, base),
            act: window(acts, base),
            side,
            occ: occ.as_flattened(),
        });
        put(&mut victim, base, v, agent_lanes(n, base));
    }
    victim
}

// --- Phase 3: forage, drink, upkeep, damage --------------------------------

struct UpkeepKernel<'a> {
    x: [i16; LANES],
    y: [i16; LANES],
    alive: [bool; LANES],
    act: [i32; LANES],
    food: [i16; LANES],
    water: [i16; LANES],
    health: [i16; LANES],
    damage: [i16; LANES],
    side: i32,
    tiles: &'a [u8],
    regrow: &'a [u8],
}

impl LaneKernel for UpkeepKernel<'_> {
    type Output = ([i16; LANES], [i16; LANES], [i16; LANES], u8);

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Self::Output {
        let zero = S::I32::splat(0);
        let side = S::I32::splat(self.side);
        let (x, y) = (S::I32::from_i16(self.x), S::I32::from_i16(self.y));
        let alive = S::I32::mask_from_array(self.alive);
        let act = S::I32::from_array(self.act);
        let here = S::I32::mask_and(alive, on_map(x, y, side));
        let idx = flat(x, y, here);
        // `ready_forest`: a forest tile whose regrow timer is 0, both read
        // from the post-move state, before this step's countdown.
        let ready = S::I32::mask_and(
            S::I32::mask_and(
                here,
                S::I32::gather_u8(self.tiles, idx).eq(S::I32::splat(i32::from(FOREST))),
            ),
            S::I32::gather_u8(self.regrow, idx).eq(zero),
        );
        // `by_water`: a water tile among the 4 neighbours on the map (for a
        // live agent; `SurvivalState::by_water` reads the neighbours only).
        let mut water_near = S::I32::mask_from_bits(0);
        for d in 1..=4 {
            let (nx, ny, _) = toward_lanes(x, y, S::I32::splat(d));
            let inside = S::I32::mask_and(alive, on_map(nx, ny, side));
            let wet = S::I32::gather_u8(self.tiles, flat(nx, ny, inside))
                .eq(S::I32::splat(i32::from(WATER)));
            water_near = S::I32::mask_or(water_near, S::I32::mask_and(inside, wet));
        }
        let forage = S::I32::mask_and(ready, act.eq(S::I32::splat(1)));
        let drink = S::I32::mask_and(water_near, act.eq(S::I32::splat(2)));
        let (food0, water0, health0) = (
            S::I32::from_i16(self.food),
            S::I32::from_i16(self.water),
            S::I32::from_i16(self.health),
        );
        let stock = S::I32::splat(i32::from(MAX_STOCK));
        let food = S::I32::select(
            forage,
            food0
                .wrapping_add(S::I32::splat(i32::from(FORAGE)))
                .min(stock),
            food0,
        );
        let water = S::I32::select(
            drink,
            water0
                .wrapping_add(S::I32::splat(i32::from(DRINK)))
                .min(stock),
            water0,
        );
        let one = S::I32::splat(1);
        let food = food.wrapping_sub(one).max(zero);
        let water = water.wrapping_sub(one).max(zero);
        let starving = S::I32::mask_or(food.eq(zero), water.eq(zero));
        let health = health0
            .wrapping_sub(S::I32::select(
                starving,
                S::I32::splat(i32::from(STARVE)),
                zero,
            ))
            .wrapping_sub(S::I32::from_i16(self.damage));
        // Game rule: health stays in 0..=MAX_HEALTH.
        let health = health.max(zero).min(S::I32::splat(i32::from(MAX_HEALTH)));
        // Dead and absent lanes keep their values; every lane is then in
        // `i16`'s range.
        (
            S::I32::select(alive, food, food0).to_i16(),
            S::I32::select(alive, water, water0).to_i16(),
            S::I32::select(alive, health, health0).to_i16(),
            S::I32::mask_bits(forage),
        )
    }
}

/// Phase 3's lane result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Upkeep {
    /// Food after the step.
    pub food: [i16; MAX_AGENTS],
    /// Water after the step.
    pub water: [i16; MAX_AGENTS],
    /// Health after the step.
    pub health: [i16; MAX_AGENTS],
    /// Which agents foraged (their tile's regrow timer restarts).
    pub forage: [bool; MAX_AGENTS],
}

/// Phase 3's lane part on the post-move state `moved`: forage, drink,
/// upkeep and damage per agent (the agent loop of `Survival::step`'s phase
/// 3, without the regrow write).
#[must_use]
pub fn upkeep(
    level: Resolved,
    moved: &SurvivalState,
    acts: &[i32; MAX_AGENTS],
    damage: &[i16; MAX_AGENTS],
) -> Upkeep {
    let n = moved.agent_count();
    let side = i32::try_from(moved.side()).unwrap_or(0);
    let mut out = Upkeep {
        food: moved.food,
        water: moved.water,
        health: moved.health,
        forage: [false; MAX_AGENTS],
    };
    for base in (0..n).step_by(LANES) {
        let (food, water, health, forage) = level.dispatch_lanes(UpkeepKernel {
            x: window(&moved.x, base),
            y: window(&moved.y, base),
            alive: alive_lanes(moved, base),
            act: window(acts, base),
            food: window(&moved.food, base),
            water: window(&moved.water, base),
            health: window(&moved.health, base),
            damage: window(damage, base),
            side,
            tiles: moved.tiles.as_flattened(),
            regrow: moved.regrow.as_flattened(),
        });
        let keep = agent_lanes(n, base);
        put(&mut out.food, base, food, keep);
        put(&mut out.water, base, water, keep);
        put(&mut out.health, base, health, keep);
        put(
            &mut out.forage,
            base,
            core::array::from_fn(|l| forage >> l & 1 == 1),
            keep,
        );
    }
    out
}

// --- Phase 4: deaths, ages ---------------------------------------------------

struct DeathKernel {
    alive: [bool; LANES],
    health: [i16; LANES],
}

impl LaneKernel for DeathKernel {
    type Output = u8;

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> u8 {
        let dies = S::I32::mask_and(
            S::I32::mask_from_array(self.alive),
            S::I32::from_i16(self.health).le(S::I32::splat(0)),
        );
        S::I32::mask_bits(dies)
    }
}

struct AgeKernel {
    alive_now: [bool; LANES],
    alive_before: [bool; LANES],
    present: [bool; LANES],
    age: [u16; LANES],
    dead_for: [u16; LANES],
}

impl LaneKernel for AgeKernel {
    type Output = ([u16; LANES], [u16; LANES]);

    #[expect(
        clippy::inline_always,
        reason = "LaneKernel::run must inline into the target_feature context"
    )]
    #[inline(always)]
    fn run<S: Lanes>(self) -> Self::Output {
        let one = S::I32::splat(1);
        let present = S::I32::mask_from_array(self.present);
        let alive = S::I32::mask_and(present, S::I32::mask_from_array(self.alive_now));
        let dead_before = S::I32::mask_and(
            S::I32::mask_and(
                present,
                S::I32::mask_not(S::I32::mask_from_array(self.alive_now)),
            ),
            S::I32::mask_not(S::I32::mask_from_array(self.alive_before)),
        );
        let (age, dead_for) = (S::I32::from_u16(self.age), S::I32::from_u16(self.dead_for));
        // Age and `dead_for` are at most 4,096 (`Survival::step`).
        (
            S::I32::select(alive, age.wrapping_add(one), age).to_u16(),
            S::I32::select(dead_before, dead_for.wrapping_add(one), dead_for).to_u16(),
        )
    }
}

/// Which agents alive in `moved` die at the health in `health` (phase 4's
/// death test).
#[must_use]
pub fn deaths(
    level: Resolved,
    moved: &SurvivalState,
    health: &[i16; MAX_AGENTS],
) -> [bool; MAX_AGENTS] {
    let n = moved.agent_count();
    let mut dies = [false; MAX_AGENTS];
    for base in (0..n).step_by(LANES) {
        let bits = level.dispatch_lanes(DeathKernel {
            alive: alive_lanes(moved, base),
            health: window(health, base),
        });
        put(
            &mut dies,
            base,
            core::array::from_fn(|l| bits >> l & 1 == 1),
            agent_lanes(n, base),
        );
    }
    dies
}

/// Phase 4's age and `dead_for` counts, after the deaths and before the
/// respawns: age grows for agents alive in `now`, `dead_for` for agents dead
/// in both `before` (the post-move state) and `now`.
pub fn ages(level: Resolved, before: &SurvivalState, now: &mut SurvivalState) {
    let n = now.agent_count();
    for base in (0..n).step_by(LANES) {
        let (age, dead_for) = level.dispatch_lanes(AgeKernel {
            alive_now: alive_lanes(now, base),
            alive_before: alive_lanes(before, base),
            present: agent_lanes(n, base),
            age: window(&now.age, base),
            dead_for: window(&now.dead_for, base),
        });
        let keep = agent_lanes(n, base);
        put(&mut now.age, base, age, keep);
        put(&mut now.dead_for, base, dead_for, keep);
    }
}

// --- The step ---------------------------------------------------------------

/// Survival's step with its order-free per-agent phases at `level`,
/// bit-identical to `Survival::step`.
pub fn step(
    cfg: &SurvivalConfig,
    state: &mut SurvivalState,
    actions: &Actions<'_, ()>,
    rng: &mut StepRng,
    out: &mut StepOut<'_>,
    level: Resolved,
) {
    let n = state.agent_count();
    let (moves, acts) = (heads(actions, n, 0), heads(actions, n, 1));
    let start = *state;
    // 1. Moves: the targets in lanes, then T1 (copied from `mod.rs` 566-580).
    let target = move_targets(level, &start, &moves);
    for a in 0..n {
        let Some(&t) = target.get(a) else { continue };
        if !start.is_alive(a) || t == start.pos(a) {
            continue;
        }
        // T1: another live mover with the same target.
        let contested = (0..n)
            .any(|b| b != a && start.is_alive(b) && target.get(b) == Some(&t) && t != start.pos(b));
        let (x, y) = if contested { start.pos(a) } else { t };
        if let (Some(sx), Some(sy)) = (state.x.get_mut(a), state.y.get_mut(a)) {
            (*sx, *sy) = (x, y);
        }
    }
    // 2. Attacks: the victims in lanes, then the sums and credits (the
    // scatter of `mod.rs` 591-597, in agent order).
    let moved = *state;
    let victim = attack_victims(level, &moved, &acts);
    let mut damage = [0i16; MAX_AGENTS];
    let mut hit_by = [[false; MAX_AGENTS]; MAX_AGENTS];
    for (a, &v) in victim.iter().enumerate().take(n) {
        let Ok(v) = usize::try_from(v) else { continue };
        if let Some(d) = damage.get_mut(v) {
            *d += ATTACK;
        }
        if let Some(h) = hit_by.get_mut(v).and_then(|r| r.get_mut(a)) {
            *h = true;
        }
    }
    upkeep_phase(cfg, state, &moved, &acts, &damage, level);
    deaths_phase(cfg, state, &moved, &hit_by, rng, out, level);
    // Rewards and the end (copied from `mod.rs` 707-719).
    for a in 0..n {
        if let Some(s) = slot(a) {
            out.set_reward(s, if state.is_alive(a) { 1.0 } else { 0.0 });
        }
    }
    state.step += 1;
    let respawn = u16::try_from(cfg.respawn).unwrap_or(0);
    let none_alive = (0..n).all(|a| !state.is_alive(a));
    if none_alive && respawn == 0 {
        out.end_episode(EpisodeEnd::Terminated);
    } else if u32::from(state.step) >= cfg.steps {
        out.end_episode(EpisodeEnd::Truncated);
    }
}

/// Phase 3: the regrow countdown and the forage writes stay scalar
/// (copied from `mod.rs` 603-626); the per-agent updates run in lanes.
fn upkeep_phase(
    cfg: &SurvivalConfig,
    state: &mut SurvivalState,
    moved: &SurvivalState,
    acts: &[i32; MAX_AGENTS],
    damage: &[i16; MAX_AGENTS],
    level: Resolved,
) {
    let regrow = u8::try_from(cfg.forest_regrow).unwrap_or(1);
    // Timers count down before this step's foraging sets new ones.
    let side = state.side();
    for row in state.regrow.iter_mut().take(side) {
        for t in row.iter_mut().take(side) {
            if *t > 0 {
                *t -= 1;
            }
        }
    }
    let up = upkeep(level, moved, acts, damage);
    for a in 0..state.agent_count() {
        if !moved.is_alive(a) {
            continue;
        }
        if up.forage.get(a) == Some(&true) {
            let (x, y) = moved.pos(a);
            if let (Ok(cx), Ok(cy)) = (usize::try_from(x), usize::try_from(y))
                && let Some(t) = state.regrow.get_mut(cy).and_then(|r| r.get_mut(cx))
            {
                *t = regrow;
            }
        }
    }
    state.food = up.food;
    state.water = up.water;
    state.health = up.health;
}

/// Phase 4: the death test and the age counts run in lanes; the events,
/// kill counts and the T5 respawn stay scalar, in slot order (copied from
/// `mod.rs` 650-705).
fn deaths_phase(
    cfg: &SurvivalConfig,
    state: &mut SurvivalState,
    moved: &SurvivalState,
    hit_by: &[[bool; MAX_AGENTS]; MAX_AGENTS],
    rng: &mut StepRng,
    out: &mut StepOut<'_>,
    level: Resolved,
) {
    let n = state.agent_count();
    let dies = deaths(level, moved, &state.health);
    for v in 0..n {
        if dies.get(v) != Some(&true) {
            continue;
        }
        if let Some(f) = state.alive.get_mut(v) {
            *f = Bool8::FALSE;
        }
        if let Some(s) = slot(v) {
            out.agent_died(s);
        }
        for a in 0..n {
            // At most 4 kills a step for 4,096 steps: no overflow.
            if hit_by.get(v).and_then(|r| r.get(a)) == Some(&true)
                && let Some(k) = state.kills.get_mut(a)
            {
                *k += 1;
            }
        }
    }
    ages(level, moved, state);
    let respawn = u16::try_from(cfg.respawn).unwrap_or(0);
    let view = i16::try_from(cfg.view).unwrap_or(2);
    for a in 0..n {
        if state.is_alive(a) {
            continue;
        }
        let dead_for = state.dead_for.get(a).copied().unwrap_or(0);
        if respawn > 0 && dead_for >= respawn {
            // StepRng only (ADR-0007): this agent's candidates, drawn in
            // slot order; T5 orders the placement as well.
            let cells = NonZeroU32::new(cells(state.side())).unwrap_or(NonZeroU32::MIN);
            let mut draws = [0u64; RESPAWN_TRIES];
            for d in &mut draws {
                *d = u64::from(rng.below_u32(cells));
            }
            if let Some((x, y)) = state.spawn_tile(view, |k| {
                draws
                    .get(usize::try_from(k).unwrap_or(0))
                    .copied()
                    .unwrap_or(0)
            }) {
                state.spawn(a, x, y);
                if let Some(s) = slot(a) {
                    out.agent_spawned(s);
                }
            }
        }
    }
}

impl AgentLaneEnvironment for Survival {
    fn step_agent_lanes(
        cfg: &SurvivalConfig,
        _: &NoAdaptive,
        state: &mut SurvivalState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
        level: Resolved,
    ) {
        step(cfg, state, actions, rng, out, level);
    }
}

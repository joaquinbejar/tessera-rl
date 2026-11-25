// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baseline (SPEC 16.1; PLAN-r1 stage 3): a forager and
//! fighter that acts from its own view only (`InfoHeuristic`), so it is not
//! omniscient. In order:
//!
//! 1. **Fight:** with health above `BRAVE`, attack the first live neighbour
//!    in the order N, E, S, W.
//! 2. **Drink:** with water at most `THIRSTY`, drink when next to water.
//! 3. **Forage:** with food at most `HUNGRY`, forage when on a ready forest.
//! 4. **Seek:** step towards the nearest cell in the window that serves the
//!    lower stock: water (to stand next to it) or a ready forest (to stand
//!    on it). Ties go to the first cell in row-major window order, and the
//!    step is the first of N, E, S, W that brings it closer onto a free
//!    tile.
//! 5. **Wander:** otherwise a uniform free move (stay included), from the
//!    heuristic's `StepRng`, and no act.
//!
//! A free tile is one the window shows passable with no agent on it, and
//! every act above checks what the mask checks, so the baseline only plays
//! legal actions. A dead agent stays and does nothing.

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, ImperfectInformation, InfoHeuristic};
use tessera_rl_num::rng::StepRng;

use super::{
    FOREST, MAX_VIEW, Survival, SurvivalConfig, SurvivalInfo, SurvivalState, WATER, map, toward,
};

/// Health above which the baseline attacks a neighbour.
pub const BRAVE: i32 = 50;
/// Water at or below which the baseline drinks.
pub const THIRSTY: i32 = 60;
/// Food at or below which the baseline forages.
pub const HUNGRY: i32 = 60;

/// A view of the information state: window cells relative to the agent.
struct View<'a> {
    info: &'a SurvivalInfo,
    radius: i32,
    width: usize,
}

impl View<'_> {
    /// Channel `c` of the cell at `(dx, dy)` from the agent, or `None`
    /// outside the window.
    fn cell(&self, c: usize, dx: i32, dy: i32) -> Option<u8> {
        if dx.abs() > self.radius || dy.abs() > self.radius {
            return None;
        }
        let col = usize::try_from(dx + self.radius).ok()?;
        let row = usize::try_from(dy + self.radius).ok()?;
        let at = c * self.width * self.width + row * self.width + col;
        self.info.window.get(at).copied()
    }

    /// Whether the agent can step onto `(dx, dy)`: passable, no agent.
    fn free(&self, dx: i32, dy: i32) -> bool {
        self.cell(0, dx, dy).is_some_and(map::passable) && self.cell(2, dx, dy) == Some(0)
    }

    fn agent(&self, dx: i32, dy: i32) -> bool {
        self.cell(2, dx, dy) == Some(1)
    }
}

/// The neighbour offsets in the order N, E, S, W, with their direction.
const DIRS: [(u32, i32, i32); 4] = [(1, 0, -1), (2, 1, 0), (3, 0, 1), (4, -1, 0)];

/// The baseline's `(move, act)` from `info`.
#[must_use]
pub fn act(cfg: &SurvivalConfig, info: &SurvivalInfo, rng: &mut StepRng) -> (u32, u32) {
    let [health, food, water, ..] = info.own;
    if health <= 0 {
        return (0, 0);
    }
    let radius = i32::try_from(cfg.view)
        .unwrap_or(2)
        .clamp(2, i32::try_from(MAX_VIEW).unwrap_or(7));
    let view = View {
        info,
        radius,
        width: usize::try_from(2 * radius + 1).unwrap_or(5),
    };
    // 1. Fight.
    if health > BRAVE
        && let Some(&(d, _, _)) = DIRS.iter().find(|&&(_, dx, dy)| view.agent(dx, dy))
    {
        return (0, 2 + d);
    }
    let by_water = DIRS
        .iter()
        .any(|&(_, dx, dy)| view.cell(0, dx, dy) == Some(WATER));
    let on_ready_forest = view.cell(0, 0, 0) == Some(FOREST) && view.cell(1, 0, 0) == Some(1);
    // 2. Drink, 3. forage.
    if water <= THIRSTY && by_water {
        return (0, 2);
    }
    if food <= HUNGRY && on_ready_forest {
        return (0, 1);
    }
    // 4. Seek the resource of the lower stock.
    let want_water = water <= food;
    let mut best: Option<(i32, i32, i32)> = None;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let useful = if want_water {
                view.cell(0, dx, dy) == Some(WATER)
            } else {
                view.cell(0, dx, dy) == Some(FOREST) && view.cell(1, dx, dy) == Some(1)
            };
            let d = dx.abs() + dy.abs();
            if useful && best.is_none_or(|(b, _, _)| d < b) {
                best = Some((d, dx, dy));
            }
        }
    }
    if let Some((d, tx, ty)) = best
        && let Some(&(dir, _, _)) = DIRS
            .iter()
            .find(|&&(_, dx, dy)| (tx - dx).abs() + (ty - dy).abs() < d && view.free(dx, dy))
    {
        return (dir, 0);
    }
    // 5. Wander: a uniform free move, stay included.
    let mut moves = [0u32; 5];
    let mut n = 1;
    for &(dir, dx, dy) in &DIRS {
        if view.free(dx, dy)
            && let Some(m) = moves.get_mut(n)
        {
            *m = dir;
            n += 1;
        }
    }
    let pick =
        rng.below_u32(NonZeroU32::new(u32::try_from(n).unwrap_or(1)).unwrap_or(NonZeroU32::MIN));
    (
        moves
            .get(usize::try_from(pick).unwrap_or(0))
            .copied()
            .unwrap_or(0),
        0,
    )
}

fn write(out: &mut ActionWriter<'_>, (m, a): (u32, u32)) {
    out.set_discrete(0, i32::try_from(m).unwrap_or(0));
    out.set_discrete(1, i32::try_from(a).unwrap_or(0));
}

/// The baseline from the full state, through the agent's own view
/// (ADR-0079 until `InfoHeuristic` is used directly).
impl Heuristic for Survival {
    fn act(
        cfg: &SurvivalConfig,
        state: &SurvivalState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let info = Self::info_state(cfg, state, agent);
        write(out, act(cfg, &info, rng));
    }
}

/// The baseline from the information state (ADR-0079 item 4.2).
impl InfoHeuristic for Survival {
    fn act_info(
        cfg: &SurvivalConfig,
        info: &SurvivalInfo,
        _: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        write(out, act(cfg, info, rng));
    }
}

// The window's directions are those of `toward`.
const _: () = {
    let (x, y) = toward(0, 0, 1);
    assert!(x == 0 && y == -1);
};

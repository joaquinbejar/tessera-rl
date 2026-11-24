// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GridKeyDoor-v1` scenes (ADR-0083): the room, the wall column with its
//! door, the key, the goal and each agent with its facing.

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};

use super::{KeyDoor, KeyDoorConfig, KeyDoorState, NO_HOLDER, step_of};

/// Cell side in pixels.
const CELL: i32 = 32;

const BACKGROUND: Rgba = Rgba::rgb(20, 20, 20);
const WALL: Rgba = Rgba::rgb(128, 128, 128);
const DOOR_CLOSED: Rgba = Rgba::rgb(150, 90, 30);
const DOOR_OPEN: Rgba = Rgba::rgb(60, 45, 30);
const KEY: Rgba = Rgba::rgb(240, 200, 40);
const GOAL: Rgba = Rgba::rgb(40, 200, 80);
const AGENTS: [Rgba; 4] = [
    Rgba::rgb(230, 60, 50),
    Rgba::rgb(60, 120, 230),
    Rgba::rgb(200, 80, 200),
    Rgba::rgb(240, 140, 40),
];
const FACING: Rgba = Rgba::rgb(255, 255, 255);

/// The pixel offset of `k` cells plus `frac` sixteenths of a cell; out of
/// range values clamp (documented: they lie far outside any canvas).
fn at(k: i32, frac: i32) -> Px16 {
    k.checked_mul(16)
        .and_then(|v| v.checked_add(frac))
        .and_then(|v| v.checked_mul(CELL))
        .map_or(Px16(i32::MAX), Px16)
}

/// Fills cell `(col, row)`.
fn cell(scene: &mut SceneWriter<'_>, col: i32, row: i32, fill: Rgba) {
    scene.rect(at(col, 0), at(row, 0), at(col, 16), at(row, 16), fill);
}

impl Render for KeyDoor {
    const MAX_PRIMITIVES: usize = 16;

    fn canvas(cfg: &KeyDoorConfig) -> (u16, u16) {
        let side = cfg
            .size
            .checked_mul(32)
            .and_then(|v| u16::try_from(v).ok())
            .unwrap_or(512);
        (side, side)
    }

    fn draw(cfg: &KeyDoorConfig, state: &KeyDoorState, scene: &mut SceneWriter<'_>) {
        let n = i32::try_from(cfg.size).unwrap_or(16);
        scene.rect(Px16(0), Px16(0), at(n, 0), at(n, 0), BACKGROUND);
        let wall = i32::from(state.wall_col);
        scene.rect(at(wall, 0), Px16(0), at(wall, 16), at(n, 0), WALL);
        let door = if state.door_open.get() {
            DOOR_OPEN
        } else {
            DOOR_CLOSED
        };
        cell(scene, wall, i32::from(state.door_row), door);
        cell(
            scene,
            i32::from(state.goal[0]),
            i32::from(state.goal[1]),
            GOAL,
        );
        if state.key_holder == NO_HOLDER {
            let (c, r) = (i32::from(state.key[0]), i32::from(state.key[1]));
            scene.rect(at(c, 4), at(r, 4), at(c, 12), at(r, 12), KEY);
        }
        let agents = usize::try_from(cfg.agents).unwrap_or(0);
        for (slot, (pos, colour)) in state.pos.iter().zip(AGENTS).enumerate().take(agents) {
            if state.is_finished(slot) {
                continue;
            }
            let (c, r) = (i32::from(pos[0]), i32::from(pos[1]));
            scene.circle(at(c, 8), at(r, 8), Px16::px(CELL * 3 / 8), colour);
            let facing = state.facing.get(slot).copied().unwrap_or(0);
            let (dc, dr) = step_of(facing);
            let tip = (at(c, 8 + 6 * dc), at(r, 8 + 6 * dr));
            scene.line((at(c, 8), at(r, 8)), tip, Px16::px(4), FACING);
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::render::{Prim, Px16, Rgba, SceneWriter};
    use tessera_rl_core::{Environment, Render};
    use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
    use tessera_rl_num::rng::{RngAuthority, StepRng};

    use super::{KeyDoor, KeyDoorConfig, KeyDoorState};

    #[test]
    fn test_keydoor_draw_reset_state_starts_with_the_background() {
        let cfg = KeyDoorConfig::default();
        let identity = RngIdentity::new(Seed::new(7), LogicalId::new(0));
        let mut rng = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
        let mut state = KeyDoorState::default();
        KeyDoor::reset(&cfg, &mut rng, &mut state);
        let blank = Prim::Circle {
            cx: Px16(0),
            cy: Px16(0),
            r: Px16(0),
            fill: Rgba::default(),
        };
        let mut buf = vec![blank; KeyDoor::MAX_PRIMITIVES];
        let mut scene = SceneWriter::new(&mut buf);
        KeyDoor::draw(&cfg, &state, &mut scene);
        let (w, h) = KeyDoor::canvas(&cfg);
        assert!(!scene.overflowed());
        assert!(scene.prims().len() > 1);
        assert!(matches!(
            scene.prims().first(),
            Some(Prim::Rect { x0: Px16(0), y0: Px16(0), x1, y1, .. })
                if *x1 == Px16::px(i32::from(w)) && *y1 == Px16::px(i32::from(h))
        ));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation of snake.

use tessera_rl_core::{HeaderView, InvariantViolation, Phase, Validate};

use super::{
    Cells, MAX_AGENTS, NO_FOOD, RING, SPAWN_LEN, Snake, SnakeConfig, SnakeState, neighbour,
    opposite,
};

/// The agent's slot holds exactly one connected, private body: a length in
/// range, cells on the board and each next to the one before, none shared, a
/// direction leading from the neck to the head, and zero outside the body.
fn check_body(
    cfg: &SnakeConfig,
    s: &SnakeState,
    k: usize,
    seen: &mut Cells,
) -> Result<(), InvariantViolation> {
    let invalid = |code| Err(InvariantViolation { code });
    let (Some(&len), Some(&head), Some(&facing), Some(ring)) =
        (s.len.get(k), s.head.get(k), s.facing.get(k), s.body.get(k))
    else {
        return invalid("a slot out of range");
    };
    if len < SPAWN_LEN || u32::from(len) > cfg.cells() {
        return invalid("a body length out of range");
    }
    if usize::from(head) >= RING {
        return invalid("a head cursor out of range");
    }
    if facing > 3 {
        return invalid("a facing out of range");
    }
    let mut before = None;
    for cell in s.cells_of(k) {
        if u32::from(cell) >= cfg.cells() {
            return invalid("a body cell off the board");
        }
        if seen.has(cell) {
            return invalid("two body segments on one cell");
        }
        seen.add(cell);
        if let Some(prior) = before {
            let joined = (0..4).any(|d| neighbour(cfg.size, prior, d) == Some(cell));
            if !joined {
                return invalid("a body that is not connected");
            }
        } else {
            let neck = s.cells_of(k).nth(1);
            if neck.is_none() || neighbour(cfg.size, cell, opposite(facing)) != neck {
                return invalid("a facing that does not lead from the neck");
            }
        }
        before = Some(cell);
    }
    let (head, len) = (usize::from(head), usize::from(len));
    for (at, &cell) in ring.iter().enumerate() {
        let inside = (head + RING - at) % RING < len;
        if !inside && cell != 0 {
            return invalid("a value in an unused ring cell");
        }
    }
    Ok(())
}

/// Imported states: bodies as `check_body` says; a dead or unused slot all
/// zero; food on the board, off every body; no food exactly on a full board.
/// These are necessary conditions, not a proof that a game reaches the state.
impl Validate for Snake {
    fn validate(cfg: &SnakeConfig, s: &SnakeState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let mut seen = Cells::default();
        for k in 0..MAX_AGENTS {
            if s.is_alive(k) {
                if k >= cfg.agents as usize {
                    return invalid("an unused slot that is alive");
                }
                check_body(cfg, s, k, &mut seen)?;
            } else {
                let empty = s.len.get(k) == Some(&0)
                    && s.head.get(k) == Some(&0)
                    && s.facing.get(k) == Some(&0)
                    && s.body.get(k).is_some_and(|r| r.iter().all(|&c| c == 0));
                if !empty {
                    return invalid("a value in a dead or unused slot");
                }
            }
        }
        let cells = cfg.cells();
        let full = (0..u16::try_from(cells).unwrap_or(0)).all(|c| seen.has(c));
        if s.food == NO_FOOD {
            if !full {
                return invalid("no food on a board with free cells");
            }
        } else if u32::from(s.food) >= cells {
            return invalid("food off the board");
        } else if seen.has(s.food) {
            return invalid("food on a body");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, a slot is alive in the header exactly when it is in
    /// the state; a running episode has an agent and food.
    fn validate_record(
        cfg: &SnakeConfig,
        header: &HeaderView<'_>,
        s: &SnakeState,
    ) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != SnakeState::default() {
                return invalid("a state other than the initial value before the first reset");
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        for slot in 0..header.agents() {
            let Some(view) = header.agent(slot) else {
                continue;
            };
            if slot < cfg.agents as usize {
                if view.alive != s.is_alive(slot) {
                    return invalid("a used slot alive in only one of header and state");
                }
                // Every live agent acts each step; the engine would drop the
                // action of a live agent marked as not acting.
                if view.acting != view.alive {
                    return invalid("a used slot acting other than exactly when alive");
                }
            } else if view.alive || view.acting {
                return invalid("an unused slot that is alive or acting");
            }
        }
        if phase == Phase::Active {
            if !s.alive.iter().any(|a| a.get()) {
                return invalid("a running episode with every agent dead");
            }
            if s.food == NO_FOOD {
                return invalid("a running episode with no food");
            }
        }
        Ok(())
    }
}

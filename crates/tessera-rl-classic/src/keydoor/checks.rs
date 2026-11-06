// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation and the scripted baseline of key and door.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, HeaderView, Heuristic, InvariantViolation, Phase, Validate};
use tessera_rl_num::rng::StepRng;

use super::{
    FORWARD, INTERACT, KeyDoor, KeyDoorConfig, KeyDoorState, LEFT, MAX_AGENTS, NO_HOLDER, RIGHT,
    WAIT, step_of,
};

/// Imported states: a wall, door, key and goal in their ranges; agents on
/// floor cells, the door only when open, the right side only through it, not
/// on the key on the floor, sharing a cell only on the goal; facings in range;
/// finished flags that say who stands on the goal; a holder among the agents,
/// and an open door only with one; unused slots zero. These are necessary
/// conditions, not a proof that a game reaches the state.
impl Validate for KeyDoor {
    fn validate(cfg: &KeyDoorConfig, s: &KeyDoorState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let n = cfg.size;
        let wall = u32::from(s.wall_col);
        if !(2..n - 2).contains(&wall) || u32::from(s.door_row) >= n {
            return invalid("a wall or door out of range");
        }
        let key_col = u32::from(s.key[0]);
        if key_col >= wall || u32::from(s.key[1]) >= n {
            return invalid("a key off the left side");
        }
        let goal_col = u32::from(s.goal[0]);
        if goal_col <= wall || goal_col >= n || u32::from(s.goal[1]) >= n {
            return invalid("a goal off the right side");
        }
        let agents = cfg.agents as usize;
        let holder_ok = s.key_holder == NO_HOLDER || usize::from(s.key_holder) < agents;
        if !holder_ok {
            return invalid("a holder that is not an agent");
        }
        if s.door_open.get() && s.key_holder == NO_HOLDER {
            return invalid("an open door that nobody had the key for");
        }
        for k in agents..MAX_AGENTS {
            let empty =
                s.pos.get(k) == Some(&[0, 0]) && s.facing.get(k) == Some(&0) && !s.is_finished(k);
            if !empty {
                return invalid("a value in an unused slot");
            }
        }
        for k in 0..agents {
            let (Some(&[col, row]), Some(&f)) = (s.pos.get(k), s.facing.get(k)) else {
                return invalid("a slot out of range");
            };
            if u32::from(col) >= n || u32::from(row) >= n {
                return invalid("an agent off the room");
            }
            if f > 3 {
                return invalid("a facing out of range");
            }
            if col == s.wall_col && row != s.door_row {
                return invalid("an agent inside the wall");
            }
            if col >= s.wall_col && !s.door_open.get() {
                return invalid("an agent past a closed door");
            }
            if s.key_at(i32::from(col), i32::from(row)) {
                return invalid("an agent on the key");
            }
            let on_goal = [col, row] == s.goal;
            if s.is_finished(k) != on_goal {
                return invalid("a finished flag that disagrees with the goal");
            }
            let shared = (0..k).any(|j| s.pos.get(j) == Some(&[col, row]));
            if shared && !on_goal {
                return invalid("two agents on one cell off the goal");
            }
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, the configured slots are alive and a running episode
    /// has someone left to finish.
    fn validate_record(
        cfg: &KeyDoorConfig,
        header: &HeaderView<'_>,
        s: &KeyDoorState,
    ) -> Result<(), InvariantViolation> {
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != KeyDoorState::default() {
                return Err(InvariantViolation {
                    code: "a state other than the initial value before the first reset",
                });
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        for slot in 0..header.agents() {
            let Some(view) = header.agent(slot) else {
                continue;
            };
            if slot < cfg.agents as usize {
                if !view.alive {
                    return Err(InvariantViolation {
                        code: "an agent that is not alive",
                    });
                }
            } else if view.alive || view.acting {
                return Err(InvariantViolation {
                    code: "an unused slot that is alive or acting",
                });
            }
        }
        if phase == Phase::Active && s.all_finished(cfg) {
            return Err(InvariantViolation {
                code: "a running episode with everyone finished",
            });
        }
        Ok(())
    }
}

const CELLS: usize = 16 * 16;

/// Steps to the nearest of `targets` from every cell, walking on floor and
/// through the door if open, around the key on the floor; other agents are
/// ignored. Unreachable is `u16::MAX`. Stack only.
#[expect(
    clippy::many_single_char_names,
    reason = "cell coordinates and the search table"
)]
fn distances(cfg: &KeyDoorConfig, s: &KeyDoorState, targets: &[(i32, i32)]) -> [u16; CELLS] {
    let n = cfg.size.cast_signed();
    let index = |c: i32, r: i32| usize::try_from(r * 16 + c).unwrap_or(0);
    let mut d = [u16::MAX; CELLS];
    for &(c, r) in targets {
        if c >= 0
            && let Some(v) = d.get_mut(index(c, r))
        {
            *v = 0;
        }
    }
    let floor = |c: i32, r: i32| !occupied_only(cfg, s, c, r);
    loop {
        let mut changed = false;
        for r in 0..n {
            for c in 0..n {
                if !floor(c, r) {
                    continue;
                }
                let best = (0u8..4)
                    .map(|f| {
                        let (dc, dr) = step_of(f);
                        let (x, y) = (c + dc, r + dr);
                        if (0..n).contains(&x) && (0..n).contains(&y) {
                            d.get(index(x, y)).copied().unwrap_or(u16::MAX)
                        } else {
                            u16::MAX
                        }
                    })
                    .min()
                    .unwrap_or(u16::MAX);
                let via = best.saturating_add(1);
                if let Some(v) = d.get_mut(index(c, r))
                    && via < *v
                {
                    *v = via;
                    changed = true;
                }
            }
        }
        if !changed {
            return d;
        }
    }
}

/// Whether the cell is impassable whoever stands on it: wall, closed door,
/// the key, or off the room. Agents do not count.
fn occupied_only(cfg: &KeyDoorConfig, s: &KeyDoorState, c: i32, r: i32) -> bool {
    let n = cfg.size.cast_signed();
    !(0..n).contains(&c)
        || !(0..n).contains(&r)
        || (c == i32::from(s.wall_col) && (r != i32::from(s.door_row) || !s.door_open.get()))
        || s.key_at(c, r)
}

/// Turns toward `want`, or the action `at_facing` once facing it.
fn toward(facing: u8, want: u8, at_facing: i32) -> i32 {
    match (want + 4 - facing) % 4 {
        0 => at_facing,
        3 => LEFT,
        _ => RIGHT,
    }
}

/// Scripted baseline (deterministic, one step): with no holder, walk to the
/// key and take it. The holder walks to the cell before the door and opens it.
/// The others wait, giving way to the holder. Once the door is open, all walk
/// to the goal, by any free neighbour nearer it. It can still deadlock: the
/// solved share is pinned by a test, and below one for several agents.
impl Heuristic for KeyDoor {
    fn act(
        cfg: &KeyDoorConfig,
        s: &KeyDoorState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, choose(cfg, s, agent.index()));
    }
}

fn choose(cfg: &KeyDoorConfig, s: &KeyDoorState, me: usize) -> i32 {
    if s.is_finished(me) {
        return WAIT;
    }
    let (Some(&[col, row]), Some(&facing)) = (s.pos.get(me), s.facing.get(me)) else {
        return WAIT;
    };
    let (col, row) = (i32::from(col), i32::from(row));
    let before_door = (i32::from(s.wall_col) - 1, i32::from(s.door_row));
    let east = 1;
    if s.door_open.get() {
        let goal = [(i32::from(s.goal[0]), i32::from(s.goal[1]))];
        return walk(cfg, s, me, facing, (col, row), &goal);
    }
    if s.key_holder == NO_HOLDER {
        for f in 0u8..4 {
            let (dc, dr) = step_of(f);
            if s.key_at(col + dc, row + dr) {
                return toward(facing, f, INTERACT);
            }
        }
        let (kc, kr) = (i32::from(s.key[0]), i32::from(s.key[1]));
        let beside = [(kc, kr - 1), (kc + 1, kr), (kc, kr + 1), (kc - 1, kr)];
        return walk(cfg, s, me, facing, (col, row), &beside);
    }
    if usize::from(s.key_holder) == me {
        if (col, row) == before_door {
            return toward(facing, east, INTERACT);
        }
        return walk(cfg, s, me, facing, (col, row), &[before_door]);
    }
    step_aside(cfg, s, me, facing, (col, row), before_door)
}

/// A waiting agent gives way: if it stands on the door's approach and the
/// holder is right behind it, or on the cell before the door, it moves to a
/// free cell no nearer the door. Otherwise it waits.
fn step_aside(
    cfg: &KeyDoorConfig,
    s: &KeyDoorState,
    me: usize,
    facing: u8,
    (col, row): (i32, i32),
    before_door: (i32, i32),
) -> i32 {
    let d = distances(cfg, s, &[before_door]);
    let at = |c: i32, r: i32| {
        usize::try_from(r * 16 + c)
            .ok()
            .and_then(|i| d.get(i).copied())
            .unwrap_or(u16::MAX)
    };
    let mine = at(col, row);
    let Some(&[hc, hr]) = s.pos.get(usize::from(s.key_holder)) else {
        return WAIT;
    };
    let (hc, hr) = (i32::from(hc), i32::from(hr));
    let in_the_way = (col - hc).abs() + (row - hr).abs() == 1 && mine < at(hc, hr);
    if mine != 0 && !in_the_way {
        return WAIT;
    }
    for f in 0u8..4 {
        let (dc, dr) = step_of(f);
        let (x, y) = (col + dc, row + dr);
        if s.can_enter(cfg, x, y, me) && at(x, y) >= mine {
            return toward(facing, f, FORWARD);
        }
    }
    WAIT
}

/// One step of the shortest walk to `targets`, or waiting if blocked or there.
#[expect(
    clippy::many_single_char_names,
    reason = "cell coordinates and the search table"
)]
fn walk(
    cfg: &KeyDoorConfig,
    s: &KeyDoorState,
    me: usize,
    facing: u8,
    (col, row): (i32, i32),
    targets: &[(i32, i32)],
) -> i32 {
    let n = cfg.size.cast_signed();
    let on_board: [(i32, i32); 4] = core::array::from_fn(|i| {
        targets
            .get(i)
            .copied()
            .filter(|&(c, r)| (0..n).contains(&c) && (0..n).contains(&r))
            .filter(|&(c, r)| !occupied_only(cfg, s, c, r))
            .unwrap_or((-1, -1))
    });
    // A target off the board indexes past every cell this search visits.
    let d = distances(cfg, s, &on_board);
    let at = |c: i32, r: i32| {
        usize::try_from(r * 16 + c)
            .ok()
            .and_then(|i| d.get(i).copied())
            .unwrap_or(u16::MAX)
    };
    let here = at(col, row);
    if here == 0 {
        return WAIT;
    }
    for f in 0u8..4 {
        let (dc, dr) = step_of(f);
        let (x, y) = (col + dc, row + dr);
        if (0..n).contains(&x)
            && (0..n).contains(&y)
            && at(x, y) < here
            && s.can_enter(cfg, x, y, me)
        {
            return toward(facing, f, FORWARD);
        }
    }
    WAIT
}

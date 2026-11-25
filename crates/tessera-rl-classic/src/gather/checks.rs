// SPDX-License-Identifier: MIT OR Apache-2.0

//! Import validation, registration and the scripted baseline of Gather.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, HeaderView, Heuristic, InvariantViolation, Phase, Validate};
use tessera_rl_num::rng::StepRng;

use super::{Gather, GatherConfig, GatherState, MAX_ITEMS};

impl tessera_rl_core::Persist for Gather {
    const ENV_ID: &'static str = super::ENV_ID;
    const ENV_VERSION: u32 = super::ENV_VERSION;
}

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Gather, id = "tessera/Gather-v1");

/// Imported states: the agent and every item on the grid, `left` equal to
/// the items present, `collected` at most [`MAX_ITEMS`], and a step count
/// within `max_steps`. Any slot may hold an item: `items` is adaptive, so
/// an episode may run with its slot's target (version 2). `Validate` sees
/// the config, not the record's header, so an imported state may hold more
/// items than the header's adaptive `items` (or the slot's target): the
/// bound checked is [`MAX_ITEMS`] (#613, REVIEW-A-r1).
impl Validate for Gather {
    fn validate(cfg: &GatherConfig, s: &GatherState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let size = i16::try_from(cfg.size).unwrap_or(i16::MAX);
        let on = |p: [i16; 2]| (0..size).contains(&p[0]) && (0..size).contains(&p[1]);
        if !on(s.agent) {
            return invalid("an agent off the grid");
        }
        // `items` is adaptive (version 2): an episode may run with its
        // slot's target, which `Validate` cannot see, so any of the
        // `MAX_ITEMS` slots may hold an item.
        let mut left = 0u8;
        for (item, present) in s.items.iter().zip(s.present.iter()) {
            if !present.get() {
                continue;
            }
            if !on(*item) {
                return invalid("an item off the grid");
            }
            if *item == s.agent {
                return invalid("an item under the agent");
            }
            left += 1;
        }
        if s.left != left {
            return invalid("a left count that disagrees with the items present");
        }
        if usize::from(s.collected) > MAX_ITEMS {
            return invalid("more items collected than the set holds");
        }
        if s.steps > cfg.max_steps {
            return invalid("a step count past max_steps");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, the slot is alive, and a running episode has not
    /// ended.
    fn validate_record(
        cfg: &GatherConfig,
        header: &HeaderView<'_>,
        s: &GatherState,
    ) -> Result<(), InvariantViolation> {
        let phase = header.phase();
        if phase == Phase::NeverReset {
            if *s != GatherState::default() {
                return Err(InvariantViolation {
                    code: "a state other than the initial value before the first reset",
                });
            }
            return Ok(());
        }
        Self::validate(cfg, s)?;
        if !header.agent(0).is_some_and(|v| v.alive) {
            return Err(InvariantViolation {
                code: "the agent is not alive",
            });
        }
        if phase == Phase::Active && (s.left == 0 || s.steps >= cfg.max_steps) {
            return Err(InvariantViolation {
                code: "a running episode that has ended",
            });
        }
        Ok(())
    }
}

/// Scripted baseline: walk to the nearest item (lowest slot on ties), rows
/// first.
impl Heuristic for Gather {
    fn act(
        _: &GatherConfig,
        s: &GatherState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let [col, row] = s.agent;
        let distance = |p: &[i16; 2]| (p[0] - col).abs() + (p[1] - row).abs();
        let target = s
            .items
            .iter()
            .zip(s.present.iter())
            .filter(|(_, present)| present.get())
            .map(|(item, _)| item)
            .min_by_key(|p| distance(p));
        let Some(&[goal_col, goal_row]) = target else {
            out.set_discrete(0, 0);
            return;
        };
        let action = if row > goal_row {
            0
        } else if row < goal_row {
            1
        } else if col < goal_col {
            2
        } else {
            3
        };
        out.set_discrete(0, action);
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit kinds, their numbers and the per-unit record.
//!
//! The numbers are the placeholders of `rts-design.md` section 3, to be
//! frozen by the calibration among scripted bots; they are not derived from
//! any existing game.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::{Bool8, Reserved};
use tessera_rl_macros::EnvState;

/// A unit kind. 0 is reserved for an empty slot in [`Unit::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// Produces Drones and stores ore; losing every Core ends the game.
    Core = 1,
    /// Produces fighters.
    Foundry = 2,
    /// Carries ore and builds; cannot attack.
    Drone = 3,
    /// Melee fighter.
    Striker = 4,
    /// Ranged fighter.
    Lancer = 5,
}

/// The fixed numbers of a kind. `speed` is ticks per cell, 0 for
/// immobile; `range` and `damage` are 0 for kinds that cannot attack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindStats {
    /// Hit points when complete.
    pub hp: u8,
    /// Ore cost; 0 for the Core, which is never built.
    pub cost: u8,
    /// Ticks to build or produce.
    pub build_ticks: u8,
    /// Ticks per cell; 0 for immobile.
    pub speed: u8,
    /// Attack range, Chebyshev.
    pub range: u8,
    /// Damage per attack.
    pub damage: u8,
    /// Sight, Chebyshev.
    pub sight: u8,
}

impl Kind {
    /// Every kind, in byte order.
    pub const ALL: [Self; 5] = [
        Self::Core,
        Self::Foundry,
        Self::Drone,
        Self::Striker,
        Self::Lancer,
    ];

    /// The kind of a stored byte; `None` for 0 and unknown bytes.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Core),
            2 => Some(Self::Foundry),
            3 => Some(Self::Drone),
            4 => Some(Self::Striker),
            5 => Some(Self::Lancer),
            _ => None,
        }
    }

    /// The kind's numbers (design section 3).
    #[must_use]
    pub const fn stats(self) -> KindStats {
        match self {
            Self::Core => KindStats {
                hp: 12,
                cost: 0,
                build_ticks: 0,
                speed: 0,
                range: 0,
                damage: 0,
                sight: 4,
            },
            Self::Foundry => KindStats {
                hp: 6,
                cost: 5,
                build_ticks: 16,
                speed: 0,
                range: 0,
                damage: 0,
                sight: 3,
            },
            Self::Drone => KindStats {
                hp: 1,
                cost: 1,
                build_ticks: 8,
                speed: 4,
                range: 0,
                damage: 0,
                sight: 3,
            },
            Self::Striker => KindStats {
                hp: 4,
                cost: 2,
                build_ticks: 12,
                speed: 4,
                range: 1,
                damage: 2,
                sight: 3,
            },
            Self::Lancer => KindStats {
                hp: 2,
                cost: 3,
                build_ticks: 16,
                speed: 6,
                range: 3,
                damage: 1,
                sight: 4,
            },
        }
    }

    /// Whether the kind is a structure: immobile, placed by a build order.
    #[must_use]
    pub const fn is_structure(self) -> bool {
        matches!(self, Self::Core | Self::Foundry)
    }
}

/// One unit slot, 16 bytes of plain data (ADR-0029). An empty slot is all
/// zero but its generation. Identity is `(slot, generation)`.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Unit {
    /// The cell it stands on.
    pub cell: u16,
    /// [`Kind`] as a byte; 0 for an empty slot.
    pub kind: u8,
    /// Hit points left.
    pub hp: u8,
    /// Whether the slot holds a unit.
    pub alive: Bool8,
    /// Whether a structure is complete (rule 7); always true for others.
    pub finished: Bool8,
    /// Ore carried, at most 1 (rule 4).
    pub carry: u8,
    /// Movement counter (rule 10).
    pub counter: u8,
    /// Ticks of build or production progress.
    pub progress: u16,
    /// [`Kind`] being produced, 0 for none.
    pub producing: u8,
    /// Zero.
    pub pad: Reserved<1>,
    /// Occupants of this slot so far, modulo 2^32.
    pub generation: u32,
}

const _: () = assert!(size_of::<Unit>() == 16);

/// Advances a movement counter by one tick of a standing move order (rule
/// 10): `(counter, steps)`, where `steps` is true when it reached `speed`,
/// and the counter then restarts at 0. An immobile kind never steps.
#[must_use]
pub const fn advance_counter(counter: u8, speed: u8) -> (u8, bool) {
    if speed == 0 {
        return (0, false);
    }
    let next = counter.saturating_add(1);
    if next >= speed {
        (0, true)
    } else {
        (next, false)
    }
}

// Fog leaves the stage 2 masks unchanged (`rts-design.md` section 14,
// ruling 4) because every target a mask can name is visible when it is
// written: every kind sees its 4-neighbours (sight at least 1) and every
// cell in its range (range at most sight).
const _: () = {
    let mut kinds: &[Kind] = &Kind::ALL;
    while let [k, rest @ ..] = kinds {
        let s = k.stats();
        assert!(s.sight >= 1 && s.range <= s.sight);
        kinds = rest;
    }
};

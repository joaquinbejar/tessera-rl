// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leduc hold'em's privileged state (ADR-0079 item 7): both private cards
//! and the public state, for a centralized critic. No player observes it,
//! and search never reads it.
//!
//! One `u8` leaf of [`STATE_LEN`] entries: `[card of slot 0, card of slot
//! 1, public rank (3 before the deal), round, slot to move, raises in the
//! round, contribution of slot 0, contribution of slot 1, history of round
//! 0, history of round 1, length of round 0, length of round 1, done]`.

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{ObsLeaf, Observation, PrivilegedState, SpaceError};
use tessera_rl_macros::Observation;

use super::{Leduc, LeducConfig, LeducState};

/// Length of the privileged state vector.
pub const STATE_LEN: usize = 13;

/// Leduc hold'em's privileged state.
#[derive(Observation)]
pub struct LeducPrivileged {
    /// See the module documentation for the layout.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [13]))]
    pub state: ObsLeaf<u8>,
}

impl PrivilegedState for Leduc {
    fn state_space(cfg: &LeducConfig) -> Result<Space, SpaceError> {
        LeducPrivileged::space(cfg)
    }

    fn write_state(_: &LeducConfig, state: &LeducState, out: &mut ObsWriter<'_, '_, ()>) {
        let values = [
            state.cards[0],
            state.cards[1],
            state.public,
            state.round,
            state.to_move,
            state.raises,
            state.contributions[0],
            state.contributions[1],
            state.history[0],
            state.history[1],
            state.history_len[0],
            state.history_len[1],
            state.done,
        ];
        let Some(mut leaf) = out.leaf_mut(LeducPrivileged::STATE) else {
            return;
        };
        for (k, v) in values.into_iter().enumerate() {
            leaf.set(k, u8::try_from(v).unwrap_or(u8::MAX));
        }
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::io::ObsWriter;
    use tessera_rl_core::{Observation, PrivilegedState};

    use super::{LeducPrivileged, STATE_LEN};
    use crate::leduc::{Leduc, LeducConfig, LeducState, NO_CARD};

    #[test]
    fn test_leduc_privileged_state_holds_both_cards_and_public_state() {
        let state = LeducState {
            cards: [2, 0],
            public: NO_CARD,
            round: 0,
            to_move: 1,
            raises: 1,
            contributions: [3, 1],
            history: [3, 0],
            history_len: [1, 0],
            done: 0,
        };
        let mut row = [0u8; STATE_LEN];
        {
            let mut leaves: [&mut [u8]; 1] = [&mut row];
            Leduc::write_state(&LeducConfig {}, &state, &mut ObsWriter::new(&mut leaves));
        }
        assert_eq!(row, [2, 0, 3, 0, 1, 1, 3, 1, 3, 0, 1, 0, 0]);
        let space = LeducPrivileged::space(&LeducConfig {}).unwrap();
        // The derived space is a `Dict` of its leaves: here one box.
        let tessera_rl_core::space::Space::Dict(leaves) = space else {
            panic!("a dict");
        };
        let [(name, tessera_rl_core::space::Space::Box { shape, .. })] = leaves.as_slice() else {
            panic!("one box leaf");
        };
        assert_eq!(name, "state");
        assert_eq!(shape, &[13]);
    }
}

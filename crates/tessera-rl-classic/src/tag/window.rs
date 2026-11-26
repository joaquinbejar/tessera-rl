// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tag's observation window written sparsely (#771): the same
//! `[channel, row, col]` bytes as `egocentric_window`, into rows the engine
//! has already zeroed (SPEC 10), so only the nonzero bytes are written.
//! Channel 0 (cells outside the grid) is whole rows above and below the grid
//! plus the column runs left and right of it; channels 1 and 2 (pursuers and
//! live evaders per cell) are at most `MAX_AGENTS` counts, gathered first and
//! written once each. No window array is built or copied.

use super::{MAX_AGENTS, TagConfig, TagState};

/// Writes the nonzero bytes of slot `me`'s window through `set(index,
/// value)`, over a destination that is all zero. A slot without a position
/// writes nothing.
#[inline]
pub(super) fn write_window(
    cfg: &TagConfig,
    state: &TagState,
    me: usize,
    mut set: impl FnMut(usize, u8),
) {
    let Some(&[cx, cy]) = state.pos.get(me) else {
        return;
    };
    let side = cfg.side();
    let plane = side * side;
    let (cx, cy) = (i64::from(cx), i64::from(cy));
    let r = i64::from(cfg.radius);
    let (width, height) = (i64::from(cfg.width), i64::from(cfg.height));
    let side_i = i64::try_from(side).unwrap_or(0);
    let clamp = |v: i64| usize::try_from(v.clamp(0, side_i)).unwrap_or(0);
    // Channel 0: window cell (col, row) is grid cell (cx + col - r, cy + row - r).
    for row in 0..side {
        let y = cy + i64::try_from(row).unwrap_or(0) - r;
        let base = row * side;
        if !(0..height).contains(&y) {
            for col in 0..side {
                set(base + col, 1);
            }
            continue;
        }
        // x < 0 exactly for col < r - cx; x >= width exactly for col >= width + r - cx.
        for col in 0..clamp(r - cx) {
            set(base + col, 1);
        }
        for col in clamp(width + r - cx)..side {
            set(base + col, 1);
        }
    }
    // Channels 1 and 2: one count per occupied (channel, cell).
    let mut hits = [(0usize, 0u8); MAX_AGENTS];
    let mut used = 0usize;
    for (j, &[x, y]) in state.pos.iter().enumerate() {
        if !state.active(cfg, j) {
            continue;
        }
        let col = usize::try_from(i64::from(x) - cx + r).ok();
        let row = usize::try_from(i64::from(y) - cy + r).ok();
        let Some(k) = col
            .zip(row)
            .and_then(|(col, row)| (col < side && row < side).then_some(row * side + col))
        else {
            continue;
        };
        let index = if cfg.is_pursuer(j) { plane } else { 2 * plane } + k;
        if let Some(hit) = hits.iter_mut().take(used).find(|h| h.0 == index) {
            // At most MAX_AGENTS (8) agents per cell.
            hit.1 += 1;
        } else if let Some(free) = hits.get_mut(used) {
            *free = (index, 1);
            used += 1;
        } else {
            debug_assert!(used < hits.len(), "more active agents than MAX_AGENTS");
        }
    }
    for &(index, count) in hits.iter().take(used) {
        set(index, count);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use tessera_rl_core::Bool8;

    use super::super::{MAX_AGENTS, TagConfig, TagState, egocentric_window};
    use super::write_window;

    fn cfg(width: u32, height: u32, pursuers: u32, evaders: u32, radius: u32) -> TagConfig {
        TagConfig {
            width,
            height,
            pursuers,
            evaders,
            radius,
            ..TagConfig::default()
        }
    }

    fn reference(cfg: &TagConfig, state: &TagState, me: usize) -> Vec<u8> {
        let side = cfg.side();
        egocentric_window(cfg, state, me)[..3 * side * side].to_vec()
    }

    fn sparse(cfg: &TagConfig, state: &TagState, me: usize) -> Vec<u8> {
        let side = cfg.side();
        // Zero, as the engine hands rows to `observe`.
        let mut out = vec![0u8; 3 * side * side];
        write_window(cfg, state, me, |k, v| out[k] = v);
        out
    }

    /// A state with every slot placed by `place(slot)` and the given caught set.
    fn state(place: impl Fn(usize) -> [i16; 2], caught: &[usize], agents: usize) -> TagState {
        let mut s = TagState::default();
        for i in 0..agents {
            s.pos[i] = place(i);
        }
        for &i in caught {
            s.caught[i] = Bool8::TRUE;
        }
        s
    }

    const SHAPES: [(u32, u32); 4] = [(3, 3), (8, 8), (3, 64), (64, 3)];

    #[test]
    fn test_window_sparse_equals_reference_for_every_radius_shape_and_position() {
        for (w, h) in SHAPES {
            for radius in 0..=7 {
                let c = cfg(w, h, 2, 3, radius);
                let (wi, hi) = (i16::try_from(w).unwrap(), i16::try_from(h).unwrap());
                for cy in 0..hi {
                    for cx in 0..wi {
                        // The observer at (cx, cy); the others spread, stacked and caught.
                        let s = state(
                            |i| match i {
                                0 | 2 => [cx, cy],
                                1 => [(cx + 1) % wi, cy],
                                3 => [(cx + wi - 1) % wi, (cy + 1) % hi],
                                _ => [(cx + 2) % wi, (cy + 2) % hi],
                            },
                            &[3],
                            5,
                        );
                        for me in 0..5 {
                            assert_eq!(
                                sparse(&c, &s, me),
                                reference(&c, &s, me),
                                "{w}x{h} r={radius} at ({cx},{cy}) me={me}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_window_sparse_counts_eight_stacked_agents_and_skips_inactive_slots() {
        for (pursuers, evaders) in [(1, 1), (4, 4), (1, 7), (7, 1)] {
            let c = cfg(8, 8, pursuers, evaders, 2);
            let agents = usize::try_from(pursuers + evaders).unwrap();
            // Every agent on one cell; one evader caught; the unused slots
            // hold positions too, which must be ignored.
            let mut s = state(|_| [4, 4], &[agents - 1], MAX_AGENTS);
            s.pos[MAX_AGENTS - 1] = [3, 3];
            for me in 0..agents {
                assert_eq!(
                    sparse(&c, &s, me),
                    reference(&c, &s, me),
                    "{pursuers}+{evaders} me={me}"
                );
            }
        }
    }

    #[test]
    fn test_window_sparse_two_calls_into_one_zeroed_buffer_equal_one_call() {
        let c = cfg(8, 8, 2, 3, 3);
        let s = state(|i| [i16::try_from(i).unwrap(), 0], &[], 5);
        let side = c.side();
        let mut twice = vec![0u8; 3 * side * side];
        write_window(&c, &s, 0, |k, v| twice[k] = v);
        write_window(&c, &s, 0, |k, v| twice[k] = v);
        assert_eq!(twice, sparse(&c, &s, 0));
    }

    proptest! {
        #[test]
        fn test_window_sparse_equals_reference_on_random_states(
            shape in 0usize..4,
            radius in 0u32..=7,
            pursuers in 1u32..=4,
            evaders in 1u32..=4,
            xs in proptest::array::uniform8(0i16..64),
            ys in proptest::array::uniform8(0i16..64),
            caught in proptest::array::uniform8(any::<bool>()),
            me in 0usize..8,
        ) {
            let (w, h) = SHAPES[shape];
            let c = cfg(w, h, pursuers, evaders, radius);
            let agents = usize::try_from(pursuers + evaders).unwrap();
            let wi = i16::try_from(w).unwrap();
            let hi = i16::try_from(h).unwrap();
            let caught: Vec<usize> = (usize::try_from(pursuers).unwrap()..agents).filter(|&i| caught[i]).collect();
            let s = state(|i| [xs[i] % wi, ys[i] % hi], &caught, agents);
            let me = me % agents;
            prop_assert_eq!(sparse(&c, &s, me), reference(&c, &s, me));
        }
    }
}

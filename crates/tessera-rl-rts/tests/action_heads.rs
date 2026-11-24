// SPDX-License-Identifier: MIT OR Apache-2.0

//! RTS keeps its action heads (ADR-0062 item 6): the default 64 unit slots
//! (`max_units`) of three discrete heads each (`N_TYPES` = 6 action types,
//! `WINDOW_CELLS` = 81 window cells, `N_PRODUCTS` = 4 products), for every
//! opponent setting, from `Space::heads` and from the built pool's `action_heads`.

use core::num::NonZeroU32;

use tessera_rl_core::Environment;
use tessera_rl_engine::{EnvPool, NoMasks, PoolOptions};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::{Rts, RtsConfig};

const ENTITIES: usize = 64;
const PER_ENTITY: [u32; 3] = [6, 81, 4];

#[test]
fn test_action_heads_rts_every_opponent_matches_pinned_table() {
    let want: Vec<u32> = PER_ENTITY.repeat(ENTITIES);
    for opponent in 0..=3 {
        let cfg = RtsConfig {
            opponent,
            ..RtsConfig::default()
        };
        let heads = Rts::action_space(&cfg)
            .expect("action space")
            .heads()
            .expect("heads")
            .into_parts();
        assert_eq!(heads, (want.clone(), vec![]), "opponent {opponent}");
        let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(0));
        let pool = EnvPool::<Rts, NoMasks>::new(cfg, options).expect("pool");
        assert_eq!(pool.action_heads(), (want.len(), 0), "opponent {opponent}");
    }
}

/// #385: the RTS's heads through `EntityHeads` are the layout it had: the
/// same space, parents (each unit's `type` head), groups of three heads,
/// and mask rows at `u * MASK_PER_UNIT` + the per-unit block offsets
/// `[type 6][target 6 x 81][product 6 x 4]`.
#[test]
fn test_action_heads_entity_heads_match_the_rts_layout() {
    use tessera_rl_core::ActionMask;
    use tessera_rl_rts::env::{
        HEADS_PER_UNIT, MASK_PER_UNIT, N_PRODUCTS, N_TYPES, WINDOW_CELLS, unit_heads,
    };
    let cfg = RtsConfig::default();
    let heads = unit_heads(&cfg);
    assert!(heads.check());
    assert_eq!(heads.entity_mask_len(), MASK_PER_UNIT);
    assert_eq!(
        Rts::entity_group_heads(&cfg).map(core::num::NonZeroU16::get),
        Some(3)
    );
    let units = usize::from(cfg.max_units);
    for u in 0..units {
        let root = u * HEADS_PER_UNIT;
        assert_eq!(Rts::head_parents(&cfg, root), [None, None]);
        let parent = u16::try_from(root).ok();
        assert_eq!(Rts::head_parents(&cfg, root + 1), [parent, None]);
        assert_eq!(Rts::head_parents(&cfg, root + 2), [parent, None]);
        let base = u * MASK_PER_UNIT;
        let (types, cells, products) = (
            usize::from(N_TYPES),
            usize::from(WINDOW_CELLS),
            usize::from(N_PRODUCTS),
        );
        assert_eq!(heads.mask_row(u, 0, 0), Some(base));
        for t in 0..types {
            let t32 = u32::try_from(t).unwrap();
            assert_eq!(heads.mask_row(u, 1, t32), Some(base + types + t * cells));
            assert_eq!(
                heads.mask_row(u, 2, t32),
                Some(base + types + types * cells + t * products)
            );
        }
        assert_eq!(heads.mask_row(u, 1, u32::from(N_TYPES)), None);
    }
}

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

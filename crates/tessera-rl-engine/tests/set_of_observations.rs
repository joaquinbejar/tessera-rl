// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Set` observations whose items are a derived struct (ADR-0082, #429)
//! through the engine: the item leaves follow the item struct's
//! declaration order (`hp` before `pos` here), typed writes through
//! `<FIELD>_ITEM` ids land in their own leaf, absent slots are zero in every
//! item leaf after `observe` whatever the environment wrote into them (or,
//! with `debug-checks` and debug assertions, such a write is reported), and
//! a set inside a set is still refused at pool build, and an item type
//! whose `ObservationLayout::LEAVES` disagrees with its space fails a debug
//! assertion when the pool builds the space.

#![expect(clippy::unwrap_used, reason = "tests: a failure is the assertion")]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, ObsSetOf, Observation, ObservationLayout,
    RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_engine::{
    AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

const MAX: usize = 4;
const HEAD: u8 = 0xAB;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Units {
    present: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct UnitsConfig {
    /// 1: the observation space nests a set inside a set, which the
    /// engine refuses.
    #[config(default = 0)]
    nested: u32,
    /// 1: the observation is `LiarObs`, whose item type misstates its
    /// leaf count.
    #[config(default = 0)]
    lying: u32,
}

/// Declared `hp` before `pos`: the exported item leaves follow this order.
#[derive(Observation)]
struct Unit {
    #[space(box(low = 0.0, high = 10.0, dtype = f32, shape = [1]))]
    hp: ObsLeaf<f32>,
    #[space(box(low = 0, high = 99, dtype = i32, shape = [2]))]
    pos: ObsLeaf<i32>,
}

#[derive(Observation)]
struct UnitsObs {
    #[space(box(low = 0, high = 255, dtype = u8, shape = [1]))]
    head: ObsLeaf<u8>,
    #[space(set(max = MAX, items = Unit))]
    units: ObsSetOf<Unit>,
}

/// A hand-written item type that claims three leaves but has two.
struct Liar;

impl tessera_rl_core::sealed::ObservationLayoutSealed for Liar {}

impl ObservationLayout for Liar {
    const LEAVES: usize = 3;
    const CONTAINS_SET: bool = false;
}

impl<C> Observation<C> for Liar {
    const LEAVES: usize = 3;

    fn space(_: &C) -> Result<Space, SpaceError> {
        let leaf = || Space::Discrete { n: 2 };
        Ok(Space::Dict(vec![
            ("a".to_owned(), leaf()),
            ("b".to_owned(), leaf()),
        ]))
    }
}

#[derive(Observation)]
struct LiarObs {
    #[space(set(max = 2, items = Liar))]
    liars: ObsSetOf<Liar>,
}

/// Slot `s`'s `hp`.
fn hp(slot: usize) -> f32 {
    0.25 * f32::from(u8::try_from(slot + 1).unwrap())
}

/// Slot `s`'s `pos`.
fn pos(slot: usize) -> [i32; 2] {
    let s = i32::try_from(slot).unwrap();
    [s + 1, s + 11]
}

/// Writes every slot's item, present or not.
struct UnitsEnv;

impl Environment for UnitsEnv {
    type State = Units;
    type Config = UnitsConfig;
    type Obs = UnitsObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &UnitsConfig) -> Result<Space, SpaceError> {
        if cfg.nested == 1 {
            let leaf = Space::Discrete { n: 2 };
            let inner = Space::Set {
                inner: Box::new(leaf),
                max: 2,
            };
            let items = Space::Dict(vec![("inner".to_owned(), inner)]);
            return Ok(Space::Dict(vec![(
                "outer".to_owned(),
                Space::Set {
                    inner: Box::new(items),
                    max: 2,
                },
            )]));
        }
        if cfg.lying == 1 {
            return LiarObs::space(cfg);
        }
        UnitsObs::space(cfg)
    }

    fn action_space(_: &UnitsConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 16 })
    }

    fn reset(_: &UnitsConfig, _: &mut StepRng, state: &mut Units) {
        *state = Units::default();
    }

    /// The action's bits are the present slots.
    fn step(
        _: &UnitsConfig,
        state: &mut Units,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        let bits = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0);
        state.present = u32::try_from(bits).unwrap_or(0);
    }

    fn observe(
        _: &UnitsConfig,
        state: &Units,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, UnitsObs>,
    ) {
        if let Some(mut head) = out.leaf_mut(UnitsObs::HEAD) {
            head.fill(HEAD);
        }
        let Some(mut set) = out.set_of_mut(UnitsObs::UNITS) else {
            return;
        };
        for slot in 0..set.capacity() {
            set.set(slot, state.present & (1 << slot) != 0);
            if let Some(mut row) = set.item(slot, Unit::HP_ITEM) {
                row.set(0, hp(slot));
            }
            if let Some(mut row) = set.item(slot, Unit::POS_ITEM) {
                let _ = row.copy_from(&pos(slot));
            }
        }
    }
}

fn pool(n: u32, workers: Option<usize>) -> EnvPool<UnitsEnv, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
    };
    let mut p = EnvPool::new(UnitsConfig::default(), options).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p
}

#[cfg(not(all(feature = "debug-checks", debug_assertions)))]
/// Checks every instance's item rows and mask against its present bits.
fn check_rows(hp_leaf: &[u8], pos_leaf: &[u8], mask: &[u8], bits: &[i32], what: &str) {
    for (i, &b) in bits.iter().enumerate() {
        for slot in 0..MAX {
            let present = b & (1 << slot) != 0;
            let at = i * MAX + slot;
            let hp_row = hp_leaf.get(at * 4..(at + 1) * 4).unwrap();
            let pos_row = pos_leaf.get(at * 8..(at + 1) * 8).unwrap();
            let (want_hp, want_pos) = if present {
                let p = pos(slot);
                let mut bytes = [0u8; 8];
                bytes[..4].copy_from_slice(&p[0].to_le_bytes());
                bytes[4..].copy_from_slice(&p[1].to_le_bytes());
                (hp(slot).to_le_bytes(), bytes)
            } else {
                ([0; 4], [0; 8])
            };
            assert_eq!(hp_row, want_hp, "{what} instance {i} slot {slot} hp");
            assert_eq!(pos_row, want_pos, "{what} instance {i} slot {slot} pos");
            assert_eq!(
                *mask.get(at).unwrap(),
                u8::from(present),
                "{what} instance {i} slot {slot}"
            );
        }
    }
}

#[test]
fn test_set_of_layout_follows_item_declaration_order() {
    assert_eq!(<Unit as ObservationLayout>::LEAVES, 2);
    const { assert!(!<Unit as ObservationLayout>::CONTAINS_SET) };
    const { assert!(<UnitsObs as Observation<UnitsConfig>>::CONTAINS_SET) };
    assert_eq!(<UnitsObs as Observation<UnitsConfig>>::LEAVES, 4);
    assert_eq!((Unit::HP_ITEM.index(), Unit::POS_ITEM.index()), (0, 1));
    assert_eq!((UnitsObs::UNITS.first(), UnitsObs::UNITS.mask()), (1, 3));
    let space = UnitsEnv::observation_space(&UnitsConfig::default()).unwrap();
    let Space::Dict(entries) = space else {
        panic!("a dict");
    };
    let Space::Set { inner, max: 4 } = &entries[1].1 else {
        panic!("a set of 4");
    };
    let Space::Dict(items) = &**inner else {
        panic!("a dict inner");
    };
    let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["hp", "pos"]);
    assert!(matches!(
        &items[0].1,
        Space::Box {
            dtype: DType::F32,
            ..
        }
    ));
    let p = pool(1, None);
    let out = p.outputs().unwrap();
    assert_eq!(out.observation(0).unwrap().len(), 1);
    assert_eq!(out.observation(1).unwrap().len(), MAX * 4);
    assert_eq!(out.observation(2).unwrap().len(), MAX * 8);
    assert_eq!(out.observation(3).unwrap().len(), MAX);
    assert!(out.observation(4).is_none());
}

#[cfg(not(all(feature = "debug-checks", debug_assertions)))]
#[test]
fn test_set_of_typed_writes_land_and_absent_slots_come_out_zero() {
    for workers in [None, Some(2)] {
        let mut p = pool(4, workers);
        let out = p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(out.observation(0).unwrap(), [HEAD; 4]);
        for leaf in 1..4 {
            assert!(out.observation(leaf).unwrap().iter().all(|&b| b == 0));
        }
        let bits = [0b0101, 0b1111, 0b0000, 0b1000];
        let out = p
            .step(tessera_rl_engine::ActionInput {
                discrete: &bits,
                continuous: &[],
            })
            .unwrap();
        let (hp_leaf, pos_leaf, mask) = (
            out.observation(1).unwrap(),
            out.observation(2).unwrap(),
            out.observation(3).unwrap(),
        );
        check_rows(
            hp_leaf,
            pos_leaf,
            mask,
            &bits,
            &format!("workers {workers:?}"),
        );
    }
}

#[cfg(not(all(feature = "debug-checks", debug_assertions)))]
#[test]
fn test_set_of_rows_emptied_mid_episode_come_out_zero() {
    for workers in [None, Some(2)] {
        let mut p = pool(2, workers);
        p.reset_all(Reseed::Keep).unwrap();
        for bits in [[0b1111, 0b1111], [0b0010, 0b0000], [0b0010, 0b1001]] {
            let out = p
                .step(tessera_rl_engine::ActionInput {
                    discrete: &bits,
                    continuous: &[],
                })
                .unwrap();
            let (hp_leaf, pos_leaf, mask) = (
                out.observation(1).unwrap(),
                out.observation(2).unwrap(),
                out.observation(3).unwrap(),
            );
            let what = format!("workers {workers:?} bits {bits:?}");
            check_rows(hp_leaf, pos_leaf, mask, &bits, &what);
        }
    }
}

#[cfg(all(feature = "debug-checks", debug_assertions))]
#[test]
fn test_set_of_absent_slot_write_is_reported_under_debug_checks() {
    let mut p = pool(1, None);
    assert_eq!(p.reset_all(Reseed::Keep).err(), Some(EngineError::Panicked));
}

#[test]
fn test_set_of_set_inside_a_set_is_refused_at_pool_build() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let cfg = UnitsConfig {
        nested: 1,
        lying: 0,
    };
    assert_eq!(
        EnvPool::<UnitsEnv, NoMasks>::new(cfg, options).err(),
        Some(EngineError::UnsupportedSpace {
            what: "a set inside a set"
        })
    );
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "the items' space and ObservationLayout::LEAVES disagree")]
fn test_set_of_item_leaf_count_mismatch_fails_at_pool_build() {
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let cfg = UnitsConfig {
        nested: 0,
        lying: 1,
    };
    let _ = EnvPool::<UnitsEnv, NoMasks>::new(cfg, options);
}

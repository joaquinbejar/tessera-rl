// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Grid `layout` asset (ADR-0077): decoding, canonical bytes and hash,
//! validation against the config, and walls in the kernel.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{Grid, GridConfig, GridLayout};
use tessera_rl_core::asset::check_asset_fixpoint;
use tessera_rl_core::config::{asset_list, canonical_bytes};
use tessera_rl_core::{Asset, AssetData, AssetError, EnvConfig};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const ROOMS: &[u8] =
    b"....#...\n....#...\n........\n....#...\n###.####\n....#...\n........\n....#...\n";

fn layout(text: &[u8]) -> Asset<GridLayout> {
    Asset::from_bytes(text).unwrap()
}

fn cfg(size: u16, layout: Asset<GridLayout>) -> GridConfig {
    GridConfig {
        size,
        max_steps: 50,
        layout,
    }
}

#[test]
fn test_grid_layout_open_default_and_bundled_hash_equal() {
    let d = GridConfig::default();
    assert_eq!(d.layout, Asset::<GridLayout>::bundled("open").unwrap());
    assert_eq!(d.layout.get().side(), 0);
    assert_eq!(d.layout.get().canonical_bytes().as_ref(), &[0xff, 0, 0]);
}

#[test]
fn test_grid_layout_text_crlf_and_canonical_inputs_hash_equal() {
    let lf = layout(ROOMS);
    let crlf: Vec<u8> = ROOMS
        .split(|&b| b == b'\n')
        .filter(|r| !r.is_empty())
        .flat_map(|r| r.iter().copied().chain(*b"\r\n"))
        .collect();
    let no_final = &ROOMS[..ROOMS.len() - 1];
    let canonical = lf.get().canonical_bytes().into_owned();
    for other in [layout(&crlf), layout(no_final), layout(&canonical)] {
        assert_eq!(other.hash(), lf.hash());
    }
    assert_eq!(
        Asset::<GridLayout>::bundled("rooms8").unwrap().hash(),
        lf.hash()
    );
    assert_eq!(lf.get().wall_count(), 12);
    for sample in [ROOMS, b"", crlf.as_slice()] {
        check_asset_fixpoint::<GridLayout>(sample).unwrap();
    }
}

#[test]
fn test_grid_layout_canonical_side_35_is_not_read_as_text() {
    // Side 35 starts with the byte '#'; the marker keeps it unambiguous.
    let row = [b'.'; 35];
    let text: Vec<u8> = (0..35)
        .flat_map(|_| row.iter().copied().chain(*b"\n"))
        .collect();
    let a = layout(&text);
    let canonical = a.get().canonical_bytes().into_owned();
    assert_eq!(canonical[..3], [0xff, 35, 0]);
    assert_eq!(layout(&canonical), a);
}

#[test]
fn test_grid_layout_decode_refuses_bad_inputs() {
    let invalid = |b: &[u8]| matches!(GridLayout::decode(b), Err(AssetError::Invalid { .. }));
    assert!(invalid(b"..\n.\n"), "ragged rows");
    assert!(invalid(b"..\n.x\n"), "unknown cell");
    assert!(invalid(&[0xff, 2]), "truncated canonical");
    assert!(invalid(&[0xff, 2, 0]), "missing bitset");
    assert!(invalid(&[0xff, 2, 0, 0x10]), "bit past the last cell");
    assert!(invalid(&[0xff, 2, 0, 0, 0]), "bitset too long");
    let big: Vec<u8> = (0..65)
        .flat_map(|_| [b'.'; 65].into_iter().chain(*b"\n"))
        .collect();
    assert_eq!(
        GridLayout::decode(&big).unwrap_err(),
        AssetError::DecodedTooLarge {
            limit: GridLayout::MAX_DECODED_BYTES
        }
    );
    assert!(matches!(
        GridLayout::decode(&[0xff, 65, 0]),
        Err(AssetError::DecodedTooLarge { .. })
    ));
}

#[test]
fn test_grid_layout_validation_names_the_layout() {
    let rooms = layout(ROOMS);
    assert!(cfg(8, rooms.clone()).validate().is_ok());
    assert!(cfg(16, layout(b"")).validate().is_ok());
    let field = |c: GridConfig| c.validate().unwrap_err().field;
    assert_eq!(field(cfg(9, rooms)), "layout");
    assert_eq!(field(cfg(4, layout(b"#...\n....\n....\n....\n"))), "layout");
    assert_eq!(field(cfg(4, layout(b".###\n####\n####\n####\n"))), "layout");
    assert!(
        cfg(4, layout(b"..##\n####\n####\n####\n"))
            .validate()
            .is_ok()
    );
}

#[test]
fn test_grid_layout_is_a_config_asset_in_canonical_form_and_list() {
    let rooms = layout(ROOMS);
    let c = cfg(8, rooms.clone());
    let list = asset_list(&c);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "layout");
    assert_eq!(list[0].hash, rooms.hash());
    let bytes = canonical_bytes(&c);
    let tail = [&[15u8][..], &rooms.hash().0].concat();
    assert!(
        bytes.windows(17).any(|w| w == tail.as_slice()),
        "tag 15 + hash"
    );
    assert_ne!(canonical_bytes(&cfg(8, layout(b""))), bytes);
}

fn pool(c: GridConfig, seed: u64) -> EnvPool<Grid, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::new(c, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

#[test]
fn test_grid_layout_goals_never_on_walls_and_cover_free_cells() {
    let rooms = layout(ROOMS);
    let mut seen = std::collections::BTreeSet::new();
    // Seeds far apart: nearby pool seeds may share instance streams.
    for seed in 0..40 {
        let p = pool(cfg(8, rooms.clone()), seed * 1_000_003);
        for i in 0..64 {
            let g = p.state(i).unwrap().goal;
            assert!(
                !rooms.get().is_wall(i32::from(g[0]), i32::from(g[1])),
                "{g:?}"
            );
            assert_ne!(g, [0, 0]);
            seen.insert(g);
        }
    }
    // 64 cells, 12 walls, the start: 51 goal cells.
    assert_eq!(seen.len(), 51);
}

#[test]
fn test_grid_layout_wall_blocks_a_move_and_shows_in_the_border_channel() {
    // The start's right neighbour is a wall; below it is free; (2, 1) is an
    // inner wall, off the border.
    let c = cfg(4, layout(b".#..\n..#.\n....\n....\n"));
    let mut p = pool(c, 3);
    let step = |p: &mut EnvPool<Grid, NoMasks>, a: i32| {
        p.step(ActionInput {
            discrete: &vec![a; 64],
            continuous: &[],
        })
        .unwrap();
    };
    step(&mut p, 2); // right: into the wall
    assert_eq!(p.state(0).unwrap().cell(), (0, 0));
    assert_eq!(p.state(0).unwrap().steps, 1);
    let obs = p.outputs().unwrap().observation(0).unwrap().to_vec();
    let cells = 16;
    assert_eq!(obs[2 * cells + 6], 255, "inner wall (2, 1) in channel 2");
    assert_eq!(obs[2 * cells + 5], 0, "free inner cell (1, 1)");
    step(&mut p, 1); // down: free
    assert_eq!(p.state(0).unwrap().cell(), (0, 1));
}

#[test]
fn test_grid_layout_canonical_config_rebuilds_open_not_walled() {
    use tessera_rl_core::config_canonical::{CanonicalDecodeError, decode_canonical};
    let open = GridConfig::default();
    assert_eq!(
        decode_canonical::<GridConfig>(&canonical_bytes(&open)),
        Ok(open)
    );
    assert_eq!(
        decode_canonical::<GridConfig>(&canonical_bytes(&cfg(8, layout(ROOMS)))),
        Err(CanonicalDecodeError::Asset { key: "layout" })
    );
}

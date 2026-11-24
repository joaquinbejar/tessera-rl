// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tensors equal the pool's output slices (#367).

#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "test assertions")]

use core::num::NonZeroU32;

use burn::prelude::{Device, Int, Tensor, TensorData};
use tessera_rl_burn::{AdapterError, actions, f32_leaf, flags, masks, rewards};
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

fn dev() -> Device {
    Device::flex()
}

fn cartpole(n: u32) -> EnvPool<CartPole> {
    let mut p = EnvPool::<CartPole>::new(
        CartPoleConfig::default(),
        PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(7)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

#[test]
fn test_adapter_cartpole_observations_equal_the_leaf_bytes() {
    let device = dev();
    let mut p = cartpole(5);
    p.step(ActionInput { discrete: &[0, 1, 0, 1, 1], continuous: &[] }).unwrap();
    let out = p.outputs().unwrap();
    let t = f32_leaf(&out, 0, 4, &device).unwrap();
    assert_eq!(t.dims(), [5, 4]);
    let got = t.into_data().to_vec::<f32>().unwrap();
    let want: Vec<f32> = out
        .observation(0)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let r = rewards(&out, &device).unwrap().into_data().to_vec::<f32>().unwrap();
    assert_eq!(r, out.rewards());
    let done = flags(out.episode_terminated(), &device).unwrap();
    assert_eq!(done.dims(), [5]);
}

#[test]
fn test_adapter_tag_masks_have_one_row_per_slot() {
    let device = dev();
    let mut p = EnvPool::<Tag, Masked>::with_masks(
        TagConfig::default(),
        PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let out = p.outputs().unwrap();
    let rows = out.masks().len() / 5;
    let m = masks(&out, 5, &device).unwrap();
    assert_eq!(m.dims(), [rows, 5]);
    let got = m.into_data().to_vec::<bool>().unwrap();
    assert_eq!(got, out.masks().iter().map(|b| b.get()).collect::<Vec<_>>());
    assert_eq!(rows, 3 * tessera_rl_classic::tag::MAX_AGENTS);
}

#[test]
fn test_adapter_refuses_bad_shapes_and_missing_leaves() {
    let device = dev();
    let p = cartpole(2);
    let out = p.outputs().unwrap();
    assert_eq!(f32_leaf(&out, 3, 4, &device).unwrap_err(), AdapterError::NoLeaf { leaf: 3 });
    assert!(matches!(f32_leaf(&out, 0, 3, &device), Err(AdapterError::Shape { .. })));
    assert!(matches!(f32_leaf(&out, 0, 0, &device), Err(AdapterError::Shape { .. })));
    assert_eq!(flags(&[], &device).unwrap_err(), AdapterError::Empty);
}

#[test]
fn test_adapter_actions_round_trip() {
    let device = dev();
    let t = Tensor::<1, Int>::from_data(TensorData::new(vec![0i64, 1, 1, 0], [4]), &device);
    let a = actions(t);
    assert_eq!(a, [0, 1, 1, 0]);
    let mut p = cartpole(4);
    p.step(ActionInput { discrete: &a, continuous: &[] }).unwrap();
}

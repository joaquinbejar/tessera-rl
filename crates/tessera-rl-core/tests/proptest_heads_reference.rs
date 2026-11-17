// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Space::heads` gives exactly what the engine's `ActionLayout::from_space`
//! gave before ADR-0062 moved it to core: the same heads in the same order,
//! and the same error wording for every unsupported space.
//!
//! `reference` is that engine code as it was at `e70e828`
//! (`tessera-rl-engine/src/layout.rs`), with `EngineError::UnsupportedSpace
//! { what }` kept as `what`. Any change to `Space::heads` must keep this
//! property, because `EnvPool::action_heads` and every action buffer are
//! built from it.

use proptest::prelude::*;
use tessera_rl_core::space::{ActionHeads, BoxBounds, DType, HeadsError, Space};

type Heads = (Vec<u32>, Vec<(f64, f64)>);

/// The removed engine algorithm, unchanged except for its error type.
mod reference {
    use super::{BoxBounds, DType, Heads, Space};

    pub fn heads(space: &Space) -> Result<Heads, &'static str> {
        let mut out = (Vec::new(), Vec::new());
        add(space, &mut out)?;
        Ok(out)
    }

    fn add(space: &Space, out: &mut Heads) -> Result<(), &'static str> {
        match space {
            Space::Discrete { n } => out.0.push(*n),
            Space::MultiDiscrete { nvec } => {
                for &n in nvec {
                    out.0.push(n);
                }
            }
            Space::MultiBinary { shape } => {
                for _ in 0..elements(shape)? {
                    out.0.push(2);
                }
            }
            Space::Box {
                shape,
                dtype: DType::F32,
                bounds: BoxBounds::Float { low, high },
            } => {
                let count = elements(shape)?;
                for i in 0..count {
                    match (pick(low, i), pick(high, i)) {
                        (Some(&low), Some(&high)) => out.1.push((low, high)),
                        _ => return Err("box bounds"),
                    }
                }
            }
            Space::Box {
                dtype: DType::F64, ..
            } => return Err("f64 box actions"),
            Space::Box { .. } => return Err("integer box actions"),
            Space::Dict(entries) => {
                for (_, inner) in entries {
                    add(inner, out)?;
                }
            }
            Space::Tuple(items) => {
                for inner in items {
                    add(inner, out)?;
                }
            }
            Space::Set { .. } => return Err("set actions"),
        }
        Ok(())
    }

    fn pick(v: &[f64], i: usize) -> Option<&f64> {
        if v.len() == 1 { v.first() } else { v.get(i) }
    }

    fn elements(shape: &[u32]) -> Result<usize, &'static str> {
        shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
            .ok_or("shape too large")
    }
}

fn shape() -> impl Strategy<Value = Vec<u32>> {
    prop::collection::vec(1u32..4, 1..3)
}

fn bounds() -> impl Strategy<Value = Vec<f64>> {
    // Lengths 0..5 cover broadcast (1), per element and too short.
    prop::collection::vec(-4.0f64..4.0, 0..5)
}

fn leaf() -> impl Strategy<Value = Space> {
    prop_oneof![
        (1u32..10).prop_map(|n| Space::Discrete { n }),
        prop::collection::vec(1u32..10, 1..4).prop_map(|nvec| Space::MultiDiscrete { nvec }),
        shape().prop_map(|shape| Space::MultiBinary { shape }),
        (shape(), bounds(), bounds()).prop_map(|(shape, low, high)| Space::Box {
            shape,
            dtype: DType::F32,
            bounds: BoxBounds::Float { low, high },
        }),
        shape().prop_map(|shape| Space::Box {
            shape,
            dtype: DType::F64,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0],
            },
        }),
        shape().prop_map(|shape| Space::Box {
            shape,
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![1],
            },
        }),
    ]
}

fn space() -> impl Strategy<Value = Space> {
    leaf().prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..4).prop_map(|items| Space::Dict(
                items
                    .into_iter()
                    .enumerate()
                    .map(|(i, s)| (format!("k{i}"), s))
                    .collect()
            )),
            prop::collection::vec(inner.clone(), 1..4).prop_map(Space::Tuple),
            inner.prop_map(|s| Space::Set {
                inner: Box::new(s),
                max: 2,
            }),
        ]
    })
}

proptest! {
    #[test]
    fn test_space_heads_any_space_equals_engine_reference(s in space()) {
        let new = s.heads().map(ActionHeads::into_parts).map_err(|e| HeadsError::what(&e));
        prop_assert_eq!(new, reference::heads(&s));
    }
}

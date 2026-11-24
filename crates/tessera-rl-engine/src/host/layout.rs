// SPDX-License-Identifier: MIT OR Apache-2.0

//! `HostLayout`: what a host pool's actions and outputs look like, fixed at
//! construction (SPEC 14.4.3 draft, ADR-0065 item 5). A binding reads it to
//! describe the pool without naming the environment's type.

use tessera_rl_core::HostEnvironment;
use tessera_rl_core::space::{DType, Space};

use crate::error::EngineError;
use crate::layout::ActionLayout;

use crate::layout::MaskHeadShape;

/// The fixed layout of a host pool (SPEC 14.4.3 draft, ADR-0065 item 5).
///
/// The id, version, turn structure and perfect-information flag are in
/// [`HostPool::registration`](super::HostPool::registration), and the
/// auto-reset mode in [`HostPool::options`](super::HostPool::options).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent layout flags that bindings read one by one"
)]
pub struct HostLayout {
    /// Instances in the pool.
    pub num_envs: usize,
    /// Agent slots per instance, the capacity (`HostEnvironment::MAX_AGENTS`).
    pub agents: usize,
    /// Slots alive after each reset (`HostEnvironment::initial_agents`).
    pub initial_agents: usize,
    /// Whether a spawn can be valid (`HostEnvironment::SPAWNS`).
    pub spawns: bool,
    /// Whether the environment refuses agents and the `refused` and
    /// `refuse_code` outputs exist (`HostEnvironment::REFUSES_AGENTS`,
    /// ADR-0070).
    pub refuses_agents: bool,
    /// Whether the pool's snapshots are reachable through `DynHostPool`
    /// and the host object (`HostPool::with_dyn_snapshots`, ADR-0073).
    pub snapshots: bool,
    /// The observation space, validated.
    pub observation_space: Space,
    /// The observation leaves, in the order of the observation outputs
    /// (`OutputField::Observation(i)`).
    pub leaves: Vec<HostLeaf>,
    /// The action space, validated.
    pub action_space: Space,
    /// Choices of each discrete action head of one agent, in order
    /// (SPEC 11.1); empty for a continuous environment.
    pub discrete_heads: Vec<u32>,
    /// Inclusive bounds of each continuous action head of one agent.
    pub continuous_heads: Vec<(f64, f64)>,
    /// Bytes of one agent's mask row; 0 without masks
    /// (`HostEnvironment::MASKS` false).
    pub mask_len: usize,
    /// Each discrete head's block in an agent's mask row (SPEC 11.3); empty
    /// without masks.
    pub mask_heads: Vec<MaskHeadShape>,
    /// Whether terminal observations are written (`same_step` auto-reset).
    pub terminal: bool,
    /// Discrete heads per entity group (ADR-0051, #385); 0 without groups
    /// or without masks.
    pub entity_group_heads: usize,
}

/// The layout of a host pool of `E`: `(num_envs, agents, initial_agents)`,
/// the validated spaces, the action layout and the mask row length as
/// `HostPool::new` built them.
pub(crate) fn host_layout<E: HostEnvironment>(
    (num_envs, agents, initial_agents): (usize, usize, usize),
    observation_space: Space,
    action_space: Space,
    actions: &ActionLayout,
    mask_len: usize,
    terminal: bool,
    entity_group_heads: usize,
) -> Result<HostLayout, EngineError> {
    // Host protocol 1 has no `Set` (ADR-0082 point 8): refused, not dropped.
    if observation_space.contains_set() {
        return Err(EngineError::UnsupportedSpace {
            what: "set observations in host protocol 1",
        });
    }
    // Each discrete head's block in an agent's mask row, as laid out (SPEC
    // 11.3); none without masks.
    let mask_heads = if E::MASKS {
        (0..actions.discrete.len())
            .map(|h| actions.mask.shape(h, &actions.discrete))
            .collect::<Option<Vec<_>>>()
            .ok_or(EngineError::UnsupportedSpace {
                what: "a discrete head without a mask block",
            })?
    } else {
        Vec::new()
    };
    Ok(HostLayout {
        num_envs,
        agents,
        initial_agents,
        spawns: E::SPAWNS,
        refuses_agents: E::REFUSES_AGENTS,
        snapshots: false,
        leaves: leaves_of(&observation_space),
        observation_space,
        action_space,
        discrete_heads: actions.discrete.clone(),
        continuous_heads: actions.continuous.clone(),
        mask_len,
        mask_heads,
        entity_group_heads,
        terminal,
    })
}

/// One observation leaf of one agent: its output is `[num_envs, agents,
/// *shape]` of `dtype`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HostLeaf {
    /// The leaf's path in the space: `Dict` keys and `Tuple` positions
    /// joined by `.`; empty for a space that is a single leaf.
    pub name: String,
    /// The element type.
    pub dtype: DType,
    /// One agent's shape: `[]` for `Discrete`, `[len]` for `MultiDiscrete`.
    pub shape: Vec<u32>,
}

/// The leaves of `space` in declaration order, as the engine lays out the
/// observation outputs (`layout::obs_leaves`); `host_layout` refuses a
/// `Set` before this runs.
pub(crate) fn leaves_of(space: &Space) -> Vec<HostLeaf> {
    let mut out = Vec::new();
    add(space, String::new(), &mut out);
    out
}

fn add(space: &Space, name: String, out: &mut Vec<HostLeaf>) {
    let join = |part: &str| {
        if name.is_empty() {
            part.to_owned()
        } else {
            format!("{name}.{part}")
        }
    };
    match space {
        Space::Discrete { .. } => out.push(HostLeaf {
            name,
            dtype: DType::I32,
            shape: Vec::new(),
        }),
        Space::MultiDiscrete { nvec } => out.push(HostLeaf {
            name,
            dtype: DType::I32,
            shape: vec![u32::try_from(nvec.len()).unwrap_or(u32::MAX)],
        }),
        Space::MultiBinary { shape } => out.push(HostLeaf {
            name,
            dtype: DType::Bool,
            shape: shape.clone(),
        }),
        Space::Box { shape, dtype, .. } => out.push(HostLeaf {
            name,
            dtype: *dtype,
            shape: shape.clone(),
        }),
        Space::Dict(entries) => {
            for (key, inner) in entries {
                add(inner, join(key), out);
            }
        }
        Space::Tuple(items) => {
            for (i, inner) in items.iter().enumerate() {
                add(inner, join(&i.to_string()), out);
            }
        }
        // Refused by `host_layout` before this runs (ADR-0082 point 8).
        Space::Set { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use tessera_rl_core::space::{BoxBounds, DType, Space};

    use super::leaves_of;

    #[test]
    fn test_host_leaves_follow_the_engine_order_with_paths() {
        let space = Space::Dict(vec![
            (
                "board".to_owned(),
                Space::Box {
                    shape: vec![2, 3],
                    dtype: DType::U8,
                    bounds: BoxBounds::UInt {
                        low: vec![0; 6],
                        high: vec![1; 6],
                    },
                },
            ),
            (
                "pair".to_owned(),
                Space::Tuple(vec![
                    Space::Discrete { n: 4 },
                    Space::MultiDiscrete { nvec: vec![2, 2] },
                ]),
            ),
            ("seen".to_owned(), Space::MultiBinary { shape: vec![5] }),
        ]);
        let got: Vec<_> = leaves_of(&space)
            .into_iter()
            .map(|l| (l.name, l.dtype, l.shape))
            .collect();
        assert_eq!(
            got,
            [
                ("board".to_owned(), DType::U8, vec![2, 3]),
                ("pair.0".to_owned(), DType::I32, vec![]),
                ("pair.1".to_owned(), DType::I32, vec![2]),
                ("seen".to_owned(), DType::Bool, vec![5]),
            ]
        );
        assert_eq!(leaves_of(&Space::Discrete { n: 3 })[0].name, "");
    }
}

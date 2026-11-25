// SPDX-License-Identifier: MIT OR Apache-2.0

//! Teams and roles (ADR-0079 item 8): `Environment::team` and
//! `Environment::role` of every agent slot, read once when a pool is built
//! into its `layout()`, never on the step path.

use tessera_rl_core::{AgentSlot, Environment};

use crate::env::BindError;

/// The team and role of every agent slot of `E` under `cfg`, in slot
/// order.
///
/// # Errors
///
/// [`BindError::Argument`] if a slot index does not fit an `AgentSlot`,
/// which `MAX_AGENTS` rules out for every registered environment.
pub(crate) fn teams_and_roles<E: Environment>(
    cfg: &E::Config,
) -> Result<(Vec<u16>, Vec<&'static str>), BindError> {
    let mut teams = Vec::with_capacity(E::MAX_AGENTS);
    let mut roles = Vec::with_capacity(E::MAX_AGENTS);
    for index in 0..E::MAX_AGENTS {
        let slot = u16::try_from(index)
            .ok()
            .and_then(|i| AgentSlot::new(i, E::MAX_AGENTS))
            .ok_or_else(|| BindError::Argument(format!("agent slot {index} is out of range")))?;
        teams.push(E::team(cfg, slot));
        roles.push(E::role(cfg, slot));
    }
    Ok((teams, roles))
}

#[cfg(test)]
mod tests {
    use super::teams_and_roles;
    use tessera_rl_classic::{
        CartPole, CartPoleConfig, PaddleDuel, PaddleDuelConfig, Snake, SnakeConfig, Tag, TagConfig,
    };
    use tessera_rl_games::{Leduc, LeducConfig};
    use tessera_rl_rts::{Rts, RtsConfig};

    #[test]
    fn test_teams_and_roles_tag_split_pursuers_then_evaders() {
        let cfg = TagConfig {
            pursuers: 2,
            evaders: 3,
            ..TagConfig::default()
        };
        let (teams, roles) = teams_and_roles::<Tag>(&cfg).unwrap();
        assert_eq!(teams, vec![0, 0, 1, 1, 1, 0, 0, 0]);
        assert_eq!(
            roles,
            vec!["pursuer", "pursuer", "evader", "evader", "evader", "", "", ""]
        );
    }

    #[test]
    fn test_teams_and_roles_two_player_games_team_is_slot() {
        let (teams, roles) = teams_and_roles::<PaddleDuel>(&PaddleDuelConfig::default()).unwrap();
        assert_eq!((teams, roles), (vec![0, 1], vec!["", ""]));
        let (teams, roles) = teams_and_roles::<Snake>(&SnakeConfig::default()).unwrap();
        assert_eq!((teams, roles), (vec![0, 1], vec!["", ""]));
    }

    #[test]
    fn test_teams_and_roles_rts_seat_swap_swaps_sides() {
        let cfg = RtsConfig::default();
        let (teams, roles) = teams_and_roles::<Rts>(&cfg).unwrap();
        assert_eq!((teams, roles), (vec![0, 1], vec!["", ""]));
        let swapped = RtsConfig {
            seat_swap: true,
            ..RtsConfig::default()
        };
        assert_eq!(teams_and_roles::<Rts>(&swapped).unwrap().0, vec![1, 0]);
    }

    #[test]
    fn test_teams_and_roles_cartpole_and_leduc_defaults() {
        let (teams, roles) = teams_and_roles::<CartPole>(&CartPoleConfig::default()).unwrap();
        assert_eq!((teams, roles), (vec![0], vec![""]));
        let (teams, roles) = teams_and_roles::<Leduc>(&LeducConfig::default()).unwrap();
        assert_eq!((teams, roles), (vec![0, 0], vec!["", ""]));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CartPole physics alone over AoS and SoA storage: the upper bound a
//! lane-wise environment (SoA columns, ADR-0036) could reach, without the
//! engine sequence around it.

use tessera_rl_classic::CartPoleState;
use tessera_rl_classic::cartpole::physics;

use crate::columns::CartPoleColumns;

/// One physics step of every state, AoS.
pub fn physics_aos(states: &mut [CartPoleState], push: &[i32]) {
    for (s, p) in states.iter_mut().zip(push) {
        s.s = physics::step(s.s, *p == 1);
    }
}

/// One physics step of every lane, SoA.
pub fn physics_soa(columns: &mut CartPoleColumns, push: &[i32]) {
    let [x, xd, th, thd] = &mut columns.s;
    for ((((x, xd), th), thd), p) in x
        .iter_mut()
        .zip(xd.iter_mut())
        .zip(th.iter_mut())
        .zip(thd.iter_mut())
        .zip(push)
    {
        [*x, *xd, *th, *thd] = physics::step([*x, *xd, *th, *thd], *p == 1);
    }
}

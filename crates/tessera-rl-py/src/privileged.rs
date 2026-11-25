// SPDX-License-Identifier: MIT OR Apache-2.0

//! `make(..., privileged_state=True)` (ADR-0079 item 7): an environment
//! that implements `PrivilegedState` adds the buffer in its maker; any
//! other refuses the option with `CapabilityError`.

use tessera_rl_core::{Environment, PrivilegedState};
use tessera_rl_engine::{EnvPool, MaskProvider};

use crate::env::{BindError, PoolArgs};

/// Adds the privileged state buffer when `args` asks for it.
pub(crate) fn apply<E: PrivilegedState, M: MaskProvider<E>>(
    pool: EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<EnvPool<E, M>, BindError> {
    if !args.privileged_state {
        return Ok(pool);
    }
    Ok(pool.with_privileged_state()?)
}

/// Refuses `privileged_state=True` for an environment whose maker did not
/// add the buffer: it has no privileged state.
pub(crate) fn check<E: Environment, M: MaskProvider<E>>(
    pool: &EnvPool<E, M>,
    args: &PoolArgs,
) -> Result<(), BindError> {
    if args.privileged_state && !pool.has_privileged_state() {
        return Err(BindError::Capability(
            "this environment has no privileged state".to_owned(),
        ));
    }
    Ok(())
}

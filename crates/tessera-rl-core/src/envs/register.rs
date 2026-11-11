// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `register!` macro (SPEC 10, ADR-0054 items 2 and 3).

/// Registers an environment under its versioned id (SPEC 10, ADR-0054).
///
/// `register!(Env, id = "tessera/<Name>-v<N>");` requires `Env: Persist`
/// and fails to compile unless `id` equals `<Env as Persist>::ENV_ID`. It
/// submits an [`EnvRegistration`](crate::envs::EnvRegistration), which
/// [`registered`](crate::envs::registered) lists.
///
/// ```ignore
/// tessera_rl::register!(Grid, id = "tessera/Grid-v1");
/// ```
#[macro_export]
macro_rules! register {
    ($env:ty, id = $id:expr $(,)?) => {
        const _: () = ::core::assert!(
            $crate::envs::same_id(<$env as $crate::Persist>::ENV_ID, $id),
            "register!: the id must equal the environment's Persist::ENV_ID",
        );
        $crate::envs::__private::inventory::submit! {
            $crate::envs::EnvRegistration::of::<$env>()
        }
    };
}

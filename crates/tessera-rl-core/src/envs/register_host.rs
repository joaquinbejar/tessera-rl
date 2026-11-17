// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `register_host!` macro (ADR-0062 item 5).

/// Registers a host environment under its versioned id (ADR-0062).
///
/// `register_host!(Env, id = "<namespace>/<Name>-v<N>");` requires
/// `Env: HostEnvironment` and fails to compile unless `id` equals
/// `<Env as HostEnvironment>::ENV_ID` and lies outside the reserved
/// `tessera/` namespace. It submits a
/// [`HostRegistration`](crate::envs::HostRegistration), which
/// [`registered_host`](crate::envs::registered_host) lists. `register!`
/// and `register_host!` share one id space: an id registered by both, or
/// twice by either, fails both lookups.
///
/// ```ignore
/// tessera_rl::register_host!(OrderBook, id = "example/OrderBook-v1");
/// ```
#[macro_export]
macro_rules! register_host {
    ($env:ty, id = $id:expr $(,)?) => {
        const _: () = ::core::assert!(
            $crate::envs::same_id(<$env as $crate::HostEnvironment>::ENV_ID, $id),
            "register_host!: the id must equal the environment's HostEnvironment::ENV_ID",
        );
        const _: () = ::core::assert!(
            !$crate::envs::in_reserved_namespace($id),
            "register_host!: the tessera/ namespace is reserved for first-party environments",
        );
        $crate::envs::__private::inventory::submit! {
            $crate::envs::HostRegistration::of::<$env>()
        }
    };
}

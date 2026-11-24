// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `register!` macro (SPEC 10, ADR-0054 items 2 and 3, ADR-0080).

/// Registers an environment under its versioned id (SPEC 10, ADR-0054).
///
/// `register!(Env, id = "tessera/<Name>-v<N>");` requires `Env: Persist`
/// and fails to compile unless `id` equals `<Env as Persist>::ENV_ID`, or
/// unless `Env::ADAPTERS` is [`Adapters::valid_for`](crate::Adapters::valid_for)
/// `Env::TURNS` (ADR-0080 item 2). It submits an
/// [`EnvRegistration`](crate::envs::EnvRegistration), which
/// [`registered`](crate::envs::registered) lists.
///
/// `register!(Env, id = "...", deprecated(since = "0.4.0", note = "use ..."))`
/// also marks the id deprecated (SPEC 16.1, ADR-0080 item 4); it fails to
/// compile unless `since` is `MAJOR.MINOR.PATCH` and `note` is non-empty.
///
/// ```ignore
/// tessera_rl::register!(Grid, id = "tessera/Grid-v1");
/// ```
#[macro_export]
macro_rules! register {
    (@checks $env:ty, $id:expr) => {
        const _: () = ::core::assert!(
            $crate::envs::same_id(<$env as $crate::Persist>::ENV_ID, $id),
            "register!: the id must equal the environment's Persist::ENV_ID",
        );
        const _: () = ::core::assert!(
            <$env as $crate::Environment>::ADAPTERS
                .valid_for(<$env as $crate::Environment>::TURNS),
            "register!: ADAPTERS is not valid for TURNS (turn-masked needs Sequential; AEC is refused for Mixed; ADR-0080)",
        );
    };
    ($env:ty, id = $id:expr $(,)?) => {
        $crate::register!(@checks $env, $id);
        $crate::envs::__private::inventory::submit! {
            $crate::envs::TypeKey::of::<$env>($id)
        }
        $crate::envs::__private::inventory::submit! {
            $crate::envs::EnvRegistration::of::<$env>()
        }
    };
    ($env:ty, id = $id:expr, deprecated(since = $since:expr, note = $note:expr $(,)?) $(,)?) => {
        $crate::register!(@checks $env, $id);
        $crate::envs::__private::inventory::submit! {
            $crate::envs::TypeKey::of::<$env>($id)
        }
        const _: () = ::core::assert!(
            $crate::envs::is_semver($since),
            "register!: deprecated since must be MAJOR.MINOR.PATCH",
        );
        const _: () = ::core::assert!(
            !$note.is_empty(),
            "register!: a deprecation needs a note",
        );
        $crate::envs::__private::inventory::submit! {
            $crate::envs::EnvRegistration::of::<$env>()
                .deprecated($crate::envs::Deprecation::new($since, $note))
        }
    };
}

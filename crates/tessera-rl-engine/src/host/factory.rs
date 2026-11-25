// SPDX-License-Identifier: MIT OR Apache-2.0

//! Make-by-id for host environments from Rust (ADR-0065 item 3).
//!
//! `register_host_factory!(E)` records a [`HostFactory`] for `E` in an
//! `inventory` list of the engine; [`make_host`] finds the factory of an
//! id, decodes the environment's config from a
//! [`ConfigSource`] with the shared decoder
//! ([`ConfigDecode`]) and builds a `HostPool<E, A>` behind
//! [`DynHostPool`].
//!
//! The environment's crate depends on core only (ADR-0062): the macro is
//! expanded by a crate that already depends on the engine, the final
//! binary's crate or a glue crate it references. `core::envs::
//! HostRegistration` stays metadata, listed by `register_host!`.
//!
//! **Linker caveat (ADR-0054).** `inventory` submissions are link-section
//! statics: a glue rlib whose only contribution to a binary is its
//! submissions can be dropped by the linker, and [`make_host`] then
//! reports [`FactoryError::UnknownId`] or [`FactoryError::NoFactory`].
//! Expand `register_host_factory!` in the final binary's crate, or
//! reference the glue crate from it (any item of it, used).

use thiserror::Error;

use tessera_rl_core::envs::{RegistryError, registered_host};
use tessera_rl_core::{ConfigDecode, ConfigDecodeError, ConfigSource, HostEnvironment};

use crate::error::EngineError;
use crate::options::{PoolOptions, SyncOptions};

use super::{DynHostPool, HostPool};

/// Pool options for make-by-id: Inline unless a [`SyncOptions`] is given.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MakeOptions {
    pool: PoolOptions,
    sync: Option<SyncOptions>,
}

impl MakeOptions {
    /// Inline, with these pool options.
    #[must_use]
    pub const fn new(pool: PoolOptions) -> Self {
        Self { pool, sync: None }
    }

    /// Sync, with these worker options.
    #[must_use]
    pub fn with_sync(self, sync: SyncOptions) -> Self {
        Self {
            sync: Some(sync),
            ..self
        }
    }

    /// The pool options.
    #[must_use]
    pub const fn pool(&self) -> &PoolOptions {
        &self.pool
    }

    /// The worker options, for Sync mode.
    #[must_use]
    pub const fn sync(&self) -> Option<&SyncOptions> {
        self.sync.as_ref()
    }
}

/// Why [`make_host`] built no pool.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum FactoryError {
    /// No host registration and no factory has this id.
    #[error("no host environment with id {id:?}")]
    UnknownId {
        /// The id asked for.
        id: String,
    },
    /// `register_host!` lists this id, but no `register_host_factory!` was
    /// linked for it.
    #[error(
        "host environment {id:?} has no factory: expand register_host_factory! for it in the \
         final binary's crate, or reference the glue crate that does"
    )]
    NoFactory {
        /// The registered id.
        id: &'static str,
    },
    /// Two linked factories share an id (checked on every call), or two
    /// registrations do (checked when the id has no factory).
    #[error(transparent)]
    Registry(#[from] RegistryError),
    /// The config source does not decode, or fails `EnvConfig::validate`.
    #[error(transparent)]
    Config(#[from] ConfigDecodeError),
    /// Construction or `with_workers` failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
}

type Build = fn(&ConfigSource<'_>, &MakeOptions) -> Result<Box<dyn DynHostPool>, FactoryError>;

/// The factory of one host environment, recorded by
/// `register_host_factory!` (ADR-0065 item 3).
#[derive(Debug)]
#[non_exhaustive]
pub struct HostFactory {
    /// The environment's id, `HostEnvironment::ENV_ID`.
    pub id: &'static str,
    build: Build,
}

impl HostFactory {
    /// `E`'s factory, for a pool of `A` agent slots. Not public API: use
    /// [`register_host_factory!`](crate::register_host_factory), which
    /// passes `A = E::MAX_AGENTS` (checked at compile time by `HostPool`).
    #[doc(hidden)]
    #[must_use]
    pub const fn new<E, const A: usize>() -> Self
    where
        E: HostEnvironment,
        E::Config: ConfigDecode + Send,
    {
        Self {
            id: E::ENV_ID,
            build: build::<E, A>,
        }
    }

    /// The record of `E`'s factory whose pools have snapshots through
    /// `DynHostPool` (ADR-0073): `register_host_factory!(E, snapshots)`.
    #[doc(hidden)]
    #[must_use]
    pub const fn new_snapshots<E, const A: usize>() -> Self
    where
        E: tessera_rl_core::HostSnapshot,
        E::Config: ConfigDecode + tessera_rl_core::config::CanonicalConfig + Send,
    {
        Self {
            id: E::ENV_ID,
            build: build_snapshots::<E, A>,
        }
    }
}

inventory::collect!(HostFactory);

/// [`build`], with the pool's snapshots reachable through `DynHostPool`.
fn build_snapshots<E, const A: usize>(
    config: &ConfigSource<'_>,
    options: &MakeOptions,
) -> Result<Box<dyn DynHostPool>, FactoryError>
where
    E: tessera_rl_core::HostSnapshot,
    E::Config: ConfigDecode + tessera_rl_core::config::CanonicalConfig + Send,
{
    let cfg = E::Config::decode(config)?;
    let pool = HostPool::<E, A>::new(cfg, options.pool().clone())?.with_dyn_snapshots();
    start(pool, options)
}

/// Decodes `E`'s config, builds the pool and starts its workers.
fn build<E, const A: usize>(
    config: &ConfigSource<'_>,
    options: &MakeOptions,
) -> Result<Box<dyn DynHostPool>, FactoryError>
where
    E: HostEnvironment,
    E::Config: ConfigDecode + Send,
{
    let cfg = E::Config::decode(config)?;
    start(HostPool::<E, A>::new(cfg, options.pool().clone())?, options)
}

/// Starts the workers `options` asks for and erases the pool's type.
fn start<E, const A: usize>(
    mut pool: HostPool<E, A>,
    options: &MakeOptions,
) -> Result<Box<dyn DynHostPool>, FactoryError>
where
    E: HostEnvironment,
    E::Config: Send,
{
    if let Some(sync) = options.sync() {
        pool = pool.with_workers(sync.clone())?;
    }
    Ok(Box::new(pool))
}

/// Looks `id` up among the host factories linked into this binary and
/// builds a pool from `config` (ADR-0065 item 3).
///
/// # Errors
///
/// [`FactoryError::Registry`] when two linked factories share any id
/// (checked on every call, whatever `id` is), or, for an id with no
/// factory, when two `register!`/`register_host!` registrations share an id
/// (the registration lookup, read only then);
/// [`FactoryError::NoFactory`] for an id `register_host!` lists without a
/// factory; [`FactoryError::UnknownId`] for any other id; and the decoding
/// or construction errors.
pub fn make_host(
    id: &str,
    config: &ConfigSource<'_>,
    options: &MakeOptions,
) -> Result<Box<dyn DynHostPool>, FactoryError> {
    let factories: Vec<&'static HostFactory> = inventory::iter::<HostFactory>.into_iter().collect();
    let mut ids: Vec<&'static str> = factories.iter().map(|f| f.id).collect();
    ids.sort_unstable();
    if let Some(dup) = ids.windows(2).find_map(|w| match w {
        [a, b] if a == b => Some(*a),
        _ => None,
    }) {
        return Err(RegistryError::DuplicateId { id: dup }.into());
    }
    if let Some(factory) = factories.iter().find(|f| f.id == id) {
        return (factory.build)(config, options);
    }
    match registered_host()?.iter().find(|r| r.id() == id) {
        Some(record) => Err(FactoryError::NoFactory { id: record.id() }),
        None => Err(FactoryError::UnknownId { id: id.to_owned() }),
    }
}

/// Registers `E`'s host factory for [`make_host`]
/// (ADR-0065 item 3). With `snapshots`, `E` implements `HostSnapshot`
/// and the pools it makes have snapshots through `DynHostPool` (ADR-0073). `E::Config` must implement `ConfigDecode` (derive
/// `EnvConfig`); without it, the expansion does not compile.
///
/// Expand it in the final binary's crate, or in a glue crate the binary
/// references: `inventory` submissions of an otherwise unused rlib can be
/// dropped by the linker (ADR-0054), and the id is then unknown.
///
/// ```ignore
/// tessera_rl_engine::register_host_factory!(my_env::Book);
/// ```
#[macro_export]
macro_rules! register_host_factory {
    ($env:ty, snapshots) => {
        $crate::__private::inventory::submit! {
            $crate::HostFactory::new_snapshots::<
                $env,
                { <$env as $crate::__private::HostEnvironment>::MAX_AGENTS },
            >()
        }
    };
    ($env:ty) => {
        $crate::__private::inventory::submit! {
            $crate::HostFactory::new::<
                $env,
                { <$env as $crate::__private::HostEnvironment>::MAX_AGENTS },
            >()
        }
    };
}

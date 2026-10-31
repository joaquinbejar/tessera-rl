// SPDX-License-Identifier: MIT OR Apache-2.0

//! Derive macros: `EnvState`, `EnvConfig`, `Observation` and `StateEnum`
//! (ADR-0012, ADR-0029, ADR-0030).
//!
//! Generated code refers to `tessera-rl-core` directly when the crate
//! depends on it, otherwise through the `tessera-rl` facade. Override the
//! path with `#[tessera(crate = path)]`.

mod env_config;
mod env_state;
mod observation;
mod paths;
mod state_enum;
mod types;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

fn expand(
    input: TokenStream,
    derive: fn(&DeriveInput) -> syn::Result<proc_macro2::TokenStream>,
) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Plain-data state: checks `#[repr(C)]`, the absence of padding and the
/// field types at compile time, and implements `Canonical` (fields in
/// declaration order) and `EnvState` (ADR-0029). Float fields may be marked
/// `#[state(allow_nan)]`.
#[proc_macro_derive(EnvState, attributes(state, tessera))]
pub fn derive_env_state(input: TokenStream) -> TokenStream {
    expand(input, env_state::derive)
}

/// An enum stored in state through its integer representation: implements
/// `StateEnum` and a validating `Canonical`. Requires `#[repr(<int>)]`,
/// unit variants and exactly one `#[default]` variant.
#[proc_macro_derive(StateEnum, attributes(tessera))]
pub fn derive_state_enum(input: TokenStream) -> TokenStream {
    expand(input, state_enum::derive)
}

/// An environment configuration: `Default` from `#[config(default = ..)]`,
/// range validation from `#[config(range = ..)]` and the canonical form,
/// excluding `#[config(adaptive)]` fields (ADR-0030).
#[proc_macro_derive(EnvConfig, attributes(config, tessera))]
pub fn derive_env_config(input: TokenStream) -> TokenStream {
    expand(input, env_config::derive)
}

/// An observation: its space per config and one `LeafId` per leaf.
#[proc_macro_derive(Observation, attributes(space, tessera))]
pub fn derive_observation(input: TokenStream) -> TokenStream {
    expand(input, observation::derive)
}

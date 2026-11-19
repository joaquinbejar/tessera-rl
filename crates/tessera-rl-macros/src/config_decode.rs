// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `ConfigDecode` impl of `#[derive(EnvConfig)]` (ADR-0065 item 3):
//! decoding starts from the generated `Default`, keys are checked first,
//! each given field is decoded with its rules, and `validate` runs last.
//! Adaptive fields are not entries, so a source naming one is refused as
//! an unknown key.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::env_config::Entry;

/// The decoder of the config `name` with static fields `entries`.
pub(crate) fn decoder(core: &TokenStream, name: &Ident, entries: &[Entry]) -> TokenStream {
    let keys = entries.iter().map(|e| &e.key);
    let arms = entries.iter().map(|Entry { key, ident, nested }| {
        if *nested {
            quote!(#key => cfg.#ident = #core::config_decode::decode_nested(item, #key, depth)?,)
        } else {
            quote!(#key => cfg.#ident = #core::config_decode::ConfigField::from_item(item, #key, depth)?,)
        }
    });
    let body = if entries.is_empty() {
        quote!()
    } else {
        quote! {
            for (key, item) in src.entries() {
                match *key {
                    #(#arms)*
                    // Refused by `check_keys` above.
                    _ => {}
                }
            }
        }
    };
    quote! {
        impl #core::ConfigDecode for #name {
            fn decode_at(
                src: &#core::ConfigSource<'_>,
                depth: u32,
            ) -> ::core::result::Result<Self, #core::ConfigDecodeError> {
                #core::config_decode::check_keys(src, &[#(#keys),*], depth)?;
                // Missing keys keep their default.
                #[allow(unused_mut, reason = "a config without fields")]
                let mut cfg = <Self as ::core::default::Default>::default();
                #body
                #core::EnvConfig::validate(&cfg)?;
                ::core::result::Result::Ok(cfg)
            }
        }
    }
}

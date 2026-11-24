// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(EnvConfig)]` (ADR-0030).
//!
//! Generates `Default` from `#[config(default = ...)]`, range validation
//! (plus a cross-field check named by the struct attribute
//! `#[config(validate = path)]`, a `fn(&Self) -> Result<(), ConfigError>`
//! run after the field checks),
//! and the canonical form (`CanonicalConfig`, keys sorted at compile time).
//! Fields marked `#[config(adaptive)]` are excluded from the canonical form
//! and the config hash. JSON Schema generation is deferred to phase 1b; the
//! header struct for adaptive values comes with the generated headers of
//! the engine (milestone 5, ADR-0036).

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{Data, DeriveInput, Expr, ExprRange, Fields, Ident, LitStr, Type};

use crate::paths::core_path;
use crate::types::last_ident;

const PRIMITIVES: &[&str] = &[
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "Fx32", "Fx64",
];

#[derive(Default)]
struct FieldAttrs {
    default: Option<Expr>,
    range: Option<ExprRange>,
    adaptive: bool,
    shape: bool,
    capacity: bool,
    asset: Option<LitStr>,
}

fn parse_attrs(field: &syn::Field) -> syn::Result<FieldAttrs> {
    let mut attrs = FieldAttrs::default();
    let mut seen: Vec<String> = Vec::new();
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("config")) {
        attr.parse_nested_meta(|meta| {
            let key = meta
                .path
                .get_ident()
                .map(ToString::to_string)
                .unwrap_or_default();
            if seen.contains(&key) {
                return Err(meta.error(format!("duplicate `{key}`")));
            }
            seen.push(key);
            if meta.path.is_ident("default") {
                attrs.default = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("range") {
                attrs.range = Some(meta.value()?.parse()?);
            } else if meta.path.is_ident("adaptive") {
                attrs.adaptive = true;
            } else if meta.path.is_ident("shape") {
                attrs.shape = true;
            } else if meta.path.is_ident("capacity") {
                attrs.capacity = true;
            } else if meta.path.is_ident("asset") {
                attrs.asset = Some(meta.value()?.parse()?);
            } else {
                return Err(meta.error(
                    "expected `default = ...`, `range = ...`, `adaptive`, `shape`, `capacity` or `asset = \"name\"`",
                ));
            }
            Ok(())
        })?;
    }
    Ok(attrs)
}

pub(crate) struct Entry {
    pub(crate) key: String,
    pub(crate) ident: Ident,
    pub(crate) nested: bool,
}

/// Whether `ty` is `Asset<..>` (ADR-0077): written as its hash, decoded
/// from a bundled name or bytes, never adaptive.
fn is_asset(ty: &Type) -> bool {
    last_ident(ty).as_deref() == Some("Asset")
}

/// An asset name: 1 to 64 bytes of `[a-z0-9_.-]` (ADR-0077 point 5).
fn valid_asset_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-')
        })
}

#[derive(Default)]
struct ConfigParts {
    defaults: Vec<TokenStream>,
    checks: Vec<TokenStream>,
    entries: Vec<Entry>,
    adaptive: Vec<String>,
    adaptive_fields: Vec<(Ident, Type)>,
    static_checks: Vec<TokenStream>,
    /// `(field, asset name)` of each asset field.
    assets: Vec<(Ident, String)>,
    /// Nested config fields, whose assets are collected too.
    nested_fields: Vec<Ident>,
}

impl ConfigParts {
    /// Records an asset field's name (ADR-0077), refusing what an asset
    /// cannot be, and refuses `asset = ...` elsewhere.
    fn push_asset(
        &mut self,
        ident: &Ident,
        key: &str,
        attrs: &FieldAttrs,
        asset: bool,
    ) -> syn::Result<()> {
        if asset {
            if attrs.adaptive || attrs.shape || attrs.capacity || attrs.range.is_some() {
                return Err(syn::Error::new_spanned(
                    ident,
                    "an asset field cannot be adaptive, a shape, a capacity or have a range (ADR-0077)",
                ));
            }
            let name = attrs
                .asset
                .as_ref()
                .map_or_else(|| key.to_owned(), LitStr::value);
            if !valid_asset_name(&name) {
                return Err(syn::Error::new_spanned(
                    ident,
                    "asset names are 1 to 64 bytes of [a-z0-9_.-] (ADR-0077)",
                ));
            }
            if self.assets.iter().any(|(_, n)| *n == name) {
                return Err(syn::Error::new_spanned(
                    ident,
                    format!("duplicate asset name `{name}`"),
                ));
            }
            self.assets.push((ident.clone(), name));
        } else if attrs.asset.is_some() {
            return Err(syn::Error::new_spanned(
                ident,
                "`asset = ...` applies only to `Asset<T>` fields",
            ));
        }
        Ok(())
    }

    fn push(&mut self, core: &TokenStream, field: &syn::Field) -> syn::Result<()> {
        let Some(ident) = &field.ident else {
            return Ok(());
        };
        let ty = &field.ty;
        let attrs = parse_attrs(field)?;
        let key = ident.to_string();
        if u16::try_from(key.len()).is_err() {
            return Err(syn::Error::new_spanned(
                ident,
                "config field name is too long",
            ));
        }
        let asset = is_asset(ty);
        let nested = !asset && !is_primitive(ty)?;
        self.push_asset(ident, &key, &attrs, asset)?;
        if nested {
            self.nested_fields.push(ident.clone());
        }
        if attrs.adaptive && (attrs.shape || attrs.capacity) {
            return Err(syn::Error::new_spanned(
                ident,
                "an adaptive field cannot affect shapes or capacities (ADR-0030)",
            ));
        }
        if attrs.adaptive && nested {
            return Err(syn::Error::new_spanned(
                ident,
                "only primitive fields can be adaptive (ADR-0030)",
            ));
        }
        if attrs.range.is_some() && nested {
            return Err(syn::Error::new_spanned(
                ident,
                "ranges apply only to primitive fields",
            ));
        }

        self.defaults
            .push(if let (true, Some(expr)) = (asset, &attrs.default) {
                // A bundled asset's name (ADR-0077 point 4).
                quote!(#ident: #core::asset::bundled_or_default(#expr),)
            } else if let Some(expr) = &attrs.default {
                quote!(#ident: #expr,)
            } else {
                quote!(#ident: ::core::default::Default::default(),)
            });
        if let Some(range) = &attrs.range {
            let reason = LitStr::new(
                &format!("must be in {}", range.to_token_stream()),
                ident.span(),
            );
            self.checks.push(quote! {
                if !(#range).contains(&self.#ident) {
                    return ::core::result::Result::Err(#core::ConfigError {
                        field: #key,
                        reason: #reason,
                    });
                }
            });
        }
        if nested {
            self.checks
                .push(quote!(#core::EnvConfig::validate(&self.#ident)?;));
            // ADR-0030: only top-level fields can be adaptive.
            // Spanned at the field, so the error points at it (SPEC 10.2).
            self.static_checks.push(quote_spanned! {ident.span()=>
                const _: () = assert!(
                    <#ty as #core::config::CanonicalConfig>::ADAPTIVE_FIELDS.is_empty(),
                    "EnvConfig: nested configs cannot have adaptive fields (ADR-0030)"
                );
            });
        }
        if attrs.adaptive && ident == "_reserved" {
            return Err(syn::Error::new_spanned(
                ident,
                "`_reserved` is used by the generated adaptive struct",
            ));
        }
        if matches!(last_ident(ty).as_deref(), Some("f32" | "f64")) {
            self.checks.push(quote! {
                if self.#ident.is_nan() {
                    return ::core::result::Result::Err(#core::ConfigError {
                        field: #key,
                        reason: "must not be NaN",
                    });
                }
            });
        }
        if attrs.adaptive {
            self.adaptive_fields.push((ident.clone(), ty.clone()));
            self.adaptive.push(key);
        } else {
            self.entries.push(Entry {
                key,
                ident: ident.clone(),
                nested,
            });
        }
        Ok(())
    }
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let core = core_path(&input.attrs)?;
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "EnvConfig does not support generics",
        ));
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            name,
            "EnvConfig can only be derived for structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "EnvConfig requires a struct with named fields",
        ));
    };

    let mut extra = None;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("config")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("validate") && extra.is_none() {
                extra = Some(meta.value()?.parse::<syn::Path>()?);
                Ok(())
            } else {
                Err(meta.error("expected one `validate = path`"))
            }
        })?;
    }
    let extra = extra.map(|path| quote!(#path(self)?;));
    let mut parts = ConfigParts::default();
    for field in &fields.named {
        parts.push(&core, field)?;
    }
    let ConfigParts {
        defaults,
        checks,
        mut entries,
        adaptive,
        adaptive_fields,
        static_checks,
        assets,
        nested_fields,
    } = parts;
    let asset_entries = asset_entries_fn(&core, &assets, &nested_fields);
    let adaptive_struct = adaptive_struct(&core, input, &adaptive_fields)?;
    entries.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
    let count = u16::try_from(entries.len())
        .map_err(|_| syn::Error::new_spanned(name, "too many config fields"))?;
    let decoder = crate::config_decode::decoder(&core, name, &entries);
    let canonical_decode = canonical_decode_impl(&core, name, &entries);
    let writes = entries.iter().map(|Entry { key, ident, nested }| {
        if *nested {
            quote!(#core::config::write_nested_entry(out, #key, &self.#ident);)
        } else {
            quote!(#core::config::write_entry(out, #key, &self.#ident);)
        }
    });

    Ok(quote! {
        impl ::core::default::Default for #name {
            fn default() -> Self {
                Self { #(#defaults)* }
            }
        }

        impl #core::EnvConfig for #name {
            fn validate(&self) -> ::core::result::Result<(), #core::ConfigError> {
                #(#checks)*
                #extra
                ::core::result::Result::Ok(())
            }

            #asset_entries
        }

        #adaptive_struct

        #decoder

        #(#static_checks)*

        impl #core::config::CanonicalConfig for #name {
            const ADAPTIVE_FIELDS: &'static [&'static str] = &[#(#adaptive),*];
            const STATIC_ENTRIES: u16 = #count;

            fn write_entries<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #(#writes)*
            }

        }

        #canonical_decode
    })
}

/// Size in bytes of a primitive config type, and the state type it is
/// stored as in the header (`bool` becomes `Bool8`).
fn header_type(ty: &Type) -> Option<(usize, TokenStream, bool)> {
    let name = last_ident(ty)?;
    let size = match name.as_str() {
        "u8" | "i8" | "bool" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" | "f32" | "Fx32" => 4,
        "u64" | "i64" | "f64" | "Fx64" => 8,
        _ => return None,
    };
    let is_bool = name == "bool";
    let stored = if is_bool { quote!(Bool8) } else { quote!(#ty) };
    Some((size, stored, is_bool))
}

/// `<Name>Adaptive`: the adaptive fields as a padding-free header struct
/// (ADR-0030, ADR-0036). Fields are ordered by size, largest first, and
/// padded to a multiple of 8 bytes.
fn adaptive_struct(
    core: &TokenStream,
    input: &DeriveInput,
    fields: &[(Ident, Type)],
) -> syn::Result<TokenStream> {
    if fields.is_empty() {
        return Ok(quote!());
    }
    let name = &input.ident;
    let vis = &input.vis;
    let adaptive = format_ident!("{}Adaptive", name);
    let mut typed = Vec::new();
    for (ident, ty) in fields {
        let Some((size, stored, is_bool)) = header_type(ty) else {
            return Err(syn::Error::new_spanned(
                ty,
                "adaptive fields must be primitives",
            ));
        };
        typed.push((size, ident, stored, is_bool));
    }
    typed.sort_by_key(|t| core::cmp::Reverse(t.0));
    let total: usize = typed.iter().map(|t| t.0).sum();
    let pad = (8 - total % 8) % 8;
    let field_defs = typed.iter().map(|(_, ident, stored, _)| {
        let stored = qualify(core, stored);
        quote!(pub #ident: #stored,)
    });
    let encodes = typed
        .iter()
        .map(|(_, ident, _, _)| quote!(#core::encoding::Canonical::encode(&self.#ident, out);));
    let decodes = typed
        .iter()
        .map(|(_, ident, _, _)| quote!(#ident: #core::encoding::Canonical::decode(input)?,));
    let inits = typed
        .iter()
        .map(|(_, ident, _, _)| quote!(#ident: #core::StateInit::state_default(),));
    let copies = typed.iter().map(|(_, ident, _, is_bool)| {
        if *is_bool {
            quote!(#ident: #core::Bool8::new(cfg.#ident),)
        } else {
            quote!(#ident: cfg.#ident,)
        }
    });
    let (pad_field, pad_encode, pad_decode, pad_state, pad_from) = if pad == 0 {
        (quote!(), quote!(), quote!(), quote!(), quote!())
    } else {
        (
            quote!(pub _reserved: #core::Reserved<#pad>,),
            quote!(#core::encoding::Canonical::encode(&self._reserved, out);),
            quote!(_reserved: #core::encoding::Canonical::decode(input)?,),
            quote!(_reserved: #core::Reserved::ZERO,),
            quote!(_reserved: #core::Reserved::ZERO,),
        )
    };
    let len = total + pad;
    Ok(quote! {
        /// Adaptive config values stored in the record header, generated by
        /// `#[derive(EnvConfig)]`.
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C)]
        #vis struct #adaptive {
            #(#field_defs)*
            #pad_field
        }

        const _: () = assert!(
            ::core::mem::size_of::<#adaptive>() == #len,
            "adaptive struct has padding"
        );

        impl #core::encoding::Canonical for #adaptive {
            const ENCODED_LEN: usize = #len;

            fn encode<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #(#encodes)*
                #pad_encode
            }

            fn decode(
                input: &mut #core::encoding::Decoder<'_>,
            ) -> ::core::result::Result<Self, #core::DecodeError> {
                ::core::result::Result::Ok(Self { #(#decodes)* #pad_decode })
            }
        }

        impl #core::StateInit for #adaptive {
            fn state_default() -> Self {
                Self { #(#inits)* #pad_state }
            }
        }

        impl #core::AdaptiveValues<#name> for #adaptive {
            fn from_config(cfg: &#name) -> Self {
                Self { #(#copies)* #pad_from }
            }
        }
    })
}

/// `Bool8` resolves through the core path; other types are written as
/// the user wrote them.
fn qualify(core: &TokenStream, stored: &TokenStream) -> TokenStream {
    if stored.to_string() == "Bool8" {
        quote!(#core::Bool8)
    } else {
        stored.clone()
    }
}

/// Primitive config values; anything else must be a nested `EnvConfig`.
fn is_primitive(ty: &Type) -> syn::Result<bool> {
    match ty {
        Type::Path(_) | Type::Group(_) => {
            Ok(last_ident(ty).is_some_and(|i| PRIMITIVES.contains(&i.as_str())))
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "config fields must be integers, f32, f64, bool, Fx32, Fx64, Asset<T> or nested #[derive(EnvConfig)] structs",
        )),
    }
}

/// `EnvConfig::asset_entries` for a config with `assets` and nested
/// configs `nested` (ADR-0077); nothing when it has neither.
fn asset_entries_fn(
    core: &TokenStream,
    assets: &[(Ident, String)],
    nested: &[Ident],
) -> TokenStream {
    if assets.is_empty() && nested.is_empty() {
        quote!()
    } else {
        let pushes = assets.iter().map(|(ident, name)| {
            quote!(out.push(#core::asset::AssetEntry {
                name: ::std::string::String::from(#name),
                hash: self.#ident.hash(),
            });)
        });
        let nested_calls = nested
            .iter()
            .map(|ident| quote!(#core::EnvConfig::asset_entries(&self.#ident, out);));
        quote! {
            fn asset_entries(&self, out: &mut ::std::vec::Vec<#core::asset::AssetEntry>) {
                #(#pushes)*
                #(#nested_calls)*
            }
        }
    }
}

/// `CanonicalDecode` for the config `name` with static fields `entries`
/// (#377): reads them back in their sorted order.
fn canonical_decode_impl(core: &TokenStream, name: &Ident, entries: &[Entry]) -> TokenStream {
    let reads = entries.iter().map(|Entry { key, ident, nested }| {
        if *nested {
            quote!(cfg.#ident = #core::config_canonical::read_nested_entry(input, #key)?;)
        } else {
            quote!(cfg.#ident = #core::config_canonical::read_entry(input, #key)?;)
        }
    });
    quote! {
        impl #core::config_canonical::CanonicalDecode for #name {
            fn read_entries(
                input: &mut #core::encoding::Decoder<'_>,
            ) -> ::core::result::Result<Self, #core::config_canonical::CanonicalDecodeError> {
                // Adaptive fields are not in the canonical form: they keep
                // their defaults (ADR-0030).
                #[allow(unused_mut, reason = "a config without static fields")]
                let mut cfg = <Self as ::core::default::Default>::default();
                #[allow(unused_variables, reason = "a config without static fields")]
                let input = input;
                #(#reads)*
                ::core::result::Result::Ok(cfg)
            }
        }
    }
}

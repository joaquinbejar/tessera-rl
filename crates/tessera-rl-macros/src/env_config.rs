// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(EnvConfig)]` (ADR-0030).
//!
//! Generates `Default` from `#[config(default = ...)]`, range validation,
//! and the canonical form (`CanonicalConfig`, keys sorted at compile time).
//! Fields marked `#[config(adaptive)]` are excluded from the canonical form
//! and the config hash. JSON Schema generation is deferred to phase 1b; the
//! header struct for adaptive values comes with the generated headers of
//! the engine (milestone 5, ADR-0036).

use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
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
            } else {
                return Err(meta.error(
                    "expected `default = ...`, `range = ...`, `adaptive`, `shape` or `capacity`",
                ));
            }
            Ok(())
        })?;
    }
    Ok(attrs)
}

struct Entry {
    key: String,
    ident: Ident,
    nested: bool,
}

#[derive(Default)]
struct ConfigParts {
    defaults: Vec<TokenStream>,
    checks: Vec<TokenStream>,
    entries: Vec<Entry>,
    adaptive: Vec<String>,
}

impl ConfigParts {
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
        let nested = !is_primitive(ty)?;
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

        self.defaults.push(if let Some(expr) = &attrs.default {
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

    let mut parts = ConfigParts::default();
    for field in &fields.named {
        parts.push(&core, field)?;
    }
    let ConfigParts {
        defaults,
        checks,
        mut entries,
        adaptive,
    } = parts;
    entries.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
    let count = u16::try_from(entries.len())
        .map_err(|_| syn::Error::new_spanned(name, "too many config fields"))?;
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
                ::core::result::Result::Ok(())
            }
        }

        impl #core::config::CanonicalConfig for #name {
            const ADAPTIVE_FIELDS: &'static [&'static str] = &[#(#adaptive),*];
            const STATIC_ENTRIES: u16 = #count;

            fn write_entries<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #(#writes)*
            }
        }
    })
}

/// Primitive config values; anything else must be a nested `EnvConfig`.
fn is_primitive(ty: &Type) -> syn::Result<bool> {
    match ty {
        Type::Path(_) | Type::Group(_) => {
            Ok(last_ident(ty).is_some_and(|i| PRIMITIVES.contains(&i.as_str())))
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "config fields must be integers, f32, f64, bool, Fx32, Fx64 or nested #[derive(EnvConfig)] structs",
        )),
    }
}

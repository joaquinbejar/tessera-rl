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
use quote::{ToTokens, format_ident, quote};
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
    adaptive_fields: Vec<(Ident, Type)>,
    static_checks: Vec<TokenStream>,
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
            // ADR-0030: only top-level fields can be adaptive.
            self.static_checks.push(quote! {
                assert!(
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
    } = parts;
    let adaptive_struct = adaptive_struct(&core, input, &adaptive_fields)?;
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

        #adaptive_struct

        const _: () = {
            #(#static_checks)*
        };

        impl #core::config::CanonicalConfig for #name {
            const ADAPTIVE_FIELDS: &'static [&'static str] = &[#(#adaptive),*];
            const STATIC_ENTRIES: u16 = #count;

            fn write_entries<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #(#writes)*
            }
        }
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
            "config fields must be integers, f32, f64, bool, Fx32, Fx64 or nested #[derive(EnvConfig)] structs",
        )),
    }
}

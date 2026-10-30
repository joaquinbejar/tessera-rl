// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(StateEnum)]` (ADR-0029).

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Ident, LitStr};

use crate::paths::core_path;

const REPRS: &[&str] = &["u8", "u16", "u32", "i8", "i16", "i32", "i64"];

fn repr_type(input: &DeriveInput) -> syn::Result<Ident> {
    let mut found = None;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("repr")) {
        attr.parse_nested_meta(|meta| {
            if let Some(ident) = meta.path.get_ident()
                && REPRS.iter().any(|r| ident == r)
            {
                found = Some(ident.clone());
            }
            Ok(())
        })?;
    }
    found.ok_or_else(|| {
        syn::Error::new_spanned(
            &input.ident,
            "StateEnum requires an integer representation: #[repr(u8)], #[repr(u16)], #[repr(u32)], #[repr(i8)], #[repr(i16)], #[repr(i32)] or #[repr(i64)]",
        )
    })
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let core = core_path(&input.attrs)?;
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "StateEnum does not support generics",
        ));
    }
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            name,
            "StateEnum can only be derived for enums",
        ));
    };
    let repr = repr_type(input)?;
    if data.variants.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "StateEnum requires at least one variant",
        ));
    }
    let mut defaults = 0usize;
    let mut arms = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "StateEnum variants must be unit variants",
            ));
        }
        if variant.attrs.iter().any(|a| a.path().is_ident("default")) {
            defaults += 1;
        }
        let v = &variant.ident;
        arms.push(quote! {
            if repr == (Self::#v as #repr) {
                return ::core::option::Option::Some(Self::#v);
            }
        });
    }
    if defaults != 1 {
        return Err(syn::Error::new_spanned(
            name,
            "StateEnum requires exactly one #[default] variant (with #[derive(Default)])",
        ));
    }
    let name_str = LitStr::new(&name.to_string(), name.span());

    Ok(quote! {
        impl #core::StateEnum for #name {
            type Repr = #repr;
            const NAME: &'static str = #name_str;

            #[inline]
            fn to_repr(self) -> #repr {
                self as #repr
            }

            #[inline]
            fn from_repr(repr: #repr) -> ::core::option::Option<Self> {
                #(#arms)*
                ::core::option::Option::None
            }
        }

        impl #core::StateInit for #name {
            #[inline]
            fn state_default() -> Self {
                <Self as ::core::default::Default>::default()
            }
        }

        impl #core::encoding::Canonical for #name {
            const ENCODED_LEN: usize = <#repr as #core::encoding::Canonical>::ENCODED_LEN;

            #[inline]
            fn encode<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #core::encoding::encode_state_enum(*self, out);
            }

            #[inline]
            fn decode(
                input: &mut #core::encoding::Decoder<'_>,
            ) -> ::core::result::Result<Self, #core::DecodeError> {
                #core::encoding::decode_state_enum(input)
            }
        }
    })
}

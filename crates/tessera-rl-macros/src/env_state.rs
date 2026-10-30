// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(EnvState)]` (SPEC 5.1, ADR-0029).
//!
//! Lane-wise Struct-of-Arrays generation for `LaneEnvironment` is deferred
//! to phase 3 (SIMD); the `NoUninit` bound for memcpy snapshots comes with
//! the engine (milestone 5).

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Fields, Ident, LitStr};

use crate::paths::core_path;
use crate::types::{check_state_field, is_float};

/// Returns whether `#[repr(...)]` contains `C`; rejects `packed`.
fn has_repr_c(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut found = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("repr")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("C") {
                found = true;
            } else if meta.path.is_ident("packed") {
                return Err(meta.error(
                    "#[repr(packed)] is not allowed in state: unaligned fields (ADR-0029)",
                ));
            }
            if meta.input.peek(syn::token::Paren) {
                let content;
                syn::parenthesized!(content in meta.input);
                content.parse::<TokenStream>()?;
            }
            Ok(())
        })?;
    }
    Ok(found)
}

fn allow_nan(attrs: &[Attribute]) -> syn::Result<bool> {
    let mut found = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("state")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("allow_nan") {
                found = true;
                Ok(())
            } else {
                Err(meta.error("expected `allow_nan`"))
            }
        })?;
    }
    Ok(found)
}

#[derive(Default)]
struct FieldParts {
    padding_checks: Vec<TokenStream>,
    bounds: Vec<TokenStream>,
    lens: Vec<TokenStream>,
    encodes: Vec<TokenStream>,
    decodes: Vec<TokenStream>,
    inits: Vec<TokenStream>,
}

impl FieldParts {
    fn push(
        &mut self,
        core: &TokenStream,
        name: &Ident,
        ident: &Ident,
        field: &syn::Field,
    ) -> syn::Result<()> {
        let ty = &field.ty;
        check_state_field(ty)?;
        let nan = allow_nan(&field.attrs)?;
        if nan && !is_float(ty) {
            return Err(syn::Error::new_spanned(
                ty,
                "#[state(allow_nan)] applies only to f32, f64 and arrays of them",
            ));
        }
        let message = LitStr::new(
            &format!(
                "EnvState: `{name}` has padding before field `{ident}`; insert a Reserved<N> field before it, with N = offset_of!({name}, {ident}) minus the end of the previous field (ADR-0029)"
            ),
            ident.span(),
        );
        self.padding_checks.push(quote! {
            if ::core::mem::offset_of!(#name, #ident) != end {
                ::core::panic!(#message);
            }
            end = ::core::mem::offset_of!(#name, #ident) + ::core::mem::size_of::<#ty>();
        });
        self.bounds.push(quote!(assert_field::<#ty>();));
        self.lens
            .push(quote!(<#ty as #core::encoding::Canonical>::ENCODED_LEN));
        self.encodes
            .push(quote!(#core::encoding::Canonical::encode(&self.#ident, out);));
        self.inits
            .push(quote!(#ident: <#ty as #core::StateInit>::state_default(),));
        self.decodes.push(if nan {
            quote!(#ident: <#ty as #core::encoding::DecodeAllowNan>::decode_allow_nan(input)?,)
        } else {
            quote!(#ident: <#ty as #core::encoding::Canonical>::decode(input)?,)
        });
        Ok(())
    }
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let core = core_path(&input.attrs)?;
    let name = &input.ident;
    if !has_repr_c(&input.attrs)? {
        return Err(syn::Error::new_spanned(
            name,
            "EnvState requires #[repr(C)] (ADR-0029)",
        ));
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            name,
            "EnvState can only be derived for structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "EnvState requires a struct with named fields",
        ));
    };

    // Field types first, so a reference reports itself rather than the
    // lifetime parameter it needs.
    let mut parts = FieldParts::default();
    let mut last = None;
    for field in &fields.named {
        let Some(ident) = &field.ident else { continue };
        parts.push(&core, name, ident, field)?;
        last = Some(ident.clone());
    }
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "EnvState does not support generic state types",
        ));
    }
    let FieldParts {
        padding_checks,
        bounds,
        lens,
        encodes,
        decodes,
        inits,
    } = parts;
    let trailing = LitStr::new(
        &match &last {
            Some(ident) => format!(
                "EnvState: `{name}` has trailing padding after field `{ident}`; add a Reserved<N> field at the end, with N = size_of::<{name}>() minus the end of `{ident}` (ADR-0029)"
            ),
            None => format!("EnvState: `{name}` has no fields"),
        },
        name.span(),
    );

    Ok(quote! {
        const _: () = {
            let mut end: usize = 0;
            #(#padding_checks)*
            if ::core::mem::size_of::<#name>() != end {
                ::core::panic!(#trailing);
            }
        };

        const _: () = {
            const fn assert_field<T: #core::encoding::Canonical + #core::StateInit + ::core::marker::Copy>() {}
            const fn assert_state<T: ::core::marker::Copy>() {}
            #(#bounds)*
            assert_state::<#name>();
        };

        impl #core::encoding::Canonical for #name {
            const ENCODED_LEN: usize = 0 #(+ #lens)*;

            #[inline]
            fn encode<S: #core::encoding::Sink>(&self, out: &mut #core::encoding::Encoder<S>) {
                #(#encodes)*
            }

            #[inline]
            fn decode(
                input: &mut #core::encoding::Decoder<'_>,
            ) -> ::core::result::Result<Self, #core::DecodeError> {
                ::core::result::Result::Ok(Self { #(#decodes)* })
            }
        }

        impl #core::StateInit for #name {
            #[inline]
            fn state_default() -> Self {
                Self { #(#inits)* }
            }
        }

        impl #core::EnvState for #name {}
    })
}

// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(Observation)]` (SPEC 10, ADR-0014).
//!
//! Each field is an `ObsLeaf<T>` with one `#[space(...)]` attribute:
//! `box(low = .., high = .., shape = .., dtype = ..)`, `discrete(n = ..)` or
//! `multi_binary(shape = ..)`. Values are expressions or closures over the
//! config (`|c: &Config| ...`), resolved at construction. The derive
//! generates `Observation<Config>` (a `Dict` in declaration order) and one
//! `LeafId<T>` constant per field, named after the field in upper case.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::meta::ParseNestedMeta;
use syn::{Data, DeriveInput, Expr, Fields, Ident, Pat, Type};

use crate::paths::core_path;
use crate::types::obs_leaf_inner;

struct BoxLeaf {
    low: Expr,
    high: Expr,
    shape: Expr,
    dtype: Option<Ident>,
}

enum Leaf {
    Box(Box<BoxLeaf>),
    Discrete { n: Expr },
    MultiBinary { shape: Expr },
}

fn parse_params(meta: &ParseNestedMeta<'_>, names: &[&str]) -> syn::Result<Vec<(String, Expr)>> {
    let mut params = Vec::new();
    meta.parse_nested_meta(|inner| {
        let Some(key) = inner.path.get_ident().map(ToString::to_string) else {
            return Err(inner.error("expected a parameter name"));
        };
        if !names.contains(&key.as_str()) {
            return Err(inner.error(format!("expected one of {names:?}")));
        }
        params.push((key, inner.value()?.parse::<Expr>()?));
        Ok(())
    })?;
    Ok(params)
}

fn take(params: &mut Vec<(String, Expr)>, key: &str, span: &Ident) -> syn::Result<Expr> {
    match params.iter().position(|(k, _)| k == key) {
        Some(i) => Ok(params.swap_remove(i).1),
        None => Err(syn::Error::new_spanned(span, format!("missing `{key}`"))),
    }
}

fn parse_leaf(field: &syn::Field, ident: &Ident) -> syn::Result<Leaf> {
    let mut leaf = None;
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("space")) {
        attr.parse_nested_meta(|meta| {
            let kind = meta.path.get_ident().map(ToString::to_string);
            let parsed = match kind.as_deref() {
                Some("box") => {
                    let mut p = parse_params(&meta, &["low", "high", "shape", "dtype"])?;
                    let dtype = match p.iter().position(|(k, _)| k == "dtype") {
                        Some(i) => match p.swap_remove(i).1 {
                            Expr::Path(path) => path.path.get_ident().cloned(),
                            other => {
                                return Err(syn::Error::new_spanned(
                                    other,
                                    "expected a dtype name",
                                ));
                            }
                        },
                        None => None,
                    };
                    Leaf::Box(Box::new(BoxLeaf {
                        low: take(&mut p, "low", ident)?,
                        high: take(&mut p, "high", ident)?,
                        shape: take(&mut p, "shape", ident)?,
                        dtype,
                    }))
                }
                Some("discrete") => {
                    let mut p = parse_params(&meta, &["n"])?;
                    Leaf::Discrete {
                        n: take(&mut p, "n", ident)?,
                    }
                }
                Some("multi_binary") => {
                    let mut p = parse_params(&meta, &["shape"])?;
                    Leaf::MultiBinary {
                        shape: take(&mut p, "shape", ident)?,
                    }
                }
                _ => {
                    return Err(
                        meta.error("expected `box(..)`, `discrete(..)` or `multi_binary(..)`")
                    );
                }
            };
            if leaf.replace(parsed).is_some() {
                return Err(meta.error("one space per leaf"));
            }
            Ok(())
        })?;
    }
    leaf.ok_or_else(|| syn::Error::new_spanned(ident, "missing #[space(...)] attribute"))
}

/// The config type named by a closure's typed parameter `|c: &Config|`.
fn closure_config(expr: &Expr) -> syn::Result<Option<Type>> {
    let Expr::Closure(closure) = expr else {
        return Ok(None);
    };
    match closure.inputs.first() {
        Some(Pat::Type(typed)) if closure.inputs.len() == 1 => match &*typed.ty {
            Type::Reference(reference) => Ok(Some((*reference.elem).clone())),
            other => Err(syn::Error::new_spanned(
                other,
                "expected `|c: &Config| ...`",
            )),
        },
        _ => Err(syn::Error::new_spanned(
            closure,
            "expected `|c: &Config| ...`",
        )),
    }
}

/// An expression evaluated against `cfg`: closures are called with it.
fn eval(expr: &Expr) -> TokenStream {
    if matches!(expr, Expr::Closure(_)) {
        quote!((#expr)(cfg))
    } else {
        quote!(#expr)
    }
}

const DTYPES: &[(&str, &str)] = &[
    ("bool", "Bool"),
    ("u8", "U8"),
    ("i8", "I8"),
    ("u16", "U16"),
    ("i16", "I16"),
    ("u32", "U32"),
    ("i32", "I32"),
    ("i64", "I64"),
    ("f32", "F32"),
    ("f64", "F64"),
];

fn leaf_space(
    core: &TokenStream,
    elem: &Type,
    leaf: &Leaf,
    dtype_checks: &mut Vec<TokenStream>,
) -> syn::Result<TokenStream> {
    Ok(match leaf {
        Leaf::Box(b) => {
            let BoxLeaf {
                low,
                high,
                shape,
                dtype,
            } = &**b;
            if let Some(dtype) = dtype {
                let Some((_, variant)) = DTYPES.iter().find(|(n, _)| dtype == n) else {
                    return Err(syn::Error::new_spanned(
                        dtype,
                        "unsupported dtype (ADR-0014)",
                    ));
                };
                let variant = format_ident!("{}", variant);
                dtype_checks.push(quote! {
                    assert!(
                        matches!(<#elem as #core::LeafDType>::DTYPE, #core::space::DType::#variant),
                        "Observation: the dtype attribute does not match the leaf type"
                    );
                });
            }
            let (low, high, shape) = (eval(low), eval(high), eval(shape));
            quote! {
                #core::space::Space::Box {
                    shape: shape(#shape)?,
                    dtype: <#elem as #core::LeafDType>::DTYPE,
                    low: ::std::vec![(#low) as f64],
                    high: ::std::vec![(#high) as f64],
                }
            }
        }
        Leaf::Discrete { n } => {
            let n = eval(n);
            quote! {
                #core::space::Space::Discrete {
                    n: u32::try_from(#n).map_err(|_| #core::SpaceError::ZeroChoices)?,
                }
            }
        }
        Leaf::MultiBinary { shape } => {
            let shape = eval(shape);
            quote!(#core::space::Space::MultiBinary { shape: shape(#shape)? })
        }
    })
}

pub(crate) fn derive(input: &DeriveInput) -> syn::Result<TokenStream> {
    let core = core_path(&input.attrs)?;
    let name = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "Observation does not support generics",
        ));
    }
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            name,
            "Observation can only be derived for structs",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "Observation requires a struct with named fields",
        ));
    };

    let mut config: Option<Type> = None;
    let mut ids = Vec::new();
    let mut spaces = Vec::new();
    let mut dtype_checks = Vec::new();
    for (index, field) in fields.named.iter().enumerate() {
        let Some(ident) = &field.ident else { continue };
        let Some(elem) = obs_leaf_inner(&field.ty) else {
            return Err(syn::Error::new_spanned(
                &field.ty,
                "Observation fields must have type ObsLeaf<T>",
            ));
        };
        let leaf = parse_leaf(field, ident)?;
        let exprs: Vec<&Expr> = match &leaf {
            Leaf::Box(b) => vec![&b.low, &b.high, &b.shape],
            Leaf::Discrete { n } => vec![n],
            Leaf::MultiBinary { shape } => vec![shape],
        };
        for expr in exprs {
            if let Some(ty) = closure_config(expr)? {
                match &config {
                    Some(seen) if quote!(#seen).to_string() != quote!(#ty).to_string() => {
                        return Err(syn::Error::new_spanned(
                            ty,
                            "all closures must take the same config type",
                        ));
                    }
                    _ => config = Some(ty),
                }
            }
        }

        let id = format_ident!("{}", ident.to_string().to_uppercase());
        ids.push(quote! {
            /// Leaf id generated by `#[derive(Observation)]`.
            pub const #id: #core::LeafId<#elem> = #core::LeafId::new(#index);
        });
        let key = ident.to_string();
        let space = leaf_space(&core, elem, &leaf, &mut dtype_checks)?;
        spaces.push(quote!((::std::string::String::from(#key), #space)));
    }
    let leaves = spaces.len();
    let field_idents = fields.named.iter().filter_map(|f| f.ident.as_ref());
    let (impl_generics, config_ty) = match &config {
        Some(ty) => (quote!(), quote!(#ty)),
        None => (quote!(<C>), quote!(C)),
    };

    Ok(quote! {
        impl #name {
            #(#ids)*
        }

        const _: () = {
            #(#dtype_checks)*
        };

        // Marker fields hold no data; reading them here keeps dead-code
        // lints quiet in the author's crate.
        const _: fn(&#name) = |obs| {
            let _ = (#(&obs.#field_idents,)*);
        };

        impl #impl_generics #core::Observation<#config_ty> for #name {
            const LEAVES: usize = #leaves;

            fn space(
                cfg: &#config_ty,
            ) -> ::core::result::Result<#core::space::Space, #core::SpaceError> {
                let _ = cfg;
                fn shape<D, const R: usize>(
                    raw: [D; R],
                ) -> ::core::result::Result<::std::vec::Vec<u32>, #core::SpaceError>
                where
                    u32: ::core::convert::TryFrom<D>,
                {
                    raw.into_iter()
                        .map(|d| u32::try_from(d).map_err(|_| #core::SpaceError::InvalidShape {
                            shape: ::std::vec::Vec::new(),
                        }))
                        .collect()
                }
                let space = #core::space::Space::Dict(::std::vec![#(#spaces),*]);
                space.validate()?;
                ::core::result::Result::Ok(space)
            }
        }
    })
}

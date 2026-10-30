// SPDX-License-Identifier: MIT OR Apache-2.0

//! Syntactic checks on field types.

use syn::{GenericArgument, PathArguments, Type};

/// Types that can never be state fields, with the reason shown to authors.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "bool",
        "`bool` is not allowed in state; use `Bool8` (ADR-0029)",
    ),
    (
        "Vec",
        "heap types are not allowed in state; use `Pool<T, CAP>` or an array",
    ),
    ("String", "heap types are not allowed in state"),
    ("Box", "heap types are not allowed in state"),
    ("Rc", "pointers are not allowed in state"),
    ("Arc", "pointers are not allowed in state"),
    ("Cell", "interior mutability is not allowed in state"),
    ("RefCell", "interior mutability is not allowed in state"),
    (
        "Option",
        "`Option` is not allowed in state; use a `Bool8` flag or `Pool`",
    ),
    ("HashMap", "maps are not allowed in state"),
    ("BTreeMap", "maps are not allowed in state"),
    ("HashSet", "sets are not allowed in state"),
    ("BTreeSet", "sets are not allowed in state"),
    ("VecDeque", "heap types are not allowed in state"),
    (
        "usize",
        "`usize` is platform-dependent; use a fixed-width integer",
    ),
    (
        "isize",
        "`isize` is platform-dependent; use a fixed-width integer",
    ),
    ("char", "`char` is not allowed in state; use `u32`"),
];

/// Rejects types that cannot be plain-data state fields.
pub(crate) fn check_state_field(ty: &Type) -> syn::Result<()> {
    match ty {
        Type::Reference(_) => Err(syn::Error::new_spanned(
            ty,
            "references are not allowed in state (ADR-0029)",
        )),
        Type::Ptr(_) => Err(syn::Error::new_spanned(
            ty,
            "raw pointers are not allowed in state (ADR-0029)",
        )),
        Type::Tuple(_) => Err(syn::Error::new_spanned(
            ty,
            "tuples are not allowed in state; use a #[derive(EnvState)] struct",
        )),
        Type::Array(array) => check_state_field(&array.elem),
        Type::Paren(inner) => check_state_field(&inner.elem),
        Type::Group(inner) => check_state_field(&inner.elem),
        Type::Path(path) => {
            let Some(last) = path.path.segments.last() else {
                return Ok(());
            };
            if let Some((_, reason)) = FORBIDDEN.iter().find(|(name, _)| last.ident == name) {
                return Err(syn::Error::new_spanned(ty, *reason));
            }
            if let PathArguments::AngleBracketed(args) = &last.arguments {
                for arg in &args.args {
                    if let GenericArgument::Type(inner) = arg {
                        check_state_field(inner)?;
                    }
                }
            }
            Ok(())
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "this type is not allowed in state (ADR-0029)",
        )),
    }
}

/// Returns whether `ty` is `f32`, `f64` or an array of them.
pub(crate) fn is_float(ty: &Type) -> bool {
    match ty {
        Type::Array(array) => is_float(&array.elem),
        Type::Paren(inner) => is_float(&inner.elem),
        Type::Group(inner) => is_float(&inner.elem),
        Type::Path(path) => path.path.is_ident("f32") || path.path.is_ident("f64"),
        _ => false,
    }
}

/// Returns the last path segment's identifier, if `ty` is a plain path.
pub(crate) fn last_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(path) if path.qself.is_none() => {
            path.path.segments.last().map(|s| s.ident.to_string())
        }
        Type::Group(inner) => last_ident(&inner.elem),
        _ => None,
    }
}

/// Returns `T` if `ty` is `ObsLeaf<T>`.
pub(crate) fn obs_leaf_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else { return None };
    let last = path.path.segments.last()?;
    if last.ident != "ObsLeaf" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    match args.args.first()? {
        GenericArgument::Type(inner) if args.args.len() == 1 => Some(inner),
        _ => None,
    }
}

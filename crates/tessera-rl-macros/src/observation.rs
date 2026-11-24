// SPDX-License-Identifier: MIT OR Apache-2.0

//! `#[derive(Observation)]` (SPEC 10, ADR-0014).
//!
//! Each field is an `ObsLeaf<T>` with one `#[space(...)]` attribute:
//! `box(low = .., high = .., shape = .., dtype = ..)`, `discrete(n = ..)` or
//! `multi_binary(shape = ..)`; or a `Set` (ADR-0082) in one of two forms:
//!
//! - `ObsSet<T>` with `set(max = .., inner(<one of those>))`: a single-leaf
//!   item, exported as the item leaf `p` and the mask `p_mask`, with a
//!   `SetId<T>` constant;
//! - `ObsSetOf<I>` with `set(max = .., items = I)` (#429), where `I` is itself
//!   `#[derive(Observation)]` with the same config type: the inner space is
//!   `I`'s `Dict`, exported as one item leaf `p/<field>` per field of `I` in
//!   `I`'s declaration order, then `p_mask`, with a `SetOf<I>` constant. The
//!   items of a set may not contain a set: refused at compile time through
//!   `ObservationLayout::CONTAINS_SET`, at the field.
//!
//! Values are expressions or closures over the config (`|c: &Config| ...`),
//! resolved at construction. The derive generates `Observation<Config>` (a
//! `Dict` in declaration order, with `CONTAINS_SET` true when a field is a
//! set), `ObservationLayout`, one id constant per field named after the
//! field in upper case (`LeafId<T>`, `SetId<T>` or `SetOf<I>`), and, for
//! every `ObsLeaf<T>` field, an `ItemId<Self, T>` constant `<FIELD>_ITEM`
//! that names the field's leaf inside a `Set` of this struct
//! (`SetOfLeaf::item`); the item ids are bound to the struct, so another
//! struct's id does not type-check. Exported names must be unique: a field
//! named like a set's `<set>_mask` is refused.

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::meta::ParseNestedMeta;
use syn::{DeriveInput, Expr, Ident, Pat, Type};

use crate::paths::core_path;
use crate::types::{named_fields, obs_leaf_inner, obs_set_inner, obs_set_of_inner};

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
        if params.iter().any(|(k, _)| *k == key) {
            return Err(inner.error(format!("duplicate `{key}`")));
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

/// One leaf kind: `box(..)`, `discrete(..)` or `multi_binary(..)`.
fn parse_kind(meta: &ParseNestedMeta<'_>, ident: &Ident) -> syn::Result<Leaf> {
    let kind = meta.path.get_ident().map(ToString::to_string);
    Ok(match kind.as_deref() {
        Some("box") => {
            let mut p = parse_params(meta, &["low", "high", "shape", "dtype"])?;
            let dtype = match p.iter().position(|(k, _)| k == "dtype") {
                Some(i) => match p.swap_remove(i).1 {
                    Expr::Path(path) => path.path.get_ident().cloned(),
                    other => {
                        return Err(syn::Error::new_spanned(other, "expected a dtype name"));
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
            let mut p = parse_params(meta, &["n"])?;
            Leaf::Discrete {
                n: take(&mut p, "n", ident)?,
            }
        }
        Some("multi_binary") => {
            let mut p = parse_params(meta, &["shape"])?;
            Leaf::MultiBinary {
                shape: take(&mut p, "shape", ident)?,
            }
        }
        _ => {
            return Err(meta.error("expected `box(..)`, `discrete(..)` or `multi_binary(..)`"));
        }
    })
}

/// The inner of a set: one leaf, or an item struct (#429).
enum SetInner {
    Leaf(Leaf),
    Items(Type),
}

/// `set(max = EXPR, inner(<leaf kind>))` or `set(max = EXPR, items = I)`
/// (ADR-0082): the capacity and the inner.
fn parse_set(meta: &ParseNestedMeta<'_>, ident: &Ident) -> syn::Result<(Expr, SetInner)> {
    let mut max = None;
    let mut inner = None;
    meta.parse_nested_meta(|param| {
        if param.path.is_ident("max") {
            if max.replace(param.value()?.parse::<Expr>()?).is_some() {
                return Err(param.error("duplicate `max`"));
            }
        } else if param.path.is_ident("inner") {
            param.parse_nested_meta(|kind| {
                if inner
                    .replace(SetInner::Leaf(parse_kind(&kind, ident)?))
                    .is_some()
                {
                    return Err(kind.error("one inner per set: `inner(..)` or `items = ..`"));
                }
                Ok(())
            })?;
        } else if param.path.is_ident("items") {
            let items = param.value()?.parse::<Type>()?;
            if inner.replace(SetInner::Items(items)).is_some() {
                return Err(param.error("one inner per set: `inner(..)` or `items = ..`"));
            }
        } else {
            return Err(param.error("expected `max`, `inner(..)` or `items = ..`"));
        }
        Ok(())
    })?;
    let max = max.ok_or_else(|| syn::Error::new_spanned(ident, "missing `max`"))?;
    let inner = inner
        .ok_or_else(|| syn::Error::new_spanned(ident, "missing `inner(..)` or `items = ..`"))?;
    Ok((max, inner))
}

/// A field's space: one leaf, or a set with its `max`.
enum FieldSpace {
    Leaf(Leaf),
    Set {
        max: Box<Expr>,
        inner: Box<SetInner>,
    },
}

fn parse_leaf(field: &syn::Field, ident: &Ident) -> syn::Result<FieldSpace> {
    let mut parsed_space = None;
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("space")) {
        attr.parse_nested_meta(|meta| {
            let parsed = if meta.path.is_ident("set") {
                let (max, inner) = parse_set(&meta, ident)?;
                FieldSpace::Set {
                    max: Box::new(max),
                    inner: Box::new(inner),
                }
            } else {
                FieldSpace::Leaf(parse_kind(&meta, ident)?)
            };
            if parsed_space.replace(parsed).is_some() {
                return Err(meta.error("one space per leaf"));
            }
            Ok(())
        })?;
    }
    parsed_space.ok_or_else(|| syn::Error::new_spanned(ident, "missing #[space(...)] attribute"))
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
    ident: &Ident,
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
                dtype_checks.push(quote_spanned! {ident.span()=>
                    const _: () = assert!(
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
                    bounds: <#elem as #core::LeafDType>::box_bounds(#low, #high),
                }
            }
        }
        Leaf::Discrete { n } => {
            dtype_checks.push(quote_spanned! {ident.span()=>
                const _: () = assert!(
                    matches!(<#elem as #core::LeafDType>::DTYPE, #core::space::DType::I32),
                    "Observation: discrete leaves must be ObsLeaf<i32>"
                );
            });
            let n = eval(n);
            quote! {
                #core::space::Space::Discrete {
                    n: u32::try_from(#n).map_err(|_| #core::SpaceError::InvalidChoices { n: u32::MAX })?,
                }
            }
        }
        Leaf::MultiBinary { shape } => {
            dtype_checks.push(quote_spanned! {ident.span()=>
                const _: () = assert!(
                    matches!(<#elem as #core::LeafDType>::DTYPE, #core::space::DType::Bool),
                    "Observation: multi_binary leaves must be ObsLeaf<Bool8>"
                );
            });
            let shape = eval(shape);
            quote!(#core::space::Space::MultiBinary { shape: shape(#shape)? })
        }
    })
}

/// Records the config type of a closure `expr`; an error if an earlier
/// closure took another one.
fn same_config(expr: &Expr, config: &mut Option<Type>) -> syn::Result<()> {
    if let Some(ty) = closure_config(expr)? {
        match config {
            Some(seen) if quote!(#seen).to_string() != quote!(#ty).to_string() => {
                return Err(syn::Error::new_spanned(
                    ty,
                    "all closures must take the same config type",
                ));
            }
            _ => *config = Some(ty),
        }
    }
    Ok(())
}

/// A leaf index as a constant expression: a fixed count plus the leaf
/// counts of the item structs of earlier `items =` sets.
#[derive(Default)]
struct Offset {
    fixed: usize,
    items: Vec<TokenStream>,
}

impl Offset {
    fn tokens(&self) -> TokenStream {
        let fixed = self.fixed;
        let items = &self.items;
        quote!(#fixed #(+ #items)*)
    }

    /// Adds `n` fixed leaves; an error at `ident` on overflow.
    fn advance(&mut self, n: usize, ident: &Ident) -> syn::Result<()> {
        self.fixed = self
            .fixed
            .checked_add(n)
            .ok_or_else(|| syn::Error::new_spanned(ident, "too many observation leaves"))?;
        Ok(())
    }
}

/// One parsed field: its ident, marker element or item type, and space.
struct Parsed<'a> {
    ident: &'a Ident,
    elem: &'a Type,
    space: FieldSpace,
}

/// Checks a field's marker type against its space and returns the parsed
/// field.
fn parse_field<'a>(field: &'a syn::Field, ident: &'a Ident) -> syn::Result<Parsed<'a>> {
    let leaf_elem = obs_leaf_inner(&field.ty);
    let set_elem = obs_set_inner(&field.ty);
    let set_of_elem = obs_set_of_inner(&field.ty);
    let Some(elem) = leaf_elem.or(set_elem).or(set_of_elem) else {
        return Err(syn::Error::new_spanned(
            &field.ty,
            "Observation fields must have type ObsLeaf<T>, ObsSet<T> or ObsSetOf<I>",
        ));
    };
    let space = parse_leaf(field, ident)?;
    let set_inner = match &space {
        FieldSpace::Leaf(_) => None,
        FieldSpace::Set { inner, .. } => Some(&**inner),
    };
    match set_inner {
        None if leaf_elem.is_some() => {}
        Some(SetInner::Leaf(_)) if set_elem.is_some() => {}
        Some(SetInner::Items(items)) if set_of_elem.is_some() => {
            if quote!(#items).to_string() != quote!(#elem).to_string() {
                return Err(syn::Error::new_spanned(
                    items,
                    "`items = I` must name the `I` of the field's `ObsSetOf<I>`",
                ));
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &field.ty,
                "ObsSet<T> fields take `#[space(set(.., inner(..)))]`, ObsSetOf<I> fields \
                 `#[space(set(.., items = I))]`, and only they do",
            ));
        }
    }
    Ok(Parsed { ident, elem, space })
}

/// The config closures of one leaf kind.
fn leaf_exprs(leaf: &Leaf) -> Vec<&Expr> {
    match leaf {
        Leaf::Box(b) => vec![&b.low, &b.high, &b.shape],
        Leaf::Discrete { n } => vec![n],
        Leaf::MultiBinary { shape } => vec![shape],
    }
}

/// The config closures of a field's space.
fn field_exprs(space: &FieldSpace) -> Vec<&Expr> {
    match space {
        FieldSpace::Leaf(leaf) => leaf_exprs(leaf),
        FieldSpace::Set { max, inner } => {
            let mut exprs = match &**inner {
                SetInner::Leaf(leaf) => leaf_exprs(leaf),
                SetInner::Items(_) => Vec::new(),
            };
            exprs.push(max);
            exprs
        }
    }
}

/// The `Space::Set` entry of a set field `key`.
fn set_entry(core: &TokenStream, key: &str, inner: &TokenStream, max: &Expr) -> TokenStream {
    let max = eval(max);
    quote!((
        ::std::string::String::from(#key),
        #core::space::Space::Set {
            inner: ::std::boxed::Box::new(#inner),
            max: u32::try_from(#max).map_err(|_| #core::SpaceError::InvalidShape {
                shape: ::std::vec::Vec::new(),
            })?,
        },
    ))
}

/// Generated tokens of one struct.
#[derive(Default)]
struct Generated {
    ids: Vec<TokenStream>,
    spaces: Vec<TokenStream>,
    checks: Vec<TokenStream>,
    bounds: Vec<TokenStream>,
}

/// One field's id constants, `Dict` entry and checks; advances `offset`
/// by its leaf count (ADR-0082: a set's item leaves, then its mask).
fn field_tokens(
    core: &TokenStream,
    name: &Ident,
    config_ty: &TokenStream,
    field: &Parsed<'_>,
    offset: &mut Offset,
    out: &mut Generated,
) -> syn::Result<()> {
    let Parsed { ident, elem, space } = field;
    let id = format_ident!("{}", ident.to_string().to_uppercase());
    let key = ident.to_string();
    let index = offset.tokens();
    match space {
        FieldSpace::Leaf(leaf) => {
            let item = format_ident!("{}_ITEM", ident.to_string().to_uppercase());
            let space = leaf_space(core, ident, elem, leaf, &mut out.checks)?;
            out.ids.push(quote! {
                /// Leaf id generated by `#[derive(Observation)]`.
                pub const #id: #core::LeafId<#elem> = #core::LeafId::new(#index);
                /// Item id generated by `#[derive(Observation)]`: this leaf
                /// inside a `Set` of this struct (`SetOfLeaf::item`).
                pub const #item: #core::ItemId<#name, #elem> = #core::ItemId::new(#index);
            });
            out.spaces
                .push(quote!((::std::string::String::from(#key), #space)));
            offset.advance(1, ident)?;
        }
        FieldSpace::Set { max, inner } => match &**inner {
            SetInner::Leaf(leaf) => {
                let space = leaf_space(core, ident, elem, leaf, &mut out.checks)?;
                out.ids.push(quote! {
                    /// Set id generated by `#[derive(Observation)]`.
                    pub const #id: #core::SetId<#elem> = #core::SetId::new(#index, #index + 1);
                });
                out.spaces.push(set_entry(core, &key, &space, max));
                offset.advance(2, ident)?;
            }
            SetInner::Items(items) => {
                let leaves = quote!(<#items as #core::ObservationLayout>::LEAVES);
                out.ids.push(quote! {
                    /// Set id generated by `#[derive(Observation)]`.
                    pub const #id: #core::SetOf<#items> =
                        #core::SetOf::new(#index, #index + #leaves);
                });
                out.checks.push(quote_spanned! {ident.span()=>
                    const _: () = assert!(
                        !<#items as #core::ObservationLayout>::CONTAINS_SET,
                        "Observation: the items of a set may not contain a set (ADR-0082)"
                    );
                });
                out.bounds
                    .push(quote!(#items: #core::Observation<#config_ty>));
                // The item space's leaves must match the config-free count the
                // ids use; checked when the pool builds the space.
                let inner = quote!({
                    let inner = <#items as #core::Observation<#config_ty>>::space(cfg)?;
                    ::core::debug_assert_eq!(
                        inner.leaf_count(),
                        ::core::option::Option::Some(<#items as #core::ObservationLayout>::LEAVES),
                        "Observation: the items' space and ObservationLayout::LEAVES disagree"
                    );
                    inner
                });
                out.spaces.push(set_entry(core, &key, &inner, max));
                offset.advance(1, ident)?;
                offset.items.push(leaves);
            }
        },
    }
    Ok(())
}

/// Refuses a field whose exported name is a set's mask, and a field `x_item`
/// beside a leaf field `x`, whose generated `X_ITEM` it would take.
fn check_names(fields: &[(&syn::Field, &Ident)]) -> syn::Result<()> {
    // Exported names (ADR-0082 point 2): a set `p` also exports `p_mask`,
    // which no other field may be named.
    let is_set =
        |f: &syn::Field| obs_set_inner(&f.ty).is_some() || obs_set_of_inner(&f.ty).is_some();
    let set_masks: Vec<String> = fields
        .iter()
        .filter(|(f, _)| is_set(f))
        .map(|(_, i)| format!("{i}_mask"))
        .collect();
    let leaves: Vec<&Ident> = fields
        .iter()
        .filter(|(f, _)| obs_leaf_inner(&f.ty).is_some())
        .map(|(_, i)| *i)
        .collect();
    for (_, ident) in fields {
        if set_masks.iter().any(|m| *ident == m) {
            return Err(syn::Error::new_spanned(
                ident,
                format!(
                    "duplicate observation leaf name `{ident}`: a set exports its mask under this name (ADR-0082)"
                ),
            ));
        }
        if let Some(leaf) = leaves
            .iter()
            .find(|l| format!("{l}_item") == ident.to_string())
        {
            let item = format!("{leaf}_ITEM").to_uppercase();
            return Err(syn::Error::new_spanned(
                ident,
                format!("`{ident}` collides with the generated `{item}` of field `{leaf}`"),
            ));
        }
    }
    Ok(())
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
    let fields = named_fields(input, "Observation")?;
    let named: Vec<(&syn::Field, &Ident)> = fields
        .iter()
        .filter_map(|f| f.ident.as_ref().map(|i| (f, i)))
        .collect();
    check_names(&named)?;
    let mut config: Option<Type> = None;
    let mut parsed = Vec::with_capacity(named.len());
    for &(field, ident) in &named {
        let field = parse_field(field, ident)?;
        for expr in field_exprs(&field.space) {
            same_config(expr, &mut config)?;
        }
        parsed.push(field);
    }
    let (impl_generics, config_ty) = match &config {
        Some(ty) => (quote!(), quote!(#ty)),
        None => (quote!(<C>), quote!(C)),
    };
    let mut offset = Offset::default();
    let mut out = Generated::default();
    for field in &parsed {
        field_tokens(&core, name, &config_ty, field, &mut offset, &mut out)?;
    }
    let contains_set = parsed
        .iter()
        .any(|f| matches!(f.space, FieldSpace::Set { .. }));
    let leaves = offset.tokens();
    let Generated {
        ids,
        spaces,
        checks,
        bounds,
    } = out;
    let field_idents = named.iter().map(|(_, i)| i);
    let where_clause = if bounds.is_empty() {
        quote!()
    } else {
        quote!(where #(#bounds),*)
    };

    Ok(quote! {
        impl #name {
            #(#ids)*
        }

        // One const item per leaf, spanned at the field (SPEC 10.2).
        #(#checks)*

        // Marker fields hold no data; reading them here keeps dead-code
        // lints quiet in the author's crate.
        const _: fn(&#name) = |obs| {
            let _ = (#(&obs.#field_idents,)*);
        };

        impl #core::sealed::ObservationLayoutSealed for #name {}

        impl #core::ObservationLayout for #name {
            const LEAVES: usize = #leaves;
            const CONTAINS_SET: bool = #contains_set;
        }

        impl #impl_generics #core::Observation<#config_ty> for #name #where_clause {
            const LEAVES: usize = #leaves;
            const CONTAINS_SET: bool = #contains_set;

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

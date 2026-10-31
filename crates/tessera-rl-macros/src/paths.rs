// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resolution of the path to `tessera-rl-core` in generated code.
//!
//! Order: an explicit `#[tessera(crate = path)]`; then `tessera-rl-core` if
//! the user's crate depends on it (environment crates, ADR-0021); then the
//! facade `tessera-rl` as `<facade>::core`; then `::tessera_rl_core`.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Attribute, Ident, Path};

/// Returns the path to the core crate for an item with `attrs`.
pub(crate) fn core_path(attrs: &[Attribute]) -> syn::Result<TokenStream> {
    if let Some(path) = explicit_crate(attrs)? {
        return Ok(quote!(#path));
    }
    Ok(resolve(
        crate_name("tessera-rl-core").ok(),
        crate_name("tessera-rl").ok(),
    ))
}

fn explicit_crate(attrs: &[Attribute]) -> syn::Result<Option<Path>> {
    let mut found = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("tessera")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("crate") {
                found = Some(meta.value()?.parse::<Path>()?);
                Ok(())
            } else {
                Err(meta.error("expected `crate = path`"))
            }
        })?;
    }
    Ok(found)
}

/// Maps what Cargo.toml declares to a path. Pure, so it is unit tested.
fn resolve(core: Option<FoundCrate>, facade: Option<FoundCrate>) -> TokenStream {
    match (core, facade) {
        (Some(FoundCrate::Name(name)), _) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(::#ident)
        }
        (None, Some(FoundCrate::Name(name))) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!(::#ident::core)
        }
        (None, Some(FoundCrate::Itself)) => quote!(crate::core),
        // `Itself` for core means core's own tests or examples, which link
        // the crate under its name.
        (Some(FoundCrate::Itself) | None, _) => quote!(::tessera_rl_core),
    }
}

#[cfg(test)]
mod tests {
    use proc_macro_crate::FoundCrate;

    use super::resolve;

    fn path(core: Option<FoundCrate>, facade: Option<FoundCrate>) -> String {
        resolve(core, facade).to_string().replace(' ', "")
    }

    #[test]
    fn test_resolve_prefers_direct_core_dependency() {
        let core = Some(FoundCrate::Name("tessera_rl_core".into()));
        let facade = Some(FoundCrate::Name("tessera_rl".into()));
        assert_eq!(path(core, facade), "::tessera_rl_core");
    }

    #[test]
    fn test_resolve_uses_facade_when_core_is_absent() {
        let facade = Some(FoundCrate::Name("tessera".into()));
        assert_eq!(path(None, facade), "::tessera::core");
        assert_eq!(path(None, Some(FoundCrate::Itself)), "crate::core");
    }

    #[test]
    fn test_resolve_defaults_to_core() {
        assert_eq!(path(None, None), "::tessera_rl_core");
        assert_eq!(path(Some(FoundCrate::Itself), None), "::tessera_rl_core");
    }
}

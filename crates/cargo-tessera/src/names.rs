// SPDX-License-Identifier: MIT OR Apache-2.0

//! Name validation for `cargo tessera new` (#351): the crate name, the
//! Python package derived from it, and the environment id.

/// Names a scaffold may not take: Rust's built-in crates and the
/// first-party Python package.
const RESERVED_CRATES: &[&str] = &[
    "alloc",
    "core",
    "proc_macro",
    "std",
    "test",
    "self",
    "crate",
    "super",
];

/// Python keywords and soft keywords: not valid as a package name.
const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield",
];

/// The namespace reserved for first-party environments (ADR-0023).
const RESERVED_NAMESPACE: &str = "tessera";

/// The longest crate name accepted.
const MAX_NAME: usize = 64;

/// A name that cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NameError {
    /// A crate name that is not lower-case ASCII letters, digits, `-` and
    /// `_`, starting with a letter.
    #[error(
        "crate name `{name}`: {reason} (lower-case ASCII letters, digits, `-` and `_`, starting with a letter, at most 64)"
    )]
    Crate {
        /// The name.
        name: String,
        /// Why.
        reason: &'static str,
    },
    /// A Python package name that cannot be imported.
    #[error("Python package `{name}` (from the crate name): {reason}")]
    Package {
        /// The package name.
        name: String,
        /// Why.
        reason: &'static str,
    },
    /// An id that is not `<namespace>/<Name>-v<N>`.
    #[error(
        "environment id `{id}`: {reason} (expected <namespace>/<Name>-v<N>, e.g. acme/Grid-v1)"
    )]
    Id {
        /// The id.
        id: String,
        /// Why.
        reason: &'static str,
    },
}

/// Checks a crate name.
///
/// # Errors
///
/// [`NameError::Crate`].
pub fn check_crate(name: &str) -> Result<(), NameError> {
    let fail = |reason| {
        Err(NameError::Crate {
            name: name.to_owned(),
            reason,
        })
    };
    if name.is_empty() || name.len() > MAX_NAME {
        return fail("empty or too long");
    }
    if !name.starts_with(|c: char| c.is_ascii_lowercase()) {
        return fail("does not start with a lower-case letter");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return fail("has a character other than a-z, 0-9, `-` and `_`");
    }
    if name.ends_with(['-', '_']) {
        return fail("ends with `-` or `_`");
    }
    let snake = package(name);
    if RESERVED_CRATES.contains(&snake.as_str()) {
        return fail("is a built-in crate name");
    }
    if snake == "tessera_rl" || snake.starts_with("tessera_rl_") {
        return fail("the tessera-rl prefix is reserved for first-party crates");
    }
    Ok(())
}

/// The Python package of a crate: `-` becomes `_`.
#[must_use]
pub fn package(name: &str) -> String {
    name.replace('-', "_")
}

/// Checks the Python package derived from a valid crate name.
///
/// # Errors
///
/// [`NameError::Package`].
pub fn check_package(package: &str) -> Result<(), NameError> {
    let fail = |reason| {
        Err(NameError::Package {
            name: package.to_owned(),
            reason,
        })
    };
    if PYTHON_KEYWORDS.contains(&package) {
        return fail("is a Python keyword");
    }
    if package.contains("__") {
        return fail("has a double underscore");
    }
    if package == "tessera_rl" {
        return fail("is the first-party package");
    }
    Ok(())
}

/// The default id of a crate: `<name>/<CamelName>-v1`.
#[must_use]
pub fn default_id(name: &str) -> String {
    let camel: String = name
        .split(['-', '_'])
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut chars = p.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect();
    format!("{name}/{camel}-v1")
}

/// A checked id: its namespace, name and version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvId {
    /// The whole id.
    pub id: String,
    /// The version `N` of `-vN`, as `HostEnvironment::ENV_VERSION`.
    pub version: u32,
}

/// Checks an environment id `<namespace>/<Name>-v<N>` outside the reserved
/// `tessera/` namespace.
///
/// # Errors
///
/// [`NameError::Id`].
pub fn check_id(id: &str) -> Result<EnvId, NameError> {
    let fail = |reason| {
        Err(NameError::Id {
            id: id.to_owned(),
            reason,
        })
    };
    let Some((ns, rest)) = id.split_once('/') else {
        return fail("no `/` between namespace and name");
    };
    if ns.is_empty()
        || !ns.starts_with(|c: char| c.is_ascii_lowercase())
        || !ns
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
    {
        return fail("the namespace is not lower-case ASCII letters, digits, `-` and `_`");
    }
    if ns == RESERVED_NAMESPACE {
        return fail("the tessera/ namespace is reserved for first-party environments");
    }
    let Some((name, version)) = rest.rsplit_once("-v") else {
        return fail("no `-v<N>` version suffix");
    };
    if name.is_empty()
        || !name.starts_with(|c: char| c.is_ascii_alphabetic())
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return fail("the name is not ASCII letters, digits and `_`, starting with a letter");
    }
    if version.is_empty()
        || version.starts_with('0')
        || !version.chars().all(|c| c.is_ascii_digit())
    {
        return fail("the version is not a positive integer without leading zeros");
    }
    let Ok(version) = version.parse::<u32>() else {
        return fail("the version does not fit in u32");
    };
    Ok(EnvId {
        id: id.to_owned(),
        version,
    })
}

#[cfg(test)]
mod tests {
    use super::{NameError, check_crate, check_id, check_package, default_id, package};

    #[test]
    fn test_check_crate_valid_names_accepted() {
        for name in ["grid", "my-env", "my_env2", "a1-b2_c3"] {
            check_crate(name).unwrap();
        }
    }

    #[test]
    fn test_check_crate_invalid_names_refused() {
        for name in [
            "",
            "Grid",
            "1grid",
            "-grid",
            "grid-",
            "grid_",
            "gr id",
            "grid!",
            "grïd",
            "core",
            "std",
            "tessera-rl",
            "tessera-rl-orderbook",
            "tessera_rl_x",
            "a".repeat(65).as_str(),
        ] {
            assert!(
                matches!(check_crate(name), Err(NameError::Crate { .. })),
                "{name:?}"
            );
        }
    }

    #[test]
    fn test_package_dashes_become_underscores() {
        assert_eq!(package("my-env-2"), "my_env_2");
    }

    #[test]
    fn test_check_package_keywords_and_dunders_refused() {
        check_package("my_env").unwrap();
        for name in ["import", "class", "a__b", "tessera_rl"] {
            assert!(
                matches!(check_package(name), Err(NameError::Package { .. })),
                "{name:?}"
            );
        }
    }

    #[test]
    fn test_default_id_camel_cases_the_name() {
        assert_eq!(default_id("my-grid_env"), "my-grid_env/MyGridEnv-v1");
        assert_eq!(check_id(&default_id("my-grid_env")).unwrap().version, 1);
    }

    #[test]
    fn test_check_id_valid_ids_give_their_version() {
        assert_eq!(check_id("acme/Grid-v1").unwrap().version, 1);
        assert_eq!(check_id("my-ns/Grid_World2-v12").unwrap().version, 12);
        assert_eq!(check_id("x/A-v-v3").map(|e| e.version).ok(), None);
    }

    #[test]
    fn test_check_id_invalid_ids_refused() {
        for id in [
            "Grid-v1",
            "/Grid-v1",
            "Acme/Grid-v1",
            "tessera/Grid-v1",
            "acme/Grid",
            "acme/-v1",
            "acme/1Grid-v1",
            "acme/Grid-v0",
            "acme/Grid-v01",
            "acme/Grid-vx",
            "acme/Grid-v99999999999",
            "acme/Gr id-v1",
            "acme/Grid-v",
        ] {
            assert!(matches!(check_id(id), Err(NameError::Id { .. })), "{id:?}");
        }
    }
}
